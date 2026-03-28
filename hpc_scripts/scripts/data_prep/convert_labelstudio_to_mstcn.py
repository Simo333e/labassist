#!/usr/bin/env python3
"""
Convert Label Studio video annotation export to MS-TCN training format.

Outputs:
  - splits.json        (train/val/test splits)
  - class_index.json   (action name -> index mapping)
  - labels/*_y.npy     (per-frame label arrays at target FPS)
  - class_distribution.csv (optional summary)
"""
import argparse
import csv
import json
import math
import random
from collections import Counter

import cv2
import numpy as np
import pathlib


IGNORE_INDEX = -100


def get_video_info(video_path):
    """Get frame count and FPS from video."""
    cap = cv2.VideoCapture(str(video_path))
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    cap.release()
    return n_frames, fps


# Label Studio uses a fixed playback FPS (typically 24fps) for its timeline,
# regardless of the video's native FPS. This means frame numbers in LS annotations
# correspond to 24fps timestamps, NOT the video's native fps.
LABEL_STUDIO_FPS = 24.0


def reduce_background_runs(sequence, default_idx, stride, max_len):
    if stride <= 1 and max_len <= 0:
        return sequence
    seq = sequence.copy()
    n = len(seq)
    i = 0
    while i < n:
        if seq[i] != default_idx:
            i += 1
            continue
        run_start = i
        while i < n and seq[i] == default_idx:
            i += 1
        run_end = i
        run_len = run_end - run_start
        allowed_len = run_len if max_len <= 0 else min(run_len, max_len)
        for pos in range(run_start, run_end):
            keep = True
            if stride > 1 and ((pos - run_start) % stride):
                keep = False
            if max_len > 0 and (pos - run_start) >= allowed_len:
                keep = False
            if not keep:
                seq[pos] = IGNORE_INDEX
    return seq


def write_class_summary(path, class_names, counts, ignored_frames):
    total = counts.sum()
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["class", "frames", "percent"])
        for name, cnt in zip(class_names, counts):
            frac = (cnt / total) if total else 0.0
            writer.writerow([name, int(cnt), f"{frac:.4f}"])
        writer.writerow(["ignored", ignored_frames, ""])


def main():
    ap = argparse.ArgumentParser(description="Convert Label Studio export into MS-TCN npy cache")
    ap.add_argument("--labelstudio_json", required=True, help="Path to Label Studio JSON export")
    ap.add_argument("--videos_dir", required=True, help="Directory containing video files")
    ap.add_argument("--out_dir", required=True, help="Output directory for cache")
    ap.add_argument("--target_fps", type=float, default=10.0, help="Target FPS for labels (must match feature extraction)")
    ap.add_argument("--val_ratio", type=float, default=0.17, help="Fraction of videos for validation")
    ap.add_argument("--test_ratio", type=float, default=0.0, help="Fraction of videos for test")
    ap.add_argument("--seed", type=int, default=42, help="Random seed for split")
    ap.add_argument("--default_label", default="none", help="Default label for unlabeled frames")
    ap.add_argument("--min_frames_per_class", type=int, default=0,
                    help="Merge classes with fewer than this many frames into default_label (0 disables)")
    ap.add_argument("--none_stride", type=int, default=1,
                    help="Keep one default-label frame every N. Others become ignore_index (-100).")
    ap.add_argument("--none_max_len", type=int, default=0,
                    help="Maximum consecutive default-label frames to keep (0 = unlimited). Extra frames become ignore_index.")
    ap.add_argument("--summary_csv", default="class_distribution.csv",
                    help="Filename for class distribution CSV (stored in out_dir)")
    args = ap.parse_args()

    args.default_label = args.default_label.strip()

    # v2 taxonomy: keep only core actions, collapse the rest to default_label ("none")
    KEEP_LABELS = {
        "none": args.default_label,                     # ensure consistent casing
        "attach pipette tip": "Attach pipette tip",
        "eject pipette tip": "Eject pipette tip",
        "aspirate reagent": "Aspirate reagent",
        "apirate pcr tube": "Apirate pcr tube",         # keep current spelling
        "dispence to pcr tube": "Dispence to pcr tube",
    }

    def remap_label(name):
        """Map raw Label Studio label to collapsed taxonomy."""
        if not name:
            return name
        normalized = name.strip()
        if not normalized:
            return normalized

        key = normalized.lower()

        # collapse setup and any low-priority labels to background
        if key == "setting up":
            return args.default_label

        if key in KEEP_LABELS:
            return KEEP_LABELS[key]

        # everything else (volume adjustments, stirring, switch pipette, etc.) → background
        return args.default_label

    random.seed(args.seed)
    out_dir = pathlib.Path(args.out_dir)
    labels_dir = out_dir / "labels"
    labels_dir.mkdir(parents=True, exist_ok=True)
    videos_dir = pathlib.Path(args.videos_dir)

    # Load Label Studio export
    with open(args.labelstudio_json) as f:
        tasks = json.load(f)

    print(f"Loaded {len(tasks)} tasks from Label Studio export")

    # Pass 1: Collect all unique action labels and frame counts
    all_labels = set()
    frame_counts = Counter()
    for task in tasks:
        for ann in task.get("annotations", []):
            for result in ann.get("result", []):
                if result.get("type") == "timelinelabels":
                    labels = [remap_label(lbl) for lbl in result["value"].get("timelinelabels", [])]
                    all_labels.update(labels)
                    for rng in result["value"].get("ranges", []):
                        start = rng.get("start", 1)
                        end = rng.get("end", start)
                        frames = max(0, end - start + 1)
                        if labels:
                            frame_counts[labels[0]] += frames

    # Add default label if not present
    if args.default_label and args.default_label not in all_labels:
        all_labels.add(args.default_label)

    # Create class index (sorted for reproducibility, default_label first if present)
    class_names = sorted(all_labels)
    if args.default_label in class_names:
        class_names.remove(args.default_label)
        class_names = [args.default_label] + class_names
    
    # Determine rare labels (excluding default)
    rare_labels = set()
    if args.min_frames_per_class > 0:
        for name in class_names:
            if name == args.default_label:
                continue
            if frame_counts.get(name, 0) < args.min_frames_per_class:
                rare_labels.add(name)

    if rare_labels:
        print(f"\nMerging {len(rare_labels)} rare classes (< {args.min_frames_per_class} frames) into '{args.default_label}':")
        for lbl in sorted(rare_labels):
            print(f"  - {lbl} ({frame_counts.get(lbl, 0)} frames)")
        class_names = [n for n in class_names if n not in rare_labels]

    class_index = {name: idx for idx, name in enumerate(class_names)}
    
    print(f"\nFound {len(class_index)} classes:")
    for name, idx in class_index.items():
        print(f"  {idx:2d}: {name}")

    # Save class index
    with open(out_dir / "class_index.json", "w") as f:
        json.dump(class_index, f, indent=2)
    print(f"\n✓ Saved class_index.json")

    # Pass 2: Process each video task
    video_stems = []
    total_counts = np.zeros(len(class_names), dtype=np.int64)
    ignored_frames = 0

    for task in tasks:
        # Extract video filename from Label Studio path
        video_path_str = task.get("data", {}).get("video", "")
        if "?d=" in video_path_str:
            # Label Studio local file format: /data/local-files/?d=videos/filename.mp4
            video_filename = video_path_str.split("?d=")[-1]
            # Handle nested paths
            video_filename = pathlib.Path(video_filename).name
        else:
            video_filename = pathlib.Path(video_path_str).name

        video_path = videos_dir / video_filename
        if not video_path.exists():
            print(f"⚠ Video not found: {video_path}, skipping...")
            continue

        stem = video_path.stem
        video_stems.append(stem)

        # Get frame count and FPS from video (for duration calculation)
        n_frames, src_fps = get_video_info(video_path)
        video_duration = n_frames / src_fps  # Duration in seconds
        print(f"\nProcessing {stem}: {n_frames} frames @ {src_fps:.1f} fps = {video_duration:.1f}s")
        print(f"  (Label Studio uses {LABEL_STUDIO_FPS} fps for its timeline)")

        # Create label array directly at target FPS (e.g., 10Hz)
        # This avoids the fps mismatch between Label Studio and video native fps
        T_target = int(math.ceil(video_duration * args.target_fps))
        default_idx = class_index.get(args.default_label, 0)
        y_resampled = np.full(T_target, default_idx, dtype=np.int64)

        # Fill labels from annotations
        # IMPORTANT: Label Studio frame numbers are in LABEL_STUDIO_FPS timeline (24fps),
        # NOT the video's native fps. We must convert LS frames to time, then to target fps.
        for ann in task.get("annotations", []):
            for result in ann.get("result", []):
                if result.get("type") == "timelinelabels":
                    label_names = [remap_label(lbl) for lbl in result["value"].get("timelinelabels", [])]
                    if not label_names:
                        continue
                    label_name = label_names[0]
                    if label_name in rare_labels:
                        label_name = args.default_label
                    label_idx = class_index[label_name]
                    
                    for rng in result["value"].get("ranges", []):
                        # Convert Label Studio frames (1-indexed, at LS fps) to time in seconds
                        ls_start_frame = rng["start"]
                        ls_end_frame = rng["end"]
                        start_time = (ls_start_frame - 1) / LABEL_STUDIO_FPS
                        end_time = ls_end_frame / LABEL_STUDIO_FPS
                        
                        # Convert time to target fps indices
                        start_idx = max(0, int(round(start_time * args.target_fps)))
                        end_idx = min(T_target, int(round(end_time * args.target_fps)))
                        
                        y_resampled[start_idx:end_idx] = label_idx
                        print(f"  LS frames [{ls_start_frame:6d}:{ls_end_frame:6d}] @ {LABEL_STUDIO_FPS}fps "
                              f"-> {start_time:.1f}s-{end_time:.1f}s -> idx [{start_idx:6d}:{end_idx:6d}] "
                              f"-> {label_names[0]} ({label_idx})")

        # Reduce background dominance if requested
        if args.default_label in class_index:
            default_idx = class_index[args.default_label]
            stride = max(1, int(args.none_stride))
            max_len = max(0, int(args.none_max_len))
            if stride > 1 or max_len > 0:
                y_reduced = reduce_background_runs(y_resampled, default_idx, stride, max_len)
            else:
                y_reduced = y_resampled
        else:
            y_reduced = y_resampled

        valid_mask = y_reduced != IGNORE_INDEX
        unique, counts = np.unique(y_reduced[valid_mask], return_counts=True)
        for idx, cnt in zip(unique, counts):
            total_counts[idx] += cnt
        ignored_frames += int((~valid_mask).sum())

        # Save label array
        np.save(labels_dir / f"{stem}_y.npy", y_reduced)
        print(f"  → Created {len(y_reduced)} labels @ {args.target_fps}Hz for {video_duration:.1f}s video")
        print(f"  → Saved {labels_dir / f'{stem}_y.npy'} ({valid_mask.sum()} valid, {(~valid_mask).sum()} ignored)")

    if not video_stems:
        print("\n❌ No videos processed! Check paths.")
        return

    # Create train/val/test splits
    random.shuffle(video_stems)
    n = len(video_stems)
    n_test = int(n * args.test_ratio)
    n_val = max(1, int(n * args.val_ratio)) if args.val_ratio > 0 else 0
    n_train = n - n_val - n_test

    # Ensure at least one training sample
    if n_train < 1:
        n_train = 1
        n_val = n - n_train - n_test

    splits = {
        "train": [f"{s}.mp4" for s in video_stems[:n_train]],
        "val": [f"{s}.mp4" for s in video_stems[n_train:n_train + n_val]],
        "test": [f"{s}.mp4" for s in video_stems[n_train + n_val:]],
    }

    with open(out_dir / "splits.json", "w") as f:
        json.dump(splits, f, indent=2)

    summary_path = out_dir / args.summary_csv
    write_class_summary(summary_path, class_names, total_counts, ignored_frames)

    print(f"\n{'='*50}")
    print(f"✓ Splits: train={len(splits['train'])}, val={len(splits['val'])}, test={len(splits['test'])}")
    print(f"✓ Classes: {len(class_index)}")
    print(f"✓ Target FPS: {args.target_fps} Hz")
    print(f"✓ Output: {out_dir}")
    print(f"✓ Class distribution summary: {summary_path}")
    print(f"{'='*50}")
    
    print("\nTrain videos:")
    for v in splits['train']:
        print(f"  - {v}")
    print("Val videos:")
    for v in splits['val']:
        print(f"  - {v}")
    if splits['test']:
        print("Test videos:")
        for v in splits['test']:
            print(f"  - {v}")


if __name__ == "__main__":
    main()



