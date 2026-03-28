#!/usr/bin/env python3
"""
Convert FineBio action annotations to MS-TCN format (10Hz labels).

FineBio annotations are CSV files with columns:
  start_sec, end_sec, task, hand_side, verb, manipulated_object, affected_object

This script:
1. Reads all annotation files for T5 (PCR) videos
2. Creates atomic action labels (verb+object combinations)
3. Outputs labels at 10Hz for MS-TCN training
"""

import argparse
import csv
import json
import math
import pathlib
import random
from collections import Counter

import cv2
import numpy as np


def get_video_duration(video_path):
    """Get video duration in seconds."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return n_frames / fps if fps > 0 else None


def parse_annotation_file(ann_path):
    """Parse a FineBio annotation file and return list of (start, end, label) tuples."""
    annotations = []
    
    with open(ann_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                start = float(row['start_sec'])
                end = float(row['end_sec'])
            except (ValueError, KeyError):
                continue
            
            # Skip task-level annotations (we want atomic actions)
            verb = row.get('verb', '').strip()
            if not verb:
                continue
            
            # Create atomic action label: verb,object
            obj = row.get('manipulated_object', '').strip()
            if obj:
                label = f"{verb},{obj}"
            else:
                label = verb
            
            annotations.append((start, end, label))
    
    return annotations


def main():
    ap = argparse.ArgumentParser(description="Convert FineBio annotations to MS-TCN format")
    ap.add_argument("--annotations_dir", required=True, help="Directory with FineBio .txt annotation files")
    ap.add_argument("--videos_dir", required=True, help="Directory with video files")
    ap.add_argument("--out_dir", required=True, help="Output directory for MS-TCN cache")
    ap.add_argument("--target_fps", type=float, default=10.0, help="Target FPS for labels")
    ap.add_argument("--task_filter", default="05", help="Filter videos by task number (e.g., '05' for T5/PCR)")
    ap.add_argument("--val_ratio", type=float, default=0.1, help="Fraction of videos for validation")
    ap.add_argument("--seed", type=int, default=42, help="Random seed")
    ap.add_argument("--min_action_frames", type=int, default=100, help="Minimum frames for a class to be included")
    args = ap.parse_args()
    
    random.seed(args.seed)
    
    ann_dir = pathlib.Path(args.annotations_dir)
    videos_dir = pathlib.Path(args.videos_dir)
    out_dir = pathlib.Path(args.out_dir)
    labels_dir = out_dir / "labels"
    labels_dir.mkdir(parents=True, exist_ok=True)
    
    # Find annotation files matching task filter
    pattern = f"*_{args.task_filter}_*.txt" if args.task_filter else "*.txt"
    ann_files = sorted(ann_dir.glob(pattern))
    
    print(f"Found {len(ann_files)} annotation files matching '{pattern}'")
    
    # Pass 1: Collect all action labels and count frames
    all_labels = set()
    label_counts = Counter()
    video_annotations = {}
    
    for ann_path in ann_files:
        # Derive video stem (P01_05_01.txt -> P01_05_01_T5)
        # Annotation format: P{person}_{task}_{trial}.txt
        # Video format: P{person}_{task}_{trial}_T{task}.mp4
        base_stem = ann_path.stem  # e.g., "P01_05_01"
        parts = base_stem.split('_')
        if len(parts) >= 3:
            task_num = parts[1]  # e.g., "05"
            # Video naming uses single digit task: T5 not T05
            task_digit = str(int(task_num))  # "05" -> "5"
            video_stem = f"{base_stem}_T{task_digit}"  # P01_05_01_T5
        else:
            video_stem = base_stem
        
        # Find video file
        video_path = None
        for ext in ['.mp4', '.MP4', '.avi']:
            candidate = videos_dir / f"{video_stem}{ext}"
            if candidate.exists():
                video_path = candidate
                break
        
        if not video_path:
            print(f"⚠ Video not found for {video_stem}, skipping")
            continue
        
        # Get video duration
        duration = get_video_duration(video_path)
        if duration is None:
            print(f"⚠ Could not read video {video_path}, skipping")
            continue
        
        # Parse annotations
        annotations = parse_annotation_file(ann_path)
        if not annotations:
            print(f"⚠ No atomic actions in {ann_path.name}, skipping")
            continue
        
        # Count frames per label
        for start, end, label in annotations:
            frames = int((end - start) * args.target_fps)
            label_counts[label] += frames
            all_labels.add(label)
        
        video_annotations[video_stem] = {
            'video_path': video_path,
            'duration': duration,
            'annotations': annotations
        }
    
    print(f"\nFound {len(video_annotations)} videos with annotations")
    print(f"Found {len(all_labels)} unique action labels")
    
    # Filter rare labels
    if args.min_action_frames > 0:
        rare_labels = {l for l, c in label_counts.items() if c < args.min_action_frames}
        print(f"Filtering {len(rare_labels)} rare labels (< {args.min_action_frames} frames)")
        all_labels -= rare_labels
    
    # Create class index (sorted, with 'none' as index 0)
    class_names = ['none'] + sorted(all_labels)
    class_index = {name: idx for idx, name in enumerate(class_names)}
    
    print(f"\nFinal class count: {len(class_index)}")
    print("Classes:")
    for name, idx in list(class_index.items())[:20]:
        print(f"  {idx:3d}: {name}")
    if len(class_index) > 20:
        print(f"  ... and {len(class_index) - 20} more")
    
    # Save class index
    with open(out_dir / "class_index.json", "w") as f:
        json.dump(class_index, f, indent=2)
    
    # Pass 2: Create label arrays
    processed_stems = []
    
    for video_stem, data in video_annotations.items():
        duration = data['duration']
        annotations = data['annotations']
        
        # Create label array at target fps
        T = int(math.ceil(duration * args.target_fps))
        labels = np.zeros(T, dtype=np.int64)  # Default to 'none' (index 0)
        
        for start, end, label in annotations:
            if label not in class_index:
                continue  # Skip rare labels
            
            label_idx = class_index[label]
            start_frame = max(0, int(start * args.target_fps))
            end_frame = min(T, int(end * args.target_fps))
            labels[start_frame:end_frame] = label_idx
        
        # Save label array
        np.save(labels_dir / f"{video_stem}_y.npy", labels)
        processed_stems.append(video_stem)
        
        # Print progress
        non_none = (labels > 0).sum()
        print(f"  ✓ {video_stem}: {T} frames, {non_none} action frames ({100*non_none/T:.1f}%)")
    
    # Create train/val splits
    random.shuffle(processed_stems)
    n_val = max(1, int(len(processed_stems) * args.val_ratio))
    val_stems = processed_stems[:n_val]
    train_stems = processed_stems[n_val:]
    
    splits = {
        "train": [f"{s}.mp4" for s in train_stems],
        "val": [f"{s}.mp4" for s in val_stems],
        "test": []
    }
    
    with open(out_dir / "splits.json", "w") as f:
        json.dump(splits, f, indent=2)
    
    print(f"\n{'='*60}")
    print(f"✓ Processed {len(processed_stems)} videos")
    print(f"✓ Classes: {len(class_index)}")
    print(f"✓ Train: {len(train_stems)}, Val: {len(val_stems)}")
    print(f"✓ Output: {out_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
