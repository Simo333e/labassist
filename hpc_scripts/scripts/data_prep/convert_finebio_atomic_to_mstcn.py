#!/usr/bin/env python3
"""
Convert FineBio atomic action annotations to MS-TCN format.

This script:
1. Parses verb+object combinations from FineBio annotations
2. Creates a class index mapping
3. Converts to per-frame .npy labels at specified FPS

Usage:
    python convert_finebio_atomic_to_mstcn.py \
        --annotation_dir /path/to/finebio_action_annotations \
        --frames_dir /path/to/finebio_full/frames \
        --output_dir /path/to/finebio_full/cache \
        --fps 10.0 \
        --view T5
"""

import argparse
import csv
import json
import re
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


# Object type mappings for grouping similar objects
OBJECT_GROUPS = {
    # Pipettes
    "yellow_pipette": "pipette",
    "blue_pipette": "pipette",
    "red_pipette": "pipette",
    "8_channel_pipette": "pipette",
    "white_pipette": "pipette",
    "green_pipette": "pipette",
    
    # Tips
    "yellow_tip": "tip",
    "blue_tip": "tip",
    "red_tip": "tip",
    "white_tip": "tip",
    
    # Tip racks
    "yellow_tip_rack": "tip_rack",
    "blue_tip_rack": "tip_rack",
    "red_tip_rack": "tip_rack",
    "white_tip_rack": "tip_rack",
    
    # Tubes/containers
    "micro_tube": "tube",
    "8_tube_stripes": "tube",
    "pcr_tube": "tube",
    "falcon_tube": "tube",
    "eppendorf_tube": "tube",
    "tube_with_spin_column": "tube",
    
    # Plates
    "cell_culture_plate": "plate",
    "96_well_plate": "plate",
    "384_well_plate": "plate",
    "petri_dish": "plate",
    
    # Lids
    "8_tube_stripes_lid": "lid",
    "cell_culture_plate_lid": "lid",
    "micro_tube_lid": "lid",
    
    # Other
    "centrifuge": "centrifuge",
    "vortex": "vortex",
    "pcr_machine": "equipment",
}


def get_object_group(obj: str) -> str:
    """Map specific object to its group."""
    if not obj or obj.strip() == "":
        return "none"
    obj = obj.strip().lower()
    return OBJECT_GROUPS.get(obj, "other")


def create_action_class(verb: str, manipulated_obj: str, affected_obj: str) -> str:
    """
    Create action class from verb + objects.
    
    Examples:
        insert + pipette + tip → insert_pipette_tip
        press + pipette + (none) → press_pipette
        eject + pipette + tip → eject_tip
        take + tube + (none) → take_tube
    """
    verb = verb.strip().lower() if verb else ""
    if not verb:
        return "none"
    
    manip_group = get_object_group(manipulated_obj)
    affect_group = get_object_group(affected_obj)
    
    # Special cases for pipetting actions
    if verb == "insert" and manip_group == "pipette":
        if affect_group == "tip":
            return "attach_tip"  # Maps to "Attach pipette tip"
        elif affect_group in ["tube", "plate"]:
            return "insert_pipette_container"
        else:
            return "insert_pipette"
    
    if verb == "eject" and (manip_group == "pipette" or affect_group == "tip"):
        return "eject_tip"  # Maps to "Eject pipette tip"
    
    if verb == "press" and manip_group == "pipette":
        return "press_pipette"  # Related to aspirate/dispense
    
    if verb == "release" and manip_group == "pipette":
        return "release_pipette"
    
    if verb == "detach" and manip_group == "pipette":
        return "detach_pipette"
    
    # General patterns
    if manip_group != "none":
        return f"{verb}_{manip_group}"
    elif affect_group != "none":
        return f"{verb}_{affect_group}"
    else:
        return verb


def parse_annotation_file(filepath: Path) -> List[Tuple[float, float, str]]:
    """
    Parse a FineBio annotation file.
    
    Returns list of (start_sec, end_sec, action_class) tuples.
    Only returns atomic actions (those with verb filled in).
    """
    actions = []
    
    with open(filepath, "r") as f:
        reader = csv.reader(f)
        header = next(reader, None)  # Skip header
        
        for row in reader:
            if len(row) < 7:
                continue
            
            start_sec = float(row[0])
            end_sec = float(row[1])
            task = row[2].strip() if len(row) > 2 else ""
            hand_side = row[3].strip() if len(row) > 3 else ""
            verb = row[4].strip() if len(row) > 4 else ""
            manipulated_obj = row[5].strip() if len(row) > 5 else ""
            affected_obj = row[6].strip() if len(row) > 6 else ""
            
            # Only process atomic actions (verb is filled)
            if verb and verb != "verb":  # Skip header row if present
                action_class = create_action_class(verb, manipulated_obj, affected_obj)
                actions.append((start_sec, end_sec, action_class))
    
    return actions


def actions_to_frame_labels(
    actions: List[Tuple[float, float, str]],
    n_frames: int,
    fps: float,
    class_to_idx: Dict[str, int]
) -> np.ndarray:
    """Convert action segments to per-frame labels."""
    labels = np.zeros(n_frames, dtype=np.int64)  # Default to 'none' (0)
    
    for start_sec, end_sec, action_class in actions:
        if action_class not in class_to_idx:
            continue
        
        start_frame = int(start_sec * fps)
        end_frame = int(end_sec * fps)
        
        # Clamp to valid range
        start_frame = max(0, min(start_frame, n_frames - 1))
        end_frame = max(0, min(end_frame, n_frames))
        
        if start_frame < end_frame:
            labels[start_frame:end_frame] = class_to_idx[action_class]
    
    return labels


def main():
    ap = argparse.ArgumentParser(description="Convert FineBio atomic annotations to MS-TCN format")
    ap.add_argument("--annotation_dir", required=True, help="Directory with FineBio annotation .txt files")
    ap.add_argument("--frames_dir", required=True, help="Directory with extracted frames")
    ap.add_argument("--output_dir", required=True, help="Output directory for cache files")
    ap.add_argument("--fps", type=float, default=10.0, help="Label FPS (should match feature FPS)")
    ap.add_argument("--view", default="T5", help="Camera view suffix (e.g., T5)")
    ap.add_argument("--frame_fps", type=float, default=25.0, help="FPS of extracted frames")
    args = ap.parse_args()

    annotation_dir = Path(args.annotation_dir)
    frames_dir = Path(args.frames_dir)
    output_dir = Path(args.output_dir)
    labels_dir = output_dir / "labels_atomic"
    labels_dir.mkdir(parents=True, exist_ok=True)

    # First pass: collect all action classes
    print("Pass 1: Collecting action classes...")
    all_actions = []
    annotation_files = sorted(annotation_dir.glob("*.txt"))
    
    for ann_file in annotation_files:
        actions = parse_annotation_file(ann_file)
        all_actions.extend([a[2] for a in actions])
    
    # Count and create class index
    action_counts = Counter(all_actions)
    print(f"\nFound {len(action_counts)} unique action classes:")
    for action, count in action_counts.most_common():
        print(f"  {action}: {count}")
    
    # Create class index (sorted by frequency, 'none' first)
    sorted_actions = ["none"] + [a for a, _ in action_counts.most_common() if a != "none"]
    class_to_idx = {action: idx for idx, action in enumerate(sorted_actions)}
    idx_to_class = {idx: action for action, idx in class_to_idx.items()}
    
    # Save class index
    class_index_path = output_dir / "class_index_atomic.json"
    with open(class_index_path, "w") as f:
        json.dump(class_to_idx, f, indent=2)
    print(f"\nSaved class index ({len(class_to_idx)} classes) to {class_index_path}")

    # Second pass: convert to per-frame labels
    print(f"\nPass 2: Converting to per-frame labels at {args.fps} Hz...")
    
    processed = []
    skipped = []
    
    for ann_file in annotation_files:
        # Match annotation to video frames
        # Annotation: P01_05_01.txt → Video: P01_05_01_T5 → Frames: frames/P01_05_01_T5/
        base_stem = ann_file.stem  # e.g., P01_05_01
        video_stem = f"{base_stem}_{args.view}"  # e.g., P01_05_01_T5
        frame_dir = frames_dir / video_stem
        
        if not frame_dir.exists():
            skipped.append(video_stem)
            continue
        
        # Count frames
        n_source_frames = len(list(frame_dir.glob("*.jpg")))
        if n_source_frames == 0:
            skipped.append(video_stem)
            continue
        
        # Calculate number of label frames at target FPS
        video_duration = n_source_frames / args.frame_fps
        n_label_frames = int(video_duration * args.fps)
        
        # Parse actions
        actions = parse_annotation_file(ann_file)
        
        # Convert to frame labels
        labels = actions_to_frame_labels(actions, n_label_frames, args.fps, class_to_idx)
        
        # Save
        label_path = labels_dir / f"{video_stem}_y.npy"
        np.save(label_path, labels)
        
        # Stats
        action_frames = np.sum(labels > 0)
        action_pct = 100 * action_frames / len(labels) if len(labels) > 0 else 0
        
        processed.append(video_stem)
        print(f"  {video_stem}: {n_label_frames} frames, {action_frames} action ({action_pct:.1f}%)")

    print(f"\n✓ Converted {len(processed)} videos")
    if skipped:
        print(f"⚠ Skipped {len(skipped)} videos (no frames found)")

    # Create splits (80% train, 10% val, 10% test)
    import random
    random.seed(42)
    random.shuffle(processed)
    
    n_train = int(0.8 * len(processed))
    n_val = int(0.1 * len(processed))
    
    train_stems = processed[:n_train]
    val_stems = processed[n_train:n_train + n_val]
    test_stems = processed[n_train + n_val:]
    
    splits = {
        "train": [f"{s}.mp4" for s in train_stems],
        "val": [f"{s}.mp4" for s in val_stems],
        "test": [f"{s}.mp4" for s in test_stems]
    }
    
    splits_path = output_dir / "splits_finebio_atomic.json"
    with open(splits_path, "w") as f:
        json.dump(splits, f, indent=2)
    print(f"\nSplits saved: {len(train_stems)} train, {len(val_stems)} val, {len(test_stems)} test")
    print(f"  → {splits_path}")

    print("\n✓ Conversion complete!")
    print(f"  Labels: {labels_dir}")
    print(f"  Class index: {class_index_path}")
    print(f"  Splits: {splits_path}")


if __name__ == "__main__":
    main()

