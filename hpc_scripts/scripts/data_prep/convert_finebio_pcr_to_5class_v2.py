#!/usr/bin/env python3
"""
Convert FineBio PCR (task 05) annotations to 5-class taxonomy.

New 5-class taxonomy (aligned with mydata):
  0: none                  - Background/no action
  1: Manipulate PCR tube   - Pipetting actions targeting 8_tube_stripes (PCR tubes)
  2: Manipulate reagent    - Pipetting actions targeting micro_tube (reagent source)
  3: Attach pipette tip    - Attaching a new tip to pipette
  4: Eject pipette tip     - Ejecting used tip from pipette

The key insight is that FineBio annotations include the `affected_object` field:
  - insert,pipette,8_tube_stripes  → Manipulate PCR tube
  - insert,pipette,micro_tube      → Manipulate reagent
  - insert,pipette,*_tip           → Attach pipette tip
  - eject,pipette,*                → Eject pipette tip

For press/release actions (which lack affected_object), we inherit the container
from the most recent insert action.

Usage:
    python convert_finebio_pcr_to_5class_v2.py \
        --annotation_dir /work3/s204157/data/labassist/annotations/finebio_action_annotations \
        --videos_dir /work3/s204157/data/labassist/finebio_videos_w640 \
        --output_dir /work3/s204157/data/labassist/combined_pcr/cache \
        --fps 10.0
"""

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np


# New 5-class taxonomy
CLASS_INDEX_5CLASS = {
    "none": 0,
    "Manipulate PCR tube": 1,
    "Manipulate reagent": 2,
    "Attach pipette tip": 3,
    "Eject pipette tip": 4,
}

# Container classification
PCR_CONTAINERS = {"8_tube_stripes", "8_tube_stripes_lid", "pcr_tube"}
REAGENT_CONTAINERS = {"micro_tube", "falcon_tube", "eppendorf_tube"}
TIP_OBJECTS = {"red_tip", "yellow_tip", "blue_tip", "white_tip", 
               "8_channel_tip", "green_tip", "tip"}


def get_video_info(video_path: Path) -> Tuple[float, int]:
    """Get FPS and frame count from video file."""
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return fps, n_frames


def is_pipette(obj: str) -> bool:
    """Check if object is a pipette."""
    obj_lower = obj.lower().strip()
    return 'pipette' in obj_lower


def classify_target(affected_obj: str) -> Optional[str]:
    """
    Classify the target container type.
    
    Returns:
        "pcr" for PCR tubes
        "reagent" for reagent containers
        "tip" for pipette tips
        None for other/unknown
    """
    if not affected_obj:
        return None
    
    affected_lower = affected_obj.lower().strip()
    
    # Check for tips first (most specific)
    if any(tip in affected_lower for tip in ['_tip', 'channel_tip']):
        return "tip"
    
    # Check for PCR containers
    if any(pcr in affected_lower for pcr in ['8_tube_stripe', 'pcr_tube', 'tube_stripe']):
        return "pcr"
    
    # Check for reagent containers
    if any(reagent in affected_lower for reagent in ['micro_tube', 'falcon', 'eppendorf']):
        return "reagent"
    
    return None


def parse_pcr_annotations(filepath: Path) -> List[Tuple[float, float, str]]:
    """
    Parse a FineBio PCR annotation file and convert to 5-class labels.
    
    Returns list of (start_sec, end_sec, class_name) tuples.
    """
    actions = []
    current_container = None  # Track current container for press/release context
    
    with open(filepath, "r") as f:
        reader = csv.reader(f)
        header = next(reader, None)  # Skip header
        
        for row in reader:
            if len(row) < 7:
                continue
            
            try:
                start_sec = float(row[0])
                end_sec = float(row[1])
            except ValueError:
                continue
            
            # task = row[2].strip() if len(row) > 2 else ""
            # hand_side = row[3].strip() if len(row) > 3 else ""
            verb = row[4].strip().lower() if len(row) > 4 else ""
            manipulated_obj = row[5].strip() if len(row) > 5 else ""
            affected_obj = row[6].strip() if len(row) > 6 else ""
            
            # Skip non-atomic actions (high-level task labels)
            if not verb:
                continue
            
            # Only process pipette-related actions
            if not is_pipette(manipulated_obj):
                continue
            
            class_name = "none"
            
            # INSERT action
            if verb == "insert":
                target = classify_target(affected_obj)
                if target == "tip":
                    class_name = "Attach pipette tip"
                    current_container = None  # Reset container context
                elif target == "pcr":
                    class_name = "Manipulate PCR tube"
                    current_container = "pcr"  # Set context for subsequent press/release
                elif target == "reagent":
                    class_name = "Manipulate reagent"
                    current_container = "reagent"
            
            # EJECT action (always tip ejection)
            elif verb == "eject":
                class_name = "Eject pipette tip"
                current_container = None
            
            # PRESS/RELEASE actions - inherit container context
            elif verb in ["press", "release"]:
                if current_container == "pcr":
                    class_name = "Manipulate PCR tube"
                elif current_container == "reagent":
                    class_name = "Manipulate reagent"
                # else: stays as "none" if no container context
            
            # DETACH action - same as current container context
            elif verb == "detach":
                if current_container == "pcr":
                    class_name = "Manipulate PCR tube"
                elif current_container == "reagent":
                    class_name = "Manipulate reagent"
                current_container = None  # Clear context after detach
            
            if class_name != "none":
                actions.append((start_sec, end_sec, class_name))
    
    return actions


def actions_to_frame_labels(
    actions: List[Tuple[float, float, str]],
    n_frames: int,
    fps: float
) -> np.ndarray:
    """Convert action segments to per-frame labels at target FPS."""
    labels = np.zeros(n_frames, dtype=np.int64)  # Default to 'none' (0)
    
    for start_sec, end_sec, class_name in actions:
        if class_name not in CLASS_INDEX_5CLASS:
            continue
        
        class_idx = CLASS_INDEX_5CLASS[class_name]
        start_frame = int(start_sec * fps)
        end_frame = int(end_sec * fps)
        
        # Clamp to valid range
        start_frame = max(0, min(start_frame, n_frames - 1))
        end_frame = max(0, min(end_frame, n_frames))
        
        if start_frame < end_frame:
            labels[start_frame:end_frame] = class_idx
    
    return labels


def main():
    ap = argparse.ArgumentParser(description="Convert FineBio PCR annotations to 5-class taxonomy")
    ap.add_argument("--annotation_dir", required=True, 
                    help="Directory with FineBio annotation .txt files")
    ap.add_argument("--videos_dir", required=True,
                    help="Directory with FineBio video files")
    ap.add_argument("--output_dir", required=True,
                    help="Output directory for labels")
    ap.add_argument("--fps", type=float, default=10.0,
                    help="Target label FPS (should match feature extraction)")
    ap.add_argument("--view", default="T5",
                    help="Camera view suffix (e.g., T5)")
    args = ap.parse_args()

    annotation_dir = Path(args.annotation_dir)
    videos_dir = Path(args.videos_dir)
    output_dir = Path(args.output_dir)
    labels_dir = output_dir / "labels_5class_v2"
    labels_dir.mkdir(parents=True, exist_ok=True)

    # Save class index
    class_index_path = output_dir / "class_index_5class_v2.json"
    with open(class_index_path, "w") as f:
        json.dump(CLASS_INDEX_5CLASS, f, indent=2)
    print(f"✓ Saved class index to: {class_index_path}")

    # Find PCR annotation files (task 05)
    pcr_files = sorted(annotation_dir.glob("*_05_*.txt"))
    print(f"\nFound {len(pcr_files)} PCR (task 05) annotation files")

    # Process each file
    processed = []
    skipped = []
    total_stats = Counter()
    
    for ann_file in pcr_files:
        # Map annotation to video: P01_05_01.txt → P01_05_01_T5.mp4
        base_stem = ann_file.stem  # e.g., P01_05_01
        video_stem = f"{base_stem}_{args.view}"  # e.g., P01_05_01_T5
        
        # Find video file
        video_path = videos_dir / f"{video_stem}.mp4"
        if not video_path.exists():
            video_path = videos_dir / f"{video_stem}.MP4"
        if not video_path.exists():
            skipped.append(f"{video_stem} (video not found)")
            continue
        
        # Get video info
        src_fps, n_src_frames = get_video_info(video_path)
        video_duration = n_src_frames / src_fps
        n_label_frames = int(np.ceil(video_duration * args.fps))
        
        # Parse annotations
        actions = parse_pcr_annotations(ann_file)
        
        # Convert to frame labels
        labels = actions_to_frame_labels(actions, n_label_frames, args.fps)
        
        # Save labels
        label_path = labels_dir / f"{video_stem}_y.npy"
        np.save(label_path, labels)
        
        # Compute stats
        unique, counts = np.unique(labels, return_counts=True)
        video_stats = dict(zip(unique.astype(int), counts.astype(int)))
        
        action_frames = sum(c for idx, c in video_stats.items() if idx > 0)
        action_pct = 100 * action_frames / len(labels) if len(labels) > 0 else 0
        
        processed.append(video_stem)
        
        # Update totals
        for idx, count in video_stats.items():
            class_name = [k for k, v in CLASS_INDEX_5CLASS.items() if v == idx][0]
            total_stats[class_name] += count
        
        print(f"  ✓ {video_stem}: {len(labels)} frames, {action_frames} action ({action_pct:.1f}%)")
    
    print(f"\n{'='*60}")
    print(f"SUMMARY")
    print(f"{'='*60}")
    print(f"Processed: {len(processed)} videos")
    print(f"Skipped: {len(skipped)} videos")
    
    if skipped:
        print(f"\nSkipped videos:")
        for s in skipped[:10]:
            print(f"  - {s}")
        if len(skipped) > 10:
            print(f"  ... and {len(skipped) - 10} more")
    
    print(f"\n{'='*60}")
    print(f"CLASS DISTRIBUTION (across all processed videos)")
    print(f"{'='*60}")
    total_frames = sum(total_stats.values())
    for class_name, count in sorted(total_stats.items(), key=lambda x: CLASS_INDEX_5CLASS[x[0]]):
        pct = 100 * count / total_frames if total_frames > 0 else 0
        print(f"  [{CLASS_INDEX_5CLASS[class_name]}] {class_name:25s}: {count:8,} frames ({pct:5.1f}%)")
    
    print(f"\n✓ Labels saved to: {labels_dir}")
    print(f"✓ Class index saved to: {class_index_path}")


if __name__ == "__main__":
    main()






