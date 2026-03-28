#!/usr/bin/env python3
"""
Convert FineBio PCR (task 05) annotations to 7-class taxonomy.

7-class taxonomy:
  0: none                     - Background/no action
  1: Adjust pipette volume    - NOT IN FINEBIO (class exists but has 0 examples)
  2: Attach pipette tip       - insert,pipette,*_tip
  3: Eject pipette tip        - eject,pipette,*
  4: Manipulate pcr tubes     - insert/press/release/detach targeting 8_tube_stripes
  5: Manipulate reagents      - insert/press/release/detach targeting micro_tube
  6: Switch pipette           - Changing from one pipette to another

Key improvements over previous versions:
1. Uses container context tracking for press/release actions
2. Detects pipette switches (put X_pipette + take Y_pipette where X != Y)
3. Maps to full 7-class taxonomy (Adjust volume will have 0 examples)

Usage:
    python convert_finebio_pcr_to_7class.py \
        --annotation_dir /work3/s204157/data/labassist/annotations/finebio_action_annotations \
        --videos_dir /work3/s204157/data/labassist/finebio_videos_w640 \
        --output_dir /zhome/45/8/156251/Desktop/thesis/labassist/hpc_scripts/data \
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


# 7-class taxonomy
CLASS_INDEX_7CLASS = {
    "none": 0,
    "Adjust pipette volume": 1,  # Not in FineBio
    "Attach pipette tip": 2,
    "Eject pipette tip": 3,
    "Manipulate pcr tubes": 4,
    "Manipulate reagents": 5,
    "Switch pipette": 6,
}

# Container classification
PCR_CONTAINERS = {"8_tube_stripes", "8_tube_stripes_lid", "pcr_tube", "pcr"}
REAGENT_CONTAINERS = {"micro_tube", "falcon_tube", "eppendorf_tube", "reagent"}
TIP_OBJECTS = {"red_tip", "yellow_tip", "blue_tip", "white_tip", 
               "8_channel_tip", "green_tip", "tip"}
PIPETTE_OBJECTS = {"red_pipette", "blue_pipette", "yellow_pipette", 
                   "white_pipette", "green_pipette", "8_channel_pipette", "pipette"}


def get_video_info(video_path: Path) -> Tuple[float, int]:
    """Get FPS and frame count from video file."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    
    return fps, frame_count


def is_pipette(obj: str) -> bool:
    """Check if object is a pipette."""
    return any(p in obj.lower() for p in PIPETTE_OBJECTS)


def get_pipette_type(obj: str) -> Optional[str]:
    """Extract pipette type from object string."""
    obj_lower = obj.lower()
    for ptype in PIPETTE_OBJECTS:
        if ptype in obj_lower:
            return ptype
    return None


def classify_target(affected_obj: str) -> Optional[str]:
    """Classify target container as 'pcr' or 'reagent'."""
    obj_lower = affected_obj.lower()
    
    # Check PCR containers
    for container in PCR_CONTAINERS:
        if container in obj_lower:
            return "pcr"
    
    # Check reagent containers
    for container in REAGENT_CONTAINERS:
        if container in obj_lower:
            return "reagent"
    
    return None


def parse_pcr_annotations(filepath: Path) -> List[Tuple[float, float, str]]:
    """
    Parse a FineBio PCR annotation file and convert to 7-class labels.
    
    Returns list of (start_sec, end_sec, class_name) tuples.
    """
    actions = []
    current_container = None  # Track current container for press/release context
    current_pipette = None    # Track current pipette in hand
    last_action_end = 0
    last_put_pipette = None
    
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
            
            verb = row[4].strip().lower() if len(row) > 4 else ""
            manipulated_obj = row[5].strip() if len(row) > 5 else ""
            affected_obj = row[6].strip() if len(row) > 6 else ""
            
            # Skip non-atomic actions (high-level task labels)
            if not verb:
                continue
            
            class_name = "none"
            
            # TAKE pipette action - track which pipette
            if verb == "take" and is_pipette(manipulated_obj):
                new_pipette = get_pipette_type(manipulated_obj)
                
                # Detect pipette switch
                if (current_pipette is not None and 
                    new_pipette is not None and 
                    current_pipette != new_pipette and
                    start_sec - last_action_end < 5.0):  # Within 5 seconds
                    class_name = "Switch pipette"
                
                current_pipette = new_pipette
                current_container = None  # Reset container when picking up pipette
            
            # PUT pipette action - track for potential switch detection
            elif verb == "put" and is_pipette(manipulated_obj):
                last_put_pipette = get_pipette_type(manipulated_obj)
                last_action_end = end_sec
                # Don't clear current_pipette yet - keep tracking until take
            
            # Only process actions with pipette
            elif is_pipette(manipulated_obj):
                # INSERT action
                if verb == "insert":
                    # Check if inserting into tip
                    if any(tip in affected_obj.lower() for tip in TIP_OBJECTS):
                        class_name = "Attach pipette tip"
                        current_container = None  # Reset container
                    else:
                        # Inserting into container
                        target = classify_target(affected_obj)
                        if target == "pcr":
                            class_name = "Manipulate pcr tubes"
                            current_container = "pcr"
                        elif target == "reagent":
                            class_name = "Manipulate reagents"
                            current_container = "reagent"
                
                # EJECT action (always tip ejection)
                elif verb == "eject":
                    class_name = "Eject pipette tip"
                    current_container = None
                
                # PRESS/RELEASE actions - inherit container context
                elif verb in ["press", "release"]:
                    if current_container == "pcr":
                        class_name = "Manipulate pcr tubes"
                    elif current_container == "reagent":
                        class_name = "Manipulate reagents"
                    # else: stays as "none" if no container context
                
                # DETACH action - same as current container context
                elif verb == "detach":
                    if current_container == "pcr":
                        class_name = "Manipulate pcr tubes"
                    elif current_container == "reagent":
                        class_name = "Manipulate reagents"
                    current_container = None  # Clear context after detach
            
            if class_name != "none":
                actions.append((start_sec, end_sec, class_name))
                last_action_end = end_sec
    
    return actions


def actions_to_frame_labels(
    actions: List[Tuple[float, float, str]],
    n_frames: int,
    fps: float
) -> np.ndarray:
    """Convert action segments to per-frame labels at target FPS."""
    labels = np.zeros(n_frames, dtype=np.int64)  # Default to 'none' (0)
    
    for start_sec, end_sec, class_name in actions:
        if class_name not in CLASS_INDEX_7CLASS:
            continue
        
        class_idx = CLASS_INDEX_7CLASS[class_name]
        start_frame = int(start_sec * fps)
        end_frame = int(end_sec * fps)
        
        # Clamp to valid range
        start_frame = max(0, min(start_frame, n_frames - 1))
        end_frame = min(n_frames, int(end_sec * fps))
        
        if start_frame < end_frame:
            labels[start_frame:end_frame] = class_idx
    
    return labels


def main():
    ap = argparse.ArgumentParser(description="Convert FineBio PCR annotations to 7-class taxonomy")
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
    labels_dir = output_dir / "labels_7class_finebio"
    labels_dir.mkdir(parents=True, exist_ok=True)

    # Save class index
    class_index_path = output_dir / "class_index_7class_finebio.json"
    with open(class_index_path, "w") as f:
        json.dump(CLASS_INDEX_7CLASS, f, indent=2)
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
            print(f"  ⚠ Video not found: {video_stem}")
            skipped.append(video_stem)
            continue
        
        # Get video info
        try:
            src_fps, frame_count = get_video_info(video_path)
        except Exception as e:
            print(f"  ⚠ Error reading video {video_stem}: {e}")
            skipped.append(video_stem)
            continue
        
        duration = frame_count / src_fps
        n_label_frames = int(duration * args.fps)
        
        # Parse annotations
        actions = parse_pcr_annotations(ann_file)
        
        # Count action types
        action_counts = Counter([a[2] for a in actions])
        
        # Convert to frame labels
        labels = actions_to_frame_labels(actions, n_label_frames, args.fps)
        
        # Update stats
        for idx in range(7):
            class_name = [k for k, v in CLASS_INDEX_7CLASS.items() if v == idx][0]
            count = (labels == idx).sum()
            total_stats[class_name] += count
        
        # Save labels
        label_path = labels_dir / f"{video_stem}_y.npy"
        np.save(label_path, labels.astype(np.int16))
        
        # Print progress
        action_frames = (labels > 0).sum()
        action_pct = 100 * action_frames / len(labels) if len(labels) > 0 else 0
        print(f"  ✓ {video_stem}: {len(labels)} frames, {action_frames} action ({action_pct:.1f}%)")
        for action, count in sorted(action_counts.items()):
            print(f"      {action}: {count} segments")
        
        processed.append(video_stem)

    print(f"\n{'='*70}")
    print(f"✓ Processed {len(processed)} videos")
    if skipped:
        print(f"⚠ Skipped {len(skipped)} videos")
    
    print(f"\nClass distribution (7-class):")
    total_frames = sum(total_stats.values())
    for class_name in CLASS_INDEX_7CLASS.keys():
        count = total_stats[class_name]
        pct = 100 * count / total_frames if total_frames > 0 else 0
        print(f"  {CLASS_INDEX_7CLASS[class_name]}: {class_name:25s} {count:8d} ({pct:.2f}%)")
    
    print(f"\n✓ Labels saved to: {labels_dir}")
    print(f"✓ Class index: {class_index_path}")


if __name__ == "__main__":
    main()




