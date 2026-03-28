#!/usr/bin/env python3
"""
Extract video snippets around labeled action segments for visual comparison.

This script:
1. Reads label files (numpy arrays at 10Hz)
2. Finds contiguous segments for each action class
3. Extracts short video clips around each segment (with padding)
4. Saves clips as individual video files or concatenated per-action montages

Usage:
    python extract_action_snippets.py \
        --video_dir /path/to/videos \
        --label_dir /path/to/labels \
        --class_index /path/to/class_index.json \
        --output_dir /path/to/output \
        --actions "Attach pipette tip,Eject pipette tip" \
        --stems "20251120_065442Z,P01_05_01_T5" \
        --padding 1.0 \
        --max_clips 5
"""

import argparse
import json
import pathlib
import subprocess
from typing import Dict, List, Tuple

import numpy as np


def find_action_segments(labels: np.ndarray, action_idx: int, fps: float = 10.0) -> List[Tuple[float, float]]:
    """
    Find contiguous segments where the action occurs.
    
    Args:
        labels: 1D array of label indices (shape: [T])
        action_idx: The action class index to find
        fps: Label frequency (default 10Hz)
        
    Returns:
        List of (start_time, end_time) tuples in seconds
    """
    segments = []
    in_segment = False
    start_idx = 0
    
    for i, label in enumerate(labels):
        if label == action_idx and not in_segment:
            # Start of new segment
            in_segment = True
            start_idx = i
        elif label != action_idx and in_segment:
            # End of segment
            in_segment = False
            segments.append((start_idx / fps, (i - 1) / fps))
    
    # Handle segment that extends to end
    if in_segment:
        segments.append((start_idx / fps, (len(labels) - 1) / fps))
    
    return segments


def extract_clip(video_path: pathlib.Path, output_path: pathlib.Path,
                 start_time: float, end_time: float, padding: float = 1.0) -> bool:
    """
    Extract a video clip using ffmpeg.
    
    Args:
        video_path: Path to source video
        output_path: Path to save clip
        start_time: Start time in seconds
        end_time: End time in seconds
        padding: Extra time to add before/after segment
        
    Returns:
        True if successful
    """
    # Add padding
    padded_start = max(0, start_time - padding)
    padded_end = end_time + padding
    duration = padded_end - padded_start
    
    cmd = [
        "ffmpeg", "-y",
        "-ss", f"{padded_start:.3f}",
        "-i", str(video_path),
        "-t", f"{duration:.3f}",
        "-c:v", "libx264",
        "-preset", "fast",
        "-crf", "23",
        "-an",  # No audio
        str(output_path)
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        return result.returncode == 0
    except Exception as e:
        print(f"  ⚠ ffmpeg error: {e}")
        return False


def find_video_file(video_dir: pathlib.Path, stem: str) -> pathlib.Path:
    """Find video file for a given stem (handles various naming conventions)."""
    # Try exact match first
    for ext in [".mp4", ".MP4", ".avi", ".mov"]:
        candidate = video_dir / f"{stem}{ext}"
        if candidate.exists():
            return candidate
        # Also try with _clean suffix
        candidate = video_dir / f"{stem}_clean{ext}"
        if candidate.exists():
            return candidate
    
    # Try globbing
    candidates = list(video_dir.glob(f"{stem}*"))
    if candidates:
        return candidates[0]
    
    return None


def main():
    ap = argparse.ArgumentParser(description="Extract video snippets around labeled actions")
    ap.add_argument("--video_dir", required=True, help="Directory containing video files")
    ap.add_argument("--label_dir", required=True, help="Directory containing label .npy files")
    ap.add_argument("--class_index", required=True, help="Path to class_index.json")
    ap.add_argument("--output_dir", required=True, help="Output directory for clips")
    ap.add_argument("--actions", required=True, 
                    help="Comma-separated list of action names to extract")
    ap.add_argument("--stems", default=None,
                    help="Comma-separated list of video stems to process (default: all)")
    ap.add_argument("--padding", type=float, default=1.0,
                    help="Seconds of padding before/after each segment")
    ap.add_argument("--max_clips", type=int, default=5,
                    help="Maximum clips per action per video")
    ap.add_argument("--label_fps", type=float, default=10.0,
                    help="Label frequency in Hz (default: 10.0)")
    args = ap.parse_args()
    
    video_dir = pathlib.Path(args.video_dir)
    label_dir = pathlib.Path(args.label_dir)
    output_dir = pathlib.Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load class index
    with open(args.class_index) as f:
        class_index = json.load(f)
    
    # Parse actions to extract
    requested_actions = [a.strip() for a in args.actions.split(",")]
    action_to_idx = {}
    for action in requested_actions:
        if action in class_index:
            action_to_idx[action] = class_index[action]
        else:
            # Try case-insensitive match
            for k, v in class_index.items():
                if k.lower() == action.lower():
                    action_to_idx[action] = v
                    break
            else:
                print(f"⚠ Action '{action}' not found in class index, skipping")
    
    if not action_to_idx:
        print("No valid actions to extract!")
        return
    
    print(f"Extracting clips for actions: {list(action_to_idx.keys())}")
    
    # Get stems to process
    if args.stems:
        stems = [s.strip() for s in args.stems.split(",")]
    else:
        # Get all stems from label directory
        stems = [p.stem.replace("_y", "") for p in label_dir.glob("*_y.npy")]
    
    print(f"Processing {len(stems)} videos")
    
    # Process each video
    for stem in stems:
        label_path = label_dir / f"{stem}_y.npy"
        if not label_path.exists():
            print(f"⚠ No labels for {stem}, skipping")
            continue
        
        video_path = find_video_file(video_dir, stem)
        if video_path is None:
            print(f"⚠ No video found for {stem}, skipping")
            continue
        
        labels = np.load(label_path)
        print(f"\n📹 {stem} ({len(labels)} frames @ {args.label_fps}Hz = {len(labels)/args.label_fps:.1f}s)")
        
        for action_name, action_idx in action_to_idx.items():
            segments = find_action_segments(labels, action_idx, args.label_fps)
            
            if not segments:
                print(f"  • {action_name}: no segments found")
                continue
            
            print(f"  • {action_name}: {len(segments)} segments")
            
            # Create action output directory
            safe_action_name = action_name.replace(" ", "_").replace("/", "-")
            action_dir = output_dir / safe_action_name
            action_dir.mkdir(exist_ok=True)
            
            # Extract clips (up to max_clips)
            for i, (start, end) in enumerate(segments[:args.max_clips]):
                duration = end - start
                clip_name = f"{stem}_{safe_action_name}_{i:02d}_{start:.1f}s-{end:.1f}s.mp4"
                clip_path = action_dir / clip_name
                
                success = extract_clip(video_path, clip_path, start, end, args.padding)
                status = "✓" if success else "✗"
                print(f"    {status} [{start:.1f}s - {end:.1f}s] ({duration:.2f}s) → {clip_name}")
    
    print(f"\n✓ Done! Clips saved to: {output_dir}")
    print("\nDirectory structure:")
    for action_name in action_to_idx.keys():
        safe_action_name = action_name.replace(" ", "_").replace("/", "-")
        action_dir = output_dir / safe_action_name
        if action_dir.exists():
            clips = list(action_dir.glob("*.mp4"))
            print(f"  {safe_action_name}/: {len(clips)} clips")


if __name__ == "__main__":
    main()
