#!/usr/bin/env python3
"""
Verify label alignment across the entire pipeline.

This script traces a specific action segment through:
1. Label Studio JSON (raw frame numbers)
2. Converted labels (*_y.npy at 10Hz)
3. Extracted frames (frames/ directory at 25fps)
4. Video file (original at 30fps)

And creates a visual verification by extracting frames at the labeled boundaries.
"""

import argparse
import json
import math
import pathlib
import subprocess
from typing import Optional

import cv2
import numpy as np


def get_video_info(video_path):
    """Get frame count, FPS, and duration from video."""
    cap = cv2.VideoCapture(str(video_path))
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    duration = n_frames / fps if fps > 0 else 0
    cap.release()
    return n_frames, fps, duration


def extract_frame_from_video(video_path, timestamp_sec, output_path):
    """Extract a single frame from video at given timestamp."""
    cmd = [
        "ffmpeg", "-y",
        "-ss", f"{timestamp_sec:.3f}",
        "-i", str(video_path),
        "-frames:v", "1",
        "-q:v", "2",
        str(output_path)
    ]
    subprocess.run(cmd, capture_output=True)
    return output_path.exists()


def find_action_in_labelstudio(ls_json_path, video_stem, action_name):
    """Find raw Label Studio frame numbers for an action in a video."""
    with open(ls_json_path) as f:
        tasks = json.load(f)
    
    for task in tasks:
        video_path_str = task.get("data", {}).get("video", "")
        if video_stem not in video_path_str:
            continue
        
        segments = []
        for ann in task.get("annotations", []):
            for result in ann.get("result", []):
                if result.get("type") == "timelinelabels":
                    labels = result["value"].get("timelinelabels", [])
                    if not labels:
                        continue
                    # Check if any label matches (case-insensitive)
                    if any(action_name.lower() in lbl.lower() for lbl in labels):
                        for rng in result["value"].get("ranges", []):
                            segments.append({
                                "label": labels[0],
                                "start_frame": rng["start"],  # 1-indexed
                                "end_frame": rng["end"]       # 1-indexed
                            })
        
        if segments:
            return segments
    
    return []


def main():
    ap = argparse.ArgumentParser(description="Verify label alignment across pipeline")
    ap.add_argument("--video_stem", required=True, help="Video stem to analyze")
    ap.add_argument("--action", required=True, help="Action name to trace")
    ap.add_argument("--segment_idx", type=int, default=0, help="Which segment (0-indexed)")
    ap.add_argument("--videos_dir", required=True, help="Directory with original videos")
    ap.add_argument("--frames_dir", required=True, help="Directory with extracted frames")
    ap.add_argument("--labels_dir", required=True, help="Directory with 10Hz labels (*_y.npy)")
    ap.add_argument("--class_index", required=True, help="Path to class_index.json")
    ap.add_argument("--labelstudio_json", default=None, help="Path to Label Studio export (optional)")
    ap.add_argument("--output_dir", required=True, help="Output directory for verification")
    args = ap.parse_args()
    
    output_dir = pathlib.Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load class index
    with open(args.class_index) as f:
        class_index = json.load(f)
    
    # Find action index
    action_idx = None
    for name, idx in class_index.items():
        if args.action.lower() in name.lower():
            action_idx = idx
            action_name = name
            break
    
    if action_idx is None:
        print(f"Action '{args.action}' not found in class index!")
        print(f"Available: {list(class_index.keys())}")
        return
    
    print(f"=== Verifying Label Alignment ===")
    print(f"Video: {args.video_stem}")
    print(f"Action: {action_name} (idx={action_idx})")
    print()
    
    # =========================================================================
    # Step 1: Video file info
    # =========================================================================
    video_path = None
    for ext in [".mp4", ".MP4"]:
        candidate = pathlib.Path(args.videos_dir) / f"{args.video_stem}{ext}"
        if candidate.exists():
            video_path = candidate
            break
    
    if not video_path:
        print(f"ERROR: Video not found for {args.video_stem}")
        return
    
    n_frames_video, video_fps, video_duration = get_video_info(video_path)
    print(f"[VIDEO FILE]")
    print(f"  Path: {video_path}")
    print(f"  Frames: {n_frames_video}")
    print(f"  FPS: {video_fps}")
    print(f"  Duration: {video_duration:.2f}s")
    print()
    
    # =========================================================================
    # Step 2: Extracted frames info
    # =========================================================================
    frames_dir = pathlib.Path(args.frames_dir) / args.video_stem
    if frames_dir.exists():
        frame_files = sorted(frames_dir.glob("*.jpg"))
        n_frames_extracted = len(frame_files)
        extracted_fps = n_frames_extracted / video_duration if video_duration > 0 else 0
    else:
        n_frames_extracted = 0
        extracted_fps = 0
        frame_files = []
    
    print(f"[EXTRACTED FRAMES]")
    print(f"  Directory: {frames_dir}")
    print(f"  Frame count: {n_frames_extracted}")
    print(f"  Implied FPS: {extracted_fps:.2f}")
    print(f"  Expected FPS: 25.0 (from ffmpeg extraction)")
    print()
    
    # =========================================================================
    # Step 3: Labels info
    # =========================================================================
    label_path = pathlib.Path(args.labels_dir) / f"{args.video_stem}_y.npy"
    if label_path.exists():
        labels = np.load(label_path)
        label_len = len(labels)
        label_duration = label_len / 10.0  # 10Hz
    else:
        print(f"ERROR: Labels not found at {label_path}")
        return
    
    print(f"[10Hz LABELS]")
    print(f"  Path: {label_path}")
    print(f"  Length: {label_len}")
    print(f"  Duration at 10Hz: {label_duration:.2f}s")
    print()
    
    # =========================================================================
    # Step 4: Find action segments in labels
    # =========================================================================
    in_segment = False
    segments = []
    for i, lbl in enumerate(labels):
        if lbl == action_idx and not in_segment:
            in_segment = True
            start_idx = i
        elif lbl != action_idx and in_segment:
            in_segment = False
            segments.append((start_idx, i - 1))
    if in_segment:
        segments.append((start_idx, len(labels) - 1))
    
    print(f"[ACTION SEGMENTS IN 10Hz LABELS]")
    print(f"  Found {len(segments)} segments of '{action_name}'")
    for i, (s, e) in enumerate(segments[:10]):  # Show first 10
        start_sec = s / 10.0
        end_sec = e / 10.0
        print(f"    {i}: indices [{s}, {e}] → [{start_sec:.2f}s, {end_sec:.2f}s]")
    if len(segments) > 10:
        print(f"    ... and {len(segments) - 10} more")
    print()
    
    if not segments:
        print("No segments found!")
        return
    
    # =========================================================================
    # Step 5: Trace specific segment
    # =========================================================================
    seg_idx = min(args.segment_idx, len(segments) - 1)
    start_10hz, end_10hz = segments[seg_idx]
    
    start_sec = start_10hz / 10.0
    end_sec = end_10hz / 10.0
    mid_sec = (start_sec + end_sec) / 2
    
    # Calculate corresponding frame indices in each domain
    start_frame_25fps = int(round(start_sec * 25))  # Frame in extracted frames
    mid_frame_25fps = int(round(mid_sec * 25))
    end_frame_25fps = int(round(end_sec * 25))
    
    start_frame_30fps = int(round(start_sec * 30))  # Frame in original video
    mid_frame_30fps = int(round(mid_sec * 30))
    end_frame_30fps = int(round(end_sec * 30))
    
    print(f"[TRACING SEGMENT {seg_idx}]")
    print(f"  10Hz label indices: [{start_10hz}, {end_10hz}]")
    print(f"  Time range: [{start_sec:.2f}s, {end_sec:.2f}s] (duration: {end_sec - start_sec:.2f}s)")
    print()
    print(f"  Corresponding frames at 25fps (extracted): [{start_frame_25fps}, {end_frame_25fps}]")
    print(f"  Corresponding frames at 30fps (video):     [{start_frame_30fps}, {end_frame_30fps}]")
    print()
    
    # =========================================================================
    # Step 6: Extract verification frames
    # =========================================================================
    print(f"[EXTRACTING VERIFICATION FRAMES]")
    
    # From extracted frames (25fps)
    if frame_files:
        for name, frame_idx in [("start", start_frame_25fps), ("mid", mid_frame_25fps), ("end", end_frame_25fps)]:
            if 0 <= frame_idx < len(frame_files):
                src = frame_files[frame_idx]
                dst = output_dir / f"{args.video_stem}_{action_name.replace(' ', '_')}_seg{seg_idx}_{name}_from_extracted_f{frame_idx}.jpg"
                import shutil
                shutil.copy(src, dst)
                print(f"  ✓ Extracted frame {frame_idx} from frames/ → {dst.name}")
            else:
                print(f"  ✗ Frame {frame_idx} out of range for extracted frames (0-{len(frame_files)-1})")
    
    # From original video (seek by time)
    for name, t_sec in [("start", start_sec), ("mid", mid_sec), ("end", end_sec)]:
        dst = output_dir / f"{args.video_stem}_{action_name.replace(' ', '_')}_seg{seg_idx}_{name}_from_video_{t_sec:.2f}s.jpg"
        if extract_frame_from_video(video_path, t_sec, dst):
            print(f"  ✓ Extracted frame at {t_sec:.2f}s from video → {dst.name}")
        else:
            print(f"  ✗ Failed to extract frame at {t_sec:.2f}s from video")
    
    # =========================================================================
    # Step 7: Label Studio raw data (if available)
    # =========================================================================
    if args.labelstudio_json:
        ls_path = pathlib.Path(args.labelstudio_json)
        if ls_path.exists():
            ls_segments = find_action_in_labelstudio(ls_path, args.video_stem, args.action)
            if ls_segments:
                print()
                print(f"[LABEL STUDIO RAW DATA]")
                print(f"  Found {len(ls_segments)} segments in Label Studio export:")
                for i, seg in enumerate(ls_segments[:10]):
                    start_f = seg["start_frame"]
                    end_f = seg["end_frame"]
                    # Convert LS frames (1-indexed, at video fps) to seconds
                    start_sec_ls = (start_f - 1) / video_fps
                    end_sec_ls = (end_f - 1) / video_fps
                    print(f"    {i}: frames [{start_f}, {end_f}] (1-indexed @ {video_fps}fps)")
                    print(f"       → [{start_sec_ls:.2f}s, {end_sec_ls:.2f}s]")
    
    print()
    print(f"=== Verification frames saved to: {output_dir} ===")
    print()
    print("Compare the '_from_extracted_' and '_from_video_' frames visually.")
    print("If they show different content, there's a timing mismatch!")


if __name__ == "__main__":
    main()
