#!/usr/bin/env python3
"""
Extract video clips directly from Label Studio JSON annotations.

This script bypasses the 10Hz label conversion and extracts clips directly
from Label Studio frame numbers → time → video clip.

KEY FIX: Label Studio uses 24fps for its timeline, NOT the video's native fps!
Frame numbers in LS annotations correspond to 24fps timestamps.

Usage:
    python extract_clips_from_labelstudio.py \
        --labelstudio_json /path/to/export.json \
        --videos_dir /path/to/videos \
        --output_dir /path/to/output \
        --actions "Attach pipette tip,Eject pipette tip" \
        --max_clips 5 \
        --padding 1.0
"""

import argparse
import json
import pathlib
import subprocess
from typing import Dict, List, Tuple, Optional

# Label Studio uses a fixed playback FPS (typically 24fps) for its timeline,
# regardless of the video's native FPS. This is the KEY insight!
LABEL_STUDIO_FPS = 24.0


def extract_clip(video_path: pathlib.Path, output_path: pathlib.Path,
                 start_time: float, end_time: float, padding: float = 1.0,
                 vflip: bool = False, crop: Optional[str] = None) -> bool:
    """Extract a video clip using ffmpeg."""
    padded_start = max(0, start_time - padding)
    padded_end = end_time + padding
    duration = padded_end - padded_start
    
    # Build filter string
    filters = []
    if crop:
        filters.append(crop)
    if vflip:
        filters.append("vflip")
    
    cmd = [
        "ffmpeg", "-y",
        "-ss", f"{padded_start:.3f}",
        "-i", str(video_path),
        "-t", f"{duration:.3f}",
    ]
    
    if filters:
        cmd.extend(["-vf", ",".join(filters)])
    
    cmd.extend([
        "-c:v", "libx264",
        "-preset", "fast",
        "-crf", "23",
        "-an",
        str(output_path)
    ])
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        return result.returncode == 0
    except Exception as e:
        print(f"  ⚠ ffmpeg error: {e}")
        return False


def find_video_file(videos_dir: pathlib.Path, video_filename: str) -> Optional[pathlib.Path]:
    """Find video file, handling various naming conventions."""
    # Direct match
    candidate = videos_dir / video_filename
    if candidate.exists():
        return candidate
    
    # Try without extension variations
    stem = pathlib.Path(video_filename).stem
    for ext in [".mp4", ".MP4", ".avi", ".mov"]:
        candidate = videos_dir / f"{stem}{ext}"
        if candidate.exists():
            return candidate
    
    return None


def parse_labelstudio_export(ls_json_path: pathlib.Path, actions: List[str]) -> Dict[str, List[dict]]:
    """
    Parse Label Studio export and extract segments for specified actions.
    
    Returns: {video_filename: [{"label": str, "start_sec": float, "end_sec": float}, ...]}
    """
    with open(ls_json_path) as f:
        tasks = json.load(f)
    
    # Normalize action names for case-insensitive matching
    action_set = {a.lower().strip() for a in actions}
    
    results = {}
    
    for task in tasks:
        # Extract video filename
        video_path_str = task.get("data", {}).get("video", "")
        if "?d=" in video_path_str:
            video_filename = video_path_str.split("?d=")[-1]
            video_filename = pathlib.Path(video_filename).name
        else:
            video_filename = pathlib.Path(video_path_str).name
        
        if not video_filename:
            continue
        
        segments = []
        
        for ann in task.get("annotations", []):
            for result in ann.get("result", []):
                if result.get("type") == "timelinelabels":
                    labels = result["value"].get("timelinelabels", [])
                    if not labels:
                        continue
                    
                    label_name = labels[0]
                    
                    # Check if this label matches any requested action
                    if label_name.lower().strip() not in action_set:
                        continue
                    
                    for rng in result["value"].get("ranges", []):
                        # CRITICAL: Convert Label Studio frames (1-indexed, at 24fps) to time
                        ls_start_frame = rng["start"]
                        ls_end_frame = rng["end"]
                        
                        # Frame numbers are 1-indexed in Label Studio
                        start_sec = (ls_start_frame - 1) / LABEL_STUDIO_FPS
                        end_sec = ls_end_frame / LABEL_STUDIO_FPS
                        
                        segments.append({
                            "label": label_name,
                            "start_sec": start_sec,
                            "end_sec": end_sec,
                            "ls_start_frame": ls_start_frame,
                            "ls_end_frame": ls_end_frame,
                        })
        
        if segments:
            results[video_filename] = segments
    
    return results


def main():
    ap = argparse.ArgumentParser(description="Extract clips directly from Label Studio JSON")
    ap.add_argument("--labelstudio_json", required=True, help="Path to Label Studio export JSON")
    ap.add_argument("--videos_dir", required=True, help="Directory containing video files")
    ap.add_argument("--output_dir", required=True, help="Output directory for clips")
    ap.add_argument("--actions", required=True, help="Comma-separated list of action names")
    ap.add_argument("--max_clips", type=int, default=5, help="Max clips per action per video")
    ap.add_argument("--padding", type=float, default=1.0, help="Padding in seconds")
    ap.add_argument("--vflip", action="store_true", help="Apply vertical flip")
    ap.add_argument("--crop", default=None, 
                    help="FFmpeg crop filter (e.g., 'crop=in_w*0.82:in_h*0.65:in_w*0.10:0')")
    ap.add_argument("--stems", default=None, help="Comma-separated video stems to process (optional)")
    args = ap.parse_args()
    
    ls_path = pathlib.Path(args.labelstudio_json)
    videos_dir = pathlib.Path(args.videos_dir)
    output_dir = pathlib.Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    actions = [a.strip() for a in args.actions.split(",")]
    stems_filter = None
    if args.stems:
        stems_filter = {s.strip() for s in args.stems.split(",")}
    
    print(f"=== Extracting clips directly from Label Studio JSON ===")
    print(f"Label Studio FPS: {LABEL_STUDIO_FPS} (this is what LS uses for its timeline)")
    print(f"Actions: {actions}")
    print()
    
    # Parse Label Studio export
    video_segments = parse_labelstudio_export(ls_path, actions)
    print(f"Found {len(video_segments)} videos with matching actions")
    
    # Process each video
    for video_filename, segments in video_segments.items():
        stem = pathlib.Path(video_filename).stem
        
        # Filter by stems if specified
        if stems_filter and stem not in stems_filter:
            continue
        
        video_path = find_video_file(videos_dir, video_filename)
        if not video_path:
            print(f"⚠ Video not found: {video_filename}")
            continue
        
        print(f"\n📹 {stem}")
        
        # Group segments by action
        by_action = {}
        for seg in segments:
            label = seg["label"]
            if label not in by_action:
                by_action[label] = []
            by_action[label].append(seg)
        
        for action_name, action_segments in by_action.items():
            safe_name = action_name.replace(" ", "_").replace("/", "-")
            action_dir = output_dir / safe_name
            action_dir.mkdir(exist_ok=True)
            
            print(f"  • {action_name}: {len(action_segments)} segments")
            
            for i, seg in enumerate(action_segments[:args.max_clips]):
                start = seg["start_sec"]
                end = seg["end_sec"]
                ls_start = seg["ls_start_frame"]
                ls_end = seg["ls_end_frame"]
                duration = end - start
                
                clip_name = f"{stem}_{safe_name}_{i:02d}_{start:.1f}s-{end:.1f}s.mp4"
                clip_path = action_dir / clip_name
                
                success = extract_clip(
                    video_path, clip_path, start, end, args.padding,
                    vflip=args.vflip, crop=args.crop
                )
                
                status = "✓" if success else "✗"
                print(f"    {status} LS frames [{ls_start}:{ls_end}] @ {LABEL_STUDIO_FPS}fps "
                      f"→ [{start:.1f}s - {end:.1f}s] ({duration:.2f}s)")
    
    print(f"\n✓ Done! Clips saved to: {output_dir}")
    
    # Print directory structure
    print("\nDirectory structure:")
    for action in actions:
        safe_name = action.replace(" ", "_").replace("/", "-")
        action_dir = output_dir / safe_name
        if action_dir.exists():
            clips = list(action_dir.glob("*.mp4"))
            print(f"  {safe_name}/: {len(clips)} clips")


if __name__ == "__main__":
    main()

