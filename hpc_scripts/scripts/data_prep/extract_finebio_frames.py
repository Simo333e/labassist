#!/usr/bin/env python3
"""
Extract frames from FineBio T5 videos at 25fps.

Usage:
    python extract_finebio_frames.py \
        --video_dir /path/to/finebio_videos_w640 \
        --output_dir /path/to/finebio_full/frames \
        --fps 25 \
        --view T5 \
        --shards 4 --shard_idx 0
"""

import argparse
import subprocess
from pathlib import Path


def extract_frames(video_path: Path, output_dir: Path, fps: float = 25.0):
    """Extract frames from video using ffmpeg."""
    stem = video_path.stem
    frame_dir = output_dir / stem
    frame_dir.mkdir(parents=True, exist_ok=True)
    
    # Check if already extracted
    existing = list(frame_dir.glob("*.jpg"))
    if len(existing) > 10:
        print(f"⏭ Skipping {stem} ({len(existing)} frames exist)")
        return len(existing)
    
    # ffmpeg command to extract frames
    # Output: 000001.jpg, 000002.jpg, etc. (1-indexed, 6 digits)
    cmd = [
        "ffmpeg",
        "-i", str(video_path),
        "-vf", f"fps={fps}",
        "-q:v", "2",  # High quality JPEG
        "-start_number", "1",
        str(frame_dir / "%06d.jpg"),
        "-y",  # Overwrite
        "-loglevel", "error"
    ]
    
    try:
        subprocess.run(cmd, check=True)
        n_frames = len(list(frame_dir.glob("*.jpg")))
        print(f"✓ {stem}: {n_frames} frames extracted")
        return n_frames
    except subprocess.CalledProcessError as e:
        print(f"✗ {stem}: ffmpeg failed - {e}")
        return 0


def main():
    ap = argparse.ArgumentParser(description="Extract frames from FineBio videos")
    ap.add_argument("--video_dir", required=True, help="Directory containing FineBio videos")
    ap.add_argument("--output_dir", required=True, help="Output directory for frames")
    ap.add_argument("--fps", type=float, default=25.0, help="Frame extraction FPS")
    ap.add_argument("--view", default="T5", help="Camera view to extract (e.g., T5)")
    ap.add_argument("--shards", type=int, default=1, help="Number of shards for parallel processing")
    ap.add_argument("--shard_idx", type=int, default=0, help="Shard index (0-based)")
    args = ap.parse_args()

    video_dir = Path(args.video_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Find all videos matching the view
    pattern = f"*_{args.view}.mp4"
    videos = sorted(video_dir.glob(pattern))
    print(f"Found {len(videos)} {args.view} videos")

    # Shard the videos
    if args.shards > 1:
        videos = [v for i, v in enumerate(videos) if i % args.shards == args.shard_idx]
        print(f"Processing shard {args.shard_idx}/{args.shards}: {len(videos)} videos")

    # Extract frames
    total_frames = 0
    for i, video_path in enumerate(videos):
        print(f"[{i+1}/{len(videos)}] Processing {video_path.name}...")
        n = extract_frames(video_path, output_dir, args.fps)
        total_frames += n

    print(f"\n✓ Extraction complete. Total frames: {total_frames}")
    print(f"Output directory: {output_dir}")


if __name__ == "__main__":
    main()

