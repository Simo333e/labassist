#!/usr/bin/env python3
"""
Extract high-resolution frames from videos with preprocessing (crop + flip).

This creates a set of frames that are:
1. High resolution (1080p or original size) - needed for YOLO to see small objects
2. Pre-cropped and flipped (matching I3D preprocessing) - so YOLO sees same view
3. Extracted at 25 FPS (matching I3D frame extraction)

Output: frames_hires_preprocessed/<stem>/%06d.jpg

Uses OpenCV for better codec compatibility than ffmpeg on some systems.
"""
import argparse
import json
import pathlib
import re
from typing import Optional

import cv2
import numpy as np
from tqdm import tqdm

# Pattern to identify FineBio stems (P##_##_##_T# format)
FINEBIO_PATTERN = re.compile(r'^P\d{2}_\d{2}_\d{2}_T\d')


def is_mydata_stem(stem: str) -> bool:
    """Check if stem belongs to mydata (needs cropping) vs FineBio (no crop)."""
    if FINEBIO_PATTERN.match(stem):
        return False
    return True


def find_video_file(videos_dir: pathlib.Path, stem: str) -> Optional[pathlib.Path]:
    """Find video file matching stem."""
    for ext in ['.mp4', '.avi', '.mov', '.mkv', '.MP4', '.AVI', '.MOV']:
        path = videos_dir / f"{stem}{ext}"
        if path.exists():
            return path
    return None


def apply_crop(img: np.ndarray, crop_top: float, crop_bottom: float,
               crop_left: float, crop_right: float) -> np.ndarray:
    """Crop image by percentage from each edge."""
    if crop_top == 0 and crop_bottom == 0 and crop_left == 0 and crop_right == 0:
        return img
    h, w = img.shape[:2]
    top = int(h * crop_top)
    bottom = int(h * (1 - crop_bottom))
    left = int(w * crop_left)
    right = int(w * (1 - crop_right))
    return img[top:bottom, left:right]


def extract_and_preprocess_frames(
    video_path: pathlib.Path,
    output_dir: pathlib.Path,
    target_fps: float = 25.0,
    crop_top: float = 0.0,
    crop_bottom: float = 0.0,
    crop_left: float = 0.0,
    crop_right: float = 0.0,
    vertical_flip: bool = False,
    horizontal_flip: bool = False,
    max_short_side: int = 1080,
    quality: int = 95,
) -> int:
    """
    Extract frames from video with preprocessing using OpenCV.
    
    Returns: number of frames extracted
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"  ERROR: Cannot open video {video_path}")
        return 0
    
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if video_fps <= 0 or total_frames <= 0:
        print(f"  ERROR: Invalid video metadata (fps={video_fps}, frames={total_frames})")
        cap.release()
        return 0
    
    # Calculate frame interval
    frame_interval = video_fps / target_fps
    
    # Calculate total output frames
    duration = total_frames / video_fps
    n_output_frames = int(duration * target_fps)
    
    frame_count = 0
    for out_idx in tqdm(range(n_output_frames), desc=f"  Extracting", leave=False):
        # Which video frame to read
        video_frame_idx = int(out_idx * frame_interval)
        
        cap.set(cv2.CAP_PROP_POS_FRAMES, video_frame_idx)
        ret, frame = cap.read()
        
        if not ret or frame is None:
            continue
        
        # Apply crop
        frame = apply_crop(frame, crop_top, crop_bottom, crop_left, crop_right)
        
        # Apply flips
        if vertical_flip:
            frame = cv2.flip(frame, 0)
        if horizontal_flip:
            frame = cv2.flip(frame, 1)
        
        # Resize if needed (keep aspect ratio, limit short side)
        h, w = frame.shape[:2]
        short_side = min(h, w)
        if short_side > max_short_side:
            scale = max_short_side / short_side
            new_w = int(w * scale)
            new_h = int(h * scale)
            frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
        
        # Save frame (1-indexed like ffmpeg)
        out_path = output_dir / f"{out_idx + 1:06d}.jpg"
        cv2.imwrite(str(out_path), frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
        frame_count += 1
    
    cap.release()
    return frame_count


def main():
    ap = argparse.ArgumentParser(description="Extract high-res preprocessed frames")
    ap.add_argument("--root", required=True, help="Dataset root containing videos/")
    ap.add_argument("--splits", required=True, help="Path to splits.json")
    ap.add_argument("--fps", type=float, default=25.0, help="Target FPS")
    ap.add_argument("--max_short_side", type=int, default=1080, 
                    help="Maximum short side (keeps 1080p for YOLO)")
    ap.add_argument("--quality", type=int, default=95, help="JPEG quality (0-100)")
    ap.add_argument("--out", default=None, help="Output directory for frames")
    
    # Cropping parameters
    ap.add_argument("--crop_top", type=float, default=0.0)
    ap.add_argument("--crop_bottom", type=float, default=0.0)
    ap.add_argument("--crop_left", type=float, default=0.0)
    ap.add_argument("--crop_right", type=float, default=0.0)
    ap.add_argument("--selective_crop", action="store_true",
                    help="Only apply crop to mydata stems")
    ap.add_argument("--vertical_flip", action="store_true")
    ap.add_argument("--horizontal_flip", action="store_true")
    
    # Sharding for parallel processing
    ap.add_argument("--shards", type=int, default=1)
    ap.add_argument("--shard_idx", type=int, default=0)
    
    args = ap.parse_args()
    
    ROOT = pathlib.Path(args.root)
    videos_dir = ROOT / "videos"
    OUT = pathlib.Path(args.out or (ROOT / "frames_hires_preprocessed"))
    
    if not videos_dir.exists():
        raise SystemExit(f"Videos directory not found: {videos_dir}")
    
    # Check crop parameters
    has_crop = (args.crop_top > 0 or args.crop_bottom > 0 or 
                args.crop_left > 0 or args.crop_right > 0)
    
    print("=" * 60)
    print("High-Resolution Preprocessed Frame Extraction")
    print("=" * 60)
    print(f"Videos: {videos_dir}")
    print(f"Output: {OUT}")
    print(f"FPS: {args.fps}")
    print(f"Max short side: {args.max_short_side}")
    print(f"JPEG quality: {args.quality}")
    if has_crop:
        crop_str = f"top={args.crop_top:.0%}, bottom={args.crop_bottom:.0%}, "
        crop_str += f"left={args.crop_left:.0%}, right={args.crop_right:.0%}"
        if args.selective_crop:
            print(f"Selective crop (mydata only): {crop_str}")
        else:
            print(f"Crop: {crop_str}")
    if args.vertical_flip:
        print("Vertical flip: enabled (mydata only)")
    if args.horizontal_flip:
        print("Horizontal flip: enabled (mydata only)")
    print("=" * 60)
    
    # Load splits
    sp = json.loads(pathlib.Path(args.splits).read_text())
    sp = sp[next(iter(sp))] if "train" not in sp else sp
    stems_all = [pathlib.Path(p).stem for p in 
                 sp.get("train", []) + sp.get("val", []) + sp.get("test", [])]
    
    # Sharding
    S = max(1, args.shards)
    I = args.shard_idx
    if I < 0 or I >= S:
        raise SystemExit(f"shard_idx {I} out of range for shards={S}")
    stems = [s for i, s in enumerate(stems_all) if (i % S) == I]
    
    print(f"Processing {len(stems)} stems (shard {I}/{S})\n")
    
    for stem in stems:
        out_dir = OUT / stem
        
        # Skip if already extracted
        if out_dir.exists() and len(list(out_dir.glob('*.jpg'))) > 100:
            n_existing = len(list(out_dir.glob('*.jpg')))
            print(f"⏭ Skipping {stem} ({n_existing} frames already exist)")
            continue
        
        # Find video
        video_path = find_video_file(videos_dir, stem)
        if video_path is None:
            print(f"⚠ No video found for {stem}, skipping")
            continue
        
        # Determine preprocessing for this stem
        is_mydata = is_mydata_stem(stem)
        
        if args.selective_crop:
            apply_crop_flag = is_mydata and has_crop
        else:
            apply_crop_flag = has_crop
        
        crop_top = args.crop_top if apply_crop_flag else 0.0
        crop_bottom = args.crop_bottom if apply_crop_flag else 0.0
        crop_left = args.crop_left if apply_crop_flag else 0.0
        crop_right = args.crop_right if apply_crop_flag else 0.0
        
        apply_vflip = args.vertical_flip and is_mydata
        apply_hflip = args.horizontal_flip and is_mydata
        
        # Extract frames
        print(f"📹 {stem}:")
        n_frames = extract_and_preprocess_frames(
            video_path=video_path,
            output_dir=out_dir,
            target_fps=args.fps,
            crop_top=crop_top,
            crop_bottom=crop_bottom,
            crop_left=crop_left,
            crop_right=crop_right,
            vertical_flip=apply_vflip,
            horizontal_flip=apply_hflip,
            max_short_side=args.max_short_side,
            quality=args.quality,
        )
        
        # Build indicator string
        indicators = []
        if apply_crop_flag:
            indicators.append("cropped")
        if apply_vflip:
            indicators.append("vflip")
        if apply_hflip:
            indicators.append("hflip")
        indicator_str = f" [{'+'.join(indicators)}]" if indicators else ""
        
        print(f"   ✓ {n_frames} frames{indicator_str}")
    
    print(f"\n✓ Frame extraction complete. Output: {OUT}")


if __name__ == "__main__":
    main()
