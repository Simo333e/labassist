#!/usr/bin/env python3
"""
Extract frames for IMFE feature computation from full FineBio atomic dataset.
Extracts at 10 Hz sampling rate without any preprocessing (no crop/flip).
"""

import argparse
import json
from pathlib import Path
import cv2
import numpy as np
from tqdm import tqdm


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--finebio_root', type=str, required=True,
                        help='Root directory of FineBio videos')
    parser.add_argument('--splits', type=str, required=True,
                        help='JSON file with train/val/test splits')
    parser.add_argument('--out_dir', type=str, required=True,
                        help='Output directory for extracted frames')
    parser.add_argument('--target_fps', type=float, default=10.0,
                        help='Target sampling rate (Hz)')
    args = parser.parse_args()

    finebio_root = Path(args.finebio_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load splits to get list of all videos
    with open(args.splits, 'r') as f:
        splits = json.load(f)
    
    all_videos = splits['train'] + splits['val'] + splits['test']
    print(f"Found {len(all_videos)} videos in splits")

    # Process each video
    for video_name in tqdm(all_videos, desc="Extracting frames"):
        stem = Path(video_name).stem
        video_dir = out_dir / stem
        video_dir.mkdir(exist_ok=True)

        # Find video file
        vpath = finebio_root / video_name
        if not vpath.exists():
            print(f"Warning: Video not found: {vpath}")
            continue

        # Open video
        cap = cv2.VideoCapture(str(vpath))
        src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        nframes = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        
        if nframes <= 0:
            print(f"Warning: Could not read frame count for {stem}")
            cap.release()
            continue

        # Calculate frame indices for target_fps sampling
        T_out = int(np.ceil(nframes / src_fps * args.target_fps))
        centers = np.linspace(0, nframes - 1, T_out).astype(int)
        centers_set = set(centers)

        # Process frames one at a time
        frame_idx = 0
        frames_saved = 0
        ok = True
        
        while ok:
            ok, frame = cap.read()
            if ok and frame_idx in centers_set:
                # Save frame without any preprocessing
                out_path = video_dir / f"{frames_saved:06d}.png"
                cv2.imwrite(str(out_path), frame)
                frames_saved += 1
            frame_idx += 1

        cap.release()
        
        if frames_saved == 0:
            print(f"Warning: No frames saved for {stem}")
            continue

        print(f"  → Extracted {frames_saved} frames")

    print(f"\nDone! Extracted frames to: {out_dir}")


if __name__ == '__main__':
    main()

