#!/usr/bin/env python3
"""
Extract frames for IMFE feature extraction from combined_pcr dataset.
Handles both FineBio PCR videos and mydata videos with proper cropping/flipping.
"""
import argparse
import json
import math
import pathlib
import cv2
import numpy as np


def apply_preprocessing(frame, do_crop, do_vflip, do_hflip):
    """Apply same preprocessing as I3D features for consistency."""
    if do_crop:
        # Selective crop: bottom 100 pixels
        frame = frame[:-100, :, :]
    if do_vflip:
        frame = cv2.flip(frame, 0)
    if do_hflip:
        frame = cv2.flip(frame, 1)
    return frame


def get_preprocessing_for_video(stem):
    """
    Determine preprocessing for a video based on its stem.
    Mirrors the logic from I3D feature extraction.
    """
    # Mydata preprocessing logic
    mydata_crop_stems = {
        "20251027_181439Z",
        "20251112_110656Z_1080p30_v2",
        "20251125_070904Z", 
        "20251125_072321Z",
        "Lab_video_cut",
    }
    mydata_vflip_stems = {"20251027_181439Z"}
    mydata_hflip_stems = {"Lab_video_cut"}
    
    do_crop = stem in mydata_crop_stems
    do_vflip = stem in mydata_vflip_stems
    do_hflip = stem in mydata_hflip_stems
    
    return do_crop, do_vflip, do_hflip


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--finebio_root", required=True, help="FineBio root dir with videos/")
    ap.add_argument("--mydata_root", required=True, help="Mydata root dir with videos/")
    ap.add_argument("--splits", required=True, help="Combined PCR splits.json")
    ap.add_argument("--fps", type=float, default=10.0)
    ap.add_argument("--out", required=True, help="Output dir for rawframes_rgb")
    args = ap.parse_args()

    FINEBIO_ROOT = pathlib.Path(args.finebio_root)
    MYDATA_ROOT = pathlib.Path(args.mydata_root)
    OUT = pathlib.Path(args.out) / "rawframes_rgb"
    OUT.mkdir(parents=True, exist_ok=True)

    # Load splits
    sp = json.loads(pathlib.Path(args.splits).read_text())
    all_stems = []
    for split_name in ["train", "val", "test"]:
        if split_name in sp:
            all_stems.extend([pathlib.Path(p).stem for p in sp[split_name]])

    # Build video paths dict
    video_paths = {}
    
    # FineBio videos (P*_05_*_T5.mp4 pattern for PCR videos)
    for vpath in (FINEBIO_ROOT / "videos").glob("P*_05_*_T5.mp4"):
        video_paths[vpath.stem] = vpath
    
    # Mydata videos
    for vpath in (MYDATA_ROOT / "videos").glob("*.mp4"):
        video_paths[vpath.stem] = vpath

    print(f"Found {len(video_paths)} total videos")
    print(f"Processing {len(all_stems)} stems from splits")

    for stem in all_stems:
        vpath = video_paths.get(stem)
        if not vpath or not vpath.exists():
            print(f"Warning: Video not found for {stem}")
            continue

        out_dir = OUT / stem
        if (out_dir / "img_00000.jpg").exists():
            print(f"Skipping {stem} (already extracted)")
            continue

        print(f"Processing {stem}...")
        out_dir.mkdir(parents=True, exist_ok=True)

        # Get preprocessing flags
        do_crop, do_vflip, do_hflip = get_preprocessing_for_video(stem)
        
        # Open video
        cap = cv2.VideoCapture(str(vpath))
        src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        nframes = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        
        if nframes <= 0:
            print(f"Warning: Could not read frame count for {stem}")
            cap.release()
            continue

        # Calculate which frame indices we need to sample at 10 Hz
        T_out = int(math.ceil((nframes / src_fps) * args.fps))
        target_indices = set([min(nframes - 1, int(round((t / args.fps) * src_fps))) for t in range(T_out)])
        
        # Create mapping: video_frame_idx -> output_frame_idx
        idx_to_out = {}
        for out_idx, vid_idx in enumerate(sorted(target_indices)):
            idx_to_out[vid_idx] = out_idx
        
        # Read and save frames one at a time (memory-efficient)
        frame_idx = 0
        frames_saved = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            
            # Only process and save frames we need
            if frame_idx in idx_to_out:
                # Apply preprocessing
                frame = apply_preprocessing(frame, do_crop, do_vflip, do_hflip)
                
                # Save frame
                out_idx = idx_to_out[frame_idx]
                fpath = out_dir / f"img_{out_idx:05d}.jpg"
                cv2.imwrite(str(fpath), frame)
                frames_saved += 1
            
            frame_idx += 1
        
        cap.release()
        
        if frames_saved == 0:
            print(f"Warning: No frames saved for {stem}")
            continue

        print(f"  → Extracted {frames_saved} frames (crop={do_crop}, vflip={do_vflip}, hflip={do_hflip})")


if __name__ == "__main__":
    main()

