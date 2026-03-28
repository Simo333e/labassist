#!/usr/bin/env python3
"""Generate domain shift visual comparison figure.

This script extracts one frame from 4 different videos in each domain (FineBio and DTU Lab)
to show visual differences between the two datasets.

To customize which frames to use from DTU Lab videos:
- Edit the 'dtu_samples' list in the main() function
- Change the 'frame' value for each video to select different moments
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def extract_frame(video_path: Path, frame_idx: int, crop_params=None):
    """Extract a single frame from video."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None
    
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    cap.release()
    
    if ret:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        if crop_params:
            h, w = frame.shape[:2]
            top = int(h * crop_params.get('top', 0))
            bottom = int(h * (1 - crop_params.get('bottom', 0)))
            left = int(w * crop_params.get('left', 0))
            right = int(w * (1 - crop_params.get('right', 0)))
            frame = frame[top:bottom, left:right]
        return frame
    return None

def main():
    output_dir = Path('/zhome/45/8/156251/Desktop/thesis/labassist/hpc_scripts/scripts/analysis/thesis_figures/output')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("\nGenerating domain shift figure...")
    
    # Paths to videos
    finebio_videos = Path('/work3/s204157/data/labassist/finebio_videos_w640')
    dtu_videos = Path('/work3/s204157/data/labassist/my_data/videos')
    
    # DTU crop parameters (matching your preprocessing)
    dtu_crop = {'top': 0.0, 'bottom': 0.25, 'left': 0.10, 'right': 0.08}
    
    # Define 4 different FineBio videos with representative frames
    finebio_samples = [
        {'video': 'P01_05_01_T5.mp4', 'frame': 500},
        {'video': 'P03_05_01_T5.mp4', 'frame': 1000},
        {'video': 'P08_05_01_T5.mp4', 'frame': 800},
        {'video': 'P09_05_01_T5.mp4', 'frame': 600},
    ]
    
    # Define 4 different DTU Lab videos with CUSTOMIZABLE frames
    # EDIT THESE FRAME NUMBERS to select specific moments from each video:
    dtu_samples = [
        {'video': '20251125_072321Z.mp4', 'frame': 6000},
        {'video': '20251112_110656Z_1080p30_v2.mp4', 'frame': 50000},
        {'video': '20251120_065442Z.mp4', 'frame': 8000},
        {'video': '20251125_070904Z.mp4', 'frame': 10000},
    ]
    
    
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    
    # Extract FineBio frames (4 different videos)
    print(f"  Extracting FineBio frames from 4 different videos...")
    for j, sample in enumerate(finebio_samples):
        video_path = finebio_videos / sample['video']
        frame_idx = sample['frame']
        
        if video_path.exists():
            frame = extract_frame(video_path, frame_idx)
            if frame is not None:
                axes[0, j].imshow(frame)
                axes[0, j].set_title(f"{sample['video']}\nFrame {frame_idx}", fontsize=8)
                axes[0, j].axis('off')
                print(f"    ✓ {sample['video']} (frame {frame_idx})")
            else:
                axes[0, j].text(0.5, 0.5, 'Failed to load', ha='center', va='center')
                axes[0, j].axis('off')
                print(f"    ✗ Failed to extract frame from {sample['video']}")
        else:
            axes[0, j].text(0.5, 0.5, 'Video not found', ha='center', va='center')
            axes[0, j].axis('off')
            print(f"    ✗ Video not found: {video_path}")
    
    # Extract DTU frames (4 different videos, with crop)
    print(f"  Extracting DTU frames from 4 different videos...")
    for j, sample in enumerate(dtu_samples):
        video_path = dtu_videos / sample['video']
        frame_idx = sample['frame']
        
        if video_path.exists():
            frame = extract_frame(video_path, frame_idx, dtu_crop)
            if frame is not None:
                frame = cv2.flip(frame, 0)
                axes[1, j].imshow(frame)
                axes[1, j].set_title(f"{sample['video']}\nFrame {frame_idx}", fontsize=8)
                axes[1, j].axis('off')
                print(f"    ✓ {sample['video']} (frame {frame_idx})")
            else:
                axes[1, j].text(0.5, 0.5, 'Failed to load', ha='center', va='center')
                axes[1, j].axis('off')
                print(f"    ✗ Failed to extract frame from {sample['video']}")
        else:
            axes[1, j].text(0.5, 0.5, 'Video not found', ha='center', va='center')
            axes[1, j].axis('off')
            print(f"    ✗ Video not found: {video_path}")
    
    # Add row labels
    axes[0, 0].text(-0.1, 0.5, 'FineBio', transform=axes[0, 0].transAxes,
                    fontsize=14, fontweight='bold', va='center', ha='right')
    axes[1, 0].text(-0.1, 0.5, 'DTU Lab', transform=axes[1, 0].transAxes,
                    fontsize=14, fontweight='bold', va='center', ha='right')
    
    plt.suptitle('Domain Shift: Visual Comparison Between FineBio and DTU Lab Recordings',
                 fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    output_path = output_dir / 'fig5_domain_shift.png'
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    fig.savefig(output_path.with_suffix('.pdf'), bbox_inches='tight')
    plt.close(fig)
    
    print(f"\n✓ Domain shift figure saved to:")
    print(f"  - {output_path}")

if __name__ == '__main__':
    main()

