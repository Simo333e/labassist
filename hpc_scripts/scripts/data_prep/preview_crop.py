#!/usr/bin/env python3
"""
Preview crop parameters on sample frames before applying to feature extraction.

This script helps tune cropping parameters for the mydata portion of combined_pcr.
It shows side-by-side comparisons of original vs cropped frames.

Usage:
    # Preview with default parameters
    python preview_crop.py --frames_dir /path/to/frames/video_stem --output crop_preview.png

    # Tune crop parameters
    python preview_crop.py --frames_dir /path/to/frames/video_stem \
        --crop_top 0.05 --crop_bottom 0.35 --crop_left 0.12 --crop_right 0.08

    # Preview multiple videos from a directory
    python preview_crop.py --frames_root /path/to/frames --num_videos 3

Example for mydata:
    python preview_crop.py \
        --frames_dir /work3/s204157/data/labassist/mydata/frames/20251027_181439Z \
        --crop_top 0.0 --crop_bottom 0.35 --crop_left 0.10 --crop_right 0.08 \
        --output crop_preview_mydata.png
"""
import argparse
import pathlib
import random
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from PIL import Image


def apply_crop(img: np.ndarray, crop_top: float, crop_bottom: float, 
               crop_left: float, crop_right: float) -> np.ndarray:
    """
    Crop image by percentage from each edge.
    
    Args:
        img: Input image array [H, W, C]
        crop_top: Fraction to remove from top (0.0-1.0)
        crop_bottom: Fraction to remove from bottom (0.0-1.0)
        crop_left: Fraction to remove from left (0.0-1.0)
        crop_right: Fraction to remove from right (0.0-1.0)
    
    Returns:
        Cropped image array
    """
    h, w = img.shape[:2]
    top = int(h * crop_top)
    bottom = int(h * (1 - crop_bottom))
    left = int(w * crop_left)
    right = int(w * (1 - crop_right))
    return img[top:bottom, left:right]


def get_crop_box(h: int, w: int, crop_top: float, crop_bottom: float,
                 crop_left: float, crop_right: float) -> tuple:
    """Get crop box coordinates for drawing rectangle."""
    top = int(h * crop_top)
    bottom = int(h * (1 - crop_bottom))
    left = int(w * crop_left)
    right = int(w * (1 - crop_right))
    return left, top, right - left, bottom - top


def load_sample_frames(frames_dir: pathlib.Path, num_samples: int = 6) -> list:
    """Load evenly spaced sample frames from a directory."""
    frame_files = sorted(frames_dir.glob("*.jpg"))
    if not frame_files:
        frame_files = sorted(frames_dir.glob("*.png"))
    
    if not frame_files:
        return []
    
    # Sample evenly across the video
    if len(frame_files) <= num_samples:
        indices = list(range(len(frame_files)))
    else:
        indices = np.linspace(0, len(frame_files) - 1, num_samples, dtype=int)
    
    samples = []
    for idx in indices:
        img = np.array(Image.open(frame_files[idx]))
        samples.append((frame_files[idx].name, img))
    
    return samples


def create_preview_figure(samples: list, crop_top: float, crop_bottom: float,
                          crop_left: float, crop_right: float, 
                          title_prefix: str = "") -> plt.Figure:
    """Create a figure showing original vs cropped frames."""
    n = len(samples)
    if n == 0:
        return None
    
    fig, axes = plt.subplots(n, 2, figsize=(14, 3.5 * n))
    if n == 1:
        axes = axes.reshape(1, 2)
    
    fig.suptitle(
        f"{title_prefix}Crop Preview: top={crop_top:.0%}, bottom={crop_bottom:.0%}, "
        f"left={crop_left:.0%}, right={crop_right:.0%}",
        fontsize=14, fontweight='bold'
    )
    
    for i, (name, img) in enumerate(samples):
        h, w = img.shape[:2]
        cropped = apply_crop(img, crop_top, crop_bottom, crop_left, crop_right)
        ch, cw = cropped.shape[:2]
        
        # Original with crop box overlay
        axes[i, 0].imshow(img)
        axes[i, 0].set_title(f"Original {w}×{h} - {name}", fontsize=10)
        axes[i, 0].axis('off')
        
        # Draw crop box
        box_x, box_y, box_w, box_h = get_crop_box(h, w, crop_top, crop_bottom, crop_left, crop_right)
        rect = patches.Rectangle(
            (box_x, box_y), box_w, box_h,
            linewidth=3, edgecolor='lime', facecolor='none', linestyle='-'
        )
        axes[i, 0].add_patch(rect)
        
        # Shade the cropped-out regions
        # Top
        if crop_top > 0:
            axes[i, 0].axhspan(0, h * crop_top, alpha=0.4, color='red')
        # Bottom
        if crop_bottom > 0:
            axes[i, 0].axhspan(h * (1 - crop_bottom), h, alpha=0.4, color='red')
        # Left
        if crop_left > 0:
            axes[i, 0].axvspan(0, w * crop_left, alpha=0.4, color='red')
        # Right
        if crop_right > 0:
            axes[i, 0].axvspan(w * (1 - crop_right), w, alpha=0.4, color='red')
        
        # Cropped result
        axes[i, 1].imshow(cropped)
        axes[i, 1].set_title(f"Cropped {cw}×{ch} ({100*cw*ch/(w*h):.1f}% of original)", fontsize=10)
        axes[i, 1].axis('off')
    
    plt.tight_layout()
    return fig


def main():
    ap = argparse.ArgumentParser(
        description="Preview crop parameters for feature extraction",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    # Input options (mutually exclusive)
    input_group = ap.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--frames_dir", type=str,
                             help="Directory with frame images for a single video")
    input_group.add_argument("--frames_root", type=str,
                             help="Root directory containing multiple video frame directories")
    
    # Crop parameters - good defaults for mydata based on your image
    ap.add_argument("--crop_top", type=float, default=0.0,
                    help="Fraction to crop from top (default: 0.0)")
    ap.add_argument("--crop_bottom", type=float, default=0.35,
                    help="Fraction to crop from bottom (default: 0.35 - removes glass shelf)")
    ap.add_argument("--crop_left", type=float, default=0.10,
                    help="Fraction to crop from left (default: 0.10 - removes equipment)")
    ap.add_argument("--crop_right", type=float, default=0.08,
                    help="Fraction to crop from right (default: 0.08 - removes bottles)")
    
    # Sampling options
    ap.add_argument("--num_samples", type=int, default=6,
                    help="Number of sample frames per video (default: 6)")
    ap.add_argument("--num_videos", type=int, default=3,
                    help="Number of videos to sample when using --frames_root (default: 3)")
    
    # Output
    ap.add_argument("--output", type=str, default="crop_preview.png",
                    help="Output preview image path (default: crop_preview.png)")
    
    args = ap.parse_args()
    
    # Validate crop parameters
    if not (0 <= args.crop_top < 1):
        sys.exit(f"crop_top must be in [0, 1), got {args.crop_top}")
    if not (0 <= args.crop_bottom < 1):
        sys.exit(f"crop_bottom must be in [0, 1), got {args.crop_bottom}")
    if not (0 <= args.crop_left < 1):
        sys.exit(f"crop_left must be in [0, 1), got {args.crop_left}")
    if not (0 <= args.crop_right < 1):
        sys.exit(f"crop_right must be in [0, 1), got {args.crop_right}")
    if args.crop_top + args.crop_bottom >= 1:
        sys.exit(f"crop_top + crop_bottom must be < 1")
    if args.crop_left + args.crop_right >= 1:
        sys.exit(f"crop_left + crop_right must be < 1")
    
    # Collect frames
    all_samples = []
    
    if args.frames_dir:
        frames_dir = pathlib.Path(args.frames_dir)
        if not frames_dir.exists():
            sys.exit(f"Frames directory not found: {frames_dir}")
        
        samples = load_sample_frames(frames_dir, args.num_samples)
        if not samples:
            sys.exit(f"No frame images found in {frames_dir}")
        
        all_samples = samples
        title_prefix = f"{frames_dir.name}: "
        
    else:  # --frames_root
        frames_root = pathlib.Path(args.frames_root)
        if not frames_root.exists():
            sys.exit(f"Frames root not found: {frames_root}")
        
        # Find video directories
        video_dirs = [d for d in sorted(frames_root.iterdir()) if d.is_dir()]
        if not video_dirs:
            sys.exit(f"No video directories found in {frames_root}")
        
        # Sample videos evenly
        if len(video_dirs) <= args.num_videos:
            selected_dirs = video_dirs
        else:
            indices = np.linspace(0, len(video_dirs) - 1, args.num_videos, dtype=int)
            selected_dirs = [video_dirs[i] for i in indices]
        
        for vdir in selected_dirs:
            samples = load_sample_frames(vdir, max(2, args.num_samples // args.num_videos))
            for name, img in samples:
                all_samples.append((f"{vdir.name}/{name}", img))
        
        title_prefix = f"{frames_root.name}: "
    
    if not all_samples:
        sys.exit("No frames loaded")
    
    print(f"Loaded {len(all_samples)} sample frames")
    print(f"Crop parameters:")
    print(f"  top:    {args.crop_top:.0%}")
    print(f"  bottom: {args.crop_bottom:.0%}")
    print(f"  left:   {args.crop_left:.0%}")
    print(f"  right:  {args.crop_right:.0%}")
    
    # Calculate effective resolution
    if all_samples:
        h, w = all_samples[0][1].shape[:2]
        new_h = int(h * (1 - args.crop_top - args.crop_bottom))
        new_w = int(w * (1 - args.crop_left - args.crop_right))
        print(f"\nOriginal size: {w}×{h}")
        print(f"Cropped size:  {new_w}×{new_h} ({100*new_w*new_h/(w*h):.1f}% of pixels)")
    
    # Create and save preview
    fig = create_preview_figure(
        all_samples, 
        args.crop_top, args.crop_bottom, 
        args.crop_left, args.crop_right,
        title_prefix
    )
    
    if fig is None:
        sys.exit("Failed to create preview figure")
    
    output_path = pathlib.Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    
    print(f"\n✓ Preview saved to: {output_path}")
    print(f"\nTo apply these crops during feature extraction, use:")
    print(f"  --crop_top {args.crop_top} --crop_bottom {args.crop_bottom} "
          f"--crop_left {args.crop_left} --crop_right {args.crop_right}")


if __name__ == "__main__":
    main()




