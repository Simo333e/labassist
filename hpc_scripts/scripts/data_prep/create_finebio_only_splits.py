#!/usr/bin/env python3
"""
Create train/val splits for FineBio PCR videos only (no mydata).
Uses 80/20 train/val split with random seed for reproducibility.
"""
import argparse
import json
import random
from pathlib import Path


def main():
    ap = argparse.ArgumentParser(description="Create FineBio-only splits")
    ap.add_argument("--labels_dir", required=True, help="FineBio labels directory")
    ap.add_argument("--output_splits", required=True, help="Output splits.json path")
    ap.add_argument("--val_fraction", type=float, default=0.2, help="Validation fraction (default: 0.2)")
    ap.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    args = ap.parse_args()

    labels_dir = Path(args.labels_dir)
    
    # Find all FineBio PCR label files (P*_05_*_T5_y.npy)
    label_files = sorted(labels_dir.glob("P*_05_*_T5_y.npy"))
    
    # Extract video stems (P01_05_01_T5.mp4 format)
    video_stems = [f.stem.replace("_y", "") + ".mp4" for f in label_files]
    
    print(f"Found {len(video_stems)} FineBio PCR videos")
    
    # Shuffle with fixed seed for reproducibility
    random.seed(args.seed)
    random.shuffle(video_stems)
    
    # Split into train/val
    n_val = max(1, int(len(video_stems) * args.val_fraction))
    val_stems = video_stems[:n_val]
    train_stems = video_stems[n_val:]
    
    splits = {
        "train": sorted(train_stems),
        "val": sorted(val_stems),
        "test": []
    }
    
    output_path = Path(args.output_splits)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w") as f:
        json.dump(splits, f, indent=2)
    
    print(f"\n✓ Created splits:")
    print(f"  Train: {len(train_stems)} videos")
    print(f"  Val:   {len(val_stems)} videos")
    print(f"  Test:  0 videos")
    print(f"\n✓ Saved to: {output_path}")


if __name__ == "__main__":
    main()


