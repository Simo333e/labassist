#!/usr/bin/env python3
"""
Generate K-fold cross-validation splits for mydata videos only.

This script creates K different train/val splits where each fold
holds out one mydata video for validation.

Usage:
    python create_mydata_kfold_splits.py \
        --splits /path/to/combined_pcr/cache/splits.json \
        --outdir /path/to/combined_pcr/cache/kfold_mydata
"""

import argparse
import json
import re
from pathlib import Path


# Pattern to identify FineBio stems (P##_##_##_T# format)
FINEBIO_PATTERN = re.compile(r'^P\d{2}_\d{2}_\d{2}_T\d')


def is_mydata_stem(stem: str) -> bool:
    """
    Check if stem belongs to mydata (not FineBio).
    
    mydata stems: timestamp format (YYYYMMDD_*) or Lab_video*
    FineBio stems: P##_##_##_T# format (e.g., P01_05_01_T5)
    
    Returns True for mydata, False for FineBio.
    """
    if FINEBIO_PATTERN.match(stem):
        return False  # FineBio
    return True  # mydata


def stem_from_video(video_path: str) -> str:
    """Extract stem from video path (remove extension)."""
    return Path(video_path).stem


def main():
    ap = argparse.ArgumentParser(description="Generate K-fold splits for mydata only")
    ap.add_argument("--splits", required=True, help="Path to original splits.json")
    ap.add_argument("--outdir", required=True, help="Output directory for fold splits")
    ap.add_argument("--k", type=int, default=None, 
                    help="Number of folds (default: Leave-One-Out, K=number of mydata videos)")
    args = ap.parse_args()

    # Load original splits
    with open(args.splits) as f:
        original_splits = json.load(f)

    # Get all stems from train + val + test
    all_videos = (
        original_splits.get("train", []) + 
        original_splits.get("val", []) + 
        original_splits.get("test", [])
    )
    all_stems = [stem_from_video(v) for v in all_videos]

    # Filter to mydata only
    mydata_stems = sorted([s for s in all_stems if is_mydata_stem(s)])
    
    print(f"Found {len(mydata_stems)} mydata videos:")
    for i, stem in enumerate(mydata_stems):
        print(f"  {i}: {stem}")

    # Determine K
    K = args.k if args.k else len(mydata_stems)
    if K > len(mydata_stems):
        raise ValueError(f"K={K} is larger than number of mydata videos ({len(mydata_stems)})")
    
    print(f"\nCreating {K}-fold cross-validation splits (Leave-One-Out)")

    # Create output directory
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Generate K folds
    for fold_idx in range(K):
        # For Leave-One-Out: each fold holds out one video
        # For K < num_videos: distribute videos across folds
        if K == len(mydata_stems):
            # Leave-One-Out
            val_stems = [mydata_stems[fold_idx]]
            train_stems = [s for i, s in enumerate(mydata_stems) if i != fold_idx]
        else:
            # K-fold with multiple videos per fold
            fold_size = len(mydata_stems) // K
            start_idx = fold_idx * fold_size
            end_idx = start_idx + fold_size if fold_idx < K - 1 else len(mydata_stems)
            val_stems = mydata_stems[start_idx:end_idx]
            train_stems = [s for s in mydata_stems if s not in val_stems]

        # Create splits dict (using .mp4 extension to match original format)
        splits = {
            "train": [f"{s}.mp4" for s in train_stems],
            "val": [f"{s}.mp4" for s in val_stems],
            "test": []
        }

        # Save fold splits
        fold_path = outdir / f"splits_fold{fold_idx}.json"
        with open(fold_path, "w") as f:
            json.dump(splits, f, indent=2)

        print(f"\nFold {fold_idx}:")
        print(f"  Train: {len(train_stems)} videos")
        print(f"  Val:   {len(val_stems)} videos → {val_stems}")
        print(f"  Saved: {fold_path}")

    # Also save a metadata file
    meta = {
        "K": K,
        "type": "Leave-One-Out" if K == len(mydata_stems) else f"{K}-fold",
        "mydata_stems": mydata_stems,
        "folds": [
            {
                "fold_idx": i,
                "val_stems": [mydata_stems[i]] if K == len(mydata_stems) else mydata_stems[i*(len(mydata_stems)//K):(i+1)*(len(mydata_stems)//K)],
                "splits_file": f"splits_fold{i}.json"
            }
            for i in range(K)
        ]
    }
    meta_path = outdir / "kfold_meta.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"\nMetadata saved: {meta_path}")

    print(f"\n✓ Created {K} fold splits in {outdir}")


if __name__ == "__main__":
    main()

