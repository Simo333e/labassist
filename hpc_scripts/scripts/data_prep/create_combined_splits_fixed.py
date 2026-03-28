#!/usr/bin/env python3
"""
Create combined splits for FineBio + FIXED mydata.

This script:
1. Takes FineBio PCR videos (P##_05_##_T5 stems)
2. Adds the FIXED mydata videos
3. Creates train/val/test splits with mydata in validation

The validation set contains mydata videos to test transfer from FineBio.
"""

import argparse
import json
import pathlib
import re


# Pattern to identify FineBio stems
FINEBIO_PATTERN = re.compile(r'^P\d{2}_\d{2}_\d{2}_T\d')


def is_finebio_stem(stem: str) -> bool:
    """Check if stem is a FineBio video."""
    return bool(FINEBIO_PATTERN.match(stem))


def main():
    ap = argparse.ArgumentParser(description="Create combined splits for FineBio + fixed mydata")
    ap.add_argument("--finebio_features_dir", required=True, 
                    help="Directory containing FineBio I3D features (*_x.npy)")
    ap.add_argument("--mydata_features_dir", required=True,
                    help="Directory containing FIXED mydata I3D features (*_x.npy)")
    ap.add_argument("--mydata_val_stems", default="20251125_070904Z",
                    help="Comma-separated list of mydata stems for validation")
    ap.add_argument("--exclude_stems", default="",
                    help="Comma-separated list of stems to exclude (e.g., broken pipette)")
    ap.add_argument("--output_splits", required=True,
                    help="Output path for combined splits.json")
    args = ap.parse_args()
    
    finebio_dir = pathlib.Path(args.finebio_features_dir)
    mydata_dir = pathlib.Path(args.mydata_features_dir)
    
    # Get all stems from feature files
    # Filter finebio_stems to ONLY include actual FineBio stems (P##_##_##_T# pattern)
    all_finebio_candidates = sorted([p.stem.replace("_x", "") for p in finebio_dir.glob("*_x.npy")])
    finebio_stems = [s for s in all_finebio_candidates if is_finebio_stem(s)]
    mydata_stems = sorted([p.stem.replace("_x", "") for p in mydata_dir.glob("*_x.npy")])
    
    print(f"Filtered out {len(all_finebio_candidates) - len(finebio_stems)} non-FineBio stems from finebio_features_dir")
    
    # Parse validation and exclusion lists
    val_stems = set(s.strip() for s in args.mydata_val_stems.split(",") if s.strip())
    exclude_stems = set(s.strip() for s in args.exclude_stems.split(",") if s.strip())
    
    print(f"Found {len(finebio_stems)} FineBio stems")
    print(f"Found {len(mydata_stems)} mydata stems")
    print(f"Validation stems: {val_stems}")
    print(f"Excluded stems: {exclude_stems}")
    
    # Build splits
    train = []
    val = []
    test = []
    
    # All FineBio goes to train
    for stem in finebio_stems:
        if stem in exclude_stems:
            print(f"  Excluding: {stem}")
            continue
        train.append(f"{stem}.mp4")
    
    # Mydata: specified stems to val, rest to train
    for stem in mydata_stems:
        if stem in exclude_stems:
            print(f"  Excluding: {stem}")
            continue
        if stem in val_stems:
            val.append(f"{stem}.mp4")
        else:
            train.append(f"{stem}.mp4")
    
    splits = {
        "train": train,
        "val": val,
        "test": test
    }
    
    print(f"\nSplits:")
    print(f"  Train: {len(train)} videos ({len([s for s in train if is_finebio_stem(s.replace('.mp4', ''))])} FineBio, {len(train) - len([s for s in train if is_finebio_stem(s.replace('.mp4', ''))])} mydata)")
    print(f"  Val: {len(val)} videos (mydata)")
    print(f"  Test: {len(test)} videos")
    
    # Save
    output_path = pathlib.Path(args.output_splits)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(splits, f, indent=2)
    
    print(f"\n✓ Saved splits to: {output_path}")


if __name__ == "__main__":
    main()


