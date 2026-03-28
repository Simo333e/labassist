#!/usr/bin/env python3
"""
Create a modified splits.json that excludes specified videos from training.

This is useful for removing problematic videos (e.g., broken equipment)
that might confuse the model.

Usage:
    python create_splits_exclude_broken.py \
        --input_splits /path/to/splits.json \
        --output_splits /path/to/splits_exclude_broken.json \
        --exclude "20251120_065442Z"
"""

import argparse
import json
import pathlib


def main():
    ap = argparse.ArgumentParser(description="Create splits with excluded videos")
    ap.add_argument("--input_splits", required=True, help="Path to original splits.json")
    ap.add_argument("--output_splits", required=True, help="Path to output splits.json")
    ap.add_argument("--exclude", required=True, 
                    help="Comma-separated list of video stems to exclude")
    args = ap.parse_args()
    
    # Load original splits
    with open(args.input_splits) as f:
        splits = json.load(f)
    
    # Parse exclusion list
    exclude_stems = [s.strip() for s in args.exclude.split(",")]
    
    # Normalize to stems (remove .mp4 extension if present)
    exclude_stems = [s.replace(".mp4", "") for s in exclude_stems]
    
    print(f"Excluding videos: {exclude_stems}")
    
    # Filter each split
    excluded_count = {"train": 0, "val": 0, "test": 0}
    
    for split_name in ["train", "val", "test"]:
        if split_name not in splits:
            continue
        
        original_videos = splits[split_name]
        filtered_videos = []
        
        for video in original_videos:
            stem = video.replace(".mp4", "")
            if stem in exclude_stems:
                print(f"  Removed from {split_name}: {video}")
                excluded_count[split_name] += 1
            else:
                filtered_videos.append(video)
        
        splits[split_name] = filtered_videos
    
    # Save new splits
    output_path = pathlib.Path(args.output_splits)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w") as f:
        json.dump(splits, f, indent=2)
    
    print(f"\n✓ Saved modified splits to: {output_path}")
    print(f"\nSummary:")
    print(f"  Train: {len(splits.get('train', []))} videos ({excluded_count['train']} excluded)")
    print(f"  Val:   {len(splits.get('val', []))} videos ({excluded_count['val']} excluded)")
    print(f"  Test:  {len(splits.get('test', []))} videos ({excluded_count['test']} excluded)")


if __name__ == "__main__":
    main()
