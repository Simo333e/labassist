#!/usr/bin/env python3
"""
Remap existing 5-class labels to the new v2 taxonomy.

Old taxonomy:
  0: none
  1: Pipette pcr tube      → 1: Manipulate PCR tube
  2: Aspirate reagent      → 2: Manipulate reagent
  3: Attach pipette tip    → 3: Attach pipette tip (unchanged)
  4: Eject pipette tip     → 4: Eject pipette tip (unchanged)

New taxonomy (v2):
  0: none
  1: Manipulate PCR tube
  2: Manipulate reagent
  3: Attach pipette tip
  4: Eject pipette tip

The indices stay the same, only the class names change.
This script copies labels and creates the new class_index file.

Usage:
    python remap_labels_to_v2.py \
        --labels_in /path/to/old/labels_5class \
        --labels_out /path/to/new/labels_5class_v2 \
        --class_index_out /path/to/class_index_5class_v2.json \
        --stems "20251027_181439Z,20251112_110656Z_1080p30_v2,..."
"""

import argparse
import json
import shutil
from pathlib import Path
from typing import List, Optional

import numpy as np


# Old class index (for reference)
OLD_CLASS_INDEX = {
    "none": 0,
    "Pipette pcr tube": 1,
    "Aspirate reagent": 2,
    "Attach pipette tip": 3,
    "Eject pipette tip": 4,
}

# New class index (v2)
NEW_CLASS_INDEX = {
    "none": 0,
    "Manipulate PCR tube": 1,
    "Manipulate reagent": 2,
    "Attach pipette tip": 3,
    "Eject pipette tip": 4,
}

# The indices are the same, so we just need to:
# 1. Copy the label files (no modification needed)
# 2. Create the new class_index file with updated names


def main():
    ap = argparse.ArgumentParser(description="Remap labels to v2 taxonomy")
    ap.add_argument("--labels_in", required=True,
                    help="Input labels directory (old 5-class)")
    ap.add_argument("--labels_out", required=True,
                    help="Output labels directory (v2)")
    ap.add_argument("--class_index_out", required=True,
                    help="Output class index file path")
    ap.add_argument("--stems", default=None,
                    help="Comma-separated list of video stems to process (default: all)")
    args = ap.parse_args()

    labels_in = Path(args.labels_in)
    labels_out = Path(args.labels_out)
    labels_out.mkdir(parents=True, exist_ok=True)

    # Save new class index
    class_index_path = Path(args.class_index_out)
    class_index_path.parent.mkdir(parents=True, exist_ok=True)
    with open(class_index_path, "w") as f:
        json.dump(NEW_CLASS_INDEX, f, indent=2)
    print(f"✓ Saved new class index to: {class_index_path}")

    # Get stems to process
    if args.stems:
        stems = [s.strip() for s in args.stems.split(",")]
    else:
        # Process all label files in input directory
        label_files = sorted(labels_in.glob("*_y.npy"))
        stems = [f.stem.replace("_y", "") for f in label_files]

    print(f"\nProcessing {len(stems)} label files...")
    
    processed = []
    skipped = []

    for stem in stems:
        src_path = labels_in / f"{stem}_y.npy"
        dst_path = labels_out / f"{stem}_y.npy"
        
        if not src_path.exists():
            skipped.append(stem)
            continue
        
        # Load labels
        labels = np.load(src_path)
        
        # The indices are the same, so just copy
        # But let's verify the labels are in valid range
        unique_vals = np.unique(labels[labels >= 0])
        if len(unique_vals) > 0 and unique_vals.max() > 4:
            print(f"  ⚠ {stem}: found label values > 4, clamping")
            labels = np.clip(labels, -100, 4)  # -100 is ignore index
        
        # Save to output
        np.save(dst_path, labels)
        
        # Stats
        unique, counts = np.unique(labels[labels >= 0], return_counts=True)
        stats = dict(zip(unique.astype(int), counts.astype(int)))
        action_frames = sum(c for idx, c in stats.items() if idx > 0)
        action_pct = 100 * action_frames / len(labels) if len(labels) > 0 else 0
        
        processed.append(stem)
        print(f"  ✓ {stem}: {len(labels)} frames, {action_frames} action ({action_pct:.1f}%)")

    print(f"\n{'='*60}")
    print(f"SUMMARY")
    print(f"{'='*60}")
    print(f"Processed: {len(processed)} files")
    print(f"Skipped: {len(skipped)} files")
    
    if skipped:
        print(f"\nSkipped (not found):")
        for s in skipped:
            print(f"  - {s}")
    
    print(f"\n✓ Labels saved to: {labels_out}")
    
    # Print class name mapping
    print(f"\n{'='*60}")
    print(f"CLASS NAME MAPPING")
    print(f"{'='*60}")
    for old_name, idx in OLD_CLASS_INDEX.items():
        new_name = [k for k, v in NEW_CLASS_INDEX.items() if v == idx][0]
        if old_name != new_name:
            print(f"  [{idx}] {old_name:25s} → {new_name}")
        else:
            print(f"  [{idx}] {old_name:25s}   (unchanged)")


if __name__ == "__main__":
    main()






