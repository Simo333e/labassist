#!/usr/bin/env python3
"""
Remap 6-class labels to 5-class by merging Aspirate/Dispense PCR tube.

Old mapping:
  0: none                   -> 0: none
  1: Apirate pcr tube       -> 1: Pipette pcr tube
  2: Aspirate reagent       -> 2: Aspirate reagent
  3: Attach pipette tip     -> 3: Attach pipette tip
  4: Dispence to pcr tube   -> 1: Pipette pcr tube  (MERGED)
  5: Eject pipette tip      -> 4: Eject pipette tip
"""
import argparse
import json
import pathlib
import numpy as np


# Remapping: old_idx -> new_idx
REMAP = {
    0: 0,  # none
    1: 1,  # Apirate pcr tube -> Pipette pcr tube
    2: 2,  # Aspirate reagent
    3: 3,  # Attach pipette tip
    4: 1,  # Dispence to pcr tube -> Pipette pcr tube (MERGE)
    5: 4,  # Eject pipette tip
}

IGNORE_INDEX = -100


def remap_labels(y: np.ndarray) -> np.ndarray:
    """Remap label array using REMAP dict, preserving ignore index."""
    y_new = np.full_like(y, IGNORE_INDEX)
    for old_idx, new_idx in REMAP.items():
        y_new[y == old_idx] = new_idx
    # Preserve any ignore indices
    y_new[y == IGNORE_INDEX] = IGNORE_INDEX
    return y_new


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels_in", required=True, help="Input labels directory (6-class)")
    ap.add_argument("--labels_out", required=True, help="Output labels directory (5-class)")
    ap.add_argument("--splits", required=True, help="Path to splits.json")
    args = ap.parse_args()

    labels_in = pathlib.Path(args.labels_in)
    labels_out = pathlib.Path(args.labels_out)
    labels_out.mkdir(parents=True, exist_ok=True)

    # Load splits to get all stems
    sp = json.loads(pathlib.Path(args.splits).read_text())
    all_stems = []
    for split in ["train", "val", "test"]:
        all_stems.extend([pathlib.Path(p).stem for p in sp.get(split, [])])

    print(f"Remapping {len(all_stems)} label files from 6-class to 5-class...")
    print(f"  Input:  {labels_in}")
    print(f"  Output: {labels_out}")

    # Statistics
    old_counts = np.zeros(6, dtype=np.int64)
    new_counts = np.zeros(5, dtype=np.int64)

    for stem in all_stems:
        in_path = labels_in / f"{stem}_y.npy"
        out_path = labels_out / f"{stem}_y.npy"

        if not in_path.exists():
            print(f"  ⚠ Missing: {in_path}")
            continue

        y_old = np.load(in_path)
        y_new = remap_labels(y_old)

        # Count old labels (only non-negative)
        for idx in range(6):
            old_counts[idx] += (y_old == idx).sum()
        # Count new labels (only non-negative)
        for idx in range(5):
            new_counts[idx] += (y_new == idx).sum()

        np.save(out_path, y_new.astype(np.int16))
        print(f"  ✓ {stem}: {len(y_old)} frames")

    print("\n" + "=" * 60)
    print("Old class distribution (6-class):")
    old_names = ["none", "Apirate pcr tube", "Aspirate reagent", 
                 "Attach pipette tip", "Dispence to pcr tube", "Eject pipette tip"]
    total = old_counts.sum()
    for idx, name in enumerate(old_names):
        pct = old_counts[idx] / total * 100 if total > 0 else 0
        print(f"  {idx}: {name:25s} {old_counts[idx]:8d} ({pct:.2f}%)")

    print("\nNew class distribution (5-class):")
    new_names = ["none", "Pipette pcr tube", "Aspirate reagent",
                 "Attach pipette tip", "Eject pipette tip"]
    total = new_counts.sum()
    for idx, name in enumerate(new_names):
        pct = new_counts[idx] / total * 100 if total > 0 else 0
        print(f"  {idx}: {name:25s} {new_counts[idx]:8d} ({pct:.2f}%)")

    print(f"\n✓ Done. Remapped labels saved to {labels_out}")


if __name__ == "__main__":
    main()

