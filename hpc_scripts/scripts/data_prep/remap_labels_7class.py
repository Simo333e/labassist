#!/usr/bin/env python3
"""
Remap FineBio 6-class PCR labels to 7-class taxonomy (compatible with mydata 7-class).

FineBio 6-class:
  0: none
  1: Apirate pcr tube
  2: Aspirate reagent
  3: Attach pipette tip
  4: Dispence to pcr tube
  5: Eject pipette tip

Mapped to 7-class:
  0: none
  1: Adjust pipette volume (N/A in FineBio)
  2: Attach pipette tip
  3: Eject pipette tip
  4: Manipulate pcr tubes (combines Apirate + Dispence to pcr tube)
  5: Manipulate reagents (Aspirate reagent only in FineBio)
  6: Switch pipette (N/A in FineBio)
"""
import argparse
import json
import pathlib
import numpy as np


# Remapping: old_finebio_idx -> new_7class_idx
REMAP = {
    0: 0,  # none -> none
    1: 4,  # Apirate pcr tube -> Manipulate pcr tubes
    2: 5,  # Aspirate reagent -> Manipulate reagents
    3: 2,  # Attach pipette tip -> Attach pipette tip
    4: 4,  # Dispence to pcr tube -> Manipulate pcr tubes (MERGE with 1)
    5: 3,  # Eject pipette tip -> Eject pipette tip
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
    ap = argparse.ArgumentParser(description="Remap FineBio 6-class labels to 7-class taxonomy")
    ap.add_argument("--labels_in", required=True, help="Input labels directory (6-class FineBio)")
    ap.add_argument("--labels_out", required=True, help="Output labels directory (7-class)")
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

    print(f"Remapping {len(all_stems)} label files from FineBio 6-class to 7-class...")
    print(f"  Input:  {labels_in}")
    print(f"  Output: {labels_out}")
    print()

    # Statistics
    old_counts = np.zeros(6, dtype=np.int64)
    new_counts = np.zeros(7, dtype=np.int64)

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
        for idx in range(7):
            new_counts[idx] += (y_new == idx).sum()

        np.save(out_path, y_new.astype(np.int16))
        print(f"  ✓ {stem}: {len(y_old)} frames")

    print("\n" + "=" * 70)
    print("Old class distribution (FineBio 6-class):")
    old_names = ["none", "Apirate pcr tube", "Aspirate reagent", 
                 "Attach pipette tip", "Dispence to pcr tube", "Eject pipette tip"]
    total = old_counts.sum()
    for idx, name in enumerate(old_names):
        pct = old_counts[idx] / total * 100 if total > 0 else 0
        print(f"  {idx}: {name:30s} {old_counts[idx]:8d} ({pct:.2f}%)")

    print("\nNew class distribution (7-class):")
    new_names = ["none", "Adjust pipette volume", "Attach pipette tip",
                 "Eject pipette tip", "Manipulate pcr tubes", "Manipulate reagents",
                 "Switch pipette"]
    total = new_counts.sum()
    for idx, name in enumerate(new_names):
        pct = new_counts[idx] / total * 100 if total > 0 else 0
        print(f"  {idx}: {name:30s} {new_counts[idx]:8d} ({pct:.2f}%)")

    print(f"\n✓ Done. Remapped labels saved to {labels_out}")


if __name__ == "__main__":
    main()




