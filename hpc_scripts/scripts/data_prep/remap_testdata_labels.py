#!/usr/bin/env python3
"""
Remap test data labels from 6-class to 5-class (training) format.

Test classes (6):
  0: none
  1: Apirate pcr tube (typo)
  2: Aspirate reagent
  3: Attach pipette tip
  4: Dispence to pcr tube
  5: Eject pipette tip

Training classes (5):
  0: none
  1: Pipette pcr tube
  2: Aspirate reagent
  3: Attach pipette tip
  4: Eject pipette tip

Mapping:
  Test 0 (none) → Train 0 (none)
  Test 1 (Apirate pcr tube) → Train 1 (Pipette pcr tube)
  Test 2 (Aspirate reagent) → Train 2 (Aspirate reagent)
  Test 3 (Attach pipette tip) → Train 3 (Attach pipette tip)
  Test 4 (Dispence to pcr tube) → Train 1 (Pipette pcr tube)
  Test 5 (Eject pipette tip) → Train 4 (Eject pipette tip)
"""

import argparse
import pathlib
import numpy as np

# Mapping from test class indices to training class indices
REMAP = {
    0: 0,  # none → none
    1: 1,  # Apirate pcr tube → Pipette pcr tube
    2: 2,  # Aspirate reagent → Aspirate reagent
    3: 3,  # Attach pipette tip → Attach pipette tip
    4: 1,  # Dispence to pcr tube → Pipette pcr tube
    5: 4,  # Eject pipette tip → Eject pipette tip
}


def main():
    ap = argparse.ArgumentParser(description="Remap test labels to training class indices")
    ap.add_argument("--input_dir", required=True, help="Input labels directory (6-class)")
    ap.add_argument("--output_dir", required=True, help="Output labels directory (5-class)")
    args = ap.parse_args()

    input_dir = pathlib.Path(args.input_dir)
    output_dir = pathlib.Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    label_files = sorted(input_dir.glob("*_y.npy"))
    print(f"Found {len(label_files)} label files in {input_dir}")

    for lf in label_files:
        labels = np.load(lf)
        
        # Apply remapping
        remapped = np.vectorize(REMAP.get)(labels)
        
        # Count changes
        changed = (labels != remapped).sum()
        
        out_path = output_dir / lf.name
        np.save(out_path, remapped.astype(np.int64))
        
        print(f"  ✓ {lf.name}: {len(labels)} frames, {changed} remapped → {out_path.name}")

    print(f"\n✓ Done! Remapped labels saved to: {output_dir}")
    print("\nClass mapping applied:")
    print("  Test 0 (none) → Train 0 (none)")
    print("  Test 1 (Apirate pcr tube) → Train 1 (Pipette pcr tube)")
    print("  Test 2 (Aspirate reagent) → Train 2 (Aspirate reagent)")
    print("  Test 3 (Attach pipette tip) → Train 3 (Attach pipette tip)")
    print("  Test 4 (Dispence to pcr tube) → Train 1 (Pipette pcr tube)")
    print("  Test 5 (Eject pipette tip) → Train 4 (Eject pipette tip)")


if __name__ == "__main__":
    main()

