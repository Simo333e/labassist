#!/usr/bin/env python3
"""
Concatenate I3D and YOLO features for multi-modal action recognition.

This script merges:
- I3D features: [T, 1024] - temporal appearance/motion features
- YOLO features: [T, 28] - object detection features

Output: [T, 1052] combined features that provide both motion patterns
and object identity information, addressing the confusion between
visually similar actions like "Pipette pcr tube" vs "Aspirate reagent".

Feature Normalization:
- I3D features: Already L2-normalized or can be standardized
- YOLO features: 
  - Object confidence [0-1]: Keep as-is
  - Positions [0-1]: Keep as-is (already normalized)
  - Proximity [0-sqrt(2)]: Scale to [0-1] by dividing by sqrt(2)
"""
import argparse
import json
import pathlib
from typing import Optional

import numpy as np


def normalize_yolo_features(X_yolo: np.ndarray) -> np.ndarray:
    """
    Normalize YOLO features to be in similar range as I3D features.
    
    YOLO features breakdown:
    - [0:14]  Object confidence: already [0, 1]
    - [14:24] Positions (cx, cy): already [0, 1] 
    - [24:28] Proximity: [0, sqrt(2)] -> scale to [0, 1]
    """
    X_norm = X_yolo.copy()
    
    # Scale proximity features (last 4 dims) from [0, sqrt(2)] to [0, 1]
    max_dist = np.sqrt(2)  # diagonal of unit square
    X_norm[:, 24:28] = X_norm[:, 24:28] / max_dist
    
    return X_norm


def main():
    ap = argparse.ArgumentParser(description="Concatenate I3D and YOLO features")
    ap.add_argument("--i3d_dir", required=True, help="Directory with I3D features (*_x.npy)")
    ap.add_argument("--yolo_dir", required=True, help="Directory with YOLO features (*_x.npy)")
    ap.add_argument("--output_dir", required=True, help="Output directory for fused features")
    ap.add_argument("--normalize_yolo", action="store_true", default=True,
                    help="Normalize YOLO features to [0,1] range")
    ap.add_argument("--copy_labels", action="store_true",
                    help="Also copy label files from a reference directory")
    ap.add_argument("--labels_dir", default=None,
                    help="Source label directory (for --copy_labels)")
    
    args = ap.parse_args()
    
    i3d_dir = pathlib.Path(args.i3d_dir)
    yolo_dir = pathlib.Path(args.yolo_dir)
    out_dir = pathlib.Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all I3D feature files
    i3d_files = sorted(i3d_dir.glob("*_x.npy"))
    
    if not i3d_files:
        raise SystemExit(f"No I3D feature files found in {i3d_dir}")
    
    print(f"Found {len(i3d_files)} I3D feature files")
    print(f"YOLO directory: {yolo_dir}")
    print(f"Output: {out_dir}")
    print(f"Normalize YOLO: {args.normalize_yolo}")
    
    success = 0
    skipped = 0
    
    for i3d_path in i3d_files:
        stem = i3d_path.stem.replace("_x", "")
        yolo_path = yolo_dir / f"{stem}_x.npy"
        out_path = out_dir / f"{stem}_x.npy"
        
        if out_path.exists():
            print(f"⏭ Skipping {stem} (already exists)")
            skipped += 1
            continue
        
        if not yolo_path.exists():
            print(f"⚠ Missing YOLO features for {stem}, skipping")
            skipped += 1
            continue
        
        # Load features
        X_i3d = np.load(i3d_path)    # [T_i3d, 1024]
        X_yolo = np.load(yolo_path)  # [T_yolo, 28]
        
        # Align temporal lengths (may differ by 1-2 frames due to rounding)
        T_i3d = X_i3d.shape[0]
        T_yolo = X_yolo.shape[0]
        T = min(T_i3d, T_yolo)
        
        if abs(T_i3d - T_yolo) > 5:
            print(f"⚠ Large temporal mismatch for {stem}: I3D={T_i3d}, YOLO={T_yolo}")
        
        X_i3d = X_i3d[:T]
        X_yolo = X_yolo[:T]
        
        # Normalize YOLO features if requested
        if args.normalize_yolo:
            X_yolo = normalize_yolo_features(X_yolo)
        
        # Concatenate features
        X_combined = np.concatenate([X_i3d, X_yolo], axis=1)  # [T, 1052]
        
        # Save
        np.save(out_path, X_combined.astype(np.float32))
        
        # Also save/update metadata
        meta_path = out_dir / f"{stem}_meta.json"
        i3d_meta_path = i3d_dir / f"{stem}_meta.json"
        yolo_meta_path = yolo_dir / f"{stem}_meta.json"
        
        meta = {
            "stem": stem,
            "T": int(T),
            "D": int(X_combined.shape[1]),
            "i3d_dim": int(X_i3d.shape[1]),
            "yolo_dim": int(X_yolo.shape[1]),
            "normalized_yolo": args.normalize_yolo,
        }
        
        # Merge metadata from sources if available
        if i3d_meta_path.exists():
            i3d_meta = json.loads(i3d_meta_path.read_text())
            meta["fps"] = i3d_meta.get("fps", 10.0)
            meta["src_fps"] = i3d_meta.get("src_fps", 25.0)
            if "crop" in i3d_meta:
                meta["crop"] = i3d_meta["crop"]
            if "vertical_flip" in i3d_meta:
                meta["vertical_flip"] = i3d_meta["vertical_flip"]
            if "horizontal_flip" in i3d_meta:
                meta["horizontal_flip"] = i3d_meta["horizontal_flip"]
        
        meta_path.write_text(json.dumps(meta, indent=2))
        
        print(f"✓ {stem}: I3D{tuple(X_i3d.shape)} + YOLO{tuple(X_yolo.shape)} → {tuple(X_combined.shape)}")
        success += 1
    
    print(f"\n✓ Feature fusion complete: {success} videos processed, {skipped} skipped")
    print(f"  Output: {out_dir}")
    print(f"  Feature dimension: 1024 (I3D) + 28 (YOLO) = 1052")


if __name__ == "__main__":
    main()

