#!/usr/bin/env python3
"""
L2-normalize I3D features to prevent NaN in training.
Creates a new directory with normalized features.
"""
import argparse
import pathlib
import numpy as np
from tqdm import tqdm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_dir", required=True, help="Input features directory")
    ap.add_argument("--output_dir", required=True, help="Output normalized features directory")
    args = ap.parse_args()

    in_dir = pathlib.Path(args.input_dir)
    out_dir = pathlib.Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    feature_files = list(in_dir.glob("*_x.npy"))
    print(f"Normalizing {len(feature_files)} feature files...")

    stats_before = []
    stats_after = []

    for fpath in tqdm(feature_files):
        x = np.load(fpath)
        
        # Record stats before
        stats_before.append((x.mean(), x.std(), x.min(), x.max()))
        
        # L2 normalize each frame (row)
        norms = np.linalg.norm(x, axis=1, keepdims=True)
        x_norm = x / (norms + 1e-8)
        
        # Record stats after
        stats_after.append((x_norm.mean(), x_norm.std(), x_norm.min(), x_norm.max()))
        
        # Save normalized features
        out_path = out_dir / fpath.name
        np.save(out_path, x_norm.astype(np.float32))
        
        # Copy metadata if exists
        meta_path = in_dir / fpath.name.replace("_x.npy", "_meta.json")
        if meta_path.exists():
            (out_dir / meta_path.name).write_text(meta_path.read_text())

    # Print summary
    before = np.array(stats_before)
    after = np.array(stats_after)
    print(f"\nBefore normalization:")
    print(f"  Mean: {before[:,0].mean():.4f}, Std: {before[:,1].mean():.4f}")
    print(f"  Min: {before[:,2].min():.4f}, Max: {before[:,3].max():.4f}")
    print(f"\nAfter normalization:")
    print(f"  Mean: {after[:,0].mean():.4f}, Std: {after[:,1].mean():.4f}")
    print(f"  Min: {after[:,2].min():.4f}, Max: {after[:,3].max():.4f}")
    print(f"\n✓ Normalized features saved to {out_dir}")


if __name__ == "__main__":
    main()




