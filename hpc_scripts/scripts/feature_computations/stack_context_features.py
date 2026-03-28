#!/usr/bin/env python3
"""Stack k temporal context frames for each timestep."""
import argparse, pathlib, json, numpy as np

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="Source feature directory")
    ap.add_argument("--out", required=True, help="Output directory")
    ap.add_argument("--k", type=int, default=10, help="Context window (k frames before current)")
    args = ap.parse_args()

    src = pathlib.Path(args.src)
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    for f in sorted(src.glob("*_x.npy")):
        stem = f.stem.replace("_x", "")
        X = np.load(f)  # [T, D]
        T, D = X.shape
        
        # Stack k context frames (causal: only past frames)
        X_ctx = []
        for t in range(T):
            # Get indices for context window [t-k+1, ..., t]
            ctx_indices = [max(0, t - i) for i in range(args.k - 1, -1, -1)]
            ctx = X[ctx_indices]  # [k, D]
            X_ctx.append(ctx.flatten())  # [k*D]
        
        X_stacked = np.stack(X_ctx, axis=0)  # [T, k*D]
        
        np.save(out / f"{stem}_x.npy", X_stacked.astype(np.float32))
        print(f"{stem}: {X.shape} -> {X_stacked.shape}")

if __name__ == "__main__":
    main()