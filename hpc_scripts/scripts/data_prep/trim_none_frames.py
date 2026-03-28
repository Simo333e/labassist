#!/usr/bin/env python3
"""
Trim leading/trailing 'none' frames from features and labels.
Keeps a small buffer of 'none' frames around actions.
"""
import argparse
import pathlib
import json
import numpy as np
from tqdm import tqdm


def find_action_bounds(labels, none_idx=0, buffer_frames=50):
    """Find first and last non-none frame, with buffer."""
    action_mask = labels != none_idx
    if not action_mask.any():
        return 0, len(labels)  # No actions, keep everything
    
    action_indices = np.where(action_mask)[0]
    first_action = max(0, action_indices[0] - buffer_frames)
    last_action = min(len(labels), action_indices[-1] + buffer_frames + 1)
    return first_action, last_action


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--xdir_in", required=True, help="Input features directory")
    ap.add_argument("--ydir_in", required=True, help="Input labels directory")
    ap.add_argument("--xdir_out", required=True, help="Output features directory")
    ap.add_argument("--ydir_out", required=True, help="Output labels directory")
    ap.add_argument("--splits", required=True, help="Path to splits.json")
    ap.add_argument("--none_idx", type=int, default=0, help="Index of 'none' class")
    ap.add_argument("--buffer", type=int, default=50, help="Frames to keep around actions (5 sec at 10Hz)")
    args = ap.parse_args()

    xdir_in = pathlib.Path(args.xdir_in)
    ydir_in = pathlib.Path(args.ydir_in)
    xdir_out = pathlib.Path(args.xdir_out)
    ydir_out = pathlib.Path(args.ydir_out)
    
    xdir_out.mkdir(parents=True, exist_ok=True)
    ydir_out.mkdir(parents=True, exist_ok=True)

    # Get all stems from splits
    sp = json.loads(pathlib.Path(args.splits).read_text())
    all_stems = []
    for split in ["train", "val", "test"]:
        all_stems.extend([pathlib.Path(p).stem for p in sp.get(split, [])])

    total_before = 0
    total_after = 0
    none_before = 0
    none_after = 0

    print(f"Trimming {len(all_stems)} videos...")
    print(f"Buffer: {args.buffer} frames ({args.buffer/10:.1f} seconds at 10Hz)")

    for stem in tqdm(all_stems):
        x_path = xdir_in / f"{stem}_x.npy"
        y_path = ydir_in / f"{stem}_y.npy"
        
        if not x_path.exists() or not y_path.exists():
            print(f"  ⚠ Missing: {stem}")
            continue
        
        x = np.load(x_path)
        y = np.load(y_path)
        
        # Ensure same length
        min_len = min(len(x), len(y))
        x, y = x[:min_len], y[:min_len]
        
        total_before += len(y)
        none_before += (y == args.none_idx).sum()
        
        # Find bounds
        start, end = find_action_bounds(y, args.none_idx, args.buffer)
        
        # Trim
        x_trimmed = x[start:end]
        y_trimmed = y[start:end]
        
        total_after += len(y_trimmed)
        none_after += (y_trimmed == args.none_idx).sum()
        
        # Save
        np.save(xdir_out / f"{stem}_x.npy", x_trimmed.astype(np.float32))
        np.save(ydir_out / f"{stem}_y.npy", y_trimmed.astype(np.int16))
        
        # Copy metadata if exists
        meta_in = xdir_in / f"{stem}_meta.json"
        if meta_in.exists():
            meta = json.loads(meta_in.read_text())
            meta["trimmed"] = True
            meta["original_T"] = int(min_len)
            meta["trim_start"] = int(start)
            meta["trim_end"] = int(end)
            (xdir_out / f"{stem}_meta.json").write_text(json.dumps(meta, indent=2))

    # Summary
    print(f"\n{'='*50}")
    print(f"Total frames: {total_before:,} → {total_after:,} ({100*total_after/total_before:.1f}%)")
    print(f"None frames:  {none_before:,} → {none_after:,} ({100*none_after/total_after:.1f}% of trimmed)")
    print(f"Reduction:    {total_before - total_after:,} frames removed")
    print(f"\n✓ Trimmed data saved to:")
    print(f"   Features: {xdir_out}")
    print(f"   Labels:   {ydir_out}")


if __name__ == "__main__":
    main()




