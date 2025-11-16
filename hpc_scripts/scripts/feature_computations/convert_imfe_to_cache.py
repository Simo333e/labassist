#!/usr/bin/env python3
import argparse
import json
import pathlib
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--imfe_features", required=True)  # <frames>/features from RT-HARE
    ap.add_argument("--out", required=True)            # your cache dir, e.g., .../features_10hz_imfe
    ap.add_argument("--fps", type=float, default=10.0)
    ap.add_argument("--frames_root", default=None, help="Path to rawframes_rgb root to match 10 Hz frame count per video")
    args = ap.parse_args()

    src = pathlib.Path(args.imfe_features)
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    for f in sorted(src.glob("*.npy")):
        stem = f.stem
        X = np.load(f)

        # Optionally resample features to match the number of extracted frames (10 Hz)
        if args.frames_root:
            fr_dir = pathlib.Path(args.frames_root) / stem
            if fr_dir.exists():
                Ty = len(list(fr_dir.glob("img_*.jpg")))
                if Ty > 0 and Ty != len(X):
                    idx = np.round(np.linspace(0, len(X) - 1, Ty)).astype(int)
                    X = X[idx]

        np.save(out / f"{stem}_x.npy", X.astype(np.float32))
        meta = {
            "stem": stem,
            "fps": float(args.fps),
            "src_fps": float(args.fps),
            "T": int(X.shape[0]),
            "D": int(X.shape[1]),
        }
        (out / f"{stem}_meta.json").write_text(json.dumps(meta))
        print(f"{stem}: X{tuple(X.shape)} saved → {(out / (stem + '_x.npy'))}")


if __name__ == "__main__":
    main()


