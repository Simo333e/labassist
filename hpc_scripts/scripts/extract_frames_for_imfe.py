#!/usr/bin/env python3
import argparse
import json
import math
import pathlib
import cv2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)     # FineBio root containing videos/
    ap.add_argument("--splits", required=True)
    ap.add_argument("--view", default="T1")
    ap.add_argument("--fps", type=float, default=10.0)
    ap.add_argument("--out", required=True)      # data_feature_extraction/<dataset>
    args = ap.parse_args()

    ROOT = pathlib.Path(args.root)
    OUT = pathlib.Path(args.out) / "rawframes_rgb"
    OUT.mkdir(parents=True, exist_ok=True)

    sp = json.loads(pathlib.Path(args.splits).read_text())
    sp = sp[next(iter(sp))] if "train" not in sp else sp
    stems = [pathlib.Path(p).stem for p in sp["train"] + sp["val"] + sp["test"]]

    vids = {v.stem: v for v in (ROOT / "videos").glob(f"*_{args.view}.mp4")}
    for stem in stems:
        vpath = vids.get(stem)
        if not vpath:
            continue
        out_dir = OUT / stem
        if (out_dir / "img_00000.jpg").exists():
            continue
        out_dir.mkdir(parents=True, exist_ok=True)

        cap = cv2.VideoCapture(str(vpath))
        src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        nframes = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        if nframes <= 0:
            cap.release()
            continue

        frames = []
        ok = True
        while ok:
            ok, f = cap.read()
            if ok:
                frames.append(f)  # BGR
        cap.release()
        if not frames:
            continue

        T_out = int(math.ceil((nframes / src_fps) * args.fps))
        centers = [min(nframes - 1, int(round((t / args.fps) * src_fps))) for t in range(T_out)]

        for i, idx in enumerate(centers):
            fpath = out_dir / f"img_{i:05d}.jpg"
            fpath.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(fpath), frames[idx])


if __name__ == "__main__":
    main()


