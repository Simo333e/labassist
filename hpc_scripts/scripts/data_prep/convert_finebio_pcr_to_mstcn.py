#!/usr/bin/env python3
"""
Convert the FineBio PCR subset (framewise label TXT at source FPS) into
MS-TCN-ready label NPZ files at a target FPS, using the existing 6-class
labassist taxonomy.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List

import cv2
import numpy as np


def load_ids(path: Path) -> List[str]:
    return [line.strip() for line in path.read_text().splitlines() if line.strip()]


def video_info(path: Path):
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return fps, n_frames


def resample_labels(y_full: np.ndarray, src_fps: float, target_fps: float) -> np.ndarray:
    """Match the resampling logic used by feature extraction (ceil duration, rounded indices)."""
    n_frames = len(y_full)
    t_out = int(np.ceil((n_frames / src_fps) * target_fps))
    indices = [
        min(n_frames - 1, int(round((t / target_fps) * src_fps)))
        for t in range(t_out)
    ]
    return y_full[indices]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--finebio_root", required=True, help="FineBio root (with pcr_T5_subset)")
    ap.add_argument("--target_root", required=True, help="Combined dataset root (cache/labels will be written here)")
    ap.add_argument("--class_index", required=True, help="Path to class_index.json (shared 6-class mapping)")
    ap.add_argument("--target_fps", type=float, default=10.0, help="Target FPS (must match feature extraction)")
    args = ap.parse_args()

    finebio = Path(args.finebio_root)
    target = Path(args.target_root)
    labels_out = target / "cache" / "labels"
    labels_out.mkdir(parents=True, exist_ok=True)

    class_index = json.loads(Path(args.class_index).read_text())
    valid_ids = set(class_index.values())
    max_label = max(valid_ids)

    ids = load_ids(finebio / "pcr_ids.txt")
    skipped = []
    written = 0

    for pid in ids:
        video_path = finebio / "pcr_T5_subset" / "videos" / f"{pid}_T5.mp4"
        lbl_txt = finebio / "pcr_T5_subset" / "labels" / f"{pid}.txt"
        if not video_path.exists():
            skipped.append(f"{pid} (missing video)")
            continue
        if not lbl_txt.exists():
            skipped.append(f"{pid} (missing labels)")
            continue

        fps, n_frames = video_info(video_path)
        y_raw = np.fromstring(lbl_txt.read_text().strip(), sep=" ", dtype=np.int64)
        if len(y_raw) != n_frames:
            if len(y_raw) > n_frames:
                y_raw = y_raw[:n_frames]
            else:
                pad = np.full(n_frames - len(y_raw), y_raw[-1] if len(y_raw) else 0, dtype=np.int64)
                y_raw = np.concatenate([y_raw, pad])

        # sanity filter: clip any stray labels outside known ids
        y_raw = np.clip(y_raw, 0, max_label)

        y_resampled = resample_labels(y_raw, src_fps=fps, target_fps=args.target_fps)
        np.save(labels_out / f"{pid}_T5_y.npy", y_resampled.astype(np.int16))
        written += 1
        print(f"✓ {pid}: {len(y_raw)} frames @ {fps:.2f} fps -> {len(y_resampled)} labels @ {args.target_fps} Hz")

    if skipped:
        print("\nSkipped entries:")
        for s in skipped:
            print(f" - {s}")
    print(f"\nDone. Wrote {written} label files to {labels_out}")


if __name__ == "__main__":
    main()


