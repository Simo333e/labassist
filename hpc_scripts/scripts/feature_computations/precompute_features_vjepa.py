#!/usr/bin/env python3
"""
Precompute V-JEPA-2 clip features and cache them as *_x.npy (shape [T, D])
aligned to existing 10 Hz labels for MS-TCN training/finetuning.

This mirrors the existing ResNet/I3D precompute scripts but uses the
Hugging Face V-JEPA-2 encoder (no need to clone Meta's repo).

Supports:
  - Video files (via decord or cv2)
  - Pre-extracted frames directory (--frames_dir)
"""
import argparse
import json
import math
import pathlib
import re
from typing import List, Sequence

import numpy as np
import torch
from PIL import Image

try:
    import decord  # type: ignore
    from decord import VideoReader, cpu
except ImportError:
    decord = None
    VideoReader = None
    cpu = None

try:
    import cv2
except ImportError:
    cv2 = None

from transformers import AutoModel, AutoVideoProcessor

# Pattern to identify FineBio stems (P##_##_##_T# format)
FINEBIO_PATTERN = re.compile(r'^P\d{2}_\d{2}_\d{2}_T\d')


def is_mydata_stem(stem: str) -> bool:
    """
    Check if stem belongs to mydata (needs cropping) vs FineBio (no crop).
    
    mydata stems: timestamp format (YYYYMMDD_*) or Lab_video*
    FineBio stems: P##_##_##_T# format (e.g., P01_05_01_T5)
    
    Returns True for mydata, False for FineBio.
    """
    if FINEBIO_PATTERN.match(stem):
        return False  # FineBio - don't crop
    return True  # mydata - apply crop


def load_splits(splits_path: pathlib.Path) -> Sequence[str]:
    data = json.loads(splits_path.read_text())
    if "train" not in data:
        # older format: {"something": {"train": [...], "val": [...], ...}}
        data = next(iter(data.values()))
    stems = []
    for split in ("train", "val", "test"):
        stems.extend([pathlib.Path(p).stem for p in data.get(split, [])])
    return stems


def compute_centers(num_frames: int, fps_src: float, fps_target: float) -> np.ndarray:
    duration = num_frames / max(1e-6, fps_src)
    num_steps = max(1, int(math.floor(duration * fps_target)))
    times = np.arange(num_steps, dtype=np.float32) / fps_target
    centers = (times * fps_src).round().astype(int)
    centers = np.clip(centers, 0, num_frames - 1)
    return centers


def clip_indices(center: int, num_frames: int, frames_per_clip: int, stride: int) -> np.ndarray:
    offsets = (np.arange(frames_per_clip) - frames_per_clip // 2) * stride
    idxs = center + offsets
    idxs = np.clip(idxs, 0, num_frames - 1).astype(int)
    return idxs


def read_batch_decord(vr: VideoReader, indices: Sequence[int]) -> List[np.ndarray]:
    batch = vr.get_batch(indices)  # (T, H, W, C) uint8
    return [frame.asnumpy() for frame in batch]


def read_batch_cv2(cap, indices: Sequence[int]) -> List[np.ndarray]:
    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, f = cap.read()
        if not ok:
            raise RuntimeError(f"Failed to read frame {idx}")
        frames.append(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
    return frames


def apply_crop(img: np.ndarray, crop_top: float, crop_bottom: float,
               crop_left: float, crop_right: float) -> np.ndarray:
    """Crop image by percentage from each edge."""
    if crop_top == 0 and crop_bottom == 0 and crop_left == 0 and crop_right == 0:
        return img
    h, w = img.shape[:2]
    top = int(h * crop_top)
    bottom = int(h * (1 - crop_bottom))
    left = int(w * crop_left)
    right = int(w * (1 - crop_right))
    return img[top:bottom, left:right]


def read_batch_frames_dir(frames_dir: pathlib.Path, indices: Sequence[int],
                          crop_top: float = 0, crop_bottom: float = 0,
                          crop_left: float = 0, crop_right: float = 0,
                          vertical_flip: bool = False,
                          horizontal_flip: bool = False) -> List[np.ndarray]:
    """Read frames from pre-extracted directory. Expects 6-digit zero-padded filenames (1-indexed)."""
    frames = []
    for idx in indices:
        # Frame files are 1-indexed: 000001.jpg, 000002.jpg, ...
        frame_path = frames_dir / f"{idx + 1:06d}.jpg"
        if not frame_path.exists():
            # Try png as fallback
            frame_path = frames_dir / f"{idx + 1:06d}.png"
        if not frame_path.exists():
            raise RuntimeError(f"Frame not found: {frame_path}")
        img = Image.open(frame_path).convert("RGB")
        img_arr = np.array(img)
        # Apply cropping
        img_arr = apply_crop(img_arr, crop_top, crop_bottom, crop_left, crop_right)
        # Apply vertical flip (to align mydata viewpoint with FineBio)
        if vertical_flip:
            img_arr = np.flipud(img_arr).copy()  # Flip vertically, copy to ensure contiguous
        # Apply horizontal flip (left/right invariance for handedness)
        if horizontal_flip:
            img_arr = np.fliplr(img_arr).copy()  # Flip horizontally, copy to ensure contiguous
        frames.append(img_arr)
    return frames


def count_frames_in_dir(frames_dir: pathlib.Path) -> int:
    """Count number of frame files in a directory."""
    jpg_count = len(list(frames_dir.glob("*.jpg")))
    png_count = len(list(frames_dir.glob("*.png")))
    return max(jpg_count, png_count)


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="Dataset root containing videos/ and cache/")
    ap.add_argument("--splits", required=True, help="Path to splits.json")
    ap.add_argument("--fps", type=float, default=10.0, help="Target FPS for features (match labels)")
    ap.add_argument("--frames_per_clip", type=int, default=64, help="Frames per clip for V-JEPA")
    ap.add_argument("--stride", type=int, default=1, help="Stride between frames within a clip")
    ap.add_argument("--batch_size", type=int, default=2, help="Clips per forward pass")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", required=True, help="Output directory for *_x.npy cache")
    ap.add_argument("--model_id", default="facebook/vjepa2-vitl-fpc64-256", help="HF model id")
    ap.add_argument("--shards", type=int, default=1, help="Split stems across shards")
    ap.add_argument("--shard_idx", type=int, default=0, help="0-based shard index")
    ap.add_argument("--use_decord", action="store_true", help="Prefer decord for video IO if available")
    ap.add_argument("--frames_dir", type=str, default="", help="Path to pre-extracted frames dir (e.g., root/frames). If set, reads from {frames_dir}/{stem}/NNNNNN.jpg")
    ap.add_argument("--frames_fps", type=float, default=25.0, help="FPS of pre-extracted frames (only used with --frames_dir)")
    
    # Cropping parameters (for focusing on work area, e.g., mydata)
    ap.add_argument("--crop_top", type=float, default=0.0,
                    help="Fraction to crop from top (0.0-1.0)")
    ap.add_argument("--crop_bottom", type=float, default=0.0,
                    help="Fraction to crop from bottom (0.0-1.0, e.g., 0.35 to remove glass shelf)")
    ap.add_argument("--crop_left", type=float, default=0.0,
                    help="Fraction to crop from left (0.0-1.0)")
    ap.add_argument("--crop_right", type=float, default=0.0,
                    help="Fraction to crop from right (0.0-1.0)")
    ap.add_argument("--selective_crop", action="store_true",
                    help="Only apply cropping to mydata stems, not FineBio (P##_##_##_T#)")
    ap.add_argument("--vertical_flip", action="store_true",
                    help="Apply vertical flip to mydata stems (aligns top-down mydata with bottom-up FineBio)")
    ap.add_argument("--horizontal_flip", action="store_true",
                    help="Apply horizontal flip to mydata stems (left/right invariance for handedness)")
    args = ap.parse_args()
    
    # Validate crop parameters
    has_crop_params = args.crop_top > 0 or args.crop_bottom > 0 or args.crop_left > 0 or args.crop_right > 0
    if has_crop_params:
        crop_msg = f"Cropping: top={args.crop_top:.0%}, bottom={args.crop_bottom:.0%}, " \
                   f"left={args.crop_left:.0%}, right={args.crop_right:.0%}"
        if args.selective_crop:
            print(f"Selective {crop_msg} (mydata only, FineBio unchanged)")
        else:
            print(f"Using {crop_msg} (all videos)")
    
    # Vertical flip (to align mydata viewpoint with FineBio)
    if args.vertical_flip:
        print("Vertical flip: enabled for mydata stems (aligns top-down → bottom-up)")
    
    # Horizontal flip (left/right invariance)
    if args.horizontal_flip:
        print("Horizontal flip: enabled for mydata stems (left/right invariance)")

    root = pathlib.Path(args.root)
    videos_dir = root / "videos"
    if not videos_dir.exists():
        videos_dir = root

    # Pre-extracted frames directory (optional)
    frames_dir_base = pathlib.Path(args.frames_dir) if args.frames_dir else None
    use_frames_dir = frames_dir_base is not None and frames_dir_base.exists()
    if use_frames_dir:
        print(f"Using pre-extracted frames from: {frames_dir_base} @ {args.frames_fps} fps")

    out_dir = pathlib.Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    stems_all = load_splits(pathlib.Path(args.splits))
    S = max(1, args.shards)
    I = args.shard_idx
    if I < 0 or I >= S:
        raise SystemExit(f"shard_idx {I} out of range for shards={S}")
    stems = [s for i, s in enumerate(stems_all) if (i % S) == I]
    print(f"Processing {len(stems)} stems on shard {I}/{S}")

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        try:
            print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        except Exception:
            pass

    model = AutoModel.from_pretrained(
        args.model_id,
        dtype="auto",
        device_map=None if device.type == "cpu" else "auto",
    ).eval()
    processor = AutoVideoProcessor.from_pretrained(args.model_id)

    frames_per_clip = args.frames_per_clip
    stride = max(1, args.stride)

    for stem in stems:
        out_npy = out_dir / f"{stem}_x.npy"
        out_meta = out_dir / f"{stem}_meta.json"
        if out_npy.exists():
            print(f"⏭ Skipping {stem} (exists)")
            continue

        cap = None
        vr = None

        # Determine if cropping should be applied to this stem
        if args.selective_crop:
            # Only crop mydata stems, not FineBio
            apply_crop_to_stem = is_mydata_stem(stem) and has_crop_params
        else:
            # Apply cropping to all stems if crop params are set
            apply_crop_to_stem = has_crop_params

        if apply_crop_to_stem:
            crop_top, crop_bottom = args.crop_top, args.crop_bottom
            crop_left, crop_right = args.crop_left, args.crop_right
        else:
            crop_top, crop_bottom, crop_left, crop_right = 0.0, 0.0, 0.0, 0.0

        # Determine if flips should be applied (only for mydata stems)
        apply_vflip_to_stem = args.vertical_flip and is_mydata_stem(stem)
        apply_hflip_to_stem = args.horizontal_flip and is_mydata_stem(stem)

        # Determine source: pre-extracted frames or video
        if use_frames_dir:
            stem_frames_dir = frames_dir_base / stem
            if not stem_frames_dir.exists():
                print(f"⚠ Frames dir not found for stem: {stem} (expected {stem_frames_dir})")
                continue
            nframes = count_frames_in_dir(stem_frames_dir)
            fps_src = args.frames_fps
            reader = lambda idxs, fdir=stem_frames_dir, ct=crop_top, cb=crop_bottom, cl=crop_left, cr=crop_right, vf=apply_vflip_to_stem, hf=apply_hflip_to_stem: read_batch_frames_dir(fdir, idxs, ct, cb, cl, cr, vf, hf)
            source_type = "frames_dir"
        else:
            # Find video file
            vpath = None
            for ext in ("mp4", "MP4", "avi", "AVI", "mov", "MOV"):
                cand = videos_dir / f"{stem}.{ext}"
                if cand.exists():
                    vpath = cand
                    break
            if vpath is None:
                matches = list(videos_dir.glob(f"{stem}.*"))
                vpath = matches[0] if matches else None
            if vpath is None:
                print(f"⚠ Video not found for stem: {stem}")
                continue

            use_decord = args.use_decord and decord is not None

            if use_decord:
                vr = VideoReader(str(vpath), ctx=cpu(0))
                nframes = len(vr)
                fps_src = float(vr.get_avg_fps())
                reader = lambda idxs, v=vr: read_batch_decord(v, idxs)
            else:
                if cv2 is None:
                    raise SystemExit("cv2 not available and decord disabled/unavailable")
                cap = cv2.VideoCapture(str(vpath))
                fps_src = cap.get(cv2.CAP_PROP_FPS) or 30.0
                nframes = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
                reader = lambda idxs, c=cap: read_batch_cv2(c, idxs)
            source_type = "video"

        if nframes <= 0:
            if cap:
                cap.release()
            print(f"⚠ Empty source: {stem}")
            continue

        centers = compute_centers(nframes, fps_src, args.fps)
        B = max(1, args.batch_size)
        print(f"{stem}: {nframes} frames @ {fps_src:.2f} fps -> {len(centers)} centers @ {args.fps} Hz; "
              f"clip={frames_per_clip}, stride={stride}, batch={B}, source={source_type}")
        all_feats = []

        for start in range(0, len(centers), B):
            batch_centers = centers[start:start + B]
            frames_list = []
            for c in batch_centers:
                idxs = clip_indices(c, nframes, frames_per_clip, stride)
                frames_list.append(reader(idxs))

            # processor expects list of frame sequences; flatten via dict keys
            inputs = processor(frames_list, return_tensors="pt")
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
            feats_tokens = model.get_vision_features(**inputs)  # [B, N_tokens, D]
            feats = feats_tokens.mean(dim=1).cpu().numpy()      # [B, D]
            all_feats.append(feats)

            if ((start // B) + 1) % 50 == 0:
                print(f"{stem}: processed {start + len(batch_centers)}/{len(centers)} clips")

        if cap:
            cap.release()

        if not all_feats:
            print(f"⚠ No features extracted for {stem}")
            continue

        X = np.concatenate(all_feats, axis=0)  # [T, D]
        np.save(out_npy, X.astype(np.float32))
        meta = {
            "stem": stem,
            "fps_src": fps_src,
            "fps_features": args.fps,
            "frames_per_clip": frames_per_clip,
            "stride": stride,
            "T": int(X.shape[0]),
            "D": int(X.shape[1]),
            "model_id": args.model_id,
            "source_type": source_type,
        }
        if apply_crop_to_stem:
            meta["crop"] = {
                "top": crop_top,
                "bottom": crop_bottom,
                "left": crop_left,
                "right": crop_right,
            }
        if apply_vflip_to_stem:
            meta["vertical_flip"] = True
        if apply_hflip_to_stem:
            meta["horizontal_flip"] = True
        out_meta.write_text(json.dumps(meta, indent=2))
        indicators = []
        if apply_crop_to_stem:
            indicators.append("cropped")
        if apply_vflip_to_stem:
            indicators.append("vflip")
        if apply_hflip_to_stem:
            indicators.append("hflip")
        indicator_str = f" [{'+'.join(indicators)}]" if indicators else ""
        print(f"✓ {stem}: X{tuple(X.shape)} saved{indicator_str} → {out_npy}")

    print("✓ V-JEPA feature extraction complete.")


if __name__ == "__main__":
    main()


