#!/usr/bin/env python3
"""
Inception-I3D feature extraction from pre-extracted frames.

Assumptions:
- ROOT/
    videos/   (original .mp4, not used here)
    frames/
      <stem>/
        000001.jpg
        000002.jpg
        ...

- Frames were extracted with ffmpeg at 25 fps:

  ffmpeg -i videos/<stem>.mp4 -vf "fps=25,scale=-1:256" -q:v 2 frames/<stem>/%06d.jpg

This script:
- Uses Inception-I3D pretrained on Kinetics-400 (from piergiaj/pytorch-i3d).
- For each video stem from splits.json, reads frames/<stem>/*.jpg.
- Builds temporal centers on a target FPS timeline (args.fps).
- Forms clips around each center and extracts 1024-D features via avg_pool.
- Saves one <stem>_x.npy per video to OUT.
"""
import argparse
import json
import math
import pathlib
import re
import sys
from typing import List

import cv2
import numpy as np
import torch
import torchvision.transforms as T

# Add models directory to path for pytorch_i3d
MODELS_DIR = pathlib.Path(__file__).resolve().parents[2] / "models"
if str(MODELS_DIR) not in sys.path:
    sys.path.insert(0, str(MODELS_DIR))

from pytorch_i3d import InceptionI3d  # noqa: E402


EXTRACTION_FPS = 25.0  # must match ffmpeg fps used when dumping frames

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


def load_frame_paths(frames_root: pathlib.Path, stem: str) -> List[pathlib.Path]:
    """Return sorted list of frame paths for a given stem, or [] if missing."""
    d = frames_root / stem
    if not d.exists():
        return []
    return sorted(d.glob("*.jpg"))


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


def load_video_tensor(frame_paths: List[pathlib.Path], tfm, 
                      crop_top: float = 0, crop_bottom: float = 0,
                      crop_left: float = 0, crop_right: float = 0,
                      vertical_flip: bool = False,
                      horizontal_flip: bool = False) -> torch.Tensor:
    """Load all frames into a single tensor [F, 3, H, W]."""
    frames = []
    for p in frame_paths:
        img_bgr = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if img_bgr is None:
            continue
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        # Apply cropping before transforms
        img_rgb = apply_crop(img_rgb, crop_top, crop_bottom, crop_left, crop_right)
        # Apply vertical flip (to align mydata viewpoint with FineBio)
        if vertical_flip:
            img_rgb = np.flipud(img_rgb).copy()  # Flip vertically, copy to ensure contiguous
        # Apply horizontal flip (left/right invariance for handedness)
        if horizontal_flip:
            img_rgb = np.fliplr(img_rgb).copy()  # Flip horizontally, copy to ensure contiguous
        frames.append(tfm(img_rgb))  # [3, H, W]
    if not frames:
        return torch.empty(0, 3, 1, 1)
    return torch.stack(frames, dim=0)  # [F, 3, H, W]


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="Dataset root containing frames/")
    ap.add_argument("--splits", required=True, help="Path to splits.json")
    ap.add_argument("--weights", required=True, help="Path to I3D weights (e.g. rgb_imagenet.pt)")
    ap.add_argument("--view", default="T1", help="Unused (kept for CLI compatibility)")
    ap.add_argument("--fps", type=float, default=10.0, help="Target FPS for feature timeline")
    ap.add_argument("--win", type=int, default=64, help="Frames per clip (I3D default: 64)")
    ap.add_argument("--stride", type=int, default=1, help="Temporal stride inside each clip")
    ap.add_argument("--imgsz", type=int, default=224, help="Spatial size (I3D default: 224)")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", default=None, help="Output dir (default: root/cache/features_10hz_i3d)")

    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--center_step", type=int, default=1, help="Keep every kth center")
    ap.add_argument("--shards", type=int, default=1, help="Split stems across N shards")
    ap.add_argument("--shard_idx", type=int, default=0, help="0-based shard index")
    
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

    ROOT = pathlib.Path(args.root)
    OUT = pathlib.Path(args.out or (ROOT / "cache/features_10hz_i3d"))
    OUT.mkdir(parents=True, exist_ok=True)

    frames_root = ROOT / "frames"

    # Load splits and stems
    sp = json.loads(pathlib.Path(args.splits).read_text())
    sp = sp[next(iter(sp))] if "train" not in sp else sp
    stems_all = [pathlib.Path(p).stem for p in sp["train"] + sp["val"] + sp["test"]]

    # Sharding
    S = max(1, int(args.shards))
    I = int(args.shard_idx)
    if I < 0 or I >= S:
        raise SystemExit(f"shard_idx {I} out of range for shards={S}")
    stems = [s for i, s in enumerate(stems_all) if (i % S) == I]

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # -------------------------------------------------------------------------
    # Load Inception-I3D backbone (Kinetics-400 RGB)
    # -------------------------------------------------------------------------
    print(f"Loading I3D weights from {args.weights}")
    i3d = InceptionI3d(num_classes=400, in_channels=3)
    state = torch.load(args.weights, map_location="cpu")
    i3d.load_state_dict(state)
    i3d.to(device).eval()

    feat_dim = 1024  # Inception-I3D avg_pool outputs 1024-D
    features = {}

    def hook(module, inp, out):
        # out: [B, 1024, T', 1, 1] after avg_pool
        # We want a single 1024-D vector per clip: average over temporal dim T'
        z = out.mean(dim=2)          # [B, 1024, 1, 1]
        features["z"] = z.flatten(1) # [B, 1024]

    i3d.avg_pool.register_forward_hook(hook)

    # -------------------------------------------------------------------------
    # Preprocessing: I3D expects RGB in [-1, 1], 224x224 crops
    # -------------------------------------------------------------------------
    tfm = T.Compose([
        T.ToTensor(),                                     # [H,W,3] uint8 -> [3,H,W] float in [0,1]
        T.ConvertImageDtype(torch.float32),
        T.Resize(256),                                    # resize short side
        T.CenterCrop(args.imgsz),                         # center crop to 224x224
        T.Normalize(mean=[0.5, 0.5, 0.5],
                    std=[0.5, 0.5, 0.5]),                  # scale to [-1, 1]
    ])

    print(f"Processing {len(stems)} stems (shard {I}/{S})")

    for stem in stems:
        out_npy = OUT / f"{stem}_x.npy"
        out_meta = OUT / f"{stem}_meta.json"
        if out_npy.exists():
            print(f"⏭ Skipping {stem} (already exists)")
            continue

        frame_paths = load_frame_paths(frames_root, stem)
        if not frame_paths:
            print(f"⚠ No frames for stem {stem}, skipping")
            continue

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

        video_tensor = load_video_tensor(
            frame_paths, tfm,
            crop_top=crop_top, crop_bottom=crop_bottom,
            crop_left=crop_left, crop_right=crop_right,
            vertical_flip=apply_vflip_to_stem,
            horizontal_flip=apply_hflip_to_stem
        )  # [F, 3, H, W]
        F = video_tensor.size(0)
        if F == 0:
            print(f"⚠ Failed to load any frames for {stem}, skipping")
            continue

        # Map target-fps timeline (args.fps) to center indices in frame space (EXTRACTION_FPS)
        src_fps = EXTRACTION_FPS
        T_out = int(math.ceil((F / src_fps) * args.fps))
        centers = [
            min(F - 1, int(round((t / args.fps) * src_fps)))
            for t in range(T_out)
        ]
        step = max(1, int(args.center_step))
        centers = centers[::step]
        if not centers:
            print(f"⚠ No centers computed for {stem}, skipping")
            continue

        half = args.win // 2
        B = max(1, int(args.batch_size))
        X_chunks = []

        def clip_tensor(center_idx: int) -> torch.Tensor:
            """
            Build a [3, T, H, W] clip around center_idx using strides on preloaded frames.
            """
            idxs = np.arange(
                center_idx - half * args.stride,
                center_idx + half * args.stride,
                args.stride,
            )
            idxs = np.clip(idxs, 0, F - 1).astype(int)
            clip = video_tensor[idxs]            # [T, 3, H, W]
            clip = clip.permute(1, 0, 2, 3)      # [3, T, H, W]
            return clip

        for s_idx in range(0, len(centers), B):
            batch_centers = centers[s_idx:s_idx + B]
            if not batch_centers:
                continue
            clips = torch.stack([clip_tensor(c) for c in batch_centers], dim=0).to(device)  # [B, 3, T, H, W]

            features.clear()
            # Use extract_features() which stops before logits, but we hook avg_pool anyway
            _ = i3d.extract_features(clips)
            Z = features["z"].detach().cpu().numpy().astype(np.float32)  # [B, 1024]
            X_chunks.append(Z)

        if not X_chunks:
            print(f"⚠ No features extracted for {stem}, skipping")
            continue

        X = np.concatenate(X_chunks, axis=0)  # [N_clips, 1024]
        np.save(out_npy, X)

        meta = {
            "stem": stem,
            "fps": float(args.fps),
            "src_fps": float(EXTRACTION_FPS),
            "n_frames": int(F),
            "T": int(X.shape[0]),
            "D": int(X.shape[1]),
            "stride": int(args.stride),
            "center_step": int(args.center_step),
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

    print(f"\n✓ Feature extraction complete. Output: {OUT}")


if __name__ == "__main__":
    main()
