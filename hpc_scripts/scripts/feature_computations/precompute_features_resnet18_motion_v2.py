#!/usr/bin/env python3
"""
ResNet18 RGB + Motion feature extractor for MS-TCN training.
Enhanced version with cropping and flip support for mydata alignment with FineBio.

Output: *_x.npy files with shape [T, D] where D=512 (RGB only) or D=1024 (RGB+Motion)
"""
import argparse
import pathlib
import json
import math
import re
from collections import deque
import numpy as np
import torch
import cv2
from torchvision.models import resnet18, ResNet18_Weights
import torchvision.transforms as T


# Pattern to identify FineBio stems (P##_##_##_T# format)
FINEBIO_PATTERN = re.compile(r'^P\d{2}_\d{2}_\d{2}_T\d')


def apply_motion_mask(diff: np.ndarray, threshold: int = 15, morph_kernel_size: int = 5) -> np.ndarray:
    """
    Apply morphological masking to frame difference to reduce noise.
    
    Args:
        diff: Frame difference [H, W, 3] in range [0, 255]
        threshold: Pixel difference threshold (changes below this are noise)
        morph_kernel_size: Size of morphological kernel for cleaning
    
    Returns:
        Masked difference with noise removed
    """
    # Convert to grayscale for masking
    diff_gray = np.mean(diff, axis=2).astype(np.uint8)
    
    # Threshold to create binary mask
    _, binary_mask = cv2.threshold(diff_gray, threshold, 255, cv2.THRESH_BINARY)
    
    # Morphological operations to clean up mask
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph_kernel_size, morph_kernel_size))
    # Opening: removes small noise blobs
    mask_cleaned = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel)
    # Closing: fills small gaps in motion regions
    mask_cleaned = cv2.morphologyEx(mask_cleaned, cv2.MORPH_CLOSE, kernel)
    
    # Apply mask to original difference (broadcast mask to 3 channels)
    mask_3ch = mask_cleaned[:, :, np.newaxis] / 255.0
    motion_masked = (diff.astype(np.float32) * mask_3ch).astype(np.uint8)
    
    return motion_masked


def is_mydata_stem(stem: str) -> bool:
    """
    Check if stem belongs to mydata (needs cropping/flipping) vs FineBio.
    
    mydata stems: timestamp format (YYYYMMDD_*) or Lab_video*
    FineBio stems: P##_##_##_T# format (e.g., P01_05_01_T5)
    """
    if FINEBIO_PATTERN.match(stem):
        return False  # FineBio - don't crop/flip
    return True  # mydata - apply crop/flip


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


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="Dataset root containing videos/")
    ap.add_argument("--splits", required=True, help="Path to splits.json")
    ap.add_argument("--fps", type=float, default=10.0, help="Target FPS for features")
    ap.add_argument("--imgsz", type=int, default=224, help="Image size for ResNet")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", required=True, help="Output directory for features")
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--motion", action="store_true", help="Include motion features (frame diff)")
    ap.add_argument("--motion_threshold", type=int, default=15,
                    help="Pixel difference threshold for motion masking (0 = no masking)")
    ap.add_argument("--motion_morph_kernel", type=int, default=5,
                    help="Morphological kernel size for noise removal")
    ap.add_argument("--shards", type=int, default=1, help="Split videos across N shards")
    ap.add_argument("--shard_idx", type=int, default=0, help="0-based shard index")
    
    # Cropping parameters (for focusing on work area)
    ap.add_argument("--crop_top", type=float, default=0.0,
                    help="Fraction to crop from top (0.0-1.0)")
    ap.add_argument("--crop_bottom", type=float, default=0.0,
                    help="Fraction to crop from bottom (0.0-1.0)")
    ap.add_argument("--crop_left", type=float, default=0.0,
                    help="Fraction to crop from left (0.0-1.0)")
    ap.add_argument("--crop_right", type=float, default=0.0,
                    help="Fraction to crop from right (0.0-1.0)")
    ap.add_argument("--selective_crop", action="store_true",
                    help="Only apply cropping to mydata stems, not FineBio")
    
    # Flip parameters (for viewpoint alignment)
    ap.add_argument("--vertical_flip", action="store_true",
                    help="Apply vertical flip to mydata stems (aligns top-down with bottom-up)")
    ap.add_argument("--horizontal_flip", action="store_true",
                    help="Apply horizontal flip to mydata stems (left/right invariance)")
    
    args = ap.parse_args()

    ROOT = pathlib.Path(args.root)
    OUT = pathlib.Path(args.out)
    OUT.mkdir(parents=True, exist_ok=True)

    # Check crop settings
    has_crop_params = args.crop_top > 0 or args.crop_bottom > 0 or args.crop_left > 0 or args.crop_right > 0
    if has_crop_params:
        crop_msg = f"Cropping: top={args.crop_top:.0%}, bottom={args.crop_bottom:.0%}, " \
                   f"left={args.crop_left:.0%}, right={args.crop_right:.0%}"
        if args.selective_crop:
            print(f"Selective {crop_msg} (mydata only)")
        else:
            print(f"Using {crop_msg} (all videos)")
    
    if args.vertical_flip:
        print("Vertical flip: enabled for mydata stems")
    if args.horizontal_flip:
        print("Horizontal flip: enabled for mydata stems")

    # Load splits and get all video stems
    sp = json.loads(pathlib.Path(args.splits).read_text())
    all_stems = []
    for split in ["train", "val", "test"]:
        all_stems.extend([pathlib.Path(p).stem for p in sp.get(split, [])])

    # Sharding
    S = max(1, args.shards)
    I = args.shard_idx
    if I < 0 or I >= S:
        raise SystemExit(f"shard_idx {I} out of range for shards={S}")
    stems = [s for i, s in enumerate(all_stems) if (i % S) == I]

    print(f"Processing {len(stems)} videos (shard {I}/{S})")

    # Find video files
    videos_dir = ROOT / "videos"
    if not videos_dir.exists():
        videos_dir = ROOT
    
    vids = {}
    for ext in ["*.mp4", "*.MP4", "*.avi", "*.AVI", "*.mov", "*.MOV"]:
        for v in videos_dir.glob(ext):
            vids[v.stem] = v
    
    print(f"Found {len(vids)} video files in {videos_dir}")

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load ResNet18
    weights = ResNet18_Weights.IMAGENET1K_V1
    model = resnet18(weights=weights).to(device).eval()

    # Hook to extract features before FC layer
    features = {}
    def hook(module, inp, out):
        features["z"] = out.flatten(1)  # [B, 512]
    model.avgpool.register_forward_hook(hook)

    # Preprocessing
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    tfm = T.Compose([
        T.ToTensor(),
        T.ConvertImageDtype(torch.float32),
        T.Resize((args.imgsz, args.imgsz), antialias=True),
        T.Normalize(mean=mean, std=std),
    ])

    for stem in stems:
        vpath = vids.get(stem)
        if not vpath:
            print(f"⚠ Video not found for stem: {stem}")
            continue

        out_npy = OUT / f"{stem}_x.npy"
        if out_npy.exists():
            print(f"⏭ Skipping {stem} (already exists)")
            continue

        # Determine transforms for this stem
        is_mydata = is_mydata_stem(stem)
        
        if args.selective_crop and is_mydata and has_crop_params:
            crop_top, crop_bottom = args.crop_top, args.crop_bottom
            crop_left, crop_right = args.crop_left, args.crop_right
        elif not args.selective_crop and has_crop_params:
            crop_top, crop_bottom = args.crop_top, args.crop_bottom
            crop_left, crop_right = args.crop_left, args.crop_right
        else:
            crop_top, crop_bottom, crop_left, crop_right = 0.0, 0.0, 0.0, 0.0
        
        apply_vflip = args.vertical_flip and is_mydata
        apply_hflip = args.horizontal_flip and is_mydata

        # Read video
        cap = cv2.VideoCapture(str(vpath))
        src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if n_frames <= 0:
            print(f"⚠ Empty video: {stem}")
            cap.release()
            continue

        indicators = []
        if crop_top > 0 or crop_bottom > 0 or crop_left > 0 or crop_right > 0:
            indicators.append("cropped")
        if apply_vflip:
            indicators.append("vflip")
        if apply_hflip:
            indicators.append("hflip")
        indicator_str = f" [{'+'.join(indicators)}]" if indicators else ""
        print(f"Processing {stem}: {n_frames} frames @ {src_fps:.1f} fps -> {args.fps} Hz{indicator_str}", flush=True)

        # Compute frame indices at target FPS
        T_out = int(math.ceil((n_frames / src_fps) * args.fps))
        frame_indices = [min(n_frames - 1, int(round((t / args.fps) * src_fps))) for t in range(T_out)]
        if not frame_indices:
            print(f"⚠ No target frames computed for {stem}")
            cap.release()
            continue

        frame_indices = deque(frame_indices)
        next_center = frame_indices[0]

        rgb_batches = []
        motion_batches = []
        X_rgb_chunks = []
        X_motion_chunks = []
        prev_sample_rgb = None
        B = args.batch_size

        def run_model_on_batch(batch_tensors, dest_chunks):
            if not batch_tensors:
                return
            tensor = torch.stack(batch_tensors).to(device)
            features.clear()
            _ = model(tensor)
            dest_chunks.append(features["z"].cpu().numpy())
            batch_tensors.clear()

        frame_idx = -1
        processed = 0
        while True:
            ok, frame_bgr = cap.read()
            if not ok:
                break
            frame_idx += 1

            while frame_indices and frame_idx == next_center:
                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                
                # Apply cropping
                frame_rgb = apply_crop(frame_rgb, crop_top, crop_bottom, crop_left, crop_right)
                
                # Apply vertical flip
                if apply_vflip:
                    frame_rgb = np.flipud(frame_rgb).copy()
                
                # Apply horizontal flip
                if apply_hflip:
                    frame_rgb = np.fliplr(frame_rgb).copy()
                
                rgb_batches.append(tfm(frame_rgb))

                if args.motion:
                    if prev_sample_rgb is None:
                        motion_rgb = np.zeros_like(frame_rgb)
                    else:
                        diff = np.abs(frame_rgb.astype(np.float32) - prev_sample_rgb.astype(np.float32))
                        
                        # Apply morphological masking to remove noise
                        if args.motion_threshold > 0:
                            motion_rgb = apply_motion_mask(
                                diff,
                                threshold=args.motion_threshold,
                                morph_kernel_size=args.motion_morph_kernel
                            )
                        else:
                            motion_rgb = diff.astype(np.uint8)
                    
                    motion_batches.append(tfm(motion_rgb))
                    prev_sample_rgb = frame_rgb.copy()
                else:
                    prev_sample_rgb = frame_rgb.copy()

                processed += 1

                if len(rgb_batches) >= B:
                    run_model_on_batch(rgb_batches, X_rgb_chunks)
                    if args.motion:
                        run_model_on_batch(motion_batches, X_motion_chunks)

                frame_indices.popleft()
                if frame_indices:
                    next_center = frame_indices[0]
                else:
                    next_center = None
                    break

            if not frame_indices:
                break

        cap.release()

        # Flush remaining samples
        run_model_on_batch(rgb_batches, X_rgb_chunks)
        if args.motion:
            run_model_on_batch(motion_batches, X_motion_chunks)

        if processed != T_out:
            print(f"⚠ Warning: expected {T_out} samples, processed {processed}")

        if not X_rgb_chunks:
            print(f"⚠ No RGB features extracted for {stem}")
            continue

        X_rgb = np.concatenate(X_rgb_chunks, axis=0)  # [T, 512]

        if args.motion:
            if not X_motion_chunks:
                print(f"⚠ No motion features extracted for {stem}")
                continue
            X_motion = np.concatenate(X_motion_chunks, axis=0)  # [T, 512]
            X = np.concatenate([X_rgb, X_motion], axis=1)  # [T, 1024]
        else:
            X = X_rgb  # [T, 512]

        X = X.astype(np.float32)
        np.save(out_npy, X)
        
        # Save metadata
        meta = {
            "stem": stem,
            "fps": args.fps,
            "src_fps": src_fps,
            "n_frames": n_frames,
            "T": X.shape[0],
            "D": X.shape[1],
            "motion": args.motion,
        }
        if crop_top > 0 or crop_bottom > 0 or crop_left > 0 or crop_right > 0:
            meta["crop"] = {
                "top": crop_top,
                "bottom": crop_bottom,
                "left": crop_left,
                "right": crop_right,
            }
        if apply_vflip:
            meta["vertical_flip"] = True
        if apply_hflip:
            meta["horizontal_flip"] = True
        (OUT / f"{stem}_meta.json").write_text(json.dumps(meta, indent=2))
        
        print(f"  ✓ {X.shape} saved → {out_npy}")

    print(f"\n✓ Feature extraction complete. Output: {OUT}")


if __name__ == "__main__":
    main()
