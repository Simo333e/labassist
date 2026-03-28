#!/usr/bin/env python3
"""
ResNet18 RGB + Motion feature extractor for MS-TCN training.
Generic version that works with any video naming convention.

Output: *_x.npy files with shape [T, D] where D=512 (RGB only) or D=1024 (RGB+Motion)
"""
import argparse
import pathlib
import json
import math
import random
from collections import deque
import numpy as np
import torch
import cv2
from torchvision.models import resnet18, ResNet18_Weights
import torchvision.transforms as T


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
    ap.add_argument("--center_jitter", type=int, default=0,
                    help="Max +/- frame jitter applied to sampled centers")
    ap.add_argument("--seed", type=int, default=1337, help="Random seed for jitter/augmentations")
    args = ap.parse_args()

    ROOT = pathlib.Path(args.root)
    OUT = pathlib.Path(args.out)
    OUT.mkdir(parents=True, exist_ok=True)

    # Load splits and get all video stems
    sp = json.loads(pathlib.Path(args.splits).read_text())
    all_stems = []
    for split in ["train", "val", "test"]:
        all_stems.extend([pathlib.Path(p).stem for p in sp.get(split, [])])

    print(f"Processing {len(all_stems)} videos from splits")

    # Find video files (any .mp4 in videos/ directory)
    videos_dir = ROOT / "videos"
    if not videos_dir.exists():
        # Try root directly
        videos_dir = ROOT
    
    # Build mapping of stem -> video path
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

    rng = random.Random(args.seed)

    for stem in all_stems:
        vpath = vids.get(stem)
        if not vpath:
            print(f"⚠ Video not found for stem: {stem}")
            continue

        out_npy = OUT / f"{stem}_x.npy"
        if out_npy.exists():
            print(f"⏭ Skipping {stem} (already exists)")
            continue

        # Read video
        cap = cv2.VideoCapture(str(vpath))
        src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if n_frames <= 0:
            print(f"⚠ Empty video: {stem}")
            cap.release()
            continue

        print(f"Processing {stem}: {n_frames} frames @ {src_fps:.1f} fps -> {args.fps} Hz", flush=True)

        # Compute frame indices at target FPS
        T_out = int(math.ceil((n_frames / src_fps) * args.fps))
        frame_indices = [min(n_frames - 1, int(round((t / args.fps) * src_fps))) for t in range(T_out)]
        if args.center_jitter > 0:
            jittered = []
            for idx in frame_indices:
                shift = rng.randint(-args.center_jitter, args.center_jitter)
                new_idx = min(n_frames - 1, max(0, idx + shift))
                if jittered and new_idx < jittered[-1]:
                    new_idx = jittered[-1]
                jittered.append(new_idx)
            frame_indices = jittered
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
        prev_sample_rgb = None  # numpy array (RGB) of previous sampled center
        B = args.batch_size

        def run_model_on_batch(batch_tensors, dest_chunks, desc):
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
                    run_model_on_batch(rgb_batches, X_rgb_chunks, "rgb")
                    if args.motion:
                        run_model_on_batch(motion_batches, X_motion_chunks, "motion")

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
        run_model_on_batch(rgb_batches, X_rgb_chunks, "rgb-final")
        if args.motion:
            run_model_on_batch(motion_batches, X_motion_chunks, "motion-final")

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
        (OUT / f"{stem}_meta.json").write_text(json.dumps(meta, indent=2))
        
        print(f" -> {X.shape} saved")

    print(f"\n✓ Feature extraction complete. Output: {OUT}")


if __name__ == "__main__":
    main()



