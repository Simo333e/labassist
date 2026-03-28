#!/usr/bin/env python3
"""
ResNet18 RGB + Motion feature extractor for MS-TCN training.
Compatible with Hailo 8 deployment (INT8 quantizable architecture).
"""
import argparse, pathlib, json, math, numpy as np, torch, cv2
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
    ap.add_argument("--root", required=True, help="FineBio root containing videos/")
    ap.add_argument("--splits", required=True)
    ap.add_argument("--view", default="T1")
    ap.add_argument("--fps", type=float, default=10.0)
    ap.add_argument("--imgsz", type=int, default=224)  # ResNet native size
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", default=None)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--motion", action="store_true", help="Include motion features")
    ap.add_argument("--motion_mode", choices=["diff", "flow"], default="diff")
    ap.add_argument("--motion_threshold", type=int, default=15,
                    help="Pixel difference threshold for motion masking (0 = no masking)")
    ap.add_argument("--motion_morph_kernel", type=int, default=5,
                    help="Morphological kernel size for noise removal")
    ap.add_argument("--shards", type=int, default=1)
    ap.add_argument("--shard_idx", type=int, default=0)
    args = ap.parse_args()

    ROOT = pathlib.Path(args.root)
    suffix = "_motion" if args.motion else ""
    OUT = pathlib.Path(args.out or (ROOT/f"cache/features_{int(args.fps)}hz_resnet18{suffix}"))
    OUT.mkdir(parents=True, exist_ok=True)

    # Load splits
    sp = json.loads(pathlib.Path(args.splits).read_text())
    sp = sp[next(iter(sp))] if "train" not in sp else sp
    stems_all = [pathlib.Path(p).stem for p in sp["train"] + sp["val"] + sp["test"]]
    
    # Sharding support
    S, I = max(1, args.shards), args.shard_idx
    stems = [s for i, s in enumerate(stems_all) if (i % S) == I]

    vids = {v.stem: v for v in (ROOT/"videos").glob(f"*_{args.view}.mp4")}
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # ResNet18 feature extractor (512D from avgpool)
    weights = ResNet18_Weights.IMAGENET1K_V1
    model = resnet18(weights=weights).to(device).eval()
    
    # Hook to extract features before FC
    features = {}
    def hook(module, inp, out):
        features["z"] = out.flatten(1)  # [B, 512]
    model.avgpool.register_forward_hook(hook)

    # ImageNet normalization
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    tfm = T.Compose([
        T.ToTensor(),
        T.Resize((args.imgsz, args.imgsz)),
        T.Normalize(mean=mean, std=std),
    ])

    for stem in stems:
        vpath = vids.get(stem)
        if not vpath:
            continue
        out_npy = OUT / f"{stem}_x.npy"
        if out_npy.exists():
            continue

        # Read video frames
        cap = cv2.VideoCapture(str(vpath))
        src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        nframes = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        if nframes <= 0:
            cap.release()
            continue

        frames = []
        while True:
            ok, f = cap.read()
            if not ok:
                break
            frames.append(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
        cap.release()
        if not frames:
            continue

        # Sample at target FPS
        T_out = int(math.ceil((nframes / src_fps) * args.fps))
        centers = [min(len(frames)-1, int(round((t/args.fps) * src_fps))) for t in range(T_out)]

        # Extract RGB features
        rgb_tensors = torch.stack([tfm(frames[c]) for c in centers])  # [T, 3, H, W]
        X_rgb = []
        B = args.batch_size
        for s in range(0, len(rgb_tensors), B):
            batch = rgb_tensors[s:s+B].to(device)
            features.clear()
            _ = model(batch)
            X_rgb.append(features["z"].cpu().numpy())
        X_rgb = np.concatenate(X_rgb, axis=0)  # [T, 512]

        if args.motion:
            # Compute motion features (frame differencing with masking)
            motion_frames = []
            for i, c in enumerate(centers):
                if i == 0:
                    # First frame: zero motion
                    motion_frames.append(np.zeros_like(frames[c]))
                else:
                    prev_c = centers[i-1]
                    diff = np.abs(frames[c].astype(np.float32) - frames[prev_c].astype(np.float32))
                    
                    # Apply morphological masking to remove noise
                    if args.motion_threshold > 0:
                        diff_masked = apply_motion_mask(
                            diff, 
                            threshold=args.motion_threshold,
                            morph_kernel_size=args.motion_morph_kernel
                        )
                        motion_frames.append(diff_masked)
                    else:
                        motion_frames.append(diff.astype(np.uint8))
            
            motion_tensors = torch.stack([tfm(m) for m in motion_frames])
            X_motion = []
            for s in range(0, len(motion_tensors), B):
                batch = motion_tensors[s:s+B].to(device)
                features.clear()
                _ = model(batch)
                X_motion.append(features["z"].cpu().numpy())
            X_motion = np.concatenate(X_motion, axis=0)  # [T, 512]
            
            # Concatenate RGB + Motion
            X = np.concatenate([X_rgb, X_motion], axis=1)  # [T, 1024]
        else:
            X = X_rgb

        np.save(out_npy, X.astype(np.float32))
        meta = {"stem": stem, "fps": args.fps, "T": X.shape[0], "D": X.shape[1], "motion": args.motion}
        (OUT / f"{stem}_meta.json").write_text(json.dumps(meta))
        print(f"{stem}: X{tuple(X.shape)} saved → {out_npy}")

if __name__ == "__main__":
    main()