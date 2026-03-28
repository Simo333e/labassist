#!/usr/bin/env python3
"""
YOLO Object Detection Feature Extraction for Action Recognition.

This script runs a trained YOLO model on video frames and extracts object detection
features that can be used as additional input to MS-TCN for action classification.

The key insight: "Pipette pcr tube" vs "Aspirate reagent" are visually similar actions
that differ primarily in the TARGET OBJECT. By detecting PCR tubes vs reagent bottles,
we can provide the model with discriminative features.

Feature encoding per frame (28 dimensions):
- Object confidence (14 dims): Max confidence score per class
- Key object positions (10 dims): Normalized (cx, cy) for 5 key objects
- Hand-object proximity (4 dims): Distance from hand to key containers

Output: One <stem>_x.npy file per video containing [T, 28] features at target FPS.
"""
import argparse
import json
import math
import pathlib
import re
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

# YOLO class mapping (from data.yaml)
CLASS_NAMES = {
    0: 'Bin',
    1: 'Pipette-w-tip',
    2: 'Pipette-wo-tip',
    3: 'MasterMix',
    4: 'Water',
    5: 'Reagents',
    6: 'pcr tube',
    7: 'hand',
    8: 'pipette_tip',
    9: 'pippe_tip_box',
    10: 'pipette_tip_on_pippette',
    11: 'Reagent open',
    12: 'ice bucket',
    13: 'pippette_tip',
}
NUM_CLASSES = 14

# Key objects for action disambiguation
# These are the objects that help distinguish between similar actions
KEY_OBJECTS = ['pcr tube', 'Reagents', 'MasterMix', 'Water', 'hand']
KEY_CLASS_IDS = {name: idx for idx, name in CLASS_NAMES.items() if name in KEY_OBJECTS}

# Frame extraction FPS (must match what was used for frame extraction)
EXTRACTION_FPS = 25.0

# Pattern to identify FineBio stems (P##_##_##_T# format)
FINEBIO_PATTERN = re.compile(r'^P\d{2}_\d{2}_\d{2}_T\d')


def is_mydata_stem(stem: str) -> bool:
    """Check if stem belongs to mydata (needs cropping) vs FineBio (no crop)."""
    if FINEBIO_PATTERN.match(stem):
        return False  # FineBio
    return True  # mydata


def compute_distance(pos1: Optional[Tuple[float, float]], 
                     pos2: Optional[Tuple[float, float]]) -> float:
    """Euclidean distance between two normalized positions."""
    if pos1 is None or pos2 is None:
        return 1.0  # max distance if object not detected
    return np.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)


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


def encode_detections(boxes, img_w: int, img_h: int) -> np.ndarray:
    """
    Encode YOLO detections into a fixed-size feature vector.
    
    Args:
        boxes: YOLO Results.boxes object
        img_w, img_h: Image dimensions for normalization
        
    Returns:
        numpy array of shape [28] containing:
        - [0:14]  Object confidence per class (max)
        - [14:24] Key object positions (cx, cy for 5 objects)
        - [24:28] Hand-object proximity (4 distances)
    """
    # Initialize feature arrays
    obj_conf = np.zeros(NUM_CLASSES, dtype=np.float32)
    key_positions: Dict[str, Tuple[float, float, float]] = {}  # name -> (cx, cy, conf)
    
    # Process each detection
    if boxes is not None and len(boxes) > 0:
        for i in range(len(boxes)):
            cls_id = int(boxes.cls[i].item())
            conf = float(boxes.conf[i].item())
            
            # Update max confidence for this class
            if conf > obj_conf[cls_id]:
                obj_conf[cls_id] = conf
            
            # Get bounding box
            x1, y1, x2, y2 = boxes.xyxy[i].cpu().numpy()
            cx = (x1 + x2) / 2 / img_w  # normalized center x
            cy = (y1 + y2) / 2 / img_h  # normalized center y
            
            # Store position for key objects (keep highest confidence detection)
            if cls_id in KEY_CLASS_IDS.values():
                class_name = CLASS_NAMES[cls_id]
                if class_name not in key_positions or conf > key_positions[class_name][2]:
                    key_positions[class_name] = (cx, cy, conf)
    
    # Extract positions for key objects (default to center if not detected)
    pos_features = []
    for obj_name in KEY_OBJECTS:
        if obj_name in key_positions:
            pos_features.extend([key_positions[obj_name][0], key_positions[obj_name][1]])
        else:
            pos_features.extend([0.5, 0.5])  # center as default (undetected)
    
    # Compute hand-to-object proximity
    hand_pos = key_positions.get('hand', (0.5, 0.5, 0))[:2] if 'hand' in key_positions else None
    proximity = []
    for obj_name in ['pcr tube', 'Reagents', 'MasterMix', 'Water']:
        obj_pos = key_positions.get(obj_name, None)
        if obj_pos:
            dist = compute_distance(hand_pos, obj_pos[:2])
        else:
            dist = 1.0  # max distance if object not detected
        proximity.append(dist)
    
    # Combine all features
    features = np.concatenate([
        obj_conf,                          # 14 dims: object confidence per class
        np.array(pos_features, dtype=np.float32),  # 10 dims: key object positions
        np.array(proximity, dtype=np.float32),     # 4 dims: hand-object proximity
    ])
    
    return features  # 28 dims total


def load_frame_paths(frames_root: pathlib.Path, stem: str) -> List[pathlib.Path]:
    """Return sorted list of frame paths for a given stem."""
    d = frames_root / stem
    if not d.exists():
        return []
    return sorted(d.glob("*.jpg"))


def find_video_file(videos_dir: pathlib.Path, stem: str) -> Optional[pathlib.Path]:
    """Find video file matching stem with common extensions."""
    for ext in ['.mp4', '.avi', '.mov', '.mkv', '.MP4', '.AVI', '.MOV']:
        path = videos_dir / f"{stem}{ext}"
        if path.exists():
            return path
    return None


def main():
    ap = argparse.ArgumentParser(description="Extract YOLO object detection features")
    ap.add_argument("--root", required=True, help="Dataset root containing frames/ or videos/")
    ap.add_argument("--splits", required=True, help="Path to splits.json")
    ap.add_argument("--use_video", action="store_true",
                    help="Read from full-resolution video instead of pre-extracted frames")
    ap.add_argument("--frames_dir", default=None,
                    help="Custom frames directory (default: root/frames). Use for hires preprocessed frames.")
    ap.add_argument("--frames_preprocessed", action="store_true",
                    help="Frames are already preprocessed (cropped+flipped), skip preprocessing")
    ap.add_argument("--weights", required=True, help="Path to YOLO weights (best.pt)")
    ap.add_argument("--fps", type=float, default=10.0, help="Target FPS for feature timeline")
    ap.add_argument("--conf", type=float, default=0.25, help="YOLO confidence threshold")
    ap.add_argument("--imgsz", type=int, default=1280, help="YOLO inference image size")
    ap.add_argument("--device", default="cuda:0", help="Device for inference")
    ap.add_argument("--out", default=None, help="Output directory")
    ap.add_argument("--batch_size", type=int, default=16, help="Batch size for YOLO inference")
    
    # Cropping parameters (must match I3D extraction for consistency)
    ap.add_argument("--crop_top", type=float, default=0.0)
    ap.add_argument("--crop_bottom", type=float, default=0.0)
    ap.add_argument("--crop_left", type=float, default=0.0)
    ap.add_argument("--crop_right", type=float, default=0.0)
    ap.add_argument("--selective_crop", action="store_true",
                    help="Only apply cropping to mydata stems")
    ap.add_argument("--vertical_flip", action="store_true",
                    help="Apply vertical flip to mydata stems")
    ap.add_argument("--horizontal_flip", action="store_true",
                    help="Apply horizontal flip to mydata stems")
    
    # Sharding for parallel processing
    ap.add_argument("--shards", type=int, default=1)
    ap.add_argument("--shard_idx", type=int, default=0)
    
    args = ap.parse_args()
    
    # Import YOLO here to avoid issues if ultralytics not installed
    try:
        from ultralytics import YOLO
    except ImportError:
        raise ImportError("Please install ultralytics: pip install ultralytics")
    
    ROOT = pathlib.Path(args.root)
    OUT = pathlib.Path(args.out or (ROOT / "cache/features_10hz_yolo"))
    OUT.mkdir(parents=True, exist_ok=True)
    
    # Determine frame source
    if args.frames_dir:
        frames_root = pathlib.Path(args.frames_dir)
    else:
        frames_root = ROOT / "frames"
    videos_root = ROOT / "videos"
    
    if args.use_video:
        if not videos_root.exists():
            raise SystemExit(f"Videos directory not found: {videos_root}")
        print(f"📹 Reading from FULL-RESOLUTION videos (better for object detection)")
    elif args.frames_preprocessed:
        print(f"📁 Reading from HIGH-RES PREPROCESSED frames: {frames_root}")
        print(f"   (Frames already cropped+flipped, skipping preprocessing)")
    else:
        print(f"📁 Reading from pre-extracted frames: {frames_root}")
        print(f"   (May lose small object detail if low-resolution)")
    
    # Check crop parameters
    has_crop_params = (args.crop_top > 0 or args.crop_bottom > 0 or 
                       args.crop_left > 0 or args.crop_right > 0)
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
    
    # Load splits
    sp = json.loads(pathlib.Path(args.splits).read_text())
    sp = sp[next(iter(sp))] if "train" not in sp else sp
    stems_all = [pathlib.Path(p).stem for p in sp.get("train", []) + sp.get("val", []) + sp.get("test", [])]
    
    # Sharding
    S = max(1, int(args.shards))
    I = int(args.shard_idx)
    if I < 0 or I >= S:
        raise SystemExit(f"shard_idx {I} out of range for shards={S}")
    stems = [s for i, s in enumerate(stems_all) if (i % S) == I]
    
    # Load YOLO model
    print(f"Loading YOLO model from {args.weights}")
    model = YOLO(args.weights)
    model.to(args.device)
    
    print(f"Processing {len(stems)} stems (shard {I}/{S})")
    print(f"Feature dimension: {NUM_CLASSES + len(KEY_OBJECTS)*2 + 4} = 28")
    
    for stem in stems:
        out_npy = OUT / f"{stem}_x.npy"
        out_meta = OUT / f"{stem}_meta.json"
        
        if out_npy.exists():
            print(f"⏭ Skipping {stem} (already exists)")
            continue
        
        # Determine cropping/flipping for this stem
        # If frames are already preprocessed, skip all preprocessing
        if args.frames_preprocessed:
            crop_top, crop_bottom, crop_left, crop_right = 0.0, 0.0, 0.0, 0.0
            apply_vflip = False
            apply_hflip = False
            apply_crop_to_stem = False  # Already preprocessed, no crop needed
        else:
            if args.selective_crop:
                apply_crop_to_stem = is_mydata_stem(stem) and has_crop_params
            else:
                apply_crop_to_stem = has_crop_params
                
            if apply_crop_to_stem:
                crop_top, crop_bottom = args.crop_top, args.crop_bottom
                crop_left, crop_right = args.crop_left, args.crop_right
            else:
                crop_top, crop_bottom, crop_left, crop_right = 0.0, 0.0, 0.0, 0.0
                
            apply_vflip = args.vertical_flip and is_mydata_stem(stem)
            apply_hflip = args.horizontal_flip and is_mydata_stem(stem)
        
        # Process from video or pre-extracted frames
        if args.use_video:
            # Read directly from video at full resolution
            video_path = find_video_file(videos_root, stem)
            if video_path is None:
                print(f"⚠ No video found for {stem}, skipping")
                continue
            
            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                print(f"⚠ Cannot open video {video_path}, skipping")
                continue
            
            video_fps = cap.get(cv2.CAP_PROP_FPS)
            n_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            src_fps = video_fps  # Use actual video fps
            n_frames = n_video_frames
            
            # Compute which video frames to sample
            T_out = int(math.ceil((n_frames / src_fps) * args.fps))
            centers = [
                min(n_frames - 1, int(round((t / args.fps) * src_fps)))
                for t in range(T_out)
            ]
            
            if not centers:
                cap.release()
                print(f"⚠ No centers computed for {stem}, skipping")
                continue
            
            # Process frames
            all_features = []
            batch_frames = []
            batch_sizes = []
            
            for center_idx in centers:
                cap.set(cv2.CAP_PROP_POS_FRAMES, center_idx)
                ret, img_bgr = cap.read()
                
                if not ret or img_bgr is None:
                    all_features.append(np.zeros(28, dtype=np.float32))
                    continue
                
                img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
                img_rgb = apply_crop(img_rgb, crop_top, crop_bottom, crop_left, crop_right)
                
                if apply_vflip:
                    img_rgb = np.flipud(img_rgb).copy()
                if apply_hflip:
                    img_rgb = np.fliplr(img_rgb).copy()
                
                h, w = img_rgb.shape[:2]
                batch_frames.append(img_rgb)
                batch_sizes.append((w, h))
                
                if len(batch_frames) >= args.batch_size:
                    results = model(batch_frames, conf=args.conf, verbose=False,
                                   imgsz=args.imgsz, device=args.device)
                    for res, (img_w, img_h) in zip(results, batch_sizes):
                        features = encode_detections(res.boxes, img_w, img_h)
                        all_features.append(features)
                    batch_frames = []
                    batch_sizes = []
            
            cap.release()
            
            # Process remaining batch
            if batch_frames:
                results = model(batch_frames, conf=args.conf, verbose=False,
                               imgsz=args.imgsz, device=args.device)
                for res, (img_w, img_h) in zip(results, batch_sizes):
                    features = encode_detections(res.boxes, img_w, img_h)
                    all_features.append(features)
        
        else:
            # Read from pre-extracted frames (original behavior)
            frame_paths = load_frame_paths(frames_root, stem)
            if not frame_paths:
                print(f"⚠ No frames for {stem}, skipping")
                continue
            
            n_frames = len(frame_paths)
            src_fps = EXTRACTION_FPS
            
            # Compute frame centers at target FPS
            T_out = int(math.ceil((n_frames / src_fps) * args.fps))
            centers = [
                min(n_frames - 1, int(round((t / args.fps) * src_fps)))
                for t in range(T_out)
            ]
            
            if not centers:
                print(f"⚠ No centers computed for {stem}, skipping")
                continue
            
            # Process frames in batches
            all_features = []
            batch_frames = []
            batch_sizes = []
            
            for center_idx in centers:
                frame_path = frame_paths[center_idx]
                
                img_bgr = cv2.imread(str(frame_path))
                if img_bgr is None:
                    all_features.append(np.zeros(28, dtype=np.float32))
                    continue
                
                img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
                img_rgb = apply_crop(img_rgb, crop_top, crop_bottom, crop_left, crop_right)
                
                if apply_vflip:
                    img_rgb = np.flipud(img_rgb).copy()
                if apply_hflip:
                    img_rgb = np.fliplr(img_rgb).copy()
                
                h, w = img_rgb.shape[:2]
                batch_frames.append(img_rgb)
                batch_sizes.append((w, h))
                
                if len(batch_frames) >= args.batch_size:
                    results = model(batch_frames, conf=args.conf, verbose=False, 
                                   imgsz=args.imgsz, device=args.device)
                    for res, (img_w, img_h) in zip(results, batch_sizes):
                        features = encode_detections(res.boxes, img_w, img_h)
                        all_features.append(features)
                    batch_frames = []
                    batch_sizes = []
            
            # Process remaining frames
            if batch_frames:
                results = model(batch_frames, conf=args.conf, verbose=False,
                               imgsz=args.imgsz, device=args.device)
                for res, (img_w, img_h) in zip(results, batch_sizes):
                    features = encode_detections(res.boxes, img_w, img_h)
                    all_features.append(features)
        
        if not all_features:
            print(f"⚠ No features extracted for {stem}, skipping")
            continue
        
        # Stack features
        X = np.stack(all_features, axis=0)  # [T, 28]
        np.save(out_npy, X.astype(np.float32))
        
        # Save metadata
        meta = {
            "stem": stem,
            "fps": float(args.fps),
            "src_fps": float(src_fps),
            "n_frames": int(n_frames),
            "T": int(X.shape[0]),
            "D": int(X.shape[1]),
            "yolo_conf": args.conf,
            "yolo_imgsz": args.imgsz,
            "use_video": args.use_video,
            "feature_breakdown": {
                "object_confidence": NUM_CLASSES,
                "key_positions": len(KEY_OBJECTS) * 2,
                "proximity": 4,
            },
            "key_objects": KEY_OBJECTS,
        }
        if apply_crop_to_stem:
            meta["crop"] = {
                "top": crop_top, "bottom": crop_bottom,
                "left": crop_left, "right": crop_right,
            }
        if apply_vflip:
            meta["vertical_flip"] = True
        if apply_hflip:
            meta["horizontal_flip"] = True
            
        out_meta.write_text(json.dumps(meta, indent=2))
        
        indicators = []
        if apply_crop_to_stem:
            indicators.append("cropped")
        if apply_vflip:
            indicators.append("vflip")
        if apply_hflip:
            indicators.append("hflip")
        indicator_str = f" [{'+'.join(indicators)}]" if indicators else ""
        
        print(f"✓ {stem}: X{tuple(X.shape)} saved{indicator_str} → {out_npy}")
    
    print(f"\n✓ YOLO feature extraction complete. Output: {OUT}")


if __name__ == "__main__":
    main()

