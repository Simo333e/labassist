#!/usr/bin/env python3
"""
Visualization script to compare raw frame differences vs masked motion.
Helps tune motion_threshold and morph_kernel parameters.
"""
import argparse
import cv2
import numpy as np
import pathlib
import matplotlib.pyplot as plt
import torch
from torchvision import transforms as T


def apply_motion_mask(diff: np.ndarray, threshold: int = 15, morph_kernel_size: int = 5) -> np.ndarray:
    """Apply morphological masking to frame difference."""
    diff_gray = np.mean(diff, axis=2).astype(np.uint8)
    _, binary_mask = cv2.threshold(diff_gray, threshold, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph_kernel_size, morph_kernel_size))
    mask_cleaned = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel)
    mask_cleaned = cv2.morphologyEx(mask_cleaned, cv2.MORPH_CLOSE, kernel)
    mask_3ch = mask_cleaned[:, :, np.newaxis] / 255.0
    motion_masked = (diff.astype(np.float32) * mask_3ch).astype(np.uint8)
    return motion_masked


def denormalize_for_display(tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    """Denormalize ImageNet-normalized tensor for display."""
    tensor = tensor.clone()
    for t, m, s in zip(tensor, mean, std):
        t.mul_(s).add_(m)
    return torch.clamp(tensor, 0, 1)


def main():
    ap = argparse.ArgumentParser(description="Compare raw vs masked motion features")
    ap.add_argument("--video", required=True, help="Path to video file")
    ap.add_argument("--frame1", type=int, default=100, help="First frame index")
    ap.add_argument("--frame2", type=int, default=105, help="Second frame index")
    ap.add_argument("--threshold", type=int, default=15, help="Motion threshold")
    ap.add_argument("--kernel", type=int, default=5, help="Morph kernel size")
    ap.add_argument("--output", default=None, help="Save comparison image")
    ap.add_argument("--simple", action="store_true", help="Simple 4-panel view (frames + diff + resnet)")
    args = ap.parse_args()
    
    # Read video
    cap = cv2.VideoCapture(args.video)
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if args.frame2 >= n_frames:
        print(f"Frame {args.frame2} exceeds video length {n_frames}")
        return
    
    # Read both frames
    cap.set(cv2.CAP_PROP_POS_FRAMES, args.frame1)
    _, frame1_bgr = cap.read()
    cap.set(cv2.CAP_PROP_POS_FRAMES, args.frame2)
    _, frame2_bgr = cap.read()
    cap.release()
    
    frame1 = cv2.cvtColor(frame1_bgr, cv2.COLOR_BGR2RGB)
    frame2 = cv2.cvtColor(frame2_bgr, cv2.COLOR_BGR2RGB)
    
    # Compute raw difference
    diff_raw = np.abs(frame2.astype(np.float32) - frame1.astype(np.float32))
    
    # Compute masked difference
    diff_masked = apply_motion_mask(
        diff_raw, 
        threshold=args.threshold, 
        morph_kernel_size=args.kernel
    )
    
    # Prepare ResNet preprocessing
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    tfm = T.Compose([
        T.ToTensor(),
        T.Resize((224, 224), antialias=True),
        T.Normalize(mean=mean, std=std),
    ])
    
    # Apply ResNet preprocessing to RGB frames
    frame1_tensor = tfm(frame1)
    frame1_resnet = denormalize_for_display(frame1_tensor, mean, std)
    frame1_resnet_np = frame1_resnet.permute(1, 2, 0).numpy()
    
    frame2_tensor = tfm(frame2)
    frame2_resnet = denormalize_for_display(frame2_tensor, mean, std)
    frame2_resnet_np = frame2_resnet.permute(1, 2, 0).numpy()
    
    # Apply ResNet preprocessing to raw difference
    diff_tensor = tfm(diff_raw.astype(np.uint8))
    diff_resnet = denormalize_for_display(diff_tensor, mean, std)
    diff_resnet_np = diff_resnet.permute(1, 2, 0).numpy()
    
    # Apply ResNet preprocessing to masked difference
    diff_masked_tensor = tfm(diff_masked)
    diff_masked_resnet = denormalize_for_display(diff_masked_tensor, mean, std)
    diff_masked_resnet_np = diff_masked_resnet.permute(1, 2, 0).numpy()
    
    if args.simple:
        # Simple 2x3 view: Show RGB and Motion streams side by side
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # Row 1: RGB Stream
        axes[0, 0].imshow(frame1)
        axes[0, 0].set_title(f'RGB: Frame {args.frame1}', fontsize=14, fontweight='bold')
        axes[0, 0].axis('off')
        
        axes[0, 1].imshow(frame2)
        axes[0, 1].set_title(f'RGB: Frame {args.frame2}', fontsize=14, fontweight='bold')
        axes[0, 1].axis('off')
        
        axes[0, 2].imshow(frame1_resnet_np)
        axes[0, 2].set_title(f'RGB → ResNet Input\n(Frame {args.frame1}, 224×224)', fontsize=13, fontweight='bold')
        axes[0, 2].axis('off')
        
        # Row 2: Motion Stream
        axes[1, 0].imshow(diff_raw.astype(np.uint8))
        axes[1, 0].set_title('Motion: Raw Difference', fontsize=14, fontweight='bold')
        axes[1, 0].axis('off')
        
        axes[1, 1].imshow(diff_masked if args.threshold > 0 else diff_raw.astype(np.uint8))
        title = f'Motion: Masked (t={args.threshold})' if args.threshold > 0 else 'Motion: Unmasked'
        axes[1, 1].set_title(title, fontsize=14, fontweight='bold')
        axes[1, 1].axis('off')
        
        axes[1, 2].imshow(diff_masked_resnet_np if args.threshold > 0 else diff_resnet_np)
        axes[1, 2].set_title('Motion → ResNet Input\n(224×224)', fontsize=13, fontweight='bold')
        axes[1, 2].axis('off')
        
        plt.suptitle('Two-Stream ResNet: RGB Stream (Top) + Motion Stream (Bottom)', 
                     fontsize=16, fontweight='bold', y=0.98)
        plt.tight_layout()
        
    else:
        # Detailed 3x4 view showing full pipeline including RGB preprocessing
        fig, axes = plt.subplots(3, 4, figsize=(20, 15))
        
        # Row 1: RGB Stream
        axes[0, 0].imshow(frame1)
        axes[0, 0].set_title(f'RGB: Frame {args.frame1}', fontsize=12, fontweight='bold')
        axes[0, 0].axis('off')
        
        axes[0, 1].imshow(frame1_resnet_np)
        axes[0, 1].set_title(f'RGB Frame {args.frame1} → ResNet', fontsize=12, fontweight='bold')
        axes[0, 1].axis('off')
        
        axes[0, 2].imshow(frame2)
        axes[0, 2].set_title(f'RGB: Frame {args.frame2}', fontsize=12, fontweight='bold')
        axes[0, 2].axis('off')
        
        axes[0, 3].imshow(frame2_resnet_np)
        axes[0, 3].set_title(f'RGB Frame {args.frame2} → ResNet', fontsize=12, fontweight='bold')
        axes[0, 3].axis('off')
        
        # Row 2: Motion - Raw difference
        axes[1, 0].imshow(diff_raw.astype(np.uint8))
        axes[1, 0].set_title('Motion: Raw Difference', fontsize=12, fontweight='bold')
        axes[1, 0].axis('off')
        
        axes[1, 1].imshow(diff_resnet_np)
        axes[1, 1].set_title('Raw Diff → ResNet Input', fontsize=12, fontweight='bold')
        axes[1, 1].axis('off')
        
        # Show masking intermediate steps
        diff_gray = np.mean(diff_raw, axis=2).astype(np.uint8)
        _, binary_mask = cv2.threshold(diff_gray, args.threshold, 255, cv2.THRESH_BINARY)
        axes[1, 2].imshow(binary_mask, cmap='gray')
        axes[1, 2].set_title(f'Threshold Mask (>{args.threshold})', fontsize=12, fontweight='bold')
        axes[1, 2].axis('off')
        
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (args.kernel, args.kernel))
        mask_cleaned = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel)
        mask_cleaned = cv2.morphologyEx(mask_cleaned, cv2.MORPH_CLOSE, kernel)
        axes[1, 3].imshow(mask_cleaned, cmap='gray')
        axes[1, 3].set_title(f'Cleaned Mask (k={args.kernel})', fontsize=12, fontweight='bold')
        axes[1, 3].axis('off')
        
        # Row 3: Motion - Masked difference
        axes[2, 0].imshow(diff_masked)
        axes[2, 0].set_title('Motion: Masked Difference', fontsize=12, fontweight='bold')
        axes[2, 0].axis('off')
        
        axes[2, 1].imshow(diff_masked_resnet_np)
        axes[2, 1].set_title('Masked Diff → ResNet Input', fontsize=12, fontweight='bold')
        axes[2, 1].axis('off')
        
        # Add text explanation in remaining panels
        axes[2, 2].text(0.5, 0.5, 
                       'Final Features:\n\n'
                       'RGB Stream:\n'
                       '  Frame → ResNet → 512D\n\n'
                       'Motion Stream:\n'
                       '  Diff → Mask → ResNet → 512D\n\n'
                       'Concatenated:\n'
                       '  [RGB_512D, Motion_512D]\n'
                       '  = 1024D → MS-TCN',
                       ha='center', va='center', fontsize=11,
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        axes[2, 2].axis('off')
        
        axes[2, 3].axis('off')
        
        plt.suptitle('Two-Stream ResNet Pipeline: RGB (Row 1) + Motion Raw (Row 2) + Motion Masked (Row 3)', 
                     fontsize=14, fontweight='bold', y=0.995)
        plt.tight_layout()
    
    if args.output:
        plt.savefig(args.output, dpi=150, bbox_inches='tight')
        print(f"Saved comparison to {args.output}")
    else:
        plt.show()
    
    # Print statistics
    raw_mean = np.mean(diff_raw)
    masked_mean = np.mean(diff_masked)
    noise_reduction = (1 - masked_mean / raw_mean) * 100 if raw_mean > 0 else 0
    
    print(f"\nStatistics:")
    print(f"  Raw motion mean: {raw_mean:.2f}")
    print(f"  Masked motion mean: {masked_mean:.2f}")
    print(f"  Noise reduction: {noise_reduction:.1f}%")
    if not args.simple:
        print(f"  Mask coverage: {np.sum(mask_cleaned > 0) / mask_cleaned.size * 100:.1f}%")


if __name__ == "__main__":
    main()

