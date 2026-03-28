#!/usr/bin/env python3
"""Generate fragmentation/stability analysis figure."""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict

CLASS_NAMES = ['none', 'Pipette pcr tube', 'Aspirate reagent', 'Attach pipette tip', 'Eject pipette tip']

def count_segments(arr):
    """Count number of contiguous segments and their lengths."""
    if len(arr) == 0:
        return [], 0
    
    segments = []
    current_class = arr[0]
    segment_start = 0
    
    for i in range(1, len(arr)):
        if arr[i] != current_class:
            segments.append({
                'class': int(current_class),
                'start': segment_start,
                'end': i,
                'length': i - segment_start
            })
            current_class = arr[i]
            segment_start = i
    
    # Last segment
    segments.append({
        'class': int(current_class),
        'start': segment_start,
        'end': len(arr),
        'length': len(arr) - segment_start
    })
    
    return segments, len(segments)

def main():
    # Try to find prediction files
    pred_dirs = [
        # Best model first (cascaded V-JEPA)
        Path('/work3/s204157/experiments/mydata_kfold_exp15_cascaded_vjepa/fold1'),
        # Fallbacks
        Path('/zhome/45/8/156251/Desktop/thesis/labassist/hpc_scripts/eval_data/fsm_test_vjepa/fold1/predictions'),
        Path('/zhome/45/8/156251/Desktop/thesis/labassist/hpc_scripts/eval_data/mydata_kfold_vjepa/exp7_combined_reg/fold1'),
    ]
    
    pred_dir = None
    for d in pred_dirs:
        if d.exists():
            pred_dir = d
            break
    
    if pred_dir is None:
        print("\nERROR: Could not find prediction directory!")
        print("Searched:")
        for d in pred_dirs:
            print(f"  - {d}")
        return
    
    output_dir = Path('/zhome/45/8/156251/Desktop/thesis/labassist/hpc_scripts/scripts/analysis/thesis_figures/output')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\nAnalyzing predictions from: {pred_dir}")
    
    fps = 10.0
    
    # Load predictions and ground truth
    gt_lengths = defaultdict(list)
    pred_lengths = defaultdict(list)
    
    gt_files = list(pred_dir.glob('*_gt.npy'))
    if not gt_files:
        # Try looking for predictions directory
        pred_subdir = pred_dir / 'predictions'
        if pred_subdir.exists():
            gt_files = list(pred_subdir.glob('*_gt.npy'))
            pred_dir = pred_subdir
    
    print(f"  Found {len(gt_files)} ground truth files")
    
    for gt_file in gt_files:
        stem = gt_file.stem.replace('_gt', '')
        pred_file = pred_dir / f'{stem}_pred.npy'
        
        if not pred_file.exists():
            print(f"  ✗ Missing prediction for {stem}")
            continue
        
        gt = np.load(gt_file)
        pred = np.load(pred_file)
        
        gt_segs, _ = count_segments(gt)
        pred_segs, _ = count_segments(pred)
        
        for s in gt_segs:
            if s['class'] > 0:  # Exclude background
                gt_lengths[s['class']].append(s['length'] / fps)
        
        for s in pred_segs:
            if s['class'] > 0:
                pred_lengths[s['class']].append(s['length'] / fps)
        
        print(f"  ✓ Processed {stem}")
    
    if not gt_lengths and not pred_lengths:
        print("\nERROR: No segments found!")
        return
    
    # Plot: Histogram of segment lengths
    n_action_classes = sum(1 for c in range(1, 7) if (c in gt_lengths or c in pred_lengths))
    
    if n_action_classes == 0:
        print("\nERROR: No action class segments found!")
        return
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    plot_idx = 0
    for class_idx in range(1, 7):  # Classes 1-6 (skip none)
        if class_idx not in gt_lengths and class_idx not in pred_lengths:
            continue
        
        if plot_idx >= 6:
            break
        
        ax = axes[plot_idx]
        
        gt_lens = gt_lengths.get(class_idx, [])
        pred_lens = pred_lengths.get(class_idx, [])
        
        bins = np.linspace(0, min(10, max([max(gt_lens, default=0), max(pred_lens, default=0)])), 50)
        
        if gt_lens:
            ax.hist(gt_lens, bins=bins, alpha=0.5, label='Ground Truth', color='green', edgecolor='black', linewidth=0.5)
        if pred_lens:
            ax.hist(pred_lens, bins=bins, alpha=0.5, label='Predicted', color='red', edgecolor='black', linewidth=0.5)
        
        # Highlight short segments (< 0.5s)
        short_pred = sum(1 for l in pred_lens if l < 0.5)
        total_pred = len(pred_lens)
        pct_short = 100 * short_pred / total_pred if total_pred > 0 else 0
        
        ax.axvline(0.5, color='black', linestyle='--', alpha=0.7, linewidth=2, label='0.5s threshold')
        ax.set_xlabel('Segment Duration (seconds)', fontsize=10)
        ax.set_ylabel('Count', fontsize=10)
        
        class_name = CLASS_NAMES[class_idx] if class_idx < len(CLASS_NAMES) else f'class_{class_idx}'
        ax.set_title(f'{class_name}\n({pct_short:.1f}% predicted < 0.5s)', fontsize=11, fontweight='bold')
        ax.legend(fontsize=8)
        ax.grid(axis='y', alpha=0.3)
        
        plot_idx += 1
    
    # Hide unused subplots
    for idx in range(plot_idx, 6):
        axes[idx].axis('off')
    
    plt.suptitle('Temporal Stability Analysis: Segment Duration Distribution\n' +
                 'Short predicted segments (< 0.5s) often trigger false alerts',
                 fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    output_path = output_dir / 'fig6_fragmentation.png'
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    fig.savefig(output_path.with_suffix('.pdf'), bbox_inches='tight')
    plt.close(fig)
    
    print(f"\n✓ Fragmentation figure saved to:")
    print(f"  - {output_path}")
    
    # Print summary statistics
    print("\n" + "="*80)
    print("Fragmentation Summary:")
    print("="*80)
    for class_idx in range(1, 7):
        if class_idx in pred_lengths:
            pred_lens = pred_lengths[class_idx]
            short = sum(1 for l in pred_lens if l < 0.5)
            total = len(pred_lens)
            pct = 100 * short / total if total > 0 else 0
            class_name = CLASS_NAMES[class_idx] if class_idx < len(CLASS_NAMES) else f'class_{class_idx}'
            print(f"  {class_name:20s}: {short:4d}/{total:4d} segments < 0.5s ({pct:5.1f}%)")

if __name__ == '__main__':
    main()

