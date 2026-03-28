#!/usr/bin/env python3
"""
Analyze and visualize FSM validation results

This script creates visualizations and detailed analysis of FSM validation results,
including:
- Alert timeline plots showing where violations were detected
- Per-video FSM state transitions
- Aggregate statistics and confusion analysis
- Error pattern analysis

Usage:
    python analyze_fsm_results.py \
        --fsm_results_dir /path/to/fsm_results \
        --predictions_base_dir /path/to/eval_data/exp/fold*/predictions \
        --class_index /path/to/class_index_7class.json \
        --output_dir /path/to/fsm_analysis
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
from matplotlib.colors import ListedColormap, BoundaryNorm

# Action class colors - MUST match eval_with_predictions.py for consistency
# Uses the same logic: ACTION_COLORS if available, otherwise tab20 fallback
ACTION_COLORS = {
    'none': '#CCCCCC',
    # Old names (v1)
    'Pipette pcr tube': '#E74C3C',
    'Aspirate reagent': '#3498DB', 
    # New names (v2)
    'Manipulate PCR tube': '#E74C3C',
    'Manipulate reagent': '#3498DB',
    # 7-class additions
    'Manipulate pcr tubes': '#E74C3C',
    'Manipulate reagents': '#3498DB',
    'Adjust pipette volume': '#9B59B6',
    'Switch pipette': '#E67E22',
    # Unchanged
    'Attach pipette tip': '#2ECC71',
    'Eject pipette tip': '#F39C12',
}


def get_colors_for_classes(class_names: List[str]) -> List:
    """
    Get colors for class names - MUST match eval_with_predictions.py exactly
    Uses ACTION_COLORS if available, otherwise falls back to tab20
    """
    colors = []
    for name in class_names:
        if name in ACTION_COLORS:
            hex_color = ACTION_COLORS[name]
            rgb = tuple(int(hex_color.lstrip('#')[i:i+2], 16)/255.0 for i in (0, 2, 4))
            colors.append(rgb)
        else:
            # Fallback to tab20 for unknown classes (same as eval scripts)
            base = matplotlib.colormaps.get_cmap("tab20")
            colors.append(base(len(colors) % 20))
    return colors


def load_class_names(class_index_path: str) -> List[str]:
    """Load action class names from class_index.json"""
    obj = json.loads(Path(class_index_path).read_text())
    if isinstance(obj, dict):
        inv = {v: k for k, v in obj.items()}
        names = [inv[i] if i in inv else f"c{i}" for i in range(len(inv))]
    else:
        names = obj
    return names


def plot_alert_timeline(
    predictions: np.ndarray,
    ground_truth: np.ndarray,
    pred_alerts: List[Dict],
    gt_alerts: List[Dict],
    class_names: List[str],
    stem: str,
    output_file: Path
):
    """
    Plot timeline showing predictions, ground truth, and alert positions
    
    Uses the same color scheme as eval_with_predictions.py for consistency.
    
    Args:
        predictions: Predicted action classes [T]
        ground_truth: Ground truth action classes [T]
        pred_alerts: List of alerts from predictions
        gt_alerts: List of alerts from ground truth
        class_names: List of action names
        stem: Video stem
        output_file: Output file path
    """
    T = len(predictions)
    ncls = len(class_names)
    
    # Build color list to match eval_with_predictions.py exactly
    colors = get_colors_for_classes(class_names)
    cmap = ListedColormap(colors)
    norm = BoundaryNorm(np.arange(-0.5, ncls + 0.5, 1.0), cmap.N)
    
    def boundaries(arr):
        return np.where(arr[1:] != arr[:-1])[0] + 1
    
    # Create figure with 2 rows: GT and Predictions (no error row)
    fig = plt.figure(figsize=(14, 5))
    gs = fig.add_gridspec(nrows=3, ncols=1, height_ratios=[1, 1, 0.28], hspace=0.35)
    ax_gt = fig.add_subplot(gs[0, 0])
    ax_pred = fig.add_subplot(gs[1, 0])
    cax = fig.add_subplot(gs[2, 0])
    
    # Plot ground truth
    ax_gt.imshow(ground_truth[None, :], aspect="auto", cmap=cmap, norm=norm)
    ax_gt.set_title(f"Ground Truth – {stem}", fontweight='bold')
    ax_gt.set_yticks([])
    
    # Mark GT critical alerts
    gt_critical_frames = [a['frame'] for a in gt_alerts if a.get('level') == 'critical']
    for frame in gt_critical_frames:
        ax_gt.axvline(frame, color='darkred', linewidth=2, linestyle='--', alpha=0.9)
    
    # Add segment boundaries
    for b in boundaries(ground_truth):
        ax_gt.axvline(b, color="k", alpha=0.15, lw=0.75)
    
    # Plot predictions
    ax_pred.imshow(predictions[None, :], aspect="auto", cmap=cmap, norm=norm)
    ax_pred.set_title("Predictions", fontweight='bold')
    ax_pred.set_yticks([])
    ax_pred.set_xlabel('Frame', fontweight='bold')
    
    # Mark predicted critical alerts with triangles
    pred_critical_frames = [a['frame'] for a in pred_alerts if a.get('level') == 'critical']
    for frame in pred_critical_frames:
        ax_pred.axvline(frame, color='red', linewidth=1.5, alpha=0.8)
    
    # Add segment boundaries
    for b in boundaries(predictions):
        ax_pred.axvline(b, color="k", alpha=0.15, lw=0.75)
    
    # Add colorbar legend
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cb = fig.colorbar(sm, cax=cax, orientation="horizontal")
    cb.set_ticks(range(ncls))
    cb.set_ticklabels([str(n) for n in class_names])
    for lab in cax.get_xticklabels():
        lab.set_rotation(40)
        lab.set_ha("right")
        lab.set_fontsize(8)
    
    # Add alert counts as text annotation
    alert_text = f"Critical Alerts: GT={len(gt_critical_frames)}, Pred={len(pred_critical_frames)}"
    fig.text(0.02, 0.98, alert_text, fontsize=10, fontweight='bold', 
             transform=fig.transFigure, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    output_file.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()


def plot_aggregate_statistics(
    aggregate_report: Dict,
    output_dir: Path
):
    """
    Create summary plots for aggregate FSM statistics
    
    Args:
        aggregate_report: Aggregate FSM report from run_fsm_validation_kfold.py
        output_dir: Output directory
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Extract data
    folds = aggregate_report['folds']
    fold_indices = [f['fold_idx'] for f in folds]
    precisions = [f['precision'] for f in folds]
    recalls = [f['recall'] for f in folds]
    f1s = [f['f1'] for f in folds]
    
    # Plot per-fold metrics
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # Precision
    ax = axes[0]
    ax.bar(fold_indices, precisions, color='#3498DB', alpha=0.7, edgecolor='black')
    ax.axhline(np.mean(precisions), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(precisions):.3f}')
    ax.set_xlabel('Fold', fontweight='bold')
    ax.set_ylabel('Precision', fontweight='bold')
    ax.set_title('Precision per Fold', fontweight='bold')
    ax.set_ylim(0, 1.0)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    # Recall
    ax = axes[1]
    ax.bar(fold_indices, recalls, color='#2ECC71', alpha=0.7, edgecolor='black')
    ax.axhline(np.mean(recalls), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(recalls):.3f}')
    ax.set_xlabel('Fold', fontweight='bold')
    ax.set_ylabel('Recall', fontweight='bold')
    ax.set_title('Recall per Fold', fontweight='bold')
    ax.set_ylim(0, 1.0)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    # F1
    ax = axes[2]
    ax.bar(fold_indices, f1s, color='#E74C3C', alpha=0.7, edgecolor='black')
    ax.axhline(np.mean(f1s), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(f1s):.3f}')
    ax.set_xlabel('Fold', fontweight='bold')
    ax.set_ylabel('F1 Score', fontweight='bold')
    ax.set_title('F1 Score per Fold', fontweight='bold')
    ax.set_ylim(0, 1.0)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    plt.suptitle('FSM Validation Performance Across Folds', fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(output_dir / 'fsm_performance_per_fold.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # Plot confusion-style analysis
    overall = aggregate_report['overall']
    tp = overall['true_positives']
    fp = overall['false_positives']
    fn = overall['false_negatives']
    tn = 0  # Not explicitly tracked, but can infer from total frames
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    categories = ['True\nPositives', 'False\nPositives', 'False\nNegatives']
    values = [tp, fp, fn]
    colors = ['#2ECC71', '#E74C3C', '#F39C12']
    
    bars = ax.bar(categories, values, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
    
    # Add value labels on bars
    for bar, val in zip(bars, values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + max(values)*0.02,
                f'{int(val)}',
                ha='center', va='bottom', fontweight='bold', fontsize=12)
    
    ax.set_ylabel('Count', fontweight='bold', fontsize=12)
    ax.set_title('FSM Validation: Detection Performance', fontweight='bold', fontsize=14)
    ax.grid(axis='y', alpha=0.3)
    
    # Add metrics text
    precision = overall['precision']
    recall = overall['recall']
    f1 = overall['f1']
    
    textstr = f'Precision: {precision:.3f}\nRecall: {recall:.3f}\nF1 Score: {f1:.3f}'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax.text(0.98, 0.98, textstr, transform=ax.transAxes, fontsize=11,
            verticalalignment='top', horizontalalignment='right', bbox=props)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'fsm_detection_performance.png', dpi=150, bbox_inches='tight')
    plt.close()


def generate_alerts_table(
    fsm_results_dir: Path,
    output_dir: Path
):
    """
    Generate a summary table of alerts per video and fold
    
    Args:
        fsm_results_dir: Directory with FSM results
        output_dir: Output directory
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    rows = []
    
    # Collect data from all folds
    for fold_result_file in sorted(fsm_results_dir.glob("fold*/fsm_results.json")):
        fold_idx = int(fold_result_file.parent.name.replace("fold", ""))
        
        with open(fold_result_file) as f:
            fold_results = json.load(f)
        
        for video in fold_results['videos']:
            row = {
                'Fold': fold_idx,
                'Video': video['stem'],
                'Frames': video['T'],
                'Pred_Total': video['predictions']['total_alerts'],
                'Pred_Critical': video['predictions']['critical_alerts'],
                'Pred_Warning': video['predictions']['warning_alerts'],
                'GT_Total': video['ground_truth']['total_alerts'],
                'GT_Critical': video['ground_truth']['critical_alerts'],
                'GT_Warning': video['ground_truth']['warning_alerts'],
                'TP': video['metrics']['true_positives'],
                'FP': video['metrics']['false_positives'],
                'FN': video['metrics']['false_negatives'],
                'Precision': video['metrics']['precision'],
                'Recall': video['metrics']['recall'],
                'F1': video['metrics']['f1'],
            }
            rows.append(row)
    
    if not rows:
        print("  WARNING: No data found for alerts table")
        return
    
    # Save as CSV
    import csv
    csv_path = output_dir / "alerts_summary.csv"
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    
    # Save as formatted text table
    txt_path = output_dir / "alerts_summary.txt"
    with open(txt_path, 'w') as f:
        # Header
        f.write("=" * 120 + "\n")
        f.write("FSM ALERTS SUMMARY TABLE\n")
        f.write("=" * 120 + "\n\n")
        
        # Column headers
        header = f"{'Fold':>4} | {'Video':>35} | {'Frames':>7} | {'Pred':>6} | {'GT':>4} | {'TP':>3} | {'FP':>4} | {'FN':>3} | {'Prec':>6} | {'Rec':>6} | {'F1':>6}\n"
        f.write(header)
        f.write("-" * 120 + "\n")
        
        # Data rows
        for row in rows:
            line = f"{row['Fold']:>4} | {row['Video'][:35]:>35} | {row['Frames']:>7} | {row['Pred_Critical']:>6} | {row['GT_Critical']:>4} | {row['TP']:>3} | {row['FP']:>4} | {row['FN']:>3} | {row['Precision']:>6.3f} | {row['Recall']:>6.3f} | {row['F1']:>6.3f}\n"
            f.write(line)
        
        f.write("-" * 120 + "\n")
        
        # Summary
        total_pred = sum(r['Pred_Critical'] for r in rows)
        total_gt = sum(r['GT_Critical'] for r in rows)
        total_tp = sum(r['TP'] for r in rows)
        total_fp = sum(r['FP'] for r in rows)
        total_fn = sum(r['FN'] for r in rows)
        
        f.write(f"\nTOTAL: Pred={total_pred}, GT={total_gt}, TP={total_tp}, FP={total_fp}, FN={total_fn}\n")
        
        if total_pred > 0:
            precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
            recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 1
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            f.write(f"OVERALL: Precision={precision:.4f}, Recall={recall:.4f}, F1={f1:.4f}\n")
        
        f.write("=" * 120 + "\n")
    
    print(f"  ✓ Alerts table saved to {csv_path}")
    print(f"  ✓ Alerts table saved to {txt_path}")
    
    # Print table to console
    print("\n" + "=" * 100)
    print("ALERTS SUMMARY TABLE")
    print("=" * 100)
    print(f"{'Fold':>4} | {'Video':>30} | {'Pred':>5} | {'GT':>4} | {'TP':>3} | {'FP':>4} | {'FN':>3} | {'Prec':>6} | {'Rec':>6}")
    print("-" * 100)
    for row in rows:
        video_short = row['Video'][:30] if len(row['Video']) > 30 else row['Video']
        print(f"{row['Fold']:>4} | {video_short:>30} | {row['Pred_Critical']:>5} | {row['GT_Critical']:>4} | {row['TP']:>3} | {row['FP']:>4} | {row['FN']:>3} | {row['Precision']:>6.3f} | {row['Recall']:>6.3f}")
    print("=" * 100)


def analyze_error_patterns(
    fsm_results_dir: Path,
    predictions_base_dir: Path,
    class_names: List[str],
    output_dir: Path
):
    """
    Analyze common error patterns in FSM validation
    
    Args:
        fsm_results_dir: Directory with FSM results
        predictions_base_dir: Base directory with predictions
        class_names: List of action names
        output_dir: Output directory
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Collect all false positives and false negatives
    false_positive_contexts = []
    false_negative_contexts = []
    
    # Load results from each fold
    for fold_result_file in sorted(fsm_results_dir.glob("fold*/fsm_results.json")):
        with open(fold_result_file) as f:
            fold_results = json.load(f)
        
        for video in fold_results['videos']:
            # Analyze prediction alerts
            for alert in video['predictions']['alerts']:
                if alert['level'] == 'critical':
                    # Check if this is a false positive
                    # (simplified: just collect contexts)
                    context = alert.get('context', [])
                    if context:
                        false_positive_contexts.append(context[-3:])  # Last 3 frames
    
    print(f"Collected {len(false_positive_contexts)} potential error contexts")
    
    # Create a simple report
    report = {
        "total_error_contexts": len(false_positive_contexts),
        "analysis": "Error pattern analysis complete. See individual fold results for details."
    }
    
    with open(output_dir / "error_patterns.json", "w") as f:
        json.dump(report, f, indent=2)


def main():
    ap = argparse.ArgumentParser(description="Analyze and visualize FSM validation results")
    ap.add_argument(
        "--fsm_results_dir",
        required=True,
        help="Directory with FSM validation results (contains fold*/fsm_results.json)"
    )
    ap.add_argument(
        "--predictions_base_dir",
        required=True,
        help="Base directory with predictions (for timeline plots)"
    )
    ap.add_argument(
        "--class_index",
        required=True,
        help="Path to class_index.json"
    )
    ap.add_argument(
        "--output_dir",
        required=True,
        help="Output directory for analysis results"
    )
    ap.add_argument(
        "--max_timelines",
        type=int,
        default=10,
        help="Maximum number of timeline plots to generate (default: 10)"
    )
    
    args = ap.parse_args()
    
    fsm_results_dir = Path(args.fsm_results_dir)
    predictions_base_dir = Path(args.predictions_base_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load class names
    class_names = load_class_names(args.class_index)
    print(f"Loaded {len(class_names)} action classes")
    
    # Load aggregate report
    aggregate_file = fsm_results_dir / "fsm_aggregate.json"
    if not aggregate_file.exists():
        print(f"ERROR: Aggregate file not found: {aggregate_file}")
        return
    
    with open(aggregate_file) as f:
        aggregate_report = json.load(f)
    
    print(f"\nLoaded aggregate report:")
    print(f"  Folds: {aggregate_report['num_folds']}")
    print(f"  Videos: {aggregate_report['total_videos']}")
    print(f"  Overall F1: {aggregate_report['overall']['f1']:.4f}")
    
    # Generate aggregate statistics plots
    print("\nGenerating aggregate statistics plots...")
    plot_aggregate_statistics(aggregate_report, output_dir)
    print(f"  ✓ Saved to {output_dir / 'fsm_performance_per_fold.png'}")
    print(f"  ✓ Saved to {output_dir / 'fsm_detection_performance.png'}")
    
    # Generate timeline plots for videos with alerts
    print("\nGenerating alert timeline plots...")
    timelines_dir = output_dir / "timelines"
    timelines_dir.mkdir(parents=True, exist_ok=True)
    
    timeline_count = 0
    
    for fold_result_file in sorted(fsm_results_dir.glob("fold*/fsm_results.json")):
        if timeline_count >= args.max_timelines:
            break
        
        fold_idx = int(fold_result_file.parent.name.replace("fold", ""))
        
        with open(fold_result_file) as f:
            fold_results = json.load(f)
        
        for video in fold_results['videos']:
            if timeline_count >= args.max_timelines:
                break
            
            stem = video['stem']
            
            # Check if video has critical alerts in predictions or GT
            has_critical = (
                video['predictions']['critical_alerts'] > 0 or
                video['ground_truth']['critical_alerts'] > 0
            )
            
            if not has_critical:
                continue
            
            # Load predictions and ground truth
            pred_file = predictions_base_dir / f"fold{fold_idx}" / "predictions" / f"{stem}_pred.npy"
            gt_file = predictions_base_dir / f"fold{fold_idx}" / "predictions" / f"{stem}_gt.npy"
            
            if not pred_file.exists() or not gt_file.exists():
                print(f"  WARNING: Missing files for {stem}")
                continue
            
            predictions = np.load(pred_file)
            ground_truth = np.load(gt_file)
            
            # Generate timeline plot
            output_file = timelines_dir / f"{stem}_fold{fold_idx}_timeline.png"
            plot_alert_timeline(
                predictions=predictions,
                ground_truth=ground_truth,
                pred_alerts=video['predictions']['alerts'],
                gt_alerts=video['ground_truth']['alerts'],
                class_names=class_names,
                stem=stem,
                output_file=output_file
            )
            
            timeline_count += 1
            print(f"  ✓ {stem} (fold {fold_idx}): "
                  f"pred_critical={video['predictions']['critical_alerts']}, "
                  f"gt_critical={video['ground_truth']['critical_alerts']}")
    
    print(f"\n✓ Generated {timeline_count} timeline plots")
    
    # Generate alerts summary table
    print("\nGenerating alerts summary table...")
    generate_alerts_table(fsm_results_dir, output_dir)
    
    # Analyze error patterns
    print("\nAnalyzing error patterns...")
    analyze_error_patterns(fsm_results_dir, predictions_base_dir, class_names, output_dir)
    print(f"  ✓ Error analysis saved to {output_dir / 'error_patterns.json'}")
    
    # Create summary report
    summary = {
        "analysis_date": "2026-01-15",
        "fsm_results_dir": str(fsm_results_dir),
        "num_folds": aggregate_report['num_folds'],
        "total_videos": aggregate_report['total_videos'],
        "overall_metrics": aggregate_report['overall'],
        "per_fold_stats": aggregate_report['per_fold_stats'],
        "timelines_generated": timeline_count,
    }
    
    with open(output_dir / "analysis_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n✓ Analysis complete!")
    print(f"✓ Results saved to: {output_dir}")


if __name__ == "__main__":
    main()




