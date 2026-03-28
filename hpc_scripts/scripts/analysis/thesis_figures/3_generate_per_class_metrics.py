#!/usr/bin/env python3
"""Generate per-class P/R/F1 table aggregated across folds."""

import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

CLASS_NAMES = ['none', 'Pipette pcr tube', 'Aspirate reagent', 'Attach pipette tip', 'Eject pipette tip']

def compute_metrics_from_cm(cm):
    """Compute per-class precision, recall, F1 from confusion matrix."""
    n_classes = cm.shape[0]
    metrics = []
    
    for i in range(n_classes):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        metrics.append({
            'Class': CLASS_NAMES[i] if i < len(CLASS_NAMES) else f'class_{i}',
            'Precision': precision,
            'Recall': recall,
            'F1': f1,
            'Support': int(cm[i, :].sum())
        })
    
    return metrics

def main():
    exp_dir = Path('/work3/s204157/experiments/mydata_kfold_exp15_cascaded_vjepa')
    output_dir = Path('/zhome/45/8/156251/Desktop/thesis/labassist/hpc_scripts/scripts/analysis/thesis_figures/output')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("\nAggregating confusion matrices across folds...")
    
    # Aggregate confusion matrices across folds
    aggregated_cm = None
    
    for fold in [0, 1, 2, 3, 4, 5]:  # All 6 folds
        # Find best epoch (check eval_sweep first, then root)
        best_epoch_file = exp_dir / f'fold{fold}' / 'eval_sweep' / 'best_epoch.txt'
        if not best_epoch_file.exists():
            best_epoch_file = exp_dir / f'fold{fold}' / 'best_epoch.txt'
        
        if best_epoch_file.exists():
            text = best_epoch_file.read_text().strip()
            # Parse format: "best_epoch=20\nmetric=f1@50\nscore=0.009009"
            best_epoch = None
            for line in text.split('\n'):
                if line.startswith('best_epoch='):
                    best_epoch = int(line.split('=')[1])
                    break
            if best_epoch is None:
                try:
                    best_epoch = int(text)
                except ValueError:
                    print(f"  ✗ Fold {fold}: Could not parse best_epoch.txt")
                    continue
        else:
            print(f"  ✗ Fold {fold}: No best_epoch.txt")
            continue
        
        # Check eval_sweep first, then root
        cm_path = exp_dir / f'fold{fold}' / 'eval_sweep' / f'epoch{best_epoch:03d}' / 'confusion_val.csv'
        if not cm_path.exists():
            cm_path = exp_dir / f'fold{fold}' / f'epoch{best_epoch:03d}' / 'confusion_val.csv'
        
        if cm_path.exists():
            cm = np.loadtxt(cm_path, delimiter=',')
            if aggregated_cm is None:
                aggregated_cm = cm
            else:
                aggregated_cm += cm
            print(f"  ✓ Fold {fold}: Added confusion matrix (epoch {best_epoch})")
        else:
            print(f"  ✗ Fold {fold}: Confusion matrix not found at {cm_path}")
    
    if aggregated_cm is not None:
        metrics = compute_metrics_from_cm(aggregated_cm)
        df = pd.DataFrame(metrics)
        
        # Create bar plot
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # All classes
        ax1 = axes[0]
        x = range(len(df))
        width = 0.25
        ax1.bar([i - width for i in x], df['Precision'], width, label='Precision', alpha=0.8)
        ax1.bar([i for i in x], df['Recall'], width, label='Recall', alpha=0.8)
        ax1.bar([i + width for i in x], df['F1'], width, label='F1', alpha=0.8)
        ax1.set_xlabel('Class')
        ax1.set_ylabel('Score')
        ax1.set_title('Per-Class Metrics (All Classes)')
        ax1.set_xticks(x)
        ax1.set_xticklabels(df['Class'], rotation=30, ha='right')
        ax1.legend()
        ax1.set_ylim([0, 1])
        ax1.grid(axis='y', alpha=0.3)
        
        # Action classes only (exclude none)
        ax2 = axes[1]
        df_actions = df[df['Class'] != 'none'].reset_index(drop=True)
        x = range(len(df_actions))
        ax2.bar([i - width for i in x], df_actions['Precision'], width, label='Precision', alpha=0.8)
        ax2.bar([i for i in x], df_actions['Recall'], width, label='Recall', alpha=0.8)
        ax2.bar([i + width for i in x], df_actions['F1'], width, label='F1', alpha=0.8)
        ax2.set_xlabel('Action Class')
        ax2.set_ylabel('Score')
        ax2.set_title('Per-Class Metrics (Actions Only)')
        ax2.set_xticks(x)
        ax2.set_xticklabels(df_actions['Class'], rotation=30, ha='right')
        ax2.legend()
        ax2.set_ylim([0, 1])
        ax2.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        fig.savefig(output_dir / 'fig3_per_class_metrics.png', dpi=300, bbox_inches='tight')
        plt.close(fig)
        
        # Format table for display
        df_display = df.copy()
        df_display['Precision'] = df_display['Precision'].apply(lambda x: f'{x:.3f}')
        df_display['Recall'] = df_display['Recall'].apply(lambda x: f'{x:.3f}')
        df_display['F1'] = df_display['F1'].apply(lambda x: f'{x:.3f}')
        
        df_display.to_csv(output_dir / 'table3_per_class_metrics.csv', index=False)
        
        print("\n" + "="*80)
        print("TABLE 3: Per-Class Precision, Recall, F1")
        print("="*80)
        print(df_display.to_string(index=False))
        print("\n✓ Saved to:")
        print(f"  - {output_dir / 'table3_per_class_metrics.csv'}")
        print(f"  - {output_dir / 'fig3_per_class_metrics.png'}")
    else:
        print("\nERROR: No confusion matrices found!")

if __name__ == '__main__':
    main()

