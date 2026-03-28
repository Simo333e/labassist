#!/usr/bin/env python3
"""Generate confusion matrix figures."""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

CLASS_NAMES = ['none', 'Pipette\npcr tube', 'Aspirate\nreagent', 'Attach\npipette tip', 'Eject\npipette tip']
CLASS_NAMES_SHORT = ['none', 'PCR tube', 'Reagent', 'Tip on', 'Tip off']

def plot_confusion_matrix(cm, class_names, output_path, title='Confusion Matrix', normalize_by_row=True):
    """Plot confusion matrix as heatmap."""
    if normalize_by_row:
        cm_norm = cm / cm.sum(axis=1, keepdims=True) * 100
        cm_norm = np.nan_to_num(cm_norm)
        fmt = '.1f'
        cbar_label = '% of Ground Truth'
    else:
        cm_norm = cm.astype(int)  # Convert to int for count display
        fmt = 'd'
        cbar_label = 'Count'
    
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(cm_norm, annot=True, fmt=fmt, cmap='Blues',
                xticklabels=class_names, yticklabels=class_names,
                ax=ax, cbar_kws={'label': cbar_label}, vmin=0, vmax=100 if normalize_by_row else None)
    ax.set_xlabel('Predicted Class', fontsize=12)
    ax.set_ylabel('Ground Truth Class', fontsize=12)
    ax.set_title(title, fontsize=14)
    
    # Highlight diagonal
    for i in range(len(class_names)):
        ax.add_patch(plt.Rectangle((i, i), 1, 1, fill=False, edgecolor='red', lw=2))
    
    plt.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    fig.savefig(output_path.with_suffix('.pdf'), bbox_inches='tight')
    plt.close(fig)
    print(f"  ✓ Saved: {output_path.name}")

def main():
    exp_dir = Path('/work3/s204157/experiments/mydata_kfold_exp15_cascaded_vjepa')
    output_dir = Path('/zhome/45/8/156251/Desktop/thesis/labassist/hpc_scripts/scripts/analysis/thesis_figures/output')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("\nAggregating confusion matrices...")
    
    # Aggregate confusion matrices
    aggregated_cm = None
    for fold in [0, 1, 2, 3, 4, 5]:  # All 6 folds
        # Check eval_sweep first, then root
        best_epoch_file = exp_dir / f'fold{fold}' / 'eval_sweep' / 'best_epoch.txt'
        if not best_epoch_file.exists():
            best_epoch_file = exp_dir / f'fold{fold}' / 'best_epoch.txt'
        
        if best_epoch_file.exists():
            text = best_epoch_file.read_text().strip()
            best_epoch = None
            for line in text.split('\n'):
                if line.startswith('best_epoch='):
                    best_epoch = int(line.split('=')[1])
                    break
            if best_epoch is None:
                try:
                    best_epoch = int(text)
                except ValueError:
                    continue
            
            # Check eval_sweep first, then root
            cm_path = exp_dir / f'fold{fold}' / 'eval_sweep' / f'epoch{best_epoch:03d}' / 'confusion_val.csv'
            if not cm_path.exists():
                cm_path = exp_dir / f'fold{fold}' / f'epoch{best_epoch:03d}' / 'confusion_val.csv'
            
            if cm_path.exists():
                cm = np.loadtxt(cm_path, delimiter=',')
                aggregated_cm = cm if aggregated_cm is None else aggregated_cm + cm
                print(f"  ✓ Fold {fold}: Added (epoch {best_epoch})")
    
    if aggregated_cm is not None:
        print("\nGenerating confusion matrix figures...")
        
        # Full confusion matrix (all classes)
        plot_confusion_matrix(
            aggregated_cm, CLASS_NAMES_SHORT,
            output_dir / 'fig4a_confusion_full.png',
            'Confusion Matrix (All Classes, Row-Normalized)',
            normalize_by_row=True
        )
        
        # Action-only (exclude 'none')
        cm_action = aggregated_cm[1:, 1:]
        plot_confusion_matrix(
            cm_action, CLASS_NAMES_SHORT[1:],
            output_dir / 'fig4b_confusion_actions.png',
            'Confusion Matrix (Action Classes Only, Row-Normalized)',
            normalize_by_row=True
        )
        
        # Also save raw counts version
        plot_confusion_matrix(
            aggregated_cm, CLASS_NAMES_SHORT,
            output_dir / 'fig4c_confusion_counts.png',
            'Confusion Matrix (Raw Counts)',
            normalize_by_row=False
        )
        
        print("\n✓ Confusion matrix figures saved to:")
        print(f"  - {output_dir}")
        
        # Print top confusions
        print("\n" + "="*80)
        print("Top Confusions (excluding diagonal):")
        print("="*80)
        cm_norm = aggregated_cm / aggregated_cm.sum(axis=1, keepdims=True) * 100
        cm_norm = np.nan_to_num(cm_norm)
        
        confusions = []
        for i in range(len(cm_norm)):
            for j in range(len(cm_norm)):
                if i != j:  # Exclude diagonal
                    confusions.append((i, j, cm_norm[i, j]))
        
        confusions.sort(key=lambda x: x[2], reverse=True)
        
        for i, (gt, pred, pct) in enumerate(confusions[:10]):
            gt_name = CLASS_NAMES_SHORT[gt].replace('\n', ' ')
            pred_name = CLASS_NAMES_SHORT[pred].replace('\n', ' ')
            print(f"{i+1:2d}. GT: {gt_name:15s} → Pred: {pred_name:15s} = {pct:5.1f}%")
    else:
        print("\nERROR: No confusion matrices found!")

if __name__ == '__main__':
    main()

