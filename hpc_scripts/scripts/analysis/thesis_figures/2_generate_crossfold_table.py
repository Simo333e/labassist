#!/usr/bin/env python3
"""Generate cross-fold performance table with mean ± std."""

import json
import numpy as np
import pandas as pd
from pathlib import Path

FOLD_NAMES = {
    0: '20251027_181439Z',
    1: '20251112_110656Z_1080p30_v2',
    2: '20251120_065442Z',
    3: '20251125_070904Z',
    4: '20251125_072321Z',
    5: 'Lab_video_cut',
}

def load_best_epoch_results(exp_dir: Path, fold: int) -> dict:
    """Load results from the best epoch for a fold."""
    fold_dir = exp_dir / f'fold{fold}'
    
    if not fold_dir.exists():
        return None
    
    # Read best epoch (check eval_sweep first, then root)
    best_epoch_file = fold_dir / 'eval_sweep' / 'best_epoch.txt'
    if not best_epoch_file.exists():
        best_epoch_file = fold_dir / 'best_epoch.txt'
    
    if best_epoch_file.exists():
        text = best_epoch_file.read_text().strip()
        # Parse format: "best_epoch=20\nmetric=f1@50\nscore=0.009009"
        for line in text.split('\n'):
            if line.startswith('best_epoch='):
                best_epoch = int(line.split('=')[1])
                break
        else:
            # Try parsing as plain integer
            try:
                best_epoch = int(text)
            except ValueError:
                best_epoch = None
        
        if best_epoch is None:
            return None
    else:
        # Find from metrics CSV
        metrics_csv = fold_dir / 'metrics_vs_epoch.csv'
        if metrics_csv.exists():
            df = pd.read_csv(metrics_csv)
            if 'F1@50' in df.columns:
                best_epoch = int(df.loc[df['F1@50'].idxmax(), 'epoch'])
            else:
                return None
        else:
            return None
    
    # Load summary.json for that epoch (check eval_sweep first, then root)
    summary_path = fold_dir / 'eval_sweep' / f'epoch{best_epoch:03d}' / 'summary.json'
    if not summary_path.exists():
        summary_path = fold_dir / f'epoch{best_epoch:03d}' / 'summary.json'
    
    if not summary_path.exists():
        print(f"Warning: Summary not found at {summary_path}")
        return None
    
    with open(summary_path) as f:
        data = json.load(f)
    
    return {
        'fold': fold,
        'recording': FOLD_NAMES.get(fold, f'fold{fold}'),
        'epoch': best_epoch,
        'micro_acc': data.get('micro_acc', 0),
        'macroF1_all': data.get('macroF1_all', 0),
        'macroF1_no_bg': data.get('macroF1_no_bg', 0),
        'edit_score': data.get('edit_score_mean', 0),
        'F1@10': data.get('segmental', {}).get('f1_at', {}).get('10', 0),
        'F1@25': data.get('segmental', {}).get('f1_at', {}).get('25', 0),
        'F1@50': data.get('segmental', {}).get('f1_at', {}).get('50', 0),
    }

def main():
    exp_dir = Path('/work3/s204157/experiments/mydata_kfold_exp15_cascaded_vjepa')
    output_dir = Path('/zhome/45/8/156251/Desktop/thesis/labassist/hpc_scripts/scripts/analysis/thesis_figures/output')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("\nLoading cross-fold results...")
    results = []
    for fold in [0, 1, 2, 3, 4, 5]:  # All 6 folds with cascaded V-JEPA
        r = load_best_epoch_results(exp_dir, fold)
        if r:
            results.append(r)
            print(f"  ✓ Fold {fold}: Loaded (epoch {r['epoch']})")
        else:
            print(f"  ✗ Fold {fold}: Not found")
    
    if not results:
        print("ERROR: No results found!")
        return
    
    df = pd.DataFrame(results)
    
    # Add mean ± std row
    metrics = ['micro_acc', 'macroF1_all', 'macroF1_no_bg', 'edit_score', 'F1@10', 'F1@25', 'F1@50']
    mean_row = {'recording': 'Mean', 'fold': '-', 'epoch': '-'}
    std_row = {'recording': 'Std', 'fold': '-', 'epoch': '-'}
    
    for m in metrics:
        mean_row[m] = df[m].mean()
        std_row[m] = df[m].std()
    
    df = pd.concat([df, pd.DataFrame([mean_row, std_row])], ignore_index=True)
    
    # Format for display
    df_display = df.copy()
    for m in metrics:
        df_display[m] = df_display[m].apply(lambda x: f'{x:.4f}' if isinstance(x, float) else x)
    
    df_display.to_csv(output_dir / 'table2_crossfold_performance.csv', index=False)
    
    print("\n" + "="*120)
    print("TABLE 2: Cross-Fold Performance (Leave-One-Out)")
    print("="*120)
    print(df_display.to_string(index=False))
    print("\n✓ Saved to:")
    print(f"  - {output_dir / 'table2_crossfold_performance.csv'}")

if __name__ == '__main__':
    main()

