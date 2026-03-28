#!/usr/bin/env python3
"""Compare all K-fold regularization experiments."""

import pandas as pd
import numpy as np
from pathlib import Path
import sys

experiments = {
    'Exp1 (Baseline)': ('exp1_baseline', 0.25, 0.15, 6),
    'Exp2 (High Dropout)': ('exp2_high_dropout', 0.50, 0.15, 6),
    'Exp3 (Strong Smoothing)': ('exp3_strong_smoothing', 0.25, 0.30, 6),
    'Exp4 (Combined Reg)': ('exp4_combined_reg', 0.50, 0.30, 3),
    'Exp5 (Early Stop)': ('exp5_early_stop', 0.25, 0.15, 6),
}

fold_names = {
    0: '20251027_181439Z',
    1: '20251112_110656Z_1080p30_v2',
    2: '20251120_065442Z (broken pipette)',
    3: '20251125_070904Z',
    4: '20251125_072321Z',
}

results = []

for exp_name, (exp_dir, dropout, smoothing, num_r) in experiments.items():
    f1_scores = []
    best_epochs = []
    
    for fold in range(5):
        sweep_csv = f"eval_data/mydata_kfold/{exp_dir}/fold{fold}/sweep.csv"
        if Path(sweep_csv).exists():
            try:
                df = pd.read_csv(sweep_csv)
                # Find best F1@50
                f1_col = 'segmental.f1_at.50'
                if f1_col in df.columns:
                    best_idx = df[f1_col].idxmax()
                    best_row = df.iloc[best_idx]
                    f1_scores.append(best_row[f1_col])
                    best_epochs.append(int(best_row['epoch']))
                    print(f"  {exp_name} Fold {fold}: F1@50={best_row[f1_col]:.4f} @ epoch {int(best_row['epoch'])}")
            except Exception as e:
                print(f"  {exp_name} Fold {fold}: ERROR - {e}")
        else:
            print(f"  {exp_name} Fold {fold}: MISSING")
    
    if f1_scores:
        results.append({
            'Experiment': exp_name,
            'Dropout': dropout,
            'Smoothing': smoothing,
            'num_R': num_r,
            'Mean F1@50': np.mean(f1_scores),
            'Std F1@50': np.std(f1_scores),
            'Min F1@50': np.min(f1_scores),
            'Max F1@50': np.max(f1_scores),
            'Folds Evaluated': len(f1_scores),
            'Avg Best Epoch': np.mean(best_epochs) if best_epochs else 0,
        })

# Create results DataFrame
df_results = pd.DataFrame(results)
df_results = df_results.sort_values('Mean F1@50', ascending=False)

print("\n" + "="*80)
print("K-FOLD REGULARIZATION EXPERIMENT COMPARISON")
print("="*80)
print(df_results.to_string(index=False, float_format='%.4f'))
print("\n" + "="*80)
print(f"🏆 WINNER: {df_results.iloc[0]['Experiment']}")
print(f"   Mean F1@50: {df_results.iloc[0]['Mean F1@50']:.4f} ± {df_results.iloc[0]['Std F1@50']:.4f}")
print(f"   Settings: dropout={df_results.iloc[0]['Dropout']:.2f}, smoothing={df_results.iloc[0]['Smoothing']:.2f}, num_R={int(df_results.iloc[0]['num_R'])}")
print("="*80)

# Save to CSV
output_csv = "eval_data/mydata_kfold/experiment_comparison.csv"
Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
df_results.to_csv(output_csv, index=False)
print(f"\n✓ Results saved to: {output_csv}")