#!/usr/bin/env python3
"""
Compare masked vs unmasked ResNet+Motion cascade training results.
Generates comparison tables and plots for thesis.
"""
import argparse
import json
import pathlib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


def load_fold_results(exp_dir: pathlib.Path, num_folds: int = 5):
    """Load results from all folds."""
    results = []
    for fold in range(num_folds):
        fold_dir = exp_dir / f"fold{fold}"
        metrics_file = fold_dir / "metrics_summary.json"
        
        if not metrics_file.exists():
            print(f"Warning: {metrics_file} not found")
            continue
        
        with open(metrics_file) as f:
            metrics = json.load(f)
        
        # Find best epoch based on F1@50
        best_epoch = max(metrics.items(), key=lambda x: x[1].get("val_f1_50", 0))
        epoch_num, epoch_metrics = best_epoch
        
        results.append({
            "fold": fold,
            "epoch": epoch_num,
            "f1_10": epoch_metrics.get("val_f1_10", 0),
            "f1_25": epoch_metrics.get("val_f1_25", 0),
            "f1_50": epoch_metrics.get("val_f1_50", 0),
            "edit": epoch_metrics.get("val_edit", 0),
            "acc": epoch_metrics.get("val_acc", 0),
        })
    
    return pd.DataFrame(results)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--masked_dir", required=True, 
                    help="Directory with masked motion results")
    ap.add_argument("--unmasked_dir", required=True,
                    help="Directory with unmasked motion results")
    ap.add_argument("--output", default="masked_vs_unmasked_comparison.png",
                    help="Output plot filename")
    args = ap.parse_args()
    
    masked_dir = pathlib.Path(args.masked_dir)
    unmasked_dir = pathlib.Path(args.unmasked_dir)
    
    print("Loading results...")
    masked_df = load_fold_results(masked_dir)
    unmasked_df = load_fold_results(unmasked_dir)
    
    if masked_df.empty or unmasked_df.empty:
        print("ERROR: No results found in one or both directories")
        return
    
    # Add method column
    masked_df["method"] = "Masked"
    unmasked_df["method"] = "Unmasked"
    
    # Combine
    combined_df = pd.concat([masked_df, unmasked_df], ignore_index=True)
    
    # Print summary statistics
    print("\n" + "="*60)
    print("COMPARISON: Masked vs Unmasked Motion Features")
    print("="*60)
    
    for method in ["Masked", "Unmasked"]:
        method_df = combined_df[combined_df["method"] == method]
        print(f"\n{method} Results:")
        print(f"  F1@10: {method_df['f1_10'].mean():.3f} ± {method_df['f1_10'].std():.3f}")
        print(f"  F1@25: {method_df['f1_25'].mean():.3f} ± {method_df['f1_25'].std():.3f}")
        print(f"  F1@50: {method_df['f1_50'].mean():.3f} ± {method_df['f1_50'].std():.3f}")
        print(f"  Edit:  {method_df['edit'].mean():.3f} ± {method_df['edit'].std():.3f}")
        print(f"  Acc:   {method_df['acc'].mean():.3f} ± {method_df['acc'].std():.3f}")
    
    # Calculate improvements
    masked_mean = masked_df.mean(numeric_only=True)
    unmasked_mean = unmasked_df.mean(numeric_only=True)
    
    print(f"\n{'Improvement (Masked - Unmasked):'}")
    print(f"  F1@10: {(masked_mean['f1_10'] - unmasked_mean['f1_10']):.3f} ({((masked_mean['f1_10'] / unmasked_mean['f1_10'] - 1) * 100):.1f}%)")
    print(f"  F1@25: {(masked_mean['f1_25'] - unmasked_mean['f1_25']):.3f} ({((masked_mean['f1_25'] / unmasked_mean['f1_25'] - 1) * 100):.1f}%)")
    print(f"  F1@50: {(masked_mean['f1_50'] - unmasked_mean['f1_50']):.3f} ({((masked_mean['f1_50'] / unmasked_mean['f1_50'] - 1) * 100):.1f}%)")
    print(f"  Edit:  {(masked_mean['edit'] - unmasked_mean['edit']):.3f} ({((masked_mean['edit'] / unmasked_mean['edit'] - 1) * 100):.1f}%)")
    
    # Create comparison plots
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    metrics = [
        ("f1_10", "F1@10", axes[0]),
        ("f1_25", "F1@25", axes[1]),
        ("f1_50", "F1@50", axes[2]),
    ]
    
    for metric, label, ax in metrics:
        # Box plot
        data_to_plot = [
            unmasked_df[metric].values,
            masked_df[metric].values
        ]
        bp = ax.boxplot(data_to_plot, labels=["Unmasked", "Masked"], 
                        patch_artist=True, widths=0.6)
        
        # Color boxes
        colors = ['#ff9999', '#99ff99']
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
        
        # Add individual points
        for i, data in enumerate(data_to_plot, 1):
            y = data
            x = np.random.normal(i, 0.04, size=len(y))
            ax.scatter(x, y, alpha=0.6, s=50, color='black', zorder=3)
        
        ax.set_ylabel(label, fontsize=12)
        ax.set_title(f"{label} Comparison", fontsize=14, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        ax.set_ylim(bottom=0)
    
    plt.tight_layout()
    plt.savefig(args.output, dpi=300, bbox_inches='tight')
    print(f"\n✓ Comparison plot saved to: {args.output}")
    
    # Save CSV
    csv_output = args.output.replace('.png', '.csv')
    combined_df.to_csv(csv_output, index=False)
    print(f"✓ Results CSV saved to: {csv_output}")
    
    # Create summary table
    summary = combined_df.groupby("method").agg({
        "f1_10": ["mean", "std"],
        "f1_25": ["mean", "std"],
        "f1_50": ["mean", "std"],
        "edit": ["mean", "std"],
        "acc": ["mean", "std"],
    }).round(3)
    
    summary_output = args.output.replace('.png', '_summary.csv')
    summary.to_csv(summary_output)
    print(f"✓ Summary table saved to: {summary_output}")


if __name__ == "__main__":
    main()




