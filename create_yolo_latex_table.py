#!/usr/bin/env python3
"""
Generate LaTeX table for YOLO object detection results from DTULabAssist.
"""
import pandas as pd
import numpy as np
from pathlib import Path

def parse_yolo_results(results_csv_path):
    """Parse YOLO results.csv file and extract key metrics."""
    df = pd.read_csv(results_csv_path)
    
    # Get final epoch metrics (last row)
    final = df.iloc[-1]
    
    # Get best epoch based on mAP50
    best_idx = df['metrics/mAP50(B)'].idxmax()
    best = df.iloc[best_idx]
    
    return {
        'final_epoch': int(final['epoch']),
        'final_precision': final['metrics/precision(B)'],
        'final_recall': final['metrics/recall(B)'],
        'final_mAP50': final['metrics/mAP50(B)'],
        'final_mAP50_95': final['metrics/mAP50-95(B)'],
        'best_epoch': int(best['epoch']),
        'best_precision': best['metrics/precision(B)'],
        'best_recall': best['metrics/recall(B)'],
        'best_mAP50': best['metrics/mAP50(B)'],
        'best_mAP50_95': best['metrics/mAP50-95(B)'],
    }

def create_yolo_latex_table(train_runs):
    """Create LaTeX table for YOLO results."""
    
    latex = r"""\begin{table}[htbp]
\centering
\caption{YOLO Object Detection Performance on DTULabAssist Dataset}
\label{tab:yolo_results}
\begin{tabular}{lcccccc}
\toprule
\textbf{Training Run} & \textbf{Epochs} & \textbf{Precision} & \textbf{Recall} & \textbf{mAP@50} & \textbf{mAP@50-95} \\
\midrule
"""
    
    for run_name, metrics in train_runs.items():
        latex += f"{run_name} (Final) & {metrics['final_epoch']} & "
        latex += f"{metrics['final_precision']:.3f} & {metrics['final_recall']:.3f} & "
        latex += f"{metrics['final_mAP50']:.3f} & {metrics['final_mAP50_95']:.3f} \\\\\n"
        
        if metrics['best_epoch'] != metrics['final_epoch']:
            latex += f"{run_name} (Best) & {metrics['best_epoch']} & "
            latex += f"{metrics['best_precision']:.3f} & {metrics['best_recall']:.3f} & "
            latex += f"{metrics['best_mAP50']:.3f} & {metrics['best_mAP50_95']:.3f} \\\\\n"
    
    latex += r"""\bottomrule
\end{tabular}
\end{table}
"""
    
    return latex

def create_combined_results_table(yolo_results, action_results):
    """Create combined LaTeX table with both YOLO and Action Recognition results."""
    
    latex = r"""\begin{table}[htbp]
\centering
\caption{DTULabAssist System Performance: Object Detection and Action Recognition}
\label{tab:combined_results}
\resizebox{\textwidth}{!}{%
\begin{tabular}{llcccccc}
\toprule
\textbf{Component} & \textbf{Model} & \textbf{Metric} & \textbf{Value} & \textbf{Std} & \textbf{Best Fold} & \textbf{Worst Fold} \\
\midrule
"""
    
    # YOLO section
    latex += r"\multirow{4}{*}{\textbf{Object Detection}} & \multirow{4}{*}{YOLOv8n} "
    latex += f"& Precision & {yolo_results['best_precision']:.3f} & - & - & - \\\\\n"
    latex += f" &  & Recall & {yolo_results['best_recall']:.3f} & - & - & - \\\\\n"
    latex += f" &  & mAP@50 & {yolo_results['best_mAP50']:.3f} & - & - & - \\\\\n"
    latex += f" &  & mAP@50-95 & {yolo_results['best_mAP50_95']:.3f} & - & - & - \\\\\n"
    latex += r"\midrule" + "\n"
    
    # Action Recognition section
    latex += r"\multirow{5}{*}{\textbf{Action Recognition}} & \multirow{5}{*}{MS-TCN + V-JEPA} "
    latex += f"& Frame Accuracy & {action_results['mean_acc']:.3f} & {action_results['std_acc']:.3f} & "
    latex += f"{action_results['best_acc']:.3f} & {action_results['worst_acc']:.3f} \\\\\n"
    latex += f" &  & Macro F1 (all) & {action_results['mean_f1_all']:.3f} & {action_results['std_f1_all']:.3f} & "
    latex += f"{action_results['best_f1_all']:.3f} & {action_results['worst_f1_all']:.3f} \\\\\n"
    latex += f" &  & Macro F1 (no bg) & {action_results['mean_f1_no_bg']:.3f} & {action_results['std_f1_no_bg']:.3f} & "
    latex += f"{action_results['best_f1_no_bg']:.3f} & {action_results['worst_f1_no_bg']:.3f} \\\\\n"
    latex += f" &  & Edit Score & {action_results['mean_edit']:.3f} & {action_results['std_edit']:.3f} & "
    latex += f"{action_results['best_edit']:.3f} & {action_results['worst_edit']:.3f} \\\\\n"
    latex += f" &  & F1@50 & {action_results['mean_f1_50']:.3f} & {action_results['std_f1_50']:.3f} & "
    latex += f"{action_results['best_f1_50']:.3f} & {action_results['worst_f1_50']:.3f} \\\\\n"
    
    latex += r"""\bottomrule
\end{tabular}%
}
\end{table}
"""
    
    return latex

def main():
    # Parse YOLO results
    print("Parsing YOLO results...")
    yolo_runs = {}
    
    # train5 results
    train5_path = Path('/zhome/45/8/156251/Desktop/thesis/labassist/runs/detect/train5/results.csv')
    if train5_path.exists():
        yolo_runs['train5'] = parse_yolo_results(train5_path)
        print(f"  ✓ train5: mAP@50 = {yolo_runs['train5']['best_mAP50']:.3f} (epoch {yolo_runs['train5']['best_epoch']})")
    
    # labassist_yolo11m22 results
    yolo11_path = Path('/zhome/45/8/156251/Desktop/thesis/labassist/DTULabAssistDataset/runs/detect/labassist_yolo11m22/results.csv')
    if yolo11_path.exists():
        yolo_runs['yolo11m'] = parse_yolo_results(yolo11_path)
        print(f"  ✓ yolo11m: mAP@50 = {yolo_runs['yolo11m']['best_mAP50']:.3f} (epoch {yolo_runs['yolo11m']['best_epoch']})")
    
    # Parse action recognition results
    print("\nParsing action recognition results...")
    action_csv = Path('/zhome/45/8/156251/Desktop/thesis/labassist/hpc_scripts/scripts/analysis/thesis_figures/output/table2_crossfold_performance.csv')
    
    if action_csv.exists():
        df = pd.read_csv(action_csv)
        # Filter out Mean and Std rows
        df_folds = df[df['fold'] != '-'].copy()
        
        action_results = {
            'mean_acc': df_folds['micro_acc'].astype(float).mean(),
            'std_acc': df_folds['micro_acc'].astype(float).std(),
            'best_acc': df_folds['micro_acc'].astype(float).max(),
            'worst_acc': df_folds['micro_acc'].astype(float).min(),
            
            'mean_f1_all': df_folds['macroF1_all'].astype(float).mean(),
            'std_f1_all': df_folds['macroF1_all'].astype(float).std(),
            'best_f1_all': df_folds['macroF1_all'].astype(float).max(),
            'worst_f1_all': df_folds['macroF1_all'].astype(float).min(),
            
            'mean_f1_no_bg': df_folds['macroF1_no_bg'].astype(float).mean(),
            'std_f1_no_bg': df_folds['macroF1_no_bg'].astype(float).std(),
            'best_f1_no_bg': df_folds['macroF1_no_bg'].astype(float).max(),
            'worst_f1_no_bg': df_folds['macroF1_no_bg'].astype(float).min(),
            
            'mean_edit': df_folds['edit_score'].astype(float).mean(),
            'std_edit': df_folds['edit_score'].astype(float).std(),
            'best_edit': df_folds['edit_score'].astype(float).max(),
            'worst_edit': df_folds['edit_score'].astype(float).min(),
            
            'mean_f1_50': df_folds['F1@50'].astype(float).mean(),
            'std_f1_50': df_folds['F1@50'].astype(float).std(),
            'best_f1_50': df_folds['F1@50'].astype(float).max(),
            'worst_f1_50': df_folds['F1@50'].astype(float).min(),
        }
        print(f"  ✓ Action Recognition: Frame Acc = {action_results['mean_acc']:.3f} ± {action_results['std_acc']:.3f}")
        print(f"  ✓ Action Recognition: F1@50 = {action_results['mean_f1_50']:.3f} ± {action_results['std_f1_50']:.3f}")
    else:
        print("  ✗ Action recognition results not found")
        action_results = None
    
    # Create output directory
    output_dir = Path('/zhome/45/8/156251/Desktop/thesis/labassist/latex_tables')
    output_dir.mkdir(exist_ok=True)
    
    # Generate YOLO table
    if yolo_runs:
        print("\nGenerating YOLO LaTeX table...")
        yolo_latex = create_yolo_latex_table(yolo_runs)
        yolo_output = output_dir / 'yolo_results.tex'
        yolo_output.write_text(yolo_latex)
        print(f"  ✓ Saved to: {yolo_output}")
    
    # Generate combined table
    if yolo_runs and action_results:
        print("\nGenerating combined LaTeX table...")
        # Use best YOLO model
        best_yolo = max(yolo_runs.values(), key=lambda x: x['best_mAP50'])
        combined_latex = create_combined_results_table(best_yolo, action_results)
        combined_output = output_dir / 'combined_results.tex'
        combined_output.write_text(combined_latex)
        print(f"  ✓ Saved to: {combined_output}")
    
    # Create summary CSV
    print("\nGenerating summary CSV...")
    summary_data = []
    
    for run_name, metrics in yolo_runs.items():
        summary_data.append({
            'Component': 'Object Detection',
            'Model': run_name,
            'Metric': 'mAP@50',
            'Value': f"{metrics['best_mAP50']:.3f}",
            'Epoch': metrics['best_epoch']
        })
        summary_data.append({
            'Component': 'Object Detection',
            'Model': run_name,
            'Metric': 'mAP@50-95',
            'Value': f"{metrics['best_mAP50_95']:.3f}",
            'Epoch': metrics['best_epoch']
        })
    
    if action_results:
        summary_data.append({
            'Component': 'Action Recognition',
            'Model': 'MS-TCN + V-JEPA',
            'Metric': 'Frame Accuracy',
            'Value': f"{action_results['mean_acc']:.3f} ± {action_results['std_acc']:.3f}",
            'Epoch': '-'
        })
        summary_data.append({
            'Component': 'Action Recognition',
            'Model': 'MS-TCN + V-JEPA',
            'Metric': 'F1@50',
            'Value': f"{action_results['mean_f1_50']:.3f} ± {action_results['std_f1_50']:.3f}",
            'Epoch': '-'
        })
    
    summary_df = pd.DataFrame(summary_data)
    summary_csv = output_dir / 'results_summary.csv'
    summary_df.to_csv(summary_csv, index=False)
    print(f"  ✓ Saved to: {summary_csv}")
    
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(summary_df.to_string(index=False))
    print("\n✓ All LaTeX tables generated successfully!")
    print(f"\nOutput directory: {output_dir}")
    print("\nFiles created:")
    print(f"  - yolo_results.tex")
    print(f"  - combined_results.tex")
    print(f"  - results_summary.csv")

if __name__ == '__main__':
    main()


