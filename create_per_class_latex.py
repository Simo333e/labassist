#!/usr/bin/env python3
"""
Generate per-class performance LaTeX table for DTULabAssist action recognition.
"""
import pandas as pd
from pathlib import Path

def create_per_class_latex_table(csv_path):
    """Create LaTeX table for per-class metrics."""
    df = pd.read_csv(csv_path)
    
    latex = r"""\begin{table}[htbp]
\centering
\caption{Per-Class Action Recognition Performance}
\label{tab:per_class_action}
\begin{tabular}{lcccc}
\toprule
\textbf{Action Class} & \textbf{Precision} & \textbf{Recall} & \textbf{F1-Score} & \textbf{Support} \\
\midrule
"""
    
    for _, row in df.iterrows():
        class_name = row['Class'].replace('_', ' ').title()
        latex += f"{class_name} & {row['Precision']:.3f} & {row['Recall']:.3f} & {row['F1']:.3f} & {int(row['Support'])} \\\\\n"
    
    latex += r"""\bottomrule
\end{tabular}
\end{table}
"""
    
    return latex

def main():
    print("Generating per-class LaTeX table...")
    
    per_class_csv = Path('/zhome/45/8/156251/Desktop/thesis/labassist/hpc_scripts/scripts/analysis/thesis_figures/output/table3_per_class_metrics.csv')
    output_dir = Path('/zhome/45/8/156251/Desktop/thesis/labassist/latex_tables')
    output_dir.mkdir(exist_ok=True)
    
    if per_class_csv.exists():
        latex = create_per_class_latex_table(per_class_csv)
        output_file = output_dir / 'per_class_action.tex'
        output_file.write_text(latex)
        print(f"  ✓ Saved to: {output_file}")
        
        # Print preview
        print("\nPer-class metrics:")
        df = pd.read_csv(per_class_csv)
        print(df.to_string(index=False))
    else:
        print(f"  ✗ File not found: {per_class_csv}")
    
    print("\n✓ Done!")

if __name__ == '__main__':
    main()


