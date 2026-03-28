#!/usr/bin/env python3
"""Generate per-recording summary table for DTU data."""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict

# Recording metadata (operator info if available)
RECORDINGS = {
    '20251027_181439Z': {'fold': 0, 'operator': 'A'},
    '20251112_110656Z_1080p30_v2': {'fold': 1, 'operator': 'B'},
    '20251120_065442Z': {'fold': 2, 'operator': 'C', 'note': 'broken pipette'},
    '20251125_070904Z': {'fold': 3, 'operator': 'D'},
    '20251125_072321Z': {'fold': 4, 'operator': 'D'},
    'Lab_video_cut': {'fold': 5, 'operator': 'E'},
}

CLASS_INDEX = {
    "none": 0,
    "Pipette pcr tube": 1,
    "Aspirate reagent": 2,
    "Attach pipette tip": 3,
    "Eject pipette tip": 4
}

def compute_recording_stats(labels_dir: Path, fps: float = 10.0):
    """Compute per-recording statistics."""
    idx_to_class = {v: k for k, v in CLASS_INDEX.items()}
    records = []
    
    for stem, meta in RECORDINGS.items():
        label_path = labels_dir / f"{stem}_y.npy"
        if not label_path.exists():
            print(f"Warning: Label file not found: {label_path}")
            continue
        
        labels = np.load(label_path)
        total_frames = len(labels)
        duration_sec = total_frames / fps
        duration_min = duration_sec / 60
        
        # Count segments (transitions)
        def count_segments(arr):
            if len(arr) == 0:
                return 0
            changes = np.concatenate([[True], arr[1:] != arr[:-1]])
            return np.sum(changes)
        
        num_segments = count_segments(labels)
        
        # Per-class stats
        record = {
            'Recording ID': stem,
            'Fold': meta['fold'],
            'Operator': meta.get('operator', '?'),
            'Duration (min)': duration_min,
            'Total Frames': total_frames,
            'Action Segments': num_segments,
        }
        
        # None stats
        none_frames = np.sum(labels == 0)
        none_min = none_frames / fps / 60
        none_pct = 100 * none_frames / total_frames
        record['None (min)'] = none_min
        record['None (%)'] = none_pct
        
        # Per action class
        for class_name, class_idx in sorted(CLASS_INDEX.items(), key=lambda x: x[1]):
            if class_name == 'none':
                continue
            class_frames = np.sum(labels == class_idx)
            class_min = class_frames / fps / 60
            # Shorten names for table (use more descriptive abbreviations)
            short_name = class_name.replace('Pipette pcr tube', 'PCR tube').replace('Aspirate reagent', 'Reagent').replace('Attach pipette tip', 'Tip on').replace('Eject pipette tip', 'Tip off')
            record[f'{short_name} (min)'] = class_min
        
        records.append(record)
    
    return pd.DataFrame(records)

def main():
    # Use 5-class PCR labels
    labels_dir = Path('/work3/s204157/data/labassist/my_data/cache_fixed_v2/labels_5class')
    output_dir = Path('/zhome/45/8/156251/Desktop/thesis/labassist/hpc_scripts/scripts/analysis/thesis_figures/output')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    df = compute_recording_stats(labels_dir)
    
    # Format for display
    df_display = df.copy()
    df_display['Duration (min)'] = df_display['Duration (min)'].apply(lambda x: f"{x:.1f}")
    df_display['None (min)'] = df_display['None (min)'].apply(lambda x: f"{x:.1f}")
    df_display['None (%)'] = df_display['None (%)'].apply(lambda x: f"{x:.1f}")
    
    for col in df_display.columns:
        if '(min)' in col and col not in ['Duration (min)', 'None (min)']:
            df_display[col] = df_display[col].apply(lambda x: f"{x:.2f}")
    
    # Save as CSV and LaTeX
    df_display.to_csv(output_dir / 'table1_recording_summary.csv', index=False)
    df_display.to_latex(output_dir / 'table1_recording_summary.tex', index=False)
    
    print("\n" + "="*100)
    print("TABLE 1: Dataset Summary (Recording Level)")
    print("="*100)
    print(df_display.to_string(index=False))
    print("\n✓ Saved to:")
    print(f"  - {output_dir / 'table1_recording_summary.csv'}")
    print(f"  - {output_dir / 'table1_recording_summary.tex'}")

if __name__ == '__main__':
    main()


