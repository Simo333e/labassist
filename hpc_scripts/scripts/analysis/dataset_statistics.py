#!/usr/bin/env python3
"""
Dataset Statistics and Visualization Script

Generates comprehensive statistics and publication-ready figures for:
1. Combined PCR dataset (FineBio PCR + mydata)
2. FineBio-only PCR videos
3. Mydata-only videos
4. FineBio atomic (full) dataset

Outputs:
- Console summary of all statistics
- CSV files with detailed statistics
- Publication-ready figures (PDF and PNG)
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import seaborn as sns

# Set style for publication-ready figures
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif', 'serif'],
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.titlesize': 14,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.spines.top': False,
    'axes.spines.right': False,
})

# Color palette - accessible and distinct
COLORS = {
    'finebio': '#2E86AB',      # Blue
    'mydata': '#A23B72',       # Magenta/Pink
    'combined': '#1B998B',     # Teal
    'background': '#E8E8E8',   # Light gray for 'none' class
}

# Action class colors - using tab20 colormap for consistency with evaluation timelines
def get_action_colors(class_names):
    """Generate consistent colors using tab20 colormap (same as evaluation scripts)."""
    import matplotlib
    base = matplotlib.colormaps.get_cmap("tab20")
    colors = {}
    for i, name in enumerate(class_names):
        rgba = base(i % 20)
        # Convert RGBA to hex
        hex_color = '#{:02x}{:02x}{:02x}'.format(
            int(rgba[0] * 255), int(rgba[1] * 255), int(rgba[2] * 255)
        )
        colors[name] = hex_color
    return colors


def is_finebio_video(stem: str) -> bool:
    """Check if video stem is from FineBio dataset (P##_05_##_T5 pattern)."""
    return stem.startswith('P') and '_05_' in stem and '_T5' in stem


def is_mydata_video(stem: str) -> bool:
    """Check if video stem is from mydata (not FineBio pattern)."""
    return not is_finebio_video(stem)


def load_labels(label_path: Path) -> np.ndarray:
    """Load label array from .npy file."""
    return np.load(label_path)


def find_action_segments(labels: np.ndarray, fps: float = 10.0) -> Dict[int, List[Tuple[float, float]]]:
    """
    Find contiguous segments for each action class.
    
    Returns:
        Dict mapping class_idx -> list of (start_time, end_time) tuples
    """
    segments = defaultdict(list)
    
    if len(labels) == 0:
        return segments
    
    current_class = labels[0]
    segment_start = 0
    
    for i in range(1, len(labels)):
        if labels[i] != current_class:
            # End of segment
            start_time = segment_start / fps
            end_time = (i - 1) / fps
            segments[int(current_class)].append((start_time, end_time))
            # Start new segment
            current_class = labels[i]
            segment_start = i
    
    # Handle last segment
    start_time = segment_start / fps
    end_time = (len(labels) - 1) / fps
    segments[int(current_class)].append((start_time, end_time))
    
    return segments


def compute_statistics(
    labels_dir: Path,
    class_index: Dict[str, int],
    fps: float = 10.0
) -> pd.DataFrame:
    """
    Compute statistics for all videos in a labels directory.
    
    Returns DataFrame with columns:
        video_stem, source, duration_sec, total_frames, 
        action_frames, background_frames, action_ratio,
        num_segments_per_class, segment_durations_per_class
    """
    label_files = sorted(labels_dir.glob("*_y.npy"))
    idx_to_class = {v: k for k, v in class_index.items()}
    num_classes = len(class_index)
    
    records = []
    
    for label_path in label_files:
        stem = label_path.stem.replace('_y', '')
        labels = load_labels(label_path)
        
        # Basic stats
        total_frames = len(labels)
        duration_sec = total_frames / fps
        
        # Count frames per class
        unique, counts = np.unique(labels[labels >= 0], return_counts=True)
        frame_counts = dict(zip(unique.astype(int), counts.astype(int)))
        
        background_frames = frame_counts.get(0, 0)  # 'none' is index 0
        action_frames = sum(c for idx, c in frame_counts.items() if idx != 0)
        action_ratio = action_frames / total_frames if total_frames > 0 else 0
        
        # Segment analysis
        segments = find_action_segments(labels, fps)
        
        record = {
            'video_stem': stem,
            'source': 'FineBio' if is_finebio_video(stem) else 'MyData',
            'duration_sec': duration_sec,
            'duration_min': duration_sec / 60,
            'total_frames': total_frames,
            'background_frames': background_frames,
            'action_frames': action_frames,
            'action_ratio': action_ratio,
        }
        
        # Per-class statistics
        for class_name, class_idx in class_index.items():
            class_frames = frame_counts.get(class_idx, 0)
            class_segments = segments.get(class_idx, [])
            segment_durations = [end - start for start, end in class_segments]
            
            record[f'frames_{class_name}'] = class_frames
            record[f'num_segments_{class_name}'] = len(class_segments)
            record[f'mean_segment_dur_{class_name}'] = np.mean(segment_durations) if segment_durations else 0
            record[f'std_segment_dur_{class_name}'] = np.std(segment_durations) if segment_durations else 0
        
        records.append(record)
    
    return pd.DataFrame(records)


def print_summary(df: pd.DataFrame, class_index: Dict[str, int], dataset_name: str):
    """Print formatted summary statistics."""
    print(f"\n{'='*70}")
    print(f" {dataset_name} Dataset Summary")
    print(f"{'='*70}")
    
    print(f"\n📊 Overall Statistics:")
    print(f"   Total videos: {len(df)}")
    
    if 'source' in df.columns:
        source_counts = df['source'].value_counts()
        for source, count in source_counts.items():
            print(f"   - {source}: {count} videos")
    
    print(f"\n⏱️  Video Durations:")
    print(f"   Total duration: {df['duration_min'].sum():.1f} min ({df['duration_sec'].sum()/3600:.2f} hours)")
    print(f"   Mean duration: {df['duration_sec'].mean():.1f} sec ({df['duration_min'].mean():.2f} min)")
    print(f"   Std duration: {df['duration_sec'].std():.1f} sec")
    print(f"   Min duration: {df['duration_sec'].min():.1f} sec")
    print(f"   Max duration: {df['duration_sec'].max():.1f} sec")
    
    print(f"\n🎬 Frame Statistics:")
    print(f"   Total frames: {df['total_frames'].sum():,}")
    print(f"   Background frames: {df['background_frames'].sum():,} ({100*df['background_frames'].sum()/df['total_frames'].sum():.1f}%)")
    print(f"   Action frames: {df['action_frames'].sum():,} ({100*df['action_frames'].sum()/df['total_frames'].sum():.1f}%)")
    
    print(f"\n📋 Per-Class Statistics:")
    for class_name, class_idx in class_index.items():
        frames_col = f'frames_{class_name}'
        segments_col = f'num_segments_{class_name}'
        mean_dur_col = f'mean_segment_dur_{class_name}'
        
        if frames_col in df.columns:
            total_frames = df[frames_col].sum()
            total_segments = df[segments_col].sum()
            pct = 100 * total_frames / df['total_frames'].sum()
            
            # Calculate mean segment duration across all videos
            all_mean_durs = df[mean_dur_col].values
            valid_durs = all_mean_durs[all_mean_durs > 0]
            avg_dur = np.mean(valid_durs) if len(valid_durs) > 0 else 0
            
            print(f"   [{class_idx}] {class_name}:")
            print(f"       Frames: {total_frames:,} ({pct:.1f}%)")
            print(f"       Segments: {int(total_segments)}")
            if avg_dur > 0:
                print(f"       Avg segment duration: {avg_dur:.2f} sec")


def plot_video_duration_distribution(df: pd.DataFrame, output_dir: Path, dataset_name: str):
    """Create histogram of video durations."""
    fig, ax = plt.subplots(figsize=(8, 5))
    
    if 'source' in df.columns and df['source'].nunique() > 1:
        # Separate by source
        for source in df['source'].unique():
            subset = df[df['source'] == source]
            color = COLORS['finebio'] if source == 'FineBio' else COLORS['mydata']
            ax.hist(subset['duration_min'], bins=15, alpha=0.7, label=source, 
                   color=color, edgecolor='white', linewidth=0.5)
        ax.legend()
    else:
        ax.hist(df['duration_min'], bins=15, alpha=0.8, 
               color=COLORS['combined'], edgecolor='white', linewidth=0.5)
    
    ax.set_xlabel('Video Duration (minutes)')
    ax.set_ylabel('Number of Videos')
    ax.set_title(f'Video Duration Distribution - {dataset_name}')
    
    # Add summary stats
    mean_dur = df['duration_min'].mean()
    ax.axvline(mean_dur, color='red', linestyle='--', linewidth=1.5, label=f'Mean: {mean_dur:.1f} min')
    ax.legend()
    
    plt.tight_layout()
    fig.savefig(output_dir / f'duration_distribution_{dataset_name.lower().replace(" ", "_")}.pdf')
    fig.savefig(output_dir / f'duration_distribution_{dataset_name.lower().replace(" ", "_")}.png')
    plt.close(fig)


def plot_class_distribution(df: pd.DataFrame, class_index: Dict[str, int], 
                           output_dir: Path, dataset_name: str):
    """Create bar chart of class distribution."""
    fig, ax = plt.subplots(1, 1, figsize=(14, 5))
    
    # Generate consistent colors using tab20 colormap
    action_colors = get_action_colors(list(class_index.keys()))
    
    # Get frame counts per class
    class_frames = {}
    for class_name in class_index.keys():
        col = f'frames_{class_name}'
        if col in df.columns:
            class_frames[class_name] = df[col].sum()
    
    total_frames = sum(class_frames.values())
    
    # Sort by count (excluding 'none')
    sorted_classes = sorted(
        [(k, v) for k, v in class_frames.items()],
        key=lambda x: (0 if x[0] == 'none' else 1, -x[1])
    )
    
    classes = [c[0] for c in sorted_classes]
    counts = [c[1] for c in sorted_classes]
    percentages = [100 * c / total_frames for c in counts]
    colors = [action_colors.get(c, '#888888') for c in classes]
    
    # Pie chart - proportions
    # Combine small classes for pie chart visibility
    pie_data = list(zip(classes, percentages, colors))
    
    wedges, texts, autotexts = ax.pie(
        percentages, labels=classes, autopct=lambda p: f'{p:.1f}%' if p > 2 else '',
        colors=colors, startangle=90, pctdistance=0.75
    )
    ax.set_title(f'Class Distribution - {dataset_name}')
    
    plt.tight_layout()
    fig.savefig(output_dir / f'class_distribution_{dataset_name.lower().replace(" ", "_")}.png')
    plt.close(fig)


def plot_segment_duration_boxplot(df: pd.DataFrame, class_index: Dict[str, int],
                                   output_dir: Path, dataset_name: str):
    """Create boxplot of segment durations per class."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Generate consistent colors using tab20 colormap
    action_colors = get_action_colors(list(class_index.keys()))
    
    # Collect all segment durations
    segment_data = []
    for _, row in df.iterrows():
        for class_name in class_index.keys():
            if class_name == 'none':
                continue
            mean_dur_col = f'mean_segment_dur_{class_name}'
            if mean_dur_col in df.columns and row[mean_dur_col] > 0:
                segment_data.append({
                    'class': class_name,
                    'duration': row[mean_dur_col],
                    'source': row.get('source', 'Combined')
                })
    
    if not segment_data:
        plt.close(fig)
        return
    
    seg_df = pd.DataFrame(segment_data)
    
    # Order classes by median duration
    class_order = seg_df.groupby('class')['duration'].median().sort_values(ascending=False).index.tolist()
    
    colors_list = [action_colors.get(c, '#888888') for c in class_order]
    
    sns.boxplot(data=seg_df, x='class', y='duration', order=class_order,
                palette=colors_list, ax=ax, width=0.6)
    
    ax.set_xlabel('Action Class')
    ax.set_ylabel('Segment Duration (seconds)')
    ax.set_title(f'Action Segment Duration Distribution - {dataset_name}')
    plt.xticks(rotation=15, ha='right')
    
    plt.tight_layout()
    fig.savefig(output_dir / f'segment_durations_{dataset_name.lower().replace(" ", "_")}.pdf')
    fig.savefig(output_dir / f'segment_durations_{dataset_name.lower().replace(" ", "_")}.png')
    plt.close(fig)


def plot_source_comparison(df: pd.DataFrame, class_index: Dict[str, int],
                           output_dir: Path, dataset_name: str):
    """Compare statistics between FineBio and MyData sources."""
    if 'source' not in df.columns or df['source'].nunique() <= 1:
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # 1. Duration comparison
    ax1 = axes[0, 0]
    sns.boxplot(data=df, x='source', y='duration_min', 
                palette=[COLORS['finebio'], COLORS['mydata']], ax=ax1)
    ax1.set_xlabel('Data Source')
    ax1.set_ylabel('Duration (minutes)')
    ax1.set_title('Video Duration by Source')
    
    # 2. Action ratio comparison
    ax2 = axes[0, 1]
    sns.boxplot(data=df, x='source', y='action_ratio',
                palette=[COLORS['finebio'], COLORS['mydata']], ax=ax2)
    ax2.set_xlabel('Data Source')
    ax2.set_ylabel('Action Ratio (action frames / total)')
    ax2.set_title('Action Density by Source')
    
    # 3. Class distribution comparison
    ax3 = axes[1, 0]
    class_names = [c for c in class_index.keys() if c != 'none']
    
    finebio_df = df[df['source'] == 'FineBio']
    mydata_df = df[df['source'] == 'MyData']
    
    x = np.arange(len(class_names))
    width = 0.35
    
    finebio_pcts = []
    mydata_pcts = []
    
    for class_name in class_names:
        col = f'frames_{class_name}'
        fb_total = finebio_df['total_frames'].sum()
        md_total = mydata_df['total_frames'].sum()
        
        finebio_pcts.append(100 * finebio_df[col].sum() / fb_total if fb_total > 0 else 0)
        mydata_pcts.append(100 * mydata_df[col].sum() / md_total if md_total > 0 else 0)
    
    ax3.bar(x - width/2, finebio_pcts, width, label='FineBio', color=COLORS['finebio'])
    ax3.bar(x + width/2, mydata_pcts, width, label='MyData', color=COLORS['mydata'])
    ax3.set_xlabel('Action Class')
    ax3.set_ylabel('Percentage of Frames')
    ax3.set_title('Class Distribution by Source')
    ax3.set_xticks(x)
    ax3.set_xticklabels(class_names, rotation=15, ha='right')
    ax3.legend()
    
    # 4. Segment count comparison
    ax4 = axes[1, 1]
    
    finebio_segs = []
    mydata_segs = []
    
    for class_name in class_names:
        col = f'num_segments_{class_name}'
        finebio_segs.append(finebio_df[col].sum())
        mydata_segs.append(mydata_df[col].sum())
    
    ax4.bar(x - width/2, finebio_segs, width, label='FineBio', color=COLORS['finebio'])
    ax4.bar(x + width/2, mydata_segs, width, label='MyData', color=COLORS['mydata'])
    ax4.set_xlabel('Action Class')
    ax4.set_ylabel('Number of Segments')
    ax4.set_title('Action Instance Count by Source')
    ax4.set_xticks(x)
    ax4.set_xticklabels(class_names, rotation=15, ha='right')
    ax4.legend()
    
    plt.tight_layout()
    fig.savefig(output_dir / f'source_comparison_{dataset_name.lower().replace(" ", "_")}.pdf')
    fig.savefig(output_dir / f'source_comparison_{dataset_name.lower().replace(" ", "_")}.png')
    plt.close(fig)


def plot_example_timelines(df: pd.DataFrame, labels_dir: Path, class_index: Dict[str, int],
                           output_dir: Path, dataset_name: str, num_examples: int = 4):
    """Plot example action timelines for representative videos."""
    idx_to_class = {v: k for k, v in class_index.items()}
    
    # Generate consistent colors using tab20 colormap
    action_colors = get_action_colors(list(class_index.keys()))
    
    # Select diverse examples (mix of sources and durations)
    examples = []
    
    if 'source' in df.columns and df['source'].nunique() > 1:
        # Get examples from each source
        for source in ['FineBio', 'MyData']:
            source_df = df[df['source'] == source].sort_values('duration_sec')
            if len(source_df) >= 2:
                # Short and long video from each source
                examples.append(source_df.iloc[0]['video_stem'])
                examples.append(source_df.iloc[len(source_df)//2]['video_stem'])
    else:
        # Just pick based on duration diversity
        sorted_df = df.sort_values('duration_sec')
        indices = np.linspace(0, len(sorted_df)-1, min(num_examples, len(sorted_df)), dtype=int)
        examples = [sorted_df.iloc[i]['video_stem'] for i in indices]
    
    examples = examples[:num_examples]
    
    if not examples:
        return
    
    fig, axes = plt.subplots(len(examples), 1, figsize=(14, 2.5*len(examples)))
    if len(examples) == 1:
        axes = [axes]
    
    for ax, stem in zip(axes, examples):
        label_path = labels_dir / f"{stem}_y.npy"
        if not label_path.exists():
            continue
        
        labels = load_labels(label_path)
        duration_sec = len(labels) / 10.0  # 10 Hz
        
        # Create timeline
        t = np.arange(len(labels)) / 10.0
        
        # Plot segments
        for class_idx in range(len(class_index)):
            class_name = idx_to_class.get(class_idx, f'class_{class_idx}')
            mask = labels == class_idx
            
            if mask.any():
                color = action_colors.get(class_name, '#888888')
                ax.fill_between(t, 0, 1, where=mask, color=color, alpha=0.8,
                               label=class_name if class_idx > 0 else None)
        
        source = 'FineBio' if is_finebio_video(stem) else 'MyData'
        ax.set_title(f'{stem} ({source}, {duration_sec:.0f}s)', fontsize=10)
        ax.set_xlim(0, duration_sec)
        ax.set_ylim(0, 1)
        ax.set_yticks([])
        ax.set_xlabel('Time (seconds)')
    
    # Add legend
    handles = [mpatches.Patch(color=action_colors[c], label=c) 
               for c in class_index.keys() if c != 'none']
    fig.legend(handles=handles, loc='upper right', bbox_to_anchor=(0.98, 0.98),
               ncol=len(handles), fontsize=9)
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(output_dir / f'example_timelines_{dataset_name.lower().replace(" ", "_")}.pdf')
    fig.savefig(output_dir / f'example_timelines_{dataset_name.lower().replace(" ", "_")}.png')
    plt.close(fig)


def create_summary_table(all_stats: Dict[str, pd.DataFrame], output_dir: Path):
    """Create a summary comparison table across all datasets."""
    summary_rows = []
    
    for dataset_name, df in all_stats.items():
        row = {
            'Dataset': dataset_name,
            'Videos': len(df),
            'Total Duration (min)': df['duration_min'].sum(),
            'Mean Duration (sec)': df['duration_sec'].mean(),
            'Std Duration (sec)': df['duration_sec'].std(),
            'Total Frames': df['total_frames'].sum(),
            'Action Frames': df['action_frames'].sum(),
            'Background %': 100 * df['background_frames'].sum() / df['total_frames'].sum(),
        }
        
        # Source breakdown if available
        if 'source' in df.columns:
            source_counts = df['source'].value_counts()
            row['FineBio Videos'] = source_counts.get('FineBio', 0)
            row['MyData Videos'] = source_counts.get('MyData', 0)
        
        summary_rows.append(row)
    
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(output_dir / 'dataset_summary.csv', index=False)
    
    print("\n" + "="*70)
    print(" DATASET COMPARISON SUMMARY")
    print("="*70)
    print(summary_df.to_string(index=False))
    
    return summary_df


def create_class_coverage_table(all_stats: Dict[str, pd.DataFrame], class_index: Dict[str, int], output_dir: Path):
    """Create a table showing which classes are covered by which dataset."""
    print("\n" + "="*70)
    print(" CLASS COVERAGE BY DATASET (% of total frames)")
    print("="*70)
    
    coverage_rows = []
    for class_name in class_index.keys():
        row = {'Class': class_name}
        frames_col = f'frames_{class_name}'
        
        for dataset_name, df in all_stats.items():
            if frames_col in df.columns:
                total = df['total_frames'].sum()
                class_frames = df[frames_col].sum()
                pct = 100 * class_frames / total if total > 0 else 0
                row[dataset_name] = pct
            else:
                row[dataset_name] = 0.0
        
        coverage_rows.append(row)
    
    coverage_df = pd.DataFrame(coverage_rows)
    coverage_df.to_csv(output_dir / 'class_coverage_by_dataset.csv', index=False)
    
    # Print with formatting
    print(coverage_df.to_string(index=False, float_format='%.1f'))
    
    # Highlight missing classes
    print("\n⚠️  LABEL TAXONOMY MISMATCH WARNING:")
    for _, row in coverage_df.iterrows():
        class_name = row['Class']
        if class_name == 'none':
            continue
        for dataset_name in all_stats.keys():
            if dataset_name in row and row[dataset_name] == 0:
                print(f"   '{class_name}' has 0% in {dataset_name}")
    
    return coverage_df


def plot_class_coverage_heatmap(all_stats: Dict[str, pd.DataFrame], class_index: Dict[str, int], output_dir: Path):
    """Create a heatmap showing class coverage across datasets."""
    # Build coverage matrix
    datasets = list(all_stats.keys())
    classes = [c for c in class_index.keys() if c != 'none']
    
    coverage_matrix = np.zeros((len(classes), len(datasets)))
    
    for i, class_name in enumerate(classes):
        frames_col = f'frames_{class_name}'
        for j, dataset_name in enumerate(datasets):
            df = all_stats[dataset_name]
            if frames_col in df.columns:
                total = df['total_frames'].sum()
                class_frames = df[frames_col].sum()
                coverage_matrix[i, j] = 100 * class_frames / total if total > 0 else 0
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Create heatmap
    im = ax.imshow(coverage_matrix, cmap='YlOrRd', aspect='auto')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('% of Total Frames')
    
    # Set ticks
    ax.set_xticks(range(len(datasets)))
    ax.set_xticklabels(datasets, rotation=45, ha='right')
    ax.set_yticks(range(len(classes)))
    ax.set_yticklabels(classes)
    
    # Add value annotations
    for i in range(len(classes)):
        for j in range(len(datasets)):
            val = coverage_matrix[i, j]
            color = 'white' if val > 5 else 'black'
            text = f'{val:.1f}%' if val > 0 else '0%'
            fontweight = 'bold' if val == 0 else 'normal'
            ax.text(j, i, text, ha='center', va='center', color=color, 
                   fontsize=10, fontweight=fontweight)
    
    ax.set_title('Action Class Coverage by Dataset\n(Highlighting Label Taxonomy Mismatch)')
    
    plt.tight_layout()
    fig.savefig(output_dir / 'class_coverage_heatmap.pdf')
    fig.savefig(output_dir / 'class_coverage_heatmap.png')
    plt.close(fig)
    
    print(f"\n✓ Saved class coverage heatmap")


def main():
    parser = argparse.ArgumentParser(description='Generate dataset statistics and figures')
    parser.add_argument('--output_dir', type=str, 
                        default='/zhome/45/8/156251/Desktop/thesis/labassist/hpc_scripts/scripts/analysis/figures',
                        help='Output directory for figures and CSVs')
    parser.add_argument('--combined_pcr_labels', type=str,
                        default='/work3/s204157/data/labassist/combined_pcr/cache/labels_5class',
                        help='Path to combined PCR 5-class labels')
    parser.add_argument('--combined_pcr_class_index', type=str,
                        default='/work3/s204157/data/labassist/combined_pcr/cache/class_index_5class.json',
                        help='Path to combined PCR class index')
    parser.add_argument('--finebio_atomic_labels', type=str,
                        default='/work3/s204157/data/labassist/finebio_full/cache_atomic/labels',
                        help='Path to FineBio atomic labels')
    parser.add_argument('--finebio_atomic_class_index', type=str,
                        default='/work3/s204157/data/labassist/finebio_full/cache_atomic/class_index.json',
                        help='Path to FineBio atomic class index')
    parser.add_argument('--fps', type=float, default=10.0, help='Label FPS')
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    all_stats = {}
    
    # =========================================================================
    # 1. Combined PCR Dataset (5-class)
    # =========================================================================
    print("\n" + "="*70)
    print(" Processing Combined PCR Dataset (5-class)")
    print("="*70)
    
    combined_labels_dir = Path(args.combined_pcr_labels)
    if combined_labels_dir.exists():
        with open(args.combined_pcr_class_index) as f:
            combined_class_index = json.load(f)
        
        df_combined = compute_statistics(combined_labels_dir, combined_class_index, args.fps)
        all_stats['Combined PCR'] = df_combined
        
        # Save detailed stats
        df_combined.to_csv(output_dir / 'stats_combined_pcr.csv', index=False)
        
        # Print summary
        print_summary(df_combined, combined_class_index, 'Combined PCR')
        
        # Generate figures
        plot_video_duration_distribution(df_combined, output_dir, 'Combined PCR')
        plot_class_distribution(df_combined, combined_class_index, output_dir, 'Combined PCR')
        plot_segment_duration_boxplot(df_combined, combined_class_index, output_dir, 'Combined PCR')
        plot_source_comparison(df_combined, combined_class_index, output_dir, 'Combined PCR')
        plot_example_timelines(df_combined, combined_labels_dir, combined_class_index, output_dir, 'Combined PCR')
        
        # Also create separate stats for FineBio-only and MyData-only
        df_finebio_only = df_combined[df_combined['source'] == 'FineBio'].copy()
        df_mydata_only = df_combined[df_combined['source'] == 'MyData'].copy()
        
        if len(df_finebio_only) > 0:
            all_stats['FineBio PCR'] = df_finebio_only
            print_summary(df_finebio_only, combined_class_index, 'FineBio PCR Only')
            df_finebio_only.to_csv(output_dir / 'stats_finebio_pcr.csv', index=False)
        
        if len(df_mydata_only) > 0:
            all_stats['MyData'] = df_mydata_only
            print_summary(df_mydata_only, combined_class_index, 'MyData Only')
            df_mydata_only.to_csv(output_dir / 'stats_mydata.csv', index=False)
    else:
        print(f"⚠ Combined PCR labels not found at: {combined_labels_dir}")
    
    # =========================================================================
    # 2. FineBio Atomic Dataset (41-class)
    # =========================================================================
    print("\n" + "="*70)
    print(" Processing FineBio Atomic Dataset (41-class)")
    print("="*70)
    
    atomic_labels_dir = Path(args.finebio_atomic_labels)
    if atomic_labels_dir.exists():
        with open(args.finebio_atomic_class_index) as f:
            atomic_class_index = json.load(f)
        
        df_atomic = compute_statistics(atomic_labels_dir, atomic_class_index, args.fps)
        all_stats['FineBio Atomic'] = df_atomic
        
        # Save detailed stats
        df_atomic.to_csv(output_dir / 'stats_finebio_atomic.csv', index=False)
        
        # Print summary (simplified for 41 classes)
        print(f"\n📊 FineBio Atomic Summary:")
        print(f"   Total videos: {len(df_atomic)}")
        print(f"   Total duration: {df_atomic['duration_min'].sum():.1f} min")
        print(f"   Mean duration: {df_atomic['duration_sec'].mean():.1f} sec")
        print(f"   Total frames: {df_atomic['total_frames'].sum():,}")
        print(f"   Number of classes: {len(atomic_class_index)}")
        
        # Duration distribution
        plot_video_duration_distribution(df_atomic, output_dir, 'FineBio Atomic')
    else:
        print(f"⚠ FineBio atomic labels not found at: {atomic_labels_dir}")
    
    # =========================================================================
    # 3. Summary Tables
    # =========================================================================
    if all_stats:
        create_summary_table(all_stats, output_dir)
        
        # Create class coverage analysis (highlighting taxonomy mismatch)
        if 'Combined PCR' in all_stats or 'FineBio PCR' in all_stats:
            coverage_stats = {k: v for k, v in all_stats.items() 
                            if k in ['Combined PCR', 'FineBio PCR', 'MyData']}
            if coverage_stats and combined_labels_dir.exists():
                create_class_coverage_table(coverage_stats, combined_class_index, output_dir)
                plot_class_coverage_heatmap(coverage_stats, combined_class_index, output_dir)
    
    print("\n" + "="*70)
    print(f" ✓ All figures and CSVs saved to: {output_dir}")
    print("="*70)
    print("\nGenerated files:")
    for f in sorted(output_dir.glob('*')):
        print(f"  - {f.name}")


if __name__ == '__main__':
    main()

