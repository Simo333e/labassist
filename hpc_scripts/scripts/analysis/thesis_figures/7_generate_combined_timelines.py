#!/usr/bin/env python3
"""Combine all 6 fold timelines into a single figure with unified legend."""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from pathlib import Path
from matplotlib.patches import Patch
from matplotlib.colors import ListedColormap, BoundaryNorm

# Fold names mapping
FOLD_NAMES = {
    0: '20251027_181439Z',
    1: '20251112_110656Z_1080p30_v2',
    2: '20251120_065442Z',
    3: '20251125_070904Z',
    4: '20251125_072321Z',
    5: 'Lab_video_cut',
}

# Best epochs for each fold
BEST_EPOCHS = {
    0: 8,
    1: 18,
    2: 4,
    3: 6,
    4: 26,
    5: 4,
}

# Action colors (matching tab20 colormap from dataset_statistics.py)
ACTION_COLORS = {
    'none': '#1f77b4',
    'Pipette pcr tube': '#aec7e8',
    'Aspirate reagent': '#ff7f0e',
    'Attach pipette tip': '#ffbb78',
    'Eject pipette tip': '#2ca02c',
}

CLASS_NAMES = ['none', 'Pipette pcr tube', 'Aspirate reagent', 'Attach pipette tip', 'Eject pipette tip']

# Display names for legend
DISPLAY_NAMES = {
    'none': 'Background',
    'Pipette pcr tube': 'PCR Tube',
    'Aspirate reagent': 'Reagent',
    'Attach pipette tip': 'Tip On',
    'Eject pipette tip': 'Tip Off',
}

def main():
    base_dir = Path('/zhome/45/8/156251/Desktop/thesis/labassist/hpc_scripts/eval_data/cascade_v2/vjepa')
    output_dir = Path('/zhome/45/8/156251/Desktop/thesis/labassist/hpc_scripts/scripts/analysis/thesis_figures/output')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("\nCombining timelines from all 6 folds...")
    
    # Create figure with 6 rows + 1 for legend
    fig = plt.figure(figsize=(20, 20))
    
    # Create grid: 6 rows for timelines + 1 small row for legend
    gs = fig.add_gridspec(7, 1, height_ratios=[1, 1, 1, 1, 1, 1, 0.15], hspace=0.05)
    
    missing_folds = []
    
    for fold_idx in range(6):
        ax = fig.add_subplot(gs[fold_idx])
        video_name = FOLD_NAMES[fold_idx]
        epoch = BEST_EPOCHS[fold_idx]
        
        # Construct path to timeline image
        timeline_path = base_dir / f'fold{fold_idx}' / f'timelines_epoch{epoch}' / 'timelines' / f'{video_name}.png'
        
        if timeline_path.exists():
            # Load and display image
            img = mpimg.imread(str(timeline_path))
            ax.imshow(img)
            ax.axis('off')
            
            print(f"  ✓ Fold {fold_idx}: {video_name} (epoch {epoch})")
        else:
            # Display placeholder
            ax.text(0.5, 0.5, f'Fold {fold_idx}: Timeline not found\n{timeline_path}',
                   ha='center', va='center', fontsize=10, color='red')
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.axis('off')
            missing_folds.append(fold_idx)
            print(f"  ✗ Fold {fold_idx}: Timeline not found at {timeline_path}")
    
    # Add unified colorbar at the bottom
    cbar_ax = fig.add_subplot(gs[6])
    
    # Create colormap
    colors = [ACTION_COLORS.get(name, '#888888') for name in CLASS_NAMES]
    rgb_colors = []
    for c in colors:
        rgb = tuple(int(c.lstrip('#')[i:i+2], 16)/255.0 for i in (0, 2, 4))
        rgb_colors.append(rgb)
    cmap = ListedColormap(rgb_colors)
    norm = BoundaryNorm(np.arange(-0.5, len(CLASS_NAMES) + 0.5, 1.0), cmap.N)
    
    # Create colorbar
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cb = fig.colorbar(sm, cax=cbar_ax, orientation='horizontal')
    cb.set_ticks(range(len(CLASS_NAMES)))
    cb.set_ticklabels([DISPLAY_NAMES.get(c, c) for c in CLASS_NAMES])
    for lab in cbar_ax.get_xticklabels():
        lab.set_rotation(40)
        lab.set_ha('right')
        lab.set_fontsize(10)
    
    # Add title
    fig.suptitle('Action Segmentation Results: Leave-One-Out Cross-Validation ',
                 fontsize=16, fontweight='bold', y=0.995)
    
    # Save figure
    output_path = output_dir / 'fig7_combined_timelines_all_folds.png'
    fig.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    fig.savefig(output_path.with_suffix('.pdf'), bbox_inches='tight', facecolor='white')
    plt.close(fig)
    
    print(f"\n✓ Combined timeline figure saved to:")
    print(f"  - {output_path}")
    print(f"  - {output_path.with_suffix('.pdf')}")
    
    if missing_folds:
        print(f"\n⚠ Warning: Missing timelines for folds: {missing_folds}")
        print(f"   Regenerate them using: regenerate_all_timelines.sh")
    else:
        print(f"\n✅ All 6 fold timelines successfully combined!")

if __name__ == '__main__':
    main()
