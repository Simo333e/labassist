#!/bin/bash
# MASTER SCRIPT: Fix mydata pipeline and prepare for finetuning
#
# This script runs ALL the steps needed to fix the label alignment issue:
# 1. Extract frames from ORIGINAL (untrimmed) videos
# 2. Convert Label Studio annotations (now aligned with original videos)
# 3. Remap to 5-class format
# 4. Compute I3D features with augmentations
# 5. Create combined splits for FineBio + fixed mydata
#
# Run this script interactively or submit individual steps as jobs.

set -euo pipefail

echo "========================================"
echo "FIXING MYDATA PIPELINE"
echo "========================================"
echo ""
echo "This fixes the label alignment issue caused by using trimmed videos."
echo "The original Label Studio annotations were made on the FULL videos."
echo ""

SCRIPTS=/zhome/45/8/156251/Desktop/thesis/labassist/hpc_scripts/scripts
BASHSCRIPTS=/zhome/45/8/156251/Desktop/thesis/labassist/hpc_scripts/bashscripts
MYDATA=/work3/s204157/data/labassist/my_data
COMBINED=/work3/s204157/data/labassist/combined_pcr_fixed

# ============================================================
# STEP 1: Extract frames from original videos
# ============================================================
echo "STEP 1: Submit frame extraction job"
echo "  bsub < ${BASHSCRIPTS}/mydata/extract_frames_original.sh"
echo ""

# ============================================================
# STEP 2: Convert labels (after frames are extracted)
# ============================================================
echo "STEP 2: Convert Label Studio annotations"
echo "  bash ${BASHSCRIPTS}/mydata/convert_labels_fixed.sh"
echo ""

# ============================================================
# STEP 3: Remap to 5-class format
# ============================================================
echo "STEP 3: Remap to 5-class format"
echo "  bash ${BASHSCRIPTS}/mydata/remap_to_5class.sh"
echo ""

# ============================================================
# STEP 4: Compute I3D features
# ============================================================
echo "STEP 4: Submit I3D feature extraction job"
echo "  bsub < ${BASHSCRIPTS}/mydata/precompute_i3d_fixed.sh"
echo ""

# ============================================================
# STEP 5: Create combined splits (FineBio + fixed mydata)
# ============================================================
echo "STEP 5: Create combined splits for finetuning"
echo "  python ${SCRIPTS}/data_prep/create_combined_splits_fixed.py \\"
echo "    --finebio_features_dir /work3/s204157/data/labassist/combined_pcr/cache/features_10hz_i3d_selective_crop_flip_win16 \\"
echo "    --mydata_features_dir ${MYDATA}/cache_fixed/features_10hz_i3d_vflip_crop_win16 \\"
echo "    --mydata_val_stems '20251125_070904Z' \\"
echo "    --exclude_stems '20251120_065442Z' \\"
echo "    --output_splits ${COMBINED}/splits.json"
echo ""

# ============================================================
# STEP 6: Finetune from FineBio pretrained model
# ============================================================
echo "STEP 6: Submit finetuning job"
echo "  bsub < ${BASHSCRIPTS}/combined_pcr/finetune_fixed.sh"
echo ""

echo "========================================"
echo "Run the steps in order, waiting for jobs to complete between steps."
echo "========================================"


