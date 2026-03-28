#!/bin/bash
# Setup combined_pcr dataset with ResNet+Motion features
# Combines FineBio PCR videos with mydata videos

set -euo pipefail

source /zhome/45/8/156251/Desktop/thesis/labassist/.env/bin/activate

SCRIPTS=/zhome/45/8/156251/Desktop/thesis/labassist/hpc_scripts/scripts

# Source directories
FINEBIO_FEATURES=/work3/s204157/data/labassist/finebio_full/cache/features_10hz_resnet18_motion
FINEBIO_LABELS=/work3/s204157/data/labassist/combined_pcr_fixed/labels_5class  # Reuse existing 5-class labels
MYDATA_FEATURES=/work3/s204157/data/labassist/my_data/cache_fixed/features_10hz_resnet18_motion_vflip_crop
MYDATA_LABELS=/work3/s204157/data/labassist/combined_pcr_fixed/labels_5class  # Already there from I3D setup

# Output directory
OUT_DIR=/work3/s204157/data/labassist/combined_pcr_resnet_motion
FEATURES_DIR=${OUT_DIR}/features
LABELS_DIR=${OUT_DIR}/labels_5class

echo "=== Setting up combined_pcr with ResNet+Motion features ==="
echo "FineBio features: ${FINEBIO_FEATURES}"
echo "Mydata features: ${MYDATA_FEATURES}"
echo "Output: ${OUT_DIR}"
echo ""

# Create directories
mkdir -p ${FEATURES_DIR}
mkdir -p ${LABELS_DIR}

# Symlink FineBio features (only PCR/task 05 videos with T5 view)
# Pattern: P{person}_05_{trial}_T5 for PCR videos
echo "=== Symlinking FineBio PCR features ==="
for f in ${FINEBIO_FEATURES}/P*_05_*_T5_x.npy; do
    if [ -f "$f" ]; then
        stem=$(basename "$f" _x.npy)
        ln -sf "$f" "${FEATURES_DIR}/${stem}_x.npy" 2>/dev/null || true
    fi
done

# Symlink mydata features
echo "=== Symlinking mydata features ==="
for f in ${MYDATA_FEATURES}/*_x.npy; do
    if [ -f "$f" ]; then
        stem=$(basename "$f" _x.npy)
        ln -sf "$f" "${FEATURES_DIR}/${stem}_x.npy" 2>/dev/null || true
    fi
done

# Symlink labels (use existing 5-class labels from combined_pcr_fixed)
echo "=== Symlinking labels ==="
for f in /work3/s204157/data/labassist/combined_pcr_fixed/labels_5class/*_y.npy; do
    if [ -f "$f" ]; then
        ln -sf "$f" "${LABELS_DIR}/" 2>/dev/null || true
    fi
done

# Copy class index
cp /work3/s204157/data/labassist/combined_pcr_fixed/class_index_5class.json ${OUT_DIR}/

# Create combined splits (excluding broken pipette, 1 mydata in val)
echo ""
echo "=== Creating combined splits ==="
python ${SCRIPTS}/data_prep/create_combined_splits_fixed.py \
    --finebio_features_dir ${FEATURES_DIR} \
    --mydata_features_dir ${MYDATA_FEATURES} \
    --mydata_val_stems "20251125_070904Z" \
    --exclude_stems "20251120_065442Z" \
    --output_splits ${OUT_DIR}/splits.json

# Count features and labels
echo ""
echo "=== Summary ==="
echo "Features: $(ls -1 ${FEATURES_DIR}/*_x.npy 2>/dev/null | wc -l)"
echo "Labels: $(ls -1 ${LABELS_DIR}/*_y.npy 2>/dev/null | wc -l)"

echo ""
echo "=== Done ==="
echo "Dataset ready at: ${OUT_DIR}"
echo "Next: Run finetuning with configs/resnet_experiments/combined_pcr_resnet_motion_finetune.yaml"
