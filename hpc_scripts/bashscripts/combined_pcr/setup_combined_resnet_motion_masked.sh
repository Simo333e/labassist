#!/bin/bash
# Setup combined_pcr dataset with ResNet+Motion (MASKED) features
# Creates symlinks to merge FineBio and mydata features

set -euo pipefail

echo "=== Setting up combined_pcr with ResNet+Motion (MASKED) features ==="

# Paths
FINEBIO_FEATURES=/work3/s204157/data/labassist/finebio_full/cache/features_10hz_resnet18_motion_masked
MYDATA_FEATURES=/work3/s204157/data/labassist/my_data/cache/features_10hz_resnet18_motion_masked_crop_flip
COMBINED_DIR=/work3/s204157/data/labassist/combined_pcr_resnet_motion_masked

# Create output directory
mkdir -p ${COMBINED_DIR}/features

echo "Linking FineBio features..."
# Link FineBio PCR videos (P*_05_*_T5 pattern)
for feat in ${FINEBIO_FEATURES}/P*_05_*_T5_x.npy; do
    if [ -f "$feat" ]; then
        ln -sf "$feat" ${COMBINED_DIR}/features/$(basename $feat)
        # Also link metadata
        meta="${feat%_x.npy}_meta.json"
        if [ -f "$meta" ]; then
            ln -sf "$meta" ${COMBINED_DIR}/features/$(basename $meta)
        fi
    fi
done

echo "Linking mydata features..."
# Link mydata features
for feat in ${MYDATA_FEATURES}/*_x.npy; do
    if [ -f "$feat" ]; then
        ln -sf "$feat" ${COMBINED_DIR}/features/$(basename $feat)
        # Also link metadata
        meta="${feat%_x.npy}_meta.json"
        if [ -f "$meta" ]; then
            ln -sf "$meta" ${COMBINED_DIR}/features/$(basename $meta)
        fi
    fi
done

# Count features
FINEBIO_COUNT=$(ls ${COMBINED_DIR}/features/P*_05_*_T5_x.npy 2>/dev/null | wc -l)
MYDATA_COUNT=$(ls ${COMBINED_DIR}/features/Lab_video*_x.npy 2>/dev/null | wc -l)
MYDATA_COUNT2=$(ls ${COMBINED_DIR}/features/202*_x.npy 2>/dev/null | wc -l)
TOTAL_MYDATA=$((MYDATA_COUNT + MYDATA_COUNT2))

echo ""
echo "=== Summary ==="
echo "FineBio PCR videos: ${FINEBIO_COUNT}"
echo "mydata videos: ${TOTAL_MYDATA}"
echo "Total: $((FINEBIO_COUNT + TOTAL_MYDATA))"
echo ""
echo "Combined features directory: ${COMBINED_DIR}/features"
echo "✓ Setup complete!"




