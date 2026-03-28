#!/bin/bash
# Update combined_pcr_fixed with Lab_video_cut features

set -euo pipefail

source /zhome/45/8/156251/Desktop/thesis/labassist/.env/bin/activate

SCRIPTS=/zhome/45/8/156251/Desktop/thesis/labassist/hpc_scripts/scripts
MYDATA=/work3/s204157/data/labassist/my_data
COMBINED=/work3/s204157/data/labassist/combined_pcr_fixed
FINEBIO_FEATURES=/work3/s204157/data/labassist/combined_pcr/cache/features_10hz_i3d_selective_crop_flip_win16

# Check if Lab_video_cut features exist
FEATURE_FILE=${MYDATA}/cache_fixed/features_10hz_i3d_vflip_crop_win16/Lab_video_cut_x.npy
if [ ! -f "$FEATURE_FILE" ]; then
    echo "ERROR: Lab_video_cut features not found at: $FEATURE_FILE"
    echo "Wait for feature extraction job to complete!"
    exit 1
fi

echo "=== Updating combined_pcr_fixed with Lab_video_cut ==="

# Symlink Lab_video_cut features
echo "Linking Lab_video_cut features..."
ln -sf ${MYDATA}/cache_fixed/features_10hz_i3d_vflip_crop_win16/Lab_video_cut_x.npy ${COMBINED}/features/
ln -sf ${MYDATA}/cache_fixed/features_10hz_i3d_vflip_crop_win16/Lab_video_cut_meta.json ${COMBINED}/features/ 2>/dev/null || true

# Regenerate splits
echo "Regenerating splits.json..."
python ${SCRIPTS}/data_prep/create_combined_splits_fixed.py \
    --finebio_features_dir ${FINEBIO_FEATURES} \
    --mydata_features_dir ${MYDATA}/cache_fixed/features_10hz_i3d_vflip_crop_win16 \
    --mydata_val_stems "20251125_070904Z" \
    --exclude_stems "20251120_065442Z" \
    --output_splits ${COMBINED}/splits.json

echo ""
echo "=== Done! ==="
echo "Feature count: $(ls -1 ${COMBINED}/features/*_x.npy 2>/dev/null | wc -l)"
echo ""
echo "Now submit training: bsub < /zhome/45/8/156251/Desktop/thesis/labassist/hpc_scripts/bashscripts/combined_pcr/finetune_fixed.sh"
