#!/bin/bash
# Set up the combined_pcr_fixed directory with merged features and labels

set -euo pipefail

# Activate environment
if [ -d "/zhome/45/8/156251/Desktop/thesis/labassist/.env" ]; then
  . "/zhome/45/8/156251/Desktop/thesis/labassist/.env/bin/activate"
fi

SCRIPTS=/zhome/45/8/156251/Desktop/thesis/labassist/hpc_scripts/scripts

# Directories
FINEBIO_FEATURES=/work3/s204157/data/labassist/combined_pcr/cache/features_10hz_i3d_selective_crop_flip_win16
FINEBIO_LABELS=/work3/s204157/data/labassist/combined_pcr/cache/labels_5class
MYDATA=/work3/s204157/data/labassist/my_data
COMBINED=/work3/s204157/data/labassist/combined_pcr_fixed

echo "=== Setting up combined_pcr_fixed directory ==="

# Create combined directories
mkdir -p ${COMBINED}/features
mkdir -p ${COMBINED}/labels_5class

# Symlink FineBio features and labels
echo "Linking FineBio features..."
for f in ${FINEBIO_FEATURES}/P*_x.npy; do
    [ -e "$f" ] || continue
    ln -sf "$f" ${COMBINED}/features/
done
for f in ${FINEBIO_FEATURES}/P*_meta.json; do
    [ -e "$f" ] || continue
    ln -sf "$f" ${COMBINED}/features/
done

echo "Linking FineBio labels..."
for f in ${FINEBIO_LABELS}/P*_y.npy; do
    [ -e "$f" ] || continue
    ln -sf "$f" ${COMBINED}/labels_5class/
done

# Symlink FIXED mydata features and labels
echo "Linking FIXED mydata features..."
for f in ${MYDATA}/cache_fixed/features_10hz_i3d_vflip_crop_win16/*_x.npy; do
    [ -e "$f" ] || continue
    ln -sf "$f" ${COMBINED}/features/
done
for f in ${MYDATA}/cache_fixed/features_10hz_i3d_vflip_crop_win16/*_meta.json; do
    [ -e "$f" ] || continue
    ln -sf "$f" ${COMBINED}/features/
done

echo "Linking FIXED mydata labels..."
for f in ${MYDATA}/cache_fixed/labels_5class/*_y.npy; do
    [ -e "$f" ] || continue
    ln -sf "$f" ${COMBINED}/labels_5class/
done

# Copy class index
cp ${MYDATA}/cache_fixed/class_index_5class.json ${COMBINED}/class_index_5class.json 2>/dev/null || \
    cp /work3/s204157/data/labassist/combined_pcr/cache/class_index_5class.json ${COMBINED}/class_index_5class.json

# Create combined splits
echo ""
echo "Creating combined splits..."
python ${SCRIPTS}/data_prep/create_combined_splits_fixed.py \
    --finebio_features_dir ${FINEBIO_FEATURES} \
    --mydata_features_dir ${MYDATA}/cache_fixed/features_10hz_i3d_vflip_crop_win16 \
    --mydata_val_stems "20251125_070904Z" \
    --exclude_stems "20251120_065442Z" \
    --output_splits ${COMBINED}/splits.json

echo ""
echo "=== Setup complete ==="
echo "Combined directory: ${COMBINED}"
echo ""
echo "Contents:"
ls -la ${COMBINED}/
echo ""
echo "Feature count: $(ls -1 ${COMBINED}/features/*_x.npy 2>/dev/null | wc -l)"
echo "Label count: $(ls -1 ${COMBINED}/labels_5class/*_y.npy 2>/dev/null | wc -l)"


