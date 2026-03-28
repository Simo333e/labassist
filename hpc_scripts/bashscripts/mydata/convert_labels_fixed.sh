#!/bin/bash
# Convert Label Studio annotations using ORIGINAL (untrimmed) mydata videos
# This fixes the label alignment issue

set -euo pipefail

# Activate environment
if [ -d "/zhome/45/8/156251/Desktop/thesis/labassist/.env" ]; then
  . "/zhome/45/8/156251/Desktop/thesis/labassist/.env/bin/activate"
fi

SCRIPTS=/zhome/45/8/156251/Desktop/thesis/labassist/hpc_scripts/scripts
MYDATA=/work3/s204157/data/labassist/my_data

# Label Studio export
LABELSTUDIO_JSON=${MYDATA}/project-3-at-2025-11-26-19-24-89666430.json

# Use ORIGINAL videos (not _clean)
VIDEOS_DIR=${MYDATA}/videos

# Output to new cache directory (to avoid confusion)
OUTPUT_DIR=${MYDATA}/cache_fixed

echo "=== Converting Label Studio annotations with ORIGINAL videos ==="
echo "Label Studio JSON: ${LABELSTUDIO_JSON}"
echo "Videos: ${VIDEOS_DIR}"
echo "Output: ${OUTPUT_DIR}"
echo ""

python ${SCRIPTS}/data_prep/convert_labelstudio_to_mstcn.py \
    --labelstudio_json "${LABELSTUDIO_JSON}" \
    --videos_dir "${VIDEOS_DIR}" \
    --out_dir "${OUTPUT_DIR}" \
    --target_fps 10.0 \
    --val_ratio 0.17 \
    --seed 42 \
    --default_label "none"

echo ""
echo "=== Label conversion complete ==="
echo "Labels saved to: ${OUTPUT_DIR}/labels/"
echo "Splits saved to: ${OUTPUT_DIR}/splits.json"
echo "Class index: ${OUTPUT_DIR}/class_index.json"


