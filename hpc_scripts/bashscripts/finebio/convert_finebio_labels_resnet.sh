#!/bin/bash
# Convert FineBio annotations to MS-TCN format with atomic actions
# Run this BEFORE pretraining to generate labels

set -euo pipefail

source /zhome/45/8/156251/Desktop/thesis/labassist/.env/bin/activate

SCRIPTS=/zhome/45/8/156251/Desktop/thesis/labassist/hpc_scripts/scripts

# FineBio paths
FINEBIO_ANNOTATIONS=/work3/s204157/data/finebio/annotations/actions
FINEBIO_VIDEOS=/work3/s204157/data/labassist/finebio_videos_w640
OUT_DIR=/work3/s204157/data/labassist/finebio_full/cache_resnet

echo "=== Converting FineBio Annotations to MS-TCN Format ==="
echo "Annotations: ${FINEBIO_ANNOTATIONS}"
echo "Videos: ${FINEBIO_VIDEOS}"
echo "Output: ${OUT_DIR}"
echo ""

# Convert ALL tasks (not just 05) for pretraining on full FineBio
# This gives us more diverse atomic actions for better pretraining
python ${SCRIPTS}/data_prep/convert_finebio_to_mstcn.py \
    --annotations_dir ${FINEBIO_ANNOTATIONS} \
    --videos_dir ${FINEBIO_VIDEOS} \
    --out_dir ${OUT_DIR} \
    --target_fps 10.0 \
    --task_filter "" \
    --val_ratio 0.1 \
    --seed 42 \
    --min_action_frames 100

# Rename the splits file for clarity
if [ -f "${OUT_DIR}/splits.json" ]; then
    cp ${OUT_DIR}/splits.json ${OUT_DIR}/splits_finebio_atomic.json
fi

echo ""
echo "=== Done ==="
echo "Labels saved to: ${OUT_DIR}/labels"
echo "Class index: ${OUT_DIR}/class_index.json"
echo "Splits: ${OUT_DIR}/splits_finebio_atomic.json"
