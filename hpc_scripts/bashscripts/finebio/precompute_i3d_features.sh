#!/bin/sh
#BSUB -q gpua100
#BSUB -J finebio_i3d[1-8]
#BSUB -gpu "num=1"
#BSUB -n 4
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=32GB]"
#BSUB -M 32GB
#BSUB -W 12:00
#BSUB -o /zhome/45/8/156251/Desktop/thesis/labassist/hpc_scripts/logs/finebio/%J_%I_i3d.out
#BSUB -e /zhome/45/8/156251/Desktop/thesis/labassist/hpc_scripts/logs/finebio/%J_%I_i3d.err

set -euo pipefail

# ============================================================
# Extract I3D features for FineBio T5 videos
# ============================================================

DATA_ROOT="/work3/s204157/data/labassist/finebio_full"
SPLITS="${DATA_ROOT}/cache/splits_finebio_atomic.json"
I3D_WEIGHTS="/work3/s204157/models/i3d/rgb_imagenet.pt"

FPS="10"
WIN="16"
STRIDE="1"
IMGSZ="224"
BATCH_SIZE="8"
CENTER_STEP="1"

FEATURES_OUT="${DATA_ROOT}/cache/features_10hz_i3d_win16"

# Sharding
SHARD_IDX=$(($LSB_JOBINDEX - 1))
NUM_SHARDS=8

mkdir -p "${FEATURES_OUT}"
mkdir -p /zhome/45/8/156251/Desktop/thesis/labassist/hpc_scripts/logs/finebio

# Activate environment
if [ -d "/zhome/45/8/156251/Desktop/thesis/labassist/.env" ]; then
  . "/zhome/45/8/156251/Desktop/thesis/labassist/.env/bin/activate"
fi

echo "============================================================"
echo "FineBio I3D Feature Extraction - Shard ${SHARD_IDX}/${NUM_SHARDS}"
echo "============================================================"
echo "Data root: ${DATA_ROOT}"
echo "Splits: ${SPLITS}"
echo "Output: ${FEATURES_OUT}"
echo "Window: ${WIN} frames @ 25fps = $(echo "scale=2; ${WIN}/25" | bc) sec"
echo "============================================================"

python /zhome/45/8/156251/Desktop/thesis/labassist/hpc_scripts/scripts/feature_computations/precompute_features_i3d.py \
    --root "${DATA_ROOT}" \
    --splits "${SPLITS}" \
    --weights "${I3D_WEIGHTS}" \
    --fps "${FPS}" \
    --win "${WIN}" \
    --stride "${STRIDE}" \
    --imgsz "${IMGSZ}" \
    --batch_size "${BATCH_SIZE}" \
    --center_step "${CENTER_STEP}" \
    --device cuda:0 \
    --out "${FEATURES_OUT}" \
    --shards "${NUM_SHARDS}" \
    --shard_idx "${SHARD_IDX}"

echo ""
echo "✓ Shard ${SHARD_IDX} complete"

