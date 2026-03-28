#!/bin/sh
#BSUB -q gpua100
#BSUB -J precompute_combined_pcr_vjepa[1-4]
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -n 4
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=24GB]"
#BSUB -M 24GB
#BSUB -W 12:00
#BSUB -o /zhome/45/8/156251/Desktop/thesis/labassist/hpc_scripts/logs/mydata/%J_%I_features_vjepa.out
#BSUB -e /zhome/45/8/156251/Desktop/thesis/labassist/hpc_scripts/logs/mydata/%J_%I_features_vjepa.err

set -euo pipefail

# Hugging Face cache on fast storage
export HF_HOME="/work3/s204157/hf_cache"
export TRANSFORMERS_CACHE="${HF_HOME}"

# === CONFIGURATION ===
DATA_ROOT="/work3/s204157/data/labassist/combined_pcr"
SPLITS="${DATA_ROOT}/cache/splits.json"
FPS="10"
FRAMES_PER_CLIP="16"
STRIDE="2"
BATCH_SIZE="1"
MODEL_ID="facebook/vjepa2-vitl-fpc64-256"

# Pre-extracted frames directory (much faster than video decoding)
FRAMES_DIR="${DATA_ROOT}/frames"
FRAMES_FPS="25"  # FPS the frames were extracted at
# =====================

FEATURES_OUT="${DATA_ROOT}/cache/features_${FPS}hz_vjepa"

mkdir -p "${FEATURES_OUT}"
mkdir -p /zhome/45/8/156251/Desktop/thesis/labassist/hpc_scripts/logs/mydata

if [ -d "/zhome/45/8/156251/Desktop/thesis/labassist/.env" ]; then
  . "/zhome/45/8/156251/Desktop/thesis/labassist/.env/bin/activate"
fi

# Convert 1-based LSF index to 0-based shard index
SHARD_IDX=$(($LSB_JOBINDEX - 1))
NUM_SHARDS=4

echo "Running shard ${SHARD_IDX}/${NUM_SHARDS}"

python /zhome/45/8/156251/Desktop/thesis/labassist/hpc_scripts/scripts/feature_computations/precompute_features_vjepa.py \
  --root "${DATA_ROOT}" \
  --splits "${SPLITS}" \
  --fps "${FPS}" \
  --frames_per_clip "${FRAMES_PER_CLIP}" \
  --stride "${STRIDE}" \
  --batch_size "${BATCH_SIZE}" \
  --device cuda:0 \
  --out "${FEATURES_OUT}" \
  --model_id "${MODEL_ID}" \
  --frames_dir "${FRAMES_DIR}" \
  --frames_fps "${FRAMES_FPS}" \
  --shards "${NUM_SHARDS}" \
  --shard_idx "${SHARD_IDX}"