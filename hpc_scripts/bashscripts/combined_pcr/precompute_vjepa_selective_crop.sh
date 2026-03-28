#!/bin/sh
#BSUB -q gpuv100
#BSUB -J precompute_vjepa_selective_crop[1-4]
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -n 4
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=24GB]"
#BSUB -M 24GB
#BSUB -W 12:00
#BSUB -o /zhome/45/8/156251/Desktop/thesis/labassist/hpc_scripts/logs/combined_pcr/%J_%I_vjepa_selective.out
#BSUB -e /zhome/45/8/156251/Desktop/thesis/labassist/hpc_scripts/logs/combined_pcr/%J_%I_vjepa_selective.err

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

# Crop parameters (applied only to mydata, not FineBio)
CROP_TOP="0.0"
CROP_BOTTOM="0.35"
CROP_LEFT="0.10"
CROP_RIGHT="0.08"
# =====================

# Output to new directory for selective cropped features
FEATURES_OUT="${DATA_ROOT}/cache/features_${FPS}hz_vjepa_selective_crop"

mkdir -p "${FEATURES_OUT}"
mkdir -p /zhome/45/8/156251/Desktop/thesis/labassist/hpc_scripts/logs/combined_pcr

if [ -d "/zhome/45/8/156251/Desktop/thesis/labassist/.env" ]; then
  . "/zhome/45/8/156251/Desktop/thesis/labassist/.env/bin/activate"
fi

# Convert 1-based LSF index to 0-based shard index
SHARD_IDX=$(($LSB_JOBINDEX - 1))
NUM_SHARDS=4

echo "Running shard ${SHARD_IDX}/${NUM_SHARDS}"
echo "Selective cropping: mydata stems cropped (top=${CROP_TOP}, bottom=${CROP_BOTTOM}, left=${CROP_LEFT}, right=${CROP_RIGHT}), FineBio unchanged"

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
  --shard_idx "${SHARD_IDX}" \
  --crop_top "${CROP_TOP}" \
  --crop_bottom "${CROP_BOTTOM}" \
  --crop_left "${CROP_LEFT}" \
  --crop_right "${CROP_RIGHT}" \
  --selective_crop

