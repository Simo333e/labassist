#!/bin/sh
#BSUB -q gpua100
#BSUB -J precompute_vjepa_stride1[1-4]
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -n 4
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=24GB]"
#BSUB -M 24GB
#BSUB -W 12:00
#BSUB -o /zhome/45/8/156251/Desktop/thesis/labassist/hpc_scripts/logs/combined_pcr/%J_%I_vjepa_stride1.out
#BSUB -e /zhome/45/8/156251/Desktop/thesis/labassist/hpc_scripts/logs/combined_pcr/%J_%I_vjepa_stride1.err

set -euo pipefail

# Hugging Face cache on fast storage
export HF_HOME="/work3/s204157/hf_cache"
export TRANSFORMERS_CACHE="${HF_HOME}"

# === CONFIGURATION ===
DATA_ROOT="/work3/s204157/data/labassist/combined_pcr"
SPLITS="${DATA_ROOT}/cache/splits.json"
FPS="10"
FRAMES_PER_CLIP="16"
STRIDE="1"      # SHORT WINDOW: stride=1 means 16 frames @ 25fps = 0.64 sec (was stride=2 = 1.28 sec)
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

# Output to new directory for short-window features
FEATURES_OUT="${DATA_ROOT}/cache/features_${FPS}hz_vjepa_selective_crop_stride1"

mkdir -p "${FEATURES_OUT}"
mkdir -p /zhome/45/8/156251/Desktop/thesis/labassist/hpc_scripts/logs/combined_pcr

if [ -d "/zhome/45/8/156251/Desktop/thesis/labassist/.env" ]; then
  . "/zhome/45/8/156251/Desktop/thesis/labassist/.env/bin/activate"
fi

# Convert 1-based LSF index to 0-based shard index
SHARD_IDX=$(($LSB_JOBINDEX - 1))
NUM_SHARDS=4

echo "Running shard ${SHARD_IDX}/${NUM_SHARDS}"
echo "SHORT WINDOW: 16 frames @ stride=1 @ 25fps = 0.64 sec (better for short actions)"
echo "Selective cropping: mydata stems cropped, FineBio unchanged"

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







