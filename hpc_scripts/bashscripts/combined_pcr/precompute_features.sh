#!/bin/sh
#BSUB -q gpua100
#BSUB -J precompute_combined_pcr_features
#BSUB -gpu "num=1"
#BSUB -n 4
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=24GB]"
#BSUB -M 24GB
#BSUB -W 06:00
#BSUB -o /zhome/45/8/156251/Desktop/thesis/labassist/hpc_scripts/logs/mydata/%J_features_combined.out
#BSUB -e /zhome/45/8/156251/Desktop/thesis/labassist/hpc_scripts/logs/mydata/%J_features_combined.err

set -euo pipefail

# === CONFIGURATION ===
DATA_ROOT="/work3/s204157/data/labassist/combined_pcr"
SPLITS="${DATA_ROOT}/cache/splits.json"
FPS="10"
CONTEXT_K="5"
CENTER_JITTER="${CENTER_JITTER:-0}"
# =====================

FEATURES_BASE="${DATA_ROOT}/cache/features_${FPS}hz_resnet18_motion"
FEATURES_CTX="${FEATURES_BASE}_ctx${CONTEXT_K}"

mkdir -p "${FEATURES_BASE}" "${FEATURES_CTX}"
mkdir -p /zhome/45/8/156251/Desktop/thesis/labassist/hpc_scripts/logs/mydata

if [ -d "/zhome/45/8/156251/Desktop/thesis/labassist/.env" ]; then
  . "/zhome/45/8/156251/Desktop/thesis/labassist/.env/bin/activate"
fi

python /zhome/45/8/156251/Desktop/thesis/labassist/hpc_scripts/scripts/feature_computations/precompute_features_resnet18_motion_generic.py \
  --root "${DATA_ROOT}" \
  --splits "${SPLITS}" \
  --fps "${FPS}" \
  --imgsz 224 \
  --batch_size 4 \
  --motion \
  --center_jitter "${CENTER_JITTER}" \
  --device cuda:0 \
  --out "${FEATURES_BASE}"

python /zhome/45/8/156251/Desktop/thesis/labassist/hpc_scripts/scripts/feature_computations/stack_context_features.py \
  --src "${FEATURES_BASE}" \
  --out "${FEATURES_CTX}" \
  --k "${CONTEXT_K}"


