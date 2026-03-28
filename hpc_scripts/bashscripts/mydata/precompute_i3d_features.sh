#!/bin/sh
#BSUB -q gpuv100
#BSUB -J precompute_mydata_i3d_10hz
#BSUB -gpu "num=1"
#BSUB -n 4
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=24GB]"
#BSUB -M 24GB
#BSUB -W 08:00
#BSUB -o /zhome/45/8/156251/Desktop/thesis/labassist/hpc_scripts/logs/mydata/%J_i3d_features.out
#BSUB -e /zhome/45/8/156251/Desktop/thesis/labassist/hpc_scripts/logs/mydata/%J_i3d_features.err

set -euo pipefail

# === CONFIGURATION ===
DATA_ROOT="/work3/s204157/data/labassist/my_data"
SPLITS="${DATA_ROOT}/cache/splits.json"
FPS="10"
VIEW="T1"
WIN="32"
STRIDE="2"
IMGSZ="160"
BATCH_SIZE="16"
CENTER_STEP="1"
# =====================

mkdir -p "${DATA_ROOT}/cache"
mkdir -p /zhome/45/8/156251/Desktop/thesis/labassist/hpc_scripts/logs/mydata

# Activate environment
if [ -d "/zhome/45/8/156251/Desktop/thesis/labassist/.env" ]; then
  . "/zhome/45/8/156251/Desktop/thesis/labassist/.env/bin/activate"
fi

python /zhome/45/8/156251/Desktop/thesis/labassist/hpc_scripts/scripts/feature_computations/precompute_features_i3d.py \
  --root "${DATA_ROOT}" \
  --splits "${SPLITS}" \
  --view "${VIEW}" \
  --fps "${FPS}" \
  --win "${WIN}" \
  --stride "${STRIDE}" \
  --imgsz "${IMGSZ}" \
  --batch_size "${BATCH_SIZE}" \
  --center_step "${CENTER_STEP}" \
  --device cuda:0 \
  --out "${DATA_ROOT}/cache/features_10hz_i3d"



