#!/bin/sh
#BSUB -q hpc
#BSUB -J convert_labelstudio
#BSUB -n 1
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=4GB]"
#BSUB -M 4GB
#BSUB -W 00:30
#BSUB -o /zhome/45/8/156251/Desktop/thesis/labassist/hpc_scripts/logs/mydata/%J_convert.out
#BSUB -e /zhome/45/8/156251/Desktop/thesis/labassist/hpc_scripts/logs/mydata/%J_convert.err

set -euo pipefail

# === CONFIGURATION ===
DATA_ROOT="/work3/s204157/data/labassist/my_data"
LABELSTUDIO_JSON="${DATA_ROOT}/project-3-at-2025-11-26-19-24-89666430.json"
VIDEOS_DIR="${DATA_ROOT}/videos"
OUT_DIR="${DATA_ROOT}/cache"
VAL_RATIO="0.17"  # ~1 video for val with 6 total
TARGET_FPS="10"
# =====================

# Create output directories
mkdir -p "${OUT_DIR}/labels"
mkdir -p /zhome/45/8/156251/Desktop/thesis/labassist/hpc_scripts/logs/mydata

# Activate environment
if [ -d "/zhome/45/8/156251/Desktop/thesis/labassist/.env" ]; then
  . "/zhome/45/8/156251/Desktop/thesis/labassist/.env/bin/activate"
fi

python /zhome/45/8/156251/Desktop/thesis/labassist/hpc_scripts/scripts/data_prep/convert_labelstudio_to_mstcn.py \
  --labelstudio_json "${LABELSTUDIO_JSON}" \
  --videos_dir "${VIDEOS_DIR}" \
  --out_dir "${OUT_DIR}" \
  --target_fps "${TARGET_FPS}" \
  --val_ratio "${VAL_RATIO}" \
  --default_label "none" \
  --min_frames_per_class 500


