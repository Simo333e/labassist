#!/bin/sh
#BSUB -q gpuv100              
#BSUB -J train_yolo_mydata
#BSUB -gpu "num=1"
#BSUB -n 8
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=32GB]"
#BSUB -M 32GB
#BSUB -W 12:00
#BSUB -o /zhome/45/8/156251/Desktop/thesis/labassist/hpc_scripts/logs/mydata/yolo/%J_yolo.out
#BSUB -e /zhome/45/8/156251/Desktop/thesis/labassist/hpc_scripts/logs/mydata/yolo/%J_yolo.err

set -euo pipefail

# === CONFIGURATION ===
DATASET_DIR="/zhome/45/8/156251/Desktop/thesis/labassist/DTULabAssistDataset"
MODEL="yolo11s.pt"                  # keep it small to avoid OOM
RUN_NAME="labassist_yolo11m2"
IMG_SIZE=1280                        
BATCH=8                             # fits comfortably on 16 GB GPUs
EPOCHS=100
WORKERS=8
# =====================

mkdir -p /zhome/45/8/156251/Desktop/thesis/labassist/hpc_scripts/logs/mydata

if [ -d "/zhome/45/8/156251/Desktop/thesis/labassist/.env" ]; then
  . "/zhome/45/8/156251/Desktop/thesis/labassist/.env/bin/activate"
fi

cd "${DATASET_DIR}"

# Optional: pin CUDA to avoid implicit fallback
export CUDA_VISIBLE_DEVICES=0
export TORCH_NCCL_BLOCKING_WAIT=1

yolo detect train \
  data=data.yaml \
  model=${MODEL} \
  imgsz=${IMG_SIZE} \
  epochs=${EPOCHS} \
  batch=${BATCH} \
  workers=${WORKERS} \
  project=runs/detect \
  name=${RUN_NAME} \
  cache=True \
  device=0 \
  optimizer=adamw \
  lr0=0.0006 \
  patience=20 \
  amp=True