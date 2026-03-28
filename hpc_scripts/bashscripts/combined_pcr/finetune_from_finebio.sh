#!/bin/sh
#BSUB -q gpuv100
#BSUB -J finetune_from_finebio
#BSUB -gpu "num=1"
#BSUB -n 8
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=24GB]"
#BSUB -M 24GB
#BSUB -W 12:00
#BSUB -o /zhome/45/8/156251/Desktop/thesis/labassist/hpc_scripts/logs/combined_pcr/%J_finetune_from_finebio.out
#BSUB -e /zhome/45/8/156251/Desktop/thesis/labassist/hpc_scripts/logs/combined_pcr/%J_finetune_from_finebio.err

set -euo pipefail

# ============================================================
# Finetune MS-TCN on combined_pcr using FineBio pretrained weights
# ============================================================
# Transfer learning approach:
# 1. Pretrain on full FineBio atomic actions (221 videos)
# 2. Finetune on combined_pcr (48 videos, 5 classes)
# ============================================================

CONFIG_PATH="/zhome/45/8/156251/Desktop/thesis/labassist/hpc_scripts/configs/I3d_experiments/combined_pcr_finetuned_from_finebio.yaml"

mkdir -p /zhome/45/8/156251/Desktop/thesis/labassist/hpc_scripts/logs/combined_pcr

# Activate environment
if [ -d "/zhome/45/8/156251/Desktop/thesis/labassist/.env" ]; then
  . "/zhome/45/8/156251/Desktop/thesis/labassist/.env/bin/activate"
fi

echo "============================================================"
echo "Finetuning MS-TCN on combined_pcr"
echo "============================================================"
echo "Transfer learning: FineBio atomic → combined_pcr 5-class"
echo ""
echo "Pretrained: /work3/s204157/experiments/finebio_atomic_pretrain"
echo "Config: ${CONFIG_PATH}"
echo "============================================================"

python /zhome/45/8/156251/Desktop/thesis/labassist/hpc_scripts/scripts/finetune_mstcn.py \
    --config "${CONFIG_PATH}"

echo ""
echo "============================================================"
echo "✓ Finetuning complete"
echo "============================================================"

