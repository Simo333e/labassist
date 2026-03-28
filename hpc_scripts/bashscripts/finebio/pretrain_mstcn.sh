#!/bin/sh
#BSUB -q gpuv100
#BSUB -J finebio_pretrain
#BSUB -gpu "num=1"
#BSUB -n 8
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=24GB]"
#BSUB -M 24GB
#BSUB -W 24:00
#BSUB -o /zhome/45/8/156251/Desktop/thesis/labassist/hpc_scripts/logs/finebio/%J_pretrain.out
#BSUB -e /zhome/45/8/156251/Desktop/thesis/labassist/hpc_scripts/logs/finebio/%J_pretrain.err

set -euo pipefail

# ============================================================
# Pretrain MS-TCN on FineBio Atomic Actions
# ============================================================
# This creates a pretrained model that can be finetuned on combined_pcr

CONFIG_PATH="/zhome/45/8/156251/Desktop/thesis/labassist/hpc_scripts/configs/finebio/finebio_atomic_pretrain.yaml"

mkdir -p /zhome/45/8/156251/Desktop/thesis/labassist/hpc_scripts/logs/finebio

# Activate environment
if [ -d "/zhome/45/8/156251/Desktop/thesis/labassist/.env" ]; then
  . "/zhome/45/8/156251/Desktop/thesis/labassist/.env/bin/activate"
fi

echo "============================================================"
echo "FineBio Atomic Action Pretraining"
echo "============================================================"
echo "Config: ${CONFIG_PATH}"
echo ""
echo "This model will be used for transfer learning to combined_pcr"
echo "============================================================"

python /zhome/45/8/156251/Desktop/thesis/labassist/hpc_scripts/scripts/finetune_mstcn.py \
    --config "${CONFIG_PATH}"

echo ""
echo "============================================================"
echo "✓ Pretraining complete"
echo "Model saved to: /work3/s204157/experiments/finebio_atomic_pretrain"
echo "============================================================"

