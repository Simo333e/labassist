#!/bin/bash
#BSUB -q gpuv100
#BSUB -J finebio_resnet_pretrain
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -n 8
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=24GB]"
#BSUB -W 24:00
#BSUB -o /zhome/45/8/156251/Desktop/thesis/labassist/hpc_scripts/logs/finebio/%J_resnet_pretrain.out
#BSUB -e /zhome/45/8/156251/Desktop/thesis/labassist/hpc_scripts/logs/finebio/%J_resnet_pretrain.err

# Pretrain MS-TCN on FineBio Atomic Actions using ResNet+Motion features

set -euo pipefail

# Activate environment
source /zhome/45/8/156251/Desktop/thesis/labassist/.env/bin/activate

CONFIG_PATH="/zhome/45/8/156251/Desktop/thesis/labassist/hpc_scripts/configs/resnet_experiments/finebio_resnet_motion_pretrain.yaml"

echo "============================================================"
echo "FineBio Atomic Action Pretraining (ResNet+Motion)"
echo "============================================================"
echo "Config: ${CONFIG_PATH}"
echo ""
echo "Features: ResNet18 + Motion (1024D)"
echo "This model will be used for transfer learning to combined_pcr"
echo "============================================================"

python /zhome/45/8/156251/Desktop/thesis/labassist/hpc_scripts/scripts/finetune_mstcn.py \
    --config "${CONFIG_PATH}"

echo ""
echo "============================================================"
echo "✓ Pretraining complete"
echo "Model saved to: /work3/s204157/experiments/finebio_resnet_motion_pretrain"
echo "============================================================"
