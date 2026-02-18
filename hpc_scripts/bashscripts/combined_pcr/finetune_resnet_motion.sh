#!/bin/bash
#BSUB -J finetune_resnet_motion
#BSUB -q gpuv100
#BSUB -n 8
#BSUB -R "rusage[mem=24G]"
#BSUB -R "span[hosts=1]"
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -W 12:00
#BSUB -o /zhome/45/8/156251/Desktop/thesis/labassist/hpc_scripts/logs/combined_pcr/%J_finetune_resnet_motion.out
#BSUB -e /zhome/45/8/156251/Desktop/thesis/labassist/hpc_scripts/logs/combined_pcr/%J_finetune_resnet_motion.err

# Finetune MS-TCN on combined_pcr using ResNet+Motion features
# This uses a pretrained model from FineBio atomic action training

set -euo pipefail

# Activate environment
source /zhome/45/8/156251/Desktop/thesis/labassist/.env/bin/activate

SCRIPTS=/zhome/45/8/156251/Desktop/thesis/labassist/hpc_scripts/scripts
CONFIG=/zhome/45/8/156251/Desktop/thesis/labassist/hpc_scripts/configs/resnet_experiments/combined_pcr_resnet_motion_finetune.yaml

echo "========================================"
echo "FINETUNING WITH RESNET+MOTION FEATURES"
echo "========================================"
echo ""
echo "Config: ${CONFIG}"
echo ""
echo "This uses:"
echo "  - FineBio pretrained model (ResNet+Motion, epoch-50)"
echo "  - FIXED mydata labels (aligned with original videos)"
echo "  - FineBio + mydata combined training"
echo "  - Vertical flip + crop for mydata alignment"
echo ""

python ${SCRIPTS}/finetune_mstcn.py --config ${CONFIG}

echo ""
echo "========================================"
echo "Finetuning complete!"
echo "========================================"
