#!/bin/bash
#BSUB -J finetune_fixed
#BSUB -q gpuv100
#BSUB -n 8
#BSUB -R "rusage[mem=24G]"
#BSUB -R "span[hosts=1]"
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -W 12:00
#BSUB -o /zhome/45/8/156251/Desktop/thesis/labassist/hpc_scripts/logs/combined_pcr/%J_finetune_fixed.out
#BSUB -e /zhome/45/8/156251/Desktop/thesis/labassist/hpc_scripts/logs/combined_pcr/%J_finetune_fixed.err

# Finetune MS-TCN on combined_pcr with FIXED mydata labels
# This should finally work now that labels are properly aligned!

set -euo pipefail

# Activate environment
if [ -d "/zhome/45/8/156251/Desktop/thesis/labassist/.env" ]; then
  . "/zhome/45/8/156251/Desktop/thesis/labassist/.env/bin/activate"
fi

SCRIPTS=/zhome/45/8/156251/Desktop/thesis/labassist/hpc_scripts/scripts
CONFIG=/zhome/45/8/156251/Desktop/thesis/labassist/hpc_scripts/configs/I3d_experiments/combined_pcr_fixed_finetune.yaml

echo "========================================"
echo "FINETUNING WITH FIXED MYDATA LABELS"
echo "========================================"
echo ""
echo "Config: ${CONFIG}"
echo ""
echo "This uses:"
echo "  - FineBio pretrained model (epoch-50)"
echo "  - FIXED mydata labels (aligned with original videos)"
echo "  - FineBio + mydata combined training"
echo ""

python ${SCRIPTS}/finetune_mstcn.py --config ${CONFIG}

echo ""
echo "========================================"
echo "Training complete!"
echo "========================================"


