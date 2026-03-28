#!/bin/sh
#BSUB -q gpuv100
#BSUB -J finebio_stage2_resnet_motion
#BSUB -gpu "num=1"
#BSUB -n 4
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=24GB]"
#BSUB -M 24GB
#BSUB -W 24:00
#BSUB -o /zhome/45/8/156251/Desktop/thesis/labassist/hpc_scripts/logs/combined_pcr/%J_finebio_stage2_resnet_motion.out
#BSUB -e /zhome/45/8/156251/Desktop/thesis/labassist/hpc_scripts/logs/combined_pcr/%J_finebio_stage2_resnet_motion.err

set -euo pipefail

# Activate environment
source /zhome/45/8/156251/Desktop/thesis/labassist/.env/bin/activate

# Change to project root
cd /zhome/45/8/156251/Desktop/thesis/labassist

# Create log directory
mkdir -p /zhome/45/8/156251/Desktop/thesis/labassist/hpc_scripts/logs/combined_pcr

echo "=========================================="
echo "Cascade Training Stage 2: FineBio-only with ResNet+Motion"
echo "Starting from: finebio_resnet_motion_pretrain/epoch-50"
echo "Training on: FineBio PCR videos only (42 videos)"
echo "Output: finebio_only_stage2_resnet_motion"
echo "=========================================="
echo ""

# Run training
python hpc_scripts/scripts/finetune_mstcn.py \
  --config hpc_scripts/configs/combined_pcr/finebio_only_stage2_resnet_motion.yaml

echo ""
echo "=========================================="
echo "Stage 2 training completed!"
echo "Next: Evaluate checkpoints and pick best for Stage 3 K-fold"
echo "=========================================="

