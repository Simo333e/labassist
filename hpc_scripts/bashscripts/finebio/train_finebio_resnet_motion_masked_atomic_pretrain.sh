#!/bin/sh
#BSUB -q gpuv100
#BSUB -J resnet_motion_masked_atomic_pretrain
#BSUB -gpu "num=1"
#BSUB -n 4
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=32GB]"
#BSUB -M 32GB
#BSUB -W 24:00
#BSUB -o /zhome/45/8/156251/Desktop/thesis/labassist/hpc_scripts/logs/finebio/%J_resnet_motion_masked_atomic_pretrain.out
#BSUB -e /zhome/45/8/156251/Desktop/thesis/labassist/hpc_scripts/logs/finebio/%J_resnet_motion_masked_atomic_pretrain.err

set -euo pipefail

# Activate environment
source /zhome/45/8/156251/Desktop/thesis/labassist/.env/bin/activate

# Change to project root
cd /zhome/45/8/156251/Desktop/thesis/labassist

# Create log directory
mkdir -p /zhome/45/8/156251/Desktop/thesis/labassist/hpc_scripts/logs/finebio

echo "=========================================="
echo "ResNet+Motion (MASKED) Atomic Action Pretraining"
echo "Dataset: Full FineBio atomic (223 videos, 36 classes)"
echo "Features: ResNet18+Motion with masking (1024D)"
echo "Motion masking: threshold=15, kernel=5"
echo "Training: 150 epochs from scratch"
echo "Output: finebio_resnet_motion_masked_pretrain"
echo "=========================================="
echo ""

# Run training
python hpc_scripts/scripts/finetune_mstcn.py \
  --config hpc_scripts/configs/resnet_experiments/finebio_resnet_motion_masked_pretrain.yaml

echo ""
echo "=========================================="
echo "Atomic pretrain completed!"
echo "Next: Use best checkpoint for Stage 2 FineBio-only training"
echo "=========================================="




