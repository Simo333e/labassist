#!/bin/sh
#BSUB -q gpuv100
#BSUB -J imfe_atomic_pretrain
#BSUB -gpu "num=1"
#BSUB -n 4
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=32GB]"
#BSUB -M 32GB
#BSUB -W 24:00
#BSUB -o /zhome/45/8/156251/Desktop/thesis/labassist/hpc_scripts/logs/finebio/%J_imfe_atomic_pretrain.out
#BSUB -e /zhome/45/8/156251/Desktop/thesis/labassist/hpc_scripts/logs/finebio/%J_imfe_atomic_pretrain.err

set -euo pipefail

# Activate environment
source /zhome/45/8/156251/Desktop/thesis/labassist/.env/bin/activate

# Change to project root
cd /zhome/45/8/156251/Desktop/thesis/labassist

# Create log directory
mkdir -p /zhome/45/8/156251/Desktop/thesis/labassist/hpc_scripts/logs/finebio

echo "=========================================="
echo "IMFE Atomic Action Pretraining"
echo "Dataset: Full FineBio atomic (223 videos, 36 classes)"
echo "Features: IMFE (2048D)"
echo "Training: 150 epochs from scratch"
echo "Output: finebio_imfe_atomic_pretrain"
echo "=========================================="
echo ""

# Run training
python hpc_scripts/scripts/finetune_mstcn.py \
  --config hpc_scripts/configs/imfe_experiments/finebio_imfe_atomic_pretrain.yaml

echo ""
echo "=========================================="
echo "Atomic pretrain completed!"
echo "Next: Use best checkpoint for Stage 2 FineBio-only training"
echo "=========================================="


