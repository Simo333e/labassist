#!/bin/bash
#BSUB -J stage2_resnet_masked
#BSUB -q gpuv100
#BSUB -W 12:00
#BSUB -n 8
#BSUB -R "rusage[mem=24GB]"
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -o /zhome/45/8/156251/Desktop/thesis/labassist/hpc_scripts/logs/combined_pcr/%J_finebio_stage2_resnet_masked.out
#BSUB -e /zhome/45/8/156251/Desktop/thesis/labassist/hpc_scripts/logs/combined_pcr/%J_finebio_stage2_resnet_masked.err

# Stage 2: Train on FineBio-only PCR videos with masked ResNet+Motion features
# This creates an intermediate checkpoint for mydata K-fold experiments

echo "=========================================="
echo "Stage 2: FineBio-only PCR Training (ResNet+Motion MASKED)"
echo "Job ID: $LSB_JOBID"
echo "Host: $(hostname)"
echo "Date: $(date)"
echo "=========================================="

source /zhome/45/8/156251/Desktop/thesis/labassist/.env/bin/activate

CONFIG="/zhome/45/8/156251/Desktop/thesis/labassist/hpc_scripts/configs/combined_pcr/finebio_only_stage2_resnet_motion_masked.yaml"

echo "Features: ResNet18+Motion with masking (threshold=15, kernel=5)"
echo "Training on: 42 FineBio PCR videos only"
echo "Starting from: Atomic pretrain (36 classes)"
echo ""

# Run training
python /zhome/45/8/156251/Desktop/thesis/labassist/hpc_scripts/scripts/finetune_mstcn.py \
  --config "${CONFIG}"

echo ""
echo "=========================================="
echo "Stage 2 training complete!"
echo "Next: Evaluate checkpoints to select best for Stage 3"
echo "=========================================="




