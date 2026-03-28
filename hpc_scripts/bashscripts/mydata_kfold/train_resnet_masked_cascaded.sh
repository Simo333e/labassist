#!/bin/bash
#BSUB -J resnet_masked_cascade[0-4]
#BSUB -q gpuv100
#BSUB -W 6:00
#BSUB -n 4
#BSUB -R "rusage[mem=16GB]"
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -o /zhome/45/8/156251/Desktop/thesis/labassist/hpc_scripts/logs/mydata_kfold/%J_fold%I_resnet_masked_cascade.out
#BSUB -e /zhome/45/8/156251/Desktop/thesis/labassist/hpc_scripts/logs/mydata_kfold/%J_fold%I_resnet_masked_cascade.err

# Stage 3: K-fold cross-validation on mydata with ResNet+Motion (MASKED)
# Uses checkpoint from Stage 2 (FineBio-only training)
# Provides clean evaluation metrics for thesis

echo "=========================================="
echo "Stage 3: mydata K-fold (ResNet+Motion MASKED Cascade)"
echo "Job ID: $LSB_JOBID"
echo "Array Index: $LSB_JOBINDEX"
echo "Fold: $LSB_JOBINDEX"
echo "Host: $(hostname)"
echo "Date: $(date)"
echo "=========================================="

source /zhome/45/8/156251/Desktop/thesis/labassist/.env/bin/activate

# Set fold index from array
export FOLD_IDX=$LSB_JOBINDEX

CONFIG="/zhome/45/8/156251/Desktop/thesis/labassist/hpc_scripts/configs/mydata_kfold/exp_resnet_masked_cascaded.yaml"

echo "Features: ResNet18+Motion with masking"
echo "Fold: ${FOLD_IDX}/5"
echo "Starting from: Stage 2 FineBio-only checkpoint"
echo ""

# Run training
python /zhome/45/8/156251/Desktop/thesis/labassist/hpc_scripts/scripts/finetune_mstcn.py \
  --config "${CONFIG}"

echo ""
echo "=========================================="
echo "Fold ${FOLD_IDX} training complete!"
echo "=========================================="




