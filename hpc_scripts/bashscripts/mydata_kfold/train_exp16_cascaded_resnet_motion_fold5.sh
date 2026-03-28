#!/bin/bash
#BSUB -J exp16_resnet_fold5
#BSUB -q gpuv100
#BSUB -W 6:00
#BSUB -n 4
#BSUB -R "rusage[mem=16GB]"
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -o /zhome/45/8/156251/Desktop/thesis/labassist/hpc_scripts/logs/mydata_kfold/%J_exp16_resnet_fold5.out
#BSUB -e /zhome/45/8/156251/Desktop/thesis/labassist/hpc_scripts/logs/mydata_kfold/%J_exp16_resnet_fold5.err

echo "=========================================="
echo "Stage 3: K-fold fold 5 on mydata (ResNet+Motion)"
echo "Job ID: $LSB_JOBID"
echo "Host: $(hostname)"
echo "Date: $(date)"
echo "=========================================="

source /zhome/45/8/156251/Desktop/thesis/labassist/.env/bin/activate

TEMPLATE_CONFIG="/zhome/45/8/156251/Desktop/thesis/labassist/hpc_scripts/configs/mydata_kfold/exp16_cascaded_resnet_motion.yaml"
FOLD_IDX=5

echo "Training fold ${FOLD_IDX} with cascaded approach (ResNet+Motion)"
echo ""

# Create temporary config with fold index substituted
FOLD_CONFIG="/tmp/exp16_cascaded_resnet_motion_fold${FOLD_IDX}_${LSB_JOBID}.yaml"
sed "s/\${FOLD_IDX}/${FOLD_IDX}/g" "${TEMPLATE_CONFIG}" > "${FOLD_CONFIG}"

echo "Generated config: ${FOLD_CONFIG}"
echo ""

# Run training
python /zhome/45/8/156251/Desktop/thesis/labassist/hpc_scripts/scripts/finetune_mstcn.py \
  --config "${FOLD_CONFIG}"

# Cleanup
rm -f "${FOLD_CONFIG}"

echo ""
echo "=========================================="
echo "Fold ${FOLD_IDX} training complete!"
echo "=========================================="





