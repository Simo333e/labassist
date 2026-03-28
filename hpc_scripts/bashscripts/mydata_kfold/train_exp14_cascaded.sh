#!/bin/bash
#BSUB -J exp14_cascaded[1-5]
#BSUB -q gpuv100
#BSUB -W 6:00
#BSUB -n 8
#BSUB -R "rusage[mem=24GB]"
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -o /zhome/45/8/156251/Desktop/thesis/labassist/hpc_scripts/logs/mydata_kfold/%J_%I_exp14_cascaded.out
#BSUB -e /zhome/45/8/156251/Desktop/thesis/labassist/hpc_scripts/logs/mydata_kfold/%J_%I_exp14_cascaded.err

# NOTE: Only submit this AFTER Stage 2 (finebio_only_stage2) training completes!
# Remember to update the checkpoint path in exp14_cascaded_from_finebio.yaml
# to point to the best epoch from Stage 2.

echo "=========================================="
echo "Stage 3: K-fold on mydata (Cascaded Approach)"
echo "Job ID: $LSB_JOBID"
echo "Array Index: $LSB_JOBINDEX"
echo "Host: $(hostname)"
echo "Date: $(date)"
echo "=========================================="

source /zhome/45/8/156251/Desktop/thesis/labassist/.env/bin/activate

TEMPLATE_CONFIG="/zhome/45/8/156251/Desktop/thesis/labassist/hpc_scripts/configs/mydata_kfold/exp14_cascaded_from_finebio.yaml"
FOLD_IDX=$((LSB_JOBINDEX - 1))  # Convert 1-based LSB_JOBINDEX to 0-based fold index

echo "Training fold ${FOLD_IDX} with cascaded approach:"
echo "  Stage 1: Atomic actions pretrain (epoch-110)"
echo "  Stage 2: FineBio-only PCR training"
echo "  Stage 3: This fold of mydata"
echo ""

# Create temporary config with fold index substituted
FOLD_CONFIG="/tmp/exp14_cascaded_fold${FOLD_IDX}_${LSB_JOBID}.yaml"
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

