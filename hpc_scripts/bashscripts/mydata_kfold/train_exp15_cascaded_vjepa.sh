#!/bin/bash
#BSUB -J exp15_vjepa[1-5]
#BSUB -q gpuv100
#BSUB -W 6:00
#BSUB -n 8
#BSUB -R "rusage[mem=24GB]"
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -o /zhome/45/8/156251/Desktop/thesis/labassist/hpc_scripts/logs/mydata_kfold/%J_%I_exp15_vjepa.out
#BSUB -e /zhome/45/8/156251/Desktop/thesis/labassist/hpc_scripts/logs/mydata_kfold/%J_%I_exp15_vjepa.err

# NOTE: Only submit this AFTER Stage 2 (finebio_only_stage2_vjepa) training completes!
# Remember to update the checkpoint path in exp15_cascaded_vjepa.yaml
# to point to the best epoch from Stage 2.

echo "=========================================="
echo "Stage 3 (V-JEPA): K-fold on mydata (Cascaded Approach)"
echo "Job ID: $LSB_JOBID"
echo "Array Index: $LSB_JOBINDEX"
echo "Host: $(hostname)"
echo "Date: $(date)"
echo "=========================================="

source /zhome/45/8/156251/Desktop/thesis/labassist/.env/bin/activate

TEMPLATE_CONFIG="/zhome/45/8/156251/Desktop/thesis/labassist/hpc_scripts/configs/mydata_kfold/exp15_cascaded_vjepa.yaml"
FOLD_IDX=$((LSB_JOBINDEX - 1))  # Convert 1-based LSB_JOBINDEX to 0-based fold index

echo "Training fold ${FOLD_IDX} with V-JEPA cascaded approach:"
echo "  Stage 1: [SKIPPED - no atomic pretraining for V-JEPA]"
echo "  Stage 2: FineBio-only PCR training from scratch (V-JEPA)"
echo "  Stage 3: This fold of mydata"
echo "  Features: V-JEPA (stride=1, selective crop, flip)"
echo ""

# Create temporary config with fold index substituted
FOLD_CONFIG="/tmp/exp15_cascaded_vjepa_fold${FOLD_IDX}_${LSB_JOBID}.yaml"
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

