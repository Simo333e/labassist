#!/bin/bash
#BSUB -J exp18_7class[1-5]
#BSUB -q gpuv100
#BSUB -W 6:00
#BSUB -n 8
#BSUB -R "rusage[mem=24GB]"
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -o /zhome/45/8/156251/Desktop/thesis/labassist/hpc_scripts/logs/mydata_kfold/%J_%I_exp18_7class.out
#BSUB -e /zhome/45/8/156251/Desktop/thesis/labassist/hpc_scripts/logs/mydata_kfold/%J_%I_exp18_7class.err

# Experiment 18: Stage 3 - K-fold on mydata with 7-class taxonomy (I3D)
# NOTE: Only submit this AFTER Stage 2 (finebio_only_stage2_7class) completes!
# Remember to update the checkpoint path in exp18_cascaded_7class.yaml to point to best Stage 2 epoch.

echo "=========================================="
echo "Experiment 18 Stage 3: K-fold on mydata (7-class, I3D)"
echo "Job ID: $LSB_JOBID"
echo "Array Index: $LSB_JOBINDEX"
echo "Host: $(hostname)"
echo "Date: $(date)"
echo "=========================================="

source /zhome/45/8/156251/Desktop/thesis/labassist/.env/bin/activate

TEMPLATE_CONFIG="/zhome/45/8/156251/Desktop/thesis/labassist/hpc_scripts/configs/mydata_kfold/exp18_cascaded_7class.yaml"
FOLD_IDX=$((LSB_JOBINDEX - 1))  # Convert 1-based to 0-based

echo "Training fold ${FOLD_IDX} with 7-class cascaded approach:"
echo "  Stage 1: Atomic actions pretrain (epoch-110)"
echo "  Stage 2: FineBio-only PCR training (7-class)"
echo "  Stage 3: This fold of mydata (7-class)"
echo ""
echo "7-class taxonomy:"
echo "  0: none"
echo "  1: Adjust pipette volume (unique to mydata)"
echo "  2: Attach pipette tip"
echo "  3: Eject pipette tip"
echo "  4: Manipulate pcr tubes"
echo "  5: Manipulate reagents"
echo "  6: Switch pipette"
echo ""

# Create temporary config with fold index substituted
FOLD_CONFIG="/tmp/exp18_7class_fold${FOLD_IDX}_${LSB_JOBID}.yaml"
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




