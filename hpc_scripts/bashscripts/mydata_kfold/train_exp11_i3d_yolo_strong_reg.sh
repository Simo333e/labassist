#!/bin/bash
#BSUB -J mydata_exp11_yolo_reg[1-5]
#BSUB -q gpuv100
#BSUB -W 2:00
#BSUB -n 4
#BSUB -R "rusage[mem=8GB]"
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -o /zhome/45/8/156251/Desktop/thesis/labassist/hpc_scripts/logs/mydata_kfold/%J_%I_exp11_yolo_reg.out
#BSUB -e /zhome/45/8/156251/Desktop/thesis/labassist/hpc_scripts/logs/mydata_kfold/%J_%I_exp11_yolo_reg.err

###############################################################################
# Experiment 11: I3D+YOLO Fused with Strong Regularization
# 
# K-fold cross-validation (5 folds) on mydata subset
# Tests if YOLO features help when properly regularized
#
# Key regularization:
#   - HIGH dropout: 0.50 (was 0.25)
#   - STRONG smoothing: 0.30 (was 0.15)
#   - REDUCED capacity: num_R=3 (was 6)
###############################################################################

echo "========================================"
echo "Experiment 11: I3D+YOLO Strong Regularization"
echo "Job ID: $LSB_JOBID"
echo "Array Index: $LSB_JOBINDEX"
echo "Host: $(hostname)"
echo "Date: $(date)"
echo "========================================"

# Activate environment
source /zhome/45/8/156251/Desktop/thesis/labassist/.env/bin/activate

# Paths
SCRIPTS_DIR="/zhome/45/8/156251/Desktop/thesis/labassist/hpc_scripts/scripts"
CONFIG_DIR="/zhome/45/8/156251/Desktop/thesis/labassist/hpc_scripts/configs/mydata_kfold"
TEMPLATE_CONFIG="${CONFIG_DIR}/exp11_i3d_yolo_strong_reg.yaml"

# Convert 1-based LSF index to 0-based fold index
FOLD_IDX=$(($LSB_JOBINDEX - 1))

echo "Training fold ${FOLD_IDX}..."
echo "Template config: ${TEMPLATE_CONFIG}"

# Create temporary config with fold index substituted
FOLD_CONFIG="/tmp/exp11_i3d_yolo_strong_reg_fold${FOLD_IDX}_${LSB_JOBID}.yaml"
sed "s/\${FOLD_IDX}/${FOLD_IDX}/g" "${TEMPLATE_CONFIG}" > "${FOLD_CONFIG}"

echo "Generated config: ${FOLD_CONFIG}"
echo "----------------------------------------"
cat "${FOLD_CONFIG}"
echo "----------------------------------------"

# Verify data paths exist
XDIR="/work3/s204157/data/labassist/combined_pcr/cache/features_10hz_i3d_yolo_fused"
YDIR="/work3/s204157/data/labassist/combined_pcr/cache/labels_5class"
SPLITS="/work3/s204157/data/labassist/combined_pcr/cache/kfold_mydata/splits_fold${FOLD_IDX}.json"

echo "Checking data paths..."
echo "XDIR: ${XDIR}"
ls -la "${XDIR}" | head -5
echo "YDIR: ${YDIR}"
ls -la "${YDIR}" | head -5
echo "SPLITS: ${SPLITS}"
cat "${SPLITS}" | head -20

# Run training
echo ""
echo "Starting training..."
python "${SCRIPTS_DIR}/finetune_mstcn.py" --config "${FOLD_CONFIG}"

# Cleanup
rm -f "${FOLD_CONFIG}"

echo ""
echo "========================================"
echo "Fold ${FOLD_IDX} completed at $(date)"
echo "========================================"

