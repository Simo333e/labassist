#!/bin/bash
#BSUB -J exp12_freeze[1-5]
#BSUB -q gpuv100
#BSUB -W 4:00
#BSUB -n 4
#BSUB -R "rusage[mem=8GB]"
#BSUB -R "span[hosts=1]"
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -o /zhome/45/8/156251/Desktop/thesis/labassist/hpc_scripts/logs/combined_pcr/%J_%I_exp12_freeze_pg.out
#BSUB -e /zhome/45/8/156251/Desktop/thesis/labassist/hpc_scripts/logs/combined_pcr/%J_%I_exp12_freeze_pg.err

###############################################################################
# Experiment 12: Freeze PG, Train Rs Only (K-fold)
# 
# Freezes the Prediction Generator to preserve FineBio temporal features
# while training Refinement stages for the new action classes.
###############################################################################

echo "========================================"
echo "Training exp12_freeze_pg"
echo "Job ID: $LSB_JOBID"
echo "Array Index: $LSB_JOBINDEX"
echo "Host: $(hostname)"
echo "Date: $(date)"
echo "========================================"

source /zhome/45/8/156251/Desktop/thesis/labassist/.env/bin/activate
cd /zhome/45/8/156251/Desktop/thesis/labassist

# Convert 1-based LSF index to 0-based fold index
FOLD_IDX=$((LSB_JOBINDEX - 1))

CONFIG_TEMPLATE="/zhome/45/8/156251/Desktop/thesis/labassist/hpc_scripts/configs/mydata_kfold/exp12_freeze_pg.yaml"

# Create temporary config with fold index substituted
TEMP_CONFIG="/tmp/exp12_freeze_pg_fold${FOLD_IDX}_${LSB_JOBID}.yaml"
sed "s/\${FOLD_IDX}/${FOLD_IDX}/g" "${CONFIG_TEMPLATE}" > "${TEMP_CONFIG}"

echo "Config: ${TEMP_CONFIG}"
echo "Fold: ${FOLD_IDX}"

python hpc_scripts/scripts/finetune_mstcn.py --config "${TEMP_CONFIG}"

rm -f "${TEMP_CONFIG}"

echo "========================================"
echo "Training complete for fold ${FOLD_IDX}!"
echo "Date: $(date)"
echo "========================================"

