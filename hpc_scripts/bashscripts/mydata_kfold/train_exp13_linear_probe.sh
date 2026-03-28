#!/bin/bash
#BSUB -J exp13_linear[1-5]
#BSUB -q gpua100
#BSUB -W 2:00
#BSUB -n 4
#BSUB -R "rusage[mem=8GB]"
#BSUB -R "span[hosts=1]"
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -o /zhome/45/8/156251/Desktop/thesis/labassist/hpc_scripts/logs/combined_pcr/%J_%I_exp13_linear_probe.out
#BSUB -e /zhome/45/8/156251/Desktop/thesis/labassist/hpc_scripts/logs/combined_pcr/%J_%I_exp13_linear_probe.err

###############################################################################
# Experiment 13: Linear Probing (K-fold)
# 
# Freezes all backbone layers, only trains conv_out classification heads.
# Very stable training, fast convergence, tests FineBio feature transfer.
###############################################################################

echo "========================================"
echo "Training exp13_linear_probe"
echo "Job ID: $LSB_JOBID"
echo "Array Index: $LSB_JOBINDEX"
echo "Host: $(hostname)"
echo "Date: $(date)"
echo "========================================"

source /zhome/45/8/156251/Desktop/thesis/labassist/.env/bin/activate
cd /zhome/45/8/156251/Desktop/thesis/labassist

# Convert 1-based LSF index to 0-based fold index
FOLD_IDX=$((LSB_JOBINDEX - 1))

CONFIG_TEMPLATE="/zhome/45/8/156251/Desktop/thesis/labassist/hpc_scripts/configs/mydata_kfold/exp13_linear_probe.yaml"

# Create temporary config with fold index substituted
TEMP_CONFIG="/tmp/exp13_linear_probe_fold${FOLD_IDX}_${LSB_JOBID}.yaml"
sed "s/\${FOLD_IDX}/${FOLD_IDX}/g" "${CONFIG_TEMPLATE}" > "${TEMP_CONFIG}"

echo "Config: ${TEMP_CONFIG}"
echo "Fold: ${FOLD_IDX}"

python hpc_scripts/scripts/finetune_mstcn.py --config "${TEMP_CONFIG}"

rm -f "${TEMP_CONFIG}"

echo "========================================"
echo "Training complete for fold ${FOLD_IDX}!"
echo "Date: $(date)"
echo "========================================"

