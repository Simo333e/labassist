#!/bin/bash
#BSUB -J exp4_retrain[1-5]
#BSUB -q gpuv100
#BSUB -W 6:00
#BSUB -n 4
#BSUB -R "rusage[mem=8GB]"
#BSUB -R "span[hosts=1]"
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -o /zhome/45/8/156251/Desktop/thesis/labassist/hpc_scripts/logs/combined_pcr/%J_%I_exp4_retrain.out
#BSUB -e /zhome/45/8/156251/Desktop/thesis/labassist/hpc_scripts/logs/combined_pcr/%J_%I_exp4_retrain.err

###############################################################################
# Train exp4_combined_reg (K-fold) - RETRAIN with epoch-110
# This retrains using the newer epoch-110 FineBio checkpoint
###############################################################################

echo "========================================"
echo "Training exp4_combined_reg (Retrain)"
echo "Job ID: $LSB_JOBID"
echo "Array Index: $LSB_JOBINDEX"
echo "Host: $(hostname)"
echo "Date: $(date)"
echo "========================================"

source /zhome/45/8/156251/Desktop/thesis/labassist/.env/bin/activate
cd /zhome/45/8/156251/Desktop/thesis/labassist

# Convert 1-based LSF index to 0-based fold index
FOLD_IDX=$((LSB_JOBINDEX - 1))

CONFIG_TEMPLATE="/zhome/45/8/156251/Desktop/thesis/labassist/hpc_scripts/configs/mydata_kfold/exp4_combined_reg.yaml"

# Create temporary config with fold index substituted
TEMP_CONFIG="/tmp/exp4_retrain_fold${FOLD_IDX}_${LSB_JOBID}.yaml"
sed "s/\${FOLD_IDX}/${FOLD_IDX}/g" "${CONFIG_TEMPLATE}" > "${TEMP_CONFIG}"

# Update exp_dir to avoid overwriting original results
sed -i "s|/experiments/mydata_kfold_exp4_combined_reg/|/experiments/mydata_kfold_exp4_combined_reg_retrain/|g" "${TEMP_CONFIG}"
sed -i "s|exp4_combined_reg_fold|exp4_combined_reg_retrain_fold|g" "${TEMP_CONFIG}"

echo "Config: ${TEMP_CONFIG}"
echo "Fold: ${FOLD_IDX}"

python hpc_scripts/scripts/finetune_mstcn.py --config "${TEMP_CONFIG}"

rm -f "${TEMP_CONFIG}"

echo "========================================"
echo "Training complete for fold ${FOLD_IDX}!"
echo "Date: $(date)"
echo "========================================"

