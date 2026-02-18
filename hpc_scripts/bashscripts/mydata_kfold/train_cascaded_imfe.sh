#!/bin/sh
#BSUB -q gpuv100
#BSUB -J mydata_exp17_imfe[1-5]
#BSUB -gpu "num=1"
#BSUB -n 4
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=24GB]"
#BSUB -M 24GB
#BSUB -W 12:00
#BSUB -o /zhome/45/8/156251/Desktop/thesis/labassist/hpc_scripts/logs/mydata_kfold/%J_%I_exp17_cascaded_imfe.out
#BSUB -e /zhome/45/8/156251/Desktop/thesis/labassist/hpc_scripts/logs/mydata_kfold/%J_%I_exp17_cascaded_imfe.err

set -euo pipefail

# Activate environment
source /zhome/45/8/156251/Desktop/thesis/labassist/.env/bin/activate

# Change to project root
cd /zhome/45/8/156251/Desktop/thesis/labassist

# Create log directory
mkdir -p /zhome/45/8/156251/Desktop/thesis/labassist/hpc_scripts/logs/mydata_kfold

# Convert LSB_JOBINDEX (1-5) to FOLD_IDX (0-4)
FOLD_IDX=$((LSB_JOBINDEX - 1))

echo "=========================================="
echo "Cascade Training Stage 3: K-Fold Validation"
echo "Experiment: exp17_cascaded_imfe"
echo "Fold: ${FOLD_IDX}"
echo "Features: IMFE (2048D)"
echo "Starting from: finebio_only_stage2_imfe"
echo "=========================================="
echo ""

# Create temporary config with substituted FOLD_IDX
TEMPLATE_CONFIG="/zhome/45/8/156251/Desktop/thesis/labassist/hpc_scripts/configs/mydata_kfold/exp17_cascaded_imfe.yaml"
FOLD_CONFIG="/tmp/exp17_cascaded_imfe_fold${FOLD_IDX}_${LSB_JOBID}.yaml"

# Substitute ${FOLD_IDX} in the template
sed "s/\${FOLD_IDX}/${FOLD_IDX}/g" "${TEMPLATE_CONFIG}" > "${FOLD_CONFIG}"

echo "Using config: ${FOLD_CONFIG}"
echo ""

# Run training
python hpc_scripts/scripts/finetune_mstcn.py \
  --config "${FOLD_CONFIG}"

# Cleanup temporary config
rm -f "${FOLD_CONFIG}"

echo ""
echo "=========================================="
echo "Fold ${FOLD_IDX} training completed!"
echo "=========================================="

