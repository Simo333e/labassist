#!/bin/sh
#BSUB -q gpua100
#BSUB -J mydata_exp7_vjepa_combined[1-4]
#BSUB -gpu "num=1"
#BSUB -n 4
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=16GB]"
#BSUB -M 16GB
#BSUB -W 6:00
#BSUB -o /zhome/45/8/156251/Desktop/thesis/labassist/hpc_scripts/logs/mydata_kfold/%J_%I_exp7_vjepa_combined.out
#BSUB -e /zhome/45/8/156251/Desktop/thesis/labassist/hpc_scripts/logs/mydata_kfold/%J_%I_exp7_vjepa_combined.err

set -euo pipefail

DATA_ROOT="/work3/s204157/data/labassist/combined_pcr"
KFOLD_DIR="${DATA_ROOT}/cache/kfold_mydata"
CONFIG_PATH="/zhome/45/8/156251/Desktop/thesis/labassist/hpc_scripts/configs/mydata_kfold/exp7_vjepa_combined_reg.yaml"
SCRIPTS_DIR="/zhome/45/8/156251/Desktop/thesis/labassist/hpc_scripts/scripts"
LOGS_DIR="/zhome/45/8/156251/Desktop/thesis/labassist/hpc_scripts/logs/mydata_kfold"

# LSB_JOBINDEX is 1-based for this job array
FOLD_IDX=${LSB_JOBINDEX}

mkdir -p "${LOGS_DIR}"
mkdir -p "${KFOLD_DIR}"

if [ -d "/zhome/45/8/156251/Desktop/thesis/labassist/.env" ]; then
  . "/zhome/45/8/156251/Desktop/thesis/labassist/.env/bin/activate"
fi

echo "============================================================"
echo "Experiment 7: V-JEPA + Combined Regularization - Fold ${FOLD_IDX}"
echo "============================================================"
echo "  num_R: 3 (reduced capacity)"
echo "  feature_dropout: 0.5 (high)"
echo "  smoothing_lambda: 0.3 (strong)"
echo "============================================================"

if [ ! -f "${KFOLD_DIR}/splits_fold${FOLD_IDX}.json" ]; then
    echo "ERROR: Fold split file not found: ${KFOLD_DIR}/splits_fold${FOLD_IDX}.json"
    exit 1
fi

# Create config for this fold by substituting FOLD_IDX
FOLD_CONFIG="/tmp/exp7_vjepa_combined_fold${FOLD_IDX}_${LSB_JOBID}.yaml"
sed "s/\${FOLD_IDX}/${FOLD_IDX}/g" "${CONFIG_PATH}" > "${FOLD_CONFIG}"

echo "Using config: ${FOLD_CONFIG}"
echo "Splits file: ${KFOLD_DIR}/splits_fold${FOLD_IDX}.json"
echo "V-JEPA features: ${DATA_ROOT}/cache/features_10hz_vjepa_selective_crop_flip_stride1"

echo ""
echo "Fold ${FOLD_IDX} validation video(s):"
python -c "import json; splits=json.load(open('${KFOLD_DIR}/splits_fold${FOLD_IDX}.json')); print('  Val:', splits['val']); print('  Train:', splits['train'])"
echo ""

echo "Starting V-JEPA + Combined Reg training for fold ${FOLD_IDX}..."
python "${SCRIPTS_DIR}/finetune_mstcn.py" --config "${FOLD_CONFIG}"

rm -f "${FOLD_CONFIG}"

echo ""
echo "============================================================"
echo "✓ Fold ${FOLD_IDX} V-JEPA + Combined Reg training complete"
echo "============================================================"


