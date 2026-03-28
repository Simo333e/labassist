#!/bin/sh
#BSUB -q gpua100
#BSUB -J mydata_exp3_strong_smoothing[1-4]
#BSUB -gpu "num=1"
#BSUB -n 4
#BSUB -R "span[hosts=1]"    
#BSUB -R "rusage[mem=16GB]"
#BSUB -M 16GB
#BSUB -W 6:00
#BSUB -o /zhome/45/8/156251/Desktop/thesis/labassist/hpc_scripts/logs/mydata_kfold/%J_%I_exp3_strong_smoothing.out
#BSUB -e /zhome/45/8/156251/Desktop/thesis/labassist/hpc_scripts/logs/mydata_kfold/%J_%I_exp3_strong_smoothing.err

set -euo pipefail

# ============================================================
# Experiment 3: Strong Smoothing (dropout=0.25, smoothing=0.30, num_R=6)
# ============================================================

# Paths
DATA_ROOT="/work3/s204157/data/labassist/combined_pcr"
KFOLD_DIR="${DATA_ROOT}/cache/kfold_mydata"
TEMPLATE_CONFIG="/zhome/45/8/156251/Desktop/thesis/labassist/hpc_scripts/configs/mydata_kfold/exp3_strong_smoothing.yaml"
SCRIPTS_DIR="/zhome/45/8/156251/Desktop/thesis/labassist/hpc_scripts/scripts"
LOGS_DIR="/zhome/45/8/156251/Desktop/thesis/labassist/hpc_scripts/logs/mydata_kfold"

FOLD_IDX=${LSB_JOBINDEX}

mkdir -p "${LOGS_DIR}"
mkdir -p "${KFOLD_DIR}"

if [ -d "/zhome/45/8/156251/Desktop/thesis/labassist/.env" ]; then
  . "/zhome/45/8/156251/Desktop/thesis/labassist/.env/bin/activate"
fi

echo "============================================================"
echo "Experiment 3: Strong Smoothing - Fold ${FOLD_IDX}"
echo "dropout=0.25, smoothing=0.30, num_R=6, epochs=50"
echo "============================================================"

# Generate K-fold splits if needed
if [ ! -f "${KFOLD_DIR}/splits_fold0.json" ]; then
    echo "Generating K-fold splits..."
    python "${SCRIPTS_DIR}/data_prep/create_mydata_kfold_splits.py" \
        --splits "${DATA_ROOT}/cache/splits.json" \
        --outdir "${KFOLD_DIR}"
fi

if [ ! -f "${KFOLD_DIR}/splits_fold${FOLD_IDX}.json" ]; then
    echo "ERROR: Fold split file not found: ${KFOLD_DIR}/splits_fold${FOLD_IDX}.json"
    exit 1
fi

# Create fold-specific config
FOLD_CONFIG="/tmp/exp3_strong_smoothing_fold${FOLD_IDX}_${LSB_JOBID}.yaml"
sed "s/\${FOLD_IDX}/${FOLD_IDX}/g" "${TEMPLATE_CONFIG}" > "${FOLD_CONFIG}"

echo "Using config: ${FOLD_CONFIG}"
echo ""
echo "Fold ${FOLD_IDX} validation video(s):"
python -c "import json; splits=json.load(open('${KFOLD_DIR}/splits_fold${FOLD_IDX}.json')); print('  Val:', splits['val']); print('  Train:', splits['train'])"
echo ""

# Train
echo "Starting training for fold ${FOLD_IDX}..."
python "${SCRIPTS_DIR}/finetune_mstcn.py" --config "${FOLD_CONFIG}"

rm -f "${FOLD_CONFIG}"

echo ""
echo "============================================================"
echo "✓ Experiment 3 Fold ${FOLD_IDX} training complete"
echo "============================================================"


