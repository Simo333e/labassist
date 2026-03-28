#!/bin/sh
#BSUB -q gpua100
#BSUB -J mydata_kfold[1-5]
#BSUB -gpu "num=1"
#BSUB -n 4
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=16GB]"
#BSUB -M 16GB
#BSUB -W 8:00
#BSUB -o /zhome/45/8/156251/Desktop/thesis/labassist/hpc_scripts/logs/mydata_kfold/%J_%I_fold.out
#BSUB -e /zhome/45/8/156251/Desktop/thesis/labassist/hpc_scripts/logs/mydata_kfold/%J_%I_fold.err

set -euo pipefail

# ============================================================
# Mydata K-Fold Cross-Validation Training
# ============================================================
# This script runs as a job array with K=6 folds (Leave-One-Out)
# Each job trains on 5 mydata videos and validates on 1
# ============================================================

# Paths
DATA_ROOT="/work3/s204157/data/labassist/combined_pcr"
KFOLD_DIR="${DATA_ROOT}/cache/kfold_mydata"
TEMPLATE_CONFIG="/zhome/45/8/156251/Desktop/thesis/labassist/hpc_scripts/configs/mydata_kfold/mydata_i3d_kfold_template.yaml"
SCRIPTS_DIR="/zhome/45/8/156251/Desktop/thesis/labassist/hpc_scripts/scripts"
LOGS_DIR="/zhome/45/8/156251/Desktop/thesis/labassist/hpc_scripts/logs/mydata_kfold"

# Get fold index from job array (LSB_JOBINDEX is 0-based for this job)
FOLD_IDX=${LSB_JOBINDEX}

mkdir -p "${LOGS_DIR}"
mkdir -p "${KFOLD_DIR}"

# Activate environment
if [ -d "/zhome/45/8/156251/Desktop/thesis/labassist/.env" ]; then
  . "/zhome/45/8/156251/Desktop/thesis/labassist/.env/bin/activate"
fi

echo "============================================================"
echo "Mydata K-Fold Cross-Validation - Fold ${FOLD_IDX}"
echo "============================================================"

# Step 1: Generate K-fold splits if they don't exist
if [ ! -f "${KFOLD_DIR}/splits_fold0.json" ]; then
    echo "Generating K-fold splits..."
    python "${SCRIPTS_DIR}/data_prep/create_mydata_kfold_splits.py" \
        --splits "${DATA_ROOT}/cache/splits.json" \
        --outdir "${KFOLD_DIR}"
fi

# Verify fold split exists
if [ ! -f "${KFOLD_DIR}/splits_fold${FOLD_IDX}.json" ]; then
    echo "ERROR: Fold split file not found: ${KFOLD_DIR}/splits_fold${FOLD_IDX}.json"
    exit 1
fi

# Step 2: Create config for this fold by substituting FOLD_IDX
FOLD_CONFIG="/tmp/mydata_i3d_kfold_fold${FOLD_IDX}_${LSB_JOBID}.yaml"
sed "s/\${FOLD_IDX}/${FOLD_IDX}/g" "${TEMPLATE_CONFIG}" > "${FOLD_CONFIG}"

echo "Using config: ${FOLD_CONFIG}"
echo "Splits file: ${KFOLD_DIR}/splits_fold${FOLD_IDX}.json"

# Show which video is held out for validation
echo ""
echo "Fold ${FOLD_IDX} validation video(s):"
python -c "import json; splits=json.load(open('${KFOLD_DIR}/splits_fold${FOLD_IDX}.json')); print('  Val:', splits['val']); print('  Train:', splits['train'])"
echo ""

# Step 3: Train the model
echo "Starting training for fold ${FOLD_IDX}..."
python "${SCRIPTS_DIR}/finetune_mstcn.py" --config "${FOLD_CONFIG}"

# Cleanup temp config
rm -f "${FOLD_CONFIG}"

echo ""
echo "============================================================"
echo "✓ Fold ${FOLD_IDX} training complete"
echo "============================================================"

