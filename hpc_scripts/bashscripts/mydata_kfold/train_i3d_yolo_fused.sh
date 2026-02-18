#!/bin/sh
#BSUB -q gpuv100
#BSUB -J mydata_exp9_yolo[1-5]
#BSUB -gpu "num=1"
#BSUB -n 4
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=16GB]"
#BSUB -M 16GB
#BSUB -W 6:00
#BSUB -o /zhome/45/8/156251/Desktop/thesis/labassist/hpc_scripts/logs/mydata_kfold/%J_%I_exp9_yolo.out
#BSUB -e /zhome/45/8/156251/Desktop/thesis/labassist/hpc_scripts/logs/mydata_kfold/%J_%I_exp9_yolo.err

set -euo pipefail

DATA_ROOT="/work3/s204157/data/labassist/combined_pcr"
KFOLD_DIR="${DATA_ROOT}/cache/kfold_mydata"
TEMPLATE_CONFIG="/zhome/45/8/156251/Desktop/thesis/labassist/hpc_scripts/configs/mydata_kfold/exp9_i3d_yolo_fused.yaml"
SCRIPTS_DIR="/zhome/45/8/156251/Desktop/thesis/labassist/hpc_scripts/scripts"
LOGS_DIR="/zhome/45/8/156251/Desktop/thesis/labassist/hpc_scripts/logs/mydata_kfold"

# Get fold index from job array (convert 1-based to 0-based)
FOLD_IDX=$((LSB_JOBINDEX - 1))

mkdir -p "${LOGS_DIR}"
mkdir -p "${KFOLD_DIR}"

# Activate environment
if [ -d "/zhome/45/8/156251/Desktop/thesis/labassist/.env" ]; then
  . "/zhome/45/8/156251/Desktop/thesis/labassist/.env/bin/activate"
fi

echo "============================================================"
echo "Experiment 9: I3D + YOLO Fused - Fold ${FOLD_IDX}"
echo "============================================================"
echo ""

# Generate K-fold splits if they don't exist
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

# Verify fused features exist
FUSED_DIR="${DATA_ROOT}/cache/features_10hz_i3d_yolo_fused"
if [ ! -d "${FUSED_DIR}" ]; then
    echo "ERROR: Fused features not found at ${FUSED_DIR}"
    echo "Run fuse_i3d_yolo_features.sh first"
    exit 1
fi

# Create config for this fold by substituting FOLD_IDX
FOLD_CONFIG="/tmp/exp9_yolo_fold${FOLD_IDX}_${LSB_JOBID}.yaml"
sed "s/\${FOLD_IDX}/${FOLD_IDX}/g" "${TEMPLATE_CONFIG}" > "${FOLD_CONFIG}"

echo "Using config: ${FOLD_CONFIG}"
echo "Splits file: ${KFOLD_DIR}/splits_fold${FOLD_IDX}.json"
echo "Features: ${FUSED_DIR}"

# Show which video is held out for validation
echo ""
echo "Fold ${FOLD_IDX} validation video(s):"
python -c "import json; splits=json.load(open('${KFOLD_DIR}/splits_fold${FOLD_IDX}.json')); print('  Val:', splits['val']); print('  Train:', splits['train'])"
echo ""

# Start training for this fold
echo "Starting training for fold ${FOLD_IDX}..."
python "${SCRIPTS_DIR}/finetune_mstcn.py" --config "${FOLD_CONFIG}"

echo ""
echo "============================================================"
echo "✓ Experiment 9 (I3D+YOLO) Fold ${FOLD_IDX} training complete"
echo "============================================================"

