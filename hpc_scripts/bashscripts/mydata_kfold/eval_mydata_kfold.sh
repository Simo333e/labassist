#!/bin/sh
#BSUB -q gpuv100
#BSUB -J eval_mydata_kfold
#BSUB -gpu "num=1"
#BSUB -n 4
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=16GB]"
#BSUB -M 16GB
#BSUB -W 2:00
#BSUB -o /zhome/45/8/156251/Desktop/thesis/labassist/hpc_scripts/logs/mydata_kfold/%J_eval_kfold.out
#BSUB -e /zhome/45/8/156251/Desktop/thesis/labassist/hpc_scripts/logs/mydata_kfold/%J_eval_kfold.err

set -euo pipefail

# ============================================================
# Evaluate all K-fold models and aggregate results
# ============================================================

# Paths
DATA_ROOT="/work3/s204157/data/labassist/combined_pcr"
KFOLD_DIR="${DATA_ROOT}/cache/kfold_mydata"
EXP_BASE_DIR="/work3/s204157/experiments/mydata_kfold"
XDIR="${DATA_ROOT}/cache/features_10hz_i3d_selective_crop_flip_win16"
YDIR="${DATA_ROOT}/cache/labels_5class"
CLASS_INDEX="${DATA_ROOT}/cache/class_index_5class.json"
SCRIPTS_DIR="/zhome/45/8/156251/Desktop/thesis/labassist/hpc_scripts/scripts"
OUTDIR="/zhome/45/8/156251/Desktop/thesis/labassist/hpc_scripts/eval_data/mydata_kfold"

EPOCH=100  # Which epoch to evaluate

mkdir -p "${OUTDIR}"

# Activate environment
if [ -d "/zhome/45/8/156251/Desktop/thesis/labassist/.env" ]; then
  . "/zhome/45/8/156251/Desktop/thesis/labassist/.env/bin/activate"
fi

# Add MS-TCN to Python path
export PYTHONPATH="${SCRIPTS_DIR}/../MS-TCN:${PYTHONPATH:-}"

echo "============================================================"
echo "Evaluating K-Fold Cross-Validation Models"
echo "============================================================"
echo "K-fold splits: ${KFOLD_DIR}"
echo "Experiments: ${EXP_BASE_DIR}"
echo "Features: ${XDIR}"
echo "Epoch: ${EPOCH}"
echo "Output: ${OUTDIR}"
echo "============================================================"
echo ""

# Evaluate each fold individually first
K=6
for FOLD_IDX in $(seq 0 $((K-1))); do
    echo "----------------------------------------"
    echo "Evaluating Fold ${FOLD_IDX}..."
    echo "----------------------------------------"
    
    FOLD_EXP_DIR="${EXP_BASE_DIR}/ms_tcn_mydata_i3d_fold${FOLD_IDX}"
    FOLD_SPLITS="${KFOLD_DIR}/splits_fold${FOLD_IDX}.json"
    FOLD_WEIGHTS="${FOLD_EXP_DIR}/epoch-${EPOCH}.model"
    FOLD_OUTDIR="${OUTDIR}/fold${FOLD_IDX}"
    
    mkdir -p "${FOLD_OUTDIR}"
    
    if [ ! -f "${FOLD_WEIGHTS}" ]; then
        echo "WARNING: Model not found: ${FOLD_WEIGHTS}"
        echo "Skipping fold ${FOLD_IDX}"
        continue
    fi
    
    python "${SCRIPTS_DIR}/eval/eval_val_timelines_plus.py" \
        --splits "${FOLD_SPLITS}" \
        --xdir "${XDIR}" \
        --ydir "${YDIR}" \
        --exp_dir "${FOLD_EXP_DIR}" \
        --class_index "${CLASS_INDEX}" \
        --weights "${FOLD_WEIGHTS}" \
        --num_layers_PG 10 \
        --num_layers_R 10 \
        --num_R 6 \
        --num_f_maps 64 \
        --outdir "${FOLD_OUTDIR}" \
        --split_name val \
        --causal \
        --make_timelines
    
    echo ""
done

echo "============================================================"
echo "Aggregating results across all folds..."
echo "============================================================"

# Run aggregation
python "${SCRIPTS_DIR}/eval/aggregate_kfold_results.py" \
    --kfold_dir "${KFOLD_DIR}" \
    --exp_base_dir "${EXP_BASE_DIR}" \
    --xdir "${XDIR}" \
    --ydir "${YDIR}" \
    --class_index "${CLASS_INDEX}" \
    --outdir "${OUTDIR}" \
    --epoch "${EPOCH}" \
    --causal

echo ""
echo "============================================================"
echo "✓ K-fold evaluation complete"
echo "Results saved to: ${OUTDIR}"
echo "============================================================"

