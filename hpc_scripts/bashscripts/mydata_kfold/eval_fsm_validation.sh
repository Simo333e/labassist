#!/bin/bash
#
# FSM Tip Hygiene Validation Pipeline
#
# This script runs the complete FSM validation pipeline:
# 1. Generate predictions for k-fold validation videos (if needed)
# 2. Run FSM validation on predictions
# 3. Generate analysis and visualizations
#
# Usage:
#   bash eval_fsm_validation.sh [EXPERIMENT_NAME] [EPOCH]
#
# Example:
#   bash eval_fsm_validation.sh exp1_baseline 50
#

set -euo pipefail

# =============================================================================
# Configuration
# =============================================================================

EXPERIMENT=${1:-"exp1_baseline"}
EPOCH=${2:-50}

# Paths
REPO_ROOT="/zhome/45/8/156251/Desktop/thesis/labassist"
SCRIPTS_DIR="${REPO_ROOT}/hpc_scripts/scripts"
DATA_ROOT="/work3/s204157/data/labassist/combined_pcr"
EVAL_BASE_DIR="${REPO_ROOT}/hpc_scripts/eval_data/mydata_kfold_fixed/${EXPERIMENT}"
KFOLD_DIR="${DATA_ROOT}/cache/kfold_mydata"

# Output directories
FSM_RESULTS_DIR="${EVAL_BASE_DIR}/fsm_validation"
FSM_ANALYSIS_DIR="${EVAL_BASE_DIR}/fsm_analysis"

# Feature and label directories
XDIR="${DATA_ROOT}/cache/combined_i3d_features"
YDIR="${DATA_ROOT}/cache/labels_7class_combined_v2"

# Class index (7-class taxonomy)
CLASS_INDEX="${REPO_ROOT}/hpc_scripts/configs/class_index_7class.json"

# Model directory pattern
EXP_BASE_DIR="/work3/s204157/experiments/mydata_kfold"

# Activate virtual environment if it exists
if [ -d "${REPO_ROOT}/.env" ]; then
    source "${REPO_ROOT}/.env/bin/activate"
fi

echo "============================================================"
echo "FSM TIP HYGIENE VALIDATION PIPELINE"
echo "============================================================"
echo "Experiment: ${EXPERIMENT}"
echo "Epoch: ${EPOCH}"
echo "Evaluation base: ${EVAL_BASE_DIR}"
echo "FSM results: ${FSM_RESULTS_DIR}"
echo "============================================================"
echo ""

# =============================================================================
# Step 1: Generate predictions for each fold (if not already done)
# =============================================================================

echo "Step 1: Checking for existing predictions..."

NUM_FOLDS=6
NEED_PREDICTIONS=false

for FOLD_IDX in $(seq 0 $((NUM_FOLDS - 1))); do
    FOLD_DIR="${EVAL_BASE_DIR}/fold${FOLD_IDX}"
    PREDICTIONS_DIR="${FOLD_DIR}/predictions"
    
    if [ ! -d "${PREDICTIONS_DIR}" ] || [ -z "$(ls -A ${PREDICTIONS_DIR} 2>/dev/null)" ]; then
        NEED_PREDICTIONS=true
        break
    fi
done

if [ "${NEED_PREDICTIONS}" = true ]; then
    echo "  Predictions not found. Generating predictions for all folds..."
    echo ""
    
    for FOLD_IDX in $(seq 0 $((NUM_FOLDS - 1))); do
        echo "  Generating predictions for Fold ${FOLD_IDX}..."
        
        FOLD_EXP_DIR="${EXP_BASE_DIR}/ms_tcn_mydata_i3d_fold${FOLD_IDX}"
        FOLD_SPLITS="${KFOLD_DIR}/splits_fold${FOLD_IDX}.json"
        FOLD_WEIGHTS="${FOLD_EXP_DIR}/epoch-${EPOCH}.model"
        FOLD_OUTDIR="${EVAL_BASE_DIR}/fold${FOLD_IDX}"
        
        if [ ! -f "${FOLD_WEIGHTS}" ]; then
            echo "    WARNING: Model not found: ${FOLD_WEIGHTS}"
            echo "    Skipping fold ${FOLD_IDX}"
            continue
        fi
        
        python "${SCRIPTS_DIR}/eval/eval_with_predictions.py" \
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
            --make_timelines \
            --save_probs
        
        echo "    ✓ Fold ${FOLD_IDX} complete"
    done
    
    echo ""
    echo "  ✓ All predictions generated"
else
    echo "  ✓ Predictions already exist, skipping generation"
fi

echo ""

# =============================================================================
# Step 2: Run FSM validation
# =============================================================================

echo "Step 2: Running FSM validation..."
echo ""

python "${SCRIPTS_DIR}/eval/run_fsm_validation_kfold.py" \
    --eval_base_dir "${EVAL_BASE_DIR}" \
    --class_index "${CLASS_INDEX}" \
    --output_dir "${FSM_RESULTS_DIR}" \
    --num_folds ${NUM_FOLDS}

echo ""
echo "  ✓ FSM validation complete"
echo ""

# =============================================================================
# Step 3: Generate analysis and visualizations
# =============================================================================

echo "Step 3: Generating analysis and visualizations..."
echo ""

python "${SCRIPTS_DIR}/eval/analyze_fsm_results.py" \
    --fsm_results_dir "${FSM_RESULTS_DIR}" \
    --predictions_base_dir "${EVAL_BASE_DIR}" \
    --class_index "${CLASS_INDEX}" \
    --output_dir "${FSM_ANALYSIS_DIR}" \
    --max_timelines 20

echo ""
echo "  ✓ Analysis complete"
echo ""

# =============================================================================
# Summary
# =============================================================================

echo "============================================================"
echo "FSM VALIDATION PIPELINE COMPLETE"
echo "============================================================"
echo "Results locations:"
echo "  FSM validation: ${FSM_RESULTS_DIR}"
echo "  Analysis:       ${FSM_ANALYSIS_DIR}"
echo ""
echo "Key files:"
echo "  - ${FSM_RESULTS_DIR}/fsm_aggregate.json"
echo "  - ${FSM_ANALYSIS_DIR}/fsm_performance_per_fold.png"
echo "  - ${FSM_ANALYSIS_DIR}/fsm_detection_performance.png"
echo "  - ${FSM_ANALYSIS_DIR}/timelines/*.png"
echo "============================================================"




