#!/bin/bash
#
# Run FSM validation across ALL V-JEPA folds (best performing model)
#

set -euo pipefail

# Configuration
REPO_ROOT="/zhome/45/8/156251/Desktop/thesis/labassist"
SCRIPTS_DIR="${REPO_ROOT}/hpc_scripts/scripts"
DATA_ROOT="/work3/s204157/data/labassist/combined_pcr"
EXP_BASE_DIR="/work3/s204157/experiments"

# V-JEPA experiment
EXPERIMENT="mydata_kfold_exp15_cascaded_vjepa"

# Best epochs per fold (from your results)
declare -A BEST_EPOCHS
BEST_EPOCHS[0]=8   # F1@50=15.22%
BEST_EPOCHS[1]=18  # F1@50=32.03%
BEST_EPOCHS[2]=4   # F1@50=9.19%
BEST_EPOCHS[3]=6   # F1@50=30.59%
BEST_EPOCHS[4]=26  # F1@50=15.23%
BEST_EPOCHS[5]=4   # F1@50=11.55%

# Paths
KFOLD_DIR="${DATA_ROOT}/cache/kfold_mydata"
XDIR="${DATA_ROOT}/cache/features_10hz_vjepa_selective_crop_flip_stride1"
YDIR="${DATA_ROOT}/cache/labels_5class"
CLASS_INDEX="${DATA_ROOT}/cache/class_index_5class.json"

# Output directories
OUTPUT_BASE="${REPO_ROOT}/hpc_scripts/eval_data/fsm_vjepa_all_folds"
FSM_DIR="${OUTPUT_BASE}/fsm_validation"
ANALYSIS_DIR="${OUTPUT_BASE}/fsm_analysis"

echo "============================================================"
echo "FSM VALIDATION - V-JEPA ALL FOLDS"
echo "============================================================"
echo "Experiment: ${EXPERIMENT}"
echo "Output: ${OUTPUT_BASE}"
echo "============================================================"
echo ""

# Activate virtual environment
if [ -d "${REPO_ROOT}/.env" ]; then
    source "${REPO_ROOT}/.env/bin/activate"
fi

# Step 1: Generate predictions for each fold
echo "Step 1: Generating predictions for all folds..."
echo ""

for FOLD_IDX in 0 1 2 3 4 5; do
    EPOCH=${BEST_EPOCHS[$FOLD_IDX]}
    
    FOLD_EXP_DIR="${EXP_BASE_DIR}/${EXPERIMENT}/fold${FOLD_IDX}"
    FOLD_SPLITS="${KFOLD_DIR}/splits_fold${FOLD_IDX}.json"
    FOLD_WEIGHTS="${FOLD_EXP_DIR}/epoch-${EPOCH}.model"
    FOLD_OUTDIR="${OUTPUT_BASE}/fold${FOLD_IDX}"
    
    echo "Processing Fold ${FOLD_IDX} (epoch ${EPOCH})..."
    
    if [ ! -f "${FOLD_WEIGHTS}" ]; then
        echo "  WARNING: Model not found: ${FOLD_WEIGHTS}"
        echo "  Skipping fold ${FOLD_IDX}"
        continue
    fi
    
    # Show validation video
    echo "  Val video: $(python3 -c "import json; print(json.load(open('${FOLD_SPLITS}'))['val'])")"
    
    # Skip if predictions already exist
    if [ -d "${FOLD_OUTDIR}/predictions" ] && [ -n "$(ls -A ${FOLD_OUTDIR}/predictions 2>/dev/null)" ]; then
        echo "  ✓ Predictions already exist, skipping..."
    else
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
        
        echo "  ✓ Predictions saved"
    fi
    echo ""
done

echo "✓ All predictions generated"
echo ""

# Step 2: Run FSM validation across all folds
echo "Step 2: Running FSM validation..."
echo ""

python "${SCRIPTS_DIR}/eval/run_fsm_validation_kfold.py" \
    --eval_base_dir "${OUTPUT_BASE}" \
    --class_index "${CLASS_INDEX}" \
    --output_dir "${FSM_DIR}" \
    --num_folds 6

echo ""
echo "✓ FSM validation complete"
echo ""

# Step 3: Generate analysis
echo "Step 3: Generating analysis..."
echo ""

python "${SCRIPTS_DIR}/eval/analyze_fsm_results.py" \
    --fsm_results_dir "${FSM_DIR}" \
    --predictions_base_dir "${OUTPUT_BASE}" \
    --class_index "${CLASS_INDEX}" \
    --output_dir "${ANALYSIS_DIR}" \
    --max_timelines 20

echo ""
echo "✓ Analysis complete"
echo ""

# Summary
echo "============================================================"
echo "FSM VALIDATION COMPLETE - ALL V-JEPA FOLDS"
echo "============================================================"
echo "Results:"
echo "  ${OUTPUT_BASE}"
echo ""
echo "Key files:"
echo "  ${FSM_DIR}/fsm_aggregate.json"
echo "  ${ANALYSIS_DIR}/alerts_summary.txt"
echo "  ${ANALYSIS_DIR}/timelines/"
echo "============================================================"

