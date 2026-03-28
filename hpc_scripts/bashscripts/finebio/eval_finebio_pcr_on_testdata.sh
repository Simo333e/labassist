#!/bin/bash
# Evaluate FineBio PCR 5-class model on test data
# Usage: ./eval_finebio_pcr_on_testdata.sh <epoch_number>

set -euo pipefail

if [ $# -lt 1 ]; then
    echo "Usage: $0 <epoch_number>"
    echo "Example: $0 50"
    exit 1
fi

EPOCH=$1

echo "=========================================="
echo "Evaluating FineBio PCR 5-class Model"
echo "Epoch: ${EPOCH}"
echo "=========================================="

# Directories
SCRIPT_DIR=/zhome/45/8/156251/Desktop/thesis/labassist/hpc_scripts
EXP_DIR=/work3/s204157/experiments/finebio_pcr_5class
WEIGHTS=${EXP_DIR}/epoch-${EPOCH}.model

# Check if weights exist
if [ ! -f "${WEIGHTS}" ]; then
    echo "Error: Weights not found: ${WEIGHTS}"
    echo "Available epochs:"
    ls -1 ${EXP_DIR}/epoch-*.model 2>/dev/null | sed 's/.*epoch-/  /' | sed 's/.model//' || echo "  None"
    exit 1
fi

# Test data paths
SPLITS=/work3/s204157/data/labassist/my_testdata/splits_test.json
XDIR=/work3/s204157/data/labassist/my_testdata/cache/features_10hz_i3d_win16
YDIR=/work3/s204157/data/labassist/my_testdata/cache/labels_5class
CLASS_INDEX=/work3/s204157/data/labassist/combined_pcr/cache/class_index_5class.json
OUTDIR=${SCRIPT_DIR}/eval_data/finebio_pcr_5class_testdata/epoch${EPOCH}

# Activate environment
source /zhome/45/8/156251/Desktop/thesis/labassist/.env/bin/activate

# Add MS-TCN to PYTHONPATH
export PYTHONPATH="${PYTHONPATH:-}:${SCRIPT_DIR}/MS-TCN"

# Create output directory
mkdir -p ${OUTDIR}

echo ""
echo "Evaluating on test data..."
echo "Weights: ${WEIGHTS}"
echo "Output: ${OUTDIR}"
echo ""

# Run evaluation
python ${SCRIPT_DIR}/scripts/eval/eval_val_timelines_plus.py \
    --splits ${SPLITS} \
    --xdir ${XDIR} \
    --ydir ${YDIR} \
    --exp_dir ${EXP_DIR} \
    --class_index ${CLASS_INDEX} \
    --weights ${WEIGHTS} \
    --num_layers_PG 10 --num_layers_R 10 --num_R 6 --num_f_maps 64 \
    --outdir ${OUTDIR} \
    --split_name test --causal --make_timelines

echo ""
echo "=========================================="
echo "Evaluation complete!"
echo "Results saved to: ${OUTDIR}"
echo "=========================================="
echo ""
echo "Summary:"
cat ${OUTDIR}/summary.txt


