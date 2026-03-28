#!/bin/bash
# Launch all 5 regularization experiments (5 folds each = 25 jobs total)
# This submits all experiments at once - they will run in parallel based on queue availability

set -euo pipefail

cd /zhome/45/8/156251/Desktop/thesis/labassist/hpc_scripts

echo "============================================================"
echo "Launching Phase 2: K-Fold Regularization Experiments"
echo "============================================================"
echo ""
echo "This will submit 5 experiments × 5 folds = 25 jobs total"
echo ""
echo "Experiments:"
echo "  1. Baseline          (dropout=0.25, smoothing=0.15, num_R=6)"
echo "  2. High Dropout      (dropout=0.50, smoothing=0.15, num_R=6)"
echo "  3. Strong Smoothing  (dropout=0.25, smoothing=0.30, num_R=6)"
echo "  4. Combined Reg      (dropout=0.50, smoothing=0.30, num_R=3)"
echo "  5. Early Stop        (dropout=0.25, smoothing=0.15, num_R=6, epochs=15)"
echo ""
echo "Transfer learning from: FineBio epoch-50 checkpoint"
echo "============================================================"
echo ""

# Submit each experiment
echo "Submitting Experiment 1: Baseline..."
JOB1=$(bsub < bashscripts/mydata_kfold/train_exp1_baseline.sh | grep -oP 'Job <\K[0-9]+')
echo "  Job ID: ${JOB1}"

echo ""
echo "Submitting Experiment 2: High Dropout..."
JOB2=$(bsub < bashscripts/mydata_kfold/train_exp2_high_dropout.sh | grep -oP 'Job <\K[0-9]+')
echo "  Job ID: ${JOB2}"

echo ""
echo "Submitting Experiment 3: Strong Smoothing..."
JOB3=$(bsub < bashscripts/mydata_kfold/train_exp3_strong_smoothing.sh | grep -oP 'Job <\K[0-9]+')
echo "  Job ID: ${JOB3}"

echo ""
echo "Submitting Experiment 4: Combined Regularization..."
JOB4=$(bsub < bashscripts/mydata_kfold/train_exp4_combined_reg.sh | grep -oP 'Job <\K[0-9]+')
echo "  Job ID: ${JOB4}"

echo ""
echo "Submitting Experiment 5: Early Stopping..."
JOB5=$(bsub < bashscripts/mydata_kfold/train_exp5_early_stop.sh | grep -oP 'Job <\K[0-9]+')
echo "  Job ID: ${JOB5}"

echo ""
echo "============================================================"
echo "✓ All experiments submitted!"
echo "============================================================"
echo ""
echo "Job IDs:"
echo "  Exp1 (Baseline):         ${JOB1}"
echo "  Exp2 (High Dropout):     ${JOB2}"
echo "  Exp3 (Strong Smoothing): ${JOB3}"
echo "  Exp4 (Combined Reg):     ${JOB4}"
echo "  Exp5 (Early Stop):       ${JOB5}"
echo ""
echo "Monitor with:"
echo "  bjobs -w"
echo ""
echo "Check logs in:"
echo "  logs/mydata_kfold/*.out"
echo "  logs/mydata_kfold/*.err"
echo ""
echo "Expected completion: ~6 hours (Exp1-4), ~2 hours (Exp5)"
echo "============================================================"


