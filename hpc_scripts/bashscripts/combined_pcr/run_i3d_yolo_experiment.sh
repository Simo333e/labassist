#!/bin/bash
# ============================================================================
# Complete I3D + YOLO Multi-Modal Experiment Pipeline
# ============================================================================
#
# This script orchestrates the full experiment:
#   1. Extract YOLO object detection features (parallel, ~8 hours)
#   2. Fuse I3D + YOLO features (serial, ~5 minutes)
#   3. Train MS-TCN with fused features (serial, ~24 hours)
#
# Usage:
#   ./run_i3d_yolo_experiment.sh           # Run all steps
#   ./run_i3d_yolo_experiment.sh yolo      # Only extract YOLO features
#   ./run_i3d_yolo_experiment.sh fuse      # Only fuse features
#   ./run_i3d_yolo_experiment.sh train     # Only train model
#
# Prerequisites:
#   - I3D features already extracted (features_10hz_i3d_selective_crop_flip_win16)
#   - YOLO model trained (DTULabAssistDataset/runs/detect/labassist_yolo11m22)
#   - FineBio pretrained checkpoint available
#
# ============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOGS_DIR="/zhome/45/8/156251/Desktop/thesis/labassist/hpc_scripts/logs/combined_pcr"
DATA_ROOT="/work3/s204157/data/labassist/combined_pcr"

mkdir -p "${LOGS_DIR}"

# Check which steps to run
RUN_STEP="${1:-all}"

echo "============================================================"
echo "I3D + YOLO Multi-Modal Action Recognition Experiment"
echo "============================================================"
echo ""
echo "This experiment tests whether YOLO object detection features"
echo "can help disambiguate 'Pipette pcr tube' vs 'Aspirate reagent'"
echo ""
echo "Step to run: ${RUN_STEP}"
echo ""

# ============================================================================
# Step 1: Extract YOLO features
# ============================================================================
run_yolo_extraction() {
    echo "============================================================"
    echo "Step 1: YOLO Feature Extraction"
    echo "============================================================"
    
    # Check if I3D features exist
    I3D_DIR="${DATA_ROOT}/cache/features_10hz_i3d_selective_crop_flip_win16"
    if [ ! -d "${I3D_DIR}" ]; then
        echo "ERROR: I3D features not found at ${I3D_DIR}"
        echo "Please run precompute_i3d_selective_crop_flip_win16.sh first"
        exit 1
    fi
    
    # Check if YOLO model exists
    YOLO_WEIGHTS="/zhome/45/8/156251/Desktop/thesis/labassist/DTULabAssistDataset/runs/detect/labassist_yolo11m22/weights/best.pt"
    if [ ! -f "${YOLO_WEIGHTS}" ]; then
        echo "ERROR: YOLO weights not found at ${YOLO_WEIGHTS}"
        echo "Please train YOLO model first"
        exit 1
    fi
    
    # Check if YOLO features already exist
    YOLO_DIR="${DATA_ROOT}/cache/features_10hz_yolo"
    if [ -d "${YOLO_DIR}" ] && [ "$(ls -A ${YOLO_DIR}/*.npy 2>/dev/null | wc -l)" -gt 40 ]; then
        echo "YOLO features already exist at ${YOLO_DIR}"
        echo "Skipping extraction (delete directory to re-extract)"
        return 0
    fi
    
    echo "Submitting YOLO feature extraction job (4 parallel shards)..."
    JOB_ID=$(bsub < "${SCRIPT_DIR}/precompute_yolo_features.sh" 2>&1 | grep -oP '(?<=Job <)\d+(?=>)')
    
    echo "Submitted job: ${JOB_ID}"
    echo "Monitor with: bjobs ${JOB_ID}"
    echo "Logs: ${LOGS_DIR}/${JOB_ID}_*_yolo_features.out"
    echo ""
    echo "IMPORTANT: Wait for this job to complete before running step 2!"
    echo "Check status: bjobs ${JOB_ID}"
    
    return 0
}

# ============================================================================
# Step 2: Fuse I3D + YOLO features
# ============================================================================
run_feature_fusion() {
    echo "============================================================"
    echo "Step 2: Feature Fusion (I3D + YOLO)"
    echo "============================================================"
    
    # Check if YOLO features exist
    YOLO_DIR="${DATA_ROOT}/cache/features_10hz_yolo"
    if [ ! -d "${YOLO_DIR}" ] || [ "$(ls -A ${YOLO_DIR}/*.npy 2>/dev/null | wc -l)" -lt 40 ]; then
        echo "ERROR: YOLO features not found or incomplete at ${YOLO_DIR}"
        echo "Please run step 1 (YOLO extraction) first and wait for completion"
        exit 1
    fi
    
    # Check if fused features already exist
    FUSED_DIR="${DATA_ROOT}/cache/features_10hz_i3d_yolo_fused"
    if [ -d "${FUSED_DIR}" ] && [ "$(ls -A ${FUSED_DIR}/*.npy 2>/dev/null | wc -l)" -gt 40 ]; then
        echo "Fused features already exist at ${FUSED_DIR}"
        echo "Skipping fusion (delete directory to re-fuse)"
        return 0
    fi
    
    echo "Submitting feature fusion job..."
    JOB_ID=$(bsub < "${SCRIPT_DIR}/fuse_i3d_yolo_features.sh" 2>&1 | grep -oP '(?<=Job <)\d+(?=>)')
    
    echo "Submitted job: ${JOB_ID}"
    echo "Monitor with: bjobs ${JOB_ID}"
    echo "This is a quick job (~5 minutes)"
    echo ""
    
    return 0
}

# ============================================================================
# Step 3: Train MS-TCN with fused features
# ============================================================================
run_training() {
    echo "============================================================"
    echo "Step 3: Training MS-TCN with I3D+YOLO Features"
    echo "============================================================"
    
    # Check if fused features exist
    FUSED_DIR="${DATA_ROOT}/cache/features_10hz_i3d_yolo_fused"
    if [ ! -d "${FUSED_DIR}" ] || [ "$(ls -A ${FUSED_DIR}/*.npy 2>/dev/null | wc -l)" -lt 40 ]; then
        echo "ERROR: Fused features not found at ${FUSED_DIR}"
        echo "Please run step 2 (feature fusion) first"
        exit 1
    fi
    
    echo "Submitting training job..."
    JOB_ID=$(bsub < "${SCRIPT_DIR}/train_i3d_yolo_fused.sh" 2>&1 | grep -oP '(?<=Job <)\d+(?=>)')
    
    echo "Submitted job: ${JOB_ID}"
    echo "Monitor with: bjobs ${JOB_ID}"
    echo "Logs: ${LOGS_DIR}/${JOB_ID}_i3d_yolo_fused.out"
    echo ""
    echo "Training will take ~24 hours"
    echo "Checkpoints saved to: /work3/s204157/experiments/combined_pcr_i3d_yolo_fused/"
    
    return 0
}

# ============================================================================
# Main execution
# ============================================================================
case "${RUN_STEP}" in
    all)
        echo "Running complete pipeline..."
        echo ""
        run_yolo_extraction
        echo ""
        echo "============================================================"
        echo "PIPELINE STATUS"
        echo "============================================================"
        echo ""
        echo "Step 1 (YOLO extraction) has been submitted."
        echo ""
        echo "NEXT STEPS (run these after each job completes):"
        echo ""
        echo "  After YOLO extraction completes:"
        echo "    $0 fuse"
        echo ""
        echo "  After feature fusion completes:"
        echo "    $0 train"
        echo ""
        echo "Or wait and run each step when the previous one finishes."
        ;;
    yolo)
        run_yolo_extraction
        ;;
    fuse)
        run_feature_fusion
        ;;
    train)
        run_training
        ;;
    status)
        echo "Checking status..."
        echo ""
        echo "I3D features:"
        I3D_COUNT=$(ls -1 ${DATA_ROOT}/cache/features_10hz_i3d_selective_crop_flip_win16/*_x.npy 2>/dev/null | wc -l)
        echo "  Count: ${I3D_COUNT} videos"
        echo ""
        echo "YOLO features:"
        YOLO_COUNT=$(ls -1 ${DATA_ROOT}/cache/features_10hz_yolo/*_x.npy 2>/dev/null | wc -l)
        echo "  Count: ${YOLO_COUNT} videos"
        echo ""
        echo "Fused features:"
        FUSED_COUNT=$(ls -1 ${DATA_ROOT}/cache/features_10hz_i3d_yolo_fused/*_x.npy 2>/dev/null | wc -l)
        echo "  Count: ${FUSED_COUNT} videos"
        echo ""
        echo "Running jobs:"
        bjobs 2>/dev/null || echo "  No jobs running"
        ;;
    *)
        echo "Usage: $0 [all|yolo|fuse|train|status]"
        echo ""
        echo "  all     - Run complete pipeline (default)"
        echo "  yolo    - Extract YOLO features only"
        echo "  fuse    - Fuse I3D+YOLO features only"
        echo "  train   - Train model only"
        echo "  status  - Check feature extraction status"
        exit 1
        ;;
esac

echo ""
echo "============================================================"
echo "Done!"
echo "============================================================"

