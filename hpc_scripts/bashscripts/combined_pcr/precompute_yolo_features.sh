#!/bin/sh
#BSUB -q gpuv100
#BSUB -J precompute_yolo[1-4]
#BSUB -gpu "num=1"
#BSUB -n 4
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=32GB]"
#BSUB -M 32GB
#BSUB -W 8:00
#BSUB -o /zhome/45/8/156251/Desktop/thesis/labassist/hpc_scripts/logs/combined_pcr/%J_%I_yolo_features.out
#BSUB -e /zhome/45/8/156251/Desktop/thesis/labassist/hpc_scripts/logs/combined_pcr/%J_%I_yolo_features.err

# ============================================================================
# YOLO Object Detection Feature Extraction
# ============================================================================
# Purpose: Extract object detection features to disambiguate similar actions.
#
# The model currently confuses "Pipette pcr tube" with "Aspirate reagent"
# because I3D captures motion patterns, not target objects.
# YOLO detects:
#   - pcr tube (class 6)
#   - Reagents (class 5)
#   - hand (class 7)
#   - etc.
#
# Features (28D per frame):
#   - Object confidence (14D): Max confidence per class
#   - Key object positions (10D): Normalized (cx, cy) for 5 key objects
#   - Hand-object proximity (4D): Distance from hand to key containers
# ============================================================================

set -euo pipefail

DATA_ROOT="/work3/s204157/data/labassist/combined_pcr"
SPLITS="${DATA_ROOT}/cache/splits.json"

# Trained YOLO model (labassist_yolo11m22)
YOLO_WEIGHTS="/zhome/45/8/156251/Desktop/thesis/labassist/DTULabAssistDataset/runs/detect/labassist_yolo11m22/weights/best.pt"

# Feature extraction parameters
FPS="10"            # Match I3D extraction rate
CONF="0.25"         # YOLO confidence threshold
IMGSZ="1280"        # YOLO inference size (matches training)
BATCH_SIZE="16"     # Batch size for YOLO inference

# Output directory
FEATURES_OUT="${DATA_ROOT}/cache/features_10hz_yolo"

# Crop parameters (MUST match I3D extraction for alignment!)
CROP_TOP="0.0"
CROP_BOTTOM="0.35"
CROP_LEFT="0.10"
CROP_RIGHT="0.08"

mkdir -p "${FEATURES_OUT}"
mkdir -p /zhome/45/8/156251/Desktop/thesis/labassist/hpc_scripts/logs/combined_pcr

if [ -d "/zhome/45/8/156251/Desktop/thesis/labassist/.env" ]; then
  . "/zhome/45/8/156251/Desktop/thesis/labassist/.env/bin/activate"
fi

# Convert 1-based LSF index to 0-based shard index
SHARD_IDX=$(($LSB_JOBINDEX - 1))
NUM_SHARDS=4

echo "============================================================"
echo "YOLO Object Detection Feature Extraction"
echo "============================================================"
echo "Running shard ${SHARD_IDX}/${NUM_SHARDS}"
echo "Model: ${YOLO_WEIGHTS}"
echo "FPS: ${FPS} Hz (matching I3D)"
echo "Confidence threshold: ${CONF}"
echo "Image size: ${IMGSZ}"
echo "Output: ${FEATURES_OUT}"
echo ""
echo "Cropping (mydata only): top=${CROP_TOP}, bottom=${CROP_BOTTOM}, left=${CROP_LEFT}, right=${CROP_RIGHT}"
echo "Vertical flip: enabled for mydata stems"
echo ""

python /zhome/45/8/156251/Desktop/thesis/labassist/hpc_scripts/scripts/feature_computations/precompute_features_yolo.py \
  --root "${DATA_ROOT}" \
  --splits "${SPLITS}" \
  --weights "${YOLO_WEIGHTS}" \
  --fps "${FPS}" \
  --conf "${CONF}" \
  --imgsz "${IMGSZ}" \
  --batch_size "${BATCH_SIZE}" \
  --device cuda:0 \
  --out "${FEATURES_OUT}" \
  --shards "${NUM_SHARDS}" \
  --shard_idx "${SHARD_IDX}" \
  --crop_top "${CROP_TOP}" \
  --crop_bottom "${CROP_BOTTOM}" \
  --crop_left "${CROP_LEFT}" \
  --crop_right "${CROP_RIGHT}" \
  --selective_crop \
  --vertical_flip

echo ""
echo "============================================================"
echo "✓ YOLO feature extraction complete for shard ${SHARD_IDX}"
echo "============================================================"

