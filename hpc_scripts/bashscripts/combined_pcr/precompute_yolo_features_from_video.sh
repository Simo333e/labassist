#!/bin/bash
#BSUB -J precompute_yolo_hires[1-4]
#BSUB -q gpua100
#BSUB -W 6:00
#BSUB -n 4
#BSUB -R "rusage[mem=8GB]"
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -o /zhome/45/8/156251/Desktop/thesis/labassist/hpc_scripts/logs/combined_pcr/%J_%I_yolo_hires.out
#BSUB -e /zhome/45/8/156251/Desktop/thesis/labassist/hpc_scripts/logs/combined_pcr/%J_%I_yolo_hires.err

###############################################################################
# YOLO Feature Extraction from FULL-RESOLUTION Video
#
# This script extracts YOLO features directly from video files at full resolution
# (1920x1080) instead of pre-extracted frames (256 pixel short side).
#
# This is CRITICAL for detecting small objects like PCR tubes which are lost
# at low resolution.
###############################################################################

echo "========================================"
echo "YOLO Feature Extraction (Full Resolution)"
echo "Job ID: $LSB_JOBID"
echo "Array Index: $LSB_JOBINDEX"
echo "Host: $(hostname)"
echo "Date: $(date)"
echo "========================================"

# Activate environment
source /zhome/45/8/156251/Desktop/thesis/labassist/.env/bin/activate

# Paths
DATA_ROOT="/work3/s204157/data/labassist/combined_pcr"
SPLITS="${DATA_ROOT}/cache/splits.json"
YOLO_WEIGHTS="/zhome/45/8/156251/Desktop/thesis/labassist/DTULabAssistDataset/runs/detect/labassist_yolo11m22/weights/best.pt"

# Output directory - new directory for high-resolution features
FEATURES_OUT="${DATA_ROOT}/cache/features_10hz_yolo_hires"

# YOLO parameters
FPS="10"
CONF_THRESHOLD="0.25"
IMGSZ="1280"  # YOLO inference size

# Crop parameters (MUST match I3D extraction for alignment!)
CROP_TOP="0.0"
CROP_BOTTOM="0.35"
CROP_LEFT="0.10"
CROP_RIGHT="0.08"

# Sharding
NUM_SHARDS="4"
SHARD_IDX="$((LSB_JOBINDEX - 1))"  # Convert 1-based to 0-based

echo "Configuration:"
echo "  DATA_ROOT: ${DATA_ROOT}"
echo "  SPLITS: ${SPLITS}"
echo "  YOLO_WEIGHTS: ${YOLO_WEIGHTS}"
echo "  FEATURES_OUT: ${FEATURES_OUT}"
echo "  FPS: ${FPS}"
echo "  CONF: ${CONF_THRESHOLD}"
echo "  IMGSZ: ${IMGSZ}"
echo "  CROP: top=${CROP_TOP}, bottom=${CROP_BOTTOM}, left=${CROP_LEFT}, right=${CROP_RIGHT}"
echo "  SHARD: ${SHARD_IDX}/${NUM_SHARDS}"
echo ""

# Run feature extraction from VIDEO (--use_video flag)
python /zhome/45/8/156251/Desktop/thesis/labassist/hpc_scripts/scripts/feature_computations/precompute_features_yolo.py \
  --root "${DATA_ROOT}" \
  --splits "${SPLITS}" \
  --weights "${YOLO_WEIGHTS}" \
  --fps "${FPS}" \
  --conf "${CONF_THRESHOLD}" \
  --imgsz "${IMGSZ}" \
  --device cuda:0 \
  --out "${FEATURES_OUT}" \
  --shards "${NUM_SHARDS}" \
  --shard_idx "${SHARD_IDX}" \
  --crop_top "${CROP_TOP}" \
  --crop_bottom "${CROP_BOTTOM}" \
  --crop_left "${CROP_LEFT}" \
  --crop_right "${CROP_RIGHT}" \
  --selective_crop \
  --vertical_flip \
  --use_video

echo ""
echo "========================================"
echo "YOLO extraction (shard ${SHARD_IDX}) complete at $(date)"
echo "========================================"

