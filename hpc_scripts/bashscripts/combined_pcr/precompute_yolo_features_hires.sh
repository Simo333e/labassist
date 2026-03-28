#!/bin/bash
#BSUB -J precompute_yolo_hires[1-4]
#BSUB -q gpuv100
#BSUB -W 4:00
#BSUB -n 4
#BSUB -R "rusage[mem=8GB]"
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -o /zhome/45/8/156251/Desktop/thesis/labassist/hpc_scripts/logs/combined_pcr/%J_%I_yolo_hires.out
#BSUB -e /zhome/45/8/156251/Desktop/thesis/labassist/hpc_scripts/logs/combined_pcr/%J_%I_yolo_hires.err

###############################################################################
# YOLO Feature Extraction from High-Resolution Preprocessed Frames
#
# This reads from frames_hires_preprocessed/ which contains:
# - High resolution frames (1080p) for better small object detection
# - Already cropped and flipped (matching I3D preprocessing)
#
# IMPORTANT: Run extract_frames_hires_preprocessed.sh FIRST!
###############################################################################

echo "========================================"
echo "YOLO Feature Extraction (High-Res Frames)"
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

# High-res preprocessed frames (already cropped+flipped)
FRAMES_DIR="${DATA_ROOT}/frames_hires_preprocessed"

# Output directory for YOLO features
FEATURES_OUT="${DATA_ROOT}/cache/features_10hz_yolo_hires"

# YOLO parameters
FPS="10"
CONF_THRESHOLD="0.25"
IMGSZ="1280"  # YOLO inference size

# Sharding
NUM_SHARDS="4"
SHARD_IDX="$((LSB_JOBINDEX - 1))"

echo "Configuration:"
echo "  DATA_ROOT: ${DATA_ROOT}"
echo "  SPLITS: ${SPLITS}"
echo "  YOLO_WEIGHTS: ${YOLO_WEIGHTS}"
echo "  FRAMES_DIR: ${FRAMES_DIR}"
echo "  FEATURES_OUT: ${FEATURES_OUT}"
echo "  FPS: ${FPS}"
echo "  CONF: ${CONF_THRESHOLD}"
echo "  IMGSZ: ${IMGSZ}"
echo "  SHARD: ${SHARD_IDX}/${NUM_SHARDS}"
echo ""

# Check that frames exist
if [ ! -d "${FRAMES_DIR}" ]; then
    echo "ERROR: Frames directory not found: ${FRAMES_DIR}"
    echo "Please run extract_frames_hires_preprocessed.sh first!"
    exit 1
fi

# Run feature extraction
# NOTE: --frames_preprocessed tells script to NOT apply crop/flip (already done)
python /zhome/45/8/156251/Desktop/thesis/labassist/hpc_scripts/scripts/feature_computations/precompute_features_yolo.py \
  --root "${DATA_ROOT}" \
  --splits "${SPLITS}" \
  --weights "${YOLO_WEIGHTS}" \
  --fps "${FPS}" \
  --conf "${CONF_THRESHOLD}" \
  --imgsz "${IMGSZ}" \
  --device cuda:0 \
  --out "${FEATURES_OUT}" \
  --frames_dir "${FRAMES_DIR}" \
  --frames_preprocessed \
  --shards "${NUM_SHARDS}" \
  --shard_idx "${SHARD_IDX}"

echo ""
echo "========================================"
echo "YOLO extraction (shard ${SHARD_IDX}) complete at $(date)"
echo "========================================"

