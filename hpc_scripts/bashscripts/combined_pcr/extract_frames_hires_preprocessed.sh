#!/bin/bash
#BSUB -J extract_hires[1-4]
#BSUB -q hpc
#BSUB -W 4:00
#BSUB -n 4
#BSUB -R "rusage[mem=8GB]"
#BSUB -o /zhome/45/8/156251/Desktop/thesis/labassist/hpc_scripts/logs/combined_pcr/%J_%I_extract_hires.out
#BSUB -e /zhome/45/8/156251/Desktop/thesis/labassist/hpc_scripts/logs/combined_pcr/%J_%I_extract_hires.err

###############################################################################
# Extract High-Resolution Preprocessed Frames
#
# This extracts frames at high resolution (1080p) with crop and flip already
# applied, so YOLO can read them directly without video decompression overhead.
#
# Key differences from original frame extraction:
# - HIGH RESOLUTION: 1080p vs 256p short side
# - PREPROCESSED: Crop + flip already applied (matching I3D preprocessing)
# - For YOLO: Better detection of small objects like PCR tubes
###############################################################################

echo "========================================"
echo "High-Resolution Frame Extraction"
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

# Output directory for high-res frames
FRAMES_OUT="${DATA_ROOT}/frames_hires_preprocessed"

# Extraction parameters
FPS="25"  # Same as I3D
MAX_SHORT_SIDE="1080"  # Keep 1080p for YOLO

# Crop parameters (MUST match I3D extraction!)
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
echo "  FRAMES_OUT: ${FRAMES_OUT}"
echo "  FPS: ${FPS}"
echo "  MAX_SHORT_SIDE: ${MAX_SHORT_SIDE}"
echo "  CROP: top=${CROP_TOP}, bottom=${CROP_BOTTOM}, left=${CROP_LEFT}, right=${CROP_RIGHT}"
echo "  SHARD: ${SHARD_IDX}/${NUM_SHARDS}"
echo ""

# Run extraction
python /zhome/45/8/156251/Desktop/thesis/labassist/hpc_scripts/scripts/feature_computations/extract_frames_hires_preprocessed.py \
  --root "${DATA_ROOT}" \
  --splits "${SPLITS}" \
  --fps "${FPS}" \
  --max_short_side "${MAX_SHORT_SIDE}" \
  --out "${FRAMES_OUT}" \
  --crop_top "${CROP_TOP}" \
  --crop_bottom "${CROP_BOTTOM}" \
  --crop_left "${CROP_LEFT}" \
  --crop_right "${CROP_RIGHT}" \
  --selective_crop \
  --vertical_flip \
  --shards "${NUM_SHARDS}" \
  --shard_idx "${SHARD_IDX}"

echo ""
echo "========================================"
echo "Frame extraction (shard ${SHARD_IDX}) complete at $(date)"
echo "========================================"

