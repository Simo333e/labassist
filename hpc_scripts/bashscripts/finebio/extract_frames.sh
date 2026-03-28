#!/bin/sh
#BSUB -q hpc
#BSUB -J finebio_frames[1-8]
#BSUB -n 4
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=8GB]"
#BSUB -M 8GB
#BSUB -W 8:00
#BSUB -o /zhome/45/8/156251/Desktop/thesis/labassist/hpc_scripts/logs/finebio/%J_%I_frames.out
#BSUB -e /zhome/45/8/156251/Desktop/thesis/labassist/hpc_scripts/logs/finebio/%J_%I_frames.err

set -euo pipefail

# ============================================================
# Extract frames from FineBio T5 videos at 25fps
# ============================================================

VIDEO_DIR="/work3/s204157/data/labassist/finebio_videos_w640"
OUTPUT_DIR="/work3/s204157/data/labassist/finebio_full/frames"
FPS="25"
VIEW="T5"

# Sharding
SHARD_IDX=$(($LSB_JOBINDEX - 1))
NUM_SHARDS=8

mkdir -p "${OUTPUT_DIR}"
mkdir -p /zhome/45/8/156251/Desktop/thesis/labassist/hpc_scripts/logs/finebio

# Activate environment
if [ -d "/zhome/45/8/156251/Desktop/thesis/labassist/.env" ]; then
  . "/zhome/45/8/156251/Desktop/thesis/labassist/.env/bin/activate"
fi

echo "============================================================"
echo "FineBio Frame Extraction - Shard ${SHARD_IDX}/${NUM_SHARDS}"
echo "============================================================"
echo "Video dir: ${VIDEO_DIR}"
echo "Output dir: ${OUTPUT_DIR}"
echo "FPS: ${FPS}"
echo "View: ${VIEW}"
echo "============================================================"

python /zhome/45/8/156251/Desktop/thesis/labassist/hpc_scripts/scripts/data_prep/extract_finebio_frames.py \
    --video_dir "${VIDEO_DIR}" \
    --output_dir "${OUTPUT_DIR}" \
    --fps "${FPS}" \
    --view "${VIEW}" \
    --shards "${NUM_SHARDS}" \
    --shard_idx "${SHARD_IDX}"

echo ""
echo "✓ Shard ${SHARD_IDX} complete"

