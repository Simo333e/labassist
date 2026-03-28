#!/bin/sh
#BSUB -q hpc
#BSUB -J finebio_labels
#BSUB -n 1
#BSUB -R "rusage[mem=4GB]"
#BSUB -M 4GB
#BSUB -W 1:00
#BSUB -o /zhome/45/8/156251/Desktop/thesis/labassist/hpc_scripts/logs/finebio/%J_labels.out
#BSUB -e /zhome/45/8/156251/Desktop/thesis/labassist/hpc_scripts/logs/finebio/%J_labels.err

set -euo pipefail

# ============================================================
# Convert FineBio atomic annotations to MS-TCN format
# ============================================================

ANNOTATION_DIR="/work3/s204157/data/labassist/annotations/finebio_action_annotations"
FRAMES_DIR="/work3/s204157/data/labassist/finebio_full/frames"
OUTPUT_DIR="/work3/s204157/data/labassist/finebio_full/cache"
FPS="10.0"
FRAME_FPS="25.0"
VIEW="T5"

mkdir -p "${OUTPUT_DIR}"

# Activate environment
if [ -d "/zhome/45/8/156251/Desktop/thesis/labassist/.env" ]; then
  . "/zhome/45/8/156251/Desktop/thesis/labassist/.env/bin/activate"
fi

echo "============================================================"
echo "FineBio Atomic Label Conversion"
echo "============================================================"
echo "Annotation dir: ${ANNOTATION_DIR}"
echo "Frames dir: ${FRAMES_DIR}"
echo "Output dir: ${OUTPUT_DIR}"
echo "Label FPS: ${FPS}"
echo "View: ${VIEW}"
echo "============================================================"

python /zhome/45/8/156251/Desktop/thesis/labassist/hpc_scripts/scripts/data_prep/convert_finebio_atomic_to_mstcn.py \
    --annotation_dir "${ANNOTATION_DIR}" \
    --frames_dir "${FRAMES_DIR}" \
    --output_dir "${OUTPUT_DIR}" \
    --fps "${FPS}" \
    --frame_fps "${FRAME_FPS}" \
    --view "${VIEW}"

echo ""
echo "✓ Label conversion complete"

