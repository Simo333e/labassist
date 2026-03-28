#!/bin/sh
#BSUB -q gpua100
#BSUB -R "select[gpu80gb]"
#BSUB -J vlm_testdata
#BSUB -gpu "num=1"
#BSUB -n 4
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=48GB]"
#BSUB -M 48GB
#BSUB -W 4:00
#BSUB -o /zhome/45/8/156251/Desktop/thesis/labassist/hpc_scripts/logs/vlm/%J_vlm_testdata.out
#BSUB -e /zhome/45/8/156251/Desktop/thesis/labassist/hpc_scripts/logs/vlm/%J_vlm_testdata.err

set -euo pipefail

export HF_HOME="/work3/s204157/data/labassist/vlms"
export TRANSFORMERS_CACHE="/work3/s204157/data/labassist/vlms"

if [ -d "/zhome/45/8/156251/Desktop/thesis/labassist/.env" ]; then
  . "/zhome/45/8/156251/Desktop/thesis/labassist/.env/bin/activate"
fi

# Your test videos
DATA_ROOT="/work3/s204157/data/labassist/my_testdata"
VIDEOS=(
    "${DATA_ROOT}/test1cut.mp4"
    "${DATA_ROOT}/test2cut.mp4"
    "${DATA_ROOT}/test3cut.mp4"
    "${DATA_ROOT}/test4cut.mp4"
    "${DATA_ROOT}/test5.0cut.mp4"
    "${DATA_ROOT}/test5.5cut.mp4"
    "${DATA_ROOT}/test6cut.mp4"
)

OUTPUT_DIR="/zhome/45/8/156251/Desktop/thesis/labassist/test/vlm_results/my_testdata"
mkdir -p "${OUTPUT_DIR}"

echo "Starting VLM segment analysis on my_testdata..."
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "Processing ${#VIDEOS[@]} videos"

for VIDEO in "${VIDEOS[@]}"; do
    STEM=$(basename "${VIDEO}" .mp4)
    echo ""
    echo "=============================================="
    echo "Processing: ${STEM}"
    echo "=============================================="
    
    python /zhome/45/8/156251/Desktop/thesis/labassist/test/test_vlm_segments.py \
        --video "${VIDEO}" \
        --segment_duration 5 \
        --frames_per_segment 30 \
        --max_size 480 \
        --output "${OUTPUT_DIR}/${STEM}_analysis.json"
done

echo ""
echo "=============================================="
echo "All videos processed!"
echo "Results saved to: ${OUTPUT_DIR}"
echo "=============================================="