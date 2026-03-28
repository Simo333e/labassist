#!/bin/sh
#BSUB -q gpua100
#BSUB -J vlm_images
#BSUB -gpu "num=1"
#BSUB -n 4
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=32GB]"
#BSUB -M 32GB
#BSUB -W 1:00
#BSUB -o /zhome/45/8/156251/Desktop/thesis/labassist/hpc_scripts/logs/vlm/%J_vlm_images.out
#BSUB -e /zhome/45/8/156251/Desktop/thesis/labassist/hpc_scripts/logs/vlm/%J_vlm_images.err

set -euo pipefail

# HuggingFace cache
export HF_HOME="/work3/s204157/data/labassist/vlms"
export TRANSFORMERS_CACHE="/work3/s204157/data/labassist/vlms"

# Activate environment
if [ -d "/zhome/45/8/156251/Desktop/thesis/labassist/.env" ]; then
  . "/zhome/45/8/156251/Desktop/thesis/labassist/.env/bin/activate"
fi

# Videos to test
VIDEOS=(
    "/work3/s204157/data/labassist/combined_pcr/videos/P02_05_01_T5.mp4"
    "/work3/s204157/data/labassist/combined_pcr/videos/20251125_072321Z.mp4"
)

OUTPUT_DIR="/zhome/45/8/156251/Desktop/thesis/labassist/test/vlm_results"
mkdir -p "${OUTPUT_DIR}"

echo "Starting VLM image-based test..."
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"

for VIDEO in "${VIDEOS[@]}"; do
    STEM=$(basename "${VIDEO}" .mp4)
    echo ""
    echo "=============================================="
    echo "Testing: ${STEM}"
    echo "=============================================="
    
    python /zhome/45/8/156251/Desktop/thesis/labassist/test/test_vlm_images.py \
        --video "${VIDEO}" \
        --num_frames 8 \
        --output "${OUTPUT_DIR}/${STEM}_images_result.json"
done

echo ""
echo "Done! Results saved to ${OUTPUT_DIR}"