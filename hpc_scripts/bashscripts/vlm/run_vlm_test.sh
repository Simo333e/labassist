#!/bin/sh
#BSUB -q gpua100       # or gpuv100 if a100 unavailable
#BSUB -J vlm_zeroshot
#BSUB -gpu "num=1"
#BSUB -n 4
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=32GB]"
#BSUB -M 32GB
#BSUB -W 1:00
#BSUB -o /zhome/45/8/156251/Desktop/thesis/labassist/hpc_scripts/logs/vlm/%J_vlm_zeroshot.out
#BSUB -e /zhome/45/8/156251/Desktop/thesis/labassist/hpc_scripts/logs/vlm/%J_vlm_zeroshot.err

set -euo pipefail

# Activate your environment
if [ -d "/zhome/45/8/156251/Desktop/thesis/labassist/.env" ]; then
  . "/zhome/45/8/156251/Desktop/thesis/labassist/.env/bin/activate"
fi

export HF_HOME="/work3/s204157/data/labassist/vlms"
export TRANSFORMERS_CACHE="/work3/s204157/data/labassist/vlms"

# Videos to test (pick 2-3 from validation set)
VIDEOS=(
    "/work3/s204157/data/labassist/combined_pcr/videos/P02_05_01_T5.mp4"
    "/work3/s204157/data/labassist/combined_pcr/videos/20251125_072321Z.mp4"
)

OUTPUT_DIR="/zhome/45/8/156251/Desktop/thesis/labassist/test/vlm_results"
mkdir -p "${OUTPUT_DIR}"

echo "Starting VLM zero-shot test..."
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"

for VIDEO in "${VIDEOS[@]}"; do
    STEM=$(basename "${VIDEO}" .mp4)
    echo ""
    echo "=============================================="
    echo "Testing: ${STEM}"
    echo "=============================================="
    
    python /zhome/45/8/156251/Desktop/thesis/labassist/test/test_vlm_zeroshot.py \
        --video "${VIDEO}" \
        --prompt specific \
        --max_frames 32 \
        --output "${OUTPUT_DIR}/${STEM}_result.json"
done

echo ""
echo "Done! Results saved to ${OUTPUT_DIR}"