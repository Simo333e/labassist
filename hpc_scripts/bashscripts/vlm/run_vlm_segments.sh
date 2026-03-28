#!/bin/sh
#BSUB -q gpuv100
#BSUB -J vlm_segments
#BSUB -gpu "num=1"
#BSUB -n 4
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=48GB]"
#BSUB -M 48GB
#BSUB -W 4:00
#BSUB -o /zhome/45/8/156251/Desktop/thesis/labassist/hpc_scripts/logs/vlm/%J_vlm_segments.out
#BSUB -e /zhome/45/8/156251/Desktop/thesis/labassist/hpc_scripts/logs/vlm/%J_vlm_segments.err

set -euo pipefail

export HF_HOME="/work3/s204157/data/labassist/vlms"
export TRANSFORMERS_CACHE="/work3/s204157/data/labassist/vlms"

if [ -d "/zhome/45/8/156251/Desktop/thesis/labassist/.env" ]; then
  . "/zhome/45/8/156251/Desktop/thesis/labassist/.env/bin/activate"
fi

# Your trimmed mydata videos - add paths here
VIDEOS=(
    "/work3/s204157/data/labassist/combined_pcr/videos/20251125_072321Z.mp4"
    # Add more trimmed videos here
)

OUTPUT_DIR="/zhome/45/8/156251/Desktop/thesis/labassist/test/vlm_results/segments"
mkdir -p "${OUTPUT_DIR}"

echo "Starting segmented VLM analysis..."
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"

for VIDEO in "${VIDEOS[@]}"; do
    STEM=$(basename "${VIDEO}" .mp4)
    echo ""
    echo "=============================================="
    echo "Processing: ${STEM}"
    echo "=============================================="
    
    python /zhome/45/8/156251/Desktop/thesis/labassist/test/test_vlm_segments.py \
        --video "${VIDEO}" \
        --segment_duration 60 \
        --frames_per_segment 8 \
        --output "${OUTPUT_DIR}/${STEM}_segments.json"
done

echo ""
echo "Done! Results saved to ${OUTPUT_DIR}"