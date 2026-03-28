#!/bin/bash
#BSUB -J extract_mydata_frames
#BSUB -q gpuv100
#BSUB -n 4
#BSUB -R "rusage[mem=8G]"
#BSUB -R "span[hosts=1]"
#BSUB -W 4:00
#BSUB -o /zhome/45/8/156251/Desktop/thesis/labassist/hpc_scripts/logs/mydata/%J_extract_frames.out
#BSUB -e /zhome/45/8/156251/Desktop/thesis/labassist/hpc_scripts/logs/mydata/%J_extract_frames.err

# Extract frames from ORIGINAL (untrimmed) mydata videos
# This fixes the label alignment issue caused by using trimmed videos

set -euo pipefail

module load ffmpeg/7.1

# Paths
MYDATA_VIDEOS=/work3/s204157/data/labassist/my_data/videos
OUTPUT_FRAMES=/work3/s204157/data/labassist/my_data/frames

mkdir -p ${OUTPUT_FRAMES}
mkdir -p /zhome/45/8/156251/Desktop/thesis/labassist/hpc_scripts/logs/mydata

echo "=== Extracting frames from ORIGINAL mydata videos ==="
echo "Source: ${MYDATA_VIDEOS}"
echo "Output: ${OUTPUT_FRAMES}"
echo ""

# Extract frames at 25fps for each video
for video in ${MYDATA_VIDEOS}/*.mp4; do
    stem=$(basename "$video" .mp4)
    outdir="${OUTPUT_FRAMES}/${stem}"
    
    if [ -d "${outdir}" ] && [ "$(ls -A ${outdir})" ]; then
        echo "⏭ Skipping ${stem} (frames already exist)"
        continue
    fi
    
    mkdir -p "${outdir}"
    
    echo "📹 Extracting frames from ${stem}..."
    ffmpeg -i "${video}" \
        -vf "fps=25,scale=-1:256" \
        -q:v 2 \
        "${outdir}/%06d.jpg" \
        -hide_banner -loglevel warning
    
    count=$(ls -1 "${outdir}" | wc -l)
    echo "  ✓ Extracted ${count} frames → ${outdir}"
done

echo ""
echo "=== Frame extraction complete ==="
echo "Frames saved to: ${OUTPUT_FRAMES}"


