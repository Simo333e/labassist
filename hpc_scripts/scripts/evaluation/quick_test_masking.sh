#!/bin/bash
# Quick test to visualize motion masking on a sample video
# Run this before full feature extraction to verify masking parameters

set -euo pipefail

source /zhome/45/8/156251/Desktop/thesis/labassist/.env/bin/activate

# Sample FineBio video
SAMPLE_VIDEO="/work3/s204157/data/labassist/my_data/videos/Lab_video_cut.mp4"

# Output directory for test
OUTDIR="/zhome/45/8/156251/Desktop/thesis/labassist/hpc_scripts/eval_data/motion_masking_test"
mkdir -p ${OUTDIR}

echo "=== Testing Motion Masking on Sample Video ==="
echo "Video: ${SAMPLE_VIDEO}"
echo "Output: ${OUTDIR}"
echo ""

# Test different threshold values
for threshold in 10 15 20 25; do
    echo "Testing threshold=${threshold}..."
    
    # Simple view: frames + raw diff + resnet input
    python /zhome/45/8/156251/Desktop/thesis/labassist/hpc_scripts/scripts/feature_computations/visualize_motion_masking.py \
        --video ${SAMPLE_VIDEO} \
        --frame1 52300 \
        --frame2 52305 \
        --threshold ${threshold} \
        --kernel 5 \
        --simple \
        --output ${OUTDIR}/simple_threshold${threshold}.png
    
    # Detailed view: full pipeline with masking
    python /zhome/45/8/156251/Desktop/thesis/labassist/hpc_scripts/scripts/feature_computations/visualize_motion_masking.py \
        --video ${SAMPLE_VIDEO} \
        --frame1 52300 \
        --frame2 52305 \
        --threshold ${threshold} \
        --kernel 5 \
        --output ${OUTDIR}/detailed_threshold${threshold}.png
done

echo ""
echo "=== Test Complete ==="
echo "Check visualizations in: ${OUTDIR}"
echo ""
echo "Generated views:"
echo "  - simple_threshold*.png: 4-panel view (frames, raw diff, ResNet input)"
echo "  - detailed_threshold*.png: 8-panel view (full masking pipeline)"
echo ""
echo "Look for threshold that:"
echo "  ✓ Removes noise/artifacts in raw difference"
echo "  ✓ Preserves actual hand/pipette motion"
echo "  ✓ Creates clean ResNet input (224x224 normalized)"
echo ""
echo "Choose threshold that best removes noise while preserving hand/pipette motion"

