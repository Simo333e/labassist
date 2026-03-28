#!/bin/bash
#BSUB -q gpuv100
#BSUB -J resnet_masked_mydata
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -n 4
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=16GB]"
#BSUB -W 4:00
#BSUB -o /zhome/45/8/156251/Desktop/thesis/labassist/hpc_scripts/logs/mydata/%J_resnet_motion_masked.out
#BSUB -e /zhome/45/8/156251/Desktop/thesis/labassist/hpc_scripts/logs/mydata/%J_resnet_motion_masked.err

# Precompute ResNet18 + Motion features WITH MASKING for mydata (6 videos)
# Applies vertical flip + cropping to align with FineBio
# Uses morphological masking to reduce noise

set -euo pipefail

# Activate environment
source /zhome/45/8/156251/Desktop/thesis/labassist/.env/bin/activate

SCRIPTS=/zhome/45/8/156251/Desktop/thesis/labassist/hpc_scripts/scripts
MYDATA_ROOT=/work3/s204157/data/labassist/my_data

# Output directory for mydata ResNet + motion features (with masking)
OUTDIR=/work3/s204157/data/labassist/my_data/cache/features_10hz_resnet18_motion_masked_crop_flip

SPLITS=${MYDATA_ROOT}/splits.json

echo "=== ResNet18 + Motion (MASKED) Feature Extraction for mydata ==="
echo "Videos dir: ${MYDATA_ROOT}/videos"
echo "Output: ${OUTDIR}"
echo "Preprocessing: crop + vflip (to align with FineBio)"
echo "Masking: threshold=15, kernel=5"
echo ""

# Check if splits file exists
if [ ! -f "${SPLITS}" ]; then
    echo "ERROR: Splits file not found: ${SPLITS}"
    exit 1
fi

python ${SCRIPTS}/feature_computations/precompute_features_resnet18_motion_v2.py \
    --root ${MYDATA_ROOT} \
    --splits ${SPLITS} \
    --fps 10 \
    --batch_size 32 \
    --out ${OUTDIR} \
    --motion \
    --motion_threshold 15 \
    --motion_morph_kernel 5 \
    --crop_top 0.36 \
    --crop_bottom 0.02 \
    --crop_left 0.02 \
    --crop_right 0.32 \
    --vertical_flip \
    --selective_crop

echo ""
echo "=== ResNet+Motion (masked) extraction complete for mydata ==="




