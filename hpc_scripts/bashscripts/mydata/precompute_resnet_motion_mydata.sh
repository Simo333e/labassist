#!/bin/bash
#BSUB -J resnet_mydata[1-6]
#BSUB -q gpuv100
#BSUB -n 4
#BSUB -R "rusage[mem=32G]"
#BSUB -R "span[hosts=1]"
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -W 06:00
#BSUB -o /zhome/45/8/156251/Desktop/thesis/labassist/hpc_scripts/logs/mydata/%J_%I_resnet_motion.out
#BSUB -e /zhome/45/8/156251/Desktop/thesis/labassist/hpc_scripts/logs/mydata/%J_%I_resnet_motion.err

# Precompute ResNet18 + Motion features for mydata videos
# With vertical flip (to align with FineBio viewpoint) and bottom crop (to remove glass shelf)

set -euo pipefail

# Activate environment
source /zhome/45/8/156251/Desktop/thesis/labassist/.env/bin/activate

SCRIPTS=/zhome/45/8/156251/Desktop/thesis/labassist/hpc_scripts/scripts
MYDATA=/work3/s204157/data/labassist/my_data

# Output directory for ResNet + motion features (with vflip and crop)
OUTDIR=${MYDATA}/cache_fixed/features_10hz_resnet18_motion_vflip_crop

# Splits file with all mydata videos
SPLITS=${MYDATA}/splits.json

# Convert 1-based LSF index to 0-based shard index
SHARD_IDX=$((LSB_JOBINDEX - 1))
NUM_SHARDS=6

echo "=== ResNet18 + Motion Feature Extraction for mydata ==="
echo "Shard: ${SHARD_IDX}/${NUM_SHARDS}"
echo "Root: ${MYDATA}"
echo "Output: ${OUTDIR}"
echo "Transforms: vertical_flip + crop_bottom=0.35 + crop_left=0.10 + crop_right=0.10"
echo ""

# Check if splits file exists
if [ ! -f "${SPLITS}" ]; then
    echo "ERROR: Splits file not found: ${SPLITS}"
    exit 1
fi

python ${SCRIPTS}/feature_computations/precompute_features_resnet18_motion_v2.py \
    --root ${MYDATA} \
    --splits ${SPLITS} \
    --fps 10 \
    --batch_size 16 \
    --out ${OUTDIR} \
    --motion \
    --shards ${NUM_SHARDS} \
    --shard_idx ${SHARD_IDX} \
    --crop_top 0.0 \
    --crop_bottom 0.35 \
    --crop_left 0.10 \
    --crop_right 0.10 \
    --selective_crop \
    --vertical_flip

echo ""
echo "=== ResNet+Motion extraction complete for shard ${SHARD_IDX} ==="
