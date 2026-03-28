#!/bin/bash
#BSUB -J resnet_masked_finebio[1-8]
#BSUB -q gpuv100
#BSUB -n 4
#BSUB -R "rusage[mem=16G]"
#BSUB -R "span[hosts=1]"
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -W 24:00
#BSUB -o /zhome/45/8/156251/Desktop/thesis/labassist/hpc_scripts/logs/finebio/%J_%I_resnet_motion_masked.out
#BSUB -e /zhome/45/8/156251/Desktop/thesis/labassist/hpc_scripts/logs/finebio/%J_%I_resnet_motion_masked.err

# Precompute ResNet18 + Motion features WITH MASKING for ALL FineBio videos
# Uses morphological masking to reduce noise and focus on actual movements

set -euo pipefail

# Activate environment
source /zhome/45/8/156251/Desktop/thesis/labassist/.env/bin/activate

SCRIPTS=/zhome/45/8/156251/Desktop/thesis/labassist/hpc_scripts/scripts
FINEBIO_VIDEOS=/work3/s204157/data/labassist/finebio_videos_w640

# Output directory for ResNet + motion features (with masking)
OUTDIR=/work3/s204157/data/labassist/finebio_full/cache/features_10hz_resnet18_motion_masked

# Splits file with ALL T5 videos (all tasks, 221 videos)
SPLITS=/work3/s204157/data/labassist/finebio_full/splits_t5_all.json

# Convert 1-based LSF index to 0-based shard index
SHARD_IDX=$((LSB_JOBINDEX - 1))
NUM_SHARDS=8

echo "=== ResNet18 + Motion (MASKED) Feature Extraction for FineBio ==="
echo "Shard: ${SHARD_IDX}/${NUM_SHARDS}"
echo "Videos dir: ${FINEBIO_VIDEOS}"
echo "Output: ${OUTDIR}"
echo "Masking: threshold=15, kernel=5"
echo ""

# Check if splits file exists
if [ ! -f "${SPLITS}" ]; then
    echo "ERROR: Splits file not found: ${SPLITS}"
    echo "Please run setup_finebio_t5_splits.sh first"
    exit 1
fi

python ${SCRIPTS}/feature_computations/precompute_features_resnet18_motion_v2.py \
    --root ${FINEBIO_VIDEOS} \
    --splits ${SPLITS} \
    --fps 10 \
    --batch_size 32 \
    --out ${OUTDIR} \
    --motion \
    --motion_threshold 15 \
    --motion_morph_kernel 5 \
    --shards ${NUM_SHARDS} \
    --shard_idx ${SHARD_IDX}

echo ""
echo "=== ResNet+Motion (masked) extraction complete for shard ${SHARD_IDX} ==="




