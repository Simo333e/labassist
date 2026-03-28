#!/bin/bash
#BSUB -J resnet_finebio_t5[1-8]
#BSUB -q gpuv100
#BSUB -n 4
#BSUB -R "rusage[mem=16G]"
#BSUB -R "span[hosts=1]"
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -W 24:00
#BSUB -o /zhome/45/8/156251/Desktop/thesis/labassist/hpc_scripts/logs/finebio/%J_%I_resnet_motion.out
#BSUB -e /zhome/45/8/156251/Desktop/thesis/labassist/hpc_scripts/logs/finebio/%J_%I_resnet_motion.err

# Precompute ResNet18 + Motion features for ALL FineBio T5 (PCR) videos
# This enables pretraining on full FineBio before finetuning on mydata

set -euo pipefail

# Activate environment
source /zhome/45/8/156251/Desktop/thesis/labassist/.env/bin/activate

SCRIPTS=/zhome/45/8/156251/Desktop/thesis/labassist/hpc_scripts/scripts
FINEBIO_VIDEOS=/work3/s204157/data/labassist/finebio_videos_w640

# Output directory for ResNet + motion features
OUTDIR=/work3/s204157/data/labassist/finebio_full/cache/features_10hz_resnet18_motion

# Splits file with ALL T5 videos (all tasks, 221 videos)
# This is used for pretraining on the full FineBio dataset
SPLITS=/work3/s204157/data/labassist/finebio_full/splits_t5_all.json

# Convert 1-based LSF index to 0-based shard index
SHARD_IDX=$((LSB_JOBINDEX - 1))
NUM_SHARDS=8

echo "=== ResNet18 + Motion Feature Extraction for FineBio T5 ==="
echo "Shard: ${SHARD_IDX}/${NUM_SHARDS}"
echo "Videos dir: ${FINEBIO_VIDEOS}"
echo "Output: ${OUTDIR}"
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
    --shards ${NUM_SHARDS} \
    --shard_idx ${SHARD_IDX}

echo ""
echo "=== ResNet+Motion extraction complete for shard ${SHARD_IDX} ==="
