#!/bin/bash
#BSUB -J vjepa_finebio_t5[1-8]
#BSUB -q gpua100
#BSUB -n 4
#BSUB -R "rusage[mem=32G]"
#BSUB -R "span[hosts=1]"
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -W 24:00
#BSUB -o /zhome/45/8/156251/Desktop/thesis/labassist/hpc_scripts/logs/finebio/%J_%I_vjepa.out
#BSUB -e /zhome/45/8/156251/Desktop/thesis/labassist/hpc_scripts/logs/finebio/%J_%I_vjepa.err

# Precompute V-JEPA features for ALL FineBio T5 (PCR) videos
# This enables pretraining on full FineBio before finetuning on mydata

set -euo pipefail

# Activate environment
source /zhome/45/8/156251/Desktop/thesis/labassist/.env/bin/activate

SCRIPTS=/zhome/45/8/156251/Desktop/thesis/labassist/hpc_scripts/scripts
FINEBIO_ROOT=/work3/s204157/data/labassist/finebio_full
FINEBIO_VIDEOS=/work3/s204157/data/labassist/finebio_videos_w640

# Output directory for V-JEPA features
OUTDIR=/work3/s204157/data/labassist/finebio_full/cache/features_10hz_vjepa

# Splits file with all T5 videos
SPLITS=/work3/s204157/data/labassist/finebio_full/splits_t5_all.json

# Convert 1-based LSF index to 0-based shard index
SHARD_IDX=$((LSB_JOBINDEX - 1))
NUM_SHARDS=8

echo "=== V-JEPA Feature Extraction for FineBio T5 ==="
echo "Shard: ${SHARD_IDX}/${NUM_SHARDS}"
echo "Videos dir: ${FINEBIO_VIDEOS}"
echo "Output: ${OUTDIR}"
echo ""

# Check if splits file exists
if [ ! -f "${SPLITS}" ]; then
    echo "ERROR: Splits file not found: ${SPLITS}"
    echo "Please run the setup script first to create the splits file."
    exit 1
fi

python ${SCRIPTS}/feature_computations/precompute_features_vjepa.py \
    --root ${FINEBIO_VIDEOS} \
    --splits ${SPLITS} \
    --fps 10 \
    --frames_per_clip 16 \
    --stride 1 \
    --batch_size 4 \
    --out ${OUTDIR} \
    --shards ${NUM_SHARDS} \
    --shard_idx ${SHARD_IDX}

echo ""
echo "=== V-JEPA extraction complete for shard ${SHARD_IDX} ==="
