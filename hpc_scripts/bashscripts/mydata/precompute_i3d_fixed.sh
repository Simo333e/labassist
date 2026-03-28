#!/bin/bash
#BSUB -J precompute_i3d_mydata_fixed[1-8]
#BSUB -q gpuv100
#BSUB -n 4
#BSUB -R "rusage[mem=16G]"
#BSUB -R "span[hosts=1]"
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -W 2:00
#BSUB -o /zhome/45/8/156251/Desktop/thesis/labassist/hpc_scripts/logs/mydata/%J_precompute_i3d.out
#BSUB -e /zhome/45/8/156251/Desktop/thesis/labassist/hpc_scripts/logs/mydata/%J_precompute_i3d.err

# Compute I3D features from FIXED mydata (original videos, correct labels)
# Includes: vertical flip (viewpoint alignment) + crop (focus on work area)

set -euo pipefail

# Activate environment
if [ -d "/zhome/45/8/156251/Desktop/thesis/labassist/.env" ]; then
  . "/zhome/45/8/156251/Desktop/thesis/labassist/.env/bin/activate"
fi

SCRIPTS=/zhome/45/8/156251/Desktop/thesis/labassist/hpc_scripts/scripts
MYDATA=/work3/s204157/data/labassist/my_data
I3D_WEIGHTS=/work3/s204157/models/i3d/rgb_imagenet.pt

# Convert 1-based LSF index to 0-based shard index
SHARD_IDX=$(($LSB_JOBINDEX - 1))
NUM_SHARDS=8

# Use the FIXED cache with correct labels
SPLITS=${MYDATA}/cache_fixed/splits.json
# Output features
OUTDIR=${MYDATA}/cache_fixed/features_10hz_i3d_vflip_crop_win16

echo "=== Computing I3D features for FIXED mydata ==="
echo "Frames: ${MYDATA}/frames"
echo "Splits: ${SPLITS}"
echo "Output: ${OUTDIR}"
echo ""

python ${SCRIPTS}/feature_computations/precompute_features_i3d.py \
    --root ${MYDATA} \
    --splits ${SPLITS} \
    --weights ${I3D_WEIGHTS} \
    --shards ${NUM_SHARDS} \
    --shard_idx ${SHARD_IDX} \
    --fps 10 \
    --win 16 \
    --stride 1 \
    --imgsz 224 \
    --batch_size 16 \
    --out ${OUTDIR} \
    --crop_top 0.0 \
    --crop_bottom 0.35 \
    --crop_left 0.0 \
    --crop_right 0.0 \
    --selective_crop \
    --vertical_flip

echo ""
echo "=== Feature extraction complete ==="
echo "Features saved to: ${OUTDIR}"


