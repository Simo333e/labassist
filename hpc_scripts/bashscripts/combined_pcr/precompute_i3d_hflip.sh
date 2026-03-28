#!/bin/bash
#BSUB -J precompute_i3d_hflip[1-8]
#BSUB -q gpua100
#BSUB -n 4
#BSUB -R "rusage[mem=16G]"
#BSUB -R "span[hosts=1]"
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -W 4:00
#BSUB -o /zhome/45/8/156251/Desktop/thesis/labassist/hpc_scripts/logs/combined_pcr/precompute_i3d_hflip_%I.out
#BSUB -e /zhome/45/8/156251/Desktop/thesis/labassist/hpc_scripts/logs/combined_pcr/precompute_i3d_hflip_%I.err

# Precompute I3D features with horizontal flip augmentation for mydata
# This creates features with both vertical AND horizontal flip for mydata videos
# to make the model invariant to left/right handedness

set -euo pipefail

# Activate environment
if [ -d "/zhome/45/8/156251/Desktop/thesis/labassist/.env" ]; then
  . "/zhome/45/8/156251/Desktop/thesis/labassist/.env/bin/activate"
fi

SCRIPTS=/zhome/45/8/156251/Desktop/thesis/labassist/hpc_scripts/scripts
DATA=/work3/s204157/data/labassist/combined_pcr
FRAMES=/work3/s204157/data/labassist/combined_pcr/frames
I3D_WEIGHTS=/work3/s204157/models/i3d/rgb_imagenet.pt

# Output: features with crop + vertical flip + horizontal flip, short window
OUTDIR=${DATA}/cache/features_10hz_i3d_selective_crop_vhflip_win16

# Use the splits that exclude the broken pipette video
SPLITS=${DATA}/cache/splits_exclude_broken.json

# Convert 1-based job index to 0-based shard index
SHARD_IDX=$((LSB_JOBINDEX - 1))

echo "=== Precomputing I3D features with horizontal flip (shard ${SHARD_IDX}/8) ==="
echo "Output: ${OUTDIR}"
echo "Splits: ${SPLITS}"

python ${SCRIPTS}/feature_computations/precompute_features_i3d.py \
    --root ${DATA} \
    --splits ${SPLITS} \
    --weights ${I3D_WEIGHTS} \
    --fps 10 \
    --win 16 \
    --stride 1 \
    --imgsz 224 \
    --batch_size 16 \
    --out ${OUTDIR} \
    --shards 8 \
    --shard_idx ${SHARD_IDX} \
    --crop_top 0.0 \
    --crop_bottom 0.35 \
    --crop_left 0.0 \
    --crop_right 0.0 \
    --selective_crop \
    --vertical_flip \
    --horizontal_flip

echo "=== Done (shard ${SHARD_IDX}) ==="
