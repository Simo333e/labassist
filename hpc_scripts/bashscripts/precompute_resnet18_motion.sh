#!/bin/sh
#BSUB -q gpuv100
#BSUB -J precompute_resnet18_motion
#BSUB -gpu "num=1"
#BSUB -n 4
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=24GB]"
#BSUB -M 24GB
#BSUB -W 12:00
#BSUB -o /zhome/45/8/156251/Desktop/thesis/labassist/hpc_scripts/logs/precompute/resnet18_motion/%J.out
#BSUB -e /zhome/45/8/156251/Desktop/thesis/labassist/hpc_scripts/logs/precompute/resnet18_motion/%J.err

set -euo pipefail

if [ -d "/zhome/45/8/156251/Desktop/thesis/labassist/.env" ]; then
  . "/zhome/45/8/156251/Desktop/thesis/labassist/.env/bin/activate"
fi

# Step 1: Extract RGB + Motion features (1024D)
python /zhome/45/8/156251/Desktop/thesis/labassist/hpc_scripts/scripts/feature_computations/precompute_features_resnet18_motion.py \
  --root /work3/s204157/data/finebio \
  --splits /work3/s204157/data/finebio/splits/splits_v1_sane.json \
  --view T1 --fps 10 --imgsz 224 \
  --batch_size 32 \
  --motion \
  --device cuda:0 \
  --out /work3/s204157/data/finebio/cache/features_10hz_resnet18_motion

# Step 2: Stack temporal context (k=5 for 1024D -> 5120D)
python /zhome/45/8/156251/Desktop/thesis/labassist/hpc_scripts/scripts/feature_computations/stack_context_features.py \
  --src /work3/s204157/data/finebio/cache/features_10hz_resnet18_motion \
  --out /work3/s204157/data/finebio/cache/features_10hz_resnet18_motion_ctx5 \
  --k 5