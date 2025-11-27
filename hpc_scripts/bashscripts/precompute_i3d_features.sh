#!/bin/sh
#BSUB -q gpuv100
#BSUB -J precompute_i3d_10hz
#BSUB -gpu "num=1"
#BSUB -n 4
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=24GB]"
#BSUB -M 24GB
#BSUB -W 12:00
#BSUB -o /zhome/45/8/156251/Desktop/thesis/labassist/hpc_scripts/logs/precompute/i3d/%J.out
#BSUB -e /zhome/45/8/156251/Desktop/thesis/labassist/hpc_scripts/logs/precompute/i3d/%J.err

set -euo pipefail

if [ -d "/zhome/45/8/156251/Desktop/thesis/labassist/.env" ]; then
  . "/zhome/45/8/156251/Desktop/thesis/labassist/.env/bin/activate"
else
  python -m venv "$HOME/venvs/finebio"
  . "$HOME/venvs/finebio/bin/activate"
fi

python /zhome/45/8/156251/Desktop/thesis/labassist/hpc_scripts/scripts/precompute_features_i3d.py \
  --root /work3/s204157/data/finebio \
  --splits /work3/s204157/data/finebio/splits/splits_v1_sane.json \
  --view T1 --fps 10 --win 32 --stride 2 --imgsz 160 \
  --batch_size 16 --center_step 1 \
  --device cuda:0 \
  --out /work3/s204157/data/finebio/cache/features_10hz_i3d