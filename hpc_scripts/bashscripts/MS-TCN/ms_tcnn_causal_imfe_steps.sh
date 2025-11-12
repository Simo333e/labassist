#!/bin/sh
#BSUB -q gpuv100
#BSUB -J ms_tcnn_causal_imfe_steps
#BSUB -gpu "num=1"
#BSUB -n 8
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=12GB]"
#BSUB -M 12GB
#BSUB -W 12:00
#BSUB -o /zhome/45/8/156251/Desktop/thesis/labassist/hpc_scripts/logs/ms_tcn/imfe/%J.out
#BSUB -e /zhome/45/8/156251/Desktop/thesis/labassist/hpc_scripts/logs/ms_tcn/imfe/%J.err

set -euo pipefail

EXP_DIR="/work3/s204157/experiments/ms_tcn_causal_steps_imfe_v1"
mkdir -p "${EXP_DIR}/logs" "${EXP_DIR}/tb"

if [ -d "/zhome/45/8/156251/Desktop/thesis/labassist/.env" ]; then
  . "/zhome/45/8/156251/Desktop/thesis/labassist/.env/bin/activate"
else
  python -m venv "$HOME/venvs/finebio"
  . "$HOME/venvs/finebio/bin/activate"
fi

python -m pip install --upgrade pip
python -m pip install pandas
python -m pip install tensorboard || true

EXP=/work3/s204157/experiments/ms_tcn_causal_steps_imfe_v2
python /zhome/45/8/156251/Desktop/thesis/labassist/hpc_scripts/MS-TCN/train.py \
  --causal \
  --run_dir /work3/s204157 \
  --bz 1 --lr 5e-4 \
  --num_f_maps 64 --num_epochs 150 \
  --num_layers_PG 10 --num_layers_R 10 --num_R 5 \
  --splits /work3/s204157/data/finebio/splits/splits_v1_sane.json \
  --xdir /work3/s204157/data/finebio/cache/features_10hz_imfe \
  --ydir /work3/s204157/data/finebio/cache/labels_10hz_steps \
  --class_index /work3/s204157/data/finebio/cache/class_index_steps.json \
  --exp_dir "$EXP"