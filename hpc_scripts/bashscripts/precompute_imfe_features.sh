#!/bin/sh
#BSUB -q gpuv100
#BSUB -J precompute_imfe_10hz
#BSUB -gpu "num=1"
#BSUB -n 4
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=24GB]"
#BSUB -M 24GB
#BSUB -W 12:00
#BSUB -o /zhome/45/8/156251/Desktop/thesis/labassist/hpc_scripts/logs/precompute/IMFE/%J.out
#BSUB -e /zhome/45/8/156251/Desktop/thesis/labassist/hpc_scripts/logs/precompute/IMFE/%J.err

set -euo pipefail

if [ -d "/zhome/45/8/156251/Desktop/thesis/labassist/.env" ]; then
  . "/zhome/45/8/156251/Desktop/thesis/labassist/.env/bin/activate"
else
  python -m venv "$HOME/venvs/imfe"
  . "$HOME/venvs/imfe/bin/activate"
fi


# Paths
ROOT=/work3/s204157/data/finebio
SPLITS=/work3/s204157/data/finebio/splits/splits_v1_sane.json
VIEW=T1
FPS=10
FRAMES_DIR=/work3/s204157/data/finebio/cache/imfe_frames_${VIEW}_${FPS}hz
IMFE_REPO=/work3/s204157/repos/RT-HARE
IMFE_CKPT='/zhome/45/8/156251/Desktop/thesis/labassist/2024-02-04-04-13-03-checkpoint-8.pt'
OUT_CACHE=/work3/s204157/data/finebio/cache/features_${FPS}hz_imfe

mkdir -p "$(dirname "$IMFE_REPO")" "$FRAMES_DIR" "$OUT_CACHE"

IMFE_CKPT_DIR="$(dirname "$IMFE_CKPT")"
mkdir -p "$IMFE_CKPT_DIR"


if [ ! -d "$IMFE_REPO/.git" ]; then
  git clone https://github.com/rickywrq/RT-HARE.git "$IMFE_REPO"
fi

echo "[Step 1] Extract frames..."

python /zhome/45/8/156251/Desktop/thesis/labassist/hpc_scripts/scripts/extract_frames_for_imfe.py \
  --root "$ROOT" --splits "$SPLITS" --view "$VIEW" --fps "$FPS" --out "$FRAMES_DIR"
echo "[Step 2] Run IMFE extractor..."

python "$IMFE_REPO/imfe_feature_extraction.py" \
  --data_dir "$FRAMES_DIR" \
  --checkpoint_path "$IMFE_CKPT"
echo "[Step 3] Convert to cache format..."

python /zhome/45/8/156251/Desktop/thesis/labassist/hpc_scripts/scripts/convert_imfe_to_cache.py \
  --imfe_features "$FRAMES_DIR/features" \
  --out "$OUT_CACHE" \
  --fps "$FPS" \
  --frames_root "$FRAMES_DIR/rawframes_rgb"


