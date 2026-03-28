#!/bin/sh
#BSUB -q gpuv100
#BSUB -J finetune_i3d_selective_crop_5class
#BSUB -gpu "num=1"
#BSUB -n 8
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=24GB]"
#BSUB -M 24GB
#BSUB -W 24:00
#BSUB -o /zhome/45/8/156251/Desktop/thesis/labassist/hpc_scripts/logs/combined_pcr/%J_finetune_i3d_selective_crop_5class.out
#BSUB -e /zhome/45/8/156251/Desktop/thesis/labassist/hpc_scripts/logs/combined_pcr/%J_finetune_i3d_selective_crop_5class.err

set -euo pipefail

CONFIG_PATH="/zhome/45/8/156251/Desktop/thesis/labassist/hpc_scripts/configs/I3d_experiments/combined_pcr_i3d_selective_crop_5class_v1.yaml"

mkdir -p /zhome/45/8/156251/Desktop/thesis/labassist/hpc_scripts/logs/combined_pcr

if [ -d "/zhome/45/8/156251/Desktop/thesis/labassist/.env" ]; then
  . "/zhome/45/8/156251/Desktop/thesis/labassist/.env/bin/activate"
fi

echo "=========================================="
echo "Fine-tuning MS-TCN with I3D features (5-class)"
echo "Features: Selective crop (mydata cropped, FineBio unchanged)"
echo "Config: ${CONFIG_PATH}"
echo "=========================================="

python /zhome/45/8/156251/Desktop/thesis/labassist/hpc_scripts/scripts/finetune_mstcn.py \
  --config "${CONFIG_PATH}"

