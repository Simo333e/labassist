#!/bin/sh
#BSUB -q gpuv100
#BSUB -J finetune_combined_pcr_vjepa_5class
#BSUB -gpu "num=1"
#BSUB -n 8
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=24GB]"
#BSUB -M 24GB
#BSUB -W 24:00
#BSUB -o /zhome/45/8/156251/Desktop/thesis/labassist/hpc_scripts/logs/mydata/%J_finetune_vjepa_5class.out
#BSUB -e /zhome/45/8/156251/Desktop/thesis/labassist/hpc_scripts/logs/mydata/%J_finetune_vjepa_5class.err

set -euo pipefail

CONFIG_PATH="/zhome/45/8/156251/Desktop/thesis/labassist/hpc_scripts/configs/vjepa_experiments/combined_pcr_vjepa_5class_v2.yaml"

if [ -d "/zhome/45/8/156251/Desktop/thesis/labassist/.env" ]; then
  . "/zhome/45/8/156251/Desktop/thesis/labassist/.env/bin/activate"
fi

echo "=========================================="
echo "Fine-tuning MS-TCN with V-JEPA features (5-class)"
echo "Config: ${CONFIG_PATH}"
echo "=========================================="

python /zhome/45/8/156251/Desktop/thesis/labassist/hpc_scripts/scripts/finetune_mstcn.py \
  --config "${CONFIG_PATH}"

