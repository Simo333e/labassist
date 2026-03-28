#!/bin/sh
#BSUB -q gpuv100
#BSUB -J combined_vjepa
#BSUB -gpu "num=1"
#BSUB -n 4
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=16GB]"
#BSUB -M 16GB
#BSUB -W 24:00
#BSUB -o /zhome/45/8/156251/Desktop/thesis/labassist/hpc_scripts/logs/combined_pcr/%J_vjepa_from_scratch.out
#BSUB -e /zhome/45/8/156251/Desktop/thesis/labassist/hpc_scripts/logs/combined_pcr/%J_vjepa_from_scratch.err

set -euo pipefail

CONFIG_PATH="/zhome/45/8/156251/Desktop/thesis/labassist/hpc_scripts/configs/combined_pcr/combined_pcr_vjepa_from_scratch.yaml"
SCRIPTS_DIR="/zhome/45/8/156251/Desktop/thesis/labassist/hpc_scripts/scripts"
LOGS_DIR="/zhome/45/8/156251/Desktop/thesis/labassist/hpc_scripts/logs/combined_pcr"

mkdir -p "${LOGS_DIR}"

if [ -d "/zhome/45/8/156251/Desktop/thesis/labassist/.env" ]; then
  . "/zhome/45/8/156251/Desktop/thesis/labassist/.env/bin/activate"
fi

echo "============================================================"
echo "Training V-JEPA from scratch on Combined PCR (48 videos)"
echo "============================================================"
echo "Config: ${CONFIG_PATH}"
echo "Training videos: 47 (5 mydata + 42 FineBio)"
echo "Validation videos: 1 (20251125_070904Z.mp4)"
echo ""

python ${SCRIPTS_DIR}/finetune_mstcn.py \
    --config "${CONFIG_PATH}"

echo ""
echo "============================================================"
echo "✓ V-JEPA training complete"
echo "============================================================"

