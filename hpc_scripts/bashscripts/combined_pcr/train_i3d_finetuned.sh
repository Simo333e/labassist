#!/bin/sh
#BSUB -q gpuv100
#BSUB -J combined_i3d
#BSUB -gpu "num=1"
#BSUB -n 4
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=16GB]"
#BSUB -M 16GB
#BSUB -W 24:00
#BSUB -o /zhome/45/8/156251/Desktop/thesis/labassist/hpc_scripts/logs/combined_pcr/%J_i3d_finetuned.out
#BSUB -e /zhome/45/8/156251/Desktop/thesis/labassist/hpc_scripts/logs/combined_pcr/%J_i3d_finetuned.err

set -euo pipefail

CONFIG_PATH="/zhome/45/8/156251/Desktop/thesis/labassist/hpc_scripts/configs/combined_pcr/combined_pcr_i3d_finetuned.yaml"
SCRIPTS_DIR="/zhome/45/8/156251/Desktop/thesis/labassist/hpc_scripts/scripts"
LOGS_DIR="/zhome/45/8/156251/Desktop/thesis/labassist/hpc_scripts/logs/combined_pcr"

mkdir -p "${LOGS_DIR}"

if [ -d "/zhome/45/8/156251/Desktop/thesis/labassist/.env" ]; then
  . "/zhome/45/8/156251/Desktop/thesis/labassist/.env/bin/activate"
fi

echo "============================================================"
echo "Training I3D finetuned on Combined PCR (48 videos)"
echo "============================================================"
echo "Config: ${CONFIG_PATH}"
echo "Pretrained checkpoint: /work3/s204157/experiments/finebio_atomic_pretrain/epoch-50.model"
echo "Training videos: 47 (5 mydata + 42 FineBio)"
echo "Validation videos: 1 (20251125_070904Z.mp4)"
echo ""

python ${SCRIPTS_DIR}/finetune_mstcn.py \
    --config "${CONFIG_PATH}"

echo ""
echo "============================================================"
echo "✓ I3D training complete"
echo "============================================================"

