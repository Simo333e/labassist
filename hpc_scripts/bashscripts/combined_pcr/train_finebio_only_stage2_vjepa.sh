#!/bin/bash
#BSUB -J finebio_vjepa_stage2
#BSUB -q gpuv100
#BSUB -W 24:00
#BSUB -n 8
#BSUB -R "rusage[mem=24GB]"
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -o /zhome/45/8/156251/Desktop/thesis/labassist/hpc_scripts/logs/combined_pcr/%J_finebio_vjepa_stage2.out
#BSUB -e /zhome/45/8/156251/Desktop/thesis/labassist/hpc_scripts/logs/combined_pcr/%J_finebio_vjepa_stage2.err

echo "=========================================="
echo "Stage 2 (V-JEPA): Training from scratch on FineBio-only PCR videos"
echo "Job ID: $LSB_JOBID"
echo "Host: $(hostname)"
echo "Date: $(date)"
echo "=========================================="

source /zhome/45/8/156251/Desktop/thesis/labassist/.env/bin/activate

CONFIG_PATH="/zhome/45/8/156251/Desktop/thesis/labassist/hpc_scripts/configs/combined_pcr/finebio_only_stage2_vjepa.yaml"

echo "This creates an intermediate V-JEPA checkpoint:"
echo "  Train from scratch on FineBio PCR → [to be used for mydata K-fold]"
echo "  Features: V-JEPA (stride=1, selective crop, flip)"
echo "  Training: 150 epochs from scratch (no pretraining)"
echo ""

python /zhome/45/8/156251/Desktop/thesis/labassist/hpc_scripts/scripts/finetune_mstcn.py \
  --config "${CONFIG_PATH}"

echo ""
echo "=========================================="
echo "Training complete!"
echo "Intermediate V-JEPA checkpoint saved to:"
echo "  /work3/s204157/experiments/finebio_only_stage2_vjepa/"
echo "=========================================="

