#!/bin/bash
#BSUB -J finebio_stage2
#BSUB -q gpuv100
#BSUB -W 12:00
#BSUB -n 8
#BSUB -R "rusage[mem=24GB]"
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -o /zhome/45/8/156251/Desktop/thesis/labassist/hpc_scripts/logs/combined_pcr/%J_finebio_stage2.out
#BSUB -e /zhome/45/8/156251/Desktop/thesis/labassist/hpc_scripts/logs/combined_pcr/%J_finebio_stage2.err

echo "=========================================="
echo "Stage 2: Training on FineBio-only PCR videos"
echo "Job ID: $LSB_JOBID"
echo "Host: $(hostname)"
echo "Date: $(date)"
echo "=========================================="

source /zhome/45/8/156251/Desktop/thesis/labassist/.env/bin/activate

CONFIG_PATH="/zhome/45/8/156251/Desktop/thesis/labassist/hpc_scripts/configs/combined_pcr/finebio_only_stage2.yaml"

echo "This creates an intermediate checkpoint:"
echo "  Atomic Actions (epoch-110) → FineBio PCR → [to be used for mydata K-fold]"
echo ""

python /zhome/45/8/156251/Desktop/thesis/labassist/hpc_scripts/scripts/finetune_mstcn.py \
  --config "${CONFIG_PATH}"

echo ""
echo "=========================================="
echo "Training complete!"
echo "Intermediate checkpoint saved to:"
echo "  /work3/s204157/experiments/finebio_only_stage2/"
echo "=========================================="

