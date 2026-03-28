#!/bin/bash
#BSUB -J combined_pcr_reg
#BSUB -q gpuv100
#BSUB -W 12:00
#BSUB -n 4
#BSUB -R "rusage[mem=8GB]"
#BSUB -R "span[hosts=1]"
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -o /zhome/45/8/156251/Desktop/thesis/labassist/hpc_scripts/logs/combined_pcr/%J_combined_reg.out
#BSUB -e /zhome/45/8/156251/Desktop/thesis/labassist/hpc_scripts/logs/combined_pcr/%J_combined_reg.err

###############################################################################
# Train combined_pcr_i3d_combined_reg
# Uses epoch-110 FineBio checkpoint with strong regularization
###############################################################################

echo "========================================"
echo "Training combined_pcr_i3d_combined_reg"
echo "Job ID: $LSB_JOBID"
echo "Host: $(hostname)"
echo "Date: $(date)"
echo "========================================"

source /zhome/45/8/156251/Desktop/thesis/labassist/.env/bin/activate
cd /zhome/45/8/156251/Desktop/thesis/labassist

CONFIG="/zhome/45/8/156251/Desktop/thesis/labassist/hpc_scripts/configs/combined_pcr/combined_pcr_i3d_combined_reg.yaml"

python hpc_scripts/scripts/finetune_mstcn.py --config "${CONFIG}"

echo "========================================"
echo "Training complete!"
echo "Date: $(date)"
echo "========================================"

