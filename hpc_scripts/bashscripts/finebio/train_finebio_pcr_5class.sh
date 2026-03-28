#!/bin/bash
#BSUB -J finebio_pcr_5class
#BSUB -o logs/finebio/%J_finebio_pcr_5class.out
#BSUB -e logs/finebio/%J_finebio_pcr_5class.err
#BSUB -q gpuv100
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -n 4
#BSUB -R "rusage[mem=16G]"
#BSUB -R "span[hosts=1]"
#BSUB -W 24:00
#BSUB -N

# Train MS-TCN on FineBio PCR videos only (5-class, no mydata)
# This tests domain generalization without any mydata in training

set -euo pipefail

echo "=========================================="
echo "Training MS-TCN on FineBio PCR 5-class"
echo "=========================================="
echo "Job ID: $LSB_JOBID"
echo "Hostname: $(hostname)"
echo "Start time: $(date)"
echo ""

# Directories
SCRIPT_DIR=/zhome/45/8/156251/Desktop/thesis/labassist/hpc_scripts
CONFIG=${SCRIPT_DIR}/configs/I3d_experiments/finebio_pcr_5class.yaml

# Activate environment
source /zhome/45/8/156251/Desktop/thesis/labassist/.env/bin/activate

# Add MS-TCN to PYTHONPATH (use :- to handle unset PYTHONPATH)
export PYTHONPATH="${PYTHONPATH:-}:${SCRIPT_DIR}/MS-TCN"

# Verify environment
echo "Python: $(which python)"
echo "CUDA_HOME: ${CUDA_HOME:-not set}"
echo "PYTHONPATH: ${PYTHONPATH}"
echo ""

# Navigate to scripts directory
cd ${SCRIPT_DIR}

# Create log directory
mkdir -p logs/finebio

# Train model
echo "Starting training..."
echo "Config: ${CONFIG}"
echo ""

python scripts/finetune_mstcn.py --config ${CONFIG}

echo ""
echo "=========================================="
echo "Training complete!"
echo "End time: $(date)"
echo "=========================================="


