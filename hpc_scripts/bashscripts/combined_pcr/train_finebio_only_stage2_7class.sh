#!/bin/bash
#BSUB -J finebio_stage2_7class_i3d
#BSUB -q gpuv100
#BSUB -W 12:00
#BSUB -n 8
#BSUB -R "rusage[mem=24GB]"
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -o /zhome/45/8/156251/Desktop/thesis/labassist/hpc_scripts/logs/combined_pcr/%J_finebio_stage2_7class_i3d.out
#BSUB -e /zhome/45/8/156251/Desktop/thesis/labassist/hpc_scripts/logs/combined_pcr/%J_finebio_stage2_7class_i3d.err

# Stage 2: Train on FineBio-only PCR videos with 7-class taxonomy (I3D features)
# This creates an intermediate checkpoint for mydata K-fold experiments (exp18)

echo "=========================================="
echo "Stage 2: FineBio-only PCR Training (7-class, I3D)"
echo "Job ID: $LSB_JOBID"
echo "Host: $(hostname)"
echo "Date: $(date)"
echo "=========================================="

source /zhome/45/8/156251/Desktop/thesis/labassist/.env/bin/activate

CONFIG="/zhome/45/8/156251/Desktop/thesis/labassist/hpc_scripts/configs/combined_pcr/finebio_only_stage2_7class.yaml"

echo "Training with 7-class taxonomy:"
echo "  Classes:"
echo "    0: none (background)"
echo "    1: Adjust pipette volume (0% in FineBio)"
echo "    2: Attach pipette tip (7.05%)"
echo "    3: Eject pipette tip (2.61%)"
echo "    4: Manipulate pcr tubes (18.71%)"
echo "    5: Manipulate reagents (5.20%)"
echo "    6: Switch pipette (0.07%)"
echo ""
echo "Starting from: Atomic pretrain (36 classes)"
echo "Training on: 42 FineBio PCR videos only"
echo ""

# Run training
python /zhome/45/8/156251/Desktop/thesis/labassist/hpc_scripts/scripts/finetune_mstcn.py \
  --config "${CONFIG}"

echo ""
echo "=========================================="
echo "Stage 2 training complete!"
echo "Next: Evaluate checkpoints to select best for Stage 3"
echo "=========================================="




