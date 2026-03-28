#!/bin/bash
#BSUB -J finetune_hflip
#BSUB -q gpua100
#BSUB -n 4
#BSUB -R "rusage[mem=16G]"
#BSUB -R "span[hosts=1]"
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -W 8:00
#BSUB -o /zhome/45/8/156251/Desktop/thesis/labassist/hpc_scripts/logs/combined_pcr/%J_finetune_hflip.out
#BSUB -e /zhome/45/8/156251/Desktop/thesis/labassist/hpc_scripts/logs/combined_pcr/%J_finetune_hflip.err

# Finetune MS-TCN on combined_pcr with horizontal flip features
# - Uses pretrained FineBio atomic model
# - Features have both vertical and horizontal flip for mydata
# - Excludes broken pipette video from training

set -euo pipefail

# Activate environment
if [ -d "/zhome/45/8/156251/Desktop/thesis/labassist/.env" ]; then
  . "/zhome/45/8/156251/Desktop/thesis/labassist/.env/bin/activate"
fi

SCRIPTS=/zhome/45/8/156251/Desktop/thesis/labassist/hpc_scripts/scripts
CONFIG=/zhome/45/8/156251/Desktop/thesis/labassist/hpc_scripts/configs/I3d_experiments/combined_pcr_hflip_exclude_broken.yaml

echo "=== Finetuning MS-TCN with horizontal flip + broken video excluded ==="
echo "Config: ${CONFIG}"

# First check if features exist
FEATURE_DIR=/work3/s204157/data/labassist/combined_pcr/cache/features_10hz_i3d_selective_crop_vhflip_win16
if [ ! -d "${FEATURE_DIR}" ]; then
    echo "ERROR: Feature directory does not exist: ${FEATURE_DIR}"
    echo "Please run precompute_i3d_hflip.sh first!"
    exit 1
fi

# Count features
NUM_FEATURES=$(ls -1 ${FEATURE_DIR}/*.npy 2>/dev/null | wc -l)
echo "Found ${NUM_FEATURES} feature files"

python ${SCRIPTS}/finetune_mstcn.py ${CONFIG}

echo "=== Training complete ==="
