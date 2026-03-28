#!/bin/sh
#BSUB -q hpc
#BSUB -J imfe_features_atomic
#BSUB -n 4
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=32GB]"
#BSUB -M 32GB
#BSUB -W 24:00
#BSUB -o /zhome/45/8/156251/Desktop/thesis/labassist/hpc_scripts/logs/finebio/%J_imfe_features_atomic.out
#BSUB -e /zhome/45/8/156251/Desktop/thesis/labassist/hpc_scripts/logs/finebio/%J_imfe_features_atomic.err

set -euo pipefail

# Activate environment
source /zhome/45/8/156251/Desktop/thesis/labassist/.env/bin/activate

# Change to project root
cd /zhome/45/8/156251/Desktop/thesis/labassist

# Create log directory
mkdir -p /zhome/45/8/156251/Desktop/thesis/labassist/hpc_scripts/logs/finebio

# Paths
FINEBIO_ROOT="/work3/s204157/data/labassist/finebio_full/videos"
SPLITS="/work3/s204157/data/labassist/finebio_full/cache/splits_finebio_atomic.json"
FRAMES_DIR="/work3/s204157/data/labassist/finebio_full/cache/imfe_frames_10hz"
IMFE_REPO="/work3/s204157/repos/RT-HARE"
IMFE_CKPT="/zhome/45/8/156251/Desktop/thesis/labassist/2024-02-04-04-13-03-checkpoint-8.pt"
OUT_CACHE="/work3/s204157/data/labassist/finebio_full/cache/features_10hz_imfe"

echo "=========================================="
echo "IMFE Feature Extraction for FineBio Atomic"
echo "Dataset: Full FineBio atomic (223 videos)"
echo "Sampling: 10 Hz"
echo "=========================================="
echo ""

# Step 1: Extract frames at 10Hz (no preprocessing for FineBio)
echo "Step 1/3: Extracting frames at 10 Hz..."
python hpc_scripts/scripts/feature_computations/extract_frames_for_imfe_finebio_atomic.py \
  --finebio_root "${FINEBIO_ROOT}" \
  --splits "${SPLITS}" \
  --out_dir "${FRAMES_DIR}" \
  --target_fps 10.0

echo ""
echo "Step 1 complete! Frames saved to: ${FRAMES_DIR}"
echo ""

# Step 2: Compute IMFE features using RT-HARE
echo "Step 2/3: Computing IMFE features..."
cd "${IMFE_REPO}"
python imfe_feature_extraction.py \
  --data_dir "${FRAMES_DIR}" \
  --checkpoint_path "${IMFE_CKPT}"

echo ""
echo "Step 2 complete! Raw IMFE features computed"
echo ""

# Step 3: Convert to cache format
echo "Step 3/3: Converting to cache format..."
cd /zhome/45/8/156251/Desktop/thesis/labassist
python hpc_scripts/scripts/feature_computations/convert_imfe_to_cache.py \
  --imfe_dir "${FRAMES_DIR}" \
  --out_dir "${OUT_CACHE}"

echo ""
echo "=========================================="
echo "IMFE feature extraction complete!"
echo "Output: ${OUT_CACHE}"
echo "Next: Train atomic pretrain model on these features"
echo "=========================================="


