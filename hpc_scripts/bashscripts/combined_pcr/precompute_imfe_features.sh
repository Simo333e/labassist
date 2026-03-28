#!/bin/sh
#BSUB -q gpuv100
#BSUB -J precompute_imfe_combined_pcr
#BSUB -gpu "num=1"
#BSUB -n 4
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=32GB]"
#BSUB -M 32GB
#BSUB -W 16:00
#BSUB -o /zhome/45/8/156251/Desktop/thesis/labassist/hpc_scripts/logs/combined_pcr/%J_imfe_features.out
#BSUB -e /zhome/45/8/156251/Desktop/thesis/labassist/hpc_scripts/logs/combined_pcr/%J_imfe_features.err

set -euo pipefail

echo "=========================================="
echo "IMFE Feature Extraction for Combined PCR"
echo "=========================================="
echo ""

# Activate environment
source /zhome/45/8/156251/Desktop/thesis/labassist/.env/bin/activate

# Paths
FINEBIO_ROOT=/work3/s204157/data/labassist/finebio_full
MYDATA_ROOT=/work3/s204157/data/labassist/my_data
SPLITS=/work3/s204157/data/labassist/combined_pcr/cache/splits.json
FPS=10
FRAMES_DIR=/work3/s204157/data/labassist/combined_pcr/cache/imfe_frames_${FPS}hz
IMFE_REPO=/work3/s204157/repos/RT-HARE
IMFE_CKPT='/zhome/45/8/156251/Desktop/thesis/labassist/2024-02-04-04-13-03-checkpoint-8.pt'
OUT_CACHE=/work3/s204157/data/labassist/combined_pcr/cache/features_${FPS}hz_imfe

# Create directories
mkdir -p "${FRAMES_DIR}" "${OUT_CACHE}"
mkdir -p /zhome/45/8/156251/Desktop/thesis/labassist/hpc_scripts/logs/combined_pcr

echo "[Step 1] Extracting frames at ${FPS} Hz with preprocessing..."
echo "  FineBio root: ${FINEBIO_ROOT}"
echo "  Mydata root: ${MYDATA_ROOT}"
echo "  Output: ${FRAMES_DIR}"
echo ""

python /zhome/45/8/156251/Desktop/thesis/labassist/hpc_scripts/scripts/feature_computations/extract_frames_for_imfe_combined_pcr.py \
  --finebio_root "${FINEBIO_ROOT}" \
  --mydata_root "${MYDATA_ROOT}" \
  --splits "${SPLITS}" \
  --fps "${FPS}" \
  --out "${FRAMES_DIR}"

echo ""
echo "[Step 2] Running IMFE feature extraction..."
echo "  RT-HARE repo: ${IMFE_REPO}"
echo "  Checkpoint: ${IMFE_CKPT}"
echo ""

cd "${IMFE_REPO}"
python imfe_feature_extraction.py \
  --data_dir "${FRAMES_DIR}" \
  --checkpoint_path "${IMFE_CKPT}"

echo ""
echo "[Step 3] Converting to cache format..."
echo "  Output: ${OUT_CACHE}"
echo ""

python /zhome/45/8/156251/Desktop/thesis/labassist/hpc_scripts/scripts/feature_computations/convert_imfe_to_cache.py \
  --imfe_features "${FRAMES_DIR}/features" \
  --out "${OUT_CACHE}" \
  --fps "${FPS}" \
  --frames_root "${FRAMES_DIR}/rawframes_rgb"

echo ""
echo "=========================================="
echo "IMFE Feature Extraction Complete!"
echo "=========================================="
echo ""
echo "Features saved to: ${OUT_CACHE}"
echo "Next step: Train Stage 2 with finebio_only_stage2_imfe.yaml"

