#!/bin/sh
#BSUB -q gpuv100
#BSUB -J fuse_i3d_yolo
#BSUB -n 4
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=8GB]"
#BSUB -M 8GB
#BSUB -W 1:00
#BSUB -o /zhome/45/8/156251/Desktop/thesis/labassist/hpc_scripts/logs/combined_pcr/%J_fuse_features.out
#BSUB -e /zhome/45/8/156251/Desktop/thesis/labassist/hpc_scripts/logs/combined_pcr/%J_fuse_features.err

# ============================================================================
# Feature Fusion: I3D + YOLO
# ============================================================================
# Concatenates I3D (1024D) and YOLO (28D) features into 1052D features
# for multi-modal action recognition.
# ============================================================================

set -euo pipefail

DATA_ROOT="/work3/s204157/data/labassist/combined_pcr"

I3D_DIR="${DATA_ROOT}/cache/features_10hz_i3d_selective_crop_flip_win16"
YOLO_DIR="${DATA_ROOT}/cache/features_10hz_yolo"
OUTPUT_DIR="${DATA_ROOT}/cache/features_10hz_i3d_yolo_fused"

LOGS_DIR="/zhome/45/8/156251/Desktop/thesis/labassist/hpc_scripts/logs/combined_pcr"

mkdir -p "${OUTPUT_DIR}"
mkdir -p "${LOGS_DIR}"

if [ -d "/zhome/45/8/156251/Desktop/thesis/labassist/.env" ]; then
  . "/zhome/45/8/156251/Desktop/thesis/labassist/.env/bin/activate"
fi

echo "============================================================"
echo "Feature Fusion: I3D + YOLO"
echo "============================================================"
echo "I3D features: ${I3D_DIR}"
echo "YOLO features: ${YOLO_DIR}"
echo "Output: ${OUTPUT_DIR}"
echo ""
echo "Resulting feature dimension: 1024 + 28 = 1052"
echo ""

python /zhome/45/8/156251/Desktop/thesis/labassist/hpc_scripts/scripts/feature_computations/concatenate_i3d_yolo_features.py \
  --i3d_dir "${I3D_DIR}" \
  --yolo_dir "${YOLO_DIR}" \
  --output_dir "${OUTPUT_DIR}" \
  --normalize_yolo

echo ""
echo "============================================================"
echo "✓ Feature fusion complete"
echo "  Output: ${OUTPUT_DIR}"
echo "============================================================"

