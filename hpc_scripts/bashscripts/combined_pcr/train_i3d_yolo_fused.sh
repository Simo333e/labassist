#!/bin/sh
#BSUB -q gpuv100
#BSUB -J combined_i3d_yolo
#BSUB -gpu "num=1"
#BSUB -n 4
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=16GB]"
#BSUB -M 16GB
#BSUB -W 24:00
#BSUB -o /zhome/45/8/156251/Desktop/thesis/labassist/hpc_scripts/logs/combined_pcr/%J_i3d_yolo_fused.out
#BSUB -e /zhome/45/8/156251/Desktop/thesis/labassist/hpc_scripts/logs/combined_pcr/%J_i3d_yolo_fused.err

# ============================================================================
# Training: I3D + YOLO Multi-Modal Action Recognition
# ============================================================================
# This experiment tests whether adding YOLO object detection features
# helps disambiguate "Pipette pcr tube" vs "Aspirate reagent".
#
# Hypothesis: I3D captures motion patterns, but not target object identity.
# YOLO features encode:
#   - Object presence (pcr tube, reagent, etc.)
#   - Object positions
#   - Hand-object proximity
#
# Expected outcome: Improved precision on "Aspirate reagent" and
# reduced confusion with "Pipette pcr tube".
# ============================================================================

set -euo pipefail

CONFIG_PATH="/zhome/45/8/156251/Desktop/thesis/labassist/hpc_scripts/configs/combined_pcr/combined_pcr_i3d_yolo_fused.yaml"
SCRIPTS_DIR="/zhome/45/8/156251/Desktop/thesis/labassist/hpc_scripts/scripts"
LOGS_DIR="/zhome/45/8/156251/Desktop/thesis/labassist/hpc_scripts/logs/combined_pcr"

mkdir -p "${LOGS_DIR}"

if [ -d "/zhome/45/8/156251/Desktop/thesis/labassist/.env" ]; then
  . "/zhome/45/8/156251/Desktop/thesis/labassist/.env/bin/activate"
fi

echo "============================================================"
echo "Training: I3D + YOLO Multi-Modal Action Recognition"
echo "============================================================"
echo ""
echo "Config: ${CONFIG_PATH}"
echo ""
echo "Feature fusion:"
echo "  - I3D: 1024D (motion/appearance)"
echo "  - YOLO: 28D (object detection)"
echo "  - Combined: 1052D"
echo ""
echo "YOLO features (28D breakdown):"
echo "  - Object confidence: 14D (max conf per class)"
echo "  - Key positions: 10D (pcr tube, reagent, hand, etc.)"
echo "  - Hand proximity: 4D (distance to containers)"
echo ""
echo "Hypothesis: YOLO features will help disambiguate"
echo "'Pipette pcr tube' vs 'Aspirate reagent'"
echo ""
echo "Training videos: 47 (5 mydata + 42 FineBio)"
echo "Validation videos: 1 (20251125_070904Z.mp4)"
echo ""

python ${SCRIPTS_DIR}/finetune_mstcn.py \
    --config "${CONFIG_PATH}"

echo ""
echo "============================================================"
echo "✓ I3D+YOLO training complete"
echo "============================================================"
echo ""
echo "Next steps:"
echo "1. Evaluate checkpoints with sweep_checkpoints_segmental.py"
echo "2. Generate confusion matrix and timelines"
echo "3. Compare with I3D-only baseline (combined_pcr_i3d_combined_reg)"
echo "============================================================"

