#!/bin/bash
# Extract verification clips DIRECTLY from Label Studio JSON
# This bypasses the 10Hz label conversion and ensures correct timing

set -euo pipefail

# Load ffmpeg module
module load ffmpeg/7.1

# Activate environment
if [ -d "/zhome/45/8/156251/Desktop/thesis/labassist/.env" ]; then
  . "/zhome/45/8/156251/Desktop/thesis/labassist/.env/bin/activate"
fi

SCRIPTS=/zhome/45/8/156251/Desktop/thesis/labassist/hpc_scripts/scripts
MYDATA=/work3/s204157/data/labassist/my_data

# Label Studio export (source of truth for annotations)
LABELSTUDIO_JSON=${MYDATA}/project-3-at-2025-11-26-19-24-89666430.json

# Original videos
VIDEOS_DIR=${MYDATA}/videos

# Output directory
OUTPUT=/zhome/45/8/156251/Desktop/thesis/labassist/hpc_scripts/eval_data/action_verification_direct

echo "=============================================="
echo "=== DIRECT CLIP EXTRACTION FROM LABEL STUDIO ==="
echo "=============================================="
echo ""
echo "This script extracts clips DIRECTLY from Label Studio JSON."
echo "It uses LABEL_STUDIO_FPS = 24.0 (the fps LS uses for its timeline)"
echo "NOT the video's native fps (30fps)."
echo ""
echo "Source: ${LABELSTUDIO_JSON}"
echo "Videos: ${VIDEOS_DIR}"
echo "Output: ${OUTPUT}"
echo ""

# Check if Label Studio JSON exists
if [ ! -f "${LABELSTUDIO_JSON}" ]; then
    echo "ERROR: Label Studio JSON not found: ${LABELSTUDIO_JSON}"
    exit 1
fi

# Actions to extract
ACTIONS="Attach pipette tip,Eject pipette tip"

# ============================================
# STEP 1: Extract ORIGINAL clips (no transformation)
# ============================================
echo "--- Step 1: Extracting ORIGINAL clips (direct from LS JSON) ---"
echo ""

python ${SCRIPTS}/data_prep/extract_clips_from_labelstudio.py \
    --labelstudio_json "${LABELSTUDIO_JSON}" \
    --videos_dir "${VIDEOS_DIR}" \
    --output_dir "${OUTPUT}/mydata_original" \
    --actions "${ACTIONS}" \
    --max_clips 5 \
    --padding 1.0

echo ""

# ============================================
# STEP 2: Extract TRANSFORMED clips (crop → vflip)
# ============================================
echo "--- Step 2: Extracting TRANSFORMED clips (crop → vflip) ---"
echo ""

# Crop parameters (match training pipeline):
# - Remove bottom 35% (glass shelf)
# - Remove left 10%, right 8%
# FFmpeg crop: crop=out_w:out_h:x:y
# crop=in_w*0.82:in_h*0.65:in_w*0.10:0

CROP_FILTER="crop=in_w*0.82:in_h*0.65:in_w*0.10:0"

python ${SCRIPTS}/data_prep/extract_clips_from_labelstudio.py \
    --labelstudio_json "${LABELSTUDIO_JSON}" \
    --videos_dir "${VIDEOS_DIR}" \
    --output_dir "${OUTPUT}/mydata_transformed" \
    --actions "${ACTIONS}" \
    --max_clips 5 \
    --padding 1.0 \
    --vflip \
    --crop "${CROP_FILTER}"

echo ""

# ============================================
# SUMMARY
# ============================================
echo "=============================================="
echo "=== EXTRACTION COMPLETE ==="
echo "=============================================="
echo ""
echo "Clips extracted to:"
echo "  Original:    ${OUTPUT}/mydata_original/"
echo "  Transformed: ${OUTPUT}/mydata_transformed/"
echo ""
echo "KEY: These clips are extracted using LABEL_STUDIO_FPS = 24.0"
echo "     Frame N in Label Studio → time = (N-1)/24 seconds"
echo ""
echo "Next steps:"
echo "  1. Open clips and verify timing matches Label Studio annotations"
echo "  2. Compare with FineBio clips for visual similarity"
echo ""

