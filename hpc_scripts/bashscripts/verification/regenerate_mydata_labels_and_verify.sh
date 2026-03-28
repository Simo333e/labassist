#!/bin/bash
# Regenerate mydata labels from Label Studio with correct 24fps handling
# Then extract verification clips to confirm alignment

set -euo pipefail

# Load ffmpeg module
module load ffmpeg/7.1

# Activate environment
if [ -d "/zhome/45/8/156251/Desktop/thesis/labassist/.env" ]; then
  . "/zhome/45/8/156251/Desktop/thesis/labassist/.env/bin/activate"
fi

SCRIPTS=/zhome/45/8/156251/Desktop/thesis/labassist/hpc_scripts/scripts
MYDATA=/work3/s204157/data/labassist/my_data

# Label Studio export (IMPORTANT: this is the source of truth)
LABELSTUDIO_JSON=${MYDATA}/project-3-at-2025-11-26-19-24-89666430.json

# Use ORIGINAL videos (not _clean trimmed versions)
VIDEOS_DIR=${MYDATA}/videos

# Output to verified cache directory
OUTPUT_DIR=${MYDATA}/cache_verified

# Verification clips output
VERIFY_OUTPUT=/zhome/45/8/156251/Desktop/thesis/labassist/hpc_scripts/eval_data/action_verification_v2

echo "=============================================="
echo "=== MYDATA LABEL REGENERATION & VERIFICATION ==="
echo "=============================================="
echo ""
echo "Source: ${LABELSTUDIO_JSON}"
echo "Videos: ${VIDEOS_DIR}"
echo "Output: ${OUTPUT_DIR}"
echo ""

# Check if Label Studio JSON exists
if [ ! -f "${LABELSTUDIO_JSON}" ]; then
    echo "ERROR: Label Studio JSON not found: ${LABELSTUDIO_JSON}"
    exit 1
fi

# ============================================
# STEP 1: Regenerate labels with 24fps fix
# ============================================
echo "--- Step 1: Converting Label Studio annotations (24fps → 10Hz) ---"
echo ""
echo "IMPORTANT: Label Studio uses 24fps for its timeline, NOT the video's native fps."
echo "The conversion script handles this with LABEL_STUDIO_FPS = 24.0"
echo ""

python ${SCRIPTS}/data_prep/convert_labelstudio_to_mstcn.py \
    --labelstudio_json "${LABELSTUDIO_JSON}" \
    --videos_dir "${VIDEOS_DIR}" \
    --out_dir "${OUTPUT_DIR}" \
    --target_fps 10.0 \
    --val_ratio 0.0 \
    --seed 42 \
    --default_label "none"

echo ""
echo "✓ Labels regenerated to: ${OUTPUT_DIR}/labels/"
echo ""

# ============================================
# STEP 2: Extract original clips for verification
# ============================================
echo "--- Step 2: Extracting ORIGINAL clips (no transformation) ---"
echo ""

# Use ALL mydata videos for verification
python ${SCRIPTS}/data_prep/extract_action_snippets.py \
    --video_dir ${VIDEOS_DIR} \
    --label_dir ${OUTPUT_DIR}/labels \
    --class_index ${OUTPUT_DIR}/class_index.json \
    --output_dir ${VERIFY_OUTPUT}/mydata_original \
    --actions "Attach pipette tip,Eject pipette tip" \
    --padding 1.0 \
    --max_clips 5

echo ""
echo "✓ Original clips saved to: ${VERIFY_OUTPUT}/mydata_original/"
echo ""

# ============================================
# STEP 3: Create transformed clips (CROP → VFLIP)
# ============================================
echo "--- Step 3: Creating TRANSFORMED clips (crop → vflip) ---"
echo ""

# Transformation parameters (match training pipeline)
# IMPORTANT: Apply CROP first (remove glass shelf), THEN VFLIP (match FineBio viewpoint)
CROP_FILTER="crop=in_w*0.82:in_h*0.65:in_w*0.10:0"
VFLIP_FILTER="vflip"
COMBINED_FILTER="${CROP_FILTER},${VFLIP_FILTER}"

mkdir -p ${VERIFY_OUTPUT}/mydata_transformed
for action_name in "Attach_pipette_tip" "Eject_pipette_tip"; do
    mkdir -p ${VERIFY_OUTPUT}/mydata_transformed/${action_name}
    
    if [ -d "${VERIFY_OUTPUT}/mydata_original/${action_name}" ]; then
        for clip in ${VERIFY_OUTPUT}/mydata_original/${action_name}/*.mp4; do
            if [ -f "$clip" ]; then
                basename_clip=$(basename "$clip")
                output_clip="${VERIFY_OUTPUT}/mydata_transformed/${action_name}/${basename_clip}"
                
                echo "  Transforming: ${basename_clip}"
                ffmpeg -y -i "$clip" -vf "${COMBINED_FILTER}" -c:v libx264 -preset fast -crf 23 -an "$output_clip" 2>/dev/null
            fi
        done
    fi
done

echo ""
echo "✓ Transformed clips saved to: ${VERIFY_OUTPUT}/mydata_transformed/"
echo ""

# ============================================
# SUMMARY
# ============================================
echo "=============================================="
echo "=== VERIFICATION COMPLETE ==="
echo "=============================================="
echo ""
echo "Generated files:"
echo "  Labels:     ${OUTPUT_DIR}/labels/"
echo "  Class idx:  ${OUTPUT_DIR}/class_index.json"
echo "  Splits:     ${OUTPUT_DIR}/splits.json"
echo ""
echo "Verification clips:"
echo "  Original:    ${VERIFY_OUTPUT}/mydata_original/"
echo "  Transformed: ${VERIFY_OUTPUT}/mydata_transformed/"
echo ""
echo "Next steps:"
echo "  1. Open clips and verify timing matches Label Studio annotations"
echo "  2. Compare transformed clips with FineBio clips"
echo "  3. If good, copy labels to cache_fixed for training:"
echo "     cp -r ${OUTPUT_DIR}/labels/* ${MYDATA}/cache_fixed/labels/"
echo ""

