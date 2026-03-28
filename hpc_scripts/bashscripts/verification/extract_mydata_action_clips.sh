#!/bin/bash
# Extract mydata action clips for visual verification
# Extracts BOTH original and transformed (vflip+crop) versions to verify alignment

set -euo pipefail

# Load ffmpeg module
module load ffmpeg/7.1

# Activate environment
if [ -d "/zhome/45/8/156251/Desktop/thesis/labassist/.env" ]; then
  . "/zhome/45/8/156251/Desktop/thesis/labassist/.env/bin/activate"
fi

SCRIPTS=/zhome/45/8/156251/Desktop/thesis/labassist/hpc_scripts/scripts
MYDATA=/work3/s204157/data/labassist/my_data

# Use the fixed labels
VIDEOS=${MYDATA}/videos
LABELS=${MYDATA}/cache_fixed/labels
CLASS_INDEX=${MYDATA}/cache_fixed/class_index.json

# Output directory
OUTPUT=/zhome/45/8/156251/Desktop/thesis/labassist/hpc_scripts/eval_data/action_verification

echo "=== Extracting mydata action clips for visual verification ==="
echo ""
echo "Strategy:"
echo "  1. Extract ORIGINAL clips (no transformations)"
echo "  2. Extract TRANSFORMED clips (vflip + crop = what model sees)"
echo "  3. Compare side-by-side to verify transformations are correct"
echo ""

# Extract the 5 usable mydata videos (excluding broken pipette videos)
USABLE_STEMS="20251027_181439Z,20251120_065442Z,20251125_070904Z,20251112_110656Z_1080p30_v2,20251125_072321Z"

# Actions to verify (the ones we'll transfer learn from FineBio)
ACTIONS="Attach pipette tip,Eject pipette tip"

echo "--- Step 1: Extract ORIGINAL clips (before transformation) ---"
python ${SCRIPTS}/data_prep/extract_action_snippets.py \
    --video_dir ${VIDEOS} \
    --label_dir ${LABELS} \
    --class_index ${CLASS_INDEX} \
    --output_dir ${OUTPUT}/mydata_original \
    --actions "${ACTIONS}" \
    --stems "${USABLE_STEMS}" \
    --padding 1.0 \
    --max_clips 5

echo ""
echo "--- Step 2: Create TRANSFORMED clips (vflip + crop) using ffmpeg ---"

# Transformation parameters (match the feature extraction pipeline)
# crop=in_w*(1-crop_left-crop_right):in_h*(1-crop_top-crop_bottom):in_w*crop_left:in_h*crop_top
# For: top=0.0, bottom=0.35, left=0.10, right=0.08
# → width: keep 82% (from 10% to 92%)
# → height: keep 65% (from 0% to 65%)

CROP_FILTER="crop=in_w*0.82:in_h*0.65:in_w*0.10:0"
VFLIP_FILTER="vflip"
# IMPORTANT: CROP first (remove glass shelf), THEN vflip (match FineBio viewpoint)
COMBINED_FILTER="${CROP_FILTER},${VFLIP_FILTER}"

# Re-extract with transformations applied
mkdir -p ${OUTPUT}/mydata_flipped
for action_name in "Attach_pipette_tip" "Eject_pipette_tip"; do
    mkdir -p ${OUTPUT}/mydata_flipped/${action_name}
    
    # Copy clips from original and apply transformations
    if [ -d "${OUTPUT}/mydata_original/${action_name}" ]; then
        for clip in ${OUTPUT}/mydata_original/${action_name}/*.mp4; do
            if [ -f "$clip" ]; then
                basename_clip=$(basename "$clip")
                output_clip="${OUTPUT}/mydata_flipped/${action_name}/${basename_clip}"
                
                echo "  Transforming: ${basename_clip}"
                ffmpeg -y -i "$clip" -vf "${COMBINED_FILTER}" -c:v libx264 -preset fast -crf 23 -an "$output_clip" 2>/dev/null
            fi
        done
    fi
done

echo ""
echo "=== Done! ==="
echo ""
echo "Clips saved to:"
echo "  ${OUTPUT}/mydata_original/    ← BEFORE transformation (raw videos)"
echo "  ${OUTPUT}/mydata_flipped/     ← AFTER vflip+crop (what model sees)"
echo ""
echo "Next steps:"
echo "  1. Run: bash bashscripts/verification/extract_finebio_action_clips.sh"
echo "  2. Visually compare:"
echo "     - mydata_original vs mydata_flipped (verify transformation is correct)"
echo "     - mydata_flipped vs finebio (verify actions look similar)"
echo "  3. If they look good → proceed with K-fold experiments"
echo ""


