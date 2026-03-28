#!/bin/bash
# Extract FineBio PCR action clips for visual comparison with mydata

set -euo pipefail

# Load ffmpeg module
module load ffmpeg/7.1

# Activate environment
if [ -d "/zhome/45/8/156251/Desktop/thesis/labassist/.env" ]; then
  . "/zhome/45/8/156251/Desktop/thesis/labassist/.env/bin/activate"
fi

SCRIPTS=/zhome/45/8/156251/Desktop/thesis/labassist/hpc_scripts/scripts
FINEBIO=/work3/s204157/data/labassist/finebio_full

# Use atomic-level labels (has attach_tip, eject_tip)
VIDEOS=/work3/s204157/data/labassist/finebio_videos_w640
LABELS=${FINEBIO}/cache/labels_atomic
CLASS_INDEX=${FINEBIO}/cache/class_index_atomic.json

# Output directory
OUTPUT=/zhome/45/8/156251/Desktop/thesis/labassist/hpc_scripts/eval_data/action_verification

echo "=== Extracting FineBio PCR action clips for comparison ==="
echo ""

# Extract from PCR task videos (task 05 = PCR)
# Use persons P01, P02, P03 (diverse performers)
FINEBIO_STEMS="P01_05_01_T5,P02_05_01_T5,P03_05_01_T5"

# FineBio atomic action names (match to mydata semantically)
ACTIONS="attach_tip,eject_tip"

echo "--- Extracting clips from FineBio PCR videos ---"
python ${SCRIPTS}/data_prep/extract_action_snippets.py \
    --video_dir ${VIDEOS} \
    --label_dir ${LABELS} \
    --class_index ${CLASS_INDEX} \
    --output_dir ${OUTPUT}/finebio \
    --actions "${ACTIONS}" \
    --stems "${FINEBIO_STEMS}" \
    --padding 1.0 \
    --max_clips 5

echo ""
echo "=== Done! ==="
echo ""
echo "Clips saved to:"
echo "  ${OUTPUT}/finebio/"
echo "    attach_tip/   ← FineBio attach pipette tip"
echo "    eject_tip/    ← FineBio eject pipette tip"
echo ""
echo "Next steps:"
echo "  1. Visually inspect clips:"
echo "     ${OUTPUT}/mydata_flipped/Attach_pipette_tip/*.mp4"
echo "     ${OUTPUT}/finebio/attach_tip/*.mp4"
echo ""
echo "  2. Check if actions look similar between datasets:"
echo "     - Hand position/movement"
echo "     - Pipette visibility"
echo "     - Action timing/duration"
echo "     - Viewpoint alignment (after vflip)"
echo ""
echo "  3. If alignment looks good → proceed to Phase 2 (K-fold training)"
echo "     If not → may need to adjust crop/flip parameters or reconsider transfer learning"
echo ""


