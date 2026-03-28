#!/bin/bash
# Extract video snippets for visual comparison between mydata and FineBio
# This helps diagnose if actions look similar or different between datasets

set -euo pipefail

# Load ffmpeg module
module load ffmpeg/7.1

# Activate environment
if [ -d "/zhome/45/8/156251/Desktop/thesis/labassist/.env" ]; then
  . "/zhome/45/8/156251/Desktop/thesis/labassist/.env/bin/activate"
fi

SCRIPTS=/zhome/45/8/156251/Desktop/thesis/labassist/hpc_scripts/scripts
DATA=/work3/s204157/data/labassist/combined_pcr

# Output directory
OUTPUT=/zhome/45/8/156251/Desktop/thesis/labassist/hpc_scripts/eval_data/action_snippets

echo "=== Extracting action snippets for visual comparison ==="

# Extract snippets from mydata videos (including broken pipette video for comparison)
echo ""
echo "--- Extracting from mydata videos ---"
python ${SCRIPTS}/data_prep/extract_action_snippets.py \
    --video_dir ${DATA}/videos \
    --label_dir ${DATA}/cache/labels_5class \
    --class_index ${DATA}/cache/class_index_5class.json \
    --output_dir ${OUTPUT}/mydata \
    --actions "Attach pipette tip,Eject pipette tip" \
    --stems "20251120_065442Z,20251125_070904Z,20251027_181439Z" \
    --padding 1.0 \
    --max_clips 5

# Extract snippets from FineBio videos  
echo ""
echo "--- Extracting from FineBio videos ---"
python ${SCRIPTS}/data_prep/extract_action_snippets.py \
    --video_dir /work3/s204157/data/labassist/finebio_videos_w640 \
    --label_dir /work3/s204157/data/labassist/finebio_full/cache/labels_atomic \
    --class_index /work3/s204157/data/labassist/finebio_full/cache/class_index_atomic.json \
    --output_dir ${OUTPUT}/finebio \
    --actions "attach_tip,eject_tip" \
    --stems "P01_05_01_T5,P02_05_01_T5,P03_05_01_T5" \
    --padding 1.0 \
    --max_clips 5

echo ""
echo "=== Done! ==="
echo "Snippets saved to: ${OUTPUT}"
echo ""
echo "Directory structure:"
echo "  ${OUTPUT}/"
echo "    mydata/"
echo "      Attach_pipette_tip/   # mydata attach tip clips"
echo "      Eject_pipette_tip/    # mydata eject tip clips (includes broken pipette)"
echo "    finebio/"
echo "      attach_tip/           # FineBio attach tip clips"
echo "      eject_tip/            # FineBio eject tip clips"
echo ""
echo "Compare the clips to see if actions look similar between datasets!"
