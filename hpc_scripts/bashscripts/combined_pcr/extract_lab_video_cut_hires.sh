#!/bin/bash
#BSUB -J extract_lab_video
#BSUB -q gpuv100
#BSUB -W 8:00
#BSUB -n 4
#BSUB -R "rusage[mem=8GB]"
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -o /zhome/45/8/156251/Desktop/thesis/labassist/hpc_scripts/logs/combined_pcr/%J_extract_lab_video.out
#BSUB -e /zhome/45/8/156251/Desktop/thesis/labassist/hpc_scripts/logs/combined_pcr/%J_extract_lab_video.err

###############################################################################
# Extract High-Resolution Frames for Lab_video_cut ONLY
#
# This video is very long (~27 min) and timed out in the previous job.
# Giving it 8 hours to ensure completion.
###############################################################################

echo "========================================"
echo "Extracting Lab_video_cut Hi-Res Frames"
echo "Job ID: $LSB_JOBID"
echo "Host: $(hostname)"
echo "Date: $(date)"
echo "========================================"

source /zhome/45/8/156251/Desktop/thesis/labassist/.env/bin/activate

DATA_ROOT="/work3/s204157/data/labassist/combined_pcr"
OUTPUT_DIR="${DATA_ROOT}/frames_hires_preprocessed"

# Create a temporary splits file with just Lab_video_cut
TEMP_SPLITS="/tmp/lab_video_cut_splits_${LSB_JOBID}.json"
echo '{"train": ["Lab_video_cut.mp4"], "val": [], "test": []}' > "${TEMP_SPLITS}"

python /zhome/45/8/156251/Desktop/thesis/labassist/hpc_scripts/scripts/feature_computations/extract_frames_hires_preprocessed.py \
  --root "${DATA_ROOT}" \
  --splits "${TEMP_SPLITS}" \
  --fps 25.0 \
  --max_short_side 1080 \
  --out "${OUTPUT_DIR}" \
  --crop_top 0.0 \
  --crop_bottom 0.35 \
  --crop_left 0.10 \
  --crop_right 0.08 \
  --selective_crop \
  --vertical_flip

echo "========================================"
echo "Frame extraction complete!"
echo "========================================"

# Now run YOLO feature extraction on this video
echo "Running YOLO feature extraction on Lab_video_cut..."

YOLO_WEIGHTS="/zhome/45/8/156251/Desktop/thesis/labassist/DTULabAssistDataset/runs/detect/labassist_yolo11m22/weights/best.pt"
FEATURES_OUT="${DATA_ROOT}/cache/features_10hz_yolo_hires"

# Re-create temp splits for YOLO
echo '{"train": ["Lab_video_cut.mp4"], "val": [], "test": []}' > "${TEMP_SPLITS}"

python /zhome/45/8/156251/Desktop/thesis/labassist/hpc_scripts/scripts/feature_computations/precompute_features_yolo.py \
  --root "${DATA_ROOT}" \
  --splits "${TEMP_SPLITS}" \
  --weights "${YOLO_WEIGHTS}" \
  --fps 10 \
  --conf 0.25 \
  --imgsz 1280 \
  --device cuda:0 \
  --out "${FEATURES_OUT}" \
  --frames_preprocessed

rm -f "${TEMP_SPLITS}"

echo "========================================"
echo "All done!"
echo "Date: $(date)"
echo "========================================"
echo ""
echo "Next steps:"
echo "1. Re-run feature fusion to include Lab_video_cut"
echo "2. Then submit training job"

