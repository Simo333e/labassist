#!/bin/bash
#BSUB -J fuse_train_exp11
#BSUB -q hpc
#BSUB -W 1:00
#BSUB -n 4
#BSUB -R "rusage[mem=4GB]"
#BSUB -w "done(27576137)"
#BSUB -o /zhome/45/8/156251/Desktop/thesis/labassist/hpc_scripts/logs/combined_pcr/%J_fuse_train.out
#BSUB -e /zhome/45/8/156251/Desktop/thesis/labassist/hpc_scripts/logs/combined_pcr/%J_fuse_train.err

###############################################################################
# Fuse Lab_video_cut features and submit training
# This job waits for job 27576137 (Lab_video_cut extraction + YOLO) to complete
###############################################################################

echo "========================================"
echo "Fusing Lab_video_cut and Submitting Training"
echo "Job ID: $LSB_JOBID"
echo "Date: $(date)"
echo "========================================"

source /zhome/45/8/156251/Desktop/thesis/labassist/.env/bin/activate
cd /zhome/45/8/156251/Desktop/thesis/labassist

# Check if Lab_video_cut YOLO features exist
YOLO_FEAT="/work3/s204157/data/labassist/combined_pcr/cache/features_10hz_yolo_hires/Lab_video_cut_x.npy"
if [ ! -f "$YOLO_FEAT" ]; then
    echo "ERROR: Lab_video_cut YOLO features not found at $YOLO_FEAT"
    echo "The previous job may have failed."
    exit 1
fi

echo "✓ Lab_video_cut YOLO features found"

# Run fusion for Lab_video_cut only
echo "Fusing Lab_video_cut features..."

I3D_DIR="/work3/s204157/data/labassist/combined_pcr/cache/features_10hz_i3d_selective_crop_flip_win16"
YOLO_DIR="/work3/s204157/data/labassist/combined_pcr/cache/features_10hz_yolo_hires"
FUSED_DIR="/work3/s204157/data/labassist/combined_pcr/cache/features_10hz_i3d_yolo_hires_fused"

python hpc_scripts/scripts/feature_computations/concatenate_i3d_yolo_features.py \
    --i3d_dir "${I3D_DIR}" \
    --yolo_dir "${YOLO_DIR}" \
    --output_dir "${FUSED_DIR}"

# Check fusion succeeded
if [ ! -f "${FUSED_DIR}/Lab_video_cut_x.npy" ]; then
    echo "ERROR: Fusion failed for Lab_video_cut"
    exit 1
fi

echo "✓ Feature fusion complete"

# Verify all mydata videos have fused features
echo ""
echo "Checking mydata videos in fused features..."
for stem in 20251027_181439Z 20251112_110656Z_1080p30_v2 20251120_065442Z 20251125_070904Z 20251125_072321Z Lab_video_cut; do
    if [ -f "${FUSED_DIR}/${stem}_x.npy" ]; then
        echo "  ✓ $stem"
    else
        echo "  ✗ $stem (MISSING!)"
        exit 1
    fi
done

echo ""
echo "All mydata videos have fused features!"
echo ""

# Submit training job
echo "Submitting exp11 training jobs..."
bsub < hpc_scripts/bashscripts/mydata_kfold/train_exp11_i3d_yolo_strong_reg.sh

echo "========================================"
echo "Done! Training jobs submitted."
echo "Date: $(date)"
echo "========================================"

