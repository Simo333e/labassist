#!/bin/bash
#BSUB -J precompute_i3d_remaining
#BSUB -q gpuv100
#BSUB -n 4
#BSUB -R "rusage[mem=32G]"
#BSUB -R "span[hosts=1]"
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -W 4:00
#BSUB -o /zhome/45/8/156251/Desktop/thesis/labassist/hpc_scripts/logs/mydata/%J_precompute_remaining.out
#BSUB -e /zhome/45/8/156251/Desktop/thesis/labassist/hpc_scripts/logs/mydata/%J_precompute_remaining.err

# Extract features for the two remaining long videos that failed due to memory
# These are the longest videos and need more memory (32GB instead of 16GB)

set -euo pipefail

# Activate environment
if [ -d "/zhome/45/8/156251/Desktop/thesis/labassist/.env" ]; then
  . "/zhome/45/8/156251/Desktop/thesis/labassist/.env/bin/activate"
fi

SCRIPTS=/zhome/45/8/156251/Desktop/thesis/labassist/hpc_scripts/scripts
MYDATA=/work3/s204157/data/labassist/my_data
I3D_WEIGHTS=/work3/s204157/models/i3d/rgb_imagenet.pt

SPLITS=${MYDATA}/cache_fixed/splits.json
OUTDIR=${MYDATA}/cache_fixed/features_10hz_i3d_vflip_crop_win16

echo "=== Computing I3D features for remaining mydata videos ==="
echo "Frames: ${MYDATA}/frames"
echo "Splits: ${SPLITS}"
echo "Output: ${OUTDIR}"
echo ""
echo "Processing with smaller batch size (8) to reduce memory usage..."
echo ""

# No sharding - process all, will skip existing
python ${SCRIPTS}/feature_computations/precompute_features_i3d.py \
    --root ${MYDATA} \
    --splits ${SPLITS} \
    --weights ${I3D_WEIGHTS} \
    --fps 10 \
    --win 16 \
    --stride 1 \
    --imgsz 224 \
    --batch_size 8 \
    --out ${OUTDIR} \
    --crop_top 0.0 \
    --crop_bottom 0.35 \
    --crop_left 0.0 \
    --crop_right 0.0 \
    --selective_crop \
    --vertical_flip

echo ""
echo "=== Feature extraction complete ==="
echo "Features saved to: ${OUTDIR}"

# Verify all features exist
echo ""
echo "Verifying all features:"
for stem in 20251027_181439Z 20251112_110656Z_1080p30_v2 20251120_065442Z 20251125_070904Z 20251125_072321Z Lab_video_cut; do
    if [ -f "${OUTDIR}/${stem}_x.npy" ]; then
        echo "  ✓ ${stem}"
    else
        echo "  ✗ ${stem} MISSING"
    fi
done

