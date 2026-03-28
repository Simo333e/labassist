#!/bin/bash
#BSUB -J precompute_i3d_lab
#BSUB -q gpuv100
#BSUB -n 4
#BSUB -R "rusage[mem=32G]"
#BSUB -R "span[hosts=1]"
#BSUB -W 04:00
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -o /zhome/45/8/156251/Desktop/thesis/labassist/hpc_scripts/logs/mydata/%J_precompute_i3d_lab.out
#BSUB -e /zhome/45/8/156251/Desktop/thesis/labassist/hpc_scripts/logs/mydata/%J_precompute_i3d_lab.err

set -euo pipefail

# Activate environment
source /zhome/45/8/156251/Desktop/thesis/labassist/.env/bin/activate

SCRIPTS=/zhome/45/8/156251/Desktop/thesis/labassist/hpc_scripts/scripts
MYDATA=/work3/s204157/data/labassist/my_data
I3D_WEIGHTS=/work3/s204157/models/i3d/rgb_imagenet.pt

OUT_DIR=${MYDATA}/cache_fixed/features_10hz_i3d_vflip_crop_win16

echo "=== Extracting I3D features for Lab_video_cut ==="
echo "Root: ${MYDATA}"
echo "Output dir: ${OUT_DIR}"

# Create a temporary splits file with just Lab_video_cut
TMP_SPLITS=/tmp/splits_lab_video.json
cat > ${TMP_SPLITS} << 'SPLITS'
{
  "train": ["Lab_video_cut.mp4"],
  "val": [],
  "test": []
}
SPLITS

python ${SCRIPTS}/feature_computations/precompute_features_i3d.py \
    --root ${MYDATA} \
    --splits ${TMP_SPLITS} \
    --weights ${I3D_WEIGHTS} \
    --fps 10 \
    --win 16 \
    --stride 1 \
    --imgsz 224 \
    --batch_size 8 \
    --out ${OUT_DIR} \
    --crop_top 0.0 \
    --crop_bottom 0.35 \
    --crop_left 0.0 \
    --crop_right 0.0 \
    --selective_crop \
    --vertical_flip

echo ""
echo "=== Done ==="
ls -la ${OUT_DIR}/Lab*
