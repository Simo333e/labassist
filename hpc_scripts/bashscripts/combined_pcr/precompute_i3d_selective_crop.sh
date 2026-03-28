# hpc_scripts/bashscripts/combined_pcr/precompute_i3d_selective_crop.sh
#!/bin/sh
#BSUB -q gpuv100
#BSUB -J precompute_i3d_selective_crop[1-4]
#BSUB -gpu "num=1"
#BSUB -n 4
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=32GB]"
#BSUB -M 32GB
#BSUB -W 12:00
#BSUB -o /zhome/45/8/156251/Desktop/thesis/labassist/hpc_scripts/logs/combined_pcr/%J_%I_i3d_selective.out
#BSUB -e /zhome/45/8/156251/Desktop/thesis/labassist/hpc_scripts/logs/combined_pcr/%J_%I_i3d_selective.err

set -euo pipefail

DATA_ROOT="/work3/s204157/data/labassist/combined_pcr"
SPLITS="${DATA_ROOT}/cache/splits.json"
I3D_WEIGHTS="/work3/s204157/models/i3d/rgb_imagenet.pt"

FPS="10"
WIN="64"
STRIDE="1"
IMGSZ="224"
BATCH_SIZE="4"
CENTER_STEP="2"

# Output to new directory for selective cropped features
FEATURES_OUT="${DATA_ROOT}/cache/features_10hz_i3d_selective_crop"

# Crop parameters (applied only to mydata, not FineBio)
CROP_TOP="0.0"
CROP_BOTTOM="0.35"
CROP_LEFT="0.10"
CROP_RIGHT="0.08"

mkdir -p "${FEATURES_OUT}"
mkdir -p /zhome/45/8/156251/Desktop/thesis/labassist/hpc_scripts/logs/combined_pcr

if [ -d "/zhome/45/8/156251/Desktop/thesis/labassist/.env" ]; then
  . "/zhome/45/8/156251/Desktop/thesis/labassist/.env/bin/activate"
fi

# Convert 1-based LSF index to 0-based shard index
SHARD_IDX=$(($LSB_JOBINDEX - 1))
NUM_SHARDS=4

echo "Running shard ${SHARD_IDX}/${NUM_SHARDS}"
echo "Selective cropping: mydata stems cropped (top=${CROP_TOP}, bottom=${CROP_BOTTOM}, left=${CROP_LEFT}, right=${CROP_RIGHT}), FineBio unchanged"

python /zhome/45/8/156251/Desktop/thesis/labassist/hpc_scripts/scripts/feature_computations/precompute_features_i3d.py \
  --root "${DATA_ROOT}" \
  --splits "${SPLITS}" \
  --weights "${I3D_WEIGHTS}" \
  --fps "${FPS}" \
  --win "${WIN}" \
  --stride "${STRIDE}" \
  --imgsz "${IMGSZ}" \
  --batch_size "${BATCH_SIZE}" \
  --center_step "${CENTER_STEP}" \
  --device cuda:0 \
  --out "${FEATURES_OUT}" \
  --shards "${NUM_SHARDS}" \
  --shard_idx "${SHARD_IDX}" \
  --crop_top "${CROP_TOP}" \
  --crop_bottom "${CROP_BOTTOM}" \
  --crop_left "${CROP_LEFT}" \
  --crop_right "${CROP_RIGHT}" \
  --selective_crop

