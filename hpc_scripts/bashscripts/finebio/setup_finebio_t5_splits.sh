#!/bin/bash
# Create splits.json for all FineBio T5 (PCR) videos
# This is needed before running V-JEPA feature extraction

set -euo pipefail

source /zhome/45/8/156251/Desktop/thesis/labassist/.env/bin/activate

FINEBIO_VIDEOS=/work3/s204157/data/labassist/finebio_videos_w640
FINEBIO_ROOT=/work3/s204157/data/labassist/finebio_full
OUTDIR=${FINEBIO_ROOT}/cache

mkdir -p ${OUTDIR}
mkdir -p ${FINEBIO_ROOT}
mkdir -p /zhome/45/8/156251/Desktop/thesis/labassist/hpc_scripts/logs/finebio

echo "=== Creating FineBio T5 splits ==="

# Create splits.json with all T5 videos in train (no val/test for pretraining)
python3 << 'EOF'
import json
import pathlib

videos_dir = pathlib.Path("/work3/s204157/data/labassist/finebio_videos_w640")
output_path = pathlib.Path("/work3/s204157/data/labassist/finebio_full/splits_t5_all.json")

# Find all T5 videos
t5_videos = sorted([f.name for f in videos_dir.glob("*_T5.mp4")])

print(f"Found {len(t5_videos)} T5 (PCR) videos")

# All in train for pretraining purposes
splits = {
    "train": t5_videos,
    "val": [],
    "test": []
}

output_path.parent.mkdir(parents=True, exist_ok=True)
with open(output_path, "w") as f:
    json.dump(splits, f, indent=2)

print(f"✓ Saved splits to: {output_path}")
print(f"  Train: {len(splits['train'])} videos")

# Also create symlink to videos dir for the V-JEPA script
videos_link = output_path.parent / "videos"
if not videos_link.exists():
    videos_link.symlink_to(videos_dir)
    print(f"✓ Created symlink: {videos_link} -> {videos_dir}")
EOF

echo ""
echo "=== Setup complete ==="
echo "Now you can run: bsub < precompute_vjepa_finebio_t5.sh"
