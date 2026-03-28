#!/bin/bash
# Create splits.json for FineBio PCR (task 05) videos with T5 view only
# This is what we need for pretraining with atomic action labels

set -euo pipefail

source /zhome/45/8/156251/Desktop/thesis/labassist/.env/bin/activate

FINEBIO_VIDEOS=/work3/s204157/data/labassist/finebio_videos_w640
FINEBIO_ROOT=/work3/s204157/data/labassist/finebio_full

mkdir -p ${FINEBIO_ROOT}

echo "=== Creating FineBio PCR (task 05) splits ==="

python3 << 'EOF'
import json
import pathlib

videos_dir = pathlib.Path("/work3/s204157/data/labassist/finebio_videos_w640")
output_path = pathlib.Path("/work3/s204157/data/labassist/finebio_full/splits_pcr_t5.json")

# Find all task 05 (PCR) videos with T5 view
# Pattern: P{person}_05_{trial}_T5.mp4
pcr_videos = sorted([f.name for f in videos_dir.glob("*_05_*_T5.mp4")])

print(f"Found {len(pcr_videos)} PCR (task 05) videos with T5 view")

# All in train for pretraining purposes
splits = {
    "train": pcr_videos,
    "val": [],
    "test": []
}

output_path.parent.mkdir(parents=True, exist_ok=True)
with open(output_path, "w") as f:
    json.dump(splits, f, indent=2)

print(f"✓ Saved splits to: {output_path}")
print(f"  Train: {len(splits['train'])} videos")

# Print first few and last few
print("\nVideos:")
for v in pcr_videos[:5]:
    print(f"  {v}")
if len(pcr_videos) > 10:
    print("  ...")
for v in pcr_videos[-5:]:
    print(f"  {v}")
EOF

echo ""
echo "=== Done ==="
