#!/bin/bash
# Create splits.json for all mydata videos

set -euo pipefail

MYDATA=/work3/s204157/data/labassist/my_data
SPLITS=${MYDATA}/splits.json

echo "=== Creating mydata splits.json ==="

# Create splits with all videos
cat > ${SPLITS} << 'EOF'
{
  "train": [
    "20251027_181439Z.mp4",
    "20251112_110656Z_1080p30_v2.mp4",
    "20251120_065442Z.mp4",
    "20251125_070904Z.mp4",
    "20251125_072321Z.mp4",
    "Lab_video_cut.mp4"
  ],
  "val": [],
  "test": []
}
EOF

echo "✓ Created: ${SPLITS}"
echo ""
echo "Contents:"
cat ${SPLITS}
