#!/bin/bash
# Create combined 7-class label directory for cascade training
# This combines FineBio PCR (remapped from 6-class) + mydata (7-class)

set -e

cd /zhome/45/8/156251/Desktop/thesis/labassist
source .env/bin/activate

FINEBIO_6CLASS="/work3/s204157/data/labassist/combined_pcr/cache/labels"
MYDATA_7CLASS="/zhome/45/8/156251/Desktop/thesis/labassist/hpc_scripts/data/labels_7class_mydata/labels"
OUTPUT_DIR="/zhome/45/8/156251/Desktop/thesis/labassist/hpc_scripts/data/labels_7class_combined"
FINEBIO_SPLITS="/work3/s204157/data/labassist/combined_pcr/cache/splits_finebio_only.json"

echo "=============================================="
echo "Creating 7-class combined label directory"
echo "=============================================="
echo ""
echo "Step 1: Remap FineBio 6-class labels to 7-class"
echo "----------------------------------------------"

# Remap FineBio labels
python hpc_scripts/scripts/data_prep/remap_labels_7class.py \
  --labels_in "$FINEBIO_6CLASS" \
  --labels_out "${OUTPUT_DIR}/finebio_remapped" \
  --splits "$FINEBIO_SPLITS"

echo ""
echo "Step 2: Copy mydata 7-class labels"
echo "----------------------------------------------"
mkdir -p "${OUTPUT_DIR}/combined"
cp "${OUTPUT_DIR}/finebio_remapped"/*.npy "${OUTPUT_DIR}/combined/"
cp "$MYDATA_7CLASS"/*.npy "${OUTPUT_DIR}/combined/"

echo ""
echo "Step 3: Verify combined labels"
echo "----------------------------------------------"
FINEBIO_COUNT=$(ls "${OUTPUT_DIR}/finebio_remapped"/*.npy | wc -l)
MYDATA_COUNT=$(ls "$MYDATA_7CLASS"/*.npy | wc -l)
COMBINED_COUNT=$(ls "${OUTPUT_DIR}/combined"/*.npy | wc -l)

echo "  FineBio remapped: $FINEBIO_COUNT files"
echo "  Mydata 7-class:   $MYDATA_COUNT files"
echo "  Combined total:   $COMBINED_COUNT files"

if [ "$COMBINED_COUNT" -eq $((FINEBIO_COUNT + MYDATA_COUNT)) ]; then
    echo "  ✓ All files copied successfully"
else
    echo "  ⚠ Warning: File count mismatch"
fi

echo ""
echo "=============================================="
echo "✓ 7-class labels created!"
echo "=============================================="
echo ""
echo "Output directory: $OUTPUT_DIR/combined/"
echo ""
echo "Class index (7 classes):"
echo "  0: none"
echo "  1: Adjust pipette volume"
echo "  2: Attach pipette tip"
echo "  3: Eject pipette tip"
echo "  4: Manipulate pcr tubes"
echo "  5: Manipulate reagents"
echo "  6: Switch pipette"
echo ""
echo "Note: FineBio videos don't have classes 1 and 6"
echo "      (Adjust pipette volume, Switch pipette)"




