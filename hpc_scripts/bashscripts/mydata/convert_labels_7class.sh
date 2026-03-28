#!/bin/bash
# Generate 7-class labels from original Label Studio annotations
# This creates labels_7class/ directory with the collapsed taxonomy

set -e

cd /zhome/45/8/156251/Desktop/thesis/labassist

echo "Converting Label Studio annotations to 7-class taxonomy..."
echo ""
echo "Class mapping:"
echo "  none (0) - background + setup/prep actions"
echo "  Adjust pipette volume (1)"
echo "  Attach pipette tip (2)"
echo "  Eject pipette tip (3)"
echo "  Manipulate pcr tubes (4) - aspirate/dispense to PCR tubes"
echo "  Manipulate reagents (5) - aspirate reagent/water, dispense to mastermix"
echo "  Switch pipette (6)"
echo ""

python hpc_scripts/scripts/data_prep/convert_labelstudio_to_mstcn_17class.py \
  --labelstudio_json /work3/s204157/data/labassist/my_data/project-3-at-2025-11-26-19-24-89666430.json \
  --videos_dir /work3/s204157/data/labassist/my_data/videos \
  --out_dir /zhome/45/8/156251/Desktop/thesis/labassist/hpc_scripts/data/labels_7class_mydata \
  --target_fps 10.0 \
  --val_ratio 0.0 \
  --test_ratio 0.0 \
  --default_label "none"

echo ""
echo "✓ Conversion complete!"
echo "✓ Output: hpc_scripts/data/labels_7class_mydata/"
echo ""
echo "Next steps:"
echo "1. Copy labels to: /work3/s204157/data/labassist/combined_pcr/cache/labels_7class/"
echo "2. Copy class_index to: /work3/s204157/data/labassist/combined_pcr/cache/class_index_7class.json"
echo "3. Update Stage 2 configs to use 7-class labels"
echo "4. Retrain cascade pipeline with 7 classes"




