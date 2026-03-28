#!/bin/bash
# Remap FIXED mydata labels to 5-class format (matching combined_pcr)

set -euo pipefail

# Activate environment
if [ -d "/zhome/45/8/156251/Desktop/thesis/labassist/.env" ]; then
  . "/zhome/45/8/156251/Desktop/thesis/labassist/.env/bin/activate"
fi

SCRIPTS=/zhome/45/8/156251/Desktop/thesis/labassist/hpc_scripts/scripts
MYDATA=/work3/s204157/data/labassist/my_data

echo "=== Remapping FIXED mydata labels to 5-class format ==="

python ${SCRIPTS}/data_prep/remap_labels_5class.py \
    --labels_in ${MYDATA}/cache_fixed/labels \
    --labels_out ${MYDATA}/cache_fixed/labels_5class \
    --splits ${MYDATA}/cache_fixed/splits.json

# Also copy the 5-class index
cat > ${MYDATA}/cache_fixed/class_index_5class.json << 'EOF'
{
  "none": 0,
  "Pipette pcr tube": 1,
  "Aspirate reagent": 2,
  "Attach pipette tip": 3,
  "Eject pipette tip": 4
}
EOF

echo "✓ Labels remapped to 5-class format"
echo "Output: ${MYDATA}/cache_fixed/labels_5class/"


