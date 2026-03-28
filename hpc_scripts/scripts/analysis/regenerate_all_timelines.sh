#!/bin/bash
#BSUB -J regen_all_timelines
#BSUB -q hpc
#BSUB -n 1
#BSUB -W 02:00
#BSUB -R "rusage[mem=8GB]"
#BSUB -o /zhome/45/8/156251/Desktop/thesis/labassist/hpc_scripts/logs/%J_regen_all_timelines.out
#BSUB -e /zhome/45/8/156251/Desktop/thesis/labassist/hpc_scripts/logs/%J_regen_all_timelines.err

# Regenerate ALL timelines with consistent colors

cd /zhome/45/8/156251/Desktop/thesis/labassist
source .env/bin/activate

COMBINED_DIR="/work3/s204157/data/labassist/combined_pcr/cache"
EXP_DIR="/work3/s204157/experiments/mydata_kfold_exp15_cascaded_vjepa"
OUT_DIR="/zhome/45/8/156251/Desktop/thesis/labassist/hpc_scripts/eval_data/cascade_v2/vjepa"

# Fold 0 - epoch 8
echo "=== Fold 0 - Epoch 8 ==="
python hpc_scripts/scripts/eval/eval_val_timelines_plus.py \
    --splits "${COMBINED_DIR}/kfold_mydata/splits_fold0.json" \
    --xdir "${COMBINED_DIR}/features_10hz_vjepa_selective_crop_flip_stride1" \
    --ydir "${COMBINED_DIR}/labels_5class" \
    --class_index "${COMBINED_DIR}/class_index_5class.json" \
    --exp_dir "${EXP_DIR}/fold0" \
    --weights "${EXP_DIR}/fold0/epoch-8.model" \
    --outdir "${OUT_DIR}/fold0/timelines_epoch8" \
    --num_R 6 \
    --make_timelines \
    --causal \
    --confidence_threshold 0.0

# Fold 1 - epoch 18
echo "=== Fold 1 - Epoch 18 ==="
python hpc_scripts/scripts/eval/eval_val_timelines_plus.py \
    --splits "${COMBINED_DIR}/kfold_mydata/splits_fold1.json" \
    --xdir "${COMBINED_DIR}/features_10hz_vjepa_selective_crop_flip_stride1" \
    --ydir "${COMBINED_DIR}/labels_5class" \
    --class_index "${COMBINED_DIR}/class_index_5class.json" \
    --exp_dir "${EXP_DIR}/fold1" \
    --weights "${EXP_DIR}/fold1/epoch-18.model" \
    --outdir "${OUT_DIR}/fold1/timelines_epoch18" \
    --num_R 6 \
    --make_timelines \
    --causal \
    --confidence_threshold 0.0

# Fold 2 - epoch 4
echo "=== Fold 2 - Epoch 4 ==="
python hpc_scripts/scripts/eval/eval_val_timelines_plus.py \
    --splits "${COMBINED_DIR}/kfold_mydata/splits_fold2.json" \
    --xdir "${COMBINED_DIR}/features_10hz_vjepa_selective_crop_flip_stride1" \
    --ydir "${COMBINED_DIR}/labels_5class" \
    --class_index "${COMBINED_DIR}/class_index_5class.json" \
    --exp_dir "${EXP_DIR}/fold2" \
    --weights "${EXP_DIR}/fold2/epoch-4.model" \
    --outdir "${OUT_DIR}/fold2/timelines_epoch4" \
    --num_R 6 \
    --make_timelines \
    --causal \
    --confidence_threshold 0.0

# Fold 3 - epoch 6
echo "=== Fold 3 - Epoch 6 ==="
python hpc_scripts/scripts/eval/eval_val_timelines_plus.py \
    --splits "${COMBINED_DIR}/kfold_mydata/splits_fold3.json" \
    --xdir "${COMBINED_DIR}/features_10hz_vjepa_selective_crop_flip_stride1" \
    --ydir "${COMBINED_DIR}/labels_5class" \
    --class_index "${COMBINED_DIR}/class_index_5class.json" \
    --exp_dir "${EXP_DIR}/fold3" \
    --weights "${EXP_DIR}/fold3/epoch-6.model" \
    --outdir "${OUT_DIR}/fold3/timelines_epoch6" \
    --num_R 6 \
    --make_timelines \
    --causal \
    --confidence_threshold 0.0

# Fold 4 - epoch 26
echo "=== Fold 4 - Epoch 26 ==="
python hpc_scripts/scripts/eval/eval_val_timelines_plus.py \
    --splits "${COMBINED_DIR}/kfold_mydata/splits_fold4.json" \
    --xdir "${COMBINED_DIR}/features_10hz_vjepa_selective_crop_flip_stride1" \
    --ydir "${COMBINED_DIR}/labels_5class" \
    --class_index "${COMBINED_DIR}/class_index_5class.json" \
    --exp_dir "${EXP_DIR}/fold4" \
    --weights "${EXP_DIR}/fold4/epoch-26.model" \
    --outdir "${OUT_DIR}/fold4/timelines_epoch26" \
    --num_R 6 \
    --make_timelines \
    --causal \
    --confidence_threshold 0.0

# Fold 5 - epoch 4
echo "=== Fold 5 - Epoch 4 ==="
python hpc_scripts/scripts/eval/eval_val_timelines_plus.py \
    --splits "${COMBINED_DIR}/kfold_mydata/splits_fold5.json" \
    --xdir "${COMBINED_DIR}/features_10hz_vjepa_selective_crop_flip_stride1" \
    --ydir "${COMBINED_DIR}/labels_5class" \
    --class_index "${COMBINED_DIR}/class_index_5class.json" \
    --exp_dir "${EXP_DIR}/fold5" \
    --weights "${EXP_DIR}/fold5/epoch-4.model" \
    --outdir "${OUT_DIR}/fold5/timelines_epoch4" \
    --num_R 6 \
    --make_timelines \
    --causal \
    --confidence_threshold 0.0

echo "=== All timelines regenerated! ==="

