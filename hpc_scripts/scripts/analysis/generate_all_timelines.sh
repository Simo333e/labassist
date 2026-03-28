#!/bin/bash
#BSUB -J gen_timelines_all
#BSUB -q hpc
#BSUB -n 1
#BSUB -W 01:00
#BSUB -R "rusage[mem=8GB]"
#BSUB -o /zhome/45/8/156251/Desktop/thesis/labassist/hpc_scripts/logs/%J_gen_timelines_all.out
#BSUB -e /zhome/45/8/156251/Desktop/thesis/labassist/hpc_scripts/logs/%J_gen_timelines_all.err

# Generate timelines for fold 0 and fold 5

cd /zhome/45/8/156251/Desktop/thesis/labassist
source .env/bin/activate

COMBINED_DIR="/work3/s204157/data/labassist/combined_pcr/cache"
EXP_DIR="/work3/s204157/experiments/mydata_kfold_exp15_cascaded_vjepa"
OUT_DIR="/zhome/45/8/156251/Desktop/thesis/labassist/hpc_scripts/eval_data/cascade_v2/vjepa"

# Fold 0 - epoch 8
echo "Generating timeline for fold 0, epoch 8..."
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

# Fold 5 - epoch 4
echo "Generating timeline for fold 5, epoch 4..."
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

echo "Done!"

