#!/bin/bash
#BSUB -J eval_fold5
#BSUB -q gpuv100
#BSUB -W 2:00
#BSUB -n 4
#BSUB -R "rusage[mem=16GB]"
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -o /zhome/45/8/156251/Desktop/thesis/labassist/hpc_scripts/logs/mydata_kfold/%J_eval_fold5.out
#BSUB -e /zhome/45/8/156251/Desktop/thesis/labassist/hpc_scripts/logs/mydata_kfold/%J_eval_fold5.err

echo "=========================================="
echo "Evaluating Fold 5 for all feature types"
echo "Job ID: $LSB_JOBID"
echo "Host: $(hostname)"
echo "Date: $(date)"
echo "=========================================="

source /zhome/45/8/156251/Desktop/thesis/labassist/.env/bin/activate

# Set PYTHONPATH to include models directory
export PYTHONPATH="/zhome/45/8/156251/Desktop/thesis/labassist:${PYTHONPATH}"

echo "=== Evaluating I3D Fold 5 ==="
python /zhome/45/8/156251/Desktop/thesis/labassist/hpc_scripts/scripts/eval/sweep_checkpoints_segmental.py \
  --splits /work3/s204157/data/labassist/combined_pcr/cache/kfold_mydata/splits_fold5.json \
  --xdir /work3/s204157/data/labassist/combined_pcr/cache/features_10hz_i3d_selective_crop_flip_win16 \
  --ydir /work3/s204157/data/labassist/combined_pcr/cache/labels_5class \
  --class_index /work3/s204157/data/labassist/combined_pcr/cache/class_index_5class.json \
  --exp_dir /work3/s204157/experiments/mydata_kfold_exp14_cascaded/fold5 \
  --outdir /work3/s204157/experiments/mydata_kfold_exp14_cascaded/fold5/eval_sweep \
  --causal --num_R 3

echo ""
echo "=== Evaluating V-JEPA Fold 5 ==="
python /zhome/45/8/156251/Desktop/thesis/labassist/hpc_scripts/scripts/eval/sweep_checkpoints_segmental.py \
  --splits /work3/s204157/data/labassist/combined_pcr/cache/kfold_mydata/splits_fold5.json \
  --xdir /work3/s204157/data/labassist/combined_pcr/cache/features_10hz_vjepa_selective_crop_flip_stride1 \
  --ydir /work3/s204157/data/labassist/combined_pcr/cache/labels_5class \
  --class_index /work3/s204157/data/labassist/combined_pcr/cache/class_index_5class.json \
  --exp_dir /work3/s204157/experiments/mydata_kfold_exp15_cascaded_vjepa/fold5 \
  --outdir /work3/s204157/experiments/mydata_kfold_exp15_cascaded_vjepa/fold5/eval_sweep \
  --causal --num_R 6

echo ""
echo "=== Evaluating ResNet+Motion Fold 5 ==="
python /zhome/45/8/156251/Desktop/thesis/labassist/hpc_scripts/scripts/eval/sweep_checkpoints_segmental.py \
  --splits /work3/s204157/data/labassist/combined_pcr/cache/kfold_mydata/splits_fold5.json \
  --xdir /work3/s204157/data/labassist/combined_pcr_resnet_motion/features \
  --ydir /work3/s204157/data/labassist/combined_pcr/cache/labels_5class \
  --class_index /work3/s204157/data/labassist/combined_pcr/cache/class_index_5class.json \
  --exp_dir /work3/s204157/experiments/mydata_kfold_exp16_cascaded_resnet_motion/fold5 \
  --outdir /work3/s204157/experiments/mydata_kfold_exp16_cascaded_resnet_motion/fold5/eval_sweep \
  --causal --num_R 6

echo ""
echo "=========================================="
echo "Fold 5 evaluation complete!"
echo "=========================================="

