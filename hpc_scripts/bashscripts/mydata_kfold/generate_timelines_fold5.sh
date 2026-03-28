#!/bin/bash
#BSUB -J timelines_fold5
#BSUB -o /zhome/45/8/156251/Desktop/thesis/labassist/hpc_scripts/logs/mydata_kfold/%J_timelines_fold5.out
#BSUB -e /zhome/45/8/156251/Desktop/thesis/labassist/hpc_scripts/logs/mydata_kfold/%J_timelines_fold5.err
#BSUB -n 1
#BSUB -R "rusage[mem=8GB]"
#BSUB -W 02:00
#BSUB -q hpc

# Load modules
module load python3/3.10.12

# Activate virtual environment
source /zhome/45/8/156251/Desktop/thesis/labassist/.env/bin/activate

# Set PYTHONPATH
export PYTHONPATH="/zhome/45/8/156251/Desktop/thesis/labassist/hpc_scripts/scripts:$PYTHONPATH"

# Navigate to project root
cd /zhome/45/8/156251/Desktop/thesis/labassist

echo "==========================================="
echo "GENERATING FOLD 5 TIMELINES"
echo "==========================================="

echo ""
echo "=== I3D Fold 5 - Epoch 30 ==="
python hpc_scripts/scripts/eval/eval_val_timelines_plus.py \
  --splits /work3/s204157/data/labassist/combined_pcr/cache/kfold_mydata/splits_fold5.json \
  --xdir /work3/s204157/data/labassist/combined_pcr/cache/features_10hz_i3d_selective_crop_flip_win16 \
  --ydir /work3/s204157/data/labassist/combined_pcr/cache/labels_5class \
  --class_index /work3/s204157/data/labassist/combined_pcr/cache/class_index_5class.json \
  --exp_dir /work3/s204157/experiments/mydata_kfold_exp14_cascaded/fold5 \
  --weights /work3/s204157/experiments/mydata_kfold_exp14_cascaded/fold5/epoch-30.model \
  --outdir /work3/s204157/experiments/mydata_kfold_exp14_cascaded/fold5/timelines_epoch30 \
  --causal --num_R 3 --make_timelines

echo ""
echo "=== V-JEPA Fold 5 - Epoch 20 ==="
python hpc_scripts/scripts/eval/eval_val_timelines_plus.py \
  --splits /work3/s204157/data/labassist/combined_pcr/cache/kfold_mydata/splits_fold5.json \
  --xdir /work3/s204157/data/labassist/combined_pcr/cache/features_10hz_vjepa_selective_crop_flip_stride1 \
  --ydir /work3/s204157/data/labassist/combined_pcr/cache/labels_5class \
  --class_index /work3/s204157/data/labassist/combined_pcr/cache/class_index_5class.json \
  --exp_dir /work3/s204157/experiments/mydata_kfold_exp15_cascaded_vjepa/fold5 \
  --weights /work3/s204157/experiments/mydata_kfold_exp15_cascaded_vjepa/fold5/epoch-20.model \
  --outdir /work3/s204157/experiments/mydata_kfold_exp15_cascaded_vjepa/fold5/timelines_epoch20 \
  --causal --num_R 6 --make_timelines

echo ""
echo "==========================================="
echo "ALL FOLD 5 TIMELINES GENERATED SUCCESSFULLY"
echo "==========================================="





