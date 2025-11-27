# LabAssist Sim-from-Video Quickstart

## 0. Prep the terminal
```bash
cd /home/simon/ws_labassist/labassist
source .env/bin/activate
source install/setup.bash
export PYTHONPATH=/home/simon/ws_labassist/labassist/.env/lib/python3.12/site-packages:/home/simon/ws_labassist/labassist/hpc_scripts:$PYTHONPATH
```

## 1. Launch the sim 
   ```bash
  ros2 launch labassist_bringup sim_from_video.launch.py \
    video:=/home/simon/labassist_data/P01_01_01_T1.mp4 \
    ckpt:=/home/simon/labassist_data/ms_tcn_epoch150.pt \
    class_index:=/home/simon/labassist_data/class_index_steps.json \
    device:=cpu \
    collect_metrics:=true \
    repo_root:=/home/simon/ws_labassist/labassist
   ```
