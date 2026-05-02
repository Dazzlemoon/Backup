#!/bin/bash -l

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --partition=batch
#SBATCH --time=96:00:00
#SBATCH --job-name=LLZO_ReaxNet_800K
#SBATCH --output=llzo_800K_%j.out
#SBATCH --error=llzo_800K_%j.err

SCRIPT_DIR=/data/home/public/qiuqizhi/reaxnet/my_experiment/LLZO
PROJECT_ROOT=/data/home/public/qiuqizhi/reaxnet

# 按你的环境修改
source activate qqz

python "$SCRIPT_DIR/run_llzo_md.py" \
  --project-root "$PROJECT_ROOT" \
  --structure "data/LLZO/init.vasp" \
  --timestep-fs 2.0 \
  --npt-temperature-k 800 \
  --npt-pressure-atm 1.0 \
  --npt-ps 100 \
  --nvt-temperature-k 800 \
  --nvt-ps 2000 \
  --friction 0.02 \
  --traj-out "$SCRIPT_DIR/llzo_800K_nvt.traj" \
  --log-out "$SCRIPT_DIR/llzo_800K_nvt.log"

