#!/bin/bash -l

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --partition=batch
#SBATCH --time=48:00:00
#SBATCH --job-name=lorem_cumulene_sog_cu30_lr_mp1_run2
#SBATCH --output=log_%j.out
#SBATCH --error=err_%j.out

# SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SCRIPT_DIR="/data/home/public/qiuqizhi/LOREM/my_experiments/cumulene/lorem-sog-cu30-lr-mp1-run2"
cd "$SCRIPT_DIR" || exit 1

echo "[DEBUG] pwd=$(pwd)"
echo "[DEBUG] using settings: $(pwd)/settings.yaml"
echo "[DEBUG] using model:    $(pwd)/model.yaml"

conda activate qqz
# Runtime environment for LOREM
export DATASETS="/data/home/public/qiuqizhi/LOREM/datasets"
export PYTHONPATH="/data/home/public/qiuqizhi/LOREM:${PYTHONPATH:-}"
# export WANDB_DISABLED=true

# 防止续跑旧 run
if [ -d run ]; then
  mv run "run_backup_$(date +%Y%m%d_%H%M%S)"
fi

python "/data/home/public/qiuqizhi/LOREM/lorem/run.py"
