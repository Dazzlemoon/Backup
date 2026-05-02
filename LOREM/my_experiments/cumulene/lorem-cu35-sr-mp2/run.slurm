#!/bin/bash -l

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --partition=batch
#SBATCH --time=48:00:00
#SBATCH --job-name=lorem_cumulene_cu35_sr_mp2
#SBATCH --output=log_%j.out
#SBATCH --error=err_%j.out

# SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SCRIPT_DIR="/data/home/public/qiuqizhi/LOREM/my_experiments/cumulene/lorem-cu35-sr-mp2"
cd "$SCRIPT_DIR" || exit 1

# # Conda env (optional but recommended on this cluster)
# if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
#     source "$HOME/miniconda3/etc/profile.d/conda.sh"
# elif [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
#     source "$HOME/anaconda3/etc/profile.d/conda.sh"
# fi

conda activate qqz

# Runtime environment for LOREM
export DATASETS="/data/home/public/qiuqizhi/LOREM/datasets"
export PYTHONPATH="/data/home/public/qiuqizhi/LOREM:${PYTHONPATH:-}"
# export WANDB_DISABLED=true

python "/data/home/public/qiuqizhi/LOREM/lorem/run.py"
