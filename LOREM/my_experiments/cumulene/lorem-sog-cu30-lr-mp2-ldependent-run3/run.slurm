#!/bin/bash -l
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --partition=batch
#SBATCH --time=48:00:00
#SBATCH --job-name=lorem_cumulene_sog_cu30_lr_mp2_ldep_run3
#SBATCH --output=log_%j.out
#SBATCH --error=err_%j.out

SCRIPT_DIR="/data/home/public/qiuqizhi/LOREM/my_experiments/cumulene/lorem-sog-cu30-lr-mp2-ldependent-run3"
cd "$SCRIPT_DIR" || exit 1

echo "[DEBUG] script_dir=${SCRIPT_DIR}"
echo "[DEBUG] using settings: ${SCRIPT_DIR}/settings.yaml"
echo "[DEBUG] using model:    ${SCRIPT_DIR}/model.yaml"

export DATASETS="/data/home/public/qiuqizhi/LOREM/datasets"
eval "$(conda shell.bash hook)"
conda activate qqz
export PYTHONPATH="/data/home/public/qiuqizhi/LOREM:/data/home/public/qiuqizhi/LOREM/marathon:${PYTHONPATH:-}"
python -c "import marathon.io; print('[DEBUG] marathon import OK')"

# 防止续跑旧 run
if [ -d run ]; then
  mv run "run_backup_$(date +%Y%m%d_%H%M%S)"
fi

python "/data/home/public/qiuqizhi/LOREM/lorem/run.py"