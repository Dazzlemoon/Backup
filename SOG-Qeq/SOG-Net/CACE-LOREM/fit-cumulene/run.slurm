#!/bin/bash -l

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --partition=batch
#SBATCH --time=96:00:00
#SBATCH --job-name=CACE_LOREM_cumulene_test_noimprove
#SBATCH --output=log_test_noimprove_%j.out
#SBATCH --error=err_test_noimprove_%j.out

# 工作目录：脚本所在目录（包含 fit-cace-SOG.py）

SCRIPT_DIR=/data/home/public/qiuqizhi/SOG-Qeq/SOG-Net/CACE-LOREM/fit-cumulene

# 激活环境（按需改为 conda activate xxx 或 source activate xxx）
source activate qqz

# python "$SCRIPT_DIR/fit-cace-LOREM-test.py" \
#   --sr-freeze-experiment \
#   --sr-exp-baseline-epochs 2 \
#   --sr-exp-freeze-epochs 3 \
#   --sr-exp-sr-weight-freeze 0.0 \
#   --sr-exp-lr-weight 10 \
#   --sr-exp-train-batches 8 \
#   --sr-exp-valid-batches 4


python "$SCRIPT_DIR/fit-cace-LOREM-test-noimprove.py" \
  --max-epochs 500 \
  --default-lr-weight 1.0 \
  --diag-every-epochs 1