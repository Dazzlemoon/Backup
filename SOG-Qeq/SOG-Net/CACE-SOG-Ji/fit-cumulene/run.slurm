#!/bin/bash -l

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --partition=batch
#SBATCH --time=96:00:00
#SBATCH --job-name=SOG_Qeq_cumulene
#SBATCH --output=log_%j.out
#SBATCH --error=err_%j.out

# 工作目录：脚本所在目录（包含 fit-cace-SOG.py）

SCRIPT_DIR=/data/home/public/qiuqizhi/SOG-Qeq/SOG-Net/CACE-SOG-Ji/fit-cumulene

# 激活环境（按需改为 conda activate xxx 或 source activate xxx）
source activate qqz

python "$SCRIPT_DIR/fit-cace-SOG.py"
