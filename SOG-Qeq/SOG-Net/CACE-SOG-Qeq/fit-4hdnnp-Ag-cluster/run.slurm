#!/bin/bash -l

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --partition=qdagnormal
#SBATCH --time=96:00:00
#SBATCH --job-name=CACE2_SOG_Qeq_NaCl

#SBATCH --output=log_%j    # 标准输出：log + JobID
#SBATCH --error=err_%j     # 错误输出：err + JobID

# 工作目录：脚本所在目录（包含 fit-cace-SOG.py）
# SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
# cd "$SCRIPT_DIR" || exit 1

cd /work/home/acrb3qk4vo/SOG-Qeq/SOG-Net/CACE-SOG-Qeq/fit-4hdnnp-Ag-cluster
# 激活环境（按需改为 conda activate xxx 或 source activate xxx）
source activate qqz

python fit-cace-SOG.py
