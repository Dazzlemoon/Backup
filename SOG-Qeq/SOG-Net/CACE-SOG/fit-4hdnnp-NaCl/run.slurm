#!/bin/bash -l
#SBATCH --parsable
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=16
#SBATCH --gres=gpu:1
#SBATCH --partition=a100
#SBATCH --mem=32G
#SBATCH --time=48:00:00
#SBATCH --job-name=CACE_SOG_NaCl
#SBATCH --mail-type=end
#SBATCH --mail-user=2393474010@sjtu.edu.cn

# 工作目录（包含 fit-cace-SOG.py 的目录）
REMOTE_ROOT=$(readlink -f /dssg/home/acct-matxzl/matxzl/QiuQizhi/SOG-Qeq/SOG-Net/CACE-SOG/fit-4hdnnp-NaCl)
cd "$REMOTE_ROOT" || exit 1

# 激活虚拟环境（按你说的 mace_qqz）
# 如果你平时是用 conda activate，可以改成 "source ~/.bashrc && conda activate dp_qqz"
source activate dp_qqz

# 推荐固定 Python 版本命令，如果环境里是 python3 就用 python3
# 将标准输出和错误输出分别写入 log / err 文件
python fit-cace-SOG.py 1>log 2>err