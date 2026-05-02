#!/bin/bash -l

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --partition=batch
#SBATCH --time=96:00:00
#SBATCH --job-name=CACE_LOREM_cumulene_ABtest5_strict
#SBATCH --output=log_test_AB5_strict_%j.out
#SBATCH --error=err_test_AB5_strict_%j.out

# 工作目录：AB 测试脚本目录
SCRIPT_DIR=/data/home/public/qiuqizhi/SOG-Qeq/SOG-Net/CACE-LOREM/fit-cumulene2
SAVE_DIR=/data/home/public/qiuqizhi/SOG-Qeq/SOG-Net/CACE-LOREM/fit-cumulene2/loss_data/CACE_LOREM_MP2_ABtest5_strict

# 激活环境（按需改为 conda activate xxx 或 source activate xxx）
source activate qqz

python "$SCRIPT_DIR/fit-cace-LOREM-testAB5-strict.py" \
  --max-epochs 500 \
  --default-lr-weight 10.0 \
  --diag-every-epochs 5 \
  --dihedral-diag-every-epochs 10

# 训练结束后，自动汇总最新 dihedral diagnostics 为一行结论
python - <<'PY'
from pathlib import Path
import pandas as pd
import numpy as np

save_dir = Path("/data/home/public/qiuqizhi/SOG-Qeq/SOG-Net/CACE-LOREM/fit-cumulene2/loss_data/CACE_LOREM_MP2_ABtest5_strict")
csvs = sorted(save_dir.glob("dihedral_diagnostics_*.csv"), key=lambda p: p.stat().st_mtime)
if not csvs:
    print("[DihedralSummary] no dihedral_diagnostics_*.csv found.")
    raise SystemExit(0)

latest = csvs[-1]
df = pd.read_csv(latest)
if len(df) == 0:
    print(f"[DihedralSummary] empty file: {latest}")
    raise SystemExit(0)

last = df.iloc[-1]
amp_ref = float(last.get("amp_ref_rel", np.nan))
amp_long = float(last.get("amp_pred_long_rel", np.nan))
corr_long = float(last.get("corr_long_vs_ref", np.nan))
f_ratio = float(last.get("mean_f_long_over_total", np.nan))

amp_ratio = amp_long / max(amp_ref, 1e-12) if np.isfinite(amp_ref) and np.isfinite(amp_long) else np.nan

# 可按需调阈值：这里给一个“能承载角度信息”的保守判据
ok_corr = np.isfinite(corr_long) and corr_long >= 0.70
ok_amp = np.isfinite(amp_ratio) and amp_ratio >= 0.50
ok_force = np.isfinite(f_ratio) and f_ratio >= 0.01
passed = bool(ok_corr and ok_amp and ok_force)

summary = (
    f"[DihedralSummary] file={latest.name} epoch={int(last.get('global_epoch', -1))} "
    f"corr_long_vs_ref={corr_long:.4f} amp_long/ref={amp_ratio:.4f} "
    f"mean_Flong/Ftotal={f_ratio:.4f} "
    f"=> {'PASS(达到角度学习阈值)' if passed else 'NOT_PASS(未达到角度学习阈值)'}"
)
print(summary)

out_path = save_dir / "dihedral_summary_latest.txt"
out_path.write_text(summary + "\n", encoding="utf-8")
print(f"[DihedralSummary] saved: {out_path}")
PY