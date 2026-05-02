#!/usr/bin/env python3
"""
Eval-only comparison for two LOREM checkpoints on the same cumulene test set.

Default comparison:
- SOG: this folder's run/checkpoints/MAE_F
- CU : ../lorem-cu30-lr-mp2/run/checkpoints/MAE_F
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from ase.io import read
from sklearn.metrics import mean_absolute_error, mean_squared_error


def parse_args():
    here = Path(__file__).resolve().parent
    default_lorem_root = Path("/data/home/public/qiuqizhi/LOREM")
    p = argparse.ArgumentParser()
    p.add_argument("--lorem-root", type=Path, default=default_lorem_root)
    p.add_argument(
        "--sog-checkpoint",
        type=Path,
        default=here / "run/checkpoints/MAE_F",
        help="Folder containing model/model.msgpack, model/model.yaml, model/baseline.yaml",
    )
    p.add_argument(
        "--cu-checkpoint",
        type=Path,
        default=here.parent / "lorem-cu30-lr-mp2" / "run/checkpoints/MAE_F",
        help="Folder containing model/model.msgpack, model/model.yaml, model/baseline.yaml",
    )
    p.add_argument(
        "--dataset",
        type=Path,
        default=None,
        help="Path to cumulene_test.xyz; default uses marathon.data.datasets / cumulene_test.xyz",
    )
    p.add_argument("--add-offset", action="store_true", help="Pass add_offset=True to Calculator")
    p.add_argument("--out-csv", type=Path, default=here / "eval_only_comparison.csv")
    p.add_argument("--out-json", type=Path, default=here / "eval_only_comparison.json")
    return p.parse_args()


def ensure_checkpoint(folder: Path):
    req = [
        folder / "model/model.msgpack",
        folder / "model/model.yaml",
        folder / "model/baseline.yaml",
    ]
    missing = [str(x) for x in req if not x.exists()]
    if missing:
        raise FileNotFoundError("Missing checkpoint files:\n" + "\n".join(missing))


def evaluate_checkpoint(calc, xyz_path: Path):
    systems = read(xyz_path, index=":")
    e_ref = [s.get_potential_energy() / len(s) for s in systems]
    f_ref = [s.get_forces() for s in systems]

    for s in systems:
        s.calc = calc
    e_pred = [s.get_potential_energy() / len(s) for s in systems]
    f_pred = [s.get_forces() for s in systems]

    f_ref_flat = np.concatenate(f_ref, axis=0)
    f_pred_flat = np.concatenate(f_pred, axis=0)

    # Keep units consistent with existing project scripts:
    # energy -> meV/atom, force -> meV/Ang
    return {
        "energy_rmse_meV_per_atom": float(np.sqrt(mean_squared_error(e_ref, e_pred)) * 1000.0),
        "energy_mae_meV_per_atom": float(mean_absolute_error(e_ref, e_pred) * 1000.0),
        "force_rmse_meV_per_A": float(np.sqrt(mean_squared_error(f_ref_flat, f_pred_flat)) * 1000.0),
        "force_mae_meV_per_A": float(mean_absolute_error(f_ref_flat, f_pred_flat) * 1000.0),
    }


def main():
    args = parse_args()
    lorem_root = args.lorem_root.resolve()
    sog_ckpt = args.sog_checkpoint.resolve()
    cu_ckpt = args.cu_checkpoint.resolve()

    # Make sure local imports work no matter where script is launched.
    sys.path.insert(0, str(lorem_root / "lorem"))

    from marathon.data import datasets  # type: ignore
    from calculator import Calculator  # type: ignore

    ensure_checkpoint(sog_ckpt)
    ensure_checkpoint(cu_ckpt)

    if args.dataset is None:
        xyz_path = datasets / "cumulene_test.xyz"
    else:
        xyz_path = args.dataset.resolve()
    if not xyz_path.exists():
        raise FileNotFoundError(f"Dataset not found: {xyz_path}")

    print(f"[INFO] dataset: {xyz_path}")
    print(f"[INFO] sog checkpoint: {sog_ckpt}")
    print(f"[INFO] cu  checkpoint: {cu_ckpt}")
    print(f"[INFO] add_offset={args.add_offset}")

    sog_calc = Calculator.from_checkpoint(sog_ckpt, add_offset=args.add_offset)
    cu_calc = Calculator.from_checkpoint(cu_ckpt, add_offset=args.add_offset)

    sog_metrics = evaluate_checkpoint(sog_calc, xyz_path)
    cu_metrics = evaluate_checkpoint(cu_calc, xyz_path)

    rows = [
        {"model": "LOREM-SOG", **sog_metrics},
        {"model": "LOREM-CU", **cu_metrics},
    ]
    df = pd.DataFrame(rows)

    print("\n=== Eval-only RMSE/MAE comparison ===")
    print(df.to_string(index=False))

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out_csv, index=False)
    with args.out_json.open("w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2, ensure_ascii=False)

    print(f"\n[INFO] saved csv : {args.out_csv}")
    print(f"[INFO] saved json: {args.out_json}")


if __name__ == "__main__":
    main()

