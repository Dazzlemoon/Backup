#!/usr/bin/env python
# coding: utf-8
import sys
import os
import argparse
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

os.environ["PYTHONWARNINGS"] = "ignore"  # 全局忽略所有 warnings


import numpy as np
import torch
import torch.nn as nn
import logging
import datetime
import csv
from pathlib import Path

import cace
from cace.representations import CaceLoremShortRange
from cace.modules import CosineCutoff, MollifierCutoff, PolynomialCutoff
from cace.modules import BesselRBF, GaussianRBF, GaussianRBFCentered

from cace.models.atomistic import NeuralNetworkPotential
from cace.tasks.train import TrainingTask

torch.set_default_dtype(torch.float32)

cace.tools.setup_logger(level='INFO')
cutoff = 5.0

#val_ratio = float(sys.argv[1])
MP = 2
N_dl = 2
# L2 (basis-covariant) consistency: rotate inputs and enforce q_l consistency.
ENABLE_L2_CONSISTENCY = False
L2_ROT_CONSISTENCY_WEIGHT = 0.05
EFFECTIVE_L2_WEIGHT = L2_ROT_CONSISTENCY_WEIGHT if ENABLE_L2_CONSISTENCY else 0.0
LR_WEIGHT_SCHEDULE = [1.0, 5.0, 10.0, 20.0]
# Fast-train knobs: run several epochs per fit() call so validation/checkpoint/diagnostics
# happen less frequently.
TRAIN_CHUNK_EPOCHS = 5
CHECKPOINT_STRIDE_EPOCHS = 10
DIAG_EVERY_GLOBAL_EPOCHS = 10
save_folder = "/data/home/public/qiuqizhi/SOG-Qeq/SOG-Net/CACE-LOREM/fit-cumulene/loss_data/CACE_LOREM_MP"+str(MP)+"_test"
DEFAULT_MAX_EPOCHS = 500
DEFAULT_START_LR = 1e-3
DEFAULT_MIN_LR = 1e-6
DEFAULT_START_DECAY_AFTER = 10
DEFAULT_LR_WEIGHT = 10.0


def parse_args():
    parser = argparse.ArgumentParser(
        description="CACE-LOREM training script with fast debug mode."
    )
    parser.add_argument(
        "--quick-test",
        action="store_true",
        help="Run a short diagnostic training instead of full schedule.",
    )
    parser.add_argument(
        "--max-train-batches",
        type=int,
        default=12,
        help="Max train batches per epoch in quick-test mode.",
    )
    parser.add_argument(
        "--max-valid-batches",
        type=int,
        default=6,
        help="Max valid/test batches in quick-test mode.",
    )
    parser.add_argument(
        "--quick-epochs",
        type=int,
        default=6,
        help="Epochs to run in quick-test mode.",
    )
    parser.add_argument(
        "--max-epochs",
        type=int,
        default=DEFAULT_MAX_EPOCHS,
        help="Total training epochs for default (non-quick, non-sr-freeze) mode.",
    )
    parser.add_argument(
        "--start-learning-rate",
        type=float,
        default=DEFAULT_START_LR,
        help="Initial learning rate (LOREM-style default: 1e-3).",
    )
    parser.add_argument(
        "--min-learning-rate",
        type=float,
        default=DEFAULT_MIN_LR,
        help="Minimum learning rate for linear decay.",
    )
    parser.add_argument(
        "--start-decay-after",
        type=int,
        default=DEFAULT_START_DECAY_AFTER,
        help="Epoch index to start linear LR decay.",
    )
    parser.add_argument(
        "--diag-every-epochs",
        type=int,
        default=1,
        help="Run LR diagnostics every N global epochs.",
    )
    parser.add_argument(
        "--default-lr-weight",
        type=float,
        default=DEFAULT_LR_WEIGHT,
        help="LR branch weight for default long training mode.",
    )
    parser.add_argument(
        "--sr-freeze-experiment",
        action="store_true",
        help="Run two-phase SR-freeze diagnostic experiment.",
    )
    parser.add_argument(
        "--sr-exp-baseline-epochs",
        type=int,
        default=3,
        help="Phase-1 epochs: normal joint training before SR freeze.",
    )
    parser.add_argument(
        "--sr-exp-freeze-epochs",
        type=int,
        default=5,
        help="Phase-2 epochs: freeze SR readout and train LR path.",
    )
    parser.add_argument(
        "--sr-exp-train-batches",
        type=int,
        default=24,
        help="Max train batches/epoch in SR-freeze experiment.",
    )
    parser.add_argument(
        "--sr-exp-valid-batches",
        type=int,
        default=12,
        help="Max valid/test batches in SR-freeze experiment.",
    )
    parser.add_argument(
        "--sr-exp-energy-weight",
        type=float,
        default=1.0,
        help="Energy loss weight used in SR-freeze experiment phases.",
    )
    parser.add_argument(
        "--sr-exp-lr-weight",
        type=float,
        default=10.0,
        help="LR branch weight used in SR-freeze experiment phases.",
    )
    parser.add_argument(
        "--sr-exp-sr-weight-freeze",
        type=float,
        default=0.0,
        help="SR branch weight in freeze phase (0.0 means LR-only supervision).",
    )
    return parser.parse_args()


class LimitedLoader:
    """Wrap a loader to expose at most max_batches batches."""

    def __init__(self, loader, max_batches: int):
        self.loader = loader
        self.max_batches = max(1, int(max_batches))

    def __iter__(self):
        for idx, batch in enumerate(self.loader):
            if idx >= self.max_batches:
                break
            yield batch

    def __len__(self):
        return min(len(self.loader), self.max_batches)


args = parse_args()
DIAG_EVERY_GLOBAL_EPOCHS = max(1, int(args.diag_every_epochs))
if args.quick_test:
    save_folder = (
        "/data/home/public/qiuqizhi/SOG-Qeq/SOG-Net/CACE-LOREM/fit-cumulene/loss_data/"
        + "CACE_LOREM_MP"
        + str(MP)
        + "_test_quick"
    )
    TRAIN_CHUNK_EPOCHS = max(1, min(2, args.quick_epochs))
    CHECKPOINT_STRIDE_EPOCHS = 1
    DIAG_EVERY_GLOBAL_EPOCHS = 1
if args.sr_freeze_experiment:
    save_folder = (
        "/data/home/public/qiuqizhi/SOG-Qeq/SOG-Net/CACE-LOREM/fit-cumulene/loss_data/"
        + "CACE_LOREM_MP"
        + str(MP)
        + "_test_sr_freeze"
    )
    TRAIN_CHUNK_EPOCHS = 1
    CHECKPOINT_STRIDE_EPOCHS = 1
    DIAG_EVERY_GLOBAL_EPOCHS = 1
now = datetime.datetime.now()
time_name=now.strftime("%Y%m%d_%H%M%S")
os.makedirs(save_folder, exist_ok=True)

print("reading data")
dataset_root = Path("/data/home/public/qiuqizhi/LOREM/datasets")
collection = cace.tasks.get_dataset_from_xyz(
    train_path=str(dataset_root / "cumulene_train.xyz"),
    valid_path=str(dataset_root / "cumulene_valid.xyz"),
    test_path=str(dataset_root / "cumulene_test.xyz"),
    seed=42,
    cutoff=cutoff,
    data_key={'energy': 'energy', 'forces': 'forces'},
    # CACE cumulene baseline (same as LOREM cace script)
    atomic_energies={1: -2.6544, 6: -5.9724},
)
batch_size = 8

train_loader = cace.tasks.load_data_loader(collection=collection,
                              data_type='train',
                              batch_size=batch_size,
                              )

valid_loader = cace.tasks.load_data_loader(collection=collection,
                              data_type='valid',
                              batch_size=8,
                              )

test_loader = cace.tasks.load_data_loader(collection=collection,
                              data_type='test',
                              batch_size=8,
                              )
if args.quick_test:
    train_loader = LimitedLoader(train_loader, args.max_train_batches)
    valid_loader = LimitedLoader(valid_loader, args.max_valid_batches)
    test_loader = LimitedLoader(test_loader, args.max_valid_batches)
    print(
        f"[QuickTest] enabled: epochs={args.quick_epochs}, "
        f"train_batches<={len(train_loader)}, valid_batches<={len(valid_loader)}"
    )
elif args.sr_freeze_experiment:
    train_loader = LimitedLoader(train_loader, args.sr_exp_train_batches)
    valid_loader = LimitedLoader(valid_loader, args.sr_exp_valid_batches)
    test_loader = LimitedLoader(test_loader, args.sr_exp_valid_batches)
    print(
        f"[SRFreezeExp] enabled: baseline_epochs={args.sr_exp_baseline_epochs}, "
        f"freeze_epochs={args.sr_exp_freeze_epochs}, "
        f"train_batches<={len(train_loader)}, valid_batches<={len(valid_loader)}"
    )

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#device = cace.tools.init_device(use_device)
print(f"device: {device}")


print("building CACE representation")
radial_basis = BesselRBF(cutoff=cutoff, n_rbf=6, trainable=True)
#cutoff_fn = CosineCutoff(cutoff=cutoff)
cutoff_fn = PolynomialCutoff(cutoff=cutoff)

cace_representation = CaceLoremShortRange(
    zs=[1, 6],
    n_atom_basis=8,
    embed_receiver_nodes=True,
    cutoff=cutoff,
    cutoff_fn=cutoff_fn,
    radial_basis=radial_basis,
    n_radial_basis=12,
    max_l=3,
    max_nu=3,
    num_message_passing=MP,
    n_scalar_features=32,
    type_message_passing=['Bchi'],
    args_message_passing={'Bchi': {'shared_channels': True, 'shared_l': True}},
    #avg_num_neighbors=1,
    device=device,
    timeit=False
           )

cace_representation.to(device)
print(f"Representation: {cace_representation}")

atomwise = cace.modules.atomwise.Atomwise(n_layers=3,
                                         output_key='CACE_energy',
                                         n_hidden=[1,1],
                                         use_batchnorm=False,
                                         add_linear_nn=True)


forces = cace.modules.forces.Forces(energy_key='CACE_energy',
                                    forces_key='CACE_forces')

print("building CACE NNP")
cace_nnp_sr = NeuralNetworkPotential(
    input_modules=None,
    representation=cace_representation,
    output_modules=[atomwise, forces]
)


q = cace.modules.MultipoleChargeHead(
    p_feature_key="p_features",
    output_key="q",
    scalar_hidden=16,
    pair_hidden=16,
)
# Expose multipole split channels in model outputs for training-time diagnostics.
for k in ["q_monopole", "q_dipole", "q_quadrupole"]:
    if k not in q.model_outputs:
        q.model_outputs.append(k)

# Coulomb/Ewald potential from multipole channels. compute_field=True to enable LR feedback.
ep = cace.modules.EwaldPotential(
    feature_key="q",
    output_key="ewald_raw",
    aggregation_mode="sum",
    compute_field=True,
    remove_self_interaction=False,
    exponent=1,
)

lr_readout = cace.modules.LoremLongRangeReadout(
    s_feature_key="s_features",
    s_l0_feature_key="s_l0_features",
    s_l1_feature_key="s_l1_features",
    s_l2_feature_key="s_l2_features",
    field_key="q_field",
    output_key="lr_energy",
    per_atom_output_key="lr_energy_atom",
    aggregation_mode="sum",
    hidden=16,
)

forces_lr = cace.modules.Forces(energy_key='lr_energy',
                                    forces_key='SOG_forces')

cace_nnp_lr = NeuralNetworkPotential(
    input_modules=None,
    representation=cace_representation,
    output_modules=[q, ep, lr_readout, forces_lr]
)

pot2 = {'CACE_energy': 'lr_energy',
        'CACE_forces': 'SOG_forces',
        'weight': 1.0
       }

pot1 = {'CACE_energy': 'CACE_energy', 
        'CACE_forces': 'CACE_forces',
       }

cace_nnp = cace.models.CombinePotential([cace_nnp_sr, cace_nnp_lr], [pot1,pot2])
#cace_nnp = cace.models.CombinePotential([cace_nnp_sr], [pot1])
cace_nnp.to(device)
print(f"LR branch weight schedule: {LR_WEIGHT_SCHEDULE}")
print(f"L2 consistency enabled: {ENABLE_L2_CONSISTENCY}")
print(f"L2 rotation consistency weight (effective): {EFFECTIVE_L2_WEIGHT}")
print(
    f"Fast settings: train_chunk_epochs={TRAIN_CHUNK_EPOCHS}, "
    f"checkpoint_stride={CHECKPOINT_STRIDE_EPOCHS}, "
    f"diag_every={DIAG_EVERY_GLOBAL_EPOCHS}"
)


def random_rotation_matrix(device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    a = torch.randn(3, 3, device=device, dtype=dtype)
    q, _ = torch.linalg.qr(a)
    if torch.linalg.det(q) < 0:
        q[:, 0] = -q[:, 0]
    return q


def _rotate_last_dim3(x: torch.Tensor, rotation: torch.Tensor) -> torch.Tensor:
    return torch.matmul(x, rotation.transpose(0, 1))


def _quad5_to_matrix(q5: torch.Tensor) -> torch.Tensor:
    # Basis: [Qxx-Qyy, 2Qzz-Qxx-Qyy, Qxy, Qxz, Qyz]
    c0 = q5[..., 0]
    c1 = q5[..., 1]
    c2 = q5[..., 2]
    c3 = q5[..., 3]
    c4 = q5[..., 4]

    qxx = (c0 - c1 / 3.0) / 2.0
    qyy = (-c0 - c1 / 3.0) / 2.0
    qzz = c1 / 3.0

    out = torch.zeros(*q5.shape[:-1], 3, 3, device=q5.device, dtype=q5.dtype)
    out[..., 0, 0] = qxx
    out[..., 1, 1] = qyy
    out[..., 2, 2] = qzz
    out[..., 0, 1] = c2
    out[..., 1, 0] = c2
    out[..., 0, 2] = c3
    out[..., 2, 0] = c3
    out[..., 1, 2] = c4
    out[..., 2, 1] = c4
    return out


def _matrix_to_quad5(qmat: torch.Tensor) -> torch.Tensor:
    c0 = qmat[..., 0, 0] - qmat[..., 1, 1]
    c1 = 2.0 * qmat[..., 2, 2] - qmat[..., 0, 0] - qmat[..., 1, 1]
    c2 = qmat[..., 0, 1]
    c3 = qmat[..., 0, 2]
    c4 = qmat[..., 1, 2]
    return torch.stack([c0, c1, c2, c3, c4], dim=-1)


def _rotate_quadrupole5(q5: torch.Tensor, rotation: torch.Tensor) -> torch.Tensor:
    qmat = _quad5_to_matrix(q5)
    rot_t = rotation.transpose(0, 1)
    qmat_rot = torch.matmul(torch.matmul(rotation.unsqueeze(0), qmat), rot_t.unsqueeze(0))
    return _matrix_to_quad5(qmat_rot)


def _build_rotated_batch_dict(batch_dict: dict, rotation: torch.Tensor) -> dict:
    rotated = {}
    for key, value in batch_dict.items():
        if not torch.is_tensor(value):
            rotated[key] = value
            continue
        if key in {"positions", "shifts", "displacement"} and value.shape[-1] == 3:
            rotated[key] = _rotate_last_dim3(value, rotation)
        elif key == "cell" and value.shape[-1] == 3 and value.shape[-2] == 3:
            rotated[key] = torch.matmul(value, rotation.transpose(0, 1))
        else:
            rotated[key] = value
    return rotated


def l2_rotation_consistency_loss(model, batch_dict, pred, output_index=None):
    # Require q(l=1) and q(l=2) to transform consistently under rotation.
    if "q_dipole" not in pred or "q_quadrupole" not in pred:
        return torch.zeros((), device=next(model.parameters()).device, dtype=torch.float32)

    dtype = pred["q_dipole"].dtype
    dev = pred["q_dipole"].device
    rotation = random_rotation_matrix(device=dev, dtype=dtype)
    rotated_batch_dict = _build_rotated_batch_dict(batch_dict, rotation)

    if hasattr(model, "models") and len(model.models) > 1:
        lr_model = model.models[1]
    else:
        lr_model = model

    rep_rot = lr_model.representation(rotated_batch_dict)
    q_rot_data = lr_model.output_modules[0](rep_rot, training=True, output_index=output_index)
    q1_rot = q_rot_data["q_dipole"]
    q2_rot = q_rot_data["q_quadrupole"]

    q1_target = torch.matmul(pred["q_dipole"].detach(), rotation.transpose(0, 1))
    q2_target = _rotate_quadrupole5(pred["q_quadrupole"].detach(), rotation)

    mse = torch.nn.functional.mse_loss
    return mse(q1_rot, q1_target) + mse(q2_rot, q2_target)


def compute_long_range_diagnostics(model, data_loader, device, lr_weight=1.0):
    model.eval()
    sum_q = 0.0
    sum_q2 = 0.0
    sum_abs_q = 0.0
    sum_qmono = 0.0
    sum_qmono2 = 0.0
    sum_abs_qmono = 0.0
    qmono_count = 0
    sum_qdip = 0.0
    sum_qdip2 = 0.0
    sum_abs_qdip = 0.0
    qdip_count = 0
    sum_qquad = 0.0
    sum_qquad2 = 0.0
    sum_abs_qquad = 0.0
    qquad_count = 0
    q_count = 0
    sum_abs_f_long = 0.0
    sum_abs_f_total = 0.0
    sum_abs_e_long = 0.0
    sum_abs_e_total = 0.0
    sum_e_long = 0.0
    sum_e_total = 0.0
    sum_e_short = 0.0
    sum_abs_e_short = 0.0
    e_count = 0
    eps = 1e-12

    # NOTE: do NOT use torch.no_grad() here because model outputs include Forces modules
    # that call torch.autograd.grad internally.
    with torch.enable_grad():
        for batch in data_loader:
            batch = batch.to(device)
            batch_dict = batch.to_dict()

            pred_total = model(batch_dict, training=False)
            pred_lr = model.models[1](batch_dict, training=False)

            q_tensor = pred_lr["q"].reshape(-1).detach()
            sum_q += q_tensor.sum().item()
            sum_q2 += (q_tensor * q_tensor).sum().item()
            sum_abs_q += torch.abs(q_tensor).sum().item()
            q_count += q_tensor.numel()
            if "q_monopole" in pred_lr:
                q_mono = pred_lr["q_monopole"].reshape(-1).detach()
                sum_qmono += q_mono.sum().item()
                sum_qmono2 += (q_mono * q_mono).sum().item()
                sum_abs_qmono += torch.abs(q_mono).sum().item()
                qmono_count += q_mono.numel()
            if "q_dipole" in pred_lr:
                q_dip = pred_lr["q_dipole"].reshape(-1).detach()
                sum_qdip += q_dip.sum().item()
                sum_qdip2 += (q_dip * q_dip).sum().item()
                sum_abs_qdip += torch.abs(q_dip).sum().item()
                qdip_count += q_dip.numel()
            if "q_quadrupole" in pred_lr:
                q_quad = pred_lr["q_quadrupole"].reshape(-1).detach()
                sum_qquad += q_quad.sum().item()
                sum_qquad2 += (q_quad * q_quad).sum().item()
                sum_abs_qquad += torch.abs(q_quad).sum().item()
                qquad_count += q_quad.numel()

            f_long = (lr_weight * pred_lr["SOG_forces"]).reshape(-1, 3).detach()
            f_total = pred_total["CACE_forces"].reshape(-1, 3).detach()
            sum_abs_f_long += torch.linalg.norm(f_long, dim=1).sum().item()
            sum_abs_f_total += torch.linalg.norm(f_total, dim=1).sum().item()

            e_long = (lr_weight * pred_lr["lr_energy"]).reshape(-1).detach()
            e_total = pred_total["CACE_energy"].reshape(-1).detach()
            e_short = (e_total - e_long).detach()
            sum_abs_e_long += torch.abs(e_long).sum().item()
            sum_abs_e_total += torch.abs(e_total).sum().item()
            sum_e_long += e_long.sum().item()
            sum_e_total += e_total.sum().item()
            sum_e_short += e_short.sum().item()
            sum_abs_e_short += torch.abs(e_short).sum().item()
            e_count += e_total.numel()

    mean_q = sum_q / max(q_count, 1)
    std_q = (max(sum_q2 / max(q_count, 1) - mean_q * mean_q, 0.0)) ** 0.5
    mean_abs_q = sum_abs_q / max(q_count, 1)
    mean_q_mono = sum_qmono / max(qmono_count, 1)
    std_q_mono = (max(sum_qmono2 / max(qmono_count, 1) - mean_q_mono * mean_q_mono, 0.0)) ** 0.5
    mean_abs_q_mono = sum_abs_qmono / max(qmono_count, 1)
    mean_q_dipole = sum_qdip / max(qdip_count, 1)
    std_q_dipole = (sum_qdip2 / max(qdip_count, 1)) ** 0.5
    mean_abs_q_dipole = sum_abs_qdip / max(qdip_count, 1)
    mean_q_quadrupole = sum_qquad / max(qquad_count, 1)
    std_q_quadrupole = (sum_qquad2 / max(qquad_count, 1)) ** 0.5
    mean_abs_q_quadrupole = sum_abs_qquad / max(qquad_count, 1)
    mean_e_long = sum_e_long / max(e_count, 1)
    mean_e_total = sum_e_total / max(e_count, 1)
    mean_e_short = sum_e_short / max(e_count, 1)
    mean_abs_e_long = sum_abs_e_long / max(e_count, 1)
    mean_abs_e_total = sum_abs_e_total / max(e_count, 1)
    mean_abs_e_short = sum_abs_e_short / max(e_count, 1)

    f_long_ratio = sum_abs_f_long / max(sum_abs_f_total, eps)
    e_long_ratio = sum_abs_e_long / max(sum_abs_e_total, eps)
    return (
        std_q,
        mean_abs_q,
        mean_q_mono,
        std_q_mono,
        mean_abs_q_mono,
        mean_q_dipole,
        std_q_dipole,
        mean_abs_q_dipole,
        mean_q_quadrupole,
        std_q_quadrupole,
        mean_abs_q_quadrupole,
        mean_e_short,
        mean_e_long,
        mean_e_total,
        mean_abs_e_short,
        mean_abs_e_long,
        mean_abs_e_total,
        f_long_ratio,
        e_long_ratio,
    )


print(f"First train loop:")
energy_loss = cace.tasks.GetLoss(
    target_name='energy',
    predict_name='CACE_energy',
    loss_fn=torch.nn.MSELoss(),
    loss_weight=0.5
)

force_loss = cace.tasks.GetLoss(
    target_name='forces',
    predict_name='CACE_forces',
    loss_fn=torch.nn.MSELoss(),
    loss_weight=0.5
)

from cace.tools import Metrics

e_metric = Metrics(
    target_name='energy',
    predict_name='CACE_energy',
    name='e/atom',
    per_atom=True
)

f_metric = Metrics(
    target_name='forces',
    predict_name='CACE_forces',
    name='f'
)

# Example usage
print("creating training task")

boost = 1

# LOREM-like stable optimizer/scheduler defaults:
# - start LR 1e-3
# - linear decay to 1e-6
# - optimizer preferring Lamb (fallback to AdamW if Lamb unavailable)
start_lr = float(args.start_learning_rate)
min_lr = float(args.min_learning_rate)
start_decay_after = max(0, int(args.start_decay_after))
max_epochs_for_decay = max(1, int(args.max_epochs))
min_ratio = max(min_lr / max(start_lr, 1e-16), 1e-8)

def _linear_lr_lambda(epoch_idx: int) -> float:
    if epoch_idx < start_decay_after:
        return 1.0
    if max_epochs_for_decay <= start_decay_after:
        return min_ratio
    progress = (epoch_idx - start_decay_after) / (max_epochs_for_decay - start_decay_after)
    progress = min(max(progress, 0.0), 1.0)
    return 1.0 - progress * (1.0 - min_ratio)

optimizer_cls = torch.optim.AdamW
optimizer_name = "AdamW(fallback)"
try:
    import torch_optimizer as torch_optimizer  # type: ignore
    optimizer_cls = torch_optimizer.Lamb
    optimizer_name = "Lamb(torch_optimizer)"
except Exception:
    if hasattr(torch.optim, "Lamb"):
        optimizer_cls = torch.optim.Lamb  # type: ignore[attr-defined]
        optimizer_name = "Lamb(torch.optim)"

optimizer_args = {'lr': start_lr, 'betas': (0.9, 0.999), 'weight_decay': 0.0}
scheduler_cls = torch.optim.lr_scheduler.LambdaLR
scheduler_args = {'lr_lambda': _linear_lr_lambda}
print(
    f"Optimizer setup: {optimizer_name}, start_lr={start_lr:.3e}, "
    f"min_lr={min_lr:.3e}, linear_decay_start_epoch={start_decay_after}, "
    f"max_epochs_for_decay={max_epochs_for_decay}"
)

task = TrainingTask(
    model=cace_nnp,
    losses=[energy_loss, force_loss],
    metrics=[e_metric, f_metric],
    device=device,
    optimizer_cls=optimizer_cls,
    optimizer_args=optimizer_args,
    scheduler_cls=scheduler_cls,
    scheduler_args=scheduler_args,
    max_grad_norm=10,
    ema=False, #True,
    ema_start=10,
    warmup_steps=5,
    consistency_loss_fn=l2_rotation_consistency_loss if ENABLE_L2_CONSISTENCY else None,
    consistency_loss_weight=EFFECTIVE_L2_WEIGHT,
    save_folder = save_folder,
    time_name = time_name
)

def set_module_trainable(module: torch.nn.Module, trainable: bool):
    for param in module.parameters():
        param.requires_grad = trainable


phase_plan = []
if args.sr_freeze_experiment:
    # Phase 1: normal joint training.
    phase_plan.append({
        "name": "baseline_joint",
        "epochs": max(1, args.sr_exp_baseline_epochs),
        "energy_weight": float(args.sr_exp_energy_weight),
        "lr_weight": float(args.sr_exp_lr_weight),
        "sr_weight": 1.0,
        "freeze_sr_readout": False,
    })
    # Phase 2: freeze SR readout and train LR branch.
    phase_plan.append({
        "name": "sr_frozen_lr_focus",
        "epochs": max(1, args.sr_exp_freeze_epochs),
        "energy_weight": float(args.sr_exp_energy_weight),
        "lr_weight": float(args.sr_exp_lr_weight),
        "sr_weight": float(args.sr_exp_sr_weight_freeze),
        "freeze_sr_readout": True,
    })
    print(f"[SRFreezeExp] phase_plan={phase_plan}")
else:
    # Default training now follows LOREM-like stable weighting:
    # fixed energy loss weight=0.5 and long horizon max_epochs,
    # with LR enhancement via configurable default_lw_weight.
    stage_plan = [
        (max(1, int(args.max_epochs)), 0.5),
    ]
    if args.quick_test:
        stage_plan = [
            (max(1, args.quick_epochs), 1.0),
        ]
        LR_WEIGHT_SCHEDULE = [1.0]
        print(f"[QuickTest] stage_plan={stage_plan}, lr_schedule={LR_WEIGHT_SCHEDULE}")
    for stage_idx, (epochs_now, energy_weight_now) in enumerate(stage_plan, start=1):
        lr_weight_now = LR_WEIGHT_SCHEDULE[min(stage_idx - 1, len(LR_WEIGHT_SCHEDULE) - 1)]
        lr_weight_phase = float(lr_weight_now) if args.quick_test else float(args.default_lr_weight)
        phase_plan.append({
            "name": f"stage_{stage_idx}",
            "epochs": int(epochs_now),
            "energy_weight": float(energy_weight_now),
            "lr_weight": lr_weight_phase,
            "sr_weight": 1.0,
            "freeze_sr_readout": False,
        })

diag_csv_path = os.path.join(save_folder, 'diagnostics_' + time_name + '.csv')
with open(diag_csv_path, 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow([
        "global_epoch", "stage", "phase", "stage_epoch", "lr_weight", "sr_weight",
        "std_q", "mean_abs_q",
        "mean_q_mono", "std_q_mono", "mean_abs_q_mono",
        "mean_q_dipole", "std_q_dipole", "mean_abs_q_dipole",
        "mean_q_quadrupole", "std_q_quadrupole", "mean_abs_q_quadrupole",
        "mean_e_short", "mean_e_long", "mean_e_total",
        "mean_abs_e_short", "mean_abs_e_long", "mean_abs_e_total",
        "f_long_over_f_total", "e_long_over_e_total"
    ])

global_epoch = 0
phase_last_diag = {}
for stage_idx, phase_cfg in enumerate(phase_plan, start=1):
    phase_name = phase_cfg["name"]
    epochs_now = int(phase_cfg["epochs"])
    energy_weight_now = float(phase_cfg["energy_weight"])
    lr_weight_now = float(phase_cfg["lr_weight"])
    sr_weight_now = float(phase_cfg["sr_weight"])
    freeze_sr_now = bool(phase_cfg["freeze_sr_readout"])

    set_module_trainable(atomwise, not freeze_sr_now)
    task.model.potential_keys[0]["weight"] = sr_weight_now
    task.model.potential_keys[1]["weight"] = lr_weight_now
    print(
        f"Stage {stage_idx} ({phase_name}): epochs={epochs_now}, "
        f"energy_weight={energy_weight_now}, force_weight=0.5, "
        f"lr_weight={lr_weight_now}, sr_weight={sr_weight_now}, "
        f"freeze_sr_readout={freeze_sr_now}"
    )
    stage_energy_loss = cace.tasks.GetLoss(
        target_name='energy',
        predict_name='CACE_energy',
        loss_fn=torch.nn.MSELoss(),
        loss_weight=energy_weight_now
    )
    task.update_loss([stage_energy_loss, force_loss])
    stage_epoch = 0
    while stage_epoch < epochs_now:
        run_epochs = min(TRAIN_CHUNK_EPOCHS, epochs_now - stage_epoch)
        task.fit(
            train_loader,
            valid_loader,
            epochs=run_epochs,
            screen_nan=False,
            # Validate once per chunk to reduce overhead.
            val_stride=run_epochs,
            bestmodel_path="best_model.pth",
            checkpoint_path="checkpoint.pt",
            checkpoint_stride=CHECKPOINT_STRIDE_EPOCHS,
        )
        stage_epoch += run_epochs
        global_epoch += run_epochs
        should_diag = (
            (global_epoch % DIAG_EVERY_GLOBAL_EPOCHS == 0)
            or (stage_epoch == epochs_now)
        )
        if should_diag:
            (
                std_q,
                mean_abs_q,
                mean_q_mono,
                std_q_mono,
                mean_abs_q_mono,
                mean_q_dipole,
                std_q_dipole,
                mean_abs_q_dipole,
                mean_q_quadrupole,
                std_q_quadrupole,
                mean_abs_q_quadrupole,
                mean_e_short,
                mean_e_long,
                mean_e_total,
                mean_abs_e_short,
                mean_abs_e_long,
                mean_abs_e_total,
                f_long_ratio,
                e_long_ratio,
            ) = compute_long_range_diagnostics(task.model, valid_loader, device, lr_weight=lr_weight_now)
            print(
                f"[Diag] epoch={global_epoch} stage={stage_idx}.{stage_epoch} "
                f"lr_weight={lr_weight_now:.1f} "
                f"std(q)={std_q:.6f} mean(|q|)={mean_abs_q:.6f} "
                f"q0(mean/std/|.|)=({mean_q_mono:.6f}/{std_q_mono:.6f}/{mean_abs_q_mono:.6f}) "
                f"q1(mean/std/|.|)=({mean_q_dipole:.6f}/{std_q_dipole:.6f}/{mean_abs_q_dipole:.6f}) "
                f"q2(mean/std/|.|)=({mean_q_quadrupole:.6f}/{std_q_quadrupole:.6f}/{mean_abs_q_quadrupole:.6f}) "
                f"E_short/E_long/E_total=({mean_e_short:.6f}/{mean_e_long:.6f}/{mean_e_total:.6f}) "
                f"||F_long||/||F_total||={f_long_ratio:.6f} E_long/E_total={e_long_ratio:.6f}"
            )
            with open(diag_csv_path, 'a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([
                    global_epoch, stage_idx, phase_name, stage_epoch, lr_weight_now, sr_weight_now,
                    std_q, mean_abs_q,
                    mean_q_mono, std_q_mono, mean_abs_q_mono,
                    mean_q_dipole, std_q_dipole, mean_abs_q_dipole,
                    mean_q_quadrupole, std_q_quadrupole, mean_abs_q_quadrupole,
                    mean_e_short, mean_e_long, mean_e_total,
                    mean_abs_e_short, mean_abs_e_long, mean_abs_e_total,
                    f_long_ratio, e_long_ratio
                ])
            phase_last_diag[phase_name] = {
                "f_long_over_f_total": f_long_ratio,
                "e_long_over_e_total": e_long_ratio,
                "mean_q_mono": mean_q_mono,
                "std_q_mono": std_q_mono,
                "std_q_dipole": std_q_dipole,
                "std_q_quadrupole": std_q_quadrupole,
            }

if args.sr_freeze_experiment:
    b = phase_last_diag.get("baseline_joint")
    f = phase_last_diag.get("sr_frozen_lr_focus")
    if b is not None and f is not None:
        eps = 1e-12
        print("[SRFreezeExp] ===== Summary =====")
        print(
            f"[SRFreezeExp] F_long/F_total: baseline={b['f_long_over_f_total']:.6e}, "
            f"frozen={f['f_long_over_f_total']:.6e}, "
            f"ratio={(f['f_long_over_f_total'] / max(b['f_long_over_f_total'], eps)):.3f}x"
        )
        print(
            f"[SRFreezeExp] E_long/E_total: baseline={b['e_long_over_e_total']:.6e}, "
            f"frozen={f['e_long_over_e_total']:.6e}"
        )
        print(
            f"[SRFreezeExp] q0(mean/std): baseline={b['mean_q_mono']:.6e}/{b['std_q_mono']:.6e}, "
            f"frozen={f['mean_q_mono']:.6e}/{f['std_q_mono']:.6e}"
        )
        print(
            f"[SRFreezeExp] std(q_dip): baseline={b['std_q_dipole']:.6e}, "
            f"frozen={f['std_q_dipole']:.6e}; "
            f"std(q_quad): baseline={b['std_q_quadrupole']:.6e}, "
            f"frozen={f['std_q_quadrupole']:.6e}"
        )

print(f"Saved diagnostics to: {diag_csv_path}")

# Hard guarantee: ensure both files exist in save_folder.
best_model_path = os.path.join(save_folder, "best_model.pth")
checkpoint_file_path = os.path.join(save_folder, "checkpoint.pt")
if not os.path.exists(best_model_path):
    task.save_model(best_model_path)
if not os.path.exists(checkpoint_file_path):
    task.checkpoint(checkpoint_file_path)

task.save_model(os.path.join(save_folder, 'cumulene-model-final.pth'))

print("Evaluating on cumulene test set")
test_loss = task.validate(test_loader)
print(f"test_loss: {test_loss:.6f}")

# print(f"Fourth train loop:")
# energy_loss = cace.tasks.GetLoss(
#     target_name='energy',
#     predict_name='CACE_energy',
#     loss_fn=torch.nn.MSELoss(),
#     loss_weight=0.001
# )

# task.update_loss([energy_loss, force_loss])
# task.fit(train_loader, valid_loader, epochs=50*boost, screen_nan=False, val_stride=10)

# task.save_model('electrolyte-model-4.pth')

print(f"Finished")


trainable_params = sum(p.numel() for p in cace_nnp.parameters() if p.requires_grad)
print(f"Number of trainable parameters: {trainable_params}")



