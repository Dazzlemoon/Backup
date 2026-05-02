#!/usr/bin/env python
# coding: utf-8
import sys
import os
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
L2_ROT_CONSISTENCY_WEIGHT = 0.05
save_folder = "/data/home/public/qiuqizhi/SOG-Qeq/SOG-Net/CACE-LOREM/fit-cumulene/loss_data/CACE_LOREM_MP"+str(MP)+"_now"
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
print(f"LR branch weight (fixed): {pot2['weight']}")
print(f"L2 rotation consistency weight: {L2_ROT_CONSISTENCY_WEIGHT}")


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
    q_count = 0
    sum_abs_f_long = 0.0
    sum_abs_f_total = 0.0
    sum_abs_e_long = 0.0
    sum_abs_e_total = 0.0
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

            f_long = (lr_weight * pred_lr["SOG_forces"]).reshape(-1, 3).detach()
            f_total = pred_total["CACE_forces"].reshape(-1, 3).detach()
            sum_abs_f_long += torch.linalg.norm(f_long, dim=1).sum().item()
            sum_abs_f_total += torch.linalg.norm(f_total, dim=1).sum().item()

            e_long = (lr_weight * pred_lr["lr_energy"]).reshape(-1).detach()
            e_total = pred_total["CACE_energy"].reshape(-1).detach()
            sum_abs_e_long += torch.abs(e_long).sum().item()
            sum_abs_e_total += torch.abs(e_total).sum().item()

    mean_q = sum_q / max(q_count, 1)
    std_q = (max(sum_q2 / max(q_count, 1) - mean_q * mean_q, 0.0)) ** 0.5
    mean_abs_q = sum_abs_q / max(q_count, 1)

    f_long_ratio = sum_abs_f_long / max(sum_abs_f_total, eps)
    e_long_ratio = sum_abs_e_long / max(sum_abs_e_total, eps)
    return std_q, mean_abs_q, f_long_ratio, e_long_ratio


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

optimizer_args = {'lr': 5e-3, 'betas': (0.99, 0.999)}
scheduler_args = {'step_size': 20, 'gamma': 0.5}

task = TrainingTask(
    model=cace_nnp,
    losses=[energy_loss, force_loss],
    metrics=[e_metric, f_metric],
    device=device,
    optimizer_args=optimizer_args,
    scheduler_cls=torch.optim.lr_scheduler.StepLR,
    scheduler_args=scheduler_args,
    max_grad_norm=10,
    ema=False, #True,
    ema_start=10,
    warmup_steps=5,
    consistency_loss_fn=l2_rotation_consistency_loss,
    consistency_loss_weight=L2_ROT_CONSISTENCY_WEIGHT,
    save_folder = save_folder,
    time_name = time_name
)

# Total epochs = 500 with staged energy loss weight:
# 0-200: 0.1, 200-300: 1, 300-400: 10, 400-500: 1000
stage_plan = [
    (200, 0.1),
    (100, 1.0),
    (100, 10.0),
    (100, 1000.0),
]
diag_csv_path = os.path.join(save_folder, 'diagnostics_' + time_name + '.csv')
with open(diag_csv_path, 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["global_epoch", "stage", "stage_epoch", "lr_weight", "std_q", "mean_abs_q", "f_long_over_f_total", "e_long_over_e_total"])

global_epoch = 0
for stage_idx, (epochs_now, energy_weight_now) in enumerate(stage_plan, start=1):
    lr_weight_now = task.model.potential_keys[1]["weight"]
    print(
        f"Stage {stage_idx}: epochs={epochs_now}, energy_weight={energy_weight_now}, "
        f"force_weight=0.5, lr_weight={lr_weight_now}"
    )
    stage_energy_loss = cace.tasks.GetLoss(
        target_name='energy',
        predict_name='CACE_energy',
        loss_fn=torch.nn.MSELoss(),
        loss_weight=energy_weight_now
    )
    task.update_loss([stage_energy_loss, force_loss])
    for stage_epoch in range(1, epochs_now + 1):
        global_epoch += 1
        task.fit(
            train_loader,
            valid_loader,
            epochs=1,
            screen_nan=False,
            val_stride=1,
            bestmodel_path="best_model.pth",
            checkpoint_path="checkpoint.pt",
        )
        std_q, mean_abs_q, f_long_ratio, e_long_ratio = compute_long_range_diagnostics(
            task.model, valid_loader, device, lr_weight=lr_weight_now
        )
        print(
            f"[Diag] epoch={global_epoch} stage={stage_idx}.{stage_epoch} "
            f"lr_weight={lr_weight_now:.1f} "
            f"std(q)={std_q:.6f} mean(|q|)={mean_abs_q:.6f} "
            f"||F_long||/||F_total||={f_long_ratio:.6f} E_long/E_total={e_long_ratio:.6f}"
        )
        with open(diag_csv_path, 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([global_epoch, stage_idx, stage_epoch, lr_weight_now, std_q, mean_abs_q, f_long_ratio, e_long_ratio])

print(f"Saved diagnostics to: {diag_csv_path}")

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



