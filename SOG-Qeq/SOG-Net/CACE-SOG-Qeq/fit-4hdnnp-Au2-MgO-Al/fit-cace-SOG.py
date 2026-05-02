#!/usr/bin/env python
# coding: utf-8

"""
用 CACE-SOG-Qeq 替代 CACE-SOG 的 SOGPotential，在 Au2-MgO-Al 数据集上训练：
- CACE 表示与 CACE-SOG 文章完全一致；
- 短程：SR_energy（Atomwise）；
- 长程：ChargeEq + SOG 核（SOG_potential），总能量 CACE_energy = SR_energy + SOG_potential。
"""
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

import cace
from cace.representations import Cace
from cace.modules import PolynomialCutoff, BesselRBF
from cace.tools.scatter import scatter_sum
from cace.tools import torch_geometric

from cace.models.atomistic import NeuralNetworkPotential
from cace.tasks.train import TrainingTask
from cace.data.extxyz_charge import read_extxyz_with_charge
from cace.tasks.load_data import random_train_valid_split

torch.set_default_dtype(torch.float32)

# ==== 保存与超参数 ====
Fourier_node = 18  # SOG 核分量数，沿用 NaCl Qeq 的设置
save_folder = "../fit-4hdnnp-Au2-MgO-Al/loss_data/SOG_Qeq_ini_c5"
now = datetime.datetime.now()
time_name = now.strftime("%Y%m%d_%H%M%S")

cace.tools.setup_logger(level="INFO")
cutoff = 5.0

# ==== 数据集读取（extxyz + charge，仿 NaCl Qeq）====
print("reading data (extxyz + charge via cace.data.extxyz_charge)")
DATA_DIR = os.path.join(os.path.dirname(__file__))
train_path = os.path.join(DATA_DIR, "Au-MgO-Al.xyz")

atomic_energies = {
    8: -18599.43617104475,
    12: -8721.75974245582,
    13: -9877.676428588728,
    79: -688.8680063349827,
}

z_map_au = {"O": 8, "Mg": 12, "Al": 13, "Au": 79}

all_data = read_extxyz_with_charge(
    path=train_path,
    cutoff=cutoff,
    atomic_energies=atomic_energies,
    z_map=z_map_au,
)
print(f"total structures in xyz: {len(all_data)}")

train_data, valid_data = random_train_valid_split(
    all_data, valid_fraction=0.1, seed=1
)

batch_size = 4

train_loader = torch_geometric.DataLoader(
    dataset=train_data,
    batch_size=batch_size,
    shuffle=True,
    drop_last=True,
)
valid_loader = torch_geometric.DataLoader(
    dataset=valid_data,
    batch_size=4,
    shuffle=False,
    drop_last=False,
)

print(f"#train structures: {len(train_data)}, #valid: {len(valid_data)}")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"device: {device}")


# ==== CACE 表示：完全沿用 CACE-SOG Au2-MgO-Al ====
print("building CACE representation")
radial_basis = BesselRBF(cutoff=cutoff, n_rbf=6, trainable=True)
cutoff_fn = PolynomialCutoff(cutoff=cutoff)

cace_representation = Cace(
    zs=[8, 12, 13, 79],
    n_atom_basis=4,
    embed_receiver_nodes=True,
    cutoff=cutoff,
    cutoff_fn=cutoff_fn,
    radial_basis=radial_basis,
    n_radial_basis=12,
    max_l=3,
    max_nu=3,
    num_message_passing=0,
    type_message_passing=["Bchi"],
    args_message_passing={"Bchi": {"shared_channels": False, "shared_l": False}},
    device=device,
    timeit=False,
)
cace_representation.to(device)
print(f"Representation: {cace_representation}")


# ==== 短程 SR 能量 ====
sr_energy = cace.modules.atomwise.Atomwise(
    n_layers=3,
    output_key="SR_energy",
    n_hidden=[32, 16],
    use_batchnorm=False,
    add_linear_nn=True,
)


# ==== Qeq: chi + system_charge + ChargeEq(SOG 核) ====
chi = cace.modules.Atomwise(
    n_layers=3,
    n_hidden=[24, 12],
    n_out=1,
    per_atom_output_key="chi",
    output_key="tot_chi",
    residual=False,
    add_linear_nn=True,
    post_process=torch.square,
    bias=False,
)


class SystemChargeFromAtomicCharges(nn.Module):
    """
    将每结构总电荷写入 data['system_charge']：
    - 若数据集中没有显式原子电荷（本 Au2-MgO-Al 即如此），假定每结构总电荷为 0。
    """

    def __init__(self, charges_key: str = "charge", output_key: str = "system_charge"):
        super().__init__()
        self.charges_key = charges_key
        self.output_key = output_key
        self.model_outputs = [output_key]

    def forward(self, data: dict, **kwargs):
        if self.charges_key not in data or data[self.charges_key] is None:
            if data.get("batch", None) is None:
                num_graphs = 1
            else:
                num_graphs = (
                    int(data["batch"].max().item()) + 1
                    if data["batch"].numel() > 0
                    else 1
                )
            data[self.output_key] = torch.zeros(
                (num_graphs,),
                device=data["positions"].device,
                dtype=data["positions"].dtype,
            )
            return data

        q = data[self.charges_key]
        if q.dim() > 1:
            q = q.view(-1)
        if data.get("batch", None) is None:
            system_q = q.sum().view(1)
        else:
            system_q = scatter_sum(q, data["batch"], dim=0)
        data[self.output_key] = system_q
        return data


system_charge_from_q = SystemChargeFromAtomicCharges(
    charges_key="charge", output_key="system_charge"
)

charge_eq = cace.modules.ChargeEq(
    dl=1.5,
    sigma=1.0,
    elements=[8, 12, 13, 79],
    feature_key="chi",
    output_key="q_eq",
    ewald_key="SOG_potential",
    system_charge=None,
    remove_self_interaction=True,
    aggregation_mode="sum",
    use_sog_kernel=True,
    sog_num_components=Fourier_node,
)


# ==== 总能量与力 ====
e_add = cace.modules.FeatureAdd(
    feature_keys=["SR_energy", "SOG_potential"],
    output_key="CACE_energy",
)

forces = cace.modules.Forces(
    energy_key="CACE_energy",
    forces_key="CACE_forces",
    calc_stress=False,
)


print("building CACE NNP (SR + Qeq-SOG)")
cace_nnp = NeuralNetworkPotential(
    input_modules=None,
    representation=cace_representation,
    output_modules=[sr_energy, chi, system_charge_from_q, charge_eq, e_add, forces],
)
cace_nnp.to(device)


# ==== 训练配置：沿用 CACE-SOG Au2-MgO-Al 的多阶段策略 ====
from cace.tools import Metrics

print("First train loop:")
energy_loss = cace.tasks.GetLoss(
    target_name="energy",
    predict_name="CACE_energy",
    loss_fn=torch.nn.MSELoss(),
    loss_weight=0.0,  # 第一阶段只用力
)

force_loss = cace.tasks.GetLoss(
    target_name="forces",
    predict_name="CACE_forces",
    loss_fn=torch.nn.MSELoss(),
    loss_weight=1.0,
)

e_metric = Metrics(
    target_name="energy",
    predict_name="CACE_energy",
    name="e/atom",
    per_atom=True,
)

f_metric = Metrics(
    target_name="forces",
    predict_name="CACE_forces",
    name="f",
)

print("creating training task")

optimizer_args = {"lr": 1e-3, "betas": (0.99, 0.999)}
scheduler_args = {"step_size": 20, "gamma": 0.9}

for i in range(10):
    task = TrainingTask(
        model=cace_nnp,
        losses=[energy_loss, force_loss],
        metrics=[e_metric, f_metric],
        device=device,
        optimizer_args=optimizer_args,
        scheduler_cls=torch.optim.lr_scheduler.StepLR,
        scheduler_args=scheduler_args,
        max_grad_norm=10,
        ema=False,
        ema_start=10,
        warmup_steps=5,
        save_folder=save_folder,
        time_name=time_name,
    )

    print("training")
    task.fit(train_loader, valid_loader, epochs=400, screen_nan=False)

task.save_model(save_folder + "Au-MgO-Al-Qeq-model.pth")
cace_nnp.to(device)

print("Second train loop:")
optimizer_args = {"lr": 1e-3, "betas": (0.99, 0.999)}
scheduler_args = {"step_size": 20, "gamma": 0.95}

energy_loss2 = cace.tasks.GetLoss(
    target_name="energy",
    predict_name="CACE_energy",
    loss_fn=torch.nn.MSELoss(),
    loss_weight=0.001,
)

for i in range(10):
    task = TrainingTask(
        model=cace_nnp,
        losses=[energy_loss2, force_loss],
        metrics=[e_metric, f_metric],
        device=device,
        optimizer_args=optimizer_args,
        scheduler_cls=torch.optim.lr_scheduler.StepLR,
        scheduler_args=scheduler_args,
        max_grad_norm=10,
        ema=False,
        ema_start=10,
        warmup_steps=5,
        save_folder=save_folder,
        time_name=time_name,
    )

    print("training")
    task.fit(train_loader, valid_loader, epochs=200, screen_nan=False)

print("Third train loop:")
optimizer_args = {"lr": 3e-4, "betas": (0.99, 0.999)}
scheduler_args = {"step_size": 20, "gamma": 0.99}

energy_loss3 = cace.tasks.GetLoss(
    target_name="energy",
    predict_name="CACE_energy",
    loss_fn=torch.nn.MSELoss(),
    loss_weight=0.0001,
)

for i in range(5):
    task = TrainingTask(
        model=cace_nnp,
        losses=[energy_loss3, force_loss],
        metrics=[e_metric, f_metric],
        device=device,
        optimizer_args=optimizer_args,
        scheduler_cls=torch.optim.lr_scheduler.StepLR,
        scheduler_args=scheduler_args,
        max_grad_norm=10,
        ema=False,
        ema_start=10,
        warmup_steps=5,
        save_folder=save_folder,
        time_name=time_name,
    )

    print("training")
    task.fit(train_loader, valid_loader, epochs=100, screen_nan=False)

print("Fourth train loop:")
energy_loss4 = cace.tasks.GetLoss(
    target_name="energy",
    predict_name="CACE_energy",
    loss_fn=torch.nn.MSELoss(),
    loss_weight=0.001,
)

optimizer_args = {"lr": 1e-4, "betas": (0.99, 0.999)}
scheduler_args = {"step_size": 20, "gamma": 0.995}

for i in range(20):
    task = TrainingTask(
        model=cace_nnp,
        losses=[energy_loss4, force_loss],
        metrics=[e_metric, f_metric],
        device=device,
        optimizer_args=optimizer_args,
        scheduler_cls=torch.optim.lr_scheduler.StepLR,
        scheduler_args=scheduler_args,
        max_grad_norm=10,
        ema=False,
        ema_start=10,
        warmup_steps=5,
        save_folder=save_folder,
        time_name=time_name,
    )

    print("training")
    task.fit(train_loader, valid_loader, epochs=40, screen_nan=False)

print("Finished")

trainable_params = sum(p.numel() for p in cace_nnp.parameters() if p.requires_grad)
print(f"Number of trainable parameters: {trainable_params}")

