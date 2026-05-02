#!/usr/bin/env python
# coding: utf-8
"""
fit-electrolyte：

- 使用 **CACE-SOG（论文 Ji et al. JCP 2026 Results-B, KF 水溶液）** 的方式训练：
  - SR：CACE 表示 + Atomwise 预测短程能量 `CACE_energy`
  - LR：从 SR 特征预测 latent `q`，通过 `SOGPotential` 得到长程能量 `SOG_potential`
  - 总能量与力由 CombinePotential 合并两个子势：SR + LR

- 数据：extxyz（含 energy/forces/固定电荷 q 列），这里用 `read_extxyz_with_charge` 直接解析。
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
from cace.modules import CosineCutoff, MollifierCutoff, PolynomialCutoff
from cace.modules import BesselRBF, GaussianRBF, GaussianRBFCentered
from cace.tools.scatter import scatter_sum
from cace.tools import torch_geometric

from cace.models.atomistic import NeuralNetworkPotential
from cace.tasks.train import TrainingTask
from cace.data.extxyz_charge import read_extxyz_with_charge
from cace.tasks.load_data import random_train_valid_split

torch.set_default_dtype(torch.float32)

cace.tools.setup_logger(level="INFO")
cutoff = 4.5
# 论文 Results-B（KF）使用 M=12 个 Gaussians（SOG multiplier 的分量数）
bandwidth_num = 12
N_dl = 1  # Fourier modes spacing（实现里用于确定 reciprocal-grid 尺度）

save_folder = os.path.join(
    os.path.dirname(__file__), "loss_data", f"SOG_rcut_{cutoff}_M_{bandwidth_num}_"
)
os.makedirs(save_folder, exist_ok=True)
now = datetime.datetime.now()
time_name = now.strftime("%Y%m%d_%H%M%S")

# electrolyte.xyz 存放在 cace-lr-fit/fit-electrolyte 目录下
train_path = "/work/home/acrb3qk4vo/SOG-Qeq/SOG-Net/CACE-SOG-Qeq/fit-electrolyte-small/electrolyte_200.xyz"
print(f"reading data from {train_path} (extxyz + charge via cace.data.extxyz_charge)")

# 电解质体系：包含 H, O, K, F 等元素；使用 read_extxyz_with_charge + 自定义 z_map
z_map_electrolyte = {
    "H": 1,
    "O": 8,
    "F": 9,
    "K": 19,
}
all_data = read_extxyz_with_charge(
    path=train_path,
    cutoff=cutoff,
    # 与论文/旧脚本一致：减去按元素平均原子能（用于能量基准对齐与更快收敛）
    atomic_energies={
        1: -0.1749365806299343,
        8: -0.08746829031496617,
        9: -4.620975436064299,
        19: -4.620975436064285,
    },
    z_map=z_map_electrolyte,
)
print(f"total electrolyte structures in xyz: {len(all_data)}")

train_dataset, valid_dataset = random_train_valid_split(
    all_data, valid_fraction=0.1, seed=1
)
batch_size = 5

train_loader = torch_geometric.DataLoader(
    dataset=train_dataset,
    batch_size=batch_size,
    shuffle=True,
    drop_last=True,
)

valid_loader = torch_geometric.DataLoader(
    dataset=valid_dataset,
    batch_size=5,
    shuffle=False,
    drop_last=False,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"device: {device}")

print("building CACE representation")
radial_basis = BesselRBF(cutoff=cutoff, n_rbf=6, trainable=True)
cutoff_fn = PolynomialCutoff(cutoff=cutoff)

# 论文 Results-B：Nembedding=4，rcut=4.5Å，T=0 或 1（此脚本默认 1，可改成 0 对齐 MP0）
MP = 1
cace_representation = Cace(
    zs=[1, 8, 9, 19],  # H, O, F, K
    n_atom_basis=4,  # 与 fit-cace-SOG-old.py 保持一致
    embed_receiver_nodes=True,
    cutoff=cutoff,
    cutoff_fn=cutoff_fn,
    radial_basis=radial_basis,
    n_radial_basis=12,  # 与 fit-cace-SOG-old.py 保持一致
    max_l=3,
    max_nu=3,
    num_message_passing=MP,
    type_message_passing=["Bchi"],
    args_message_passing={"Bchi": {"shared_channels": False, "shared_l": False}},
    device=device,
    timeit=False,
    forward_features=["atomic_numbers"],
)

cace_representation.to(device)
print(f"Representation: {cace_representation}")

# -------------------------
# SR 子势：CACE + Atomwise(SR_energy) + Forces
# -------------------------
sr_atomwise = cace.modules.atomwise.Atomwise(
    n_layers=3,
    output_key="CACE_energy",
    n_hidden=[32, 16],
    use_batchnorm=False,
    add_linear_nn=True,
)
sr_forces = cace.modules.Forces(
    energy_key="CACE_energy",
    forces_key="CACE_forces",
    calc_stress=False,
)
print("building SR NNP (CACE + Atomwise)")
cace_nnp_sr = NeuralNetworkPotential(
    input_modules=None,
    representation=cace_representation,
    output_modules=[sr_atomwise, sr_forces],
).to(device)

# -------------------------
# LR 子势：latent q + SOGPotential + Forces
# -------------------------
q_latent = cace.modules.Atomwise(
    n_layers=3,
    n_hidden=[24, 12],
    n_out=1,
    per_atom_output_key="q",
    output_key="tot_q",
    residual=False,
    add_linear_nn=True,
    bias=False,
)
ep = cace.modules.SOGPotential(
    N_dl=N_dl,
    bandwidth_num=bandwidth_num,
    feature_key="q",
    output_key="SOG_potential",
    remove_self_interaction=False,
    aggregation_mode="sum",
    Periodic=True,
)
lr_forces = cace.modules.Forces(
    energy_key="SOG_potential",
    forces_key="SOG_forces",
    calc_stress=False,
)
print("building LR NNP (latent q + SOGPotential)")
cace_nnp_lr = NeuralNetworkPotential(
    input_modules=None,
    representation=cace_representation,
    output_modules=[q_latent, ep, lr_forces],
).to(device)

# -------------------------
# CombinePotential：总能量/力 = SR + LR
# -------------------------
pot_sr = {"CACE_energy": "CACE_energy", "CACE_forces": "CACE_forces"}
pot_lr = {"CACE_energy": "SOG_potential", "CACE_forces": "SOG_forces", "weight": 1.0}
print("building CombinePotential (SR + LR)")
cace_nnp = cace.models.CombinePotential([cace_nnp_sr, cace_nnp_lr], [pot_sr, pot_lr]).to(device)

print("First train loop:")
energy_loss = cace.tasks.GetLoss(
    target_name="energy",
    predict_name="CACE_energy",
    loss_fn=torch.nn.MSELoss(),
    loss_weight=0.1,
)

force_loss = cace.tasks.GetLoss(
    target_name="forces",
    predict_name="CACE_forces",
    loss_fn=torch.nn.MSELoss(),
    loss_weight=1000,
)

from cace.tools import Metrics

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
optimizer_args = {"lr": 5e-3, "betas": (0.99, 0.999)}
scheduler_args = {"step_size": 20, "gamma": 0.5}

for i in range(5):
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
    task.fit(train_loader, valid_loader, epochs=40, screen_nan=False, val_stride=10)

task.save_model("electrolyte-model-SOG.pth")
cace_nnp.to(device)

print("Second train loop:")
energy_loss = cace.tasks.GetLoss(
    target_name="energy",
    predict_name="CACE_energy",
    loss_fn=torch.nn.MSELoss(),
    loss_weight=1,
)
task.update_loss([energy_loss, force_loss])
print("training")
task.fit(train_loader, valid_loader, epochs=100, screen_nan=False, val_stride=10)
task.save_model("electrolyte-model-SOG-2.pth")
cace_nnp.to(device)

print("Third train loop:")
energy_loss = cace.tasks.GetLoss(
    target_name="energy",
    predict_name="CACE_energy",
    loss_fn=torch.nn.MSELoss(),
    loss_weight=10,
)
task.update_loss([energy_loss, force_loss])
task.fit(train_loader, valid_loader, epochs=100, screen_nan=False, val_stride=10)
task.save_model("electrolyte-model-SOG-3.pth")

print("Fourth train loop:")
energy_loss = cace.tasks.GetLoss(
    target_name="energy",
    predict_name="CACE_energy",
    loss_fn=torch.nn.MSELoss(),
    loss_weight=1000,
)
task.update_loss([energy_loss, force_loss])
task.fit(train_loader, valid_loader, epochs=100, screen_nan=False, val_stride=10)
task.save_model("electrolyte-model-SOG-4.pth")

print("Finished")
trainable_params = sum(p.numel() for p in cace_nnp.parameters() if p.requires_grad)
print(f"Number of trainable parameters: {trainable_params}")
