#!/usr/bin/env python
# coding: utf-8
"""
fit-electrolyte：

- 使用 CACE-SOG-Qeq（CACE 表示 + ChargeEq + SOG 核）在水+电解质体系 electrolyte.xyz 上训练；
- 不创建/不使用 SOGPotential，仅用 ChargeEq 的长程能量（key = SOG_potential）；
- 在模型 to(device) 后用 Ji 等 2026 提出的 BSA 初始化 SOG 参数；
- 数据使用 cace-lr-fit/fit-electrolyte/electrolyte.xyz（extxyz + forces + q 作为原子电荷）。
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
# 为了与旧脚本 fit-cace-SOG-old.py 的网络结构保持一致：
# - cutoff 从 5.29 调整为 4.5
# - CACE 表示中 n_atom_basis 改为 4，n_radial_basis 改为 12，num_message_passing 改为 1
cutoff = 4.5
Fourier_node = 18  # SOG 核的分量数（ChargeEq 内部 SOG 参数个数）

save_folder = os.path.join(
    os.path.dirname(__file__), "loss_data", "N_" + str(Fourier_node) + "_BSA_"
)
os.makedirs(save_folder, exist_ok=True)
now = datetime.datetime.now()
time_name = now.strftime("%Y%m%d_%H%M%S")

# electrolyte.xyz 存放在 cace-lr-fit/fit-electrolyte 目录下
train_path = "/work/home/acrb3qk4vo/SOG-Qeq/SOG-Net/CACE-SOG-Qeq/fit-electrolyte-small3/electrolyte_200.xyz"
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
    atomic_energies=None,
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
    num_message_passing=1,  # 与 fit-cace-SOG-old.py 中 MP=1 一致
    type_message_passing=["Bchi"],
    args_message_passing={"Bchi": {"shared_channels": False, "shared_l": False}},
    device=device,
    timeit=False,
    forward_features=["atomic_numbers"],
)

cace_representation.to(device)
print(f"Representation: {cace_representation}")

# ----- 短程：SR 能量 -----
sr_energy = cace.modules.atomwise.Atomwise(
    n_layers=3,
    output_key="SR_energy",
    n_hidden=[32, 16],
    use_batchnorm=False,
    add_linear_nn=True,
)

# ----- 电负性 chi（Qeq 输入） -----
chi = cace.modules.Atomwise(
    n_layers=3,
    n_hidden=[24, 12],
    n_out=1,
    per_atom_output_key="chi",
    output_key="tot_chi",
    residual=False,
    add_linear_nn=True,
    # Keep chi signed (no squaring) so it can take positive/negative values.
    # Qeq uses `chi` linearly (with an extra minus sign inside ChargeEq).
    post_process=None,
    bias=False,
)

# ----- 让 chi 更“按元素分开”：加 per-element bias + 全局 scale（加速让 q_eq 变大） -----
# 说明：
# - Qeq 的解只对 chi 的相对差异敏感；当 chi 差异过小，q_eq 会被压到接近 0。
# - 这里引入可训练偏置 b_Z，并给一个偏激但可控的初始化，让一开始就有更大的 Δchi。
chi_bias = cace.modules.ElementwiseFeatureBias(
    elements=[1, 8, 9, 19],
    feature_key="chi",
    output_key="chi_biased",
    # Elementwise bias initialization: use Pauling electronegativity (H/O/F/K).
    # Note: module has `zero_mean_bias=True`, so the effective bias is mean-centered.
    init_bias={1: 2.20, 8: 3.44, 9: 3.98, 19: 0.82},
    use_scale=True,
    init_scale=5.0,
    zero_mean_bias=True,
)

class SystemChargeFromAtomicCharges(nn.Module):
    """
    将每结构总电荷写入 data['system_charge']：
    system_charge[g] = sum_{i in graph g} charge[i]
    若 batch 中无 charge，则默认设为 0（电中性）。
    """
    def __init__(self, charges_key: str = "charge", output_key: str = "system_charge"):
        super().__init__()
        self.charges_key = charges_key
        self.output_key = output_key
        self.model_outputs = [output_key]

    def forward(self, data: dict, **kwargs):
        if self.charges_key not in data or data[self.charges_key] is None:
            # fallback: 电中性
            if data.get("batch", None) is None:
                num_graphs = 1
            else:
                num_graphs = int(data["batch"].max().item()) + 1 if data["batch"].numel() > 0 else 1
            data[self.output_key] = torch.zeros((num_graphs,), device=data["positions"].device, dtype=data["positions"].dtype)
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

system_charge_from_q = SystemChargeFromAtomicCharges(charges_key="charge", output_key="system_charge")

# ----- Qeq：使用 SOG 核构造 A；长程能量输出 key 命名为 SOG_potential -----
charge_eq = cace.modules.ChargeEq(
    dl=1.5,
    sigma=1.0,
    elements=[1, 8, 9, 19],
    feature_key="chi_biased",
    output_key="q_eq",
    ewald_key="SOG_potential",
    system_charge=None,
    remove_self_interaction=True,
    aggregation_mode="sum",
    use_sog_kernel=True,
    sog_num_components=Fourier_node,
)

# ----- 总能量 = SR_energy + SOG_potential -----
e_add = cace.modules.FeatureAdd(
    feature_keys=["SR_energy", "SOG_potential"],
    output_key="CACE_energy",
)

forces = cace.modules.Forces(
    energy_key="CACE_energy",
    forces_key="CACE_forces",
    calc_stress=False,
)

print("building CACE NNP (ChargeEq long-range, renamed to SOG_potential)")
cace_nnp = NeuralNetworkPotential(
    input_modules=None,
    representation=cace_representation,
    output_modules=[sr_energy, chi, chi_bias, system_charge_from_q, charge_eq, e_add, forces],
)
cace_nnp.to(device)

# ----- BSA 初始化：与 Ji 等 2026 一致，用 1/r 的 BSA 覆盖 SOG 初值 -----
charge_eq.init_sog_from_bsa(r_cut=cutoff, b=2.0)
print("ChargeEq SOG params initialized from BSA (r_cut=%.2f, b=2)." % cutoff)

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

task.save_model("hydrocarbon-model.pth")
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
task.save_model("hydrocarbon-model-2.pth")
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
task.save_model("hydrocarbon-model-3.pth")

print("Fourth train loop:")
energy_loss = cace.tasks.GetLoss(
    target_name="energy",
    predict_name="CACE_energy",
    loss_fn=torch.nn.MSELoss(),
    loss_weight=1000,
)
task.update_loss([energy_loss, force_loss])
task.fit(train_loader, valid_loader, epochs=100, screen_nan=False, val_stride=10)
task.save_model("hydrocarbon-model-4.pth")

print("Finished")
trainable_params = sum(p.numel() for p in cace_nnp.parameters() if p.requires_grad)
print(f"Number of trainable parameters: {trainable_params}")
