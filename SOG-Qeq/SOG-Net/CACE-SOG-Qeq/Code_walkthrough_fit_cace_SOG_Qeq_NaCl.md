## 从代码角度解读 NaCl 训练脚本（`fit-4hdnnp-NaCl/fit-cace-SOG.py`）与 Qeq-SOG 模型

本文件从 **代码实现** 的角度，逐段解释 `fit-4hdnnp-NaCl/fit-cace-SOG.py` 和相关模块（尤其是 Qeq-SOG 的 `ChargeEq`），帮助你快速理解“脚本到底做了什么”。公式仍用 `$...$` / `$$...$$`。

建议结合 `README_fit_cace_SOG_Qeq_NaCl.md` 一起阅读：README 以“物理/训练配置”为主，本文件偏“源码级结构”。

---

## 1. 文件开头：路径与基础 import

```python
import sys
import os
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

os.environ["PYTHONWARNINGS"] = "ignore"
```

- 将 `CACE-SOG-Qeq` 根目录加入 `sys.path`，确保 `import cace` 等指向当前仓库版本，而不是全局安装的旧包。
- 关闭 Python 的 warning 输出，避免训练日志被过多 warning 淹没。

接着是常规依赖：

```python
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
from cace.data.extxyz_charge import get_dataset_from_extxyz_with_charge
```

**关键点**：

- `Cace`：主表示层（图网络/张量基表示）。
- `scatter_sum`：用来按 `batch` 维度对 per-atom 电荷求和，得到每结构总电荷。
- `torch_geometric`：用 `torch_geometric.DataLoader` 承载 `AtomicData`。
- `get_dataset_from_extxyz_with_charge`：自定义的 extxyz + 电荷读取函数（见 `cace/data/extxyz_charge.py`）。

---

## 2. 数据集读取：extxyz + per-atom charge

脚本里数据集读取部分是：

```python
cace.tools.setup_logger(level="INFO")
cutoff = 5.29
Fourier_node = 18  # SOG 核的分量数（ChargeEq 内部 SOG 参数个数）

save_folder = os.path.join(os.path.dirname(__file__), "loss_data", "N_" + str(Fourier_node) + "_")
os.makedirs(save_folder, exist_ok=True)
now = datetime.datetime.now()
time_name = now.strftime("%Y%m%d_%H%M%S")

print("reading data (extxyz + charge via cace.data.extxyz_charge)")
collection = get_dataset_from_extxyz_with_charge(
    train_path=os.path.join(os.path.dirname(__file__), "NaCl.xyz"),
    cutoff=cutoff,
    valid_fraction=0.1,
    seed=1,
    atomic_energies={11: -4417.07609365649, 17: -12516.880649933015},
)
batch_size = 5
```

这里用的不是 ASE，而是你在 `cace/data/extxyz_charge.py` 里写的手工解析函数：

- 对每一帧 `NaCl.xyz`，按
  `species, pos(3), forces(3), charge(1)` 的列顺序解析。
- 把每个原子的 `charge` 放到 `AtomicData.additional_info["charge"]`。
- 使用 `get_neighborhood(...)` 构造边列表 `edge_index` 与 PBC 偏移 `shifts`、`unit_shifts`。
- 构建 `AtomicData(...)`，形成 `collection.train` / `collection.valid`。

随后用 **PyG 的 DataLoader** 包装：

```python
train_dataset = collection.train
valid_dataset = collection.valid

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
```

这一步的结果是：后续 `TrainingTask` 迭代时，拿到的 `batch` 是一个 `AtomicData` 合并成的 PyG `Batch`，包含：

- `positions, cell, edge_index, shifts, unit_shifts, atomic_numbers, batch, ptr, ...`
- 以及附加的 `charge` 字段（在 `batch["charge"]` 或 `batch.to_dict()["charge"]` 里）。

---

## 3. 表示层 `Cace` 与短程能量 / 电负性网络

表示层构建：

```python
radial_basis = BesselRBF(cutoff=cutoff, n_rbf=6, trainable=True)
cutoff_fn = PolynomialCutoff(cutoff=cutoff)

cace_representation = Cace(
    zs=[11, 17],
    n_atom_basis=2,
    embed_receiver_nodes=True,
    cutoff=cutoff,
    cutoff_fn=cutoff_fn,
    radial_basis=radial_basis,
    n_radial_basis=8,
    max_l=3,
    max_nu=3,
    num_message_passing=0,
    type_message_passing=["Bchi"],
    args_message_passing={"Bchi": {"shared_channels": False, "shared_l": False}},
    device=device,
    timeit=False,
    forward_features=["atomic_numbers"],
)
```

**要点**：

- 表示层接收 `positions, cell, edge_index, shifts, atomic_numbers` 等，输出 per-atom 特征 `node_feats`。
- `forward_features=["atomic_numbers"]` 确保 `atomic_numbers` 在经过表示层后仍保留在 `data` 中（后续 `ChargeEq` 需要）。

短程能量网络：

```python
sr_energy = cace.modules.atomwise.Atomwise(
    n_layers=3,
    output_key="SR_energy",
    n_hidden=[32, 16],
    use_batchnorm=False,
    add_linear_nn=True,
)
```

电负性网络：

```python
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
```

- `sr_energy`：对 CACE 表示做 3 层 MLP，输出 per-atom 或 per-structure 的 `SR_energy`。
- `chi`：输出 per-atom 电负性 `chi_i`，用 `square` 保证非负。

---

## 4. 每结构总电荷：`SystemChargeFromAtomicCharges`

脚本中有一个小模块专门用于把 per-atom `charge` 汇总成每结构总电荷：

```python
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
            data[self.output_key] = torch.zeros(
                (num_graphs,), device=data["positions"].device, dtype=data["positions"].dtype
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

system_charge_from_q = SystemChargeFromAtomicCharges(charges_key="charge", output_key="system_charge")
```

这段做的事情就是：

$$
Q_{\text{tot}}^{(g)} = \sum_{i \in \text{graph } g} q_i^{\text{(data)}}
$$

并写入 `data["system_charge"]`，供后续 `ChargeEq` 使用。

---

## 5. Qeq 模块 `ChargeEq`：SOG 核 + 线性系统求解

构造 Qeq 模块：

```python
charge_eq = cace.modules.ChargeEq(
    dl=1.5,
    sigma=1.0,
    elements=[11, 17],
    feature_key="chi",
    output_key="q_eq",
    ewald_key="SOG_potential",
    # 使用 data['system_charge']（由上面的 SystemChargeFromAtomicCharges 写入）
    system_charge=None,
    remove_self_interaction=True,
    aggregation_mode="sum",
    use_sog_kernel=True,
    sog_num_components=Fourier_node,
)
```

在 `cace/modules/charge_eq.py` 中，关键逻辑是：

1. 从 `data` 中取：
   - 坐标 `r = data["positions"]`
   - 晶胞 `box = data["cell"]`
   - 电负性 `chi = data[self.feature_key]`（即 `chi` 网络输出）
   - 原子序数 `Z`（用于索引元素硬度参数）
   - 总电荷向量 `system_charge`（来自 `data["system_charge"]` 或构造函数常数）。

2. 构造核矩阵 $A$：
   - 若 `use_sog_kernel=True`：

     ```python
     A_now = self._build_A_sog(r_now, box_now)
     ```

     `_build_A_sog` 中：

     - 先构造 pairwise 距离 $r_{ij}$；
     - 再用 SOG 核：

       $$
       A_{ij} = \sum_{\ell=1}^{L} w_\ell \exp(-\alpha_\ell r_{ij}^2),
       $$

       其中 $\alpha_\ell = \exp(\text{sog\_log\_alpha}_\ell)$、$w_\ell = \text{sog\_weights}_\ell$ 或来自共享的 SOGPotential。

   - 否则，从 `EwaldPotential` 获取一个简单的 $1/r$ 核矩阵。

3. 组装增广线性系统并用 `torch.linalg.solve` 求解：

```python
A_plus_J = A_mat + torch.diag(J.to(dtype))
coeffs = torch.ones((N_atoms + 1, N_atoms + 1), device=device, dtype=dtype)
coeffs[:N_atoms, :N_atoms] = A_plus_J
coeffs[N_atoms, N_atoms] = 0.0
Q_tot = system_Q / self.normalization_factor
chi_vector = torch.cat(
    [-chi.view(-1), torch.tensor([Q_tot], device=device, dtype=dtype)]
)
sol = torch.linalg.solve(coeffs, chi_vector)
q_eq = sol[:N_atoms]
lambda_eq = sol[N_atoms]
```

并用

```python
ewald_energy = 0.5 * q_eq.unsqueeze(1).T @ A_now @ q_eq.unsqueeze(1)
```

得到长程能量，最后写入 `data[self.ewald_key]`（在此脚本中为 `SOG_potential`）。

---

## 6. 总能量与力：`FeatureAdd` + `Forces`

总能量聚合：

```python
e_add = cace.modules.FeatureAdd(
    feature_keys=["SR_energy", "SOG_potential"],
    output_key="CACE_energy",
)
```

力模块：

```python
forces = cace.modules.Forces(
    energy_key="CACE_energy",
    forces_key="CACE_forces",
    calc_stress=False,
)
```

- `FeatureAdd` 简单地做：
  $$
  E_{\text{tot}} = E_{\text{SR}} + E_{\text{long}}.
  $$
- `Forces` 在 `forward` 里调用 `get_outputs(...)`，本质上是：
  - 用 `torch.autograd.grad` 计算

    $$
    \mathbf{F}_i = -\\frac{\\partial E_{\text{tot}}}{\\partial \\mathbf{r}_i}.
    $$

  - 并写入 `data["CACE_forces"]`。

---

## 7. 模型装配：`NeuralNetworkPotential`

最后组装模型：

```python
print("building CACE NNP (ChargeEq long-range, renamed to SOG_potential)")
cace_nnp = NeuralNetworkPotential(
    input_modules=None,
    representation=cace_representation,
    output_modules=[sr_energy, chi, system_charge_from_q, charge_eq, e_add, forces],
)
cace_nnp.to(device)
```

执行顺序为：

1. `Preprocess`（内部默认 input 模块）。
2. `Cace`：构建表示。
3. `sr_energy`：短程能量。
4. `chi`：电负性。
5. `system_charge_from_q`：按结构汇总总电荷。
6. `charge_eq`：解 Qeq，得到 `q_eq` 与 `SOG_potential`。
7. `e_add`：`SR_energy + SOG_potential → CACE_energy`。
8. `forces`：从 `CACE_energy` 求导得到 `CACE_forces`。

`NeuralNetworkPotential` 在 `forward` 结束后，会从数据字典中抽取所有模块声明的 `model_outputs` 汇总成最终返回的 `dict`。

---

## 8. 训练循环：`TrainingTask`

训练部分核心结构（以第一阶段为例）：

```python
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

e_metric = Metrics(... 'e/atom' ...)
f_metric = Metrics(... 'f' ...)

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
```

`TrainingTask.fit` 内部会：

- 迭代 `train_loader`：
  - 将 `batch` 移到 `device`；
  - 调用 `model(batch, training=True)`；
  - 计算 loss 并 `backward()`；
  - `optimizer.step()` + scheduler 更新。
- 定期在 `valid_loader` 上评估能量/力误差，并记录 log/metric。

后续阶段（2/3/4）只是改变 `energy_loss` 的权重和训练轮数，复用相同的数据流和模型结构。

---

## 9. 小结

从代码层面看，`fit-cace-SOG.py` 做了以下几件事：

1. **数据侧**：用 `get_dataset_from_extxyz_with_charge` 手工解析 NaCl extxyz，保证 per-atom `charge` 与 `forces` 被读入，并通过 `SystemChargeFromAtomicCharges` 转成 per-graph 总电荷约束。
2. **模型侧**：构建 `Cace` 表示 + SR `Atomwise` + 电负性网络 + SOG 核 `ChargeEq` + 能量聚合 + 力模块，形成一个物理约束良好的 Qeq-SOG 势。
3. **训练侧**：通过 `TrainingTask` 分阶段调整能量/力权重，训练约 500 个 epoch，对齐 NaCl 的能量与力，同时学习 SOG 核参数与元素硬度。

这份脚本既是一个可直接跑的 NaCl Qeq-SOG 训练示例，也是一块模板：可以在此基础上替换数据集/元素/核形式（例如更复杂的 SOG+FFT/NUFFT matvec + PCG 求解），构建其他体系的 Qeq-SOG 模型。

