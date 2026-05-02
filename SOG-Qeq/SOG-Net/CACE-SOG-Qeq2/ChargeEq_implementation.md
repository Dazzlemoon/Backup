## CACE 中 Qeq 模块 `ChargeEq` 的实现解析

本文基于 `SOG-Qeq/cace/cace/modules/charge_eq.py`，解释 CACE 里 Qeq 模块 `ChargeEq` 的具体实现方式、数学形式以及与 Ewald/LES 的联系，帮助你在阅读和修改时有一份“源码对照表”。

源码位置（相对当前项目）：

```1:3:/dssg/home/acct-matxzl/matxzl/QiuQizhi/SOG-Qeq/cace/cace/modules/charge_eq.py
from typing import Dict, List
import torch
import torch.nn as nn
```

---

### 1. 类的构造函数：参数与成员变量

```8:23:/dssg/home/acct-matxzl/matxzl/QiuQizhi/SOG-Qeq/cace/cace/modules/charge_eq.py
class ChargeEq(nn.Module):
    def __init__(self,
                 dl: float = 1.5,
                 sigma: float = 1.0,
                 elements: List[int] = None,
                 feature_key: str = 'chi',
                 output_key: str = 'q_eq',
                 ewald_key: str = 'ewald_potential',
                 system_charge: float = 0.0,
                 remove_self_interaction: bool = True,
                 aggregation_mode: str = 'sum',
                 compute_field: bool = True,
                 norm_factor: float = (1./90.0474)**0.5, 
                 scaling_factor: float = 1.0,
                 system_charge_key: str = 'system_charge',  # Key for system charge in data
                 ):
        super().__init__()
```

- **关键参数含义**
  - **`dl`**：Ewald 相关长度尺度，用于 `EwaldPotential`；控制实空间 / 倒空间和截断等，影响库伦矩阵 \(A\) 的数值。
  - **`sigma`**：高斯展宽参数，同样进入 Ewald 计算（电荷平滑）。
  - **`elements`**：元素原子序数组，如 `[1, 8]`，用于建立元素到电荷硬度参数 \(J\) 的索引。
  - **`feature_key`**：从数据字典中读取电负性的键（通常是前面网络输出的 `chi`）。
  - **`output_key`**：把求解得到的平衡电荷 \(q_{\text{eq}}\) 写回数据字典时使用的键名。
  - **`ewald_key`**：把 Ewald/库伦能量写回数据字典时使用的键名（一般为 `ewald_potential`）。
  - **`system_charge`**：体系总电荷，若数据中未提供 `system_charge_key`，则采用这里的默认值（如 0）。
  - **`remove_self_interaction`**：是否在 Ewald 计算中去掉自相互作用项。
  - **`aggregation_mode`**：`"sum"` 表示对每个 batch 中的不同构型的 Ewald 能量做求和等聚合方式。
  - **`compute_field`**：控制 Ewald 计算时是否顺带求电场（有利于后续力或响应性质）。
  - **`norm_factor`**：将总电荷归一化到与 `ewald.py` 中能量单位一致的因子。
  - **`system_charge_key`**：数据字典中存储体系总电荷的键名。

构造函数内部还完成了若干重要初始化：

```26:52:/dssg/home/acct-matxzl/matxzl/QiuQizhi/SOG-Qeq/cace/cace/modules/charge_eq.py
        self.feature_key = feature_key
        self.output_key = output_key
        self.ewald_key = ewald_key
        self.model_outputs = [output_key, ewald_key]
        self.normalization_factor = norm_factor  # 1/2\epsilon_0
        self.scaling_factor = scaling_factor
        self.compute_field = compute_field
        self.system_charge = system_charge
        self.aggregation_mode = aggregation_mode
        self.system_charge_key = system_charge_key

        self.ep = EwaldPotential(
            dl=dl,
            sigma=sigma,
            remove_self_interaction=remove_self_interaction,
            aggregation_mode=aggregation_mode,
        )
        self.elements = elements
        Z_max = max(elements)
        Z_index_map = torch.full((Z_max + 1,), -1)
        for i, z in enumerate(elements):
            Z_index_map[z] = i
        self.register_buffer('Z_index_map', Z_index_map)

        init_J = torch.ones(len(elements)) # initialize J to 1 for all elements
        self.J_raw = nn.Parameter(data=init_J, requires_grad=True)
```

- **与 LES / Ewald 的连接**
  - `self.ep = EwaldPotential(...)`：这里真正把 Qeq 模块和 Ewald（也就是 LES 的长程静电核）连起来。
  - `Z_index_map`：建立 `原子序 Z → 元素索引` 的查表张量，后面根据 `atomic_numbers` 给每个原子分配对应的 \(J\) 值。
  - `self.J_raw`：
    - 对每种元素维护一个可训练参数 \(J_{\text{raw}}\)。
    - 实际使用时会平方 `J_elem = J_raw^2` 以确保 \(J > 0\)，对应 Qeq 里元素的“硬度”参数。

> 总结：构造函数负责把 Qeq 中的**超参数（dl, sigma 等）**与**可学习参数（每种元素的硬度 J）**组织起来，同时准备好与 `EwaldPotential` 的接口。

---

### 2. `forward`：从 `chi` 到 \(q_{\text{eq}}\) 和 Ewald 能量

`forward` 的输入是一个特征字典 `data: Dict[str, torch.Tensor]`，通常由 CACE 的 `NeuralNetworkPotential` 管线逐模块传递。

```54:69:/dssg/home/acct-matxzl/matxzl/QiuQizhi/SOG-Qeq/cace/cace/modules/charge_eq.py
    def forward(self, data: Dict[str, torch.Tensor], **kwargs):

        if data["batch"] is None:
            n_nodes = data['positions'].shape[0]
            batch_now = torch.zeros(n_nodes, dtype=torch.int64, device=data['positions'].device)
        else:
            batch_now = data["batch"]

        box = data['cell']
        r = data['positions']
        chi = data[self.feature_key]
        Z = data['atomic_numbers']
        element_types = torch.unique(Z)
        assert len(element_types) == len(self.elements), \
            f"Number of unique elements {len(element_types)} != expected number {len(self.elements)}."
        if chi.dim() == 1:
            chi = chi.unsqueeze(1)
```

- **数据读取**
  - `positions`：原子坐标 \(r_i\)。
  - `cell`：晶胞矩阵，用于 Ewald 处理周期性边界。
  - `chi`：上游网络（例如 `Atomwise` 模块）输出的**每个原子的电负性特征**。
  - `atomic_numbers`：原子序 \(Z_i\)，用来决定每个原子的硬度 \(J_i\)。
  - `batch`：指定每个原子属于哪一个构型（支持一个 batch 多个结构）。

接下来根据元素类型选择对应的 \(J\)：

```72:80:/dssg/home/acct-matxzl/matxzl/QiuQizhi/SOG-Qeq/cace/cace/modules/charge_eq.py
        J_raw = self.J_raw
        J_elem = torch.square(J_raw) # positive
        idx = self.Z_index_map[Z]
        J_i = J_elem[idx]

        n, d = r.shape
        assert d == 3, 'r dimension error'
        assert n == chi.size(0), 'chi dimension error'
```

- **逻辑**
  - `self.J_raw` 是按元素存储的，可训练张量。
  - 通过 `Z_index_map[Z]` 将每个原子映射到相应元素索引，再从 `J_elem` 取出该元素的硬度。
  - 最终得到 `J_i`：与原子数相同的一维向量，每个分量是该原子的 \(J_i > 0\)。

接下来处理体系总电荷：

```81:90:/dssg/home/acct-matxzl/matxzl/QiuQizhi/SOG-Qeq/cace/cace/modules/charge_eq.py
        unique_batches = torch.unique(batch_now)  # Get unique batch indices

        if (self.system_charge_key not in data or data[self.system_charge_key] is None
            ) and self.system_charge is not None:
            system_charge = self.system_charge
            system_charge = torch.full((len(unique_batches),), 
                                       system_charge, device=data['positions'].device)
        else:
            system_charge = data[self.system_charge_key]
```

- 若数据中未提供 `system_charge`，则使用构造函数里的标量（默认 0），并在 batch 维上复制。

#### 2.1 按构型循环：为每个结构求解 Qeq

```92:112:/dssg/home/acct-matxzl/matxzl/QiuQizhi/SOG-Qeq/cace/cace/modules/charge_eq.py
        results = []
        ewald_results = []

        for i in unique_batches:
            mask = batch_now == i  # Create a mask for the i-th configuration
            r_now, chi_now, box_now = r[mask], chi[mask], box[i]
            system_charge_now = system_charge[i]
            J_i_now = J_i[mask]
            A_now = self._compute_A_matrix(r_now, box_now)
            q_eq, lambda_eq = self._compute_q_eq(A_now, chi_now, J_i_now, system_charge_now)
            results.append(q_eq)
            ewald_energy = 0.5 * q_eq.unsqueeze(1).T @ A_now @ q_eq.unsqueeze(1)
            ewald_results.append(ewald_energy)
        all_q_eq = torch.cat(results, dim=0)
        if all_q_eq.dim() == 1:
            all_q_eq = all_q_eq.unsqueeze(1)
        data[self.output_key] = all_q_eq
        all_ewald = torch.stack(ewald_results, dim=0).sum(axis=1) if self.aggregation_mode == "sum" else torch.stack(ewald_results, dim=0)
        if all_ewald.dim() != 1:
            all_ewald = all_ewald.squeeze(-1)
        data[self.ewald_key] = all_ewald

        return data
```

- 对每个 batch 中的构型 \(k\)：
  1. 基于 `mask` 取出该构型的坐标 `r_now`、电负性 `chi_now`、硬度 `J_i_now` 和晶胞 `box_now`。
  2. 调用 `_compute_A_matrix` 得到该构型的库伦矩阵 \(A\)。
  3. 调用 `_compute_q_eq` 在总电荷约束下求解平衡电荷 \(q_{\text{eq}}\) 和拉格朗日乘子 \(\lambda_{\text{eq}}\)。
  4. 用二次型 \(E_{\text{Ewald}} = \frac{1}{2} q^\top A q\) 计算该构型的 Ewald 能量。
  5. 把所有构型的 \(q_{\text{eq}}\) 拼接成 `all_q_eq`，写入 `data[self.output_key]`。
  6. 把所有构型的 Ewald 能量聚合后写入 `data[self.ewald_key]`。

> 这一步就是 **“从神经网络给出的 `chi`，通过 Qeq 模块求得电荷和长程 Ewald 能量”** 的核心入口。

---

### 3. `_compute_A_matrix`：构建库伦 / Ewald 矩阵

```118:123:/dssg/home/acct-matxzl/matxzl/QiuQizhi/SOG-Qeq/cace/cace/modules/charge_eq.py
    def _compute_A_matrix(self, r, cell):
        N_atoms = len(r)
        q_eye = torch.eye(N_atoms, device=r.device)
        _, A_mat = self.ep.compute_potential_triclinic(r, q_eye, cell, compute_field=self.compute_field)

        return A_mat
```

- **输入**
  - `r`：当前构型的原子坐标。
  - `cell`：当前构型的晶胞（一般是 3×3 矩阵）。

- **步骤**
  - 构造单位矩阵 `q_eye`，可以理解为“在每个原子上放一个单位电荷”的集合。
  - 调用 `EwaldPotential.compute_potential_triclinic`：
    - 该函数假定每个原子轮流带单位电荷，计算出在其它原子位置处的势能，从而得到势/能量与电荷向量之间的线性映射矩阵。
    - 输出中第二个量 `A_mat` 就相当于 Qeq 方程中的核矩阵 \(A\)：
      \[
      E_{\text{Ewald}} = \frac{1}{2} q^\top A q
      \]
  - 返回 \(A_{\text{mat}}\)，维度是 \(N_{\text{atoms}} \times N_{\text{atoms}}\)。

> 简单说：`_compute_A_matrix` 把 LES/Ewald 的长程库伦物理封装成了一个对称矩阵 \(A\)，后面 Qeq 只需在这个核上做线性代数。

---

### 4. `_compute_q_eq`：Qeq 线性方程的构建与求解

```125:139:/dssg/home/acct-matxzl/matxzl/QiuQizhi/SOG-Qeq/cace/cace/modules/charge_eq.py
    def _compute_q_eq(self, A_mat, chi, J, system_Q):
        device, dtype = A_mat.device, A_mat.dtype
        N_atoms = A_mat.size(0)
        A_plus_J = A_mat + torch.diag(J.to(dtype))
        coeffs = torch.ones((N_atoms+1, N_atoms+1), device=device, dtype=dtype)
        coeffs[:N_atoms, :N_atoms] = A_plus_J
        coeffs[N_atoms,  N_atoms]  = 0.0
        Q_tot = system_Q / self.normalization_factor # normalized to be consistent with ewald.py
        chi_vector = torch.cat([-chi.view(-1),
                                torch.tensor([Q_tot], device=device, dtype=dtype)])
        chi_vector = chi.unsqueeze(1) if chi.dim() == 1 else chi_vector
        sol = torch.linalg.solve(coeffs, chi_vector)
        q_eq = sol[:N_atoms]
        lambda_eq = sol[N_atoms]
        return q_eq, lambda_eq
```

#### 4.1 物理/数学背景（简化）

经典 Qeq 模型的能量形式可以写作：

\[
E(q) = \frac{1}{2} q^\top A q + \frac{1}{2} \sum_i J_i q_i^2 + \sum_i \chi_i q_i
\]

再加上总电荷约束：

\[
\sum_i q_i = Q_{\text{tot}}
\]

对 \(q_i\) 引入拉格朗日乘子 \(\lambda\) 做极值：

\[
\frac{\partial}{\partial q_i}\left(E(q) - \lambda\left(\sum_j q_j - Q_{\text{tot}}\right)\right) = 0
\]

得到线性方程组：

\[
(A + J) q + \chi - \lambda \mathbf{1} = 0, \quad \sum_i q_i = Q_{\text{tot}}
\]

可以写成矩阵形式：

\[
\begin{pmatrix}
A + J & -\mathbf{1} \\
\mathbf{1}^\top & 0
\end{pmatrix}
\begin{pmatrix}
q \\
\lambda
\end{pmatrix}
=
\begin{pmatrix}
-\chi \\
Q_{\text{tot}}
\end{pmatrix}.
\]

#### 4.2 对应到源码

- `A_plus_J = A_mat + torch.diag(J.to(dtype))`
  - 这里 \(A_{\text{mat}} + \text{diag}(J)\) 就是上式的 \(A + J\)。

- `coeffs` 的构造：
  - `coeffs` 初始化为全 1 矩阵，接着前 \(N\) 行列替换为 `A_plus_J`。
  - `coeffs[N_atoms, N_atoms] = 0.0`：最后一个元素给出拉格朗日乘子方程的 0。
  - 在很多 Qeq 实现中，最后一行/列对应的是约束 \(\sum_i q_i = Q_{\text{tot}}\) 与 \(-\lambda\) 项；本实现里具体对号可通过配合 `ewald.py` 的实现进一步确认，但整体结构就是上面矩阵的离散版本。

- `Q_tot = system_Q / self.normalization_factor`
  - 把给定的总电荷 \(Q_{\text{sys}}\) 按照 `normalization_factor` 进行单位归一化，以与 Ewald 矩阵 \(A\) 的单位保持一致。

- `chi_vector = torch.cat([-chi.view(-1), torch.tensor([Q_tot], ...)])`
  - 构造右端向量：
    - 前 \(N\) 项是 \(-\chi_i\)
    - 最后一项是 \(Q_{\text{tot}}\)
  - 这对应上面矩阵方程右侧的 \([-\chi; Q_{\text{tot}}]\)。

- `sol = torch.linalg.solve(coeffs, chi_vector)`
  - 解线性方程：
    \[
    \text{coeffs} \times \begin{pmatrix} q \\ \lambda \end{pmatrix}
    = \text{chi\_vector}
    \]

- `q_eq = sol[:N_atoms]`, `lambda_eq = sol[N_atoms]`
  - 提取解向量前 \(N\) 分量作为平衡电荷 \(q_{\text{eq}}\)，最后一分量是拉格朗日乘子 \(\lambda_{\text{eq}}\)。

> 因此，`_compute_q_eq` 就是典型 Qeq 线性方程在 PyTorch 中的矩阵求解实现，兼容自动求导，从而允许对 `chi` 与 \(J\) 反向传播梯度。

---

### 5. 与 CACE‑LES / 训练脚本的接口关系

结合你之前的 `fit_cace_new.py`，数据流可以总结为：

- **上游模块**
  - CACE 表征 `Cace` 提供局域原子特征。
  - `Atomwise` 网络预测每个原子的 `chi`（电负性特征），存入 `data['chi']`。

- **`ChargeEq` 模块**
  1. 在 `forward` 里读取：
     - `positions`, `cell`, `atomic_numbers`, `chi`, `batch` 等。
  2. 通过 EwaldPotential 构造库伦核矩阵 \(A\)。
  3. 构建带约束的线性方程，解出 \(q_{\text{eq}}\) 和 \(\lambda\)。
  4. 将：
     - `q_eq` 写入 `data[self.output_key]`（例如 `data['q_eq']`）
     - Ewald 能量 \(E_{\text{Ewald}} = \frac{1}{2} q^\top A q\) 写入 `data[self.ewald_key]`（例如 `data['ewald_potential']`）。

- **下游模块**
  - `FeatureAdd` 将 `SR_energy` 和 `ewald_potential` 相加得到 `CACE_energy`。
  - `Forces` 根据 `CACE_energy` 对坐标求导得到 `CACE_forces`。
  - 训练时，用 `energy` 和 `forces` 的损失约束整个链条，使 `chi` 网络、元素硬度 \(J\) 和表征一起学习出合适的电荷分布和长程势能。

---

### 6. 建议的阅读 / 实验路径

若你想深入理解并可能修改 `ChargeEq`，可以参考：

- **第一步：对照本文 + 源码**
  - 一边看 `charge_eq.py`，一边对照本文中的矩阵方程，确认每一行代码对应哪个物理/数学量。
  - 特别看 `_compute_q_eq` 部分，把矩阵方程自己手写一遍，会非常有帮助。

- **第二步：查阅 `EwaldPotential` 源码**
  - 在 `cace/modules/ewald.py` 中查看 `EwaldPotential.compute_potential_triclinic` 的实现，理解：
    - 它如何从 `(r, cell)` 生成矩阵 \(A\)。
    - `aggregation_mode` 和 `remove_self_interaction` 的具体含义。

- **第三步：做小规模数值实验**
  - 写一个 toy 脚本，只包含几个原子（比如 2~3 个电荷），固定坐标和 `chi`，调用 `ChargeEq` 看看：
    - 得到的 `q_eq` 是否满足总电荷约束。
    - 改变 `chi` 或 `J` 是否会朝物理合理的方向变化（如更大 `chi` 对应更容易带正/负电）。

- **第四步：结合 LES / Qeq 理论**
  - 对照相关论文里 Qeq 的公式和 LES 的潜在 Ewald 框架，验证这里的实现与理论一致：
    - 能量形式
    - 线性方程结构
    - 约束和单位归一化方式

掌握了这些之后，你就可以比较有把握地：

- 调整或扩展 Qeq（例如增加不同元素的参数化方式、加入额外的电荷约束等）；
- 或者把 `ChargeEq` 迁移 / 嵌入到你自己的 SOG‑Net / 其它 MLIP 框架里。 

