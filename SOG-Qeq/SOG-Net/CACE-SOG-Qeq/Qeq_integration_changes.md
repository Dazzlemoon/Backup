## CACE-SOG-Qeq 中集成 Qeq (`ChargeEq`) 的修改说明

本文件总结了在 `CACE-SOG-Qeq` 中，为了将 **CACE-SOG** 与 **Qeq/ChargeEq 方法** 结合而做的代码改动位置，便于你对照阅读与后续扩展。

---

### 1. 新增模块：`cace.modules.EwaldPotential`（简化版）

- **文件位置**
  - `cace/modules/ewald.py`

- **主要内容**
  - 新增类 `EwaldPotential(nn.Module)`，用于给 Qeq 模块提供核矩阵 \(A\)：
    - 接口：`compute_potential_triclinic(r, q, cell, compute_field=False)`
    - 输入：
      - `r`: 形状 `(N, 3)` 的原子坐标
      - `q`: 形状 `(N,)` 或 `(N, n_q)` 的电荷
      - `cell`: 形状 `(3, 3)` 的晶胞矩阵（当前实现中仅保留接口，不显式用于周期展开）
    - 输出：
      - `pot`: 对每一列电荷向量 `q[:, k]`，计算得到的标量库伦能量 $ E_k = \tfrac{1}{2} q_k^\top A q_k $
      - `A_mat`: 形状 `(N, N)` 的核矩阵 \(A\)，定义为简单的 \( A_{ij} = 1 / |r_i - r_j| \)
  - 当前实现为 **简化的 1/r 库伦核**，没有执行完整的 Ewald 周期求和，但保留了与原 CACE 接口兼容的调用方式，便于 `ChargeEq` 使用：
    - 在 `ChargeEq` 中通过
      ```python
      _, A_now = self.ep.compute_potential_triclinic(
          r_now,
          torch.eye(r_now.size(0), device=r_now.device),
          box_now,
          compute_field=self.compute_field,
      )
      ```
      获得用于 Qeq 线性方程的 \(A\) 矩阵。

---

### 2. 新增模块：`cace.modules.ChargeEq`（Qeq 实现）

- **文件位置**
  - `cace/modules/charge_eq.py`

- **主要内容**
  - 新增类 `ChargeEq(nn.Module)`，实现 **Qeq/charge equilibration** 过程，整体结构参考了你在 `SOG-Qeq/cace/cace/modules/charge_eq.py` 中的原始 CACE 实现：
    - 构造函数参数（与原版保持一致）：
      - `elements`: 元素原子序列表，如 `[1, 8]`
      - `feature_key`: 读取上游网络输出的电负性特征键名（默认 `'chi'`）
      - `output_key`: 写回平衡电荷的键名（默认 `'q_eq'`）
      - `ewald_key`: 写回长程能量的键名（默认 `'ewald_potential'`）
      - `system_charge`: 体系总电荷（默认 0.0）
      - 其余如 `dl`, `sigma`, `aggregation_mode`, `system_charge_key` 等与原 CACE 用法兼容
    - 内部结构：
      - 持有一个 `EwaldPotential` 实例：
        ```python
        self.ep = EwaldPotential(
            dl=dl,
            sigma=sigma,
            remove_self_interaction=remove_self_interaction,
            aggregation_mode=aggregation_mode,
        )
        ```
      - 为每种元素维护一个可训练的硬度参数向量 `self.J_raw`，在 forward 中使用平方 `J_elem = J_raw**2` 保证正值。
      - 使用 `Z_index_map` 将每个原子序 `Z` 映射到对应的元素硬度 `J_i`。
  - `forward` 逻辑（按 batch 中每个构型循环）：
    1. 从 `data` 中读取：
       - `positions`, `cell`, `atomic_numbers`, 上游网络输出的 `chi` 等。
    2. 对当前构型：
       - 调用 `EwaldPotential.compute_potential_triclinic` 得到核矩阵 `A_now`（见上节）。
       - 调用 `_compute_q_eq(A_now, chi_now, J_i_now, system_charge_now)`：
         - 构建并求解 Qeq 线性方程组，得到平衡电荷 `q_eq` 与拉格朗日乘子 `lambda_eq`。
       - 计算 Ewald 能量：
         
         $$ E_{\text{Ewald}} = \frac{1}{2} q_{\text{eq}}^\top A_{\text{now}} q_{\text{eq}} $$
         
    3. 聚合所有构型：
       - 将所有构型的 `q_eq` 拼接后写回 `data[self.output_key]`（如 `data['q_eq']`）
       - 将所有构型的 Ewald 能量聚合后写回 `data[self.ewald_key]`（如 `data['ewald_potential']`）
  - `_compute_q_eq`：
    - 与原 CACE 版本一致，构建扩展矩阵并解线性方程：
      $$
      \begin{pmatrix}
      A + J & 1 \\
      1^\top & 0
      \end{pmatrix}
      \begin{pmatrix}
      q \\
      \lambda
      \end{pmatrix}
      =
      \begin{pmatrix}
      -\chi \\
      Q_{\text{tot}}
      \end{pmatrix}
      $$
    - 其中 `A` 来自前一步 `EwaldPotential` 返回的 `A_mat`，`J` 为按原子展开后的硬度对角阵。

---

### 3. 模块导出更新：`cace.modules.__init__.py`

- **文件位置**
  - `cace/modules/__init__.py`

- **修改内容**
  - 在原有的导出基础上，新增：
    ```python
    from .sog import *

    from .ewald import *

    from .charge_eq import *
    ```
  - 这样，在脚本中可以直接使用：
    - `cace.modules.EwaldPotential`
    - `cace.modules.ChargeEq`
    - 同时保持原有的 `SOGPotential` 等模块接口不变。

---

### 4. Qeq 示例脚本路径修正：water 系统 `fit_cace_new.py`

- **文件位置**
  - `water/water_perspective/cacelr-Qeq-r-4.5-nl-1-nu-3/fit_cace_new.py`

- **修改目的**
  - 原始脚本假定在集群环境下已经安装了某个绝对路径下的 `cace` 包：
    ```python
    import sys
    sys.path.append('/global/home/users/dongjinkim/software/cace/')
    ```
  - 在当前仓库中，这样的路径并不存在，会导致无法导入本地的 `cace` 模块。

- **具体修改**
  - 将文件开头替换为基于当前脚本位置的相对路径推断：
    ```python
    #!/usr/bin/env python
    # coding: utf-8

    import os
    import sys

    # 使用当前脚本位置，自动加入本仓库的 CACE-SOG-Qeq 路径，避免依赖集群上的绝对路径
    ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
    if ROOT_DIR not in sys.path:
        sys.path.insert(0, ROOT_DIR)
    ```
  - 这样，`import cace` 将优先从当前仓库根目录下的 `cace/` 包中导入：
    - 使用的是你在 `CACE-SOG-Qeq/cace` 中带有 `SOGPotential`、`ChargeEq` 等扩展的本地 CACE 版本；
    - 不再依赖远端 `/global/home/...` 这样的集群特定路径。

---

### 5. 如何在 CACE-SOG 中“接 Qeq”

有了以上改动之后，你可以在任何基于 `CACE-SOG-Qeq/cace` 的脚本中，像在 CACE‑LES 示例中那样集成 Qeq，仅需在 **短程能量 + SOG 长程** 管线中插入 `chi` 和 `ChargeEq`：

- **典型结构（示意）**
  1. CACE 表征：
     - `cace_representation = Cace(...)`
  2. 短程能量网络：
     - `sr_energy = cace.modules.Atomwise(..., output_key='SR_energy', ...)`
  3. 电负性网络：
     - `chi = cace.modules.Atomwise(..., per_atom_output_key='chi', output_key='tot_chi', post_process=torch.square, ...)`
  4. Qeq 模块：
     - `Qeq = cace.modules.ChargeEq(elements=cace_representation.zs, feature_key='chi', output_key='q_eq', ...)`
  5. 长程能量（两种典型接法）：
     - **CACE‑LES 风格**：直接使用 `Qeq` 给出的 `ewald_potential`：
       - `FeatureAdd(['SR_energy', 'ewald_potential'] -> 'CACE_energy')`
     - **SOG‑Net 风格**：把 `q_eq` 当作 SOG 的输入电荷（需要在脚本层面将 `SOGPotential.feature_key` 设为 `'q_eq'`），由 SOG 卷积给出长程能量，然后再与 `SR_energy` 相加。

你后续若希望针对具体体系（NaCl、水、peptides 等）写出完整的 **CACE-SOG-Qeq** 训练脚本，只需在对应 `fit-cace-SOG.py` 一类脚本中：

1. 引入上面新增的 `ChargeEq`；
2. 在 CACE‑SOG 的 pipeline 中插入 `chi` → `ChargeEq`；
3. 决定是直接使用 `ewald_potential` 作为长程能量，还是基于 `q_eq` 再接入 `SOGPotential`。

如果你希望，我也可以针对某一个具体体系（例如 `pure-SR-Nacl` 或 `fit-dipeptides`）再帮你写一份完整的 **CACE-SOG-Qeq 训练脚本** 草案，供你直接运行与调整。

