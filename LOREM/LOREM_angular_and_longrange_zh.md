# LOREM：角向信息如何进入网络，以及长程分支的数据流

本文档说明 **Learning Long-Range Representations with Equivariant Messages**（Rumiantsev 等，2026，arXiv:2507.19382）与实现 **`lorem/lorem.py`** 中 **$\mathrm{Lorem}$** 的对应关系：短程里角度从哪里来；**`lr=True`** 时长程势如何从 **电荷多通道** 算出来，再如何 **回灌** 到标量节点特征。数学公式：**行内** 用 `$...$`，**行间** 用 `$$...$$`。

更完整的整体架构见同目录 **`LOREM_ARCHITECTURE_zh.md`**；多通道库仑与严格多极的差别见 **`Multipole_vs_channel_SOG_zh.md`**。

---

## 1. 角向信息在网络结构中的位置

角度 **不是** 单独一项监督信号，而是通过 **边的单位方向** 与 **球谐基底** 显式进入前向计算。

### 1.1 边上：归一化位移 + 球谐展开

在 **`Initial`** 中，相对位移 $\mathbf{R}_{ij}$ 先被归一化为单位向量 $\hat{\mathbf{R}}_{ij}$，并保留标量距离 $r_{ij}$：

- **径向**：仅依赖 $r_{ij}$ 的径向基（如 Bernstein）× 截断函数；
- **角向**：对 $\hat{\mathbf{R}}_{ij}$ 调用 **`e3x.so3.spherical_harmonics`**，得到各 $l\le$ `max_degree` 的 $Y_{lm}$，即论文中的 **angular expansion**。

因此，**每条截断边**上的角向依赖都体现在 **`spherical_expansion`**（下文记为边上的 `spherical`）中。

### 1.2 可学习系数与球谐收缩 → 等变边/节点特征

边标量特征 `edges_scalar`（由种类嵌入、径向收缩等得到）经线性层产生按 $l$ 与特征通道的 **系数**，再与同一条边上的 **`spherical`** 做收缩，得到 **等变边特征** `edges_spherical`，再按目标节点索引 `i` 做 **`segment_sum`**，得到 **`nodes_spherical`**（球谐型等变节点特征），并经过 **`e3x.nn.TensorDense`**。

直观理解：网络学习的是「在 **固定的** $Y_{lm}(\hat{\mathbf{R}}_{ij})$ 基底上，各模式权重为多少」；**取向**已通过球谐进入表示，而不只是标量函数 $f(r_{ij})$。

### 1.3 等变信息进入标量：球谐范数

对 **`nodes_spherical`** 使用 **`spherical_norm_last_axis`**（按 $l$ 块求范数，类似 power spectrum），再乘 **`l_factors`** 后 **`Update`** 注入 **`nodes_scalar`**。这样标量通道中包含 **旋转不变、但与局部几何取向有关的量**。

### 1.4 可选：等变消息传递

若 **`equivariant_message_passing=True`**，每一轮仍使用 **同一条边上缓存的** `spherical`（角向基不随层改变），用更新后的边系数与 **`e3x.nn.MessagePass`**、**`e3x.nn.Tensor`** 更新 **`nodes_spherical`**，再经球谐范数回灌标量。

---

## 2. 长程分支（`lr=True`）中的数据流

长程部分在代码中位于 **`Lorem.__call__`** 末尾的 **`if self.lr:`** 块。核心思想与论文一致：**从节点特征预测多通道电荷（标量 + 低 $l$ 等变分量展平）→ Ewald（周期）或全边 $1/r$（非周期）求势 → 将势与 `nodes_spherical` 张量积混合后更新标量特征 → 再加一项原子能量**。

下表概括 **张量语义**（具体变量名以代码为准）。

| 步骤 | 作用 |
|------|------|
| 电荷头 | `nodes_scalar` → MLP → **`scalar_charges`**（每原子 1 维）；`nodes_spherical` → **`TensorDense(..., max_degree=max_degree_lr)`** → **`spherical_charges`**；**`charges = concat(...)`**，最后一维为 **多通道电荷**。 |
| 周期体系 | **`jax.vmap`** 对 **`charges`** 的最后一维：**每个通道**各调用一次 **`jaxpme.Ewald(...).potentials`**，输入 **`cell`、`positions`、短程边 `i,j`、`k_grid`、`smearing`** 等，得到 **`potentials`**。 |
| 非周期 | 使用 **`full_R_ij`、`full_i`、`full_j`**：先算 **`1/r`**（$r=0$ 处屏蔽），**`segment_sum(charges[full_j] * (1/r), full_i)`** 将邻居电荷经库仑权重聚到中心原子 → **`potentials`**。 |
| 拆分势 | **`scalar_potential`** 取第 0 通道；**`spherical_potential`** 取其余通道并 reshape。 |
| 回灌 | **`spherical_potential`** 经 **`Dense`** 后与 **`nodes_spherical`** 做 **`e3x.nn.Tensor`**（张量积）；再 **`spherical_norm`**，与 **`scalar_potential`** 拼接，**`Update(nodes_scalar)`**。 |
| 能量 | **`energy += MLP(nodes_scalar)`**。 |

**说明：**

- 长程 **相互作用核** 对每个电荷通道是 **同一类** Ewald 或 $1/r$（**不显式**写成静电学里偶极–偶极公式的 $\hat{\mathbf{r}}$ 多项式）。
- **几何与高阶信息**在长程中的体现主要来自：**(i)** 由 **`TensorDense(max_degree_lr)`** 从 **`nodes_spherical`** 读出的 **低 $l$ 电荷分量**；**(ii)** 回灌阶段 **`Tensor(spherical_potential, nodes_spherical)`** 与短程球谐特征对齐。

---

## 3. 力与可微性

**`predict`** 中对 **`batch.positions`**（周期）或 **`batch.full_edges`**（非周期）求梯度，使能量对坐标可微；长程贡献的力分别来自 Ewald 与全边 $1/r$ 项的导数（见 **`lorem.py`** 中 **`energy` / `predict`**）。

---

## 4. 代码入口

| 模块 | 文件与位置 |
|------|------------|
| 主模型 | `lorem/lorem.py`：`Lorem`、`Initial`、`Update`、`RadialCoefficients` |
| 角向 | `Initial`：`normalize` + `spherical_harmonics`；前向中 `edges_spherical`、`nodes_spherical` |
| 长程 | `Lorem.__call__`：`if self.lr:` 块（Ewald / `segment_sum` + 回灌） |

---

*若论文 PDF 与 arXiv 版本一致，可与文中 Fig.1 对照；实现细节以本仓库 `lorem/lorem.py` 为准。*
