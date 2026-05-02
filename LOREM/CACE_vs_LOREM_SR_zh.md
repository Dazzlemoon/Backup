# CACE 与 LOREM 短程网络：架构异同与代码实现对照

本文对比 **本仓库中的 LOREM 短程部分**（`LOREM/lorem/lorem.py`，JAX/Flax + e3x）与 **CACE-SOG-Ji 中的 CACE 表示层**（`SOG-Qeq/SOG-Net/CACE-SOG-Ji/cace/representations/cace_representation.py`，PyTorch）。二者都属于 **截断图上的等变/不变原子势表示**，但 **基函数、消息形式与代码栈均不同**；**不能**把 LOREM 的短程实现等同于 CACE。

---

## 1. 一句话结论

| 维度 | LOREM 短程 | CACE（`Cace`）短程 |
|------|------------|-------------------|
| **深度学习框架** | JAX + Flax | PyTorch |
| **几何基（角向）** | **球谐**（`e3x.so3.spherical_harmonics`，按 \(l,m\) 不可约分解） | **笛卡尔单项式** \(x^{l_x}y^{l_y}z^{l_z}\)（`AngularComponent`，按 `lxlylz` 列表） |
| **径向** | Bernstein 等可配置径向基 + `RadialCoefficients` 里 MLP 与径向收缩 | 可插拔 `radial_basis`（如 Bessel 等）× `cutoff_fn` |
| **边/节点通道** | 种类嵌入 → 边系数；通道维为 `num_features` 等 | one-hot → **发送/接收节点嵌入**；`EdgeEncoder` 得到 **`n_atom_basis²` 维**边通道 |
| **多体相关阶** | 不显式用 CACE 的 “A→B 对称化阶数”；等变部分用 e3x **张量层**迭代 | 显式 **`Symmetrizer`**：由 `max_l` 与 **`max_nu`** 构造对称化 **B 基**（多体积） |
| **消息传递** | `segment_sum` + `Update`；可选 **`e3x.nn.MessagePass`**（`equivariant_message_passing`） | **`MessageAr`**、**`MessageBchi`**、**`NodeMemory`** 等组合，每步再 **`symmetrizer`** |
| **节点特征形状** | 标量向量 + e3x 风格的球谐张量（`nodes_spherical`） | **4 维张量** `[节点, radial, angular, channel]`，角向维为 `lxlylz` 长度 |

---

## 2. LOREM 短程：实现要点（论文与代码一致处）

**文件**：`lorem/lorem.py`（类 `Lorem`、`Initial`、`RadialCoefficients`、`Update`）

- **构图**：相对位移 \(\mathbf{R}_{ij}\)，截断内为边。
- **Initial**：单位化 \(\hat{\mathbf{r}}_{ij}\)，径向基 × 余弦截断；角向为 **球谐**；原子种类为 `nn.Embed`。
- **RadialCoefficients**：对 **拼接的节点标量特征**（首轮为种类嵌入）用 MLP 生成系数，与径向基 **收缩** 得到边标量；再与球谐组合、聚合为节点，经 `TensorDense` 等到 **等变节点特征**。
- **标量更新**：用 **球谐范数**（power spectrum 风格权重 `l_factors`）把等变信息压回标量，`Update` 为 **MLP + LayerNorm** 的残差块。
- **多步 MP**：`for _ in range(num_message_passing)`；可选打开 **`equivariant_message_passing`** 使用 **`e3x.nn.MessagePass`** 更新球谐特征。
- **能量**：对 `nodes_scalar` 的 **MLP 残差** 输出每原子能量（每层可累加）。

**与 CACE 的本质差异**：角向是 **标准球谐不可约基** + e3x 张量运算；**没有** CACE 的 `lxlylz` 笛卡尔积基，也 **没有** `Symmetrizer(max_nu, ...)` 这种显式多体对称积构造。

---

## 3. CACE 短程：实现要点

**文件**：`SOG-Qeq/SOG-Net/CACE-SOG-Ji/cace/representations/cace_representation.py`（类 `Cace`）

- **节点编码**：`NodeEncoder` one-hot → `NodeEmbedding`（发送端；接收端可共享或独立）。
- **边编码**：默认 `EdgeEncoder(directed=True)`，与嵌入维配合，得到 **`n_edge_channels = n_atom_basis²`** 的边通道（见 `cace_representation.py` 中 `self.n_edge_channels = n_atom_basis**2`）。
- **边几何**：
  - `get_edge_vectors_and_lengths` 得单位边向量与长度；
  - `radial_basis(edge_lengths) * cutoff_fn`；
  - **`AngularComponent(max_l)`**：对单位向量分量做 **递归笛卡尔单项式**（注释与实现均表明为 *edge basis* 的角向部分，而非球谐列表），输出形状 `[n_edges, angular_dim]`。
- **边属性张量**：`elementwise_multiply_3tensors(radial×cutoff, angular, encoded_edges)` → **`[n_edges, radial_dim, angular_dim, embedding_dim]`**。
- **初值节点特征**：对 **接收原子** `scatter_sum` 得 `node_feat_A`，再 **`SharedRadialLinearTransform`** 混合径向维。
- **对称化 B 基**：`node_feat_B = self.symmetrizer(node_attr=node_feat_A)`。`Symmetrizer`（见 `cace/modules/symmetrize_basis.py`）按 **`max_nu`** 对 A 基角向指标做 **乘积与组合**，得到多体阶意义下的不变/对称特征（与 ACE/CACE 类文献中的 A/B 基一致，**非** LOREM 中的 e3x 流程）。

**消息传递**（`forward` 中循环）：

- **`NodeMemory`**（可选）：对 `node_feat_A` 的残差记忆项；
- **`MessageBchi`**：用 **当前 B 特征** 经 MLP 得权重，与 **边张量 `edge_attri`** 相乘构造消息（见 `cace/modules/interaction.py` 中 `MessageBchi`）；
- **`MessageAr`**：从发送端复制 `node_feat_A`，并按距离、截断对 **分角向组** 做指数型径向衰减调制（见 `MessageAr.forward`）；
- 聚合后 **`radial_transform`**，再 **`symmetrizer`** 进入下一轮 `node_feat_B`；
- 使用 **`mp_norm_factor ∝ 1/√avg_num_neighbors`** 做归一化。

**与 LOREM 的本质差异**：角向为 **笛卡尔张量积基 + 显式 `max_nu` 对称化**；消息机制为 **Ar / Bχ / Memory** 三条线与 CACE 自定义图运算，**不是** e3x 的 `MessagePass` 同款。

---

## 4. 相同点（高层）

- 均在 **截断半径** 内构图，用 **相对位移** 保证平移不变；总能量常分解为 **原子贡献之和**（具体读出层在各自 head / 组合势模块中）。
- 都使用 **径向基 × 截断函数** 处理距离；都用 **多层消息传递** 扩大感受野（步数分别由 `num_message_passing` 等配置）。
- **都与 MACE 不等价**：MACE 是另一套体展开与对称收缩设计；论文中 LOREM 与 MACE/CACE-LES 等为 **并列基线**，非同一实现。

---

## 5. 长程部分（避免混淆）

- **LOREM**：短程结束后，若 `lr=True`，从节点特征预测 **标量 + 低 \(l\) 等变电荷**，用 **Ewald**（周期）或 **全边 \(1/r\)**（非周期），再张量积回灌标量特征（见 `lorem/lorem.py` 中 `if self.lr:`）。
- **CACE-SOG-Ji**：长程常在 **`cace/modules/ewald.py`** 等中用 **SOGPotential / Ewald** 等与 **电荷分支** 结合，与 LOREM 的 **等变电荷 + jax-pme** 是 **不同代码路径与物理参数化**；比较“短程架构”时一般 **不把 SOG 与 LOREM 长程混为一谈**。

---

## 6. 代码路径速查

| 内容 | LOREM | CACE（CACE-SOG-Ji） |
|------|--------|----------------------|
| 主模块 | `lorem/lorem.py` | `cace/representations/cace_representation.py` |
| 角向基 | `e3x.so3.spherical_harmonics`（在 `Initial`） | `cace/modules/angular.py` 中 `AngularComponent` |
| 对称化 / 多体阶 | （无同名 `Symmetrizer`） | `cace/modules/symmetrize_basis.py` |
| 消息层 | `lorem/lorem.py`：`segment_sum` + 可选 `e3x.nn.MessagePass` | `cace/modules/interaction.py`：`MessageAr`、`MessageBchi`、`NodeMemory` |

---

## 7. 小结

- **不相同**：LOREM 短程是 **e3x 球谐 + RadialCoefficients +（可选）e3x 等变消息传递**；CACE 短程是 **笛卡尔 lxlylz 边基 + Symmetrizer(max_nu) + MessageAr/Bχ/Memory** 的 **PyTorch 专用流水线**。
- **“像”的地方**：都是 **局部图神经网络**，都强调几何与通道混合；但 **基函数与消息定义不同**，因此论文与实验上应视为 **两种架构**，仅在任务上可比性能，而非代码级同一短程网络。

---

*若你后续升级任一侧的 `forward` 签名或默认超参，请同步更新本文中的类名与文件路径。*
