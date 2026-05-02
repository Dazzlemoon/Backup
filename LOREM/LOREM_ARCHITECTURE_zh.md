# LOREM 架构说明：论文观点与代码实现对照

本文档基于仓库内论文 *Learning Long-Range Representations with Equivariant Messages*（Rumiantsev 等，2026，arXiv:2507.19382）与实现代码目录 `lorem/`，说明 **LOREM（文中写作 Lorem）** 的整体结构、各模块在代码中的位置，并澄清与 **MACE / CACE** 的关系。

---

## 1. 论文中的 LOREM 在解决什么问题

- 多数 MLIP 在 **截断半径内的图** 上做消息传递，难以刻画静电、色散、电子离域等 **长程** 效应。
- 已有工作用 **标量** 电荷 + \(1/r^p\) 等形式做长程修正；LOREM 的核心思想是使用 **等变（equivariant）“电荷”** 做长程消息传递，使长程部分仍能携带更高阶几何信息。
- 实现上，周期体系借助 **Ewald**（与文中 “Ewald message passing” 一脉相承），非周期体系用全对或长程边的 **Coulomb（\(p=1\)）** 形式；论文指出实践中 **仅用 \(p=1\)** 往往足够。

---

## 2. 架构总览（与 `lorem/lorem.py` 中 `Lorem` 类对应）

实现入口类为 **`lorem.Lorem`**（Flax `nn.Module`），主要超参见类属性（如 `cutoff`、`max_degree`、`num_message_passing`、`lr`、`equivariant_message_passing` 等）。

数据流与论文 Fig. 1 一致，可概括为：

1. **短程**：在半径 \(r_c\) 内构图；边上用 **径向展开 + 球谐角向展开 + 种类嵌入**，经 **`RadialCoefficients`** 得到边特征，再聚合到节点，维护 **标量节点特征** 与 **球谐型等变节点特征**；通过 **球谐范数（power spectrum 风格）** 把等变信息压回标量，经 **`Update` 残差块** 更新。
2. **重复 \(M\) 次**：由 `num_message_passing` 控制；可选 **`equivariant_message_passing=True`** 时使用 `e3x.nn.MessagePass` 做等变消息传递（见下文）。
3. **能量**：多处 **`MLP` 残差** 输出每原子能量，再累加（论文中的 residual atomic energy prediction）。
4. **长程（`lr=True`）**：由节点特征预测 **标量 + 低 \(l\) 的等变电荷**，经 **Ewald（周期）** 或 **\(1/r\) 求和（非周期、全边列表）** 得到势，再通过 **张量积** 与原有球谐特征混合，再次 `Update` + `MLP` 加到能量上。

以下按代码模块对照论文小节。

---

## 3. 短程部分：代码如何实现

### 3.1 初始嵌入 — `Initial`

文件：`lorem/lorem.py` 中类 **`Initial`**。

- **径向**：距离 \(r_{ij}\) 经 `RadialEmbedding`（如 Bernstein + 余弦截断 `cutoff_fn`），与 **`RadialCoefficients`** 中 MLP 输出的系数做收缩（论文中的 \(\rho_{ij,c}\) 与可学习系数矩阵思想一致）。
- **角向**：`e3x.so3.spherical_harmonics(R_ij, max_degree, ...)`，对应论文 **Angular expansion**。
- **种类**：`ChemicalEmbedding` 为 `nn.Embed`，将原子序数映射为向量。
- 输出：`radial`、`spherical`、`species`、`cutoffs`、`r_ij`，供后续使用。

### 3.2 边到节点的第一次聚合（含“无初始球谐节点特征”时的构造）

`Lorem.__call__` 中在 **`initialize_node_features`** 为真时用 `species` 初始化标量节点特征，否则标量初值为 **零向量**（与论文“种类嵌入”表述略有开关差异，由 `model.yaml` 控制）。

- 用 **`RadialCoefficients`** 输入 **拼接的 species 特征** `[species[i], species[j]]` 与径向展开得到 **边标量 `edges_scalar`**，聚合到目标节点 `i`，再 **`Update`** 更新 **`nodes_scalar`**。
- 将球谐系数与 **`spherical`** 按 l 通道组合，segment-sum 到节点得到 **`nodes_spherical`**，再 **`e3x.nn.TensorDense`**（对应论文中的张量/等变线性层）。
- 用 **`spherical_norm_last_axis`**（自定义 JVP 的球谐范数）+ **`l_factors`** 权重，将等变信息注入标量，再 **`Update`**。

这与论文 **Short-range message passing** 中“边特征 \(K_{ij}\) → 标量消息 + 球谐预因子 → 张量积 → 聚合 → 更新 \(S_i\) → 球谐范数更新 \(P_i\)”的叙述一致；**第一步**没有“与前一轮球谐节点特征做张量积”，代码里通过 **`TensorDense` 直接得到初始 `nodes_spherical`**，与论文脚注中“初始步用自张量积/略去与 \(S_j\) 的乘积”的说明相符。

### 3.3 消息传递循环 — `num_message_passing`

每一轮：

- 用 **当前** `nodes_scalar[i]`、`nodes_scalar[j]` 拼接（而非仅种类）作为 **`RadialCoefficients`** 的 pair 特征，重新计算边特征并聚合，更新标量节点。
- 若 **`equivariant_message_passing`** 为真：再次用系数与 **`spherical`**（边上固定的角向基）组合，经 **`e3x.nn.MessagePass`** 与 **`e3x.nn.Tensor`** 更新 **`nodes_spherical`**，再用球谐范数更新标量。
- 每轮末尾 **`energy += MLP(nodes_scalar)`**（残差式原子能量）。

论文中的 **\(M\)** 即 `num_message_passing`。

### 3.4 `Update` 块

类 **`Update`**：残差 + 两层 MLP + **LayerNorm**，与论文 **Update block** 公式一致（先加 `MLP(Y)`，再 LayerNorm，再自残差 `MLP(X)`，再 LayerNorm）。

---

## 4. 长程部分：代码如何实现

在 **`if self.lr:`** 分支中：

1. **电荷头**  
   - 标量：`MLP(nodes_scalar) -> scalar_charges`（每原子 1 维）。  
   - 等变：`TensorDense(features=1, max_degree=max_degree_lr)(nodes_spherical)`，再 reshape，与论文中 **低 \(l_{\max,\mathrm{LR}}\)** 的等变电荷一致；默认 `max_degree_lr=2` 时与文中“约 10 个电荷通道”的构造思想一致（标量 + \(l=0\) 球谐部分拼接等，见论文脚注）。

2. **势函数**  
   - **`k_grid is not None`（周期）**：`jaxpme.Ewald` 的 `calculator.potentials(...)`，对电荷最后一维 **vmap**，并行各通道。  
   - **`full_R_ij is not None`（非周期）**：对全边用 **屏蔽的 \(1/r\)**（\(r=0\) 处屏蔽）做 `segment_sum`，把 **长程边** 上的库仑式相互作用聚合到原子。

3. **回灌到短程特征**  
   - 将势的标量部分与球谐部分拆开，球谐势经 **`Dense`** 与 **`nodes_spherical` 做 `e3x.nn.Tensor` 张量积**，球谐范数与标量势拼接后 **`Update`**，最后 **`energy += MLP`**。

力：通过 **`predict` / `jax.value_and_grad`** 对边位移、**`positions`**（周期）或 **`full_edges`**（非周期）求导得到，与论文中可微能量一致。

---

## 5. 仓库中其它文件分工（训练与推理管线）

| 文件 | 作用 |
|------|------|
| `lorem/run.py` | 读取当前工作目录下 `settings.yaml`、`model.yaml`，用 `marathon.io.from_dict` 构建 `Lorem`，训练循环、checkpoint、日志（如 wandb）。 |
| `lorem/lorem.py` | **模型定义**（上文全部）。 |
| `lorem/sample.py` | 将 ASE `Atoms` 转为图批次（边、掩码、周期/非周期下的 `full_edges` 等）。 |
| `lorem/transforms.py` | `ToSample`、`SetUpEwald`（\(k\) 网格与 smearing）等数据变换。 |
| `lorem/calculator.py` | ASE `Calculator` 封装：加载 checkpoint、`jit` 推理。 |
| `lorem/metrics_*.py` / `lorem/results_*.py` | 各数据集评测与作图（常 `exec` 引用 `experiments/` 下脚本）。 |

依赖关系：训练依赖 **JAX + Flax + e3x + jax-pme**，数据路径由环境变量 **`DATASETS`** 指定（见主 `README.md`）。

---

## 6. 短程网络与 MACE、CACE 是否“相同”？

**结论：不相同；是同一大类思想下的不同架构与实现。**

| 维度 | LOREM（本仓库 `lorem/lorem.py`） | MACE | CACE（如 CACE-LES / 仓库中 `experiments/cumulene/cace`） |
|------|----------------------------------|------|--------------------------------------------------------|
| **实现框架** | JAX + Flax + **e3x** | 常见为 PyTorch + **e3nn** 等 | 常见为 PyTorch，自有 **CACE** 模块 |
| **短程几何** | 径向 Bernstein/Gaussian 类基 + **球谐**；`RadialCoefficients` + `TensorDense` / 可选 `MessagePass` | MACE 体序、对称收缩等 **MACE 特有** 结构 | CACE 的上下文感知与消息传递 **与 LOREM 不是同一套算子** |
| **长程** | 学习的 **标量 + 等变电荷** + Ewald / \(1/r\) | 另有 MACE 静电等扩展，机制不同 | CACE-LES 等为 **标量** 长程等，论文中明确与 LOREM 对比 |
| **在本仓库中的位置** | `lorem/` 核心模型 | `experiments/.../MACE/` 为 **基线评测脚本** | `experiments/.../cace/` 为 **基线评测脚本** |

论文亦将 **MACE、PET、CACE-LES** 等作为 **对比基线**，而非同一套代码路径。数据集说明中也提到部分数据与 **CACE-LES** 工作相关，但 **LOREM 模型代码并不复用 MACE/CACE 的网络定义**。

**概念上的联系**：三者都是 **SE(3)/E(3) 等变** 或近似等变的 MLIP，短程都用到 **截断图 + 距离与方向信息**；但 **层结构、张量积形式、消息函数、长程头** 均不同，不能视为“同一短程网络换皮”。

---

## 7. 配置文件如何对应论文章节

典型 `experiments/cumulene/.../model.yaml` 中：

- `cutoff` → 论文 \(r_c\)；  
- `max_degree` → 球谐最大角动量（短程角向通道）；  
- `num_message_passing` → 论文 \(M\)；  
- `lr: true/false` → 是否启用 **长程分支**；  
- `equivariant_message_passing` → 是否在短程循环中使用 **等变消息传递**（`e3x.nn.MessagePass`）；  
- `max_degree_lr`（代码默认 2）→ 论文 \(l_{\max,\mathrm{LR}}\)；  
- `initialize_node_features` → 是否用种类嵌入直接初始化标量节点特征。

更细的实验与附录（如不同 \(l_{\max,\mathrm{LR}}\)、Ewald 实现对比）以论文正文与附录为准。

---

## 8. 参考文献与代码索引

- 论文 PDF：仓库内 `Rumiantsev 等 - 2026 - Learning Long-Range Representations with Equivariant Messages.pdf`（若文件名不同，以你本地为准）
- 核心实现：`lorem/lorem.py`（类 `Lorem`、`Initial`、`RadialCoefficients`、`Update` 等）
- 训练入口：`lorem/run.py`

---

## 9. 相关文档

- 与 **CACE 短程** 的对比见同目录下 **`CACE_vs_LOREM_SR_zh.md`**（若存在）。

---

*文档随仓库路径编写；若你升级了 `lorem.py` 中的类名或前向参数，请同步对照更新本节中的类名。*
