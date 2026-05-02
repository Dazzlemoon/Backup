# MACE-polar-1 与 LOREM 长程机制对比（结合论文与代码）

本文对比：

- `MACE-polar-1.pdf`（对应代码在 `mace`）
- `Rumiantsev 等 - 2026 - Learning Long-Range Representations with Equivariant Messages.pdf`（LOREM，代码在 `LOREM/lorem`）

重点回答三个问题：

1. 两者长程部分“具体”有什么不同？
2. 这些差异在代码里如何体现？
3. MACE-polar-1 是否等同于 LOREM 的“等变电荷”架构思路？

---

## 1. 先给结论

两者都属于“**等变对象 + 长程核**”路线，但不是同一架构：

- **相同点**：都把长程相互作用写成类似 $1/r$ 的全局传播，并且传播的不是纯标量，而包含等变分量（多极/球谐分量）。
- **关键不同点**：
  - LOREM 把长程模块主要当作**全局消息传递层**（更新特征后再由 MLP 读出能量）。
  - MACE-polar-1 把长程模块做成**显式物理能量项 + 可极化迭代 + 电荷/自旋守恒平衡**的组合。

因此：**MACE-polar-1 借鉴了“等变长程通信”思想，但不是 LOREM 的同构实现**。

---

## 2. 论文层面的数学形式对比

## 2.1 LOREM：等变“电荷”做长程消息

LOREM 的核心长程公式（论文 Eq. (4)）是：

$$
V_{i,l,m}
=
\sum_{j=1}^{N}
\sum_{\mathbf{n}\in\mathbb{Z}^3}
\frac{Q_{j,l,m}}{|\mathbf{r}_i - \mathbf{r}_j + n_1\mathbf{c}_1+n_2\mathbf{c}_2+n_3\mathbf{c}_3|^p}
$$

其中 $Q_{j,l,m}$ 是等变“电荷”通道，常用 $p=1$。

然后把得到的势 $V$ 回注到局部特征里，再做残差能量读出。可理解为：

$$
\text{local features} \xrightarrow{\text{predict }Q}
\text{long-range potential }V
\xrightarrow{\text{feature update}}
\Delta E
$$

更像“**全局消息通道**”。

---

## 2.2 MACE-polar-1：显式密度-势-能量 + 可极化迭代

MACE-polar-1 的论文给出更强的物理分解（Eq. (1)）：

$$
E_{\text{total}}
=
E_{\text{local}} + E_{\text{non-local}} + E_{\text{electrostatic}}
$$

它不是仅把长程当特征通道，而是明确建模了电荷/自旋密度：

$$
\rho^\uparrow(\mathbf{r})=\sum_{i,l,m} p^\uparrow_{i,lm}\,\phi_{lm}(\mathbf{r}-\mathbf{r}_i),
\quad
\rho^\downarrow(\mathbf{r})=\sum_{i,l,m} p^\downarrow_{i,lm}\,\phi_{lm}(\mathbf{r}-\mathbf{r}_i)
$$

以及

$$
\rho=\rho^\uparrow+\rho^\downarrow,\qquad s=\rho^\uparrow-\rho^\downarrow
$$

在每次极化更新中，先由密度得到势（论文 Eq. (16)(17)）：

$$
v^{(u),\uparrow\downarrow}(\mathbf{r})
=
\int \frac{\rho^{(u),\uparrow\downarrow}(\mathbf{r}')}{|\mathbf{r}-\mathbf{r}'|}d\mathbf{r}'
+ \frac12 v_{\text{app}}(\mathbf{r})
$$

再做局部更新，并通过 Fukui 权重做守恒平衡（论文 Eq. (13)(14) 机制）来满足总电荷和总自旋约束。

这更接近“**弱 SCF 风格的可极化迭代**”（但论文强调是非自洽简化形式，避免完整 SCF 代价）。

---

## 3. 代码中的对应实现（最关键）

## 3.1 LOREM 代码：长程是一个 feature update block

核心文件：`LOREM/lorem/lorem.py`

- 长程开关：`if self.lr:`
- 先从局部特征构造 `scalar_charges` 与 `spherical_charges`，拼成 `charges`
- 周期体系：`jaxpme.Ewald(...).potentials(...)`
- 非周期体系：显式 $1/r$ 求和
- 用势更新标量特征后，再做一轮 residual energy 累加

代码路径（符号级）：

- `scalar_charges` / `spherical_charges`：`lorem.py` 中 `if self.lr` 段
- Ewald：`calculator = Ewald(...)` + `calculator.potentials(...)`
- 非周期求和：`potentials = segment_sum(charges[full_j] * one_over_r, ...)`
- 回注更新：`updates = concatenate([scalar_potential, norms])`
- 能量：`energy += masked(MLP(...), nodes_scalar, ...)[..., 0]`

补充：`LOREM/lorem/transforms.py` 的 `SetUpEwald` 负责 `k_grid` 与 `smearing`。

**含义**：LOREM 的长程量主要作为“消息/特征”使用，不单独显式拆一个物理可解释的 $E_{\text{electrostatic}}$ 输出头。

---

## 3.2 MACE-polar-1 代码：显式密度、势、库仑能、迭代平衡

核心文件：`mace/mace/modules/extensions.py` 中 `class PolarMACE`

### (a) 长程物理模块对象

- `GTOElectrostaticFeatures`：把多极密度映射成场特征
- `GTOElectrostaticEnergy`：显式计算库仑能
- `compute_k_vectors_flat`：构造 reciprocal-space $k$ 网格

这些来自 `graph_longrange`（代码中导入名）。

### (b) 自旋分辨多极与递推

- `spin_charge_density` 是两通道（up/down）多极系数
- `for i in range(self.num_recursion_steps):` 做极化递推
- 每步通过 `field_dependent_charges_maps[i]` 产生增量

### (c) Fukui 平衡（守恒约束）

- `fukui_source_map` 预测每原子 Fukui 权重
- 用批内归一化后，把差额补到 monopole：
  - `Q_p_S = total_charge + (total_spin - 1)`
  - `Q_m_S = total_charge - (total_spin - 1)`
  - 按归一化 Fukui 分配 deficit

这是论文 Eq. (13)(14)(15) 的直接工程化实现。

### (d) 显式能量拼装

`PolarMACE.forward` 中有清晰的能量路径：

$$
E = E_{\text{local/backbone}}
 + E_{\text{electron/local-field}}
 + E_{\text{electrostatic}}
 + E_{\text{external-field coupling}}
$$

可见变量：

- `total_energy = e0 + inter_e`
- `le_total = self.local_electron_energy(...)`（可选加到总能）
- `electro_energy = self.coulomb_energy(...)`
- `total_energy += electro_energy + ...`

这与论文里的分解思想是一致的。

---

## 4. “长程部分”最本质的区别

可以压缩成 5 条：

1. **目标函数层级不同**  
   - LOREM：长程主要是表示学习通道。  
   - MACE-polar-1：长程是物理可解释能量组成的一部分。

2. **状态变量不同**  
   - LOREM：单套等变 charges（无显式自旋上下通道守恒机制）。  
   - MACE-polar-1：显式 $\uparrow/\downarrow$ 两通道多极密度。

3. **是否有显式守恒平衡**  
   - LOREM：无类似 Fukui 归一化电荷平衡的显式步骤。  
   - MACE-polar-1：每步都做 Fukui equilibration 保证总量约束。

4. **是否显式输出库仑能**  
   - LOREM：不单独显式 `electrostatic_energy` 头。  
   - MACE-polar-1：显式 `electrostatic_energy`，并回传力。

5. **是否迭代极化**  
   - LOREM：标准前向里是单次 LR block（可配多次 SR message passing，但不是同构 SCF 迭代）。  
   - MACE-polar-1：`num_recursion_steps` 的固定点式场-电荷递推。

---

## 5. MACE-polar-1 是否“也是 LOREM 的等变电荷思路”？

结论分两层：

- **是（思想层）**：  
  都使用了“等变长程载体 + $1/r$ 核传播”的核心观念，都超越了只用标量电荷的长程项。

- **不是（架构层）**：  
  MACE-polar-1 在该思想上进一步引入了
  - 自旋分辨密度（$\rho^\uparrow,\rho^\downarrow$）  
  - Fukui 归一化平衡（电荷/自旋守恒）  
  - 显式库仑能与非局域能拆分  
  - 可极化递推更新  
  这些并不是 LOREM 原始实现中的同构组件。

一句话总结：**MACE-polar-1 与 LOREM 共享“等变长程通信”基因，但 MACE-polar-1 更偏“物理约束增强的可极化能量模型”，LOREM 更偏“等变长程消息网络”。**

---

## 6. 你可直接对照的源码入口

- MACE-polar 预训练模型入口：`mace/mace/calculators/foundations_models.py` (`mace_polar`)
- MACE-polar 核心实现：`mace/mace/modules/extensions.py` (`PolarMACE`)
- LOREM 核心实现：`LOREM/lorem/lorem.py` (`class Lorem`)
- LOREM Ewald 准备：`LOREM/lorem/transforms.py` (`SetUpEwald`)

如果你希望，我可以再给你补一版“**逐行对照表**”（把 `PolarMACE.forward` 与 `Lorem.__call__` 按步骤并排映射），用于你后续做复现实验或改模型。

---

## 7. 自旋分辨密度与 Fukui 归一化平衡（补充详解）

## 7.1 自旋分辨密度是什么

传统只看总电荷密度：

$$
\rho(\mathbf r)=\rho^\uparrow(\mathbf r)+\rho^\downarrow(\mathbf r)
$$

MACE-polar-1 同时追踪两路通道：

- $\rho^\uparrow(\mathbf r)$：自旋上通道
- $\rho^\downarrow(\mathbf r)$：自旋下通道

并定义自旋密度：

$$
s(\mathbf r)=\rho^\uparrow(\mathbf r)-\rho^\downarrow(\mathbf r)
$$

这让模型不仅能表示“总电荷怎么分布”，还能表示“自旋怎么分布”，对自由基、开壳层、不同多重度体系更关键。

在代码里，对应 `PolarMACE.forward` 的 `spin_charge_density`（形状可理解为 $N\times2\times D$，2 是 $\uparrow/\downarrow$ 两通道，$D$ 是多极展开维度）。

## 7.2 Fukui 归一化平衡是什么

局部网络先给出单极初猜 $\tilde p_{i,00}^{\uparrow/\downarrow}$，通常不严格满足全体系约束。  
于是每步用 Fukui 权重做全局回填，保证目标总电荷 $Q$ 与总自旋 $S$：

$$
p_{i,00}^{\uparrow,\text{new}}
=
\tilde p_{i,00}^{\uparrow}
+
\frac{f_i^\uparrow}{\sum_j f_j^\uparrow}
\left(
\frac{Q+S}{2}-\sum_j \tilde p_{j,00}^{\uparrow}
\right)
$$

$$
p_{i,00}^{\downarrow,\text{new}}
=
\tilde p_{i,00}^{\downarrow}
+
\frac{f_i^\downarrow}{\sum_j f_j^\downarrow}
\left(
\frac{Q-S}{2}-\sum_j \tilde p_{j,00}^{\downarrow}
\right)
$$

含义是：把“全局 deficit”按原子 Fukui 权重分配，从而每轮都满足守恒。

## 7.3 与 LOREM 的关键区别

- LOREM 的等变 charges 主要是长程消息载体，不包含显式的“每轮守恒回填”步骤。
- MACE-polar-1 的 `spin_charge_density` 是状态变量，每轮都有：
  - 场驱动增量更新
  - Fukui 归一化守恒修正（主要作用于单极）

---

## 8. MACE-polar-1 电荷递推在代码里如何实现

设递推状态为：

$$
P^{(u)}\equiv \text{spin\_charge\_density}^{(u)}
$$

在 `PolarMACE.forward` 中，每轮 `for i in range(self.num_recursion_steps):` 可抽象成：

1. 用当前状态算长程势特征

$$
V^{(u)}=\mathcal{V}\!\left(P^{(u)}\right)
$$

2. 网络输出“原始增量 + 本轮 Fukui 权重”

$$
(\Delta P_{\text{raw}}^{(u)}, f^{(u)})
=
\mathcal{G}_u\!\left(h_{\text{local}},V^{(u)},P^{(u)}\right)
$$

3. 先做原始增量更新

$$
P_{\text{tmp}}^{(u+1)}=P^{(u)}+\Delta P_{\text{raw}}^{(u)}
$$

4. 再做单极守恒回填（Fukui）

$$
P^{(u+1)}=\text{FukuiBalance}\!\left(P_{\text{tmp}}^{(u+1)};Q,S\right)
$$

### “每步增长多少”是否固定？

不是固定步长。每轮总改变量是：

$$
\Delta P^{(u)}=
\Delta P_{\text{raw}}^{(u)}
+
\Delta P_{\text{fukui}}^{(u)}
$$

它由网络输出与当前 deficit 共同决定，可正可负，不是常数。

---

## 9. 2 原子 toy 数值示例（1 轮 vs 2 轮）

为直观起见，仅看单极子 $p_{00}$，忽略自旋拆分细节（等价看合并后的净电荷通道）。

- 两原子 A/B，距离 $r=3.0\ \text{\AA}$
- 初始守恒后：

$$
p_A^{(0)}=+0.26,\quad p_B^{(0)}=-0.26
$$

### 第 1 轮

$$
v_A^{(0)}=\frac{p_B^{(0)}}{r}=-0.0867,\quad
v_B^{(0)}=\frac{p_A^{(0)}}{r}=+0.0867
$$

设 toy 更新律：

$$
\Delta p_i^{(0)}=-0.5\,v_i^{(0)}
$$

得：

$$
p_A^{(1)}=0.3033,\quad p_B^{(1)}=-0.3033
$$

### 第 2 轮

$$
v_A^{(1)}=-0.1011,\quad v_B^{(1)}=+0.1011
$$

$$
p_A^{(2)}=0.3539,\quad p_B^{(2)}=-0.3539
$$

如果取示意总能：

$$
E_{\text{tot}}=E_{\text{local}}+E_{\text{coul}}+E_{\text{nonlocal}}
$$
$$
E_{\text{local}}=-1.2000,\quad
E_{\text{coul}}=\frac{p_Ap_B}{r},\quad
E_{\text{nonlocal}}=-0.02(p_A^2+p_B^2)
$$

则得到：

- 初始：$E_{\text{tot}}=-1.22523\ \text{eV}$
- 1 轮后：$E_{\text{tot}}=-1.23435\ \text{eV}$
- 2 轮后：$E_{\text{tot}}=-1.24675\ \text{eV}$

这个 toy 说明：最终长程能量仍由“电荷/多极”给出，但这个电荷是**递推后的终态**，不是一次局部读出。

---

## 10. 常见误解澄清：MACE-polar-1 不看 spin 时是否就等于 LOREM 电荷

不等价。最多只能说“最后都能得到一个标量电荷输出”。

- LOREM：短程读出 scalar/spherical charges，作为长程消息载体。
- MACE-polar-1：短程先给初猜，然后长程迭代更新，并在每轮执行 Fukui 守恒回填；最终标量电荷来自

$$
p_{i,00}^{\uparrow}+p_{i,00}^{\downarrow}
$$

所以语义不同：  
LOREM 更偏“长程消息表示”；MACE-polar-1 更偏“受物理约束的可极化密度状态”。

---

## 11. 论文中对该设计优势与算力披露

## 11.1 报告的优势（示例）

论文给出了多处定量改进（相对短程或无显式电静力基线）：

- 摘要：蛋白-配体任务约 **4 倍改进**
- S30L 超分子复合物：从 `MACE-OMOL` 的 **7.31 kcal/mol** 降至 `MACE-POLAR-1-L` 的 **3.52 kcal/mol**（`M` 为 4.78）
- 离子氢键任务约 **3 倍改善**
- Cl$_2$/Cl$^-$ 局域化示例：约 **1.952 |e|** 局域在远端 Cl$^-$ 团簇，约 **0.048 |e|** 在 Cl$_2$ 团簇

## 11.2 算力/效率披露边界

- 论文明确给出训练规模：OMol25 100M 结构，使用 **64 × NVIDIA H200** 训练。
- 但主文重点在精度与化学任务表现，并未系统给出完整推理吞吐/`ns/day` 类型速度表。

