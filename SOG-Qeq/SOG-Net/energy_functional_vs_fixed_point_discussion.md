# Baldwin 2026：Energy Functional vs Fixed Point（训练公式与实现讨论）

本文档基于论文 **Baldwin et al., 2026, Design Space of Self-Consistent Electrostatic MLIPs** 的理论与训练章节整理，重点回答：

1. Energy Functional 架构的训练公式是什么？  
2. Fixed Point 架构的训练公式是什么？  
3. 这两类方法是否只能依托 MACE 消息传递训练？  
4. 是否可以用 CACE 的方式训练？

---

## 1) Energy Functional 架构：核心方程与训练

### 1.1 能量函数形式（论文式 (17)/(33) 对应）
论文将总能量写成「局域项 + 非局域/泛函项 + 库仑项 + 外场项」的结构。可写为：

$$
E(\mathbf p;\theta) \;=\; E_{\mathrm{local}}(\theta) \;+\; G_{\mathrm{ML}}(\mathbf p;\theta) \;+\; E_{\mathrm{Coulomb}}[\rho(\mathbf p)].
$$

其中粗粒化电荷密度由基函数展开：

$$
\rho(\mathbf r)=\sum_k p_k \,\phi_k(\mathbf r).
$$

推理时需要对 $\mathbf p$ 做约束优化（固定总电荷），得到自洽解 $p^*$，再用 $E(p^*;\theta)$ 输出能量。

---

### 1.2 直接训练（Direct Training）目标（论文式 (75)）
论文给出的 direct 训练（在 DFT 电荷上监督）可写为：

$$
\mathcal L_{\mathrm{direct}}^{\mathrm{EF}}
=
\left\|E_{\mathrm{ML}}(\theta,\mathbf p_{\mathrm{DFT}})-E_{\mathrm{DFT}}\right\|^2
\;+\;
\left\|\nabla_{\mathbf p}E_{\mathrm{ML}}(\theta,\mathbf p)\right\|_{\mathbf p_{\mathrm{DFT}}}^{2}.
$$

含义：
- 第一项：在 DFT 电荷描述处，能量要对齐 DFT；
- 第二项：要求 DFT 电荷在模型能量面上是驻点（stationary point）。

这需要可用的 DFT 原子多极矩/电荷分解作为监督。

---

## 2) Fixed Point 架构：核心方程与训练

### 2.1 固定点迭代定义（论文式 (41)）
论文定义固定点更新：

$$
\mathbf p^{(t+1)} = F_{\mathrm{ML}}\!\left(\theta,\{\!(z_i,\mathbf r_i)\!\}_i,\;v_{\mathrm{eff}}[\mathbf p^{(t)}]\right),
$$

$$
v_{\mathrm{eff}}[\mathbf p](\mathbf r)
=
\int \frac{\rho(\mathbf r')}{|\mathbf r-\mathbf r'|}\,d\mathbf r'
 + v_{\mathrm{app}}(\mathbf r) + \mu,
$$

其中 $\mu$ 是控制总电荷的拉格朗日乘子（类似费米能级角色）。迭代到收敛得 $p^*$。

能量单独计算（论文式 (42)）：

$$
E =
E_{\mathrm{local}}
 + E_{\mathrm{nonlocal}}(\{u_i\}_i)
 + \frac12\iint \frac{\rho(\mathbf r)\rho(\mathbf r')}{|\mathbf r-\mathbf r'|}\,d\mathbf r\,d\mathbf r'
 + \int v_{\mathrm{app}}(\mathbf r)\rho(\mathbf r)\,d\mathbf r.
$$

---

### 2.2 直接训练（Direct Training）目标（论文式 (76)/(78)）
固定点 direct 监督写为：

$$
\mathcal L_{\mathrm{direct}}^{\mathrm{FP}}
=
\left\|F_{\mathrm{ML}}(\theta,\mathbf v_{\mathrm{DFT}})-\mathbf p_{\mathrm{DFT}}\right\|^2,
$$

其中论文实践常用

$$
\mathbf v_{\mathrm{DFT}}=\mathbf v[\mathbf p_{\mathrm{DFT}}]
$$

得到：

$$
\mathcal L_{\mathrm{direct}}^{\mathrm{FP}}
=
\left\|F_{\mathrm{ML}}(\theta,\mathbf v[\mathbf p_{\mathrm{DFT}}])-\mathbf p_{\mathrm{DFT}}\right\|^2.
$$

这等价于“让 DFT 电荷成为模型 SC 循环的一个解”。

---

## 3) 论文中的统一训练框架（并非只限 direct）

论文还比较了两种更“推理一致”的训练方式：

1. **Implicit Differentiation**（论文式 (80)-(85)）：  
   定义
   $$
   \mathbf f(\theta,\mathbf p)=F_{\mathrm{ML}}(\theta,\mathbf v[\mathbf p])-\mathbf p,
   $$
   在自洽解 $\mathbf p^*$ 满足 $\mathbf f(\theta,\mathbf p^*)=0$，用隐式求导
   $$
   \frac{\partial \mathbf p^*}{\partial \theta}
   =
   -\left(\frac{\partial \mathbf f}{\partial \mathbf p^*}\right)^{-1}
   \frac{\partial \mathbf f}{\partial \theta}.
   $$

2. **Unrolling SC loop**：  
   训练时只跑固定步数 SC 迭代并反向传播（把 SC 过程当深层网络展开）。

论文指出 direct 往往存在训练/推理不一致、稳定性和泛化问题；implicit 或 shortcut-SCF + 后期 implicit/fully-unrolled 更实用。

---

## 4) 是否“只能依托 MACE 消息传递”？

**结论：不是。**

论文在实现章节写的是“在 MACE 框架中实现并对比”（Implementation in the MACE Framework），这是一个**实现载体**，不是理论上唯一可行载体。  

Energy Functional / Fixed Point 的关键需求本质上是：
- 一个可学习的局域表示器（几何与元素到特征）；
- 一个粗粒化电荷密度参数化 $\rho(\mathbf r;\mathbf p)$；
- 长程库仑/外场算子；
- 一个可微分的自洽求解流程（优化或固定点迭代）；
- 与能量/力/偶极等目标的联合训练。

这些需求并不绑定到 MACE 的消息传递形式。

---

## 5) 可以用 CACE 方式训练吗？

**结论：可以，且你们当前 SOG-Qeq/CACE 的工作就是这条路线的实例化之一。**

从架构映射看：

- MACE 中的局域表示器 $h_i$ 角色  
  $\rightarrow$ CACE 表示器输出特征；

- $F_{\mathrm{ML}}$（由有效势更新电荷）  
  $\rightarrow$ CACE 中可由 `ChargeEq` 模块 + 神经网络映射实现；

- $G_{\mathrm{ML}}$ / $E_{\mathrm{nonlocal}}$  
  $\rightarrow$ CACE 的可学习非局域能量项（如你们的 SOG/QEq 组合）；

- 库仑项与外场项  
  $\rightarrow$ CACE 中已有 Ewald/SOG/外场接口即可承载；

- 训练法  
  $\rightarrow$ 同样可做 direct / implicit / unroll（取决于你们当前求解器是否可微、线性系统/Jacobian 求解是否稳定）。

---

## 6) 用 CACE 落地时的关键注意点

1. **总电荷约束实现方式**：$\mu$（或等价约束）更新要稳定，避免 SC 迭代震荡。  
2. **尺度一致性**：`q_eq` 与物理电荷尺度（如 `normalization_factor`）必须在训练和评估口径上统一。  
3. **损失配比**：能量/力/偶极/电荷联合训练时，权重要避免某一项压制其他项。  
4. **训练-推理一致性**：若推理用“收敛 SC”，训练阶段至少后期也应接近该流程（implicit 或更多 unroll）。  
5. **可微分数值稳定性**：implicit differentiation 涉及线性系统求解，需监控条件数、收敛阈值与混合参数。

---

## 7) 简短回答（给你的三个问题）

- **Energy Functional 训练公式**：核心是式 (75) 的 direct 目标（能量对齐 + 在 DFT 电荷处驻点约束），也可用 implicit/unroll 训练。  
- **Fixed Point 训练公式**：核心是式 (76)/(78) 的 direct 目标（DFT 电荷是 SC 方程解），以及式 (80)-(85) 的 implicit 求导训练。  
- **是否只能 MACE？能否 CACE？**：不只能 MACE；完全可以用 CACE 做，前提是满足上面列出的表示、库仑算子、自洽迭代与可微训练条件。

