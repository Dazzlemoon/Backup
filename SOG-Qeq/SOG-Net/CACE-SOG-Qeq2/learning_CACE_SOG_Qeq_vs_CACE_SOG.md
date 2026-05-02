# 当前学习方式（CACE-SOG-Qeq）与 CACE-SOG 学习方式的对比

本文对比两类设置下“谁在学什么、长程能量从哪来、损失如何反传”，便于你在同一套代码里选型或混合使用。公式使用 `$...$` 与 `$$...$$`。

---

## 1. 概念区分

- **当前学习方式（CACE-SOG-Qeq）**：以 **ChargeEq + Ewald（或可选 SOG 核）** 为核心的长程路径；长程能量主要来自 Qeq 模块的二次型 $E_{\text{long}} = \frac12 q^\top A q$，$A$ 由 Ewald 或可选的“可学习 SOG 核”给出。
- **CACE-SOG 学习方式**：以 **SOGPotential** 为核心的长程路径；长程能量由 SOG 卷积给出 $E_{\text{long}} = E_{\text{SOG}}(r, q; w_\ell, s_\ell)$，其中 $q$ 可以是上游 NN 直接给的电荷，也可以是 ChargeEq 给出的 $q_{\text{eq}}$；**可学习的是 SOG 的幅度/宽度参数**（如 `wl`, `sl`），用于拟合“真实体系的长程/屏蔽/多体 tail”。

---

## 2. 当前方式（CACE-SOG-Qeq，ChargeEq 为主）

### 2.1 数据流与谁在学

- **长程能量来源**：  
  $$
  E_{\text{long}} = \frac12 q^\top A q,\quad q = q_{\text{eq}}\ \text{（Qeq 解）}.
  $$
  - $A$ 要么来自 **EwaldPotential**（固定物理核，不学习），  
  - 要么来自 **可选的 SOG 核**（`use_sog_kernel=True` 时的 `sog_log_alpha`, `sog_weights`），在“逼近 $1/r$”的函数族里做**小幅可学习**。

- **被学习的量**：
  - **$\chi$**：由 CACE 上游 NN（如 Atomwise）从结构特征预测，**NN 参数**通过能量/力损失反传；
  - **$J$（硬度）**：`ChargeEq.J_raw`，按元素的可训练参数；
  - **（可选）SOG 核参数**：若开启 `use_sog_kernel`，则 `sog_log_alpha`、`sog_weights` 参与 $A$ 的构造，通过 $E_{\text{QEq}}$ 进入损失一起更新。

- **损失与反传**：  
  总能量一般为 $E = E_{\text{SR}} + E_{\text{long}}$，其中 $E_{\text{long}} = \text{ewald\_potential}$（即 $\frac12 q^\top A q$）。梯度经 Qeq 的线性求解、$A$ 的构造，回传到 $\chi$、$J$ 以及（若启用）SOG 参数。**核 $A$ 的角色是“库仑骨架”，学习被约束在接近 $1/r$ 的 SOG 族内。**

### 2.2 小结（当前方式）

| 项目         | 内容 |
|--------------|------|
| 长程能量形式 | $\frac12 q^\top A q$，$q=q_{\text{eq}}$ |
| $A$ 的来源   | Ewald（固定）或可学习 SOG（逼近 $1/r$） |
| 学习对象     | $\chi$（NN）、$J$、可选 SOG 核参数 |
| 物理含义     | Qeq 电荷 + 库仑二次型，核保持“库仑型” |

---

## 3. CACE-SOG 方式（SOGPotential 为主）

### 3.1 数据流与谁在学

- **长程能量来源**：  
  $$
  E_{\text{long}} = E_{\text{SOG}}(r, q; w_\ell, s_\ell),
  $$
  由 **SOGPotential** 的前向计算得到（实空间或周期 SOG 卷积）。  
  - 输入电荷 $q$：可以是 NN 直接输出的电荷，也可以是 **ChargeEq 的 $q_{\text{eq}}$**（此时需把 `SOGPotential.feature_key` 设为 `'q_eq'`）。

- **被学习的量**：
  - **SOG 核参数**：`SOGPotential` 的 `wl`、`sl`（或等价地 amplitude / shift）为 **nn.Parameter**，初值通常取“逼近 $1/r$”的 SOG 拟合，训练中**自由更新**，用于拟合真实数据里的长程/屏蔽/多体 tail；
  - 若前面接了 ChargeEq，则 $\chi$、$J$ 也照常学习；但**长程部分进入损失的是 SOG 的输出**，而不是必需要求 $E_{\text{long}} = \frac12 q^\top A q$ 与 SOG 一致。

- **损失与反传**：  
  总能量 $E = E_{\text{SR}} + E_{\text{SOG}}$。梯度经 SOG 卷积回传到 `wl`、`sl` 以及（若存在）产生 $q$ 的模块（如 $\chi$ 网络、ChargeEq 的 $J$）。**这里“长程核”的学习主要由 SOGPotential 承担，允许在更大范围内拟合数据。**

### 3.2 小结（CACE-SOG 方式）

| 项目         | 内容 |
|--------------|------|
| 长程能量形式 | $E_{\text{SOG}}(r, q; w_\ell, s_\ell)$（SOG 卷积） |
| 核参数       | SOGPotential 的 `wl`, `sl` 等，可学习、可偏离纯 $1/r$ |
| 学习对象     | SOG 核参数为主；若接 Qeq，则还有 $\chi$、$J$ |
| 物理含义     | 长程/屏蔽/多体效应由 SOG 核灵活拟合 |

---

## 4. 核心差异对比

| 维度           | 当前方式（CACE-SOG-Qeq）           | CACE-SOG 方式                     |
|----------------|------------------------------------|-----------------------------------|
| 长程能量谁算   | ChargeEq（$A$ + 求 $q_{\text{eq}}$） | SOGPotential（卷积，输入 $q$）    |
| 核是否可学习   | 可选（SOG 核时在“逼近 $1/r$”族内） | 是，SOG 参数自由学                |
| 核的物理约束   | 强：希望 $A$ 保持库仑型             | 弱：SOG 核可学出屏蔽/tail         |
| 电荷从哪来     | Qeq 自洽解                         | $q_{\text{eq}}$ 或其它 NN 电荷    |
| 典型总能量     | $E_{\text{SR}} + \frac12 q^\top A q$ | $E_{\text{SR}} + E_{\text{SOG}}$ |

---

## 5. 混合用法（你文档里已有）

- **CACE‑LES 风格**：  
  长程只用 ChargeEq 的 `ewald_potential`，即 $E_{\text{long}} = \frac12 q^\top A q$，不接 SOGPotential；学习的就是当前方式（$\chi$、$J$、可选 SOG 核）。

- **SOG‑Net 风格**：  
  长程用 SOGPotential，输入为 `q_eq`；即 $E_{\text{long}} = E_{\text{SOG}}(r, q_{\text{eq}}; w_\ell, s_\ell)$。此时“长程如何随 $r,q$ 变化”主要由 **SOGPotential 的参数**学习，ChargeEq 负责提供物理合理的 $q_{\text{eq}}$，二者可同时反传、联合训练。

---

## 6. 总结一句话

- **当前学习方式（CACE-SOG-Qeq）**：长程由 Qeq 的二次型 $q^\top A q$ 给出，$A$ 要么固定（Ewald），要么在“逼近 $1/r$”的 SOG 族里小范围可学习；CACE 主要学 $\chi$、$J$ 和（可选）SOG 核，**强调核的物理性**。
- **CACE-SOG 学习方式**：长程由 SOGPotential 的卷积给出，SOG 的幅度/宽度参数**自由学习**，用于拟合真实体系的长程、屏蔽、多体效应；若接 Qeq，则还学 $\chi$、$J$，但长程形状主要由 SOG 模块决定。

二者可以并存于同一套代码中，通过“是否接入 SOGPotential、长程能量取 `ewald_potential` 还是 `SOG_potential`”来切换或混合。
