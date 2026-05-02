## Qeq 中 SOG（高斯和）形式的 A 矩阵：当前求解方式与可行的加速路线

本文回答三个问题：

- **当前用的是什么方法**来“做矩阵求逆/解线性系统”？
- **哪里慢**（时间/显存复杂度是什么）？
- **A 是高斯和（SOG）这种好结构能否利用**，有哪些工程上可落地的加速方法？

文中公式使用 `$...$` 与 `$$...$$`。

---

## 1. 你在 Qeq 里真正要“求逆”的是什么？

对单个构型（省略 batch 下标），经典 Qeq 能量可以写成：

$$
E(q)=\frac12 q^\top A q + \frac12 q^\top J q + \chi^\top q,
$$

并带总电荷约束：

$$
\mathbf{1}^\top q = Q.
$$

其中：

- $A\in\mathbb{R}^{N\times N}$：长程库仑核矩阵（你当前也允许用 SOG 近似来构造）。
- $J=\mathrm{diag}(J_i)$：元素硬度（对角项，代码里每种元素一个可训练参数）。
- $\chi$：电负性特征（上游网络输出）。

一阶条件对应增广块系统：

$$
\begin{pmatrix}
A+J & \mathbf{1}\\
\mathbf{1}^\top & 0
\end{pmatrix}
\begin{pmatrix}
q\\ \lambda
\end{pmatrix}
=
\begin{pmatrix}
-\chi\\ Q
\end{pmatrix}.
$$

因此你要加速的是：在每个构型上**快速求解这个线性系统**（或者等价地快速施加 $(A+J)^{-1}$ 及其约束版本）。

---

## 2. 当前 CACE-SOG-Qeq 代码中用的是什么方法？

在 `cace/modules/charge_eq.py` 里（每个构型）：

- **显式构造 $A$**：
  - 若 `use_sog_kernel=True`：调用 `_build_A_sog`，直接构造一个密集的 $N\times N$ 矩阵（SOG 形式仍是显式算出每个 $A_{ij}$）。
  - 否则：从 `EwaldPotential.compute_potential_triclinic` 拿到 $A$（当前仓库的 `ewald.py` 是“简化版 1/r”，也会直接返回 $A$）。

- **直接求解增广系统**：在 `_compute_q_eq` 里用 `torch.linalg.solve` 解 $(N+1)\times(N+1)$ 系统。

对应源码关键行（供你对照）：

- 构造/选择 $A$：`ChargeEq.forward` 里 `A_now = self._build_A_sog(...)` 或 `compute_potential_triclinic(..., q_eye, ...)`。
- 直接求解：`sol = torch.linalg.solve(coeffs, chi_vector)`。

### 2.1 复杂度瓶颈

对每个构型（$N$ 原子）：

- **构造 $A$**：时间/显存都是 $O(N^2)$。
- **直接求解**：`torch.linalg.solve` 对密集矩阵通常是 $O(N^3)$ 时间、$O(N^2)$ 显存。

这在 $N\gtrsim 10^3$ 时通常就会成为训练/推理的主要瓶颈。

---

## 3. A 是 SOG（高斯和）时，结构是什么？能带来什么？

当你用 SOG 形式近似库仑核时：

$$
A_{ij} = K(r_{ij}) \approx \sum_{\ell=1}^{L} w_\ell\,\exp(-\alpha_\ell r_{ij}^2),
$$

或用宽度参数 $s_\ell$ 表示：

$$
K(r)=\sum_{\ell=1}^L w_\ell\,\exp(-r^2/s_\ell^2).
$$

这类核的关键优点在于：

- **对任意向量 $v$，$Av$ 是“高斯核对点源的卷积求和”**：

$$
(Av)_i = \sum_j K(|r_i-r_j|) v_j = \sum_{\ell=1}^L w_\ell\sum_j \exp(-\alpha_\ell|r_i-r_j|^2) v_j.
$$

- 也就是说：你不一定要显式构造 $A$；只要能快速做 `apply_A(v)`，就可以用迭代法解线性系统。

这就是“利用 SOG 结构”的核心：

> 把“显式矩阵 + 直接求解”换成“快速 matvec + 迭代求解”。

---

## 4. 可行的加速路线（按落地优先级）

下面几条路线并不互斥，通常是“先做 4.1 / 4.2，必要时再上 4.3/4.4”。

### 4.1 用迭代法替代 `torch.linalg.solve`（PCG/MINRES/GMRES）

令 $M=A+J$。因为 $J$ 是正对角，很多时候 $M$ 会更接近 SPD（对称正定），此时优先用 **PCG**；若不保证 SPD，可用 **MINRES/GMRES**。

关键是：迭代法每一步只需要：

- 矩阵向量乘 `Mv = A v + J\odot v`
- 一个（或多个）预条件器 `P^{-1}v`

**只要 `apply_A(v)` 快，整体就快。**

> 你目前的实现是直接解增广系统。迭代法版本一般会配合 Schur complement/投影来处理电中性约束（见你已有的 `SOG_Qeq_A_matrix_SOG_acceleration.md`）。

### 4.2 不构造 A：把 `_build_A_sog` 改成 `apply_A_sog(v)`

你现在的 `_build_A_sog` 显式构造 $A$：

- 计算所有 pair 距离 $r_{ij}$（$O(N^2)$）
- 再按高斯和求和

要利用 SOG 的优势，应当改为算子形式：

- 输入：`positions r`、`v`、(cell)
- 输出：`Av`

这样才能和 PCG 组合，避免 $O(N^2)$ 显存与 $O(N^3)$ 求解。

### 4.3 周期体系：SOG + FFT/NUFFT 做快速 matvec

在周期边界下，高斯核求和可以走“网格化 + 频域卷积”路线：

- **gridding**：把点源 $\sum_j v_j\delta(x-r_j)$ 近似投到网格
- **FFT/NUFFT**：对每个高斯分量做卷积（高斯在 $k$ 空间仍是高斯）
- **degridding**：从网格插值回粒子位置得到 $(Av)_i$

复杂度可做到接近 $\tilde O(N\log N)$（常数与网格大小、精度有关）。

你仓库里 `cace/modules/sog.py` 已经有 `pytorch_finufft` 的 NUFFT 实现思路（虽然当前 Qeq 的 A-sog 还没复用这条路径）。

### 4.4 非周期/开边界：FGT/IFGT /（高斯）FMM / 层次矩阵

如果不是周期体系（或你不想上网格）：

- **FGT/IFGT（Fast Gaussian Transform）**：针对高斯核求和的经典加速。
- **(Gaussian) FMM**：把远场贡献用多极展开压缩。
- **H-matrix / HSS / ACA**：把核矩阵分块近似成低秩，适合做近似求解或预条件。

这些方法工程量更大，但在大体系上效果显著。

---

## 5. 预条件器（决定 PCG 是否好用）

迭代法的“快”往往取决于预条件器。对 $M=A+J$ 常见选择：

- **Jacobi（对角预条件）**：$P=\mathrm{diag}(M)$，实现最简单。
- **Block-Jacobi / 分域**：按空间划分块，对每块做小规模直接求解。
- **近邻截断 + 稀疏近似**：把短程部分做稀疏近似作为预条件。
- **SOG 分量分层**：把最“长程”的几个分量用粗精度快速算，作预条件的近似 inverse。

经验上：先上 Jacobi/Block-Jacobi 就能让很多体系的迭代步数显著下降。

---

## 6. 你现在的实现能否“利用高斯和这种好形式”？结论

- **当前实现**：没有真正利用。
  - 即使 `use_sog_kernel=True`，也仍然显式构造 $A$（$O(N^2)$）并且直接解（$O(N^3)$）。

- **可以利用**，而且路线清晰：
  - 把 `A` 从“矩阵”改为“算子”（`apply_A(v)`）
  - 用 PCG/MINRES 替代 `torch.linalg.solve`
  - 周期体系优先走 SOG+FFT/NUFFT（你已有 NUFFT 相关代码可借鉴）
  - 叠加合适预条件器

这套组合通常是 Qeq 在大体系下最有效的加速方案。

---

## 7. 对你当前仓库的最小改动建议（工程落地顺序）

- **第 1 步（最小可用）**：
  - 在 `ChargeEq` 里新增一个迭代求解分支（PCG/MINRES），先保留显式 $A$ 但不直接 solve（验证正确性）。

- **第 2 步（真正加速）**：
  - 把 SOG 从 `_build_A_sog` 改成 `apply_A_sog(v)`，配合 PCG。

- **第 3 步（周期体系大幅提速）**：
  - 用 FFT/NUFFT 实现 `apply_A_sog(v)`（参考 `cace/modules/sog.py` 的 NUFFT 思路）。

如果你希望我把“第 1 步/第 2 步”直接实现成代码（新增 PCG 解算、并替换 `torch.linalg.solve`），我也可以在现有 `ChargeEq` 上继续往下改。
