# 在 CACE-SOG-Qeq 中采用 DP-QEq 式矩阵加速的可行性说明

本文说明：能否用 DP-QEq 中“不构造矩阵 + 投影梯度 + LBFGS”的方式，对 CACE-SOG-Qeq 里当前的**矩阵构造与直接求逆**进行加速；结论是**可以**，并给出实现要点与注意事项。公式使用 `$...$` 与 `$$...$$`。

---

## 1. 当前 CACE-SOG-Qeq 的求解方式

在 `ChargeEq`（`charge_eq.py`）中：

- **构造 $A$**：`_compute_A_matrix(r, cell)` 通过令 $q = I$（单位矩阵）调用 `EwaldPotential.compute_potential_triclinic(r, q_eye, cell)`，取返回的 `q_field` 作为 $A$（即 $\phi = A q$ 在 $q$ 为各单位向量时的拼成矩阵）。
- **求电荷**：`_compute_q_eq(A_mat, chi, J, system_Q)` 组装增广线性系统
  $$
  \begin{pmatrix} A+J & \mathbf{1} \\ \mathbf{1}^\top & 0 \end{pmatrix} \begin{pmatrix} q \\ \lambda \end{pmatrix} = \begin{pmatrix} -\chi \\ Q \end{pmatrix},
  $$
  并用 `torch.linalg.solve` 直接求逆得到 $(q,\lambda)$。

因此当前实现同时存在：**显式构造 $N\times N$ 矩阵 $A$** 和 **对 $(N+1)\times(N+1)$ 增广矩阵求逆**，在大 $N$ 时内存与算力成本都较高。

---

## 2. DP-QEq 的加速思路（简要）

DP-QEq（见 `DP-QEq/Qeq_matrix_acceleration.md`）的做法是：

1. **不构造 $A$**：库仑能 $E_{\text{Coulomb}}(q)$ 用 PME 等直接按 $q$ 算出来，梯度 $\nabla_q E_{\text{Coulomb}} = A q$ 由 JAX 自动微分或由势场接口给出。
2. **不求逆线性系统**：把 Qeq 写成带约束最小化 $\min_q E_{\text{QEq}}(q) \ \text{s.t.} \ \mathbf{1}^\top q = Q$，用 **投影梯度 + LBFGS** 迭代，只依赖“能量 + 一阶梯度”，不组矩阵、不求逆。

因此，若在 CACE 里也能“只算 $E(q)$ 和 $\nabla_q E(q)$、不组 $A$、不直接求逆”，就可以套用同一套加速思路。

---

## 3. 在 CACE 中是否具备“不构造 $A$”的条件？

可以。理由如下。

- **$E_{\text{QEq}}$ 的组成**：
  $$
  E(q) = \frac{1}{2} q^\top A q + \frac{1}{2} \sum_i J_i q_i^2 + \sum_i \chi_i q_i.
  $$
  对 $q$ 的梯度为
  $$
  \nabla_q E = A q + J\odot q + \chi.
  $$

- **Ewald 接口的语义**：`compute_potential_triclinic(r, q, cell, compute_field=True)` 在传入**任意电荷向量** $q$（形状 `(N,)` 或 `(N,1)`）时：
  - 返回的 **`pot`** 对应库仑部分的能量（倒空间等），即 $E_{\text{coul}}(q) = \frac{1}{2} q^\top A q$ 的数值（或与之成固定比例，取决于 `norm_factor`）；
  - 返回的 **`q_field`** 即为势 $\phi = A q$，即 $\nabla_q \big(\frac{1}{2} q^\top A q\big) = A q$。

因此，**一次调用** `compute_potential_triclinic(r, q, cell, compute_field=True)` 就能得到：
- 库仑能量标量，
- $A q$（用于与 $J$、$\chi$ 拼成完整梯度），

而**不需要**先构造 $N\times N$ 的 $A$。当前用 $q = I$ 只是为了“批量”得到 $A$ 的每一列；若只为求 $q_{\text{eq}}$，完全可以用“对当前迭代的 $q$ 算一次能量 + 一次 $Aq$”的方式替代。

---

## 4. 在 CACE 中实现 DP-QEq 式加速的要点

### 4.1 新增“能量 + 梯度”接口（不组 $A$）

对单构型 $(r,\mathrm{cell})$ 和给定 $q$，实现标量函数与梯度：

- **库仑**：调用一次
  `pot, q_field = self.ep.compute_potential_triclinic(r, q, cell, compute_field=True)`  
  - $E_{\text{coul}}(q) = \mathrm{pot}$（注意与现有 `norm_factor`、是否含实空间等保持一致）；  
  - $\nabla_q E_{\text{coul}} = \mathrm{q\_field}$（即 $A q$）。
- **On-site**：$E_{\text{on}} = \sum_i \big( \chi_i q_i + \frac{1}{2} J_i q_i^2 \big)$，梯度为 $\chi + J\odot q$。
- **总能量与总梯度**：
  $$
  E(q) = E_{\text{coul}}(q) + E_{\text{on}}(q), \quad \nabla_q E = A q + J\odot q + \chi.
  $$

这样，**全程不构造 $A$**，只在对当前 $q$ 的一次 Ewald 调用中得到“能量 + $Aq$”。

### 4.2 投影梯度（保持 $\mathbf{1}^\top q = Q$）

与 DP-QEq 相同：在每步得到 $g = \nabla_q E$ 后，投影到约束流形上：

$$
g \leftarrow g - \frac{\mathbf{1}^\top g}{N} \mathbf{1},
$$

使更新方向满足 $\mathbf{1}^\top (\delta q) = 0$，从而不破坏总电荷约束。若使用 PyTorch，可写为：

```python
def project_grad(grad, constraint_ones):
    # constraint_ones: (1, N) 或 (N,) 全 1
    a = constraint_ones @ grad.reshape(-1, 1)
    b = (constraint_ones ** 2).sum()
    return grad - (a / b).flatten() * constraint_ones.flatten()
```

### 4.3 用 LBFGS 求约束最小化

- **优化变量**：当前步的电荷 $q \in \mathbb{R}^N$。
- **目标**：$\min_q E(q)$，约束 $\mathbf{1}^\top q = Q$。
- **每步迭代**：
  1. 用上文的接口算 $E(q)$ 和 $\nabla_q E$；
  2. 对梯度做投影得到 $\tilde{g}$；
  3. 用 LBFGS（如 `torch.optim.LBFGS` 或自写）在 $q$ 上做一步更新。
- **约束满足**：初始 $q^{(0)}$ 取为满足 $\mathbf{1}^\top q^{(0)} = Q$（例如上一步解或均匀分配）；每步用投影梯度保证搜索方向在切空间内，再配合线搜索或 LBFGS 的步长，使 $q$ 始终保持在流形上（或每步后做一次简单投影/缩放使 $\mathbf{1}^\top q = Q$）。

这样就不再需要组增广矩阵、也不调用 `torch.linalg.solve`。

### 4.4 与现有实现的数值一致性与单位

- 使用与 `ChargeEq` 相同的 `EwaldPotential`、`norm_factor`、`remove_self_interaction` 等，这样 $E_{\text{coul}}(q)$ 和 $A q$ 与当前“先建 $A$ 再求逆”的结果在数值上一致。
- On-site 中的 $\chi$、$J$、以及 $Q$ 的归一化（如 `system_charge / self.normalization_factor`）与 `_compute_q_eq` 中保持一致，避免同一构型得到不同的 $q_{\text{eq}}$。

---

## 5. 预期收益与适用场景

| 方面 | 当前 CACE 做法 | 采用 DP-QEq 式加速后 |
|------|----------------|----------------------|
| 内存 | 存 $N\times N$ 的 $A$ 和增广矩阵 | 不存 $A$，只存当前 $q$ 与 LBFGS 状态 |
| 每步求 $q$ | 构造 $A$ + 一次 $(N+1)\times(N+1)$ 求逆 | 若干次“$E(q)$ + $\nabla_q E(q)$”（每次一次 Ewald 调用），无求逆 |
| 大 $N$ | 构造与求逆都约 $O(N^2)$ 或更贵 | 单次 Ewald 与梯度约 $O(N)$ 或 $O(N\log N)$（视 Ewald 实现），总代价取决于 LBFGS 迭代次数 |

- **更适合**：$N$ 较大、希望省内存和减少显式矩阵操作时；或与 DP-QEq 保持同一套“能量+梯度+LBFGS”逻辑，便于对比与迁移。
- **需注意**：LBFGS 是迭代法，需要设定容差与最大迭代次数；用**上一步的 $q$ 作为初猜**（如 MD 中）可显著减少迭代次数；对小体系，直接求逆可能更简单、更稳。

---

## 6. 小结

- **可以用** DP-QEq 的加速思路在 CACE-SOG-Qeq 里做矩阵求解加速：不构造 $A$、不直接求逆增广矩阵，而是用“能量 + 梯度 + 投影梯度 + LBFGS”在约束 $\mathbf{1}^\top q = Q$ 下最小化 $E_{\text{QEq}}(q)$。
- **可行性**：CACE 的 `EwaldPotential.compute_potential_triclinic(r, q, cell, compute_field=True)` 对任意 $q$ 能给出库仑能量与 $A q$，足以在不建 $A$ 的前提下提供 LBFGS 所需的一阶信息。
- **实现要点**：为 ChargeEq 增加“能量+梯度”路径（仅用 Ewald 单次调用 + on-site）、投影梯度、以及 LBFGS（或其它一阶约束优化）；保持与现有 `norm_factor`、$\chi$、$J$、$Q$ 一致，即可在 CACE-SOG-Qeq 中复现 DP-QEq 式的矩阵加速。

若你希望，我可以再给出一份更贴近 `charge_eq.py` 的伪代码或补丁式修改建议（例如新方法名、与现有 `_compute_q_eq` 的切换方式等）。
