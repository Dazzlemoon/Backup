## CACE-SOG-Qeq2 中 Qeq+SOG 加速实现概览（对应 Qeq_A_matrix_gaussian_sum_acceleration.md 的 4.1 与 4.2）

本文说明：

- 在 `CACE-SOG-Qeq2` 中，如何在 `ChargeEq` 里实现 Qeq 线性系统的 **迭代求解** 与 **SOG 算子形式 `apply_A_sog`**（对应笔记 `Qeq_A_matrix_gaussian_sum_acceleration.md` 中的 4.1、4.2）。
- 相比旧版 CACE-SOG-Qeq 做了哪些改进。
- 后续如何用 `fit-4hdnnp-NaCl/NaCl.xyz` 做一个简单的 **时间对比 benchmark**，验证是否有加速效果。

---

## 1. 目标回顾：4.1 与 4.2 想做什么？

在 `Qeq_A_matrix_gaussian_sum_acceleration.md` 里：

- **4.1**：用 **迭代法（PCG/MINRES 等）替代 `torch.linalg.solve`**，避免对增广块矩阵做稠密分解。
- **4.2**：把 “显式构造 A 矩阵” 改为 **算子形式 `apply_A_sog(v)`**，只要能快速做 `v ↦ Av`，就能配合迭代法加速。

在 CACE-SOG-Qeq2 中，这两步的“最小可用版本”已经实现。

---

## 2. 在哪实现了 4.1 + 4.2？

核心代码在：

- 文件：`cace/modules/charge_eq.py`
- 类：`ChargeEq`

关键改动如下。

### 2.1 新增迭代求解开关与参数（4.1）

在 `ChargeEq.__init__` 中新增了三个参数：

- `use_iterative_solver: bool = False`：是否启用迭代法（PCG）求解 Qeq。
- `max_cg_iters: int = 200`：PCG 最大迭代步数。
- `cg_tol: float = 1e-6`：PCG 收敛容忍度。

并在实例上保存：

- `self.use_iterative_solver`
- `self.max_cg_iters`
- `self.cg_tol`

### 2.2 forward 分支：显式解 vs 迭代解

在 `ChargeEq.forward` 中，对每个 batch 内的构型，原来统一走 “构造 A → `_compute_q_eq` 用 `torch.linalg.solve`” 的路径；现在增加了一个分支：

- 当 **`use_sog_kernel and use_iterative_solver` 为 True** 时：
  - 不再显式构造 `A_now`。
  - 直接调用 `_compute_q_eq_iterative(...)`：
    - 内部通过 PCG 解约束问题（见 2.4）。
  - 长程能量通过算子形式计算：`A_q = self._apply_A_sog(r_now, q_eq, box_now)`，`ewald_energy = 0.5 * (q_eq * A_q).sum()`。

- 否则（保持向后兼容）：
  - 若 `use_sog_kernel`：用 `_build_A_sog` 显式构造 `A_sog`。
  - 否则：从 `EwaldPotential.compute_potential_triclinic` 拿到 `A_now`。
  - 再用 `_compute_q_eq` + `torch.linalg.solve` 求解增广块系统。

这样：

- **老接口**（不启用迭代求解）不会受影响；
- **新接口**（SOG+迭代）走的是 “算子 + PCG” 的逻辑。

### 2.3 `_apply_A_sog`：SOG 的算子形式 Av（4.2 的第一步）

函数：`ChargeEq._apply_A_sog(self, r, v, cell) -> torch.Tensor`

含义：

- 输入：原子位置 `r`，向量 `v`，晶胞 `cell`。
- 输出：`Av`，其中
  - `A_ij = K(|r_i - r_j|)`，`K` 为 SOG 核。

当前实现仍然是 **O(N²)**：

- 内部依然构造了一个临时的 `A_sog`（逻辑几乎与 `_build_A_sog` 相同），随后做一次矩阵向量乘 `Av = A_sog @ v`。
- 不过：
  - **A 不再保存在模块状态中**（只在函数内部构造后立即乘完丢弃）。
  - 外部只依赖 `apply_A_sog(v)` 这一接口，后续要改成 FFT/NUFFT/FMM 形式时，只需替换函数体。

也就是说，4.2 中“把 `_build_A_sog` 改成 `apply_A_sog(v)`”的接口层工作已经完成，接下来可以进一步把内部从显式 O(N²) 改成更快的 NUFFT/FMM 版本。

### 2.4 `_pcg` + `_compute_q_eq_iterative`：PCG + Schur 消元（4.1）

新增的两个核心函数：

- `_pcg(matvec, b, M=None, tol, max_iter)`：
  - 完整的 **PCG 实现**，完全在 torch 中，可微。
  - `matvec`：封装 `M v`（这里 `M = A + J`）。
  - `M`（大写参数）：预条件器 `P^{-1}v`，目前用简单的 Jacobi 预条件（`P ≈ J`）。

- `_compute_q_eq_iterative(A_mat, r, cell, chi, J, system_Q)`：
  - 实现了笔记和 `SOG_Qeq_A_matrix_SOG_acceleration.md` 中描述的 **Schur complement 路线**：
    1. 记 \(M = A + J\)，约束为 \(1^\top q = Q\)。
    2. 先解两个子问题：
       - \(M u = 1\)
       - \(M v = -\chi\)
    3. 用
       - \(\lambda = (Q - 1^\top v) / (1^\top u)\)
       - \(q = v + \lambda u\)
    4. 其中所有 `M^{-1}` 的作用，都是通过 `_pcg` + `matvec` 实现的，而不显式构造/分解 M。
  - 对 `matvec` 的实现：
    - 若 `A_mat` 非空：`M v = A_mat @ v + J_vec * v`（用于调试/对比）。
    - 若 `A_mat` 为空：`M v = self._apply_A_sog(r, v, cell) + J_vec * v`（真正 4.1+4.2 组合的路径）。
  - 预条件器：`precond(v) = v / (J_vec + 1e-8)`（Jacobi）。

综上，**4.1 的“用迭代法解 Mq=b”已经通过 `_pcg`+`_compute_q_eq_iterative` 落地，且可与 4.2 中的 `apply_A_sog` 组合使用。**

---

## 3. 相比旧版 CACE-SOG-Qeq 的改进点总结

与旧版（无 Qeq 加速）相比，CACE-SOG-Qeq2 中的 `ChargeEq` 主要改进有：

- **新增迭代求解模式**：
  - 通过 `use_iterative_solver=True`，可以在保持接口不变的前提下，切换到 PCG+算子模式。
  - 保留 `torch.linalg.solve` 分支，方便对比/回退。
- **A 从“长期存在的矩阵”变成“临时算子”**：
  - `_apply_A_sog` 给出了统一的 `v ↦ Av` 接口。
  - 实现上仍然是 O(N²)，但 A 不再被长时间存储，有利于后续替换为 NUFFT/FMM 实现。
- **Schur 消元 + PCG 结构清晰**：
  - 逻辑与笔记中的推导一致，形式标准，方便在更多体系上复用。
- **预条件器接口已就位**：
  - 目前用简单的 Jacobi 预条件器，后续可以改进（Block-Jacobi、近邻截断等），无需改 PCG 主体。

在复杂度层面：

- 当前 `_apply_A_sog` 仍是 O(N²)，PCG 每步也是 O(N²)，但：
  - 避免了 O(N³) 的稠密分解。
  - 在中等规模体系上，如果 PCG 的迭代步数不大（几十步内收敛），整体成本将明显优于直接 `torch.linalg.solve`。
  - 为后续引入 NUFFT 加速打开了接口。

---

## 4. 后续：如何用 NaCl.xyz 做时间加速测试？

下面是一种建议的 benchmark 路线，可以基于现有的 NaCl 脚本/数据来验证迭代求解是否加速。

### 4.1 基本思路

选取相同的 NaCl 数据集（如 `fit-4hdnnp-NaCl/NaCl.xyz`），对比：

1. **直接解版本（baseline）**：
   - `use_sog_kernel=True, use_iterative_solver=False`。
   - 即 SOG 近似 + 显式构造 A + `torch.linalg.solve`。
2. **迭代解版本（CACE-SOG-Qeq2 改进）**：
   - `use_sog_kernel=True, use_iterative_solver=True`。
   - 使用 `_apply_A_sog` + `_pcg` + Schur 消元。

在两个版本中：

- 其余网络结构、批大小、epoch 配置保持一致。
- 记录每个 epoch 的 wall-clock 时间（或训练前几百个 batch 的平均 step 时间）。
- 记录最终的能量/力 RMSE，用于确保精度相近。

### 4.2 可以如何在 NaCl 上落地

例如在 `CACE-SOG-Qeq2/fit-4hdnnp-NaCl/` 下新建两个脚本：

- `fit-cace-SOG-direct.py`：
  - 与当前 CACE-SOG-Qeq 的 NaCl 训练脚本类似。
  - `ChargeEq(..., use_sog_kernel=True, use_iterative_solver=False)`。
- `fit-cace-SOG-iter.py`：
  - 只改动 `ChargeEq` 构造行：
    - `ChargeEq(..., use_sog_kernel=True, use_iterative_solver=True, max_cg_iters=K, cg_tol=tol)`。
  - 其余完全相同。

在训练循环中：

- 用 Python 的 `time` 或 `datetime` 记录每个 epoch 的用时。
- 最后把时间和 RMSE 写入 csv 或打印到日志中，方便对比。

这样，你就可以定量回答：

- 在当前 NaCl 体系规模下，**PCG+算子 vs 直接 `solve`** 的时间比例；
- PCG 所需迭代步数大致多少（可以在 `_pcg` 中加简单的计数/统计）。

### 4.3 后续进一步加速的空间

在上述基础上，如果迭代解在中大体系上已经有明显加速，还可以：

- 把 `_apply_A_sog` 的内部实现替换为 **SOG+FFT/NUFFT**（参考 `CACE-SOG` 中的 `SOGPotential` 和 `SOG_FFT_NUFFT_matvec_math.md`）；
- 或引入更强的预条件器（例如局部块分解）以减少 PCG 步数。

这些都可以在不改变 `ChargeEq` 外部接口的前提下逐步实现。

---

## 5. 小结

- CACE-SOG-Qeq2 中已经实现了 `Qeq_A_matrix_gaussian_sum_acceleration.md` 中的 **4.1（迭代求解）** 和 **4.2（SOG 算子接口）** 的最小版本：
  - 新增 `use_iterative_solver` / `max_cg_iters` / `cg_tol`；
  - 提供 `_apply_A_sog` 算子接口；
  - 用 `_pcg` + `_compute_q_eq_iterative` 实现 Schur 消元 + PCG 解约束 Qeq。
- 当前 `_apply_A_sog` 仍是 O(N²) 的直接实现，但 A 不再长期存储，为 NUFFT/FMM 等进一步加速预留了接口。
- 可以通过构造 `direct` vs `iter` 两个 NaCl 训练脚本，直接在 `NaCl.xyz` 上比较时间与 RMSE，验证加速效果，并为后续频域/预条件改进提供基线。

