# ChargeEq `normalization_factor` 说明

本文说明 `CACE-SOG-Qeq` 中 `ChargeEq.normalization_factor` 的数学含义，以及为什么在评估时 `scaled q_eq` 才与数据中的 `q_ref` 同尺度。

## 1. 背景

在 Qeq 中，目标是求平衡电荷向量 `q`，常见优化问题为：

$$
\min_{\mathbf q}\;E(\mathbf q)
= \frac12 \mathbf q^\top \mathbf A \mathbf q
+ \frac12 \sum_i J_i q_i^2
+ \sum_i \chi_i q_i
$$

并满足总电荷约束：

$$
\sum_i q_i = Q_{\text{tot}}
$$

其中：
- `A`：由 Ewald/SOG 核构造的相互作用矩阵；
- `J_i`：元素硬度；
- `chi_i`：网络预测的电负性项；
- `Q_tot`：体系总电荷。

## 2. 代码中 `normalization_factor` 的作用

在当前实现中，`ChargeEq` 内部不是直接用物理总电荷 `Q_phys`，而是使用缩放后的约束右端：

$$
Q_{\text{int}}=\frac{Q_{\text{phys}}}{f},\quad f=\text{normalization\_factor}
$$

对应代码逻辑（`charge_eq.py`）：

- `self.normalization_factor = norm_factor`
- `Q_tot = system_Q / self.normalization_factor`

因此内部求解满足：

$$
\sum_i q_i^{(\text{int})}=Q_{\text{int}}=\frac{Q_{\text{phys}}}{f}
$$

## 3. 线性方程形式

引入拉格朗日乘子 `lambda` 后，求解的线性系统可写为：

$$
\begin{bmatrix}
\mathbf A+\mathrm{diag}(\mathbf J) & \mathbf 1\\
\mathbf 1^\top & 0
\end{bmatrix}
\begin{bmatrix}
\mathbf q^{(\text{int})}\\
\lambda
\end{bmatrix}
=
\begin{bmatrix}
-\boldsymbol\chi\\
Q_{\text{phys}}/f
\end{bmatrix}
$$

这正是代码中 `_compute_q_eq(...)` 的数学表达。

## 4. 为什么 `scaled q_eq` 才接近 `q_ref`

模型输出的 `q_eq` 是内部变量 `q^(int)`，而数据文件中的 `q_ref` 是物理尺度电荷 `q^(phys)`。两者关系为：

$$
\mathbf q^{(\text{phys})} \approx f\,\mathbf q^{(\text{int})}
$$

所以评估时：
- `raw`：直接比较 `q_ref` 与 `q_eq`（通常会偏大）；
- `scaled`：比较 `q_ref` 与 `q_eq * normalization_factor`（应显著更接近）。

## 5. 结合本次 NaCl 实验的现象

你的输出显示：
- `mean(q_eq_sum / q_ref_sum) ≈ 9.48933`
- `1 / normalization_factor ≈ 9.48933`

两者几乎完全一致，说明当前模型确实满足：

$$
q_{\text{eq}} \approx \frac{q_{\text{ref}}}{f}
$$

因此：
- `raw q_eq` 主要用于模型内部求解与能量计算；
- `scaled q_eq = q_eq * f` 才是与 extxyz `q_ref` 同单位、同尺度的对比量。

## 6. 结论

`normalization_factor` 不是“额外修补项”，而是 Qeq 内部单位/尺度一致性的核心因子。  
当比较电荷精度时应明确比较对象：

- 关心模型内部求解变量：看 `q_eq`（raw）；
- 关心与数据标注电荷的一致性：看 `q_eq * normalization_factor`（scaled）。

## 7. `90.0474` 这个因子在项目里怎么“算出来”

从本项目代码角度，最直接、可复现实的定义是：

$$
C \equiv \frac{1}{f^2},\quad f=\text{normalization\_factor}
$$

也就是：

$$
f=\sqrt{\frac{1}{C}}
$$

在默认实现中（`charge_eq.py`）：

$$
f=(1/90.0474)^{1/2}
$$

所以反推：

$$
C=\frac{1}{f^2}=90.0474
$$

这说明 `90.0474` 是当前 Qeq/Ewald 实现所采用的“内部等效库仑前因子”常数，`normalization_factor` 则是与之配套的电荷缩放系数。

---

若和常见分子模拟常数比较（`eV`、`Å`、电荷单位 `e`），常见两体库仑写法为：

$$
E_{ij}=k_{\mathrm{eV\cdot\AA}}\frac{q_i q_j}{r_{ij}},\quad
k_{\mathrm{eV\cdot\AA}}\approx 14.3996
$$

而本项目内部常数满足近似关系：

$$
90.0474 \approx 2\pi \times 14.33
$$

数值上与常见库仑常数只差一个与实现约定（Ewald/Fourier 归一化、核定义、单位吸收方式）相关的固定倍率。  
因此在本代码里应以 `ChargeEq` 的实际定义为准：`C=90.0474`、`f=(1/C)^{1/2}`，并保持训练/评估一致。

