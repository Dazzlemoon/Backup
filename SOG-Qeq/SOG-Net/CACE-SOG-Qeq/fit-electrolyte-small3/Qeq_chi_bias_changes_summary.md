## Qeq 电荷“快速变大”改动总结（CACE-SOG-Qeq）

本文件总结了为了解决“Qeq 解出的 $q_{eq}$ 幅度过小（接近 0）”而做的代码改动。核心思路是：在进入 `ChargeEq` 之前，对逐原子电负性特征 $\chi_i$ 增加 **按元素的可训练偏置**（并可选加全局缩放），让元素之间的 $\Delta\chi$ 在训练初期就足够大，从而让线性系统的解 $q$ 更容易达到更大幅度。

---

## 1. 背景：为什么要拉大 $\Delta\chi$

Qeq 的解可粗略理解为线性系统

$$
Aq = b
$$

其中：
- $A$ 的主要尺度来自硬度（对角项）$J$ 以及远程核；
- $b$ 的有效驱动力来自 $\chi$ 的相对差异（整体平移可被约束项吸收）。

当元素间电负性差 $\Delta\chi$ 很小、而 $J$ 的量级不小（例如 $J\sim 2$–$5$）时，就会自然得到很小的电荷幅度：

$$
q \sim \frac{\Delta\chi}{J_{\mathrm{eff}}}.
$$

因此我们通过“元素偏置 + 缩放”的方式，让 $\Delta\chi$ 在训练初期变大，从而 **更快** 得到更大的 $q_{eq}$。

---

## 2. 新增模块：`ElementwiseFeatureBias`

### 2.1 新文件

- 新增：`CACE-SOG-Qeq/cace/modules/elementwise_bias.py`
- 并在：`CACE-SOG-Qeq/cace/modules/__init__.py` 追加
  - `from .elementwise_bias import *`

### 2.2 模块功能

对逐原子特征做以下变换：

$$
\chi_i^{(\mathrm{biased})} = s\,\chi_i + b_{Z_i},
$$

其中：
- $b_Z$：每个元素一个可训练参数（`nn.Parameter` 向量）；
- $s$：全局可训练缩放（可开关 `use_scale`）；
- 可选：将偏置做去均值（`zero_mean_bias=True`）：

$$
b_Z \leftarrow b_Z - \frac{1}{N_\mathrm{elem}}\sum_{Z'} b_{Z'}.
$$

实现要点：
- 用 `atomic_numbers` 将每个原子映射到 `elements=[...]` 的索引；
- 输出写入 `data[output_key]`，默认只新增该 key，不破坏原始 `chi`。

---

## 3. 训练脚本接入（fit-electrolyte-small2）

### 3.1 修改的脚本

- 修改：`CACE-SOG-Qeq/fit-electrolyte-small2/fit-cace-Qeq-SOG.py`

### 3.2 接入方式

1) `chi` 头部仍输出逐原子 `chi`（key=`"chi"`）。

2) 新增 `chi_bias` 模块，把 `"chi"` 变换为 `"chi_biased"`：

- `feature_key="chi"`
- `output_key="chi_biased"`

3) 将 `ChargeEq` 的 `feature_key` 从 `"chi"` 改为 `"chi_biased"`：

- 旧：`feature_key="chi"`
- 新：`feature_key="chi_biased"`

4) 将 `chi_bias` 插入 `NeuralNetworkPotential.output_modules`，位置在 `chi` 之后、`ChargeEq` 之前：

- 旧（示意）：`[..., chi, system_charge_from_q, charge_eq, ...]`
- 新（示意）：`[..., chi, chi_bias, system_charge_from_q, charge_eq, ...]`

---

## 4. 为了“快”，当前采用的初始化（可按需调整）

为了让训练一开始就产生更大的 $\Delta\chi$，在 `fit-cace-Qeq-SOG.py` 中对 `chi_bias` 使用了偏激但可控的初始化：

- 偏置初始化（单位与 $\chi$ 一致；使用 H/O/F/K 的 Pauling 电负性）：

$$
b_H=2.20,\quad b_O=3.44,\quad b_F=3.98,\quad b_K=0.82
$$

- 缩放初始化：

$$
s = 5.0
$$

备注：
- 若出现 loss 震荡或 NaN，建议优先把 $s$ 从 $5.0$ 降到 $2.0$ 或 $1.0$；
- 若想进一步增大电荷幅度，可增大 $|b_Z|$ 或 $s$，但需要注意数值稳定性与泛化。

---

## 5. 预期效果与风险

### 5.1 预期效果

- 训练初期就能显著拉大 $\Delta\chi$，因此更容易得到更大的 $q_{eq}$ 幅度（不再坍缩到 $10^{-2}$）。

### 5.2 主要风险

- $\chi$ 被放大后，Qeq 线性系统的解 $q$ 可能变大，从而使长程能量/力的梯度变强，带来训练不稳定。
- 如果不加任何约束，偏置 $b_Z$ 可能漂移过大（虽然去均值能避免整体平移漂移，但仍可能产生过强分离）。

### 5.3 推荐的后续增强（可选）

为更稳健，可考虑加轻微正则：

$$
\mathcal{L}_b = \lambda_b \sum_Z b_Z^2,\qquad
\mathcal{L}_s = \lambda_s (s-1)^2,
$$

并从较小的 $\lambda_b,\lambda_s$（例如 $10^{-4}$–$10^{-2}$）开始尝试。

