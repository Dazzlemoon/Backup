# `fit-electrolyte-small3` 中 `chi` 设计与结果说明

本文总结 `fit-electrolyte-small3` 训练脚本中 `chi` 的设计逻辑、训练后学到的关键参数，以及当前结果是否合理。

---

## 1. 设计目标

在 Qeq 框架中，模型不会直接预测电荷，而是先预测每个原子的 `chi`（电负性特征），再通过线性方程在总电荷约束下求解 `q_eq`。  
因此本设计的目标是：

- 让 `chi` 保留正负信息（避免早期信息压缩）；
- 通过元素偏置和全局缩放，拉开元素间 `Delta chi`；
- 让 Qeq 求解得到更有分辨率的电荷分布。

---

## 2. `chi` 的数据流（small3 当前版本）

在 `fit-cace-Qeq-SOG.py` 中：

1. `Atomwise` 输出原始 `chi_raw`；
2. 设定 `post_process=None`，即
   - `chi = chi_raw`（有正有负，不再平方）；
3. 通过 `ElementwiseFeatureBias` 得到
   - `chi_biased = scale * chi + b_Z`；
4. `ChargeEq` 使用 `feature_key="chi_biased"` 进入 Qeq 方程求解。

简写为：

`chi_raw -> chi(=chi_raw) -> chi_biased -> ChargeEq`

---

## 3. 初始化参数（训练前）

### 3.1 全局缩放

- `init_scale = 5.0`

说明：这是初值，不是固定常数；训练中会被优化器更新。

### 3.2 元素偏置（raw bias 初值）

使用 Pauling 电负性常用值：

- H: 2.20
- O: 3.44
- F: 3.98
- K: 0.82

对应配置：

- `init_bias={1:2.20, 8:3.44, 9:3.98, 19:0.82}`
- `zero_mean_bias=True`

因此前向中实际使用的是去均值偏置：

`b_eff = b_raw - mean(b_raw)`

---

## 4. 训练后读到的关键参数（你当前输出）

你在 notebook 中打印得到：

- `final scale: 3.69958758354187`
- `raw bias: [1.541018, 3.6757243, 3.6538665, 1.2603487]`（顺序 `[H,O,F,K]`）
- `effective bias used in forward: [-0.9917214, 1.1429849, 1.1211271, -1.2723907]`

按元素展开：

- H(Z=1): raw=1.54102, used=-0.991721
- O(Z=8): raw=3.67572, used=+1.14298
- F(Z=9): raw=3.65387, used=+1.12113
- K(Z=19): raw=1.26035, used=-1.27239

解释：

- `scale` 从 5.0 学到约 3.70，说明模型自动降低了放大量级；
- O/F 的有效偏置为正，H/K 为负，元素分组清晰。

---

## 5. 当前 `chi` 分布结果（你提供的统计）

- `chi_biased min/max: -1.2999 / 1.5023`
- `chi_raw min/max: -0.0833 / 0.1822`

按元素均值（valid）：

- H: `chi_biased` 约 -1.119；`chi_raw` 约 -0.034
- O: `chi_biased` 约 +1.296；`chi_raw` 约 +0.041
- F: `chi_biased` 约 +1.314；`chi_raw` 约 +0.052
- K: `chi_biased` 约 -0.908；`chi_raw` 约 +0.098

结论：`chi_raw` 幅度较小但可正可负，经过 `scale + bias` 后 `chi_biased` 被拉开到约 `[-1.3, 1.5]`，符合“增强元素可分性”的设计意图。

---

## 6. 结果是否合理

从建模目标看，当前结果总体合理：

- `chi_biased` 元素分布明显分离（说明 bias/scale 工作正常）；
- `scale` 训练后收敛到中等值（未保持过高初值，通常更稳）；
- 进入 Qeq 的输入信号强度比 `chi_raw` 明显更可用。

但要判断“物理上是否更优”，仍需结合以下指标：

1. 按元素的 `q_eq` 统计（均值/分位数/符号）；
2. 验证集能量和力误差是否优于基线；
3. 是否出现训练不稳定（loss 震荡、NaN、梯度爆炸）。

建议最小验证清单：

- 比较 small2 vs small3 的 `val_e/atom_rmse`、`val_f_rmse`；
- 比较 `q_eq` 的按元素分布是否更符合体系化学直觉；
- 检查不同随机种子下结果是否稳定。

---

## 7. 常见误解说明

- “raw bias 为什么不是 5？”  
  这是两个不同参数：
  - `5.0` 是 `scale` 的初始值；
  - `raw bias` 是按元素偏置，初始为 Pauling 值并在训练中更新。

---

如果后续继续迭代该方案，建议把 `scale` 和 `bias` 的训练轨迹（每 N epoch 打印一次）也记录到日志，便于判断收敛和稳定性。

