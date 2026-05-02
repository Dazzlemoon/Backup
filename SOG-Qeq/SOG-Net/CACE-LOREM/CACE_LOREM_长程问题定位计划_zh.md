# CACE-LOREM 长程学习问题定位计划

## 目标

定位以下核心问题，并给出可复现实验结论：

1. 为什么 CACE-LOREM 长程通道已开启，但 `F_long/F_total` 仍长期偏小。
2. 为什么提高 `lr_weight` 后，能量占比有变化，但角度信息学习仍弱。
3. 问题主要来自损失配置、读出链路缩幅，还是短程/长程梯度竞争。

## 阶段 0：先做短时冒烟测试（10-20 分钟）

- 使用 `fit-cumulene/fit-cace-LOREM-test.py --quick-test` 运行短测。
- 确认以下最小输出可正常生成：
  - `diagnostics_*.csv`
  - `loss*.csv`
  - `rmse*.csv`
- 重点检查短测中的：
  - `std_q_dipole`, `std_q_quadrupole`
  - `f_long_over_f_total`, `e_long_over_e_total`

判据：

- 若 `f_long_over_f_total` 仍接近 `1e-4~1e-3`，说明不是训练时长问题，存在结构性/标度问题。

## 阶段 1：排查“梯度竞争”是否主因

### 1.1 SR 冻结实验（优先）

- 冻结短程读出（或短程分支），只训练长程相关头部若干 epoch。
- 对比冻结前后：
  - `f_long_over_f_total` 是否显著上升
  - 二面角曲线是否更接近目标形状

判据：

- 若冻结后长程占比明显上升，说明共享骨干上的梯度竞争是主因之一。

### 1.2 梯度范数记录

- 记录每 N step 的梯度范数：
  - 长程头（`MultipoleChargeHead`、`LoremLongRangeReadout`）
  - 短程头（`Atomwise`）
  - 共享表示层（`CaceLoremShortRange`）

判据：

- 若长程头梯度长期显著小于短程头（例如小一个数量级以上），说明长程信号被淹没。

## 阶段 2：排查“读出链路缩幅”是否主因

### 2.1 逐层幅值追踪

- 增加日志，逐步记录以下张量的均值/标准差：
  - `q`
  - `q_dipole`
  - `q_quadrupole`
  - `q_field`
  - `lr_energy_atom`
  - `lr_energy`

判据：

- 若某一层开始出现显著缩小（如方差骤降），该层为重点改造对象。

### 2.2 `ewald_raw` 与 `lr_energy` 相关性

- 统计同 batch 下 `ewald_raw` 与 `lr_energy` 的相关系数。

判据：

- 相关性过低说明 `LoremLongRangeReadout` 未有效利用长程场信息。

## 阶段 3：排查“损失与权重”是否主因

### 3.1 损失扫参（小网格）

- 仅做小规模实验，网格如下：
  - `energy_weight`: `[0.5, 1.0, 2.0]`
  - `force_weight`: `[0.5, 1.0]`
  - `lr_weight`: `[1, 5, 10, 20]`

观测指标：

- `f_long_over_f_total` 是否能稳定进入 `>=1e-2`
- `std_q_dipole/std_q_quadrupole` 是否抬升并保持

### 3.2 检查“只放大输出”而非“增强可学习性”

- 对比两种策略：
  - 只加大 `lr_weight`
  - 加辅助损失（针对长程能量/长程力）

判据：

- 若只加权重效果有限，而加辅助损失明显改善，说明核心是监督路径不足。

## 阶段 4：与 LOREM 做对齐验证

- 对齐同批次统计：
  - `q0_mean/q0_std`
  - `q1_norm_mean`
  - `q2_norm_mean`
  - `sr_energy_sum/lr_energy_sum`
- 保持相同数据切分和评价口径，比较趋势而不是单点值。

判据：

- 若 CACE-LOREM 的 `q1/q2` 振幅长期明显低于 LOREM，同步印证角度信息不足。

## 交付物清单

每一阶段都应产生可复查文件：

- `diagnostics_*.csv`
- 一页图表（推荐 notebook）
- 一段结论（问题是否定位成功、下一步改动建议）

## 建议执行顺序

1. 阶段 0（短测确认）
2. 阶段 1（梯度竞争）
3. 阶段 2（链路缩幅）
4. 阶段 3（损失与权重）
5. 阶段 4（与 LOREM 对齐）

这样可以先排除“训练不够久”的伪问题，再进入真正的机制定位。

---

## 已落地：AB4-strict 默认训练方案（严格 LR-focus + SR 渐进回归 + 阶段门控日志）

对应脚本：`fit-cumulene2/fit-cace-LOREM-testAB4-strict.py`  
对应提交任务：`fit-cumulene2/run.slurm`

### 1) 默认 phase 结构（5 阶段）

非 `quick_test`、非 `sr_freeze_experiment` 模式下，默认采用：

1. `joint_warmup`（约 15% epochs）：`sr_weight=1.0`
2. `lr_focus_strict`（约 45% epochs）：`sr_weight=0.0` 且 `freeze_sr_readout=True`
3. `joint_return_0p1`（约 15% epochs）：`sr_weight=0.1`
4. `joint_return_0p3`（约 15% epochs）：`sr_weight=0.3`
5. `joint_return_0p5`（余下 epochs）：`sr_weight=0.5`

设计目的：

- 第 2 阶段尽可能接近 LR-only 学习条件，验证 LR 分支是否能稳定承载角度信息；
- 第 3-5 阶段逐步回归 SR，降低“一步回到 `sr_weight=1.0`”导致 LR 形状被冲掉的风险。

### 2) 阶段门控日志（Phase Gate）

在每个阶段结束时，脚本会基于该阶段最新 `DihedralDiag` 输出：

- `corr_long_vs_ref`
- `amp_long/ref`（`amp_pred_long_rel / amp_ref_rel`）
- `mean_Flong/Ftotal`

并给出 `[PhaseGate] ... => PASS/FAIL`。

当前默认门槛：

- `corr_long_vs_ref >= 0.70`
- `amp_long/ref >= 0.20`
- `mean_Flong/Ftotal >= 0.10`

说明：门槛用于定位和告警，不会自动中断训练；用于快速判断“当前阶段是否达成 LR 形状承载目标”。

### 3) `run.slurm` 已同步

`run.slurm` 已改为运行 `fit-cace-LOREM-testAB4-strict.py`，并统一输出目录到：

- `loss_data/CACE_LOREM_MP2_ABtest4_strict`

训练结束后 `DihedralSummary` 也从同一目录读取，避免目录错配。
