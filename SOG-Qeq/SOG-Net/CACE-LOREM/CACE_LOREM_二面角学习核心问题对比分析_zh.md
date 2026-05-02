# CACE-LOREM 二面角学习核心问题对比分析（先不改代码）

## 1. 问题结论（先给结论）

当前 CACE-LOREM 的长程分量在 cumulene 二面角任务中更像“常数偏置项”，而不是“承载势能面形状的主项”。  
这不是单一超参数问题，而是**表示路径 + 回注方式 + 梯度竞争**共同造成的结果。

从现象看：

- `E_long` 可以随 `lr_weight` 放大，但 `F_long/F_total` 仍偏小。
- 二面角曲线总振幅很小、相关性低，且 LR 分量曲线更平坦。
- `q1/q2`（偶极/四极）整体幅值与结构依赖性不足，难以形成稳定角向响应。

---

## 2. LOREM 与 CACE-LOREM 的关键实现差异（代码层）

### 2.1 多极电荷读出路径

LOREM（`LOREM/lorem/lorem.py`）：

- `q0`：由标量分支 MLP 直接读出。
- `q(l>0)`：由 `nodes_spherical` 经 `TensorDense` 直接线性投影读出（保持角向通道的线性可混合）。

CACE-LOREM（`cace/modules/lorem_longrange.py` 的 `MultipoleChargeHead`）：

- `q0`：由 `p_features` 的 MLP 读出。
- `q1/q2`：`s_l1/s_l2` 先被 `gate(p)` 缩放，再在径向与通道维求和得到。

影响：

- `gate * sum` 路径的表达更“受限”，容易把角向信息压成小幅值输出。
- 对角向分量的可学习混合能力弱于 LOREM 的直接线性投影。

### 2.2 长程回注（LR feedback）信息保留方式

LOREM：

- 先算 LR potential，再与球谐张量特征做 Tensor 交互，之后进入标量更新与能量头。
- 路径更接近“角向张量 -> 张量交互 -> 标量更新”。

CACE-LOREM（`LoremLongRangeReadout`）：

- `q_field` 与 `s_l0/s_l1/s_l2` 相互作用后，对每个 `l` 分支做角向范数（norm）压缩，再拼接进 `energy_head`。

影响：

- 角向符号/相位信息会被过早标量化，LR 分支更容易退化为平滑偏置。

### 2.3 力监督中的有效梯度占比

在组合模型中（`fit-cumulene/fit-cace-LOREM-test-noimprove.py` + `cace/models/combined.py`）：

- 总能量/总力来自 SR 与 LR 的加权求和。
- 当 SR 分支更强时，LR 分支虽“开着”，但在力损失中的有效梯度占比仍可能长期偏低。

影响：

- 即使提高 `lr_weight`，若 LR 表示能力不足或回注丢角向信息，学习结果仍可能是“放大后的平偏置”。

### 2.4 Ewald 模块实现一致性风险（结构性风险点）

在 `cace/modules/ewald.py` 中存在重复定义与返回形状分支差异（尤其 `q_field` 维度处理）。  
非周期体系多数走 realspace 路径，问题不一定立即显现，但这是潜在不稳定源。

---

## 3. 为什么会“长程通道开了但学不到二面角”

可归纳为三个层面的瓶颈叠加：

1. **可表达性瓶颈**：`q1/q2` 读出路径对角向信息的线性混合能力不足。  
2. **信息保真瓶颈**：LR 回注时过早 norm 化，导致角向细节在进入能量头前被压缩。  
3. **优化动力学瓶颈**：SR 对总力主导，LR 有效梯度长期偏小。

因此表现为：

- LR 能量绝对值可大，但对二面角变化的响应斜率小（`F_long` 小）。
- LR 曲线“抬升了基线”，却没学到“形状”。

---

## 4. 不改代码前提下的“可执行诊断方案”（先验证再改）

你已在评估 notebook 增加逐结构导出，这是正确方向。建议固定执行以下检查：

1. **逐二面角结构导出** `q0/q1/q2`（含 per-atom 与 structure summary）。
2. **导出分解曲线**：`E_total / E_short / E_long` 对二面角曲线。
3. **导出力占比曲线**：`F_long/F_total` 随二面角与训练 epoch 的变化。
4. **看“振幅而非均值”**：对每个量计算 `max-min` 振幅，判断是否承载形状。

判据（经验）：

- 若 `amp(E_long)` 远小于 `amp(E_total)` 且相关性低，说明 LR 未承载形状；
- 若 `q1/q2` 对二面角变化振幅接近噪声级，说明角向电荷本体未学成。

---

## 5. 可在 CACE-LOREM 中优先考虑的改造方向（先方案，后实现）

以下是“先设计，不动代码”的建议清单，按优先级排序。

### 方向 A（最高优先级）：把 `q1/q2` 读出改为“线性投影型”

目标：

- 从 `s_l1/s_l2` 到 `q1/q2` 的映射由 gate-sum 改为可学习线性投影（保留角向分量）。

预期收益：

- 增强角向信息表达能力，提高 `q1/q2` 的结构依赖振幅。

风险：

- 幅值可能骤增，需要配套正则或初始化约束。

### 方向 B（高优先级）：减少 LR 回注中的过早标量化

目标：

- 在 `LoremLongRangeReadout` 中，避免过早仅保留角向 norm；
- 保留更多角向分量信息进入后续能量头（或分支并联：norm 路径 + 角向路径）。

预期收益：

- LR 分支更容易学习二面角形状而非仅常数偏置。

风险：

- 参数量上升，训练稳定性与正则需求提高。

### 方向 C（中高优先级）：优化 LR 梯度预算，而非只放大 `lr_weight`

目标：

- 通过阶段训练/损失重加权，使 LR 在关键阶段拿到足够梯度；
- 避免 SR 一直“抢占”全部力监督。

预期收益：

- `F_long/F_total` 更可控地提升，并能转化为曲线形状改进。

风险：

- 过强 LR 训练可能牺牲整体 MAE，需要阶段性权重回调。

### 方向 D（中优先级）：统一 Ewald 接口与形状契约

目标：

- 清理重复定义，统一 `q_field` 的约定形状；
- 确保 periodic/non-periodic 分支在下游 readout 语义一致。

预期收益：

- 降低隐藏 bug 与路径差异导致的训练不确定性。

---

## 6. 建议的最小 A/B 验证路线

建议先定义三组对照目标（后续实现时按此验收）：

- **Baseline**：现有 CACE-LOREM（当前 noimprove 配置）。
- **A-only**：仅替换 `q1/q2` 读出路径。
- **A+B**：替换读出 + LR 回注保角向信息。

每组固定比较：

- 二面角曲线：`corr(pred, ref)`、`amp(pred)`、`amp(E_long)`；
- 训练诊断：`q1_norm_amp`、`q2_norm_amp`、`F_long/F_total`；
- 常规指标：`U_MAE`, `F_MAE`（防止整体退化）。

验收原则：

- 不是只看 loss 降低，而是看 **LR 曲线是否真正承载二面角形状**。

---

## 7. 一句话总结

当前主要矛盾不是“长程没开”，而是“**长程表达与回注方式把角向信息压弱了**”，再叠加 SR 梯度竞争，最终让 LR 学成了偏置项。  
优先改造 `q1/q2` 读出与 LR 回注信息保真路径，是解决“学不到二面角信息”的最可能主线。

---

## 8. 本次已完成的方向 A 改造与一致性评估

### 8.1 已完成改造（代码已修改）

已在 `cace/modules/lorem_longrange.py` 的 `MultipoleChargeHead` 中完成方向 A：

- 将 `q1/q2` 的读出从原来的 `gate(p) * s_l` 再求和，改为**直接线性投影读出**：
  - `q1 = Linear(flatten(s_l1)) -> 3`
  - `q2_raw6 = Linear(flatten(s_l2)) -> 6`，再通过固定 detrace 映射到 5 分量 `q2`
- 为控制改造初期幅值突增，增加了保守稳定策略：
  - `Linear` 使用小增益 Xavier 初始化（`gain=0.1`）
  - 增加可学习输出缩放参数（`tanh(scale)` 形式）约束初始幅值

这对应你提出的风险控制要求（“幅值可能骤增，需要初始化约束”）。

### 8.2 改完后与 LOREM 的一致性结论

结论：**更接近 LOREM，但还不是完全一致**。

一致的部分：

- 都从 `l>0` 的角向特征中直接读出多极（不再依赖 scalar gate-sum）。
- 都是“可学习线性映射到多极通道”。

仍不一致的部分：

- LOREM 使用球谐张量域上的 `TensorDense`；当前 CACE-LOREM 是 Cartesian 分组 `s_l1/s_l2` 的扁平线性层。
- LOREM 的 LR 回注仍是张量交互主导；当前 CACE-LOREM 在 `LoremLongRangeReadout` 里仍有按角向 `norm` 的标量化压缩。

因此，方向 A 能显著改善“角向电荷读出能力”，但若要达到 LOREM 在二面角形状承载上的效果，通常还需要继续推进方向 B（回注信息保真）和方向 C（梯度预算）。

---

## 9. 本次已完成的方向 B 改造与一致性评估

### 9.1 已完成改造（代码已修改）

已在 `cace/modules/lorem_longrange.py` 的 `LoremLongRangeReadout` 中完成方向 B：

- 保留原有 `norm` 路径（`s0/s1/s2` 的角向范数标量特征）；
- 新增**角向分量并联路径**（directional branch）：
  - 对 `s1_updates/s2_updates` 仅在 channel 维做均值，保留角向分量符号与结构变化；
  - 将 `s1_dir/s2_dir` 与 `mono + norm` 特征并联拼接后一起进入 `energy_head`。
- 增加稳定措施：
  - 角向并联分支加入可学习缩放（`tanh(scale)`）并以小值初始化，降低训练初期震荡。

换言之，当前 LR 回注已从“仅 norm 标量化”升级为“**norm + directional 并联**”。

### 9.2 与 LOREM 的区别（改完 B 之后）

改完 A+B 后，与 LOREM 的关系是：**在目标上更接近，在数学形式上仍不完全一致**。

更接近的点：

- 都不再完全依赖早期标量化；
- 都允许角向信息更直接地影响 LR 能量读出。

仍存在的关键差异：

- LOREM 的回注是球谐张量域中的 `Tensor` 交互（更“结构等变原生”）；
- 当前 CACE-LOREM 仍是 Cartesian 分组特征上的工程化并联拼接（directional + norm），不是同构的球谐张量操作。

因此，A+B 是“朝 LOREM 行为靠近”的强改进，但不是严格同构复现。

### 9.3 是否“已经能够学到角度信息”？

结论：**更有可能学到，但不能仅凭结构修改直接判定“已学到”**。  
必须通过训练后诊断数据确认，重点看：

- `amp(E_long)` 是否显著提升并与二面角变化相关；
- `corr(E_long_curve, ref_curve)` 是否明显改善；
- `q1/q2` 对二面角的振幅与可分辨性是否提升；
- `F_long/F_total` 是否从“近零”上升到可解释区间，且不破坏 `U_MAE/F_MAE`。

若这些指标同步改善，才能认为“角度信息确实被 LR 分支学到”。

---

## 10. 本次新增改动：去掉 `q1/q2` 的 `tanh(scale)` 并恢复线性头标准增益

### 10.1 已完成改动（代码已修改）

已在 `cace/modules/lorem_longrange.py` 的 `MultipoleChargeHead` 中完成以下调整：

- 去掉 `q1/q2` 上的 `tanh(scale)` 限幅：
  - 由 `q = tanh(scale) * Linear(...)` 改为 `q = softplus(scale_raw) * Linear(...)`。
- `l1_output_scale` / `l2_output_scale` 保留为可学习参数，但改为 softplus 参数化：
  - 采用 `scale = softplus(scale_raw)`，保证缩放始终为正且不受 `[-1,1]` 上界限制。
  - 初始化为 `softplus(scale_raw)=1.0`（通过 `scale_raw=log(expm1(1.0))` 实现），使初始幅值不再被额外压小。
- 将 `l1_proj_head` / `l2_proj_head` 的 Xavier 初始化增益从 `0.1` 恢复到 `1.0`。

### 10.2 动机与预期影响

这次改动的核心目标是：减少高阶多极（尤其 `q1/q2`）在训练早期被过度抑制，给 LR 分支分配到更可见的梯度。

此前路径里存在“双重保守缩放”：

- 线性头权重采用 `gain=0.1`；
- 输出再乘 `tanh(scale)`（初值 `scale=0.1` 时 `tanh(0.1)≈0.1`）。

两者叠加会显著缩小 `q1/q2` 初始有效幅值，容易导致：

- `q1/q2` 长期停留在极小量级；
- `E_long` 更像平移偏置；
- `F_long/F_total` 难以上升到可解释区间。

改为 `softplus(scale_raw)` 且初值为 1.0 后：

- 不再有人为 `tanh` 上界限制；
- 不再在初期额外压低高阶多极通道；
- 与 LOREM 的“直接线性读出高阶通道”行为更接近（虽数学实现仍非完全同构）。

### 10.3 与“像 LOREM 一样让 LR 占一定梯度”的关系

该改动主要作用于“多极读出幅值可达性”，即为 LR 梯度流创造必要条件。  
但它本身不保证最终一定通过二面角验收，仍需配合第 9 节的诊断判据看训练结果是否同步改善：

- `amp(E_long)` 与相关性是否提升；
- `q1/q2` 对二面角的振幅是否脱离噪声级；
- `F_long/F_total` 是否持续上升且不破坏总体 `U_MAE/F_MAE`。

若上述指标仍无明显改善，再考虑继续推进方向 C（阶段式梯度预算：如短程冻结/权重调度）会更稳妥。

---

## 11. 方向 C 的默认落地：`fit-cace-LOREM-testAB3.py` 三阶段训练（已实现）

### 11.1 默认 `phase_plan` 已改为三阶段

已在 `fit-cumulene2/fit-cace-LOREM-testAB3.py` 将默认（非 `quick_test`、非 `sr_freeze_experiment`）训练流程改为：

1. `joint_warmup`：SR+LR 联合训练（`freeze_sr_readout=False`, `sr_weight=1.0`）
2. `lr_focus`：冻结短程 readout，压低短程权重（`freeze_sr_readout=True`, `sr_weight=0.1`）
3. `joint_finetune`：恢复联合微调（`freeze_sr_readout=False`, `sr_weight=1.0`）

默认 epoch 按 `max_epochs` 自动分配为约 `20% / 60% / 20%`，并在脚本中保证每段至少 1 epoch。

### 11.2 训练意图

该默认三阶段直接对应“让 LR 分到足够梯度预算”的目标：

- 第 1 段稳定整体收敛与基线；
- 第 2 段通过冻结 SR 读出并降低 `sr_weight`，避免 SR 长期抢占力监督；
- 第 3 段再恢复联合，以降低只训 LR 造成的全局 MAE 退化风险。

### 11.3 `run.slurm` 已同步到 AB3

`fit-cumulene2/run.slurm` 已同步更新：

- 训练脚本改为 `fit-cace-LOREM-testAB3.py`；
- 默认保存目录改为 `loss_data/CACE_LOREM_MP2_ABtest3`；
- `job-name` / `output` / `error` 文件名改为 AB3；
- 训练后 `DihedralSummary` 汇总脚本读取路径改为同一 AB3 目录，避免“训练输出和汇总目录不一致”。

### 11.4 预期验收信号（AB3）

若三阶段有效，通常应看到：

- 第 2 段期间 `mean(F_long/F_total)` 比 `joint_warmup` 有可见上升；
- `amp_pred_long_rel` 与 `corr_long_vs_ref` 相对单阶段训练提升；
- `q1/q2` 的统计量（如 `mean_q1_norm`, `mean_q2_norm`）不再长期贴近噪声级。

若 AB3 仍未改善，再考虑在训练损失中加入 profile 监督（而非仅诊断）会更直接。

