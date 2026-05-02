# CACE-LOREM AB 系列训练脚本与日志对应总览

本文总结以下 5 个脚本的改进点与差异，并给出当前 `fit-cumulene2` 目录中已有日志文件和脚本对应关系：

- `fit-cace-LOREM-testAB.py`
- `fit-cace-LOREM-testAB2.py`
- `fit-cace-LOREM-testAB3.py`
- `fit-cace-LOREM-testAB4-strict.py`
- `fit-cace-LOREM-testAB-try.py`

---

## 1. 各脚本改进脉络（按演进顺序）

### A. `fit-cace-LOREM-testAB.py`

- **定位**：AB 系列的基础版（单阶段默认训练）。
- **默认训练**：
  - `DEFAULT_MAX_EPOCHS = 500`
  - `DEFAULT_START_LR = 1e-3`
  - 默认 `phase_plan` 实际是单阶段 `stage_1`（`sr_weight=1.0`, `freeze_sr_readout=False`）。
- **已有能力**：
  - 支持 `--sr-freeze-experiment` 两阶段诊断模式（baseline + sr_frozen），但不是默认流程。
  - 具备 `DihedralDiag` 与 LR 分解诊断导出。
- **默认保存目录**：`loss_data/CACE_LOREM_MP2_ABtest`

### B. `fit-cace-LOREM-testAB2.py`

- **定位**：AB.py 的并行版本（主要用于另一组作业/目录隔离）。
- **与 AB.py 的关系**：训练逻辑基本一致，仍是单阶段默认训练 + 可选 `sr_freeze_experiment`。
- **关键差异**：
  - 默认保存目录改为 `loss_data/CACE_LOREM_MP2_ABtest2`。
- **默认保存目录**：`loss_data/CACE_LOREM_MP2_ABtest2`

### C. `fit-cace-LOREM-testAB3.py`

- **定位**：把“阶段训练”变为默认策略。
- **核心改进**：默认 `phase_plan` 改为 3 阶段（非 quick-test）：
  1. `joint_warmup`（`sr_weight=1.0`）
  2. `lr_focus`（`sr_weight=0.1`, `freeze_sr_readout=True`）
  3. `joint_finetune`（`sr_weight=1.0`）
- **目的**：尝试让 LR 在中段获得更多梯度，再回归联合优化。
- **默认保存目录**：`loss_data/CACE_LOREM_MP2_ABtest3`

### D. `fit-cace-LOREM-testAB-try.py`

- **定位**：LR-only 能力验证脚本（“LR 分支本体是否能学角度”）。
- **核心改进**：
  - 默认 `DEFAULT_MAX_EPOCHS = 50`
  - 默认 `DEFAULT_START_LR = 3e-3`（更激进）
  - 默认阶段为单段 `lr_only_probe`：
    - `sr_weight=0.0`
    - `freeze_sr_readout=True`
- **目的**：排除“LR 架构能力不足”假设，验证 LR-only 条件下是否能学到二面角形状。
- **默认保存目录**：`loss_data/CACE_LOREM_MP2_ABtry_lr_only`

### E. `fit-cace-LOREM-testAB4-strict.py`

- **定位**：严格版阶段训练（在 AB3 基础上进一步强化）。
- **核心改进**：
  1. 默认保存目录改为 `ABtest4_strict`。
  2. 默认阶段改为 5 段（严格 LR-focus + 渐进回归）：
     - `joint_warmup`（约 15%，`sr_weight=1.0`）
     - `lr_focus_strict`（约 45%，`sr_weight=0.0`, `freeze_sr_readout=True`）
     - `joint_return_0p1`（约 15%，`sr_weight=0.1`）
     - `joint_return_0p3`（约 15%，`sr_weight=0.3`）
     - `joint_return_0p5`（剩余，`sr_weight=0.5`）
  3. 新增阶段门控日志 `PhaseGate`，阈值：
     - `corr_long_vs_ref >= 0.70`
     - `amp_long/ref >= 0.20`
     - `F_long/F_total >= 0.10`
     - 每阶段输出 PASS/FAIL（日志门控，不强制中断）。
- **默认保存目录**：`loss_data/CACE_LOREM_MP2_ABtest4_strict`

---

## 2. 当前日志文件与训练脚本对应（`fit-cumulene2` 目录）

当前检测到的日志文件：

- `log_test_AB_81.out`
- `log_test_AB2_86.out`
- `log_test_AB3_87.out`
- `log_ABtry_LRonly_88.out`

对应关系如下：

1. `log_test_AB_81.out`
   - 对应脚本：`fit-cace-LOREM-testAB.py`
   - 依据：AB 命名、日志内容里保存目录为 `.../ABtest/...`

2. `log_test_AB2_86.out`
   - 对应脚本：`fit-cace-LOREM-testAB2.py`
   - 依据：AB2 命名、日志内容显示 `.../ABtest/...` 与 `...ABtest2...` 曾有目录错配记录（后续已修正到更一致的 slurm 版本）

3. `log_test_AB3_87.out`
   - 对应脚本：`fit-cace-LOREM-testAB3.py`
   - 依据：AB3 命名，日志中可见 `Default3Phase` 三阶段输出特征

4. `log_ABtry_LRonly_88.out`
   - 对应脚本：`fit-cace-LOREM-testAB-try.py`
   - 依据：日志中含 `LROnlyDefault`、`lr_only_probe`，且保存目录为 `...ABtry_lr_only...`

> 备注：`fit-cace-LOREM-testAB4-strict.py` 为最新严格版，若运行后建议日志命名为 `log_test_AB4_strict_<jobid>.out`，并与保存目录 `ABtest4_strict` 保持一致。

---

## 3. 每版脚本在“长程角度学习”问题上的定位

- `AB / AB2`：用于确认“默认联合训练”下 LR 难学到角度形状的现象。
- `AB3`：验证“中段 LR-focus + 最后回联合”是否足以改善。
- `AB-try`：验证“LR-only 条件下 LR 分支本体可学习角度信息”。
- `AB4-strict`：把 LR-only 能力迁移到联合训练，采用严格 LR-focus + 渐进 SR 回归 + 阶段门控监控。

---

## 4. 建议后续使用方式

1. 用 `AB-try` 作为 LR 能力基线（确认 LR-only 仍 PASS）。
2. 主实验切换到 `AB4-strict`，重点看每个阶段的 `PhaseGate` 输出。
3. 若 `lr_focus_strict` 阶段 PASS、但回归阶段 FAIL，优先继续放缓 SR 回归（例如增加 `sr_weight=0.2/0.4` 过渡段）。

