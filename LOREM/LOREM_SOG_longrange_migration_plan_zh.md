# LOREM 中将库伦求和替换为 SOG 高斯和的实现建议

本文讨论如何在 `LOREM/lorem` 中，把当前“等变电荷 + 库伦/Ewald 求和”的长程分支，改造成 Ji et al. (2026) 的 SOG（sum-of-Gaussians）形式。

本次已落地改动说明见：[`LOREM_SOG_code_changes_summary_zh.md`](./LOREM_SOG_code_changes_summary_zh.md)。

参考：

- 论文：`SOG-Qeq/SOG-Net/CACE-SOG/Ji 等 - 2026 - Accurate learning of long-range interatomic potentials by coupling Cartesian atomic cluster expansio.pdf`
- 参考代码：`SOG-Qeq/SOG-Net/CACE-SOG-Ji`（重点 `cace/modules/ewaldC.py`）

---

## 1. 当前 LOREM 长程实现位置（需要改的核心）

文件：`LOREM/lorem/lorem.py`

在 `if self.lr:` 分支中，当前流程为：

1. 从特征读出潜变量：
   - `scalar_charges`
   - `spherical_charges`
   - `charges = concat([scalar, spherical])`
2. 两种库伦求和：
   - 周期：`Ewald(...).potentials(...)`
   - 非周期：`segment_sum(charges/r)`
3. 将 `potentials` 回注到 `nodes_scalar`，再用 MLP 得到 `energy_long`

也就是说，**真正需要替换的是第 2 步“求势核”**；前后读出/回注框架可以先保持。

---

## 2. SOG 目标形式（与论文对齐）

论文方法（Eq. 6–8）：

- 结构因子：
  - `S(k, η) = Σ_i q_{i,η} exp(-i k·r_i)`
- Fourier 乘子（高斯和）：
  - `f_hat(k) = Σ_l w_l exp(-k^2 / s_l^2)`（等价写法可不同）
- 长程能量：
  - `E_lr = (1 / 2V) Σ_η Σ_k f_hat(k) |S(k,η)|^2`

`CACE-SOG-Ji/cace/modules/ewaldC.py` 里使用 NUFFT/FFT 实现了这个思想（含 periodic 与 real-space Gaussian 近似分支）。

---

## 3. 推荐改造策略（最小侵入）

建议分 3 步走，避免一次性推翻 LOREM 训练稳定性。

### Step A：在 `Lorem` 中新增 `lr_kernel_type`（已完成）

在 `Lorem` 配置新增开关：

- `lr_kernel_type = "coulomb"`（默认，保持现状）
- `lr_kernel_type = "sog"`（新）

并新增 SOG 超参：

- `sog_num_gaussians`（M，默认 12）
- `sog_init_mode`（`bsa_coulomb` / `uniform`）
- `sog_use_nufft`（True/False）
- `sog_periodic_mode`（`kspace` / `realspace_gaussian`）

这样可保证老实验不受影响。

当前状态：

- `lr_kernel_type`：已在 `lorem.py` 与 `run.py` 中打通；
- `sog_num_gaussians`：已可由 `settings.yaml` 配置；
- `sog_init_mode`：已实现 `uniform` 与 `dimer_cc`（硬编码初始化），`bsa_coulomb` 尚待实现。

### Step B：把“求势”抽象成独立函数

在 `lorem.py` 中将当前

- `Ewald(...).potentials(...)`
- `segment_sum(charges/r)`

抽为统一接口，例如：

- `_compute_lr_potentials_coulomb(...)`
- `_compute_lr_potentials_sog(...)`

最终都返回与现有一致的 `potentials` 张量形状，后面的回注与 `energy_long` MLP 可以不动。

### Step C：先做“并行诊断模式”

先不替换训练主路，增加诊断输出：

- `diag_energy_long_coulomb_raw`
- `diag_energy_long_sog_raw`
- 对应 `corr/amp` 对比

验证 SOG 分支数值稳定、梯度正常后，再把训练主路切到 SOG。

---

## 4. 与 CACE-SOG-Ji 对齐时的关键实现点

## 4.1 潜变量通道维

LOREM 当前 `charges = [scalar + spherical]`，天然是多通道（η 维）。
这与论文 `Σ_η` 一致，不需要改读取方式。

## 4.2 k=0 模处理

论文与 CACE-SOG 代码都强调零频处理/平滑核。
迁移时要明确：

- 是否去除 `k=0`
- 是否做中性约束或惩罚（可选）

## 4.3 初始化（强烈建议用 BSA）

论文 Eq. 13–14 给了 Coulomb-tail 的 SOG 初始化：

- `w_l` 与 `s_l` 可由 `(b, s)` 解析得到
- 默认 `M=12` 常见有效

建议优先实现 `bsa_coulomb` 初始化，避免随机初始化导致收敛慢。

当前补充：

- 已新增 `dimer_cc` 初始化模式（来自 `CACE-SOG-Ji` 的硬编码参数思路）用于快速对齐试验；
- 该模式是“经验/硬编码”路线，不是严格的程序化 BSA（Eq.13/14）生成器。

## 4.4 周期/非周期双分支

对应 LOREM 现有双分支：

- periodic：SOG-kspace（优先）
- non-periodic：先做 Gaussian real-space 近似（与 CACE-SOG-Ji 思路一致）

---

## 5. 代码层建议（文件级）

建议新增文件（而不是把 `lorem.py` 塞太满）：

- `LOREM/lorem/sog_kernel.py`
  - `class SOGKernel` 或函数集合
  - 包含 `f_hat(k)`、BSA 初始化、periodic/nonperiodic 势计算

修改文件：

- `LOREM/lorem/lorem.py`
  - 新增 `lr_kernel_type` 分支
  - 在 `if self.lr:` 中调用 `sog_kernel` 或现有 Coulomb 分支
- `LOREM/lorem/run.py`
  - 配置参数透传
  - 新增 SOG 诊断输出（权重、带宽、长程能量统计）
- `LOREM/lorem/transforms.py`
  - 若 SOG periodic 需要额外网格参数，可在此扩展预处理

---

## 6. 验证计划（避免“改了但不知道是否变好”）

第一阶段（功能正确性）：

- 固定小 batch，比较 Coulomb/SOG 前向是否无 NaN，梯度是否可回传
- 检查 `energy_long` 尺度与 `q` 通道统计是否合理

第二阶段（行为对齐）：

- 在“以 1/r 为主”的数据上，SOG 应接近 Coulomb（不明显变差）
- 在“非 1/r 尾部”数据上，SOG 期望优于 Coulomb（论文结论）

第三阶段（你当前关心的二面角任务）：

- 重点看 `corr_long_vs_ref`、`amp_long/ref`、`F_long/F_total`
- 与 LR-only / 联合训练结果对照，判断是否减轻“长程学不到形状”

---

## 7. 风险与注意事项

- `jax` 版本下若引入 NUFFT 依赖，工程复杂度高；可先做纯 `jax.numpy` 的离散 k-space 原型再优化。
- 若直接把 `energy_long` 从“潜势回注+MLP”改成“纯二次型能量”，行为会变化较大；建议先保持现有回注框架，只替换求势核。
- SOG 参数（`w_l`, `s_l`）需加稳定约束（如 `softplus` 保证尺度正），防止训练初期发散。

---

## 8. 推荐落地顺序（实操）

1. 新增 `sog_kernel.py`（只实现 periodic 前向，先不进训练主路）
2. `lorem.py` 增加 `lr_kernel_type`，支持 `coulomb/sog`（已完成）
3. 打开并行诊断（同一 batch 同时输出 Coulomb/SOG long energy）
4. 小规模训练验证稳定性
5. 全量训练与现有 AB/AB-strict 指标对比

这样可以把改动风险压到最低，并且每一步都有可观测指标验证。

---

## 9. 目前可直接使用的配置示例

在 `settings.yaml` 中可直接切换训练主核：

- Coulomb（默认）：
  - `lr_kernel_type: coulomb`

- SOG（uniform 初始化）：
  - `lr_kernel_type: sog`
  - `sog_num_gaussians: 12`
  - `sog_init_mode: uniform`

- SOG（dimer-CC 硬编码初始化）：
  - `lr_kernel_type: sog`
  - `sog_num_gaussians: 12`
  - `sog_init_mode: dimer_cc`

