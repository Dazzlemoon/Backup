# LOREM SOG 改动结果说明（本次实现）

本文记录本次在 `LOREM/lorem` 中已落地的代码改动。当前版本在保持原有共享 SOG 参数调用不变的前提下，新增了按多极阶数 `l` 分开的 SOG 参数设计（可开关）。

---

## 1. 已完成改动概览

本次完成了五项核心改动：

1. 新增 `sog_kernel.py`，实现 **periodic** 情况下的 SOG 前向势计算；
2. 在 `lorem.py` 中新增 `lr_kernel_type` 开关，支持 `coulomb/sog` 两条 periodic 分支；
3. 在 `run.py` 中暴露配置入口，可通过 `settings.yaml` 选择训练主核（默认 `coulomb`）；
4. 在 `lorem.py` 中新增 `sog_init_mode`，支持 `dimer_cc` 硬编码参数初始化；
5. 在 `lorem.py`/`run.py` 中新增 `sog_l_dependent_params`，支持按 `l` 使用不同 SOG 参数。

默认行为保持不变：`lr_kernel_type="coulomb"`、`sog_init_mode="uniform"`、`sog_l_dependent_params=false`，不影响现有训练与评估流程。

---

## 2. 新增文件

### `LOREM/lorem/sog_kernel.py`

新增函数：

- `compute_sog_periodic_potentials(...)`

实现要点：

- 基于 `cell` 与 `k_grid_shape` 构建 reciprocal `k` 向量；
- 使用可训练参数构造 SOG multiplier：
  - 宽度参数通过 `softplus` 保证为正；
  - 形式为高斯和；
- 去除 `k=0` 模（稳定性/与 Ewald 常见处理一致）；
- 通过结构因子映射得到每个原子、每个通道的 `potentials`；
- 支持两种参数形状：
  - 共享模式：`sog_*` 为 `[M]`；
  - 通道模式：`sog_*` 为 `[C, M]`（每个通道独立 kernel）。

输出形状与 `lorem.py` 回注逻辑兼容：`[num_nodes, num_channels]`。

---

## 3. 修改文件

### `LOREM/lorem/lorem.py`

主要改动：

1. 新增 import：
   - `from sog_kernel import compute_sog_periodic_potentials`

2. 在 `Lorem` 模型新增配置字段：
   - `lr_kernel_type: str = "coulomb"`
   - `sog_num_gaussians: int = 12`
   - `sog_init_mode: str = "uniform"`（可选 `uniform` / `dimer_cc`）
   - `sog_l_dependent_params: bool = False`

3. 在 periodic 长程分支中增加条件逻辑：
   - `lr_kernel_type == "sog"`：调用 `compute_sog_periodic_potentials(...)`
   - 否则：继续走原 `Ewald(...).potentials(...)`

4. 新增中间诊断输出（SOG 路径）：
   - `diag_sog_amplitudes`
   - `diag_sog_log_widths`
   - `diag_sog_amplitudes_per_l`（仅 `sog_l_dependent_params=true`）
   - `diag_sog_log_widths_per_l`（仅 `sog_l_dependent_params=true`）

5. 新增 dimer-CC 硬编码初始化模式（`sog_init_mode="dimer_cc"`）：
   - 约束 `sog_num_gaussians == 12`；
   - `sog_amplitudes` 使用 dimer-CC 的固定向量；
   - `shift_1 = linspace(-3, 2, 12)` 映射为当前实现所用 `width`，再转换为 `sog_log_widths`。

6. 新增按 `l` 分开参数（`sog_l_dependent_params=true`）：
   - 先学习 `max_degree_lr+1` 组参数：`sog_*_per_l`，形状为 `[(L+1), M]`；
   - 再按通道把每组 `l` 参数展开为 `[C, M]`；
   - 当前通道到 `l` 的映射：
     - 通道 0（`scalar_charge`）-> `l=0`
     - 球谐通道中 `l=0` 的 1 个分量 -> `l=0`
     - `l=1` 的 3 个分量 -> `l=1`
     - `l=2` 的 5 个分量 -> `l=2`
     - ...直到 `max_degree_lr`

### `LOREM/lorem/run.py`

新增配置透传能力（读取 `settings.yaml`）：

- `lr_kernel_type`（默认 `coulomb`）
- `sog_num_gaussians`（默认 `12`）
- `sog_init_mode`（默认不覆盖模型默认值）
- `sog_l_dependent_params`（默认 `false`）

并在模型构建前注入到 `model_config["model"]`，支持不改代码、仅改配置切换主核与参数模式。

---

## 4. 配置示例

在 `settings.yaml` 中启用按 `l` 分开参数（仅 SOG kernel 下生效）：

```yaml
lr_kernel_type: sog
sog_num_gaussians: 12
sog_init_mode: uniform
sog_l_dependent_params: true
```

保持旧版共享参数路径：

```yaml
lr_kernel_type: sog
sog_l_dependent_params: false
```

---

## 5. 未改动部分（当前版本边界）

以下内容当前仍未改动：

- 非周期分支（`full_R_ij` 路径）仍为原 `1/r + segment_sum`；
- 未新增并行对照输出（如同一步同时输出 Coulomb/SOG 的 long energy）。

---

## 6. 兼容性与风险说明

- 默认 `lr_kernel_type="coulomb"` 且 `sog_l_dependent_params=false`，旧实验默认行为不变；
- 新 SOG 分支当前仅覆盖 periodic 前向；
- 使用相对导入路径（`from sog_kernel import ...`）与当前 `lorem.py` 目录结构一致。

---

## 7. 建议下一步

1. 增加 `diag_energy_long_*` 并行对照日志；
2. 新增 `sog_init_mode=bsa_coulomb`（程序化 BSA 初始化）；
3. 再实现 non-periodic 的 SOG 近似分支，补齐双分支一致性。

