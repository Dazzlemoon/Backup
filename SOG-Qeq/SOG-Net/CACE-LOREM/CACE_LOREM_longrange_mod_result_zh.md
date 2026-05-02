# CACE-LOREM 长程网络修改结果（单极+偶极+四极，Coulomb/Ewald，回灌）

本文档记录在 `CACE-LOREM` 工作目录内完成的长程网络改造。目标是按 LOREM 流程实现：

1. 由短程状态构造 `l<=2` 的等变电荷通道（单极/偶极/四极）；
2. 用 Coulomb/Ewald 计算长程势；
3. 将高阶势与 `S` 交互形成“等变更新”，再标量化，与单极势拼接后经 MLP 得到长程能量。

---

## 1. 修改文件清单

- 新增：`cace/modules/lorem_longrange.py`
  - `MultipoleChargeHead`
  - `LoremLongRangeReadout`
- 更新：`cace/modules/__init__.py`
  - 导出 `lorem_longrange` 中的新模块
- 更新：`cace/representations/cace_lorem_short_range.py`
  - 输出 `s_l1_features` / `s_l2_features` 供长程 `q1/q2` 线性读出

---

## 2. 新增长程模块说明

## 2.1 `MultipoleChargeHead`

路径：`cace/modules/lorem_longrange.py`

输入：
- `p_features`（默认键 `p_features`，取最后一层）
- `s_l1_features`（优先）或 `s_features` 的 `l=1` 子块（回退）
- `s_l2_features`（优先）或 `s_features` 的 `l=2` 子块（回退）

输出：
- `q`（默认键 `q`），通道布局固定为
  - `q[:, 0:1]`：单极（1 维）
  - `q[:, 1:4]`：偶极（3 维）
  - `q[:, 4:9]`：四极（5 维）
- 同时保留分量键：
  - `q_monopole`
  - `q_dipole`
  - `q_quadrupole`

构造方式（核心）：
- 单极：`q0 = MLP(P_i)`
- 偶极：`q1_i = W_1 * pool(S_i^(l=1))`
- 四极：`q2_i = W_2 * pool(S_i^(l=2))`

其中 `pool` 表示仅在径向/通道维做聚合，保留角向 `l` 分块，再做线性映射。  
这使 `q1/q2` 的读出路径与 LOREM 图中“从 S 的低阶块读出多极通道”更加一致。

---

## 2.2 `LoremLongRangeReadout`

路径：`cace/modules/lorem_longrange.py`

输入：
- 分组 `S` 特征（优先）：
  - `s_l0_features`
  - `s_l1_features`
  - `s_l2_features`
  （若缺失则回退到 `s_features`）
- `q_field`（来自 Ewald/Coulomb 模块的每原子每通道势）

输出：
- `lr_energy_atom`（每原子长程能）
- `lr_energy`（按 batch 聚合后的图级长程能）

流程：
1. `q_field` 拆分：
   - 单极势：`phi0 = q_field[:, :1]`
   - 偶极势：`phi1 = q_field[:, 1:4]`
   - 四极势：`phi2 = q_field[:, 4:9]`
2. 按 `l` 分组回灌耦合（笛卡尔分桶）：
   - `S^(0)` 由 `phi0` 门控
   - `S^(1)` 由线性映射后的 `phi1` 门控
   - `S^(2)` 由线性映射后的 `phi2` 门控
3. 分别对每个 `l` 块在角向维取范数，得到 `l=0/1/2` 的标量不变量；
4. 将 `phi0` 与三组不变量拼接后经 MLP 输出长程能量。

这对应 LOREM 的“高阶势与球特征交互 -> 分组标量化 -> 与单极势融合读出”的思想。

---

## 3. 与 Ewald/Coulomb 的衔接建议

可直接串接现有 `EwaldPotential`（`cace/modules/ewald.py`）：

- 设置 `feature_key='q'`
- 设置 `compute_field=True`，以得到 `q_field` 供回灌模块使用

示例顺序（`output_modules`）：

1. `MultipoleChargeHead(output_key='q', ...)`
2. `EwaldPotential(feature_key='q', compute_field=True, output_key='ewald_raw', ...)`
3. `LoremLongRangeReadout(s_feature_key='s_features', field_key='q_field', output_key='lr_energy', ...)`
4. `Forces(energy_key='lr_energy', forces_key='lr_forces')`

---

## 4. 关键说明（与“标准偶极/四极静电”关系）

- 当前实现是 **LOREM 风格工程路线**：
  - 多通道电荷 + Coulomb/Ewald + 回灌读出
- 它不等同于完整显式的多极张量核（如逐项写出偶极-偶极、四极-四极解析式）
- 但满足你当前需求：在 CACE 笛卡尔框架中实现 `l<=2` 电荷通道、长程求势与回灌

---

## 5. 当前状态

- 代码已完成并通过语法检查（`py_compile`）
- 相关文件无 lints 报错
- 尚未自动替换现有训练脚本；需在具体 `fit-*.py` 中切换 representation 与 output_modules 串接

---

## 6. 兼容性备注

- 为支持新电荷头，`CaceLoremShortRange` 的输出补充了：
  - `edge_index`
  - `shifts`
- 保留并扩展了已有：
  - `node_feats`
  - `s_features`
  - `s_l0_features`
  - `s_l1_features`
  - `s_l2_features`
  - `p_features`
  - `b_features`

这些改动不破坏已有模块默认键名；若没有分组 `S` 键，`q1/q2` 会回退到 `s_features` 的低阶切片读出。
