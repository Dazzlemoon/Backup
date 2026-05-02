# CACE-SOG-LOREM 短程网络修改结果（LOREM 风格）

本文档记录本次在 `CACE-SOG-LOREM` 中完成的短程网络改造：在保持 CACE 笛卡尔角向基与消息传递模块的前提下，新增一个 **LOREM 风格的 Y/S/P 数据流表示类**，用于后续长程实验。

---

## 1. 已完成的代码修改

- 新增表示模块：`cace/representations/cace_lorem_short_range.py`
- 新增类：`CaceLoremShortRange`
- 导出入口已更新：`cace/representations/__init__.py`

---

## 2. 新类的设计目标

`CaceLoremShortRange` 不是逐行复刻 LOREM（JAX/e3x），而是用 CACE 现有组件实现相同的短程语义分工：

- **Y（固定角向基，内部使用）**：`angular_component = AngularComponent(edge_vectors)`
- **S（随 MP 更新）**：`s_now`（由 A-basis 表示构成，循环内更新）
- **P（随 MP 更新）**：`p_now`（标量节点状态，由 `S` 的角向范数更新）

具体映射：

1. 边特征 `edge_attri = radial * cutoff * angular * edge_type`
2. 聚合到节点得 `s_now`（对应 LOREM 的球面分支角色）
3. 计算 `s_now` 的角向范数并更新 `p_now`
4. 重复 MP：`MessageAr` / `MessageBchi` / `NodeMemory` 更新 `s_now`
5. 每轮用 `s_now` 再更新 `p_now`

---

## 3. 输出键（便于调试和下游复用）

新类 `forward` 输出：

- `node_feats`: `p_features` 堆叠结果（保持与 `Atomwise(feature_key="node_feats")` 兼容）
- `s_features`: 每轮 S 的堆叠
- `p_features`: 每轮 P 的堆叠
- `b_features`: 每轮对称化不变量 B 的堆叠

其中 `node_feats` 默认等于 `p_features`，可直接接现有 `Atomwise` 头训练。

---

## 4. 与原 `Cace` 表示的关系

- 原 `Cace`：输出以 `node_feat_B` 堆叠后的 `node_feats` 为主
- 新 `CaceLoremShortRange`：显式维护并输出 Y/S/P，更接近 LOREM 图中的短程流程语义
- 两者都使用 CACE 的笛卡尔角向基，不引入 e3nn/e3x 依赖

---

## 5. 当前状态与下一步建议

### 当前状态

- 新增模块已通过语法检查（`py_compile`）
- 目前未自动替换已有训练脚本，属于“可选 representation”

### 下一步建议

在训练脚本中将：

```python
from cace.representations import Cace
```

替换为：

```python
from cace.representations import CaceLoremShortRange
```

并把实例化 `Cace(...)` 改为 `CaceLoremShortRange(...)`（参数大体兼容，新增 `n_scalar_features` 可选）。

---

## 6. 备注

该改造实现的是“LOREM 风格短程流程”的 CACE 版本（Y 固定、S/P 迭代更新），用于支持后续你规划的 CACE+LOREM 长程融合实验；它与 LOREM 原生 e3x 张量路径在数学上并不完全等价。
