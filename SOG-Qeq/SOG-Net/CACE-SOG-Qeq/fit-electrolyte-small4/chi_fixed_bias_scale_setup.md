# `fit-electrolyte-small4` 固定 `chi_biased` 设置说明

## 目标

将 `chi_biased` 改为**固定公式**：

`chi_biased = s * chi + b_Z`

其中：

- `b_Z` 使用 H/O/F/K 的 Pauling 电负性常用值；
- `b_Z` 不做去均值；
- `b_Z` 不可训练（固定常数）；
- `s = 5` 固定常数，不可训练。

---

## 已做修改（`fit-cace-Qeq-SOG.py`）

1. 保持 `chi` 分支为有符号输出：
   - `post_process=None`（即 `chi = chi_raw`，不平方）。

2. 删除原来的可训练 `ElementwiseFeatureBias` 配置：
   - 不再使用 `zero_mean_bias=True`；
   - 不再使用可训练 `bias` 和可训练 `scale`。

3. 新增固定模块 `FixedElementwiseFeatureBias`：
   - 内部使用 `register_buffer("bias", ...)` 和 `register_buffer("scale", ...)`；
   - 因为不是 `nn.Parameter`，所以优化器不会更新它们；
   - 前向严格执行：
     - `chi_biased = scale * chi + bias[element]`
   - 不进行去均值操作。

4. 固定常数设置如下：
   - `init_bias={1: 2.20, 8: 3.44, 9: 3.98, 19: 0.82}`
   - `init_scale=5.0`

元素顺序：

- `elements=[1, 8, 9, 19]` 对应 `[H, O, F, K]`

---

## 对训练行为的影响

- `chi_bias` 不再随训练漂移；
- `scale` 也不再从 5.0 学到其他值；
- Qeq 输入的元素偏置将始终保持 Pauling 常数，不会出现“训练后偏离初始化”的情况。

---

## 备注

- 此配置更强约束、可解释性更强，但自由度更小；
- 若能量/力误差上升，属于预期风险（模型可调参数被减少）。

