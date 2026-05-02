# CACE-LOREM：L2 基协变一致性改造说明（不引入 e3x）

本文档说明本次在 `CACE-LOREM` 中完成的 **L2（基协变级）** 改造：  
在不切换到 e3x/e3nn 的前提下，通过显式旋转一致性约束，让 `l=1/l=2` 长程电荷读出更接近严格等变行为。

---

## 1. 改动目标

在现有 CACE 框架下，保持原有训练/推理主链路不变，新增一个训练正则项：

- 对同一 batch 随机旋转得到 `R·x`；
- 要求旋转后预测的 `q1/q2` 与原预测经过对应 `D^(1)(R)/D^(2)(R)` 变换后尽量一致；
- 将该一致性损失按权重加入总训练损失。

---

## 2. 代码改动清单

### 2.1 `cace/modules/lorem_longrange.py`（方案 A 节点内等变读出）

`MultipoleChargeHead` 的 `q1/q2` 读出已从
`mean(pool)->Linear` 改为 MessageBchi 风格的“标量门控 × 等变块”：

- 先由不变量 `p_features` 生成门控权重 `w_{k,c}`（分别对应 `l=1/l=2`）；
- 对 `S^(l)` 做逐 `(radial, channel)` 加权：`S^(l) * w`（权重在角向维共享）；
- 仅在 `radial/channel` 维求和，保留角向维：
  - `l=1` 直接得到 3 分量 `q1`
  - `l=2` 先得到 6 分量 `q2_raw6`
- 对 `q2_raw6` 施加固定去迹投影（6->5）得到 `q2`。

当前固定投影假设 `l=2` 笛卡尔分量顺序为 `[xx, xy, xz, yy, yz, zz]`，输出 5 维为：

- `xx-yy`
- `2zz-xx-yy`
- `xy`
- `xz`
- `yz`

这一步对应了“节点内等变读出（不新增边消息）”。

### 2.2 `cace/tasks/train.py`

新增可插拔一致性约束接口：

- `consistency_loss_fn: Optional[Callable[..., torch.Tensor]] = None`
- `consistency_loss_weight: float = 0.0`

在 `train_step()` 中：

- 先计算原有监督损失；
- 若配置了 `consistency_loss_fn` 且权重大于 0，则额外计算
  `loss += consistency_loss_weight * consistency_loss`；
- 其余训练流程（反向传播、优化器、EMA、保存）不变。

### 2.3 `fit-cumulene/fit-cace-LOREM-now.py`

新增 L2 控制参数：

- `L2_ROT_CONSISTENCY_WEIGHT = 0.05`

新增函数：

- `random_rotation_matrix()`：随机采样 `SO(3)` 旋转矩阵；
- `_rotate_last_dim3()`：对坐标类张量做旋转；
- `_quad5_to_matrix()` / `_matrix_to_quad5()`：四极 5 分量与 3x3 对称去迹张量互转；
- `_rotate_quadrupole5()`：用 `Q' = R Q R^T` 旋转四极；
- `_build_rotated_batch_dict()`：构造旋转后的 batch；
- `l2_rotation_consistency_loss()`：计算 `q1/q2` 的旋转一致性损失。

并在 `TrainingTask(...)` 中接入：

- `consistency_loss_fn=l2_rotation_consistency_loss`
- `consistency_loss_weight=L2_ROT_CONSISTENCY_WEIGHT`

---

## 3. L2 一致性损失定义

记当前 batch 的原预测为：

- `q1`：偶极通道（3 维）
- `q2`：四极通道（5 维）

随机采样旋转 `R`，在旋转输入上得到预测：

- `q1_rot_pred`
- `q2_rot_pred`

构造目标：

- `q1_target = D^(1)(R) q1`（在当前实现中等价于笛卡尔向量旋转）
- `q2_target = D^(2)(R) q2`（通过四极矩阵旋转 `Q' = R Q R^T` 再映射回 5 分量）

一致性损失：

$$
\mathcal{L}_{\text{L2}} =
\mathrm{MSE}(q1_{\text{rot\_pred}}, q1_{\text{target}})
+
\mathrm{MSE}(q2_{\text{rot\_pred}}, q2_{\text{target}})
$$

总损失：

$$
\mathcal{L}_{\text{total}} =
\mathcal{L}_{\text{sup}} + \lambda_{\text{L2}} \mathcal{L}_{\text{L2}}
$$

其中当前设置 $\lambda_{\text{L2}} = 0.05$。

---

## 4. 本次实现与“严格数学等变”的关系

- 本改造是 **L2：基协变级**，显式加入了旋转一致性约束；
- 它比纯结构先验（L0/L1）更接近严格等变；
- 但由于未引入完整 CG/e3x 张量代数闭合，仍属于“强约束近似等变”，不是 L3 的算子级严格闭合。

---

## 5. L0~L3 严格度分级（本项目语境）

### L0：流程模仿级

- 按 `l` 分块，采用经验门控与读出；
- 无显式旋转一致性约束；
- 代价最低，迭代最快。

### L1：块约束级

- 禁止跨 `l` 非法混合，保持块内读出；
- 结构先验更强，但未显式做旋转一致性训练；
- 代价低到中等。

### L2：基协变级（本次实现）

- 显式加入 `D^(1)/D^(2)` 一致性损失；
- 在不换框架前提下显著提升旋转协变约束强度；
- 训练开销中等（需额外旋转分支前向）。

### L3：算子闭合级

- 在 `l<=2` 上实现严格合法耦合表与张量算子闭合；
- 最接近严格群等变；
- 开发与维护成本最高。

---

## 6. 使用与调参建议

- 初始可用：`L2_ROT_CONSISTENCY_WEIGHT = 0.01 ~ 0.05`；
- 若主损失下降明显变慢，可先降到 `0.01`；
- 若 `q1/q2` 旋转一致性仍差，可增到 `0.1` 并观察：
  - `std(q)`
  - `||F_long|| / ||F_total||`
  - `E_long / E_total`
  - 二面角曲线形状是否改善。

