## `log_5197318` 训练进度与产物分析

本文基于：
- 训练日志：`fit-4hdnnp-NaCl/log_5197318`
- 训练脚本：`fit-4hdnnp-NaCl/fit-cace-SOG.py`

---

## 1) 现在训练到第几个 epoch？

日志尾部显示当前处在 **Fourth train loop**，并且已经跑到 **epoch 50**（日志最后一个验证点是 `Epoch 50 ...`）：

- `Fourth train loop:` 出现在日志靠后（约第 1305 行附近）
- 末尾包含：
  - `Epoch 40, ...`、`Epoch 50, ...`
  - 最后记录到 `Epoch 50, Train Loss: ..., Val Loss: ...`

因此就“当前这一次 loop 的 epoch 计数”而言：**第 4 段训练正在进行，已到 epoch 50**。

> 注意：日志里每个 loop 的 epoch 都从 1 重新计数；如果你关心“全局累计训练了多少 epoch”，见下文第 2 节。

---

## 2) `fit-cace-SOG.py` 一共会跑多少个 epoch？

脚本里明确写了 4 个阶段（其中第 1 阶段还包含一个 `for i in range(5)` 的重复训练）：

### 2.1 First train loop（循环 5 次）

脚本（`fit-cace-SOG.py`）：
- `for i in range(5): ... task.fit(... epochs=40, val_stride=10)`

因此 First train loop 总 epoch 数：
- **5 × 40 = 200 epochs**

### 2.2 Second / Third / Fourth train loop

脚本依次调用：
- Second：`task.fit(... epochs=100, val_stride=10)`
- Third：`task.fit(... epochs=100, val_stride=10)`
- Fourth：`task.fit(... epochs=100, val_stride=10)`

因此这三段合计：
- **100 + 100 + 100 = 300 epochs**

### 2.3 总计

合计总训练 epoch 数为：
- **200 + 300 = 500 epochs**

---

## 3) 会生成多少个 `.pth` 文件？

这里要区分“脚本显式保存的 `.pth`”和“训练过程中自动保存的 `.pth`”。

### 3.1 脚本显式保存的 4 个 `.pth`

脚本最后会尝试保存 4 个固定文件名（每段 loop 结束后各一次）：
- `hydrocarbon-model.pth`
- `hydrocarbon-model-2.pth`
- `hydrocarbon-model-3.pth`
- `hydrocarbon-model-4.pth`

理论上：**4 个 `.pth`**。

但需要注意：`TrainingTask.save_model()` 当前实现里存在一个条件判断问题：
- 在 `ema=False` 的默认配置下，`self.ema_model` 为 `None`
- `save_model()` 的非 EMA 分支里又写了 `if self.ema_model: torch.save(...)`
- 这会导致 **在 `ema=False` 时不会真正执行 `torch.save`**，从而可能出现“脚本写了 save_model 但实际没有生成对应 `.pth`”的情况。

所以这 4 个文件名是“计划产物”，但是否实际落盘要以目录中是否存在为准。

### 3.2 训练过程中自动保存的 `.pth`（在 `save_folder` 里）

`TrainingTask.train_step()` 和 `TrainingTask.fit()` 里还有若干 `torch.save(...)`，文件名固定（会被反复覆盖更新），因此“文件数量”是固定的，但“保存次数”随训练过程变化：

- **1 个**：当训练 loss 刷新最小值时保存（同名覆盖）
  - `{save_folder}/loss{time_name}_model.pth`
  - 日志前段出现大量 `保存失败: 'NeuralNetworkPotential' object has no attribute 'models'`，说明曾尝试保存但失败；后续是否成功取决于运行时是否仍触发异常。

- **最多 4 个**：每次验证（`val_stride=10`）时，如果能量/力的 MAE 或 RMSE 同时满足阈值并刷新“历史最好”，就保存（同名覆盖）
  - `{save_folder}/min_mae_e{time_name}_model.pth`
  - `{save_folder}/min_mae_f{time_name}_model.pth`
  - `{save_folder}/min_rmse_e{time_name}_model.pth`
  - `{save_folder}/min_rmse_f{time_name}_model.pth`
  - 日志后段明确出现过 `Save best MAE e model:` / `Save best RMSE e model:`，说明这些保存逻辑至少被触发过。

### 3.3 小结：`.pth` 文件“最多多少个文件名”

按“不同文件名个数”统计，上限为：
- `hydrocarbon-model*.pth`：4 个（但可能因 `save_model` 条件问题而未生成）
- `save_folder` 下：`loss...` 1 个 + `min_*` 4 个 = 5 个

合计理论上最多：
- **9 个不同文件名的 `.pth`**

> 备注：如果你还传了 `bestmodel_path`（默认 `best_model.pth`），理论上也会生成/覆盖 1 个，但它同样走 `save_model()`，在 `ema=False` 时也可能不会真的保存，因此本文不把它当作“可靠产物”计数。

---

## 4) 会生成多少个 checkpoint 文件？

`TrainingTask.fit()` 默认参数：
- `checkpoint_path='checkpoint.pt'`
- `checkpoint_stride=10`

每次 `fit(... epochs=..., ...)` 内部都会在 `epoch % 10 == 0` 时调用 `self.checkpoint('checkpoint.pt')`，因此：
- **每个 fit 阶段会反复覆盖同一个文件名 `checkpoint.pt`**
- 目录里最终通常只会看到 **1 个 `checkpoint.pt` 文件**（内容为最近一次保存的状态）

如果按“写入次数”而不是“文件名数量”统计，那么总 checkpoint 写入次数为：
- First loop：5 次 fit × (40/10)=4 次/fit ⇒ 20 次
- Second：100/10=10 次
- Third：10 次
- Fourth：10 次
- 合计写入次数：**20 + 10 + 10 + 10 = 50 次**

但落盘文件名数量仍然是：
- **1 个：`checkpoint.pt`（不断覆盖）**

