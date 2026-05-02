# CACE-LOREM `fit-cumulene/fit-cace-SOG.py` 训练流程说明

本文档说明当前脚本如何执行 CACE-LOREM 的「短程 + 长程」联合训练，并标注与 LOREM cumulene 配置对齐的关键超参数。

---

## 1. 目标与输出

脚本目标：在 cumulene 数据上训练总能量/力模型，其中

- 短程能量：`CACE_energy`
- 长程能量：`lr_energy`（由多极通道 + Ewald + 回灌读出得到）
- 总能量：`CACE_energy = CACE_energy(sr) + lr_energy(lr)`（通过 `CombinePotential`）

---

## 2. 关键参数

当前脚本中关键量级：

- `cutoff = 5.0`
- `MP = 2`（短程消息传递步数）
- `optimizer_args["lr"] = 1e-3`

其余 CACE 架构参数（如 `max_l=3`, `max_nu=3`, `n_atom_basis=8`, `n_radial_basis=12`）保留 CACE 侧设置。

---

## 3. 数据与加载

1. 从 `LOREM/datasets` 读取：
   - `cumulene_train.xyz`
   - `cumulene_valid.xyz`
   - `cumulene_test.xyz`
2. 使用 `cace.tasks.get_dataset_from_xyz(...)` 生成集合；
3. 用 `load_data_loader(...)` 构建 train/valid/test loader。

---

## 4. 短程分支（SR）

### 表示
- 使用 `CaceLoremShortRange(...)`
- 输出包含 `node_feats`, `s_features`, `s_l0/s_l1/s_l2`, `p_features` 等

### 读出
- `Atomwise(..., output_key='CACE_energy')` 生成短程能量
- `Forces(energy_key='CACE_energy', forces_key='CACE_forces')`

形成 `cace_nnp_sr`。

---

## 5. 长程分支（LR，LOREM 风格）

`output_modules` 顺序如下：

1. `MultipoleChargeHead(output_key='q')`  
   从短程状态构造多极通道电荷：
   - 单极 1 通道
   - 偶极 3 通道
   - 四极 5 通道

2. `EwaldPotential(feature_key='q', compute_field=True, output_key='ewald_raw', exponent=1)`  
   用 Coulomb/Ewald 计算长程势，并输出 `q_field`。

3. `LoremLongRangeReadout(field_key='q_field', output_key='lr_energy')`  
   按 `l=0,1,2` 分组回灌：
   - 单极势门控 `S^(0)`
   - 偶极势门控 `S^(1)`
   - 四极势门控 `S^(2)`
   各组取范数得到不变量，拼接单极势后经 MLP 输出 `lr_energy`。

4. `Forces(energy_key='lr_energy', forces_key='SOG_forces')`

形成 `cace_nnp_lr`。

---

## 6. 总能量/总力合并

通过：

```python
cace_nnp = cace.models.CombinePotential([cace_nnp_sr, cace_nnp_lr], [pot1, pot2])
```

其中：

- `pot1['CACE_energy'] = 'CACE_energy'`（短程）
- `pot2['CACE_energy'] = 'lr_energy'`（长程）
- 力同样按键合并。

因此训练目标中的 `CACE_energy` / `CACE_forces` 即总量。

---

## 7. 训练策略

- 优化器：Adam (`lr=1e-3`, `betas=(0.99, 0.999)`)
- 调度器：StepLR (`step_size=20`, `gamma=0.5`)
- 梯度裁剪：`max_grad_norm=10`
- 分阶段能量损失权重：
  - 200 epoch: 0.1
  - 100 epoch: 1.0
  - 100 epoch: 10.0
  - 100 epoch: 1000.0
- 力损失权重固定：1000.0

---

## 8. 运行后产物

- 模型与日志保存到：
  - `/data/home/public/qiuqizhi/SOG-Qeq/SOG-Net/CACE-LOREM/fit-cumulene/loss_data/CACE_LOREM_MP*/`
- 稳定输出的检查点文件（位于 `save_folder`）：
  - `best_model.pth`（按验证损失最优）
  - `min_mae_e.pth`（按验证集能量 MAE 最优）
  - `min_mae_f.pth`（按验证集力 MAE 最优）
- 最终模型文件：
  - `cumulene-model-final.pth`
- 脚本末尾会执行 test 集评估并打印 `test_loss` 与可训练参数量。

---

## 9. 备注

- 当前脚本已切换到 EwaldPotential 路线（不是 SOGPotential）。
- 若需要和旧基线做 ablation，可仅替换 LR 分支模块而保持 SR 分支不变。
