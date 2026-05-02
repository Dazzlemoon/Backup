# CACE-LES 参数对齐说明（cumulene）

本文档说明 `fit-cace-SOG.py` 在 `fit-cumulene-out3-change` 中如何按 LOREM 论文里 CACE-LES 的 cumulene 参数口径进行修改。

## 1. 论文中可对应的参数口径

根据论文附录对 CACE-LES（cumulene/biodimers）的描述，可对应到以下关键信息：

- cutoff: `5 Å`
- 径向基函数: `6 Bessel radial functions`
- `c = 8`
- `lmax = 3`
- `one message-passing layer`（即 MP=1）
- `one-dimensional hidden variable`（隐藏维度取 1）
- `sigma = 1`
- `dl = 2`
- 训练策略：总 500 epoch，能量权重 `0.1 -> 1 -> 10 -> 1000`（每 100 epoch 变化），力权重 `1000`，学习率 `5e-3`，每 20 step 衰减 2 倍

## 2. 本次在脚本中的对应修改

目标文件：

- `/data/home/public/qiuqizhi/SOG-Qeq/SOG-Net/CACE-SOG-Ji/fit-cumulene-out3-change/fit-cace-SOG.py`

已修改项：

- `MP = 2` -> `MP = 1`
- `args_message_passing['Bchi']`:
  - `shared_channels=False` -> `shared_channels=True`
  - `shared_l=False` -> `shared_l=True`
- 短程能量头 `atomwise`:
  - `n_hidden=[32,16]` -> `n_hidden=[1,1]`
- 电荷头 `q`:
  - `n_hidden=[24,12]` -> `n_hidden=[1,1]`

## 3. 按你的额外要求保留的设置

你要求保留：

- `q n_out=3`

因此该脚本是“按论文轻量化参数口径对齐”，但在 `q` 头输出维度上保留了 3 通道设置，不是严格的标量 `q` 版本。

## 4. 参数量量级说明

- 修改前（原始 out3/同类配置）参数量为百万级（约 `3,180,972`）。
- 本次轻量化后（且保留 `q n_out=3`），参数量预期降到约 `7.8e4` 量级（约 7.8 万）。

这已经接近论文中 CACE-LES 报告的 `~70k` 量级，但由于 `q n_out=3`，通常会比严格标量 `q` 版本略大。

