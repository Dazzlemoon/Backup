## `fit_cace_new.py` 中 Qeq 在 CACE‑LES 里的集成说明（water 系统）

本文件解释 `fit_cace_new.py` 这份脚本是如何在 CACE 框架中引入 Qeq（charge equilibration）并与 LES 长程静电能量结合的，并给出一个建议的阅读 / 学习路径。

---

### 1. 整体目标与结构概览

- **脚本目标**
  - 使用 CACE 表征 + 短程原子能量网络
  - 通过 **Qeq** 从神经网络预测的电负性 `chi` 中求解原子电荷 `q_eq`
  - 利用 CACE 中实现的 **长程库伦 / Ewald（LES 风格）** 能量（键值名为 `ewald_potential`）
  - 最终得到：
    - 总能量：`CACE_energy = SR_energy + ewald_potential`
    - 总力：`CACE_forces = -∂CACE_energy/∂R`

- **代码主流程（从上到下）**
  1. 设置路径与依赖、读取 H\(_2\)O 数据集（能量 & 力）
  2. 构建 CACE 表征 `Cace(...)`
  3. 定义若干 `modules`：
     - `sr_energy`：短程原子能量项
     - `chi`：预测每个原子的 Qeq 电负性
     - `Qeq`：根据 `chi` 求解电荷、Ewald 能量、电场
     - `e_add`：`SR_energy + ewald_potential → CACE_energy`
     - `forces`：由 `CACE_energy` 自动求导得到 `CACE_forces`
  4. 用这些模块构造 `NeuralNetworkPotential`
  5. 多阶段训练（调整能量损失权重）并保存模型

---

### 2. 数据与设备：为 Qeq/LES 准备训练信号

相关代码大致位于：

```26:44:/dssg/home/acct-matxzl/matxzl/QiuQizhi/SOG-Qeq/les_fit/MLIPs/CACE-LES/water/water_perspective/cacelr-Qeq-r-4.5-nl-1-nu-3/fit_cace_new.py
logging.info("reading data")
collection = cace.tasks.get_dataset_from_xyz(train_path='../train-H2O_RPBE-D3.xyz',
                                 valid_fraction=0.1,
                                 seed=1,
                                 cutoff=cutoff,
                                 data_key={'energy': 'energy', 'forces':'forces'}, 
                                 atomic_energies={1: -5.853064337340629, 8: -2.926532168670322} # avg
                                 )
...
train_loader = cace.tasks.load_data_loader(...)
valid_loader = cace.tasks.load_data_loader(...)
```

- **关键点**
  - 数据集中提供了体系总能量 `energy` 与原子力 `forces`，**并没有显式的电荷标签**。
  - Qeq 部分（电荷和 Ewald 能量）是通过物理模型 + 神经网络学到的 `chi` 间接拟合出来的。
  - `atomic_energies` 提供元素基准能量，用于加速/稳定短程能量学习。

---

### 3. CACE 表征：短程 + 为 Qeq 提供结构信息

表征相关代码：

```51:73:/dssg/home/acct-matxzl/matxzl/QiuQizhi/SOG-Qeq/les_fit/MLIPs/CACE-LES/water/water_perspective/cacelr-Qeq-r-4.5-nl-1-nu-3/fit_cace_new.py
radial_basis = BesselRBF(cutoff=cutoff, n_rbf=6, trainable=True)
cutoff_fn = PolynomialCutoff(cutoff=cutoff)

cace_representation = Cace(
    zs=[1,8],
    n_atom_basis=2,
    embed_receiver_nodes=True,
    cutoff=cutoff,
    cutoff_fn=cutoff_fn,
    radial_basis=radial_basis,
    n_radial_basis=12,
    max_l=3,
    max_nu=3,
    num_message_passing=0,
    type_message_passing=['Bchi'],
    args_message_passing={'Bchi': {'shared_channels': False, 'shared_l': False}},
    device=device,
    timeit=False,
    forward_features=['atomic_numbers']
)
```

- **作用**
  - `Cace` 将原子结构（原子种类、位置）映射成一组局域原子特征。
  - 这些特征同时供给：
    - 短程能量模块 `sr_energy`
    - 电负性网络 `chi`（进而影响 Qeq 和长程静电）
  - `cutoff = 4.5 Å`：定义短程截断半径，长程部分由 Qeq + Ewald 处理。

---

### 4. Qeq 集成的关键模块与数据流

#### 4.1 短程能量模块 `sr_energy`

```78:82:/dssg/home/acct-matxzl/matxzl/QiuQizhi/SOG-Qeq/les_fit/MLIPs/CACE-LES/water/water_perspective/cacelr-Qeq-r-4.5-nl-1-nu-3/fit_cace_new.py
sr_energy = cace.modules.atomwise.Atomwise(n_layers=3,
                                         output_key='SR_energy',
                                         n_hidden=[32,16],
                                         use_batchnorm=False,
                                         add_linear_nn=True)
```

- **功能**
  - 将 CACE 原子特征映射成每个原子的短程能量并做加和。
  - 输出键为 `SR_energy`，后续会与长程 Ewald 能量相加。

#### 4.2 电负性网络 `chi`

```86:95:/dssg/home/acct-matxzl/matxzl/QiuQizhi/SOG-Qeq/les_fit/MLIPs/CACE-LES/water/water_perspective/cacelr-Qeq-r-4.5-nl-1-nu-3/fit_cace_new.py
chi = cace.modules.Atomwise(
    n_layers=3,
    n_hidden=[24,12],
    n_out=1,
    per_atom_output_key='chi',
    output_key = 'tot_chi',
    residual=False,
    add_linear_nn=True,
    post_process=torch.square, # square the chi values
)
```

- **功能**
  - 基于 CACE 原子特征预测每个原子的一个标量：原始 `chi_raw`。
  - 经过 `post_process=torch.square` 得到非负的 `chi`，保证物理上的稳定性。
  - 输出：
    - 原子级：`chi`（键名 `per_atom_output_key='chi'`）
    - 体系级：`tot_chi`（所有原子 `chi` 的聚合）
  - **Qeq 会把这个 `chi` 当作输入特征使用。**

#### 4.3 Qeq 模块 `ChargeEq`

```97:107:/dssg/home/acct-matxzl/matxzl/QiuQizhi/SOG-Qeq/les_fit/MLIPs/CACE-LES/water/water_perspective/cacelr-Qeq-r-4.5-nl-1-nu-3/fit_cace_new.py
Qeq = cace.modules.ChargeEq(
    dl=1.5,
    sigma=1.0,
    elements=cace_representation.zs,
    feature_key='chi',
    output_key='q_eq',
    system_charge=0.0,
    remove_self_interaction=True,
    aggregation_mode='sum',
    compute_field=True,
)
```

- **核心逻辑**
  - `feature_key='chi'`：告诉 Qeq 模块从前一层输出中读取每个原子的 `chi` 值。
  - Qeq 内部基于某种电荷平衡模型（类似传统 QEq）求解线性方程组：
    - 给定电负性 / 硬度参数、原子间库伦相互作用核
    - 在总电荷约束 `system_charge=0.0` 下求出平衡电荷 `q_eq`
  - 根据这些电荷，计算：
    - 体系长程库伦 / Ewald 能量（在本脚本中以 `ewald_potential` 的形式出现在特征字典中）
    - 如果 `compute_field=True`，还会给出与电场相关的量（用于力或响应性质）。
  - 长程部分与 LES 的联系：
    - 在 CACE‑LES 中，**LES 的“潜在 Ewald”实现就体现在这个电荷 + Ewald 能量的模块里**。
    - 神经网络并不直接输出电荷，而是输出电负性 `chi`，再经物理可解释的 Qeq 模块转为电荷和库伦能。

> 想进一步理解实现细节时，可在 CACE 源码里搜索 `class ChargeEq`，查看它如何构建库伦矩阵 / Ewald 和求解电荷。

#### 4.4 能量合成与力计算

```111:120:/dssg/home/acct-matxzl/matxzl/QiuQizhi/SOG-Qeq/les_fit/MLIPs/CACE-LES/water/water_perspective/cacelr-Qeq-r-4.5-nl-1-nu-3/fit_cace_new.py
e_add = cace.modules.FeatureAdd(feature_keys=['SR_energy', 'ewald_potential'],
                 output_key='CACE_energy')

forces = cace.modules.Forces(energy_key='CACE_energy',
                             forces_key='CACE_forces')

cace_nnp = NeuralNetworkPotential(
    representation=cace_representation,
    output_modules=[sr_energy, chi, Qeq, e_add, forces]
)
```

- **数据流顺序（按 `output_modules` 列表）**
  1. `sr_energy`：从表征得到 `SR_energy`
  2. `chi`：从表征得到 `chi`（原子级）和 `tot_chi`
  3. `Qeq`：读取 `chi`，求解 `q_eq`，并计算 `ewald_potential` 等
  4. `e_add`：
     - 从当前特征字典里取 `SR_energy` 与 `ewald_potential`
     - 相加得到体系总能量 `CACE_energy`
  5. `forces`：
     - 从 `CACE_energy` 对原子坐标自动求导，得到 `CACE_forces`

- **这一步实现了“把 Qeq/LES 集成进 CACE‑LES 能量里”**：
  - 短程能量和长程 Ewald 能量在**统一的特征字典**中被相加，梯度也会自然通过 Qeq 模块和表征模块反向传播。

---

### 5. 训练策略：通过能量和力端到端学习 Qeq

损失与训练结构：

```125:138:/dssg/home/acct-matxzl/matxzl/QiuQizhi/SOG-Qeq/les_fit/MLIPs/CACE-LES/water/water_perspective/cacelr-Qeq-r-4.5-nl-1-nu-3/fit_cace_new.py
energy_loss = cace.tasks.GetLoss(
    target_name='energy',
    predict_name='CACE_energy',
    loss_fn=torch.nn.MSELoss(),
    loss_weight=0.1
)

force_loss = cace.tasks.GetLoss(
    target_name='forces',
    predict_name='CACE_forces',
    loss_fn=torch.nn.MSELoss(),
    loss_weight=1000
)
```

```161:221:/dssg/home/acct-matxzl/matxzl/QiuQizhi/SOG-Qeq/les_fit/MLIPs/CACE-LES/water/water_perspective/cacelr-Qeq-r-4.5-nl-1-nu-3/fit_cace_new.py
for i in range(5):
    task = TrainingTask(
        model=cace_nnp,
        losses=[energy_loss, force_loss],
        metrics=[e_metric, f_metric],
        ...
    )
    task.fit(..., epochs=40, ...)
...
# 后续几次阶段调高 energy_loss 的 loss_weight，再继续训练并保存模型
```

- **端到端训练特点**
  - 没有电荷的监督标签，Qeq 参数、`chi` 网络、短程能量网络都通过匹配 `CACE_energy` 和 `CACE_forces` 自动调整。
  - 力的损失权重一开始就比较大（1000），保证几何和局域形状的正确性。
  - 随着训练阶段推进，多次提高 `energy_loss` 的权重，使能量拟合更加精细。
  - 由于 `CACE_energy` 里包含了 Qeq/Ewald 能量，**训练过程中网络会“学会”如何通过合适的 `chi → q_eq → Ewald` 来再现真实的长程静电贡献**。

---

### 6. 建议的学习 / 阅读路径

如果你想系统理解这部分代码，推荐按以下顺序：

- **步骤 1：整体跑通 & 熟悉输入输出**
  - 通读 `fit_cace_new.py` 一遍，画出你自己的数据流图：
    - 输入：结构 + `energy` + `forces`
    - 中间：`Cace` 表征 → `sr_energy`、`chi`、`Qeq`、`e_add`、`forces`
    - 输出：`CACE_energy`、`CACE_forces`
  - 在小数据集上试着跑一遍训练，观察：
    - 日志里的 `e/atom` 和 `f` 指标如何随 epoch 变化
    - 学习率、scheduler 的影响

- **步骤 2：对比“无 Qeq” 的 CACE 基线**
  - 在仓库中找一个不含 Qeq 的 CACE 训练脚本（如果有，例如只包含 `sr_energy` 和 `forces` 的版本）。
  - 对比两份脚本，重点看：
    - 多了 `chi`、`ChargeEq`、`FeatureAdd` 这几个模块
    - 输出键从 `SR_energy` 变成 `CACE_energy`，力也改由 `CACE_energy` 导出
  - 通过对比更容易看出 **“为集成 Qeq/LES 需要额外加的最小改动”**。

- **步骤 3：深入 CACE 源码理解 Qeq 实现**
  - 在 CACE 仓库中搜索：
    - `class ChargeEq`
    - 以及 Qeq 相关的 kernel/Ewald 计算函数。
  - 搞清楚：
    - 该模块如何构建库伦矩阵或 Ewald 核
    - 如何施加总电荷约束 `system_charge=0.0`
    - 输出中 `ewald_potential` 的具体定义（是总能量还是某种 per-atom 分配）
  - 建议先只看正向计算公式，后面再确认梯度和 PyTorch 自动求导链路。

- **步骤 4：分析梯度流与物理含义**
  - 从 `CACE_energy` 反向追踪：
    - `CACE_energy = SR_energy + ewald_potential`
    - `ewald_potential` 来自 `q_eq` 和位置
    - `q_eq` 来自 Qeq 线性系统，系数依赖 `chi` 和结构
    - `chi` 来自神经网络，输入是 CACE 表征
  - 画出一个简化的计算图，理解：
    - 改变原子位置如何影响电荷与能量
    - `chi` 的变化如何改变分配到每个原子的电荷，从而调节长程能量

- **步骤 5：做一些小实验加深理解**
  - 修改 Qeq 超参数（例如 `dl`, `sigma`），比较：
    - 训练收敛速度和最终误差
    - 预测电荷分布是否更加物理合理
  - 调整 `post_process`（比如去掉平方，仅做线性），看是否会导致训练不稳定或电荷发散。
  - 通过这些实验，你可以更直观地体会 Qeq/LES 部分对模型的贡献。

---

### 7. 总结：Qeq 是如何“插入” CACE‑LES 的？

- **接口层面**
  - 利用 CACE 的 `NeuralNetworkPotential` 框架，将 `ChargeEq` 作为一个普通的 `output_module` 插入到 pipeline 中。
  - 通过约定好的特征键：
    - `chi`：来自前一层 Atomwise 网络
    - `q_eq`、`ewald_potential`：由 Qeq 模块写回特征字典
  - 最终通过 `FeatureAdd` 把短程能量和 Ewald 能量合并成 `CACE_energy`。

- **物理 / 学习层面**
  - 神经网络预测的是“潜在电负性” `chi`，Qeq 模块则保证电荷和静电势满足物理约束（电中性、库伦相互作用等）。
  - LES 的“潜在 Ewald”思想：**把长程静电嵌入到一个可学习但受物理约束的潜在空间（电荷 / 电负性）里**，实现可泛化、可解释的长程补丁。

掌握了上面的结构和数据流之后，你再结合 CACE 和 LES 的论文一起看，会更容易把理论和代码对应起来。若你愿意，我也可以根据你接下来提出的更具体问题（比如：如何改成其它体系 / 其它截断半径）一起继续细化。 

