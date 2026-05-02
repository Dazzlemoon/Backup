## CACE-SOG-Qeq 中 NaCl 训练脚本的说明（`fit-4hdnnp-NaCl/fit-cace-SOG.py`）

本文简要说明当前 `CACE-SOG-Qeq` 文件夹里用于 NaCl 的 Qeq+SOG 模型设置、训练配置以及期望结果。公式使用 `$...$` 与 `$$...$$`。

---

## 1. 使用的模型结构（高层图）

整体模型是一个 **短程 CACE 神经网络 + Qeq 长程电荷平衡** 的组合：

- **表示层**：`Cace(...)`
  - 元素：`zs = [11, 17]`（Na, Cl）
  - 半径截断：`cutoff = 5.29`
  - 基函数：`BesselRBF` + 多项式 cutoff
  - 只做一次 message passing（`num_message_passing = 0, type_message_passing = ["Bchi"]`），输出每原子表示 `node_feats`。

- **短程能量**：`sr_energy = Atomwise(...)`
  - 三层 MLP，输入 CACE 表示，输出 `SR_energy`（每结构短程能量）。

- **电负性网络**：`chi = Atomwise(...)`
  - 三层 MLP，输出 per-atom 电负性 `chi_i`（`post_process=torch.square` 确保非负）。

- **Qeq 电荷平衡**：`ChargeEq(...)`
  - 输入：`chi_i`、原子类型 `Z_i`、体系总电荷 `Q_{\text{tot}}`。
  - 内部硬度矩阵：`J = diag(J_i^2)`，`J_i` 为按元素可训练参数。
  - 核矩阵 $A$ 采用 **SOG（高斯和）近似的 $1/r$ 核**：
    $$
    A_{ij} = K(r_{ij}) = \sum_{\ell=1}^{L} w_\ell \exp(-\alpha_\ell r_{ij}^2),
    $$
    其中 $L = \text{Fourier\_node} = 18$。
  - Qeq 线性系统：
    $$
    \begin{pmatrix}
    A+J & \mathbf{1} \\
    \mathbf{1}^\top & 0
    \end{pmatrix}
    \begin{pmatrix}
    q \\
    \lambda
    \end{pmatrix}
    =
    \begin{pmatrix}
    -\chi \\
    Q_{\text{tot}}
    \end{pmatrix},
    $$
    由 `torch.linalg.solve` 直接求解，得到平衡电荷 `q_eq`。
  - 长程能量（在本脚本中命名为 `SOG_potential`）：
    $$
    E_{\text{long}} = \frac12 q^\top A q.
    $$

- **总能量聚合**：`FeatureAdd(["SR_energy", "SOG_potential"] → "CACE_energy")`

- **力**：`Forces(energy_key="CACE_energy", forces_key="CACE_forces", calc_stress=False)`
  - 从总能量对坐标求梯度得到原子力。

### 1.1 SOG 高斯和参数：全体系共用同一组（代码中的约定）

在 **CACE+SOG** 的当前实现里，高斯和（SOG）核是对 **距离** 的标量函数，且 **所有原子对共用同一组参数**，不是“不同两个原子用不同参数”：

- **核形式**：$K(r_{ij}) = \sum_{\ell=1}^{L} w_\ell \exp(-r_{ij}^2/s_\ell^2)$（或等价的 $\sum_\ell \text{weights}_\ell \exp(-\alpha_\ell r_{ij}^2)$），只依赖 $r_{ij}$，不依赖原子种类或 (i,j) 的“类型”。
- **参数**：全体系共用 **一份** $(w_\ell, s_\ell)$（或 `sog_weights` / `sog_log_alpha`），即 **一套** 长度为 $L$ 的向量，对所有 $(i,j)$ 使用同一 $K(r)$。

**代码依据**：

- `cace/modules/sog.py` 中 `SOGPotential`：`self.wl`、`self.sl` 为 shape `[L]` 的模块参数，在 `compute_potential_SOG_realspace` 里对任意 (i,j) 用同一组 `wl`/`sl` 计算 $K(r_{ij})$。
- `cace/modules/charge_eq.py` 中 `_build_A_sog`：若使用 `shared_sog_potential`，则用其 `wl`/`sl`；否则用本模块的 `sog_weights`/`sog_log_alpha`。构造 $A_{ij} = K(r_{ij})$ 时，对所有 (i,j) 使用同一套参数，无 per-atom 或 per-pair 的下标。

**关于论文**：你提到的 Ji 等 2025 的 PDF（Machine-Learning Interatomic Potentials for Long-Range Systems）在本地路径下，此处无法打开。若论文中有 “per-species”“per-pair” 或按原子类型区分的 SOG 参数描述，需要你对照论文公式与图注自行确认；**当前仓库实现是“全体系一套 SOG 参数”**。

最终模型 `NeuralNetworkPotential` 的输出 keys 主要包括：

- `CACE_energy`：总能量
- `CACE_forces`：总力
- `q_eq`：平衡电荷
- `SOG_potential`：长程库仑/Qeq 能量

---

## 2. 训练数据与总电荷的处理

- **数据源**：`fit-4hdnnp-NaCl/NaCl.xyz`，是带
  `Properties=species:S:1:pos:R:3:forces:R:3:charge:R:1` 的 extxyz 文件。
- **读取方式**：不依赖 ASE，而是使用 `cace.data.extxyz_charge.get_dataset_from_extxyz_with_charge`：
  - 手工解析每帧的：
    - `species`（Na/Cl）
    - `pos`（原子坐标）
    - `forces`（力）
    - `charge`（per-atom 电荷）
    - `Lattice`（晶胞）
    - `energy`（若存在）
  - 构造 `AtomicData`，per-atom 电荷写入 `additional_info["charge"]`。

- **每结构总电荷**：
  - 利用模块 `SystemChargeFromAtomicCharges(charges_key="charge", output_key="system_charge")`：
    $$
    Q_{\text{tot}}^{(g)} = \sum_{i \in \text{graph } g} q_i^{\text{(data)}},
    $$
    并写入 `data["system_charge"]`。
  - `ChargeEq(system_charge=None, system_charge_key="system_charge")` 时会优先使用这个 per-graph 总电荷作为约束。

- **DataLoader**：直接对 `AtomicData` 列表使用 `torch_geometric.DataLoader` 构建 `train_loader` 与 `valid_loader`。

---

## 3. 训练轮数（epochs）与调度

在 `fit-cace-SOG.py` 中训练分为四个阶段：

1. **阶段 1**：
   - 外层循环 5 次，每次调用一个新的 `TrainingTask`，训练 `epochs=40`。
   - 能量 loss 权重：`0.1`，力 loss 权重：`1000`。
   - 学习率调度：`StepLR(step_size=20, gamma=0.5)`。
2. **阶段 2**：
   - 接着 `task.update_loss` 把能量 loss 权重改为 `1`，力 loss 仍为 `1000`。
   - 再训练 `epochs=100`。
3. **阶段 3**：
   - 能量 loss 权重改为 `10`，力 loss 仍为 `1000`。
   - 再训练 `epochs=100`。
4. **阶段 4**：
   - 能量 loss 权重改为 `1000`（与力项同量级），力 loss 仍为 `1000`。
   - 再训练 `epochs=100`。

因此，等效总 epoch 数约为：

- 阶段 1：`5 * 40 = 200` epoch
- 阶段 2：`100` epoch
- 阶段 3：`100` epoch
- 阶段 4：`100` epoch

合计约 **`500` 个 epoch**（其中阶段 1 等价于 5 次重启的 40 epoch 训练，方便 early-stopping 或中途观察）。

---

## 4. 损失函数（loss function）

在每个训练阶段，`TrainingTask` 的损失由两个部分组成：

- **能量损失**（`energy_loss`）：
  - 使用 `torch.nn.MSELoss()`：
    $$
    \mathcal{L}_E = \text{MSE}(E_{\text{pred}}, E_{\text{ref}}),
    $$
  - 权重在各阶段依次为 `0.1 → 1 → 10 → 1000`。

- **力损失**（`force_loss`）：
  - 使用 `torch.nn.MSELoss()`：
    $$
    \mathcal{L}_F = \text{MSE}(\mathbf{F}_{\text{pred}}, \mathbf{F}_{\text{ref}}),
    $$
  - 权重始终为 `1000`。

总损失为：

$$
\mathcal{L} = w_E \,\mathcal{L}_E + w_F \,\mathcal{L}_F.
$$

其中 $(w_E, w_F)$ 按阶段改变，用于从“先对齐力场细节”逐渐过渡到“同时精确总能量与力”的训练过程。

---

## 5. SOG 核的权重与带宽参数如何更新

在 Qeq SOG 核中，`ChargeEq` 使用参数：

- `sog_log_alpha`：长度为 `L` 的可训练参数，对应高斯核中 $\alpha_\ell$ 的对数：
  $$
  \alpha_\ell = \exp(\text{sog\_log\_alpha}_\ell) > 0.
  $$
- `sog_weights`：长度为 `L` 的可训练权重：
  $$
  w_\ell = \text{sog\_weights}_\ell.
  $$

在 `_build_A_sog` 中，核矩阵为：

$$
A_{ij} = \sum_{\ell=1}^{L} w_\ell \exp(-\alpha_\ell r_{ij}^2).
$$

因为 $A$ 进入 Qeq 能量：

$$
E_{\text{QEq}}(q) = \frac12 q^\top A q + \frac12 q^\top J q + \chi^\top q,
$$

而 `q_eq` 又参与总能量与力的计算，整体 loss 通过反向传播自然会对 `sog_log_alpha` 与 `sog_weights` 产生梯度。优化器（例如 `Adam`）在每次 `loss.backward()` 之后，对这些参数做梯度更新：

- `alpha_\ell` 与 `w_\ell` 会自动调节，以便：
  - 更好地拟合参考 NaCl 数据中的长程静电行为；
  - 在数值上提供更稳定且物理解读清晰的 Qeq 核矩阵。

注意：本配置中 **不再显式使用 `SOGPotential`**。SOG 核仅体现在 `ChargeEq` 构造的 $A$ 上，所有 SOG 参数的更新都通过 Qeq 路径完成。

---

## 6. 期望达到的结果

这一套 CACE-SOG-Qeq 模型在 NaCl 上的训练目标可以概括为：

- **能量与力的精度**：
  - 期望在验证集上：
    - 能量误差达到每原子 $10^{-3}\,\text{eV}$ 量级或更好；
    - 力的均方根误差（RMSE）在 $10^{-2}\,\text{eV/Å}$ 量级左右。
  - 具体数值可在 `log_*` 与保存的模型评估中查看。

- **电荷与总电荷约束**：
  - 对每个构型：
    - `system_charge` 精确等于输入数据中 per-atom `charge` 的总和；
    - Qeq 解出的 `q_eq` 在数值上逼近期望电荷分布，同时满足总电荷约束。

- **物理可解释性**：
  - 通过 SOG 形式的核矩阵 $A$，Qeq 的长程能量行为与真实库仑相互作用保持一致（在局部截断下的近似意义上）；
  - 高斯分量的权重与带宽（`w_\ell, \alpha_\ell`）提供了一种可解释的“屏蔽核”近似，可以在后续分析中检查其与 $1/r$ 的拟合质量。

- **与后续加速方法的衔接**：
  - 当前实现仍是显式构造 $A$ 并直接求解线性系统，复杂度为 $O(N^2)$（构造） + $O(N^3)$（求解）。
  - 但由于 $A$ 已经被写成 SOG 核形式，为后续引入：
    - SOG + FFT/NUFFT 快速 matvec
    - PCG/MINRES 等迭代法
  提供了直接的接口，这在 `Qeq_A_matrix_gaussian_sum_acceleration.md` 与 `SOG_FFT_NUFFT_matvec_math.md` 中已有详细分析。

整体而言，本脚本对应的是一套“**可学习的 SOG-Qeq 长程核 + CACE 短程神经网络**”的 NaCl 基准实现，用于验证：

- 在共享或可学习的 SOG 核下，
- Qeq 是否能够在保证物理约束（电荷守恒）的同时，
- 达到与原始 CACE-SOG 势类似甚至更好的能量/力拟合效果，
- 并为后续的算子化加速方案打基础。

---

## 8. 训练结束后会产生哪些输出文件？

以 `fit-4hdnnp-NaCl/` 目录为中心，主要会生成以下几类文件：

- **标准输出 / 标准错误日志**
  - `log`, `log_*.out`（或 `slurm-*.out`）：
    - 每个 epoch 的训练/验证损失：`loss`, `energy_loss`, `force_loss`。
    - 评估指标：`e/atom`（能量误差）、`f`（力 RMSE）。
    - 学习率、epoch 编号、用时等信息。
  - `err`, `err_*.out`：
    - 训练中的 warning（例如 CUDA context 初始化提示）。
    - 若训练异常终止，这里会有完整的 Python 堆栈。

- **模型权重文件（PyTorch `state_dict`）**
  - `hydrocarbon-model.pth`：阶段 1 结束后保存的模型。
  - `hydrocarbon-model-2.pth`：阶段 2 后。
  - `hydrocarbon-model-3.pth`：阶段 3 后。
  - `hydrocarbon-model-4.pth`：阶段 4（全部训练）结束后的最终模型。
  - 每个 `.pth` 文件中包含：
    - CACE 表示层权重。
    - 短程 SR `Atomwise` 网络权重。
    - 电负性网络 `chi` 参数。
    - Qeq 硬度参数 `J_raw`。
    - SOG 核参数（`sog_log_alpha`, `sog_weights` 或共享 SOG 参数）。
    - 以及（若保存）优化器的状态，用于继续训练或做 fine-tune。

- **损失曲线与中间快照（`loss_data/N_18_*/` 子目录）**
  - 训练过程中 `TrainingTask` 会在 `save_folder = loss_data/N_18_...` 中记录：
    - 每个 epoch 的训练/验证损失（通常是文本或 `npy/npz` 格式）。
    - 若开启“保存 best model”，会在验证集 loss 最优时额外保存一个模型快照。
  - 这些文件可以用于：
    - 绘制能量/力损失随 epoch 的收敛曲线。
    - 回溯训练中是否出现过拟合或震荡。
    - 选择验证集表现最佳的模型作为最终 NaCl 势函数。

- **批处理作业输出（集群环境）**
  - 若通过 `sbatch run.slurm` 提交作业，Slurm 会额外生成：
    - `slurm-<jobid>.out`：包含脚本运行的标准输出/错误。
  - 本 README 中所述的训练/日志逻辑在本地直接运行 `python fit-cace-SOG.py` 时同样适用，只是输出会进入 `log` / `err` 文件，而非 `slurm-*.out`。

总之，完成一次 NaCl 训练后，你可以通过：

- 查看 `log_*.out` / `loss_data/*` 判断训练是否收敛、loss 大小如何；
- 通过 `hydrocarbon-model-4.pth`（或验证集 best 的 `.pth`）在 notebook 中用 `cace.tasks.EvaluateTask` 做预测；
- 对比能量/力误差以及学到的 SOG 核参数，评估这套 CACE-SOG-Qeq NaCl 模型是否达到了预期的精度与物理解读。


