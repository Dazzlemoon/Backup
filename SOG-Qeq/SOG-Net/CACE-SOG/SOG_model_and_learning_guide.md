## CACE‑SOG 中 SOG‑Net 模型实现与学习路径说明

本文基于：

- 论文：*Machine-Learning Interatomic Potentials for Long-Range Systems*（Ji, Liang, Xu，PRL 2025，SOG‑Net）
- 代码：`SOG-Net/CACE-SOG/` 目录，尤其是：
  - `cace/modules/sog.py`
  - `cace/models/combined.py`
  - 示例脚本：
    - `fit-4hdnnp-NaCl/fit-cace-SOG.py`
    - `pure-SR-Nacl/fit-cace-SOG.py`
    - `fit-dipeptides/fit-dipeptides-SOG.py`

目标是解释：**论文中提出的 SOG‑Net 模型是如何在 CACE‑SOG 代码里落地的**，以及**你可以如何系统地学习与修改这套代码**。

---

### 1. 论文中 SOG‑Net 模型的核心结构（简要回顾）

论文中，总能量写成短程（SR）+ 长程（LR）两部分：

$$
E = E_{\text{SR}} + E_{\text{LR}}.
$$

- **短程能量 $E_{\text{SR}}$**：
  - 使用任意局域描述子（这里是 CACE）+ 拟合网络：
    $$
    E_{\text{SR}} = \sum_{i=1}^N E_{\text{SR}, i}
                  = \sum_{i=1}^N f_{\theta_{\text{SR}}}(D_i),
    $$
    其中 $D_i$ 为原子 $i$ 的局域特征（满足平移、旋转、排列对称性）。

- **潜在变量 $q_{i,\eta}$**：
  - SOG‑Net 使用一个「latent network」从局域环境生成每个原子、每个通道 $\eta$ 上的潜在量 $q_{i,\eta}$。
  - 这些 $q_{i,\eta}$ 不一定是真正的电荷，可以理解为「不同衰减形式长程成分的权重」。

- **SOG 长程卷积层**：
  - 在倒空间中，用 **Sum‑of‑Gaussians（SOG）核** 对 $q_{i,\eta}$ 做卷积，得到 LR 能量：
    - 核的幅度和宽度（variance/shift）由可训练参数控制；
    - 通过 NUFFT/快速卷积使得复杂度接近线性；
  - 每一层（每个 \(\eta\)）有一组 SOG 多项式系数，在训练中自动学习不同的长程衰减行为（例如 \(1/r\)、\(1/r^6\)、偶极衰减等混合）。

代码层面，你会看到：

- **局域部分**：一个普通的 CACE‑NNP；
- **潜在变量生成**：一个 per‑atom `Atomwise` 网络，输出特征键叫 `q`；
- **SOG 层**：`SOGPotential` 模块，对 `q` 做长程卷积；
- **组合总能量**：`CombinePotential` 把 SR 和 LR 两个 NNP 的输出按照权重加总。

---

### 2. CACE‑SOG 的整体代码结构与 SOG‑Net 对应关系

#### 2.1 目录结构（与 SOG‑Net 模型直接相关的部分）

- **核心模块**
  - `cace/modules/sog.py`：SOG 长程势能 `SOGPotential` 的 PyTorch 实现。
  - `cace/models/combined.py`：`CombinePotential`，负责把多个 NNP（例如 SR + LR）组合成一个总模型。
- **示例任务脚本**
  - `fit-4hdnnp-NaCl/fit-cace-SOG.py`：NaCl 三维周期体系，CACE + SOG，典型「点电荷+SOG」示例。
  - `pure-SR-Nacl/fit-cace-SOG.py`：只用 SOG‑Net 做长程修正的 NaCl 示例。
  - `fit-dipeptides/fit-dipeptides-SOG.py`：多原子种类、分子体系（Spice dipeptides）的示例。

#### 2.2 SR 部分：CACE 表征 + 原子能量网络

在 `fit-4hdnnp-NaCl/fit-cace-SOG.py` 中（NaCl 示例）：

```84:99:SOG-Net/CACE-SOG/fit-4hdnnp-NaCl/fit-cace-SOG.py
atomwise = cace.modules.atomwise.Atomwise(n_layers=3,
                                         output_key='CACE_energy',
                                         n_hidden=[32,16],
                                         use_batchnorm=False,
                                         add_linear_nn=True)

forces = cace.modules.forces.Forces(energy_key='CACE_energy',
                                    forces_key='CACE_forces')

print("building CACE NNP")
cace_nnp_sr = NeuralNetworkPotential(
    input_modules=None,
    representation=cace_representation,
    output_modules=[atomwise, forces]
)
```

- 这部分对应论文中的 \(E_{\text{SR}} = \sum f_{\theta_{\text{SR}}}(D_i)\)：
  - `cace_representation = Cace(...)` 生成 \(D_i\)；
  - `atomwise` 是 \(f_{\theta_{\text{SR}}}\)；
  - `forces` 从 `CACE_energy` 自动求导得到 SR 力。

#### 2.3 潜在变量网络：生成 \(q_i\)

继续看 NaCl 示例中的潜在网络（Latent network）：

```102:111:SOG-Net/CACE-SOG/fit-4hdnnp-NaCl/fit-cace-SOG.py
q = cace.modules.Atomwise(
    n_layers=3,
    n_hidden=[24,12],
    n_out=1,
    per_atom_output_key='q',
    output_key = 'tot_q',
    residual=False,
    add_linear_nn=True,
    bias=False)
```

- 对应论文图 1 中的 **Latent Network**：
  - 输入同样是 CACE 表征的 per‑atom 特征；
  - 输出的是每个原子一个标量 \(q_i\)（这里只有一个通道 \(\eta\)，更一般可以是多通道 \(q_{i,\eta}\)）。
  - `per_atom_output_key='q'`：在数据字典里留下 per‑atom 的 `data['q']`，正是后面 `SOGPotential` 读取的特征。

#### 2.4 SOG 长程层：`SOGPotential`

`SOGPotential` 的实现位于：

```11:24:SOG-Net/CACE-SOG/cace/modules/sog.py
class SOGPotential(nn.Module):
    def __init__(self,
                 N_dl=1,  # Fourier modes
                 bandwidth_num = 12,
                 external_field = None, # external field
                 external_field_direction: int = 0, # external field direction, 0 for x, 1 for y, 2 for z
                 charge_neutral_lambda: float = None,
                 remove_self_interaction=False,
                 feature_key: str = 'q',
                 output_key: str = 'SOG_potential',
                 aggregation_mode: str = "sum",
                 compute_field: bool = False,
                 Periodic: bool = False,
                 ):
```

关键点：

- **输入特征**：`feature_key='q'`
  - 从数据字典 `data['q']` 中读取潜在变量 $q_i$，对应论文中的 $q_{i,\eta}$。
- **SOG 核参数**：
  - `bandwidth_num` 决定 SOG 展开中高斯的个数；
  - `self.amplitude_1` 和 `self.shift_1` 是每个高斯核的初始幅度与「宽度」（shift），与论文中的「Sum‑of‑Gaussians multiplier」对应；
  - 当 `Periodic=True` 时，`wl` 和 `sl` 会按论文里适合倒空间卷积的形式重新参数化。
- **长程卷积实现**：
  - 周期性盒子：`compute_potential_SOG_triclinic` 或 `compute_potential_SOG_triclinic_NUFFT` 对应论文中的「Long Range Convolution Network」与 NUFFT 加速；
  - 非周期系统：`compute_potential_SOG_realspace` 使用实空间高斯和。

长程势能的核心公式在倒空间实现（截取一段逻辑）：

```123:176:SOG-Net/CACE-SOG/cace/modules/sog.py
cell_inv = torch.linalg.inv(box)
G = 2 * torch.pi * cell_inv.T
...
nvec = torch.stack(torch.meshgrid(n1, n2, n3, indexing="ij"), dim=-1).reshape(-1, 3)
...
kvec = (nvec.float() @ G).to(device)
...
k_dot_r = torch.matmul(r_raw, kvec.T)  # [n, M]
...
S_k_real = (q.unsqueeze(2) * cos_k_dot_r.unsqueeze(1)).sum(dim=0)
S_k_imag = (q.unsqueeze(2) * sin_k_dot_r.unsqueeze(1)).sum(dim=0)
S_k_sq = S_k_real**2 + S_k_imag**2  # [M]
...
kfac = self.wl.view(1, 1, 1, -1) * torch.exp(k_sq.unsqueeze(-1) * min_term)
...
pot = (factors * kfac * S_k_sq).sum() / volume
```

对应论文中：

- $S(\mathbf{k}) = \sum_i q_i e^{i \mathbf{k}\cdot\mathbf{r}_i}$（结构因子）；
- SOG 多项式 $(g_{\theta_\eta} \circ \rho_\eta)(k)$ 通过高斯核 `kfac` 实现；
- 能量大致为
  $$
  E_{\text{LR}} \sim \sum_{\mathbf{k}} |S(\mathbf{k})|^2 \cdot \text{SOG}(k),
  $$
  这在代码里就是对 `S_k_sq * kfac` 的加权求和。

#### 2.5 SR+LR 组合：`CombinePotential`

组合模块定义在：

```7:29:SOG-Net/CACE-SOG/cace/models/combined.py
class CombinePotential(nn.Module):
    def __init__(
        self,
        potentials: List[nn.Module],
        potential_keys: List[Dict],
        operation = None,
    ):
        """
        Combine multiple potentials into a single potential.
        ...
        pot1 = {'CACE_energy': 'CACE_energy_intra',
        'CACE_forces': 'CACE_forces_intra',
        'weight': 1.
        }
        pot2 = {'CACE_energy': 'CACE_energy_inter',
        'CACE_forces': 'CACE_forces_inter',
        'weight': 0.01,
        }
        """
```

在 NaCl 示例中：

```127:136:SOG-Net/CACE-SOG/fit-4hdnnp-NaCl/fit-cace-SOG.py
pot2 = {'CACE_energy': 'SOG_potential', 
        'CACE_forces': 'SOG_forces',
        'weight': 1
       }

pot1 = {'CACE_energy': 'CACE_energy', 
        'CACE_forces': 'CACE_forces',
       }

cace_nnp = cace.models.CombinePotential([cace_nnp_sr, cace_nnp_lr], [pot1,pot2])
```

- `cace_nnp_sr`：只含 SR 能量 `CACE_energy` 和 `CACE_forces`；
- `cace_nnp_lr`：只含 LR 能量 `SOG_potential` 和 `SOG_forces`；
- `CombinePotential`：
  - 先分别调用两个子模型，得到它们各自的输出；
  - 按键名把 LR 输出重命名为统一的 `CACE_energy` / `CACE_forces` 并乘以权重；
  - 最终在 `default_operation` 中通过 `torch.stack(...).sum(0)` 把多路能量/力叠加，总体上实现：
    $$
    E = E_{\text{SR}} + w \, E_{\text{LR}},
    $$
    对应论文的 $E = E_{\text{SR}} + E_{\text{LR}}$。

---

### 3. 从「论文公式」到「代码对象」的一一对应

下表是一个快速对应关系（非严格数学符号，而是帮助你在脑子里对齐）：

- **短程部分**
  - 论文：$D_i$（短程描述子）  
    代码：`Cace(...)` 输出的 per‑atom 特征；
  - 论文：$E_{\text{SR},i} = f_{\theta_{\text{SR}}}(D_i)$  
    代码：`Atomwise(output_key='CACE_energy', ...)`；
  - 论文：$E_{\text{SR}} = \sum_i E_{\text{SR},i}$  
    代码：`NeuralNetworkPotential(..., output_modules=[atomwise, forces])` 自动在内部对 per‑atom 能量求和。

- **潜在变量部分**
  - 论文：Latent network 输出 $q_{i,\eta}$  
    代码：`Atomwise(per_atom_output_key='q', output_key='tot_q', ...)`。

- **SOG 卷积层**
  - 论文：结构因子 $S_\eta(\mathbf{k}) = \sum_i q_{i,\eta} e^{i\mathbf{k}\cdot\mathbf{r}_i}$  
    代码：`compute_potential_SOG_triclinic` 函数中 `S_k_real / S_k_imag / S_k_sq` 的计算；
  - 论文：SOG Multiplier $(g_{\theta_\eta}\circ\rho_\eta)(k)$  
    代码：`self.wl`, `self.sl` 等参数，以及生成的 `kfac`；
  - 论文：$E_{\text{LR}} = \sum_{\mathbf{k}} |S(\mathbf{k})|^2\cdot\text{SOG}(k)$  
    代码：`pot = (factors * kfac * S_k_sq).sum() / volume`。

- **总模型**
  - 论文：$E = E_{\text{SR}} + E_{\text{LR}}$  
    代码：`CombinePotential([cace_nnp_sr, cace_nnp_lr], [pot1, pot2])`。

---

### 4. 推荐的学习与阅读路径

下面是一个渐进式的学习顺序，适合你边看论文边对照代码。

#### 步骤 1：先跑通一个最简单的 NaCl 示例

1. 重点脚本：`pure-SR-Nacl/fit-cace-SOG.py`。
2. 建议：
   - 通读这个脚本一遍，画出数据流图：
     - 输入：`NaCl.xyz` 中的结构 + 能量 + 力；
     - 中间：
       - `Cace` 表征；
       - SR NNP：`atomwise` + `forces`；
       - 潜在网络：`q = Atomwise(..., per_atom_output_key='q')`；
       - 长程 SOG：`SOGPotential(N_dl=1, Periodic=True, ...)`；
       - `CombinePotential` 合并 SR 与 LR；
     - 输出：总能量 `CACE_energy`，总力 `CACE_forces`。
   - 选择较小的训练轮数，实际跑一次训练，看 loss 和 metric 日志的收敛情况。

通过这一步，你会建立起「脚本层」上的直觉：SOG 只是 CACE‑NNP 后面再接一个长程层。

#### 步骤 2：深入 `SOGPotential` 源码（`cace/modules/sog.py`）

1. 建议顺序：
   - 先看 `__init__`，搞清楚每个参数的物理与数学意义：
     - `N_dl`, `bandwidth_num`, `amplitude_1`, `shift_1`；
     - `Periodic` 打开/关闭时的差异。
   - 然后看 `forward`：
     - 关注 `r = data['positions']`、`q = data[self.feature_key]` 的形状和含义；
     - 了解 batch 结构（`batch_now` 和 `unique_batches` 的逻辑）。
   - 最后看 `compute_potential_SOG_triclinic` 与 `compute_potential_SOG_realspace`：
     - 对照论文里的 SOG‑Net 图，确认：
       - 如何从实空间坐标构造 \(\mathbf{k}\) 网格；
       - 如何计算 \(S(\mathbf{k})\)；
       - 如何通过高斯核 `kfac` 得到 SOG multiplier；
       - 如何组合得到能量 `pot`。
2. 如果你熟悉 Ewald/FFT，会发现 SOG 层与传统 Ewald 在形式上很像，只是核从 \(1/k^2\) 变成了可学习的高斯组合。

#### 步骤 3：对比多个体系的训练脚本

1. 对比 NaCl 和二肽：
   - `fit-4hdnnp-NaCl/fit-cace-SOG.py`
   - `fit-dipeptides/fit-dipeptides-SOG.py`
2. 观察：
   - CACE 表征的 `zs`、`cutoff`、`max_l`、`max_nu` 等如何随体系而变；
   - 潜在网络的结构是否有调整（层数、hidden size）；
   - SOG 参数 `bandwidth_num` / `N_dl` 是否随体系调节；
   - 损失权重 `loss_weight` 如何平衡能量与力。

通过对比不同体系，你会更容易提炼出「SOG‑Net 的不变部分」和「随体系可调的超参数」。

#### 步骤 4：结合论文，手写一遍关键公式与代码映射

1. 从论文中摘出关键几个公式（比如 $E_{\text{SR}}$、$S(\mathbf{k})$、SOG multiplier、$E_{\text{LR}}$）。
2. 对照 `sog.py` 里的实现，在纸上写出：
   - 哪个变量对应 $q_i$、$\mathbf{r}_i$、$\mathbf{k}$；
   - 哪一段代码实现了 $S(\mathbf{k})$；
   - 哪一段是 SOG multiplier；
   - 最终能量表达式和代码里的 `pot` 是如何对应的。

这一步做完，你基本可以「不用看注释，直接读懂 SOGPotential」。

#### 步骤 5：做小实验加深理解

可以尝试几个方向的小实验（在 NaCl 示例上动手最方便）：

1. **改变 SOG 展开阶数**
   - 修改 `bandwidth_num`（以及 `amplitude_1`/`shift_1` 初始值的长度）；
   - 观察：
     - 训练速度是否明显变化；
     - 最终能量/力误差有多大差别；
     - 模型对远距离相互作用的敏感度是否改变。
2. **切换周期/非周期模式**
   - 在简单的测试脚本中比较 `Periodic=True` 和 `Periodic=False` 的结果；
   - 对比 `compute_potential_SOG_triclinic` 与 `compute_potential_SOG_realspace` 的数值行为。
3. **把 SOG 关掉，只保留 SR 模型**
   - 在 `CombinePotential` 中只保留 SR 部分（示例里已经给了注释掉 LR 的选项）；
   - 对比：
     - SR‑only vs SR+SOG 的能量与力误差；
     - 模型对系统尺寸（原子数、盒子大小）变化的鲁棒性。

这些实验能让你更直观地感受：**SOG 层在「补长程」上到底贡献了什么**。

---

### 5. 如果你想继续深入 / 修改模型

在完全弄清楚以上内容后，你可以尝试：

- 在 `CACE-SOG-Qeq` 中，把 Qeq 生成的$q_{\text{eq}}$ 换成 SOG 层的输入 `q`，即做「Qeq+SOG」的混合长程方案；
- 在 `SOGPotential` 中：
  - 增加多通道 q（多个 $\eta$）并让不同 $\eta$ 共享或不共享 SOG 核；
  - 替换当前固定初始化的 `amplitude_1`/`shift_1` 为完全可训练的参数，并比较数值稳定性；
- 在 `CombinePotential` 中尝试不同的加权策略或归一化方式，以适应更复杂多物理场系统。

如果你告诉我你接下来最想针对哪一个体系（NaCl / 水 / 多肽 / 其它），我可以再帮你写一份更「任务定制版」的阅读与实验路线图，甚至是具体的超参数和脚本改写建议。

