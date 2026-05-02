## 在 CACE-SOG-Qeq 中引入可学习的 SOG 核 \(A_{\text{SOG}}\)（原型说明）

本文说明我们对 `ChargeEq` 做的一个**最小改动**：在保持原有 Ewald 形式不变的前提下，增加一个可选分支，用 **SOG（高斯和）近似 \(1/r\)** 来构造 Qeq 的核矩阵 \(A\)，并使其参数可训练。当前实现仍然显式构造 \(A\)（未接入 PCG/FFT 加速），主要目的是为后续 SOG 加速与参数共享实验打基础。

---

### 1. 代码改动位置与开关

- 文件：`cace/modules/charge_eq.py`
- 类：`ChargeEq`

新增了两个构造参数：

```python
class ChargeEq(nn.Module):
    def __init__(...,
                 system_charge_key: str = "system_charge",
                 # 可选：用 SOG 近似 1/r 构造 A，而不是直接从 EwaldPotential 取 A
                 use_sog_kernel: bool = False,
                 sog_num_components: int = 4):
        ...
        self.use_sog_kernel = use_sog_kernel
```

- 默认 `use_sog_kernel=False`，完全保留原有 Ewald 版本的行为，不影响现有模型；
- 若在构造 `ChargeEq` 时设置 `use_sog_kernel=True`，则 Qeq 的核矩阵由 **SOG 核**构造。

---

### 2. 新增的 SOG 核参数

在 `__init__` 中，当 `use_sog_kernel=True` 时，会额外初始化一组高斯和参数：

```python
if self.use_sog_kernel:
    # 初始化一组覆盖不同 length scale 的宽度
    init_sigmas = torch.linspace(0.5, 5.0, sog_num_components)
    init_alphas = 1.0 / (init_sigmas ** 2 + 1e-6)
    self.sog_log_alpha = nn.Parameter(torch.log(init_alphas))
    # 初始化权重为接近库仑核的衰减（粗略均匀）
    self.sog_weights = nn.Parameter(torch.ones(sog_num_components) / sog_num_components)
```

- `sog_log_alpha`：用 log-parameterization 确保 \(\alpha_\ell>0\)，对应高斯核 \(\exp(-\alpha_\ell r^2)\) 的宽度；
- `sog_weights`：每个高斯分量的权重 \(w_\ell\)，初始化为平均权重；
- 这两者都是 `nn.Parameter`，因此可以在训练中通过反向传播同时更新。

**目标**：在训练开始前，用一组覆盖多尺度的高斯核粗略拟合 \(1/r\)；训练过程中，只在这个物理合理的族内做微调，学到“更贴近真实体系的有效库仑核”。

---

### 3. 前向中 A 的构造逻辑

在原有 `forward` 循环中，我们原先是直接用 `EwaldPotential` 生成核矩阵：

```python
for i in unique_batches:
    ...
    J_i_now = J_i[mask]
    # 第二个返回量在本实现中就是 A 矩阵
    _, A_now = self.ep.compute_potential_triclinic(
        r_now,
        torch.eye(r_now.size(0), device=r_now.device),
        box_now,
        compute_field=self.compute_field,
    )
    q_eq, lambda_eq = self._compute_q_eq(A_now, chi_now, J_i_now, system_charge_now)
```

现在根据开关改成：

```python
J_i_now = J_i[mask]

# 构造核矩阵 A：
# - 默认：从 EwaldPotential 取出等效的 A；
# - 可选：用 SOG 高斯和近似 1/r 构造 A_sog（可学习）。
if self.use_sog_kernel:
    A_now = self._build_A_sog(r_now, box_now)
else:
    _, A_now = self.ep.compute_potential_triclinic(
        r_now,
        torch.eye(r_now.size(0), device=r_now.device),
        box_now,
        compute_field=self.compute_field,
    )
```

除 A 的来源不同外，后续 Qeq 求解 `_compute_q_eq` 的数学形式完全保持不变：

```python
q_eq, lambda_eq = self._compute_q_eq(A_now, chi_now, J_i_now, system_charge_now)
ewald_energy = 0.5 * q_eq.unsqueeze(1).T @ A_now @ q_eq.unsqueeze(1)
```

---

### 4. SOG 版核矩阵 `_build_A_sog` 的实现（原型）

新增的内部方法：

```python
def _build_A_sog(self, r: torch.Tensor, cell: torch.Tensor) -> torch.Tensor:
    """
    用 SOG（高斯和）近似 1/r 构造核矩阵 A_sog。

    当前实现：
    - 作为原型，直接用最小镜像下的欧氏距离近似 |r_i - r_j|；
    - 未显式使用 FFT/NUFFT 加速，仍然显式构造 A（O(N^2)），
      但核形式已经是 SOG，可在后续替换为算子形式以配合 PCG 加速。
    """
    device, dtype = r.device, r.dtype
    N = r.size(0)

    # pairwise distance matrix
    diff = r.unsqueeze(0) - r.unsqueeze(1)  # [N, N, 3]
    dist = torch.linalg.norm(diff, dim=-1) + 1e-8  # [N, N]

    # SOG 核： sum_l w_l * exp(-alpha_l * r^2)
    alphas = torch.exp(self.sog_log_alpha).to(device=device, dtype=dtype)  # [L]
    weights = self.sog_weights.to(device=device, dtype=dtype)  # [L]
    r2 = dist ** 2  # [N, N]

    # [L, N, N]
    sog_terms = torch.exp(-alphas.view(-1, 1, 1) * r2.unsqueeze(0))
    A_sog = (weights.view(-1, 1, 1) * sog_terms).sum(dim=0)  # [N, N]

    A_sog = A_sog.to(dtype=dtype)
    return A_sog
```

当前实现有几个重要的**设计取舍**：

- **物理形状**：`A_sog[i,j]` 是形如
  $$
  A_{ij}^{\text{SOG}} = \sum_\ell w_\ell \exp(-\alpha_\ell r_{ij}^2)
  $$
  的核，对 \(r\) 的依赖与 SOG 长程势相同，天然逼近 \(1/r\)；
- **周期性处理**：目前仅用直接坐标差异 `r_i-r_j` 计算欧氏距离，未做严格 PBC wrap（作为原型足够）。若需要严格 LES 周期性，可在此处参照 `EwaldPotential.compute_potential_triclinic` 的做法进行最小镜像或倒空间处理；
- **复杂度**：仍然是显式构造 \(N\times N\) 的核矩阵，复杂度约为 \(O(N^2)\)。但一旦核形式固定为 SOG，高层求解可以很自然地替换为：
  - 不显式构造 \(A\)，只实现 `apply_A(v)`；
  - 使用 SOG+FFT/NUFFT 做快速 matvec；
  - 用 PCG/MINRES 迭代解线性系统——这就是 `SOG_Qeq_A_matrix_SOG_acceleration.md` 中提出的路线。

---

### 5. 训练与反向传播：SOG 参数如何被“同时更新”

由于 `sog_log_alpha` 与 `sog_weights` 都是 `nn.Parameter`，且：

- 既参与 `_build_A_sog` 的构造；
- 又通过 \(E_{\text{QEq}} = \tfrac12 q^\top A_{\text{SOG}} q + \cdots\) 影响损失；

因此在 PyTorch 的计算图中：

- 只要你的 loss 中包含依赖 `q_eq` / Qeq 能量的项（例如总能量、力、或电荷正则项），调用
  ```python
  loss.backward()
  ```
  时，链式法则会自动将梯度反向传播到 `sog_log_alpha` 与 `sog_weights` 上；
- 如果同时还有其它模块（例如将来某个 `SOGPotential`）也使用这组参数，那么多条路径上的梯度会在这两个参数上**自动累加**，在优化器 `step()` 时一起更新。

换句话说：**技术上已准备好“统一可学习的 SOG 核参数”这一接口，是否启用/共享则由你在构建模型时的配置决定。**

---

### 6. 与完整 SOG 加速方案的关系

当前这一步只是把 Qeq 的 \(A\) 从“**Ewald 黑盒输出**”扩展为“**可选的、显式 SOG 核（可学习）**”。在 `SOG_Qeq_A_matrix_SOG_acceleration.md` 中所描述的更完整方案包括：

- 不显式构造 \(A\)，只定义 `apply_A(v)`；
- 在 `apply_A(v)` 内使用 SOG+FFT/NUFFT 等快速卷积；
- 用 PCG/MINRES + Schur 补来迭代解线性系统。

这些都可以在当前原型的基础上平滑迁移：

- 保留 `sog_log_alpha` / `sog_weights` 及其物理初始化；
- 将 `_build_A_sog` 中的“显式构造 \(A\)”替换为“算子化 matvec”；
- 在 `_compute_q_eq` 之外增加一个基于算子的迭代求解器，实现真正的 \(O(N\log N)\) 级别加速。

---

### 7. 小结

- 我们为 `ChargeEq` 增加了一个可选开关 `use_sog_kernel`，在开启时：
  - Qeq 的核矩阵 \(A\) 用 SOG（高斯和）近似 \(1/r\) 构造；
  - 新增的 SOG 参数 `sog_log_alpha`、`sog_weights` 为可训练参数，可通过反向传播更新；
  - Qeq 的数学形式（\(\tfrac12 q^\top A q + \tfrac12 J q^2 + \chi q\) 及总电荷约束）保持不变。
- 当前 `_build_A_sog` 仍然显式构造 \(N^2\) 级别的核矩阵，主要目标是：
  - 对齐 CACE-SOG-Qeq 文档中关于 “用 SOG 近似 Qeq 核矩阵” 的设计思路；
  - 为后续引入“算子 + 迭代求解 + FFT/NUFFT 加速”提供一个可运行的、物理上合理的原型。
- 在此基础上，你可以进一步：
  - 与 `SOGPotential` 共享部分或全部 SOG 参数（例如宽度 \(\sigma_\ell\)）；
  - 将 `_build_A_sog` 替换为真正的 SOG-FFT/PCG 实现，实现对 Qeq 矩阵求解的加速。

