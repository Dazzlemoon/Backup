# SOG 参数初始化：论文说明与 fit-4hdnnp-NaCl 当前实现

本文档说明：(1) Ji 等 2026 文章中 CACE-SOG 的 SOG 参数初始化方法；(2) 当前 `fit-4hdnnp-NaCl` 中 ChargeEq 使用的 SOG 初始化方式；(3) 二者差异与可选对齐方式。公式用 `$...$` 与 `$$...$$`。

---

## 1. 文章中 CACE-SOG 的 SOG 初始化（Ji 等 2026）

### 1.1 动机

文章 **Section II.C “Initialization in training procedure”** 指出：早期工作中对 $\theta_{\mathrm{SOG}}$ 采用“权重设为 1、带宽对数均匀”的启发式初始化，容易与真实长程衰减偏离。本文改用 **基于 1/r 核的双边级数近似 (BSA)** 得到的高斯和参数作为初值，使 SOG 乘子更接近目标衰减、加快训练并减轻过拟合。

### 1.2 BSA 公式（实空间）

对幂律核 $1/r^{2\alpha}$ 的积分表示做梯形离散（$\alpha=1/2$ 对应库仑 $1/r$），得到 $1/r$ 的 BSA：

$$
\frac{1}{r} \approx F_{\mathrm{SOG}}(r)
= \frac{2\log b}{\sqrt{2\pi s^2}}
\sum_{m=-\infty}^{\infty} b^{-m}
\exp\Big(-\frac{r^2}{2 b^{2m} s^2}\Big).
$$

取 $m \geq 0$ 的项并截断到 $m = 0,\ldots,M-1$，得到长程部分 $F^M_{\mathrm{SOG}}(r)$。文中取：

- **$b = 2$**
- **$s = r_{\mathrm{cut}} / 1.9892536839080267$**（常数来自补充材料：使 $F^M_{\mathrm{SOG}}(r_{\mathrm{cut}}) = 1/r_{\mathrm{cut}}$ 连续，且 $b=2$ 时相对误差约 $\leq 10^{-3}$）

即实空间长程核为（$M$ 为高斯个数）：

$$
F^M_{\mathrm{SOG}}(r)
= \frac{2\log b}{\sqrt{2\pi s^2}}
\sum_{\ell=0}^{M-1} b^{-\ell}
\exp\Big(-\frac{r^2}{2 b^{2\ell} s^2}\Big).
$$

### 1.3 Fourier 空间初始化（文章中的可训练参数）

将 $F^M_{\mathrm{SOG}}$ 做 Fourier 变换得到闭式（文中 Eq. (13)），并以此初始化 **Fourier 乘子** $\hat{f}_{\theta_{\mathrm{SOG}}}(k) = \sum_{\ell=0}^{M-1} w_\ell e^{-k^2/s_\ell^2}$ 的 **权重与带宽**：

$$
w_\ell = 4\pi b^{2\ell} s^2 \log b,\qquad
s_\ell = \frac{\sqrt{2}}{b^{2\ell} s^2},\qquad
\ell = 0,\ldots,M-1.
$$

即 **$\theta_{\mathrm{SOG}} = \{w_\ell, s_\ell\}_{\ell=0}^{M-1}$** 由 $b$、$s$（及 $r_{\mathrm{cut}}$）和 $M$ 唯一确定。文中默认 **$M=12$**，且 $M$ 可根据库仑长程尾的截断误差式 (15) 选取。

---

## 2. fit-4hdnnp-NaCl 当前的 SOG 初始化

### 2.1 使用的模块与参数形式

- **脚本**：`fit-4hdnnp-NaCl/fit-cace-SOG.py`
- **长程**：仅用 **ChargeEq** 的 SOG 核构造 $A$ 并计算长程能量，输出 key 为 `SOG_potential`；**不**使用 `SOGPotential`。
- **SOG 分量数**：`Fourier_node = 18`，作为 `ChargeEq(..., sog_num_components=18)` 的 `sog_num_components`（即 $L$）。
- **核形式**（`cace/modules/charge_eq.py` 中当 `shared_sog_potential is None` 时）：
  $$
  A_{ij} = K(r_{ij}) = \sum_{\ell=1}^{L}
  \mathrm{weights}_\ell\,
  \exp\big(-\alpha_\ell\, r_{ij}^2\big).
  $$
  可训练参数为 **$\mathrm{sog\_log\_alpha}$**（即 $\log\alpha_\ell$）和 **$\mathrm{sog\_weights}$**（即 $\mathrm{weights}_\ell$）。

### 2.2 当前初始化代码（ChargeEq）

在 `cace/modules/charge_eq.py` 的 `__init__` 中，当 `use_sog_kernel=True` 且 `shared_sog_potential is None` 时：

```python
# 初始化一组覆盖不同 length scale 的宽度，单位与坐标一致
# 用 log-parameterization 保证 alpha > 0
init_sigmas = torch.linspace(0.5, 5.0, sog_num_components)
init_alphas = 1.0 / (init_sigmas ** 2 + 1e-6)
self.sog_log_alpha = nn.Parameter(torch.log(init_alphas))
# 初始化权重为接近库仑核的衰减（粗略均匀）
self.sog_weights = nn.Parameter(torch.ones(sog_num_components) / sog_num_components)
```

即：

- **$\sigma_\ell$**：在 $[0.5, 5.0]$ 上线性均匀取 $L$ 个值（与 `cutoff` 无显式关系）。
- **$\alpha_\ell = 1/(\sigma_\ell^2 + 10^{-6})$**，**$\mathrm{sog\_log\_alpha}_\ell = \log \alpha_\ell$**。
- **$\mathrm{sog\_weights}_\ell = 1/L$**（均匀权重）。

因此，当前实现是 **与截断半径无关的启发式**：用固定区间上的 $\sigma$ 线性分布 + 等权，**没有**使用文章中的 BSA 或 $r_{\mathrm{cut}}$。

---

## 3. 论文与当前实现的对比

| 项目           | Ji 等 2026 (CACE-SOG)              | fit-4hdnnp-NaCl (ChargeEq SOG)     |
|----------------|-------------------------------------|-------------------------------------|
| 参数形式       | Fourier：$w_\ell, s_\ell$            | 实空间：$\alpha_\ell, \mathrm{weights}_\ell$ |
| 初始化依据     | BSA 近似 $1/r$，$b=2$，$s=r_{\mathrm{cut}}/1.989\ldots$ | $\sigma_\ell \in [0.5, 5.0]$ 线性 + 等权 |
| 与 $r_{\mathrm{cut}}$ 关系 | 有（$s \propto r_{\mathrm{cut}}$） | 无                                   |
| 高斯个数       | 默认 $M=12$                         | $L=18$（`Fourier_node`）            |

若希望与文章一致，可在 ChargeEq 侧用 BSA 推导出的 **实空间** $\alpha_\ell$ 和 $\mathrm{weights}_\ell$ 做初值（见下节）。

---

## 4. 可选：按 BSA 为 ChargeEq 做与论文一致的初始化

BSA 实空间长程核（$m \geq 0$ 截断）为

$$
F^M_{\mathrm{SOG}}(r)
= \frac{2\log b}{\sqrt{2\pi s^2}}
\sum_{\ell=0}^{M-1} b^{-\ell}
\exp\Big(-\frac{r^2}{2 b^{2\ell} s^2}\Big).
$$

与 ChargeEq 的 $K(r) = \sum_\ell \mathrm{weights}_\ell \exp(-\alpha_\ell r^2)$ 对应：

- **$\alpha_\ell = \dfrac{1}{2 b^{2\ell} s^2}$**，其中 $s = r_{\mathrm{cut}} / 1.9892536839080267$，$b=2$。
- **$\mathrm{weights}_\ell = \dfrac{2\log b}{\sqrt{2\pi s^2}}\, b^{-\ell}$**（可选再乘一常数使整体尺度与现有实现一致）。

在 `fit-4hdnnp-NaCl` 中 `cutoff = 5.29`，若取 $r_{\mathrm{cut}} = \mathtt{cutoff}$，则：

- $s = 5.29 / 1.9892536839080267 \approx 2.659$
- 对 $\ell = 0,\ldots,L-1$ 算出 $\alpha_\ell$ 与 $\mathrm{weights}_\ell$，赋给 `sog_log_alpha` 和 `sog_weights` 的初值，即可得到与论文 BSA 一致的 SOG 初始化（实空间形式）。

实现时需在构造 `ChargeEq` 之后、训练前，用上述公式覆盖 `sog_log_alpha` 和 `sog_weights` 的 `data`，或增加一个 `ChargeEq.init_sog_from_bsa(r_cut, M, b=2)` 的接口并在 `fit-cace-SOG.py` 中调用。

---

## 5. 小结

- **文章**：SOG 参数由 **BSA 近似 1/r** 得到，$b=2$，$s = r_{\mathrm{cut}}/1.989\ldots$，Fourier 空间 $w_\ell,\,s_\ell$ 由式 (14) 给出，默认 $M=12$。
- **fit-4hdnnp-NaCl**：ChargeEq 内 SOG 使用 **与 $r_{\mathrm{cut}}$ 无关** 的启发式：$\sigma_\ell \in [0.5, 5.0]$ 线性、等权，$L=18$。
- 若需与论文一致，可按 Section 4 用 BSA 导出实空间 $\alpha_\ell$、$\mathrm{weights}_\ell$ 并设为 ChargeEq 的 SOG 初值。
