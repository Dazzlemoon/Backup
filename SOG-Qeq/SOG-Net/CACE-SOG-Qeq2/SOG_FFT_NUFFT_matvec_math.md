## 周期体系下：用 SOG + FFT/NUFFT 快速计算 $y = Av$ 的数学原理

本文只讲“数学怎么来的”，目标是解释：当 $A$ 的核是 **SOG（高斯和）**时，为什么在周期体系中可以把 $y=Av$ 变成一次（或少数几次）FFT/NUFFT，从而把复杂度从 $O(N^2)$ 降到接近 $\tilde O(N\log N)$。

文中公式使用 `$...$` 与 `$$...$$`。

---

## 1. 问题设定：$A$ 是高斯和核的周期卷积

给定周期晶胞（一般为三斜晶胞）$\Omega$，体积 $V=|\det(\mathbf{B})|$，其中 $\mathbf{B}\in\mathbb{R}^{3\times 3}$ 是晶胞矩阵（列向量为晶格基矢）。

有 $N$ 个粒子位置 $\{\mathbf{r}_i\}_{i=1}^N$（在晶胞内，或视为模晶格等价），以及一组权重（例如 Qeq 迭代里的向量分量）$\{v_i\}$。

我们希望计算

$$
y_i = (Av)_i = \sum_{j=1}^N K(\mathbf{r}_i-\mathbf{r}_j)\,v_j,
$$

其中核 $K$ 取 **SOG 形式**：

$$
K(\mathbf{r}) = \sum_{\ell=1}^{L} w_\ell\,\exp(-\alpha_\ell \|\mathbf{r}\|^2).
$$

在周期体系中，真正的周期核是“对所有晶格平移求和”的周期延拓：

$$
K_{\mathrm{per}}(\mathbf{r}) = \sum_{\mathbf{n}\in\mathbb{Z}^3} K(\mathbf{r}+\mathbf{B}\mathbf{n}).
$$

于是

$$
y_i = \sum_{j=1}^N K_{\mathrm{per}}(\mathbf{r}_i-\mathbf{r}_j)\,v_j.
$$

这已经是一个**周期卷积型的核求和**。

---

## 2. 关键事实：高斯的傅里叶变换仍是高斯

在 $\mathbb{R}^3$ 上，

$$
\mathcal{F}\{e^{-\alpha\|\mathbf{r}\|^2}\}(\mathbf{k})
= \int_{\mathbb{R}^3} e^{-\alpha\|\mathbf{r}\|^2}\,e^{-i\mathbf{k}\cdot\mathbf{r}}\,d\mathbf{r}
= \left(\frac{\pi}{\alpha}\right)^{3/2} e^{-\|\mathbf{k}\|^2/(4\alpha)}.
$$

因此对 SOG 核：

$$
\widehat{K}(\mathbf{k})
= \sum_{\ell=1}^L w_\ell\left(\frac{\pi}{\alpha_\ell}\right)^{3/2} \exp\left(-\frac{\|\mathbf{k}\|^2}{4\alpha_\ell}\right).
$$

这一步是 SOG 能被 FFT/NUFFT 高效处理的根本原因：**频域乘子是解析、平滑、快速衰减的高斯和**。

---

## 3. 周期化之后：核求和变成倒格点的傅里叶级数

周期化核 $K_{\mathrm{per}}$ 的傅里叶级数在倒格点上展开。

- 倒格矢：令 $\mathbf{G} = 2\pi\,\mathbf{B}^{-T}$，则任意倒格点向量可以写成

$$
\mathbf{k}_{\mathbf{m}} = \mathbf{G}\,\mathbf{m},\qquad \mathbf{m}\in\mathbb{Z}^3.
$$

- 周期核的级数（忽略收敛细节，SOG 的高斯使得一切非常良性）：

$$
K_{\mathrm{per}}(\mathbf{r})
= \frac{1}{V}\sum_{\mathbf{m}\in\mathbb{Z}^3} \widehat{K}(\mathbf{k}_{\mathbf{m}})\,e^{i\mathbf{k}_{\mathbf{m}}\cdot\mathbf{r}}.
$$

将其代入 $y_i$：

$$
\begin{aligned}
 y_i
&= \sum_{j=1}^N v_j\,\frac{1}{V}\sum_{\mathbf{m}} \widehat{K}(\mathbf{k}_{\mathbf{m}})\,e^{i\mathbf{k}_{\mathbf{m}}\cdot(\mathbf{r}_i-\mathbf{r}_j)}\\
&= \frac{1}{V}\sum_{\mathbf{m}} \widehat{K}(\mathbf{k}_{\mathbf{m}})\,e^{i\mathbf{k}_{\mathbf{m}}\cdot\mathbf{r}_i}\underbrace{\sum_{j=1}^N v_j e^{-i\mathbf{k}_{\mathbf{m}}\cdot\mathbf{r}_j}}_{S(\mathbf{k}_{\mathbf{m}})}.
\end{aligned}
$$

这里出现了典型的“结构因子”形式：

$$
S(\mathbf{k}) = \sum_{j=1}^N v_j\,e^{-i\mathbf{k}\cdot\mathbf{r}_j}.
$$

于是算法可以分成三步：

1. 计算所有倒格点上的 $S(\mathbf{k}_{\mathbf{m}})$（这是 **非均匀点到规则频域网格** 的变换）。
2. 频域逐点乘以乘子 $\widehat{K}(\mathbf{k}_{\mathbf{m}})$。
3. 回到粒子位置：

$$
y_i = \frac{1}{V}\sum_{\mathbf{m}} \big(\widehat{K}(\mathbf{k}_{\mathbf{m}})S(\mathbf{k}_{\mathbf{m}})\big) e^{i\mathbf{k}_{\mathbf{m}}\cdot\mathbf{r}_i},
$$

这是 **规则频域网格到非均匀点** 的变换。

这两个“非均匀点 ↔ 规则频域网格”正是 NUFFT 的两种基本类型。

---

## 4. NUFFT 的两种方向与 $Av$ 的对应关系

为了把三斜晶胞统一成 $[0,2\pi)^3$ 上的相位，我们常做分数坐标变换：

- 分数坐标：$\mathbf{s}_i = \mathbf{B}^{-1}\mathbf{r}_i \in [0,1)^3$
- 相位坐标：$\mathbf{x}_i = 2\pi\mathbf{s}_i \in [0,2\pi)^3$

则

$$
\mathbf{k}_{\mathbf{m}}\cdot\mathbf{r}_i
= (\mathbf{G}\mathbf{m})\cdot \mathbf{r}_i
= 2\pi \mathbf{m}\cdot \mathbf{s}_i
= \mathbf{m}\cdot \mathbf{x}_i.
$$

此时：

- **Type-1 NUFFT（NU points → uniform Fourier modes）** 计算

$$
F_{\mathbf{m}} \approx \sum_{j=1}^N c_j\,e^{\pm i\mathbf{m}\cdot\mathbf{x}_j}.
$$

取 $c_j=v_j$ 且符号取负号，就得到了 $S(\mathbf{k}_{\mathbf{m}})$。

- **Type-2 NUFFT（uniform Fourier modes → NU points）** 计算

$$
f(\mathbf{x}_i) \approx \sum_{\mathbf{m}} F_{\mathbf{m}}\,e^{\pm i\mathbf{m}\cdot\mathbf{x}_i}.
$$

取 $F_{\mathbf{m}} = \widehat{K}(\mathbf{k}_{\mathbf{m}})S(\mathbf{k}_{\mathbf{m}})$，符号取正号，就得到 $y_i$（差一个 $1/V$ 的缩放）。

因此 $Av$ 的 NUFFT 版计算流程是：

1. **NUFFT type-1**：$S_{\mathbf{m}} \leftarrow \sum_j v_j e^{-i\mathbf{m}\cdot\mathbf{x}_j}$
2. **频域乘子**：$\widetilde{S}_{\mathbf{m}} \leftarrow \widehat{K}(\mathbf{k}_{\mathbf{m}})\,S_{\mathbf{m}}$
3. **NUFFT type-2**：$y_i \leftarrow \frac{1}{V}\sum_{\mathbf{m}} \widetilde{S}_{\mathbf{m}} e^{+i\mathbf{m}\cdot\mathbf{x}_i}$

---

## 5. 为什么是 $\tilde O(N\log N)$？误差从哪来？

NUFFT 的思想是把非均匀点的指数求和转成：

- 把点源用一个紧支撑的窗函数（通常也是高斯/KB 等）**散射到规则网格**（gridding）
- 在网格上做 FFT（$O(M\log M)$，$M$ 是网格点数）
- 再从网格 **插值回粒子位置**（degridding）

当网格大小 $M$ 与粒子数 $N$ 同阶时，总复杂度接近 $O(N\log N)$。

误差主要由三部分控制：

- 网格截断（频域模式数有限）
- gridding/degridding 的窗函数插值误差
- 浮点误差

工程上用 `eps`（如 `finufft` 的 `eps=1e-4`）控制目标精度。

---

## 6. SOG 的额外便利：频域乘子可预计算且是高斯和

对每个分量 $\ell$，频域乘子是

$$
\widehat{K}_\ell(\mathbf{k}) = w_\ell\left(\frac{\pi}{\alpha_\ell}\right)^{3/2}\exp\left(-\frac{\|\mathbf{k}\|^2}{4\alpha_\ell}\right).
$$

所以总乘子是它们的和：

$$
\widehat{K}(\mathbf{k}) = \sum_{\ell=1}^L \widehat{K}_\ell(\mathbf{k}).
$$

这意味着：

- 你只需要做一次 type-1 和一次 type-2 NUFFT；
- 中间在频域做一次逐点乘法（乘子可在给定晶胞与 $\{\alpha_\ell,w_\ell\}$ 后预计算/缓存）。

---

## 7. 与 Qeq 的关系：用在 PCG 的 `apply_A(v)`

在 Qeq 里你要解的是 $(A+J)q + \mathbf{1}\lambda = -\chi$。

如果用迭代法（PCG/MINRES），核心操作是反复计算

$$
(A+J)u = Au + J\odot u.
$$

当 $Au$ 由上面的 SOG+NUFFT 快速给出后：

- 每次迭代开销主要是 2 次 NUFFT + 少量逐点操作；
- 总开销约为“迭代步数 × 单次 matvec 成本”。

这就是为什么把 $A$ 改成 SOG 后，非常适合用 NUFFT/FFT 加速 Qeq。

---

## 8. 一个极简的“公式到实现”映射（与你仓库 `sog.py` 对齐）

在 `cace/modules/sog.py` 的 `compute_potential_SOG_triclinic_NUFFT` 里，你能看到和本文一致的结构：

- 把 $\mathbf{r}$ 映射到分数/相位坐标（`r_frac`, `r_in`）
- 构造频域网格 $\mathbf{m}$（`meshgrid(n1,n2,n3)`）
- 计算乘子 $kfac \approx \widehat{K}(\mathbf{k}_{\mathbf{m}})$（对 SOG 分量求和）
- `finufft_type1` 得到 $S_{\mathbf{m}}$
- 逐点乘 `con_sog = kfac * recon`
- `finufft_type2` 回到粒子点得到 $y_i$（再做体积缩放）

所以本文给出的推导可以直接作为“为什么这些张量操作是对的”的数学解释。

---

## 9. 小结

- SOG 核在频域仍是高斯和，因此周期化后可以写成倒格点傅里叶级数。
- $Av$ 可以分解为：
  - 结构因子 $S(\mathbf{k})$（type-1 NUFFT）
  - 频域乘子 $\widehat{K}(\mathbf{k})$（逐点乘）
  - 回到粒子点（type-2 NUFFT）
- 这为 Qeq 的迭代求解提供了高效的 `apply_A(v)`，从而把“直接解密集系统”替换为“快速 matvec + 迭代法”。
