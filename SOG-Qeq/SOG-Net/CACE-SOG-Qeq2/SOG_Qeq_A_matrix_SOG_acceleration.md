## 用 SOG（高斯和）逼近 Qeq 中的库仑核 \(1/r\)，以及如何加速 Qeq 线性方程求解

本文面向你当前关注的问题：

- Qeq 里构造的核矩阵 \(A\)（本质上来自长程库仑势）能否用 **SOG（sum of Gaussians）** 形式逼近，即把 \(1/r\) 写成多个高斯核的加权和；
- 逼近后的 SOG 参数能否与 `SOGPotential` 里的参数（例如 `wl/sl`）设置成相同；
- 一旦 \(A\) 变成 SOG 形式，Qeq 的“矩阵求逆/线性方程求解”有没有更快的方法（避免显式构造/分解大矩阵）。

文中所有数学公式都用 `$...$` 或 `$$...$$`。

---

### 1. Qeq 的线性方程：你真正要解的是什么？

经典 Qeq 能量（省略常数）可以写成：

$$
E(q)
=
\frac12 q^\top A q
\;+\;
\frac12 \sum_{i=1}^N J_i q_i^2
\;+\;
\sum_{i=1}^N \chi_i q_i,

$$


并带有总电荷约束：

$$
\mathbf{1}^\top q = Q.
$$

其中：

- $q\in\mathbb{R}^N$ 是原子电荷；
- $A\in\mathbb{R}^{N\times N}$ 是库仑核矩阵（周期体系通常对应 Ewald 形式；非周期可近似为 $A_{ij}\approx 1/|r_i-r_j|$ 并处理自项）；
- $J_i>0$ 是“硬度”（hardness），形成对角矩阵 $J=\mathrm{diag}(J_1,\dots,J_N)$；
- $\chi\in\mathbb{R}^N$ 是电负性特征（由网络预测）；
- $\mathbf{1}$ 是全 1 向量。

对拉格朗日函数

$$
\mathcal{L}(q,\lambda)=E(q)-\lambda(\mathbf{1}^\top q-Q)
$$

求一阶条件得到块线性系统（与代码里的 `torch.linalg.solve` 对应）：

$$
\begin{pmatrix}
A+J & \mathbf{1}\\
\mathbf{1}^\top & 0
\end{pmatrix}
\begin{pmatrix}
q\\
\lambda
\end{pmatrix}
=
\begin{pmatrix}
-\chi\\
Q
\end{pmatrix}.
$$

记

$$
M := A+J,\qquad b:=-\chi,
$$

则系统为：

$$
Mq + \mathbf{1}\lambda = b,\qquad \mathbf{1}^\top q = Q.
$$

**所以你真正想加速的是**：在给定几何构型时，反复求解带约束的线性系统（尤其是 $M^{-1}$ 的作用）。

---

### 2. 为什么 \(1/r\) 可以用高斯和逼近？（关键数学依据）

一个非常关键的恒等式是：

$$
\frac{1}{r}
=
\frac{2}{\sqrt{\pi}}\int_0^\infty e^{-t^2 r^2}\,dt,
\qquad r>0.
$$

证明思路很简单：令 $u=tr$，则

$$
\int_0^\infty e^{-t^2 r^2}\,dt
=
\frac{1}{r}\int_0^\infty e^{-u^2}\,du
=
\frac{1}{r}\cdot \frac{\sqrt{\pi}}{2}.
$$

于是得到上式。

这意味着：只要对积分做数值求积（quadrature），就得到一个**高斯和**近似：

$$
\frac{1}{r}
\approx
\sum_{\ell=1}^{L} w_\ell\, e^{-\alpha_\ell r^2},
$$

其中可取

$$
\alpha_\ell = t_\ell^2,\qquad
w_\ell = \frac{2}{\sqrt{\pi}}\,\omega_\ell,
$$

而 $\{t_\ell,\omega_\ell\}$ 来自某种对 $\int_0^\infty$ 的求积方案（例如变换后的 Gauss–Legendre / Gauss–Laguerre / 双指数变换等）。

这就是“用 SOG 逼近 \(1/r\)”最干净的一条数学路线：**库仑核是高斯核的连续叠加，离散化后就是高斯和**。

---

### 3. 把 Qeq 的 \(A\) 写成 SOG 形式意味着什么？

在最朴素（非周期、忽略自项）的情况下，库仑核矩阵可写为：

$$
A_{ij} = \frac{1}{\lVert r_i-r_j\rVert},\qquad i\neq j.
$$

用 SOG 逼近后，你可以构造

$$
A_{ij}
\approx
\sum_{\ell=1}^{L} w_\ell \exp\!\Big(-\alpha_\ell \lVert r_i-r_j\rVert^2\Big),
\qquad i\neq j,
$$

并在对角上按需要设定（例如 $A_{ii}=0$ 或加入自能修正项）。

这样做的直接好处不是“$A$ 变稀疏”，而是：

- 你**可以不用显式形成 $A$**；
- 你只需要能快速计算 $y=A x$（矩阵-向量乘，matvec）；
- 由于每一项都是“高斯核卷积”，可以用 FFT/NUFFT 思想把 $A x$ 做到接近 $O(N\log N)$（或按网格大小标度），从而配合迭代法加速求解。

---

### 4. 能否把 \(A\) 的 SOG 参数设置成与 `SOGPotential` 相同？

**可以尝试，但要区分“参数共享”有几种层级**。

#### 4.1 共享“形状参数”（宽度） vs 共享“幅度参数”

如果你把 `SOGPotential` 的核（简写）记为：

$$
K_{\text{SOG}}(r)
=
\sum_{\ell=1}^{L} \tilde w_\ell \exp(-\tilde\alpha_\ell r^2),
$$

那么把 Qeq 的 $A$ 也写成类似形式：

$$
A_{ij}\approx \sum_{\ell=1}^{L} w_\ell \exp(-\alpha_\ell \lVert r_i-r_j\rVert^2).
$$

你可以选择：

- **只共享宽度**：$\alpha_\ell \equiv \tilde\alpha_\ell$，但 $w_\ell$ 独立；
- **同时共享宽度和幅度**：$\alpha_\ell \equiv \tilde\alpha_\ell$ 且 $w_\ell\equiv \tilde w_\ell$；
- **共享初值但不绑定**：初始化相同，训练时分别更新。

#### 4.2 物理一致性与可辨识性（identifiability）风险

要注意 Qeq 的 $A$ 在物理上代表“库仑相互作用核”（或其 Ewald 版本），它不应该任意改变形状，否则 Qeq 得到的 $q_{\text{eq}}$ 会变成“为了拟合能量而扭曲的电荷”，失去物理意义。

因此更稳妥的做法通常是：

- 若你希望 Qeq 电荷保持物理性：让 \(A\) 逼近的核尽量接近 \(1/r\)（或 Ewald），即 **固定**（或强约束）住 $w_\ell,\alpha_\ell$；
- `SOGPotential` 的参数用于学习“真实体系里剩余的长程尾部/屏蔽/多体效应”，它可以自由学习。

换句话说：**“让 Qeq 的 \(A\) 用 SOG 表示”主要是为了加速 matvec，而不是为了把它学成任意核。**

---

### 5. 用 SOG 表示 \(A\) 后，Qeq 求解怎么加速？（重点）

你要解的是带约束的块系统：

$$
\begin{pmatrix}
M & \mathbf{1}\\
\mathbf{1}^\top & 0
\end{pmatrix}
\begin{pmatrix}
q\\
\lambda
\end{pmatrix}
=
\begin{pmatrix}
b\\
Q
\end{pmatrix},
\qquad M=A+J.
$$

核心思路是：**不要显式求逆，也不要对 $M$ 做稠密分解；改用迭代法，只需要快速 matvec：$x\mapsto Ax$。**

#### 5.1 用 Schur complement 把约束消掉（公式层面更清晰）

由

$$
Mq + \mathbf{1}\lambda = b
$$

得

$$
q = M^{-1}(b-\mathbf{1}\lambda).
$$

代入约束 $\mathbf{1}^\top q = Q$：

$$
\mathbf{1}^\top M^{-1}(b-\mathbf{1}\lambda) = Q
$$

即

$$
\lambda
=
\frac{\mathbf{1}^\top M^{-1} b - Q}{\mathbf{1}^\top M^{-1}\mathbf{1}}.
$$

然后

$$
q = M^{-1}(b-\mathbf{1}\lambda).
$$

这告诉你：只要你能算两次（或少数几次）线性系统

$$
M x = b,\qquad M y = \mathbf{1},
$$

就能得到 $\lambda$ 和 $q$。因此“带约束”问题可以转成“解若干个 $M$ 的线性系统”。

#### 5.2 用 PCG / MINRES：只需 matvec，不显式形成 \(A\)

如果 $M$ 是对称正定（在合理 $J_i>0$、$A$ 半正定的设置下通常成立），可以用预条件共轭梯度（PCG）解：

$$
M x = rhs.
$$

PCG 每一步只需要：

- 一个 matvec：$v\mapsto Mv = Av + Jv$；
- 若干点积和 saxpy（向量线性组合）。

其中：

- $Jv$ 是对角乘法，$O(N)$；
- $Av$ 若用 SOG 表示并用 FFT/NUFFT 计算，可做到近似 $O(N\log N)$ 或更低常数。

所以，**一旦你把 $A$ 变成“可快速 matvec 的 SOG 卷积算子”，就可以用 PCG 彻底绕开稠密求逆。**

#### 5.3 如何对 SOG 核做快速 matvec（与 `SOGPotential` 同源）

如果

$$
(Ax)_i = \sum_{j\neq i} \Big(\sum_{\ell=1}^L w_\ell e^{-\alpha_\ell \lVert r_i-r_j\rVert^2}\Big)\, x_j,
$$

则每一项

$$
(A^{(\ell)}x)_i := \sum_{j\neq i} w_\ell e^{-\alpha_\ell \lVert r_i-r_j\rVert^2} x_j
$$

都是“高斯核对点源的卷积”。在周期盒子中可走 FFT/NUFFT 路线：

1. **gridding**：把 $\sum_j x_j \delta(\cdot-r_j)$ 近似投到网格；
2. **FFT**：到 $k$ 空间；
3. **乘以高斯 multiplier**：$\exp(-c_\ell k^2)$（对应实空间高斯核的傅里叶像）；
4. **IFFT**：回到实空间；
5. **gathering**：在 $r_i$ 位置插值取值。

这和你在 `SOGPotential` 里看到的“结构因子 + multiplier + 求和”的计算图是同一类思想。

因此：**用 SOG 表示 \(A\) 的最大价值是：你可以复用 SOGPotential 的快速卷积基础设施，把 \(Ax\) 做快，从而让 PCG/MINRES 变快。**

#### 5.4 预条件（Preconditioning）：决定迭代步数

仅 matvec 快还不够，PCG 收敛步数取决于条件数。常用预条件包括：

- **Jacobi（对角）预条件**：
  $$
  P = \mathrm{diag}(M),\qquad P^{-1}\approx M^{-1}.
  $$
  它便宜但可能不够强。

- **block / two-level 预条件**（更适合核矩阵）：
  - 近场（短距离）部分用稠密小块精确或 ILU；
  - 远场（长距离）部分用 SOG-FFT 的快速算子近似。

- **固定几何的重用**：
  在 MD 或同一 batch 内几何变化小的情况下，可以复用上一步的迭代解作 warm-start（初值），迭代次数通常会下降。

#### 5.5 复杂度直观对比

设每个构型有 $N$ 个原子：

- 直接 `torch.linalg.solve`（稠密）通常是 $O(N^3)$；
- PCG：
  - 每步成本 $\approx$ matvec 成本；
  - 若 matvec 用 SOG+FFT/NUFFT：每步 $\tilde O(N\log N)$；
  - 总成本 $\approx$ 迭代步数 $\times$ $\tilde O(N\log N)$。

当 $N$ 较大时，只要迭代步数不是特别大，PCG 会比 $O(N^3)$ 显著更有优势。

---

### 6. 为什么 Qeq 的核要尽量逼近 \(1/r\)，而不是“随便学一个核”？

在上面的讨论里，我们一直强调：无论是直接用 Ewald \(1/r\)，还是用 SOG 近似 \(1/r\)，**Qeq 里的 \(A\) 都应该尽量保持“库仑核”的物理形状**，而不是变成一个完全自由的“黑盒核”。这背后有几个物理与数值上的原因：

- **Qeq 的角色是“静电骨架”，不是万能能量拟合器**  
  - 经典 Qeq / PQEq 的思想是：把总能量拆成
    $$
    E_{\text{Total}} = E_{\text{short}} + E_{\text{QEq}},\quad
    E_{\text{QEq}} \approx \text{带极化的静电能} \sim \tfrac12 q^\top A q+\cdots.
    $$
  - 这里的 \(A\) 本质上是“（高斯/Ewald/极化修正后的）库仑核”，回答的问题是：**“给定一组电荷 \(q\)，它们在介质中的库仑相互作用有多大？”**
  - 如果 \(A\) 变成任意学出来的核，\(q\) 就不再对应电势/场/多极矩的物理解读，而只是某个黑箱二次型的参数，**Qeq 电荷的物理性会严重下降**。

- **屏蔽、多体、残余长程效应不应该全部塞进 \(A\)**  
  - 真实体系确实有介电屏蔽、多体效应、金属/电解质中的集体行为，但工业/物理上常见的分层做法是：
    - 用“形状接近 \(1/r\)”的核（Ewald、SOG、PQEq）承担**主导的库仑骨架**；
    - 把复杂部分拆出来交给其它模块：极化壳、介电常数、短程 NN 残差势等。
  - 试图用“随意的 \(A_{ij}(r_{ij})\)”去拟合所有多体/屏蔽效应，在数学上做不到：真正的多体是依赖整体环境的，根本压不进一个简单的 pairwise 核里，最终只会得到“在训练集上勉强 work 的 pseudo-kernel”，**泛化与可解释性都很差**。

- **物理形状良好的 \(A\) 带来更好的数值稳定性和可迁移性**  
  - Qeq 解的是二次型的线性系统，其条件数、正定性与 \(A\) 结构强相关：
    - 接近物理 \(1/r\)（加合理硬度/屏蔽）的 \(A\) 通常是“对称 + 大致正定”，数值上更好解；
    - 任意学出来的核很容易引入非物理负模态或病态条件数，导致 Qeq 解发散或极不稳定。
  - 真实电磁相互作用在不同材料中仍然“看起来像库仑 \(1/r\) + 局部修正”，因此只要 \(A\) 的大体形状保持这个物理结构，**跨体系迁移性会好很多**。

- **模型分工清晰：\(A\) 负责库仑骨架，NN 负责“剩余复杂度”**  
  - “库仑 \(1/r\) + 少数物理参数”是一种强先验，相当于在极小的参数空间里编码了大部分长程行为；
  - 短程 NN / 残差势只需要在这个骨架上学习“真实体系里剩下的那一点复杂度”（多体、化学环境、极化 tail 等），参数利用效率更高、样本需求更合理。

从这个角度看：

- **用 SOG 逼近 \(1/r\)**：是为了在**保持库仑物理形状**的前提下，获得更好的数值性质（FFT/NUFFT 可加速、核平滑等）；  
- **不让 \(A\) 随意学**：是为了让 Qeq 电荷仍然是“有物理解读的库仑电荷”，而不是任意二次型下的虚构变量；  
- **真正复杂的屏蔽/多体长程 tail**：应该交给 PQEq 的极化壳、介电参数，或 CACE/DP-QEq/ReaxNet 里的短程/残差网络去学习，而不是污染 Qeq 自身的核。

---

### 7. 这种“把 A 也做成 SOG”与“直接用 SOGPotential 做长程能量”的区别

很关键的一点是：

- **SOGPotential**：是在能量层面直接学习一个长程卷积核，并用潜在变量（`q` 或 `q_eq`）输出长程能量；
- **Qeq 的 A**：是用于“自洽求电荷”的核矩阵，决定 $q_{\text{eq}}$ 的分配方式。

如果你把 Qeq 的 $A$ 也变成可训练的 SOG 核，并且又用 SOGPotential 学长程能量，可能会出现“长程被两套自由度同时解释”的不可辨识问题。更稳妥的策略通常是：

- $A$ 用 SOG 只是**数值近似/加速**，参数固定为逼近 \(1/r\)；
- 让 SOGPotential 去学剩余的真实长程 tail（或屏蔽、极化、多体等）。

---

### 8. 你可以怎么落地到代码（建议）

如果你要在当前 `ChargeEq` 体系里做这件事，推荐的最小可行实现是：

1. **保留 Qeq 的数学形式不变**（$M=A+J$，约束不变）；
2. **替换 “显式构造 A + 直接 solve” 为 “算子形式 matvec + PCG”**：
   - 写一个 `apply_A(x, positions, cell)`，内部用 SOG-FFT/NUFFT 算 $Ax$；
   - 写一个 `apply_M(x)=apply_A(x)+J*x`；
   - 用 PCG 解 `Mx=b` 和 `My=1`，再用 Schur complement 得到 $\lambda,q$；
3. **SOG 参数选择**：
   - 先用固定的 $\{w_\ell,\alpha_\ell\}$ 做 \(1/r\) 逼近（数值分析角度的 SOG），不要一开始就绑定到 `SOGPotential` 的可训练参数；
   - 等你确认数值稳定与加速效果后，再考虑“共享宽度/共享初值”等更激进的耦合方式。

---

### 9. 小结

- **能否用 SOG 逼近 \(1/r\)**：可以，依据是
  $$
  \frac{1}{r}=\frac{2}{\sqrt{\pi}}\int_0^\infty e^{-t^2 r^2}\,dt \approx \sum_\ell w_\ell e^{-\alpha_\ell r^2}.
  $$
- **能否与 `SOGPotential` 参数相同**：可以尝试，但更建议“共享初值/共享宽度而不共享幅度”，并保持 $A$ 的物理一致性（尽量逼近库仑核）。
- **对 Qeq 求解的加速手段**：核心是把稠密求逆变为“快速 matvec + 迭代解”，即
  - SOG 让 $Ax$ 可以用 FFT/NUFFT 快速算；
  - 用 PCG/MINRES 解 $Mx=rhs$；
  - 用 Schur complement 处理电中性约束。

如果你希望我把这份文档进一步“对齐你当前仓库的具体代码文件”，我也可以在文末补上一个更工程化的章节：列出需要改哪些类/函数（例如 `ChargeEq._compute_q_eq`、新增 PCG solver、以及复用 `SOGPotential` 的 NUFFT 路径来实现 `apply_A`）。

