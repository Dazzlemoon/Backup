# 在 CACE 中仿照 Fixed Point 架构训练：公式与代码实现建议

本文是面向你当前仓库（`SOG-Qeq/SOG-Net/CACE-SOG-Qeq`）的实现笔记，目标是回答：

1. 能否仿照论文 Fixed Point 架构做 CACE 训练？  
2. 对应数学公式是什么？  
3. 具体代码该怎么落地（结合你现有 `ChargeEq` / `SOG` 代码）？

---

## 1. 结论先行

**可以。**  

你当前的 `cace/modules/charge_eq.py` 已经具备 Fixed Point 的关键组件：

- 电势驱动的电荷更新（`q_eq`）
- 总电荷约束（`system_charge` + `normalization_factor`）
- 长程核矩阵（Ewald 或 SOG）
- 输出长程能量（`ewald_key` / `SOG_potential`）

也就是说，框架已经“接近 Fixed Point 版本”。差别主要在于：

- 你当前 `ChargeEq._compute_q_eq()` 是**一次解线性方程**（类 QEq 闭式），  
- 论文 Fixed Point 是显式迭代
  $$
  p^{(t+1)}=F_{\mathrm{ML}}(v_{\mathrm{eff}}[p^{(t)}]).
  $$

两者可统一为：  
**把你现在的闭式 `q_eq` 看作一个“可微近似固定点求解器”特例**；若要严格复刻论文，可再加显式 SC 迭代更新器。

---

## 2. 数学公式（CACE 版本映射）

下面给出与当前代码更贴近的写法。

### 2.1 粗粒化电荷密度

$$
\rho(\mathbf r; \mathbf q)=\sum_i q_i\,\phi_i(\mathbf r),
$$

其中 $\phi_i$ 可是点电荷（或高斯/多极基）。

### 2.2 有效势

$$
v_{\mathrm{eff}}[\mathbf q](\mathbf r)=
\int \frac{\rho(\mathbf r')}{|\mathbf r-\mathbf r'|}d\mathbf r'
+ v_{\mathrm{app}}(\mathbf r) + \mu.
$$

在离散基上通常变成
$$
\mathbf v = A\mathbf q + \mathbf v_{\mathrm{app}} + \mu\mathbf 1.
$$

### 2.3 你当前 `ChargeEq` 的线性系统（QEq 形式）

代码里核心是

$$
\begin{bmatrix}
A+\mathrm{diag}(J) & \mathbf 1\\
\mathbf 1^\top & 0
\end{bmatrix}
\begin{bmatrix}
\mathbf q\\ \lambda
\end{bmatrix}
=
\begin{bmatrix}
-\chi\\ Q_{\mathrm{tot}}/f
\end{bmatrix},
$$

其中：
- $J$：元素硬度（`J_elem/J_i`）
- $\chi$：电负性特征（`feature_key`, 如 `chi_biased`）
- $f$：`normalization_factor`
- 约束 $\mathbf 1^\top\mathbf q = Q_{\mathrm{tot}}/f$

### 2.4 总能量（Fixed Point/自洽电荷后）

可写为：

$$
E = E_{\mathrm{SR}}(\theta_{\mathrm{sr}})
+ E_{\mathrm{NL}}(\theta_{\mathrm{nl}};\mathbf q^*)
+ \frac12\,\mathbf q^{*\top}A\mathbf q^*
+ \mathbf q^{*\top}\mathbf v_{\mathrm{app}}.
$$

在你当前工程里对应 `FeatureAdd(SR_energy + SOG_potential)` 这类组合。

---

## 3. 代码落地方案（建议三层）

## 3.1 层 A：最小改造（推荐，先跑通）

直接沿用现有 `ChargeEq`（一次线性求解）+ CACE 表示器，训练时做：

- 输入：结构 -> 表示器 -> `chi`（或 `chi_biased`）与 `J`
- `ChargeEq` 解出 `q_eq`
- 输出 `CACE_energy/CACE_forces`，按能量/力/偶极/电荷损失训练

这已经是“自洽电荷+长程耦合”架构。

---

## 3.2 层 B：显式 Fixed Point 迭代（更贴论文）

新增一个模块（示意）：

```python
class FixedPointChargeUpdate(nn.Module):
    def __init__(self, fml, max_steps=40, mixing=0.5, tol=1e-5):
        super().__init__()
        self.fml = fml                 # F_ML: (geom_feat, v_eff) -> q_new
        self.max_steps = max_steps
        self.mixing = mixing
        self.tol = tol

    def forward(self, geom_feat, q0, build_veff):
        q = q0
        for _ in range(self.max_steps):
            v = build_veff(q)          # A q + v_app + mu
            q_new = self.fml(geom_feat, v)
            q_next = (1 - self.mixing) * q + self.mixing * q_new
            if torch.max(torch.abs(q_next - q)) < self.tol:
                q = q_next
                break
            q = q_next
        return q
```

然后在主模型里：
1. 表示器提特征 `h_i`  
2. `q0 = q_local(h_i)` 作为初值  
3. `q* = FixedPointChargeUpdate(...)`  
4. 用 `q*` 计算长程项和总能量。

---

## 3.3 层 C：训练方式（对应论文三类）

### (1) Direct（简单但推理不一致风险）
- 监督 `q*` 对齐 `q_ref` 或多极矩
- 监督能量/力

典型损失：
$$
\mathcal L = w_E\|E-E^{\mathrm ref}\|^2 + w_F\|F-F^{\mathrm ref}\|^2 + w_q\|q-q^{\mathrm ref}\|^2 + \cdots
$$

### (2) Unroll SC（中等复杂）
- 训练时只跑固定 `K` 步 SC 并反传  
- 先用小步数快速训练，再增大步数

### (3) Implicit Differentiation（最稳健）
- 训练时收敛到固定点后，用隐式求导  
- 对大系统更省显存，但要处理雅可比线性求解稳定性

---

## 4. 与你当前仓库的具体对应点

- `cace/modules/charge_eq.py`  
  - 已有：`feature_key -> chi`, `J_raw -> J_elem/J_i`, `system_charge` 约束  
  - 可增：显式 `FixedPointChargeUpdate` 或保持闭式解作为特例

- `fit-.../fit-cace-Qeq-SOG.py`  
  - 已有 `chi` / `chi_biased`、`SystemChargeFromAtomicCharges`、`FeatureAdd`  
  - 可改：在 `charge_eq` 前后加入 `q0` 初始化与迭代器

- 训练脚本
  - 先保留当前 loss（E/F + 可选 q/dipole）
  - 若切换 fixed-point 显式迭代，建议先 `unroll K=5~8`，后期再 `K↑` 或 implicit

---

## 5. 最小可运行伪代码（整合）

```python
# 1) local representation
h = representation(data)
chi = chi_head(h)              # [N, 1]
J   = hardness_head(h, z)      # [N]

# 2) build operators
def build_veff(q, mu):
    # A can be Ewald or SOG kernel matrix
    return A @ q + v_app + mu

# 3) initial guess (local)
q0 = q_local_init(h).detach()  # or zeros

# 4) fixed-point solve
q_star = fp_solver(geom_feat=h, q0=q0, build_veff=build_veff)   # iterative
# or: q_star = charge_eq_closed_form(chi, J, A, Q_total)         # your current way

# 5) energy
E_lr = 0.5 * q_star.T @ A @ q_star + (q_star * v_app).sum()
E_sr = sr_energy_head(h)
E_nl = nonlocal_energy_head(h, q_star)   # optional
E = E_sr + E_nl + E_lr

# 6) loss
loss = wE * mse(E, E_ref) + wF * mse(F, F_ref) + wq * mse(q_star, q_ref)
```

---

## 6. 实操建议（按优先级）

1. **先用你当前闭式 `ChargeEq` 跑通并调稳**（最划算）。  
2. 再引入“显式 SC 迭代更新器”做 A/B 实验（闭式 vs fixed-point）。  
3. 训练策略优先：`shortcut unroll` -> `implicit` 精调。  
4. 保持 `normalization_factor` 在训练/评估口径一致（你此前已踩过这个坑）。

---

## 7. 一句话总结

你可以完全用 CACE 复现 Fixed Point 思路；  
当前 `CACE-SOG-Qeq` 已经具备核心机制，下一步主要是把“闭式 QEq 解”扩展成“显式迭代 fixed-point + 对应训练策略”。

