# Qeq 的 A 与 SOGPotential 共用同一套 SOG 参数，且训练中不调用 SOGPotential

本文说明如何让 **ChargeEq** 在构造核矩阵 $A$ 时使用与 **SOGPotential** 完全相同的 $(w_\ell, s_\ell)$ 参数，同时**在训练里只使用 ChargeEq 的长程能量**（即不把 SOGPotential 接进前向与损失）。公式使用 `$...$` 与 `$$...$$`。

---

## 1. 目标

- **共用参数**：Qeq 的 $A_{ij} = K(r_{ij})$ 与 SOGPotential 的实空间核一致，即
  $$
  K(r) = \sum_\ell w_\ell\, e^{-r^2/s_\ell^2},
  $$
  且 $(w_\ell, s_\ell)$ 来自**同一组** `nn.Parameter`（即 SOGPotential 的 `wl`, `sl`）。
- **训练中不用 SOGPotential**：损失里长程项只取 ChargeEq 的 `ewald_potential`（$E_{\text{long}} = \frac12 q^\top A q$），不调用 `SOGPotential.forward()`；梯度只经 ChargeEq 反传到这套共享的 SOG 参数。

这样既保证“Qeq 的 A 与 SOG 核同一套参数”，又满足“训练时不使用 SOGPotential 这个函数”。

---

## 2. 代码改动概要

- **文件**：`cace/modules/charge_eq.py`
- **新增构造参数**：`shared_sog_potential: Optional[nn.Module] = None`。
- **逻辑**：
  - 若 `use_sog_kernel=True` 且 `shared_sog_potential` 不为 `None`，则 ChargeEq 不再创建自己的 `sog_log_alpha` / `sog_weights`，而是在 `_build_A_sog` 中直接使用 `shared_sog_potential.wl` 与 `shared_sog_potential.sl` 构造 $A$，核形式与 `sog.compute_potential_SOG_realspace` 一致：$K(r)=\sum_\ell w_\ell \exp(-r^2/s_\ell^2)$。
  - 将传入的 `shared_sog_potential` 赋给 `self.shared_sog_potential`，作为子模块挂到 ChargeEq 下，这样其 `wl`/`sl` 会出现在 `model.parameters()` 中，并通过 ChargeEq 的前向得到梯度。

---

## 3. 使用方式（训练脚本）

```python
from cace.modules import ChargeEq, SOGPotential

# 1) 先创建 SOGPotential，仅用于提供 (wl, sl)，不加入后续 NNP 的 forward 链
sog_pot = SOGPotential(
    bandwidth_num=12,
    feature_key="q",  # 无关，因不调用 forward
    output_key="SOG_potential",
    Periodic=False,
)

# 2) ChargeEq 使用同一套 SOG 参数，且 use_sog_kernel=True
charge_eq = ChargeEq(
    elements=[1, 8],
    feature_key="chi",
    output_key="q_eq",
    ewald_key="ewald_potential",
    use_sog_kernel=True,
    shared_sog_potential=sog_pot,  # 共用 wl, sl
)

# 3) 构建 NNP 时只接 ChargeEq，不接 SOGPotential
# 例如：data -> cace_rep -> chi -> charge_eq -> q_eq, ewald_potential
#      总能量 = SR_energy + ewald_potential（不接 SOG_potential）
modules = [..., chi, charge_eq, e_add]  # e_add = FeatureAdd(['SR_energy','ewald_potential'])
model = NeuralNetworkPotential(modules=modules, ...)

# 4) 训练：loss 只依赖 charge_eq 的输出，SOGPotential.forward 从未被调用；
#    梯度经 ChargeEq 反传到 sog_pot.wl / sog_pot.sl，实现“共用一套参数、仅 Qeq 参与训练”。
```

要点：

- `sog_pot` 只作为 `ChargeEq` 的**子模块**存在（通过 `shared_sog_potential` 传入），其参数会随 `model.parameters()` 一起被优化器更新。
- 前向与损失中**不要**再接入 `SOGPotential`（不要用 `SOG_potential` 或 `sog_pot(data)`），长程能量只使用 `ewald_potential`。

---

## 4. 核公式一致性

- **SOGPotential**（`compute_potential_SOG_realspace`）：
  - `min_term = -1 / sl**2`，`K_ij = sum_l wl * exp(r_ij^2 * min_term)`，即 $K(r)=\sum_\ell w_\ell e^{-r^2/s_\ell^2}$。
- **ChargeEq._build_A_sog**（当 `shared_sog_potential` 非空）：
  - 使用同一组 `wl`、`sl`，计算 `min_term = -1/(sl**2 + 1e-12)`，`A_sog[i,j] = sum_l wl * exp(r_ij^2 * min_term)`，与上面一致。

因此 Qeq 的 $A$ 与 SOGPotential 的实空间核在数学上共用同一套 $(w_\ell, s_\ell)$。

---

## 5. 小结

- 通过 `ChargeEq(..., use_sog_kernel=True, shared_sog_potential=sog_pot)`，Qeq 的 $A$ 与 SOGPotential 共用同一套 SOG 参数（wl, sl）。
- 训练时**不把 SOGPotential 加入前向**，长程只使用 ChargeEq 的 `ewald_potential`；梯度经 ChargeEq 更新共享的 SOG 参数即可。
- 若之后需要推理或后处理时用 SOGPotential 算长程，只需用同一 `sog_pot`（或从 checkpoint 里加载同一组 wl/sl），核与 Qeq 的 $A$ 保持一致。
