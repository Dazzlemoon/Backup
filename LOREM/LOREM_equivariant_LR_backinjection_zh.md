# LOREM：低 $l$ 等变电荷、长程势与回灌张量积（代码数据流）

本文档说明 **`lorem/lorem.py`** 中 **`if self.lr:`** 分支如何实现：**从 `nodes_spherical` 读出低角动量截断的等变电荷 → 多通道 Ewald / $1/r$ 求势 → 用 `Tensor` 与短程球谐特征混合 → 球谐范数与标量势一并注入 `nodes_scalar`**。数学公式约定：**行内** `$...$`，**行间** `$$...$$`。

更宏观的架构见 **`LOREM_angular_and_longrange_zh.md`**；多通道库仑与严格多极静电的差别见 **`Multipole_vs_channel_SOG_zh.md`**。

---

## 1. 输入：`nodes_scalar` 与 `nodes_spherical`

长程分支执行前，每个原子已有：

- **`nodes_scalar`**：形状约 **`[N, d]`**，标量节点特征（`d = num_features`）。
- **`nodes_spherical`**：e3x 风格 **等变特征**，由截断边上球谐与 `TensorDense` 等得到，形状约为 **`[N, 1, num_lm, s]`**：
  - $\mathrm{num\_lm} = (\texttt{max\_degree}+1)^2$；
  - $s =$ **`num_spherical_features`**；
  - 中间维对应 **球谐分量 $l,m$**，最后一维为 **等变通道**。

---

## 2. 电荷头：标量 MLP + `TensorDense`（低 $l$ 等变电荷）

```187:194:lorem/lorem.py
            scalar_charges = masked(MLP(features=[2 * d, 1]), nodes_scalar, node_mask)
            spherical_charges = e3x.nn.TensorDense(
                features=1,
                use_bias=False,
                max_degree=max_degree_lr,
                include_pseudotensors=False,
            )(nodes_spherical).reshape(num_nodes, -1)
            charges = jnp.concatenate([scalar_charges, spherical_charges], axis=-1)
```

- **`scalar_charges`**：形状 **`[N, 1]`**。由 **`nodes_scalar`** 经 MLP 得到，视作 **一个标量电荷通道**。
- **`TensorDense(..., max_degree=max_degree_lr, features=1)`**：对 **`nodes_spherical`** 做 **SO(3) 等变线性映射**，输出角动量截断为 **`max_degree_lr`**（默认常为 2）、每不可约块 **1 个特征通道**，再 **`reshape(num_nodes, -1)`** 拉平为 **`[N, C_sph]`**。
  - 当 **`max_degree_lr = 2`** 时，$l=0,1,2$ 分量数为 $1+3+5=9$，通常 **$C_{\mathrm{sph}}=9$**（具体 flatten 顺序以 e3x 为准）。
- **`charges`**：在最后一维拼接：**`[N, 1 + C_sph]`**。

语义：**第 0 维对应标量电荷；后续 $C_{\mathrm{sph}}$ 维为「低 $l$ 等变电荷系数」**，后续长程核对每一维 **独立** 做同一种库仑型求势（见下节）。

---

## 3. 长程势：按通道的 Ewald（周期）或 $1/r$（非周期）

周期时 **`jax.vmap`** 对 **`charges`** 最后一维逐通道调用 **`jaxpme.Ewald(...).potentials`**；非周期时对 **`full_R_ij`** 用屏蔽的 **`1/r`** 做 **`segment_sum`**。输出 **`potentials`** 与 **`charges`** 通道数对齐：**`[N, 1 + C_sph]`**。

要点：**实现上**每个通道都被当作 **标量源** 参与 **库仑型** 长程；**不**在长程核里单独展开偶极–偶极张量公式。物理含义与局限见 **`Multipole_vs_channel_SOG_zh.md`**。

---

## 4. 回灌：`Dense` → `Tensor` → 球谐范数 → `Update`

```224:236:lorem/lorem.py
            scalar_potential = potentials[..., 0][..., None]
            spherical_potential = potentials[..., 1:].reshape(num_nodes, 1, -1, 1)

            spherical_potential = e3x.nn.Dense(s, use_bias=False)(spherical_potential)
            spherical_updates = e3x.nn.Tensor(include_pseudotensors=False)(
                spherical_potential, nodes_spherical
            )

            norms = spherical_norm_last_axis(spherical_updates, max_degree)
            norms = (norms * l_factors[None, None, :, None]).reshape(num_nodes, -1)
            updates = jnp.concatenate([scalar_potential, norms], axis=-1)
            nodes_scalar = Update(d)(nodes_scalar, updates, node_mask)
```

逐步含义：

1. **`scalar_potential`**：取 **`potentials` 第 0 通道**，即与 **标量电荷** 对应的长程势，形状 **`[N, 1]`**。
2. **`spherical_potential`**：取 **第 $1$ 至末通道**，reshape 为 **`[N, 1, C_sph, 1]`**，便于 e3x 中与张量积对齐。
3. **`Dense(s)`**：将势的通道维线性映射到 **`s = num_spherical_features`**，与 **`nodes_spherical`** 最后一维对齐，便于 **`Tensor`** 缩并。
4. **`Tensor(spherical_potential, nodes_spherical)`**：**Clebsch–Gordan 型等变张量积**：一边是 **长程得到的「球谐势侧」**，一边是 **短程 `nodes_spherical`**；输出 **`spherical_updates`** 与局部等变特征同型，表示 **长程势对局域等变特征的调制**。
5. **`spherical_norm_last_axis`**：对 **`spherical_updates`** 按 **$l$ 块** 求范数（power spectrum 风格），再乘 **`l_factors`**，拉平为 **`norms`**，得到 **可写入标量分支的旋转不变量**。
6. **`updates = concat(scalar_potential, norms)`**：**标量长程势** 与 **等变长程诱导的标量统计量** 一并送入 **`Update(d)`**，更新 **`nodes_scalar`**（形式与短程里「范数回灌」一致）。
7. 随后 **`energy += MLP(nodes_scalar)`** 增加一项原子能量。

---

## 5. 数据流小结（单向箭头）

**`nodes_spherical` → `TensorDense(max_degree_lr)` → `spherical_charges`（拉平）**

**`scalar_charges` ∥ `spherical_charges` → `charges`**

**`charges` →（逐通道）Ewald 或 `segment_sum` $1/r$ → `potentials`**

**`potentials` → 拆成 `scalar_potential` 与 `spherical_potential`**

**`spherical_potential` → `Dense(s)` → `Tensor(·, nodes_spherical)` → `spherical_updates`**

**`spherical_norm_last_axis` → `norms`；与 `scalar_potential` 拼接 → `Update(nodes_scalar)` → `MLP` → 能量**

---

## 6. 与物理表述的对应

- **等变电荷**：由 **`TensorDense`** 从完整 **`nodes_spherical`** 线性读出、截断到 **`max_degree_lr`**，再拉平成多路 **「假标量源」** 参与 $1/r$ 型长程。
- **回灌**：长程不仅贡献标量势，还通过 **`Tensor`** 把 **各通道势** 与 **当前局域球谐特征** 耦合，经 **范数** 进入 **`nodes_scalar`**，使能量头能利用 **长程 × 局域几何** 的组合信息。

---

*实现以仓库内 `lorem/lorem.py` 及所用 **e3x** 版本为准；若需核对精确形状，可在前向中对张量打印 `shape`。*
