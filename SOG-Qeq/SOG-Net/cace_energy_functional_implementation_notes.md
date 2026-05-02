# 在 CACE 中仿照 Energy Functional 架构训练：公式与代码实现建议

本文档回答你提出的问题：  
**能否按 Energy Functional（能量泛函）架构做 CACE 训练？如果可以，公式和代码怎么实现？**

结论先说：**可以**。而且你当前 `CACE-SOG-Qeq` 已经有不少可复用组件（`chi/chi_biased`、`ChargeEq`、`SOG/ewald`、`FeatureAdd`）。

---

## 1. Energy Functional 架构在 CACE 中的目标

我们希望定义一个关于电荷系数（或多极矩）$\mathbf p$ 的可微能量泛函：

$$
E_{\text{tot}}(\mathbf R, \mathbf p; \theta)
=
E_{\text{local}}(\mathbf R; \theta_{\text{sr}})
+
G_{\text{ML}}(\mathbf R,\mathbf p; \theta_g)
+
E_{\text{Coul}}(\mathbf p; \mathbf R)
+
E_{\text{app}}(\mathbf p; \mathbf R),
$$

并在总电荷约束下求解

$$
\mathbf p^*(\mathbf R;\theta) = \arg\min_{\mathbf p,\ \mathbf 1^\top \mathbf p = Q_{\text{tot}}} E_{\text{tot}}(\mathbf R,\mathbf p;\theta).
$$

最终预测能量

$$
\hat E(\mathbf R;\theta)=E_{\text{tot}}(\mathbf R,\mathbf p^*;\theta).
$$

力通过

$$
\hat{\mathbf F}=-\frac{\partial \hat E}{\partial \mathbf R}
$$

得到（自动微分）。

---

## 2. 与你当前 CACE-SOG-Qeq 的变量映射

- `chi` / `chi_biased`：可作为 $G_{\text{ML}}$ 中与 $\mathbf p$ 相关项的输入。  
- `SOG_potential` 或 `ewald_key`：可作为 $E_{\text{Coul}}$ 的实现。  
- `SR_energy`：即 $E_{\text{local}}$。  
- `system_charge`：总电荷约束。  
- `q_eq`：现在是闭式解输出；在能量泛函架构里可替换为“最优化求得的 $\mathbf p^*$”。

---

## 3. 数学形式（一个可实现的具体版本）

给一个最实用版本（从你当前工程易迁移）：

### 3.1 泛函定义

$$
E_{\text{tot}} = E_{\text{SR}}(\mathbf R)
+ \underbrace{\sum_i \chi_i p_i + \frac12\sum_i J_i p_i^2}_{\text{可看作 }G_{\text{ML}}\text{ 的一部分}}
+ \frac12 \mathbf p^\top A(\mathbf R)\mathbf p
+ \mathbf p^\top \mathbf v_{\text{app}}.
$$

这其实是 QEq 二次泛函；若要更“Energy Functional”，可把二次项扩展成更高阶/神经网络：

$$
G_{\text{ML}}(\mathbf R,\mathbf p)=\sum_i U_i(h_i,\ p_i,\ \text{neighbors}),
$$

其中 $U_i$ 可以是 one-body nonlinear 或 many-body nonlinear（参考 Baldwin 文中分类）。

### 3.2 约束最优化

$$
\min_{\mathbf p} E_{\text{tot}}(\mathbf p)\quad
\text{s.t.}\quad \mathbf 1^\top \mathbf p = Q_{\text{tot}}.
$$

可以用拉格朗日乘子 $\lambda$：

$$
\mathcal L(\mathbf p,\lambda)=E_{\text{tot}}(\mathbf p)+\lambda(\mathbf 1^\top\mathbf p-Q_{\text{tot}}).
$$

若 $E_{\text{tot}}$ 是二次型，得到线性方程组（你当前 `ChargeEq` 就是这类）；  
若是非线性 $G_{\text{ML}}$，可用 LBFGS/Newton/Trust-Region 求 $\mathbf p^*$。

---

## 4. 代码实现方式（推荐两阶段）

## 4.1 第一阶段：二次泛函版本（最小改造，先稳定）

这一步几乎可直接复用 `ChargeEq` 思路，把它明确包装成“能量泛函最小化器”：

```python
class EnergyFunctionalQEq(nn.Module):
    def __init__(self, feature_key="chi_biased", output_key="p_star", energy_key="EF_energy"):
        super().__init__()
        self.feature_key = feature_key
        self.output_key = output_key
        self.energy_key = energy_key
        # J 按元素参数化（可训练）
        self.J_raw = nn.Parameter(torch.ones(num_elements))

    def forward(self, data):
        # 1) 取 chi、元素类型、A 矩阵、总电荷
        chi = data[self.feature_key].view(-1)
        Z   = data["atomic_numbers"]
        A   = build_A_from_ewald_or_sog(data)      # [N, N]
        Q   = data["system_charge"]                # scalar
        J_i = map_element_hardness(self.J_raw, Z)  # [N]

        # 2) 解约束二次问题（线性系统）
        # [A+diag(J)  1] [p]   = [-chi]
        # [1^T        0] [λ]     [ Q  ]
        p_star, lam = solve_kkt(A, J_i, chi, Q)

        # 3) 计算长程能量项
        E_lr = 0.5 * p_star @ A @ p_star
        E_chiJ = (chi * p_star).sum() + 0.5 * (J_i * p_star**2).sum()
        data[self.output_key] = p_star.unsqueeze(1)
        data[self.energy_key] = E_lr + E_chiJ
        return data
```

然后主模型总能量：

```python
E_total = SR_energy + EF_energy   # 对应 CACE_energy
```

---

## 4.2 第二阶段：非线性泛函版本（更接近论文 Energy Functional）

将 `E_chiJ` 替换成神经泛函：

```python
class GMLNonlinear(nn.Module):
    def __init__(self, in_dim, hidden=128):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, 1)
        )

    def forward(self, h_i, p_i):
        # 最简单 one-body: U_i(h_i, p_i)
        x = torch.cat([h_i, p_i.unsqueeze(-1)], dim=-1)
        return self.mlp(x).sum()
```

总泛函变为：

```python
E_tot(p) = E_SR + GML(h, p) + 0.5 * p^T A p + p^T v_app
```

并对 `p` 做约束优化（每个 structure 单独求解）：

```python
def minimize_p(E_fn, p0, Q, max_iter=50):
    # 方式A：投影梯度 + 约束修正
    # 方式B：LBFGS + penalty( sum(p)-Q )
    # 方式C：带 λ 的 KKT Newton
    ...
    return p_star
```

---

## 5. 训练策略（对应论文三种）

### (1) Direct

在参考电荷（如果有 `p_ref`）处监督能量/力/电荷：

$$
\mathcal L = w_E\|E-E^{ref}\|^2 + w_F\|F-F^{ref}\|^2 + w_p\|p^*-p^{ref}\|^2 + w_D\|D-D^{ref}\|^2.
$$

### (2) Implicit Differentiation（推荐最终版）

当 $p^*$ 由“最优化器隐式定义”时，用隐式求导反传，避免对整个优化轨迹反传。

### (3) Unroll（工程上简单）

训练时只跑 K 步优化（或 SC），把优化过程当计算图展开。

---

## 6. 在你仓库中的最小改造建议

1. 新增模块：`cace/modules/energy_functional.py`  
   - 实现 `EnergyFunctionalQEq`（先二次版）。
2. 在训练脚本（如 `fit-cace-Qeq-SOG.py`）中替换/并行 `ChargeEq`：  
   - 输出 `p_star` 与 `EF_energy`。  
3. 用 `FeatureAdd(["SR_energy", "EF_energy"], output_key="CACE_energy")` 合并。
4. 保持 `system_charge` 管线不变（你已实现 `SystemChargeFromAtomicCharges`）。
5. 先跑二次版稳定训练，再切到非线性 `GML` + 约束优化器。

---

## 7. 与当前闭式 ChargeEq 的关系

- 你现在的 `ChargeEq` 可以看作“二次能量泛函最小化”的一个高效特例。  
- 因此“仿照 Energy Functional”并不是推翻现有，而是：
  1) 从“求电荷”提升为“显式写能量泛函并最小化”；  
  2) 逐步把二次项扩展为可学习非线性泛函 $G_{\text{ML}}$。

---

## 8. 一句落地建议

先在你现有框架里做 **EnergyFunctionalQEq（二次版）**，跑通后再加 **GML 非线性项 + implicit differentiation**，这是风险最小且最贴近论文路线的实现路径。

---

## 9. 与传统 Qeq 训练逐条对比表

下表把“传统 Qeq（闭式二次）”与“Energy Functional（可扩展泛函）”在工程训练上逐条对比：

| 维度 | 传统 Qeq 训练 | Energy Functional 训练（CACE版） |
|---|---|---|
| 电荷表示 | 常为原子点电荷 $q_i$ | 可为点电荷/高斯/多极系数 $\mathbf p$ |
| 核心方程 | 闭式/线性系统（KKT）一次求解 | 约束最小化 $\mathbf p^*=\arg\min E(\mathbf p)$（二次可退化为Qeq） |
| 训练对象 | 主要是 $\chi,\eta$（或其网络头）+ SR 模块 | $E_{\text{local}}$、$G_{\text{ML}}$、电荷头、可能还有迭代器参数 |
| 数据需求（最小） | 仅能量/力也可训；有时不需要显式 $q_{\text{ref}}$ | 仅能量/力也可训；若要稳定电荷物理性，建议加偶极/电荷监督 |
| 数据需求（增强） | 若监督电荷：需原子电荷分解 | 若 direct 监督 $\mathbf p$：需多极/电荷分解；implicit 方式可不强依赖 |
| 损失函数 | 常见 $L_E+L_F(+L_q)$ | 常见 $L_E+L_F+L_D(+L_p)$，并可加入电荷约束/平滑正则 |
| 训练-推理一致性 | 往往较好（同一闭式求解器） | 取决于训练法：direct 可能不一致；implicit/unroll 更一致 |
| 数值稳定性 | 高（线性系统可控） | 中等到高（受最优化器、步长、收敛阈值影响） |
| 可表达性 | 受二次型限制（易欠拟合复杂极化/转移） | 更强（可加 one-body/many-body 非线性泛函） |
| 计算开销 | 低到中 | 中到高（每步要做最小化或更多迭代） |
| 可解释性 | 强（$\chi,\eta$ 清晰） | 中（非线性 $G_{\text{ML}}$ 可解释性下降） |

补充：
- 你当前 `ChargeEq` 路线对应“传统Qeq训练”的高质量神经增强版本。  
- 若把 $G_{\text{ML}}$ 做成非线性并显式最小化，就进入“Energy Functional训练”范式。

---

## 10. 两种方法优劣比较（给实验决策用）

## 10.1 传统 Qeq（闭式二次）优点
- 训练/推理流程简单，调参成本低；
- KKT 线性系统可控，稳定性好；
- 对总电荷约束处理自然；
- 对中小数据集、工程落地友好。

## 10.2 传统 Qeq（闭式二次）局限
- 二次泛函表达能力有限，容易在强非线性极化、远程电荷转移场景欠拟合；
- 对“导体/绝缘体同时覆盖”“复杂界面外场响应”等现象可能不够灵活；
- 可能出现“能量/力够准，但电荷分布物理性欠佳”。

## 10.3 Energy Functional（非线性泛函）优点
- 可表达更复杂的电荷-环境耦合（one-body nonlinear / many-body）；
- 更有机会同时提升力、偶极、外场响应与跨场景泛化；
- 能自然承接 implicit differentiation / unroll 等“推理一致”训练。

## 10.4 Energy Functional（非线性泛函）代价
- 优化器与收敛策略复杂，容易出现训练不稳；
- 计算开销更高（尤其每样本都做约束最小化）；
- 若缺少电荷/偶极监督，可能学到“可用但非物理”的 $\mathbf p^*$。

---

## 11. 从哪些数据可以观察到“新现象”

如果你要比较“传统Qeq vs Energy Functional”，建议重点用以下数据切片：

1. **外加电场 slab 数据（导体 + 绝缘体）**  
   - 观察：总偶极-电场曲线斜率、屏蔽行为、界面电势变化。  
   - 指标：Dipole RMSE、field-response slope error、charge profile shift。

2. **带电缺陷 / 分离缺陷对（长程电荷转移）**  
   - 观察：电荷是否在远距缺陷间不合理“泄漏/平均化”。  
   - 指标：缺陷局域电荷、总能差、随间距的电荷转移曲线。

3. **碎裂/解离路径（fragmentation）**  
   - 观察：碎裂后是否逼近整数电荷分配，是否平滑。  
   - 指标：fragment charge vs distance 曲线、能量曲线连续性。

4. **多相界面（如金属-水、离子液-固体）**  
   - 观察：极化诱导、界面层电荷重排、跨相转移。  
   - 指标：界面偶极、层分辨电荷密度、力误差分解。

5. **跨电荷态泛化集（训练未覆盖的总电荷）**  
   - 观察：总电荷变化时能量/力/偶极是否稳定外推。  
   - 指标：charge-state extrapolation error、SC 收敛率、失败样本比例。

### 建议同时记录的训练与推理诊断
- SC/最小化收敛步数分布（均值、P95、失败率）；  
- $\|\nabla_{\mathbf p}E\|$ 的统计（是否接近驻点）；  
- 训练态与推理态误差差距（判断一致性问题）；  
- `q` 的物理尺度一致性（是否需要 normalization_factor 还原比较）。

