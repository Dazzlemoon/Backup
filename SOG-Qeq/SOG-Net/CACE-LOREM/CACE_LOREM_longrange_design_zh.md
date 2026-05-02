# CACE 架构下实现「类 LOREM」长程：设计说明

本文档说明在 **保留 CACE 主干**（笛卡尔角向基、对称化不变特征、消息传递）的前提下，如何以 **多标量长程势** 对接 **Learning Long-Range Representations with Equivariant Messages**（Rumiantsev 等，2026，arXiv:2507.19382）中的 **长程思想**，并讨论该路线作为 **开发策略** 的利弊。数学公式：**行内** `$...$`，**行间** `$$...$$`。

---

## 1. 目标：要复现什么、不必复现什么

| 层次 | LOREM（论文 + `lorem/lorem.py`） | 本仓库推荐的「CACE + 类 LOREM 长程」 |
|------|----------------------------------|----------------------------------------|
| 短程几何 | e3x **球谐** $Y_{lm}$，**等变** `nodes_spherical` | CACE **不变** B 特征 + 边基中 **笛卡尔单项式**角向 $L_l(\hat{\mathbf{r}})$（与球谐线性等价，见 CACE 原文） |
| 长程核 | 多通道 **标量** 电荷 + **Ewald** / $1/r$ | 与现有 **SOGPotential / Ewald** 一致：**多通道标量** 逐通道长程 |
| 长程 → 短程回灌 | `Tensor` + 球谐范数 + `Update` | **不强制** e3x：可用 **`concat`（长程势, 不变特征）+ MLP / 残差块** |

**结论**：「类 LOREM」在这里指 **长程用可学习多源 + 显式库仑型求和**，而不是在 CACE 里 **逐行复刻** LOREM 的 **e3x 张量积回灌**。

---

## 2. 为何以 CACE 不变特征为主是合理开发路径

1. **CACE 读出天然是旋转不变量**（对称化 B 基），与 **标量长程势** $\Phi_{i,c}$ 同属 **标量融合**，**无需**再为「变成标量」而做 **CG + 张量积**（CACE 原文强调笛卡尔下构造不变量，避免显式稠密 CG 流水线）。
2. **角向信息**已由 **笛卡尔单项式基**（`AngularComponent`，$l_x+l_y+l_z=l$）进入边特征与消息传递，**不必**并行再维护一套球谐 `nodes_spherical` 才能描述局域方向模式。
3. **与现有 CACE-SOG 一致**：`Atomwise` 对 **`node_feats` 拉平** → 多通道 **`q`** → **`SOGPotential`** 按通道求长程势；仅将「融合方式」从**纯相加总能量**扩展为 **长程势再馈入读出**（若需要），改动面可控。

---

## 3. 推荐数据流（概念）

1. **表示**：`Cace` 前向 → `data["node_feats"]`（与现脚本一致）。
2. **电荷头（可多路）**：`Atomwise(feature_key="node_feats", n_out=C, per_atom_output_key="q", ...)`，得到每原子 **$q_{i,c}$**（标量通道）。
3. **长程**：`SOGPotential` / `Ewald`（`feature_key="q"`）→ 每原子每通道 **标量势** $\Phi_{i,c}$（或总势分解到原子，视模块实现）。
4. **融合（可选，「类 LOREM 回灌」的简化版）**：  
   $$
   h_i' = \mathrm{MLP}\big([\,\mathrm{flatten}(\mathrm{node\_feats}_i)\,;\,\Phi_{i,0},\ldots,\Phi_{i,C-1}\,]\big)
   $$
   再接入 **短程能量头**或**联合读出**；或仅把 **$\sum_c \Phi_{i,c}$** 与总能量相加（与当前 **CombinePotential** 类似）。
5. **力**：对坐标自动微分，与 LOREM/CACE-SOG 一致。

---

## 4. 与 LOREM 全文的差异（预期内）

- **无** `e3x.nn.Tensor(spherical\_potential, nodes\_spherical)`：**不显式**构造「长程势 × 等变球谐」的 CG 型耦合；若数据需要强角向–长程交叉项，可增大 MLP 容量或 **手工构造少量不变量**（例如已有 B 特征中的多体角项）。
- **物理**：多通道 $\Phi_{i,c}$ 仍是 **库仑型核上的标量源**，与 **`Multipole_vs_channel_SOG_zh.md`**（LOREM 目录）中讨论一致：**近似** 而非严格多极 Ewald。

---

## 5. 是否「好的开发方法」——适用场景

**适合**：

- 在 **CACE-SOG** 上快速迭代 **长程权重、通道数、融合 MLP**；
- 数据集主要考核 **能量/力 RMSE**，对 **严格介电 / LO–TO** 等需完整多极静电的场景 **无硬性要求**。

**需升级时**：

- 若实验表明必须 **显式等变电荷 + 张量积回灌**，再考虑 **混合架构**（例如 CACE 不变分支 + 并行 e3n 球谐分支），成本显著高于本方案。

---

## 6. 代码与文档索引

| 内容 | 位置 |
|------|------|
| CACE 表示与笛卡尔角向 | `cace/representations/cace_representation.py`，`cace/modules/angular.py` |
| 多通道 `q` + SOG | `cace/modules/atomwise.py`，`cace/modules/ewald*.py` |
| 联合势 | `cace/models/combined.py`（`CombinePotential`） |
| LOREM 长程与 e3x 回灌 | `/data/home/public/qiuqizhi/LOREM/lorem/lorem.py`，`LOREM_angular_and_longrange_zh.md`，`LOREM_equivariant_LR_backinjection_zh.md` |

---

## 7. 小结

在 **CACE 不变特征 + 笛卡尔单项式角向** 前提下，以 **多标量势** 实现 **类 LOREM 长程**、并用 **concat + MLP**（或仅能量相加）代替 **e3x Tensor + 球谐范数**，是 **工程上清晰、与 CACE 归纳偏置一致** 的开发路线；它与 LOREM **论文中的完整等变回灌** **不等价**，但在多数 **力场拟合** 任务中可作为 **第一阶段** 方案。

---

*文档随实现迭代可增补具体类名与 `forward` 签名；数值约定以本目录 `cace` 与训练脚本为准。*
