# ReaxNet 论文与代码概览（含 LLZO 检查）

## 1. 论文信息

- 题目：A foundation machine learning potential with polarizable long-range interactions for materials modelling
- 期刊：Nature Communications (2025)
- DOI：[10.1038/s41467-025-65496-3](https://doi.org/10.1038/s41467-025-65496-3)
- 代码仓库（论文给出）：[https://github.com/reaxnet/reaxnet](https://github.com/reaxnet/reaxnet)

## 2. 论文主要内容（快速版）

这篇文章的核心是：在等变图神经网络势（类似 NequIP 框架）中，显式加入**可极化长程相互作用**，构建一个可跨元素泛化的 foundation MLIP。

### 2.1 主要方法

- 总能量分解为三部分：`E0 + EPQEq + ED3`  
  - `E0`：短程/局域项（神经网络输出）  
  - `EPQEq`：基于 PQEq 的可极化静电项（核心贡献）  
  - `ED3`：DFT-D3 色散修正
- 与常见 QEq 不同，文中强调通过 core-shell 的可极化建模更好描述外电场响应和电荷重排。

### 2.2 主要结果

- 在不同电荷态数据集上，显式长程项能显著改善能量/力预测。
- 在基础模型层面（基于 MPtrj 训练），长程项对精度和可迁移性有明确增益。
- 展示了多类应用：  
  - 固态电解质中的 Li 离子扩散  
  - 铁电材料相变（BaTiO3）  
  - 固态电池界面反应分子动力学（如 Li/Li3PS4）

### 2.3 与 LLZO 相关的论文结论

论文中明确包含了 `c-LLZO (Li7La3Zr2O12)` 的扩散实验：  

- 使用 c-LLZO 进行 800-1800 K 的 MD 扩散分析；
- 给出 Arrhenius 与 MSD 结果；
- 对比了含/不含长程项模型与 AIMD 参考，说明可极化长程项对扩散性质建模有帮助。

### 2.4 LLZO 数值实验是怎么做的（论文 Methods 细化）

按论文 Methods（Molecular dynamics simulations）可概括为：

1. **初始结构来源**  
   - c-LLZO（空间群 `Ia-3d`）起始于文献中的实验结构（论文引用 ref.50）。
2. **构型构建**  
   - 扩展为 `2 × 2 × 2` 超胞；  
   - 在位点分布上，随机放置 `180` 个 Li 到 `24(d)` 位点，另放置 `268` 个 Li 到 `96(h)` 位点；  
   - 先做结构弛豫，得到 MD 起点。
3. **动力学流程**  
   - 先做 `NpT`（Nosé-Hoover thermostat + Parrinello-Rahman barostat，1 atm）用于温度下结构平衡；  
   - 再做 `NVT` 提取扩散性质；  
   - 时间步长 `2 fs`。
4. **扩散统计方式**  
   - 每个温度点先平衡 `100 ps`，再跑 `2 ns` 轨迹；  
   - 温度范围 `800-1800 K`（每 200 K 一个点）；  
   - 用 Li 的 MSD 计算自扩散系数，并做 Arrhenius 拟合。

### 2.5 LLZO 的主要发现与结果

- 该基础模型能够复现实验/文献 AIMD 参考下的扩散系数与活化能趋势（论文图 3）。
- 相比不含显式长程项的对照模型（w/o-lr），含可极化长程项模型在 LLZO 扩散预测上更优。
- 由于能跑更大体系与更长时间，扩散统计不确定性更低，结果稳定性更好（论文强调“reduced uncertainties”）。
- 作者据此认为该框架可为固态电解质中的扩散分析提供更可靠的数据基础。

### 2.6 论文有没有说明 LLZO 数据怎么得到

有说明“结构与轨迹来源”，但没有把 LLZO 写成一个独立命名的公开“数据集包”：

- **结构来源**：LLZO 初始晶体结构来自已发表实验结构（ref.50），不是论文新建并单独发布的 LLZO 数据集名称。
- **训练数据来源**：foundation 模型主训练集是 `MPtrj`（不是 LLZO 专属数据集）。
- **结果数据可获得性**：论文 Data availability 写到 MD 轨迹数据在 `reaxnet` 仓库可获取，但未在正文中单列“LLZO 数据集下载链接（独立条目）”。

## 3. 当前 `reaxnet` 文件夹里有什么

> 基于当前目录实际可见文件整理（不是远端仓库的完整快照）。

### 3.1 顶层

- `README.md`：安装方式、依赖、基础说明、示例入口（文档中提及 notebooks）。
- `ARTICLE_AND_CODE_OVERVIEW.md`：本说明文档。

### 3.2 `examples/`（示例 notebook 与结构样例）

- `examples/basic.ipynb`：基础能量/力预测示例。
- `examples/non_bond.ipynb`：长程非键相互作用示例。
- `examples/fine_tuning.ipynb`：预训练模型微调示例。
- `examples/Li2PO2N.cif`：示例结构文件。

### 3.3 `data/`（应用案例相关输入/输出结构）

- `data/LLZO/`：包含 `init.vasp` 与多个温度点的 `*_final.vasp`（`800/1000/1200/1400/1600/1800 K`）。
- `data/BaTiO3/`：包含 `init.vasp` 与 `final.vasp`。

说明：这些文件更像是结构快照或阶段结果，不等同于完整 MD 轨迹。

### 3.4 `egnn/`（神经网络与图计算主模块）

- `nequip.py`、`e3nn_layer.py`、`nn_util.py`：等变网络结构与层实现。
- `data.py`、`dataloader.py`：ASE/图数据构建与 batch 逻辑。
- `compute.py`：能量-力-应力计算流程（JAX 自动微分）。
- `loss.py`：训练损失。

### 3.5 `jax_nb/`（长程非键相互作用实现）

- `jax_nb.py`：PQEq、静电、非键项主流程。
- `jax_d3.py`：DFT-D3 相关实现。
- `parameters.py`：相关参数。

### 3.6 `pretrained/`

- `README.md`：给出预训练模型下载链接（Figshare）。
- `__init__.py`：模块初始化文件。

## 4. 关于“LLZO 实验代码是否在当前目录里”

结论分两层：

### 4.1 论文层面：有

- 论文正文明确有 `Li7La3Zr2O12 (c-LLZO)` 扩散实验与图表描述。

### 4.2 当前本地 `reaxnet` 目录层面：有 LLZO 相关结构文件，但未见完整 MD 轨迹

我在当前目录重新检索后发现：

- 存在 LLZO 目录：`data/LLZO/`，含 `init.vasp` 和不同温度下的 `*_final.vasp` 结构结果；
- 示例 notebook 已存在：`examples/basic.ipynb`、`examples/non_bond.ipynb`、`examples/fine_tuning.ipynb`；
- 但仍未看到常见完整轨迹文件（如 `*.traj`、`*.xyz`、`*.extxyz`、`*.lammpstrj` 等）。

因此更准确的说法是：  
**你当前本地 `reaxnet` 已有 LLZO 相关结构文件与部分结果快照，但尚未发现完整 MD 轨迹文件和 LLZO 专用一键复现实验脚本。**

## 5. 你如果要复现 LLZO，建议下一步

1. 基于 `data/LLZO/init.vasp` 与各温度 `*_final.vasp`，先确认结构是否可直接作为复现实验起点；  
2. 补齐轨迹输出与后处理流程（MSD、扩散系数、Arrhenius 拟合）；  
3. 按论文 Methods 校准 MD 参数（温度区间、步长、系综、时长）；  
4. 用当前代码中的 `ASE + JAX` 接口补一个 `LLZO` 专用 driver（NVT/NpT + MSD + Arrhenius）。

如果你需要，我可以下一步直接给你在 `reaxnet` 里补一个 `LLZO_REPRO_PLAN.md`，把“输入文件格式 + 运行脚本骨架 + 结果后处理”写成可执行清单。
