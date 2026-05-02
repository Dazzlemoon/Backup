# Assessment and Application of Universal MLIPs in SSE: 论文与代码概览

## 1. 论文基本信息

- 题目：Assessment and Application of Universal Machine Learning Interatomic Potentials in Solid-State Electrolyte Research
- 期刊：ACS Materials Letters (2025), 7, 3403-3412
- DOI：[10.1021/acsmaterialslett.5c00336](https://doi.org/10.1021/acsmaterialslett.5c00336)
- 仓库链接（论文中提及）：[https://github.com/dhw059/SSE](https://github.com/dhw059/SSE)

## 2. 这篇文章主要讲了什么

### 2.1 研究背景与核心问题

固态电解质（SSE）是下一代锂电池的关键材料，但传统 DFT 与经验势方法在以下方面存在明显瓶颈：

- 成本高，难以做大规模筛选；
- 很难同时兼顾多材料体系的可迁移性；
- 对扩散、动力学、温度效应等问题的建模效率不足。

论文聚焦的问题是：**通用预训练 MLIP 在 SSE 场景下到底好不好用，边界在哪里，哪些模型更可靠。**

### 2.2 方法设计

论文构建了一个针对 SSE 的系统化 benchmark，评估多个通用 MLIP（文中涉及 GRACE、DPA、MatterSim、MACE、SevenNet、CHGNet、TensorNet、M3GNet、ORB 等）在多类任务上的表现，包括：

- 能量/力/应力预测；
- 热力学与相稳定相关性质（如形成能、ehull）；
- 电化学稳定窗口；
- 力学性质（体模量、剪切模量）；
- 声子热力学量（热容、熵、自由能）；
- Li 离子扩散相关动力学性质。

### 2.3 主要结论（可直接用于读者速览）

- 多数性质上表现较强的模型包括：GRACE-2L-OAM、MACE-MPA、MatterSim、DPA-3.1-3M、SevenNet-MF-ompa（文中不同指标下也对 ORB 系列有讨论）。
- 论文在案例研究中选择 MatterSim 做进一步应用，强调其在精度-效率上的平衡。
- 在 Li6PS5Cl 中，约 40-50% 的 S/Cl 无序度可提升 Li+ 迁移通道连通性并增强离子输运。
- 在 Na_xLi_(3-x)YCl6 中，更高 Li 含量有助于拓宽迁移通道、降低扩散势垒。
- 结论整体强调：**通用 MLIP 可用于 SSE 机理研究与筛选，但仍需结合具体任务做针对性 benchmark，而不是“一个模型通吃”。**

## 3. 本目录（`SSE`）里都有什么

> 下面基于当前目录中的可见文件进行归纳，重点关注“与论文复现和性质评估相关”的内容。

### 3.1 论文与说明文件

- `README.md`：项目简述、图示、MatCalc 依赖说明与引用信息。
- `assessment-and-application-of-universal-machine-learning-interatomic-potentials-in-solid-state-electrolyte-research.pdf`：论文正文 PDF。
- `assessment-and-application-of-universal-machine-learning-interatomic-potentials-in-solid-state-electrolyte-research (1).pdf`：论文 PDF 的另一份拷贝。
- `SI-chapter-5-Universal Machine Learning Interatomic Potentials are Ready for Solid Ion Conductor.pdf`：相关补充材料/章节 PDF。

### 3.2 核心 Python 脚本（性质预测与对比）

- `predict_props_general_ehull_mlip_.py`：基于通用预训练 MLIP 计算 `energy above hull (ehull)`，并与参考值比较（MAE/RMSE）。
- `predict_props_ehull_from_pretrain_mlip.py`：类似地做 ehull 评估，脚本中可切换模型与数据源配置。
- `predict_props_general_windows_mlip.py`：评估还原电位/氧化电位/电化学稳定窗口等电化学性质。
- `predict_props_ele_chem_windowa_from_pretrain_mlip.py`：电化学稳定窗口评估脚本（预训练模型版本）。
- `predict_props_general_kgvrh_mlip.py`：评估体模量/剪切模量（KVRH/GVRH）等弹性性质。
- `predict_props_general_phonon_mlip.py`：基于声子计算热容、熵、Helmholtz 自由能等温度相关热力学量。
- `predict_props_from_pretrain.py`：综合型脚本，包含结构弛豫、相图条目构建、ehull 计算等流程。

### 3.3 过滤与候选集相关目录

- `alexandria-PBE_filter/`：包含筛选 notebook 与多个 JSON 结果文件（带有 band gap、氧化还原窗口、弹性等筛选条件）。
- `DenseGNN_filter/`：包含筛选 notebook 与多版筛选结果 JSON。
- `LiIonML-master/`：以 notebook 为主的数据处理与机器学习流程示例（如数据库处理、建模笔记本）。

## 4. 代码使用上的注意点（读代码前建议先看）

- 脚本里存在很多硬编码路径（如 `/home/datasets/...`），通常需要改成你本机的数据路径。
- 多个脚本依赖 `matcalc`、`pymatgen`、`mp_api`、`matplotlib`、`sklearn` 等库，并默认使用 GPU（可自动回退 CPU）。
- 脚本里有 Materials Project API key 的写法示例；正式使用时建议改为环境变量读取，避免明文写在代码里。
- 各脚本的“模型名称 / 数据集选择”通常通过注释切换，运行前建议先统一整理成配置参数。

## 5. 如果你想快速上手这份仓库

建议先按以下顺序阅读和执行：

1. 先读 `README.md` + 论文 PDF，明确评估指标和任务定义；
2. 从 `predict_props_general_ehull_mlip_.py` 或 `predict_props_general_windows_mlip.py` 入手，先跑通一个任务；
3. 再看 `predict_props_general_phonon_mlip.py` 和 `predict_props_general_kgvrh_mlip.py`，扩展到热力学/力学指标；
4. 最后使用各 `*_filter/` 目录的 JSON 与 notebook 做候选筛选与结果复核。

---

如果你希望，我可以下一步再帮你做一个“最小可运行版本”文档（包含环境安装、依赖版本、示例命令、输入输出示例），便于你在本机直接复现实验流程。
