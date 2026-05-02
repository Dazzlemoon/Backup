## Ji 等 2026：CACE-SOG 数值实验整理

本文档严格根据 Ji 等（2026，J. Chem. Phys. 164, 024109）“Accurate learning of long-range interatomic potentials by coupling Cartesian atomic cluster expansion and sum-of-Gaussians neural networks”的正文数值实验部分整理而成，按小节顺序概述每个数据集、对比方法和主要结论。

---

## 1. Molecular dimers（III.A）

- **体系与数据集**：
  - 采用文献 [Ref. 51] 的分子二聚体数据集，是 MLIPs 常用的 LR 基准。
  - 参考能量与力：HSE06 混合泛函 + many-body dispersion 校正。
  - 原始数据集中包含六类 dimers，每类具有不同的理想幂律衰减 \(1/r^p\)。
  - 文中重点选取三类代表：
    - **CC（charge–charge）**：\(p=1\)
    - **PP（polar–polar）**：\(p=3\)
    - **AA（apolar–apolar）**：\(p=6\)
  - 训练/测试划分：
    - 训练：10 个构型，分子间距约 5–12 Å；
    - 测试：3 个构型，分子间距约 12–15 Å。

- **模型与超参数（SR/LR）**：
  - **SR 表示（所有模型共享）**：CACE
    - cutoff \(r_\text{cut}=5\) Å；
    - Bessel 径向函数数目：6；
    - 角向与高阶参数：\(c=8,\ \mu_{\max}=2,\ \nu_{\max}=2\)；
    - 元素 embedding 维度 \(N_\text{embedding}=3\)；
    - 消息传递层：\(T=1\)（尽管在该任务中距离超过 cutoff 时 MP 帮助有限）。
  - **对比的 MLIPs**：
    - **CACE-only**：只有 SR CACE + 全连接 NN，没有 LR 层。
    - **CACE-LES**：CACE + latent Ewald summation（Ewald 基于 \(1/r\) 核）。
      - latent 变量维度 \(P=1\)，Ewald smeared σ=1 Å。
    - **CACE-SOG**：CACE + SOG-Net（本工作）
      - latent 变量维度 \(P=1\)；
      - SOG 高斯数 \(M=12\)，使用 BSA 初始化（近似 1/r Coulomb）。

- **对比结果**：
  - 指标：能量与力的 RMSE（表 I）。
  - 主要结论：
    - 所有带 LR 描述符的 MLIP（CACE-LES、CACE-SOG）在能量和力上都**远优于** CACE-only。
    - 对 **CC（1/r）**：
      - CACE-LES 在测试集能量上略优于 CACE-SOG；
      - 说明 Ewald + 1/r 核对纯 Coulomb 系统仍然非常合适。
    - 对 **PP（1/r³）与 AA（1/r⁶）**：
      - CACE-SOG 在能量与力 RMSE 上**显著优于** CACE-LES；
      - 体现 SOG-Net 的高斯核在捕捉非 1/r 幂律 LR 尾部时更具适应性。
    - 模型规模对比（表 II）：
      - CACE-based 模型（CACE-only、CACE-LES、CACE-SOG）的可训练参数数目约为 DP-SR、DP-SOG 的 1/30–1/20；
      - CACE-SOG 相比 CACE-LES 仅多出约 \(2M\) 个参数（每个 Gauss 的权重和带宽），但在精度上通常更优。

---

## 2. Aqueous KF solutions（水溶液 KF，III.B）

- **体系与数据集**：
  - 数据集：[Ref. 34] 中的水溶液 KF，覆盖从 0 到约 2 mol/L 的宽浓度范围。
  - 包含：
    - 体相电解质溶液；
    - 电解质–蒸汽界面构型。
  - 参考能量与力：
    - 水：柔性 SPC/Fw 力场，固定电荷：O −0.8476e，H +0.4238e；
    - 离子：K\(^+\)、F\(^-\) 单价。

- **模型与设置**：
  - SR：CACE 表示，对三种模型（CACE-only、CACE-LES、CACE-SOG）完全一致：
    - 6 Bessel 径向函数，\(c=12,\ \mu_{\max}=3,\ \nu_{\max}=3\)；
    - \(N_\text{embedding}=4\)；
    - cutoff \(r_\text{cut}=4.5\) Å；
    - 消息传递层 \(T=0\) 或 1（对比有/无消息传递）。
  - LR：
    - CACE-LES 与 CACE-SOG 均沿用 dimer 中的设置（Ewald vs SOG，高斯数 \(M=12\) 等）。

- **对比与结果**：
  - 模型：CACE-only、CACE-LES、CACE-SOG（含 MP0/MP1 变体）。
  - 指标：
    - Charge 预测 MAE；
    - Force MAE。
  - 主要观察：
    - 随训练样本数增加，CACE-SOG 的原子电荷预测与参考值高度吻合（图 3(b)）。
    - 力预测的 \(R^2\) 接近 1，说明力场刻画精确（图 3(c)）。
    - 与 CACE-LES 的对比（图 3(d),(e)）：
      - 在 charge 预测 MAE 上，CACE-SOG 在大多数样本数下**优于** CACE-LES 一个数量级以上；
      - 在 force MAE 上，CACE-SOG 也普遍优于 CACE-LES；消息传递的有无只带来轻微改进。
    - 物理原因：
      - 水溶液界面存在非平凡屏蔽和 LJ 尾部，真实 LR 衰减偏离 1/r；
      - CACE-LES 的 Ewald 1/r 核对这种非 Coulomb 尾部欠缺灵活性；
      - CACE-SOG 通过学习型高斯核自适应这些非 1/r 衰减，因此在 charge/force 上显著更准。

---

## 3. Global charge transfer & multiple charge states（III.C）

这一节主要测试 CACE-SOG 在**全局电荷转移**与**多电荷态**体系上的能力。

### 3.1 Na\(_9\)Cl\(_8^+\) / Na\(_8\)Cl\(_8^+\) 离子簇

- **体系**：
  - 数据集来自 Ko 等 [Ref. 6]：当从中性 Na\(_9\)Cl\(_8\) 簇上移除一个 Na 原子，产生 Na\(_9\)Cl\(_8^+\) 与 Na\(_8\)Cl\(_8^+\)。
  - 5000 个构型，90% 训练 / 10% 测试。
  - 以静电 1/r 相互作用为主。

- **设置**：
  - SR：CACE，参数与先前 CACE-LES 工作一致：
    - \(r_\text{cut}=5.29\) Å，6 Bessel 函数，\(c=8,\ \mu_{\max}=3,\ \nu_{\max}=3,\ N_\text{embedding}=2\)，\(T=0\)。
  - LR：
    - CACE-LES：latent 维度 1，σ=1.5 Å；
    - CACE-SOG：M=12 高斯，相同网格。

- **对比模型**：
  - CACE-only；
  - CACE-LES；
  - CACE-SOG（带/不带 BSA 初始化：w/o vs w）。

- **结果（表 III & 图 4）**：
  - 在能量与力 RMSE 上：
    - 物理上 1/r 核即可很好描述该体系，所以 CACE-LES 与 CACE-SOG 均能达到很高精度；
    - 但 CACE-SOG 在使用 BSA 初始化（“w”）时，**能量/力 RMSE 最低**，优于 CACE-only 和 CACE-LES。
  - 收敛速度：
    - CACE-SOG w/o（无物理初始化）收敛慢且精度略低；
    - BSA 初始化（尤其是同时初始化 bandwidth）显著加速收敛，并提升最终精度；
    - 图 4 显示，不同初始化策略（Type-1/2/3）中，带 BSA 的方案在训练时间与精度之间取得了最佳折中。

### 3.2 Au\(_2\)–MgO(001) 与 Cu-BTA（benzotriazole on Cu(111)）

- **体系**：
  - Au\(_2\)–MgO(001) 与 Cu-BTA 数据集均来自 Ko 等和 Maruf 等工作 [Refs. 6, 52]：
    - Au\(_2\)–MgO(001)：5000 构型；
    - Cu-BTA：400 构型。
  - 目的：测试 MLIP 是否能捕捉界面处的电荷转移（如向 Au 或 Cu）以及 cutoff 之外的相互作用。

- **设置**：
  - 数据集划分：90% 训练 / 10% 测试。
  - CACE 参数：6 Bessel 函数，\(c=12,\ \mu_{\max}=3,\ \nu_{\max}=3,\ N_\text{embedding}=4,\ T=0\)；
  - cutoff：Au\(_2\)–MgO 用 5.5 Å，Cu-BTA 用 5.0 Å。
  - LR：
    - CACE-LES：Ewald with \(k_c = \pi, \sigma = 1\) Å；
    - CACE-SOG：M=12 高斯，相同网格。

- **对比模型**（表 IV）：
  - CC-ACE（charge-constrained ACE）；
  - 4G-HDNNP；
  - NequIP-LR；
  - Polar-LR；
  - CACE-LES；
  - CACE-SOG（本文）。

- **结果（表 IV）**：
  - Au\(_2\)–MgO(001)：
    - 在能量 RMSE（meV/atom）和力 RMSE（meV/Å）上，CACE-SOG 与 CACE-LES 均显著优于其它基线；
    - CACE-SOG 在力 RMSE 上**最好**（约 5.73 meV/Å），CACE-LES 次之（约 7.91）。
  - Cu-BTA：
    - LR 主体仍然接近 Coulomb，因此 CACE-SOG 与 CACE-LES 表现接近；
    - 两者在能量/力 RMSE 上均优于其它基线。
  - Ag\(_3^{+/-}\) cluster：
    - 测试不同电荷态下的能量面学习；
    - CACE-SOG 在能量/力 RMSE 上略优于 CACE-LES，说明 SOG 层在几乎无 LR 效应时仍能改善近场截断误差。

---

## 4. Liquid–vapor interfacial water（III.D）

- **体系与数据集**：
  - 液–汽界面水数据集 [Ref. 10]：
    - 500 个界面构型，厚度约 65 Å；
    - 每个构型含 ~522 个水分子；
    - 参考：revPBE0-D3 DFT。

- **模型设置**：
  - CACE：\(r_\text{cut}=5.5\) Å，6 Bessel，\(c=12,\ \mu_{\max}=3,\ \nu_{\max}=3,\ N_\text{embedding}=3,\ T=0\)。
  - LR latent 变量：
    - CACE-LES 与 CACE-SOG 均采用 4 维 latent；
    - CACE-LES：Ewald（\(k_c=\pi,\ \sigma=1\) Å）；
    - CACE-SOG：M=12，高斯网格与 LES 相同。

- **动力学模拟与对比**：
  - 使用训练好的 CACE-LES 与 CACE-SOG，在 NVT 300 K 下各跑 500 ps MD（步长 1 fs，Nosé–Hoover）。
  - 比较以下物理量与 DFT：
    - 氧密度分布（thinner 65 Å slab 与 thicker 120 Å slab）；
    - 纵向 dipole density correlation function DDCF \(D_z(k_z)\)；
    - 水偶极与 z 轴夹角 \(\theta\) 的平均值 \(\langle \cos\theta \rangle\)。
  - 结论：
    - 对于密度分布，CACE-LES 与 CACE-SOG 都与 DFT 相符，且能泛化到更厚的水层（120 Å）；
    - 在 DDCF 与取向序参数上，CACE-SOG 在界面附近略优于 CACE-LES；
    - 说明 CACE-SOG 能更准确地恢复界面极化和 dipole 层的结构。

---

## 5. Water/Pt(111) interfacial systems（III.E）

- **体系与数据集**：
  - 水/Pt(111) 界面，数据来自 [Ref. 57]：
    - 48 041 个构型；
    - 3×4 正交 Pt(111) slab + 32 个水分子。
    - 参考：PBE+D3 DFT。

- **模型与模拟**：
  - CACE 参数与前述水体系一致；
  - LR：CACE-LES vs CACE-SOG（同 M=12 等设置）。
  - 训练后，以两种 MLIP 在 350 K 下做 1 ns NVT MD（步长 1 fs，Nosé–Hoover）。

- **对比结果**：
  - 结构性质：
    - O–O 和 O–H RDF：CACE-SOG 的峰位置和高度都与 DFT 更吻合；
    - CACE-LES 虽然合理，但在峰值等细节上有可见偏差。
  - 界面有序性：
    - 通过平面平均密度剖面分析 Pt–O、Pt–H 的分布；
    - 观察到典型的双峰 Pt–O 结构（2–4 Å），CACE-SOG 准确再现 DFT 的双峰和峰值高度；
    - CACE-LES 对该双峰结构的描述略逊一筹。

---

## 6. 小结：数值实验覆盖与对比对象

**Ji 2026 中 CACE-SOG 数值实验共涵盖：**

1. **分子二聚体（CC, PP, AA）**：考察不同幂律 LR 尾部；
2. **水溶液 KF（bulk + 界面）**：测试电解质与界面屏蔽效应；
3. **带电簇与电荷转移体系**：
   - Na\(_9\)Cl\(_8^+\)/Na\(_8\)Cl\(_8^+\) 簇；
   - Ag\(_3^{+/-}\) cluster。
4. **固–液/分子–固体界面体系**：
   - Au\(_2\)–MgO(001)；
   - Cu-BTA on Cu(111)；
   - 液–汽界面水；
   - 水/Pt(111)。

**主要对比方法包括：**

- CACE-only（无 LR 模块）；
- CACE-LES（CACE + latent Ewald）；
- DP-SR、DP-SOG（前作）；
- CC-ACE、4G-HDNNP、NequIP-LR、Polar-LR 等。

**总体结论：**

- 对纯 Coulomb 系统（如 CC dimer、部分带电簇），CACE-SOG 与 CACE-LES 精度相当，BSA 初始化可提升训练效率与精度；
- 对非 1/r 幂律 LR 尾部（PP/AA dimers、KF 溶液、界面体系等），CACE-SOG 在能量/力 /charge 等指标上显著**优于** CACE-LES 与纯 SR 模型；
- 在复杂界面体系（Au\(_2\)–MgO、Cu-BTA、water/Pt、水–汽界面）中，CACE-SOG 更好地恢复了 RDF、密度剖面、极化与有序度等高阶物理量，展示了其在广义 LR 相互作用上的灵活性与准确性。 

