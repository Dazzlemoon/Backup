# JAX 自动微分与 PyTorch 自动微分对比

本文从机制、用法和适用场景上对比 JAX 与 PyTorch 的自动微分（autodiff），便于在 DP-QEq（JAX）与 CACE-SOG-Qeq（PyTorch）之间做技术选型或迁移时参考。公式使用 `$...$` 与 `$$...$$`。

---

## 1. 核心设计哲学

| 维度 | PyTorch | JAX |
|------|---------|-----|
| **执行模式** | **Define-by-run**（边运行边建图）：前向执行时动态记录依赖，反向时按记录求导。 | **Define-then-run**（先定义再变换）：把“纯函数”做**追踪（trace）**得到计算图，再对图做 **transform**（如 `grad`、`jit`、`vmap`）。 |
| **计算图** | 每次前向都会建一张新图，用完后可释放；图与具体输入绑定。 | 对“抽象输入”做一次 trace 得到固定图，可被多次复用、优化、求导。 |
| **状态与副作用** | 允许 in-place、全局随机状态、可变对象；更接近“普通 Python + 张量”。 | 强调**纯函数 + 不可变数组**；随机用显式 `key`，状态通过函数参数传递。 |

因此：**PyTorch 的 autodiff 是“执行时记录”的自动微分；JAX 的 autodiff 是“对追踪后的函数做数学变换”的自动微分。**

---

## 2. 自动微分如何发生

### 2.1 PyTorch：动态图 + 反向磁带

- **前向**：对 `requires_grad=True` 的张量做运算时，每个算子会往当前线程的 **autograd 图** 里插入节点，并记录“谁依赖谁”。
- **反向**：调用 `loss.backward()` 时，从 `loss` 出发按图**反向遍历**，对每个节点用链式法则算梯度，并写入对应张量的 `.grad`。
- **特点**：
  - 图是**动态**的：不同输入可以走不同分支（if/for 随数据变化），每次前向都是一次新记录。
  - 梯度与“当前这次前向”严格对应；调试时容易设断点、打印中间量。
  - 默认**不**对图做全局优化；若要“静态图 + 编译”，需用 `torch.compile`、TorchScript 等。

### 2.2 JAX：追踪 + 变换

- **前向**：`grad(f)(x)` 时，JAX 先用**抽象值（tracer）**对 `f` 做一次“假执行”，不真算数值，只记录出现的算子与数据流，得到 `f` 的**抽象计算图**。
- **求导**：对这张图做**自动微分变换**，得到“梯度函数”对应的新图，再对真实输入 `x` 执行，得到 $\nabla f(x)$。
- **特点**：
  - 图在 **trace 阶段**就固定了；控制流（if/for）若依赖具体数值，必须用 `jax.lax.cond`、`jax.lax.scan` 等**显式写进图里**，否则只会 trace 到“某一分支”。
  - 求导、JIT、向量化等都是对“同一个函数”做**可组合的变换**（`grad(jit(f))`、`vmap(grad(f))` 等），容易做高阶导、批求导。
  - 一次 trace 可被多次复用，便于 XLA 编译和跨设备执行。

---

## 3. 典型用法对比

### 3.1 一阶梯度

- **PyTorch**：  
  `y = f(x)`，`y.backward()`，然后从 `x.grad` 取梯度；或 `torch.autograd.grad(y, x)`。
- **JAX**：  
  `grad_f = jax.grad(f)`，然后 `grad_f(x)` 得到 $\nabla f(x)$；无“就地写入 .grad”，梯度以**返回值**形式给出。

### 3.2 高阶导数

- **PyTorch**：  
  对 `grad_x = torch.autograd.grad(y, x, create_graph=True)` 再对 `grad_x` 求导，可实现二阶导，但写法较啰嗦，且要记得 `create_graph=True`。
- **JAX**：  
  `jax.grad(jax.grad(f))` 或 `jax.hessian(f)` 等，**同一套变换组合**即可；高阶导与一阶导在用法上一致。

### 3.3 向量化与批处理

- **PyTorch**：  
  批维度通常由用户自己加（如 `batch_dim`），或靠广播；自定义批行为要手写或借助 `torch.vmap`（较新）。
- **JAX**：  
  `vmap(f)` 直接得到“沿某轴批处理”的版本，和 `grad`、`jit` 组合很自然，例如 `vmap(grad(loss))(batch_x)`。

### 3.4 控制流

- **PyTorch**：  
  普通 Python 的 `if`、`for` 直接参与建图；不同 batch 可以走不同分支，灵活，但图会随数据变化。
- **JAX**：  
  trace 时若用普通 `if/for` 且条件依赖输入，只会 trace 到**当前 trace 时走到的分支**；若要“图里包含分支/循环”，需用 `jax.lax.cond`、`jax.lax.scan`、`jax.lax.while_loop` 等，把控制流变成**显式算子**。

### 3.5 随机数

- **PyTorch**：  
  `torch.manual_seed(...)` 等全局状态；同一函数多次调用，随机性由全局状态决定。
- **JAX**：  
  `key = jax.random.PRNGKey(seed)`，然后 `key, subkey = jax.random.split(key)`，把 `subkey` 传入函数；随机性**显式传递**，便于复现和并行。

---

## 4. 性能与编译

| 方面 | PyTorch | JAX |
|------|---------|-----|
| **默认执行** | 算子逐条执行（eager），无整体图优化。 | 默认也是逐条执行，但设计围绕“可 JIT”；`jit(f)` 后整函数编译成 XLA。 |
| **图优化** | 需显式用 `torch.compile`（或 TorchScript）才做融合、重排等。 | JIT 后通常做融合、常量折叠、设备放置等，与 **XLA** 深度结合。 |
| **跨设备** | 主要 GPU（CUDA）；分布式、多机靠额外库。 | 同一套代码较易在 CPU/GPU/TPU 间切换，XLA 负责下发。 |

---

## 5. 与 Qeq 实现的对应关系

- **DP-QEq（JAX）**：  
  - 能量 `get_Energy_Qeq_2(charges, positions, ...)` 是**纯函数**；  
  - `jax.grad(get_Energy_Qeq_2)(...)` 直接得到对 `charges` 的梯度，无需建 Hessian；  
  - 与 `jaxopt.LBFGS`、投影梯度组合自然；随机与状态都通过参数传递，适合 JIT 和 vmap。

- **CACE-SOG-Qeq（PyTorch）**：  
  - `EwaldPotential`、`ChargeEq` 是 `nn.Module`，前向里用 `torch` 算子；  
  - 梯度靠 `loss.backward()` 或 `torch.autograd.grad`，**同样不需要手写 Hessian**；  
  - 控制流、调试、与现有 PyTorch 生态（优化器、DataLoader、部署）一致；若要做“DP-QEq 式”LBFGS，只需在 PyTorch 里实现能量+投影梯度+LBFGS 即可，自动微分只负责提供一阶梯度。

---

## 6. 小结表

| 对比项 | PyTorch 自动微分 | JAX 自动微分 |
|--------|------------------|--------------|
| 图构建时机 | 运行时按实际执行路径记录 | 对函数 trace 一次得到抽象图 |
| 控制流 | 原生 if/for 直接参与图 | 需 lax.cond/scan 等显式算子 |
| 梯度存放 | 写入张量 `.grad` | 以返回值形式返回 |
| 高阶导 | 支持，需 create_graph 等 | 直接 grad(grad(f)) 等组合 |
| 批求导 / vmap | 多靠手写或较新的 vmap | vmap 与 grad/jit 同属 transform，组合自然 |
| 随机数 | 全局 seed | 显式 key 传递 |
| 编译/优化 | 可选（torch.compile） | 围绕 JIT + XLA 设计 |
| 典型场景 | 灵活训练、调试、工业部署 | 科研、科学计算、组合变换、TPU |

**结论**：两者都能为 Qeq 提供“能量 + 一阶梯度”，无需手写 Hessian。差异主要在于**图如何建、何时建、能否被编译与组合**：PyTorch 更偏“命令式、易调试”，JAX 更偏“函数式、易变换与编译”。在 CACE 中实现 DP-QEq 式 LBFGS 时，用 PyTorch 的 `autograd.grad` 或 `backward` 提供梯度即可，数学上与 JAX 版本等价。
