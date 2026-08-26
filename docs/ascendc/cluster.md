# AscendC Cluster 模式定义

> 对照 NVIDIA Thread Block Cluster：[NV Cluster 定义与三种调用模式](../cuda/cluster.md)  
> AscendC 同时覆盖 **SIMT**（Grid–Block–Thread）与 **SIMD / Vec-Tile**（多核 + Pipe）两套编程面。

**结论**：AscendC 的 **Cluster** 是「如何把逻辑工作切到物理/逻辑 Core」的调度与分组边界。  
- **SIMT**：层次与 NV 对齐 → **`grid → block`**，Cluster 描述 **core 切分**（共调度/核组边界）。  
- **SIMD**：显式三层 → **`cluster → blockNum → block`**。

---

## 1. 术语对齐（NV ↔ AscendC）

| 概念 | NVIDIA（SIMT） | AscendC SIMT | AscendC SIMD |
|------|----------------|--------------|--------------|
| 顶层任务网格 | Grid（按 Block 计，或模式 C 按 Cluster 计） | **Grid**（`gridDim` / `blockIdx`） | **Cluster**（一次算子任务的核组顶层） |
| 可选共调度组 | **Cluster**（同 GPC） | **Cluster**（如何切分/绑定 Core） | （已提升为顶层） |
| 逻辑执行单元数 | `gridDim`（Block 个数） | Block 个数 ≈ 映射到 AIV 上的线程块任务 | **`blockNum`**（`<<<blockNum, ...>>>` / `GetBlockNum`） |
| 单核/单块实例 | Thread Block | Thread Block（同刻一 AIV 上一 Block） | **Block**（一逻辑 AI Core 实例，`GetBlockIdx`） |
| 块内并行 | Warp → Thread | Warp → Thread | Vec/Tile + Pipe（非线程模型） |

设计原则：

1. **SIMT 与 NV 保持一致**：开发者看到的主轴仍是 `grid → block (→ thread)`；Cluster **不替代** Grid/Block，只回答「这些 Block 如何落到 Core / 核组」。
2. **SIMD 显式分层**：没有 Thread，并行轴是多核；用 `cluster → blockNum → block` 把「核组 → 逻辑核数 → 当前核」说清。

---

## 2. SIMT 架构：与 NV 一致

### 2.1 主轴层次

```
Grid                         ← 线程块网格（与 NV 同构）
 └─ [Cluster]                ← 可选：如何切分 / 共调度 Core（对齐 NV Cluster 语义）
     └─ Thread Block         ← 同刻映射到一颗 AIV 上的一块任务
         └─ Warp (32 threads)
             └─ Thread
```

| 层 | AscendC SIMT 含义 | 与 NV 对应 |
|----|-------------------|------------|
| **Grid** | `gridDim`；Block 总数上限等同平台约束（如 ≤ 65535） | Grid |
| **Cluster** | **Core 切分策略**：一组 Block / 任务如何绑定到物理 AIV（或核组），表达共调度与资源边界 | Thread Block Cluster → GPC |
| **Block** | `blockDim` / `blockIdx`；块内共享 UB（`__ubuf__`） | Thread Block → SM |
| **Thread** | `threadIdx`；寄存器私有 | Thread |

Launch 形态（与 NV 经典路径同构）：

```cpp
// SIMT：主轴仍是 grid → block
kernel<<<gridDim, blockDim, dyn_ubuf_size, stream>>>(...);
```

内置变量（SIMT）：`gridDim`、`blockDim`、`blockIdx`、`threadIdx`。

### 2.2 Cluster 在 SIMT 下的职责

在 SIMT 下 **不要** 把 Cluster 做成另一套与 `grid/block` 平行的主索引；其职责对齐 NV：

| 职责 | 说明 |
|------|------|
| **切分 Core** | 规定哪些 Thread Block 落到哪些 AIV / 核组（数量、绑定、共驻） |
| **共调度边界** | 需要跨 Block 同步或共享近端缓冲时，Cluster 给出可依赖的硬件边界（类比 NV 同 GPC） |
| **与 Grid 正交** | `gridDim` 仍描述「有多少 Block」；Cluster 描述「这些 Block 如何成组落到 Core」 |

对照 NV 三种调用模式（见 [cuda/cluster.md](../cuda/cluster.md)）的语义落点：

| NV 模式 | AscendC SIMT 侧对应理解 |
|---------|-------------------------|
| A `__cluster_dims__` | 编译期固定「每 Cluster 含多少 Block / 绑多少 Core」 |
| B `cudaLaunchKernelEx` + ClusterDimension | 运行时按负载选择核组大小 / Core 切分 |
| C `__block_size__`（按 Cluster 计 Grid） | Host 直接按「核组个数」描述任务；块大小与核组内块数收进属性 |

> 实现上平台 API 名称可不同于 CUDA；**分层语义**应保持：Grid/Block 主轴不变，Cluster = Core 切分。

### 2.3 SIMT 数据切分（示意）

```cpp
// 与 CUDA grid-stride / block-stride 同构
int tid = blockIdx.x * blockDim.x + threadIdx.x;
int stride = gridDim.x * blockDim.x;
for (int i = tid; i < n; i += stride) {
    // per-thread 元素
}
```

Cluster 只影响「这些 Block 是否共驻、如何绑核」，**不改变** `blockIdx/threadIdx` 的计算习惯。

---

## 3. SIMD 架构：`cluster → blockNum → block`

SIMD / AscendC Vec 路径没有 Thread 网格；并行来自 **多逻辑 AI Core** + 核内 **向量/Tile + Pipe**。Cluster 提升为顶层分组。

### 3.1 三层定义

```
Cluster                      ← 一次算子/任务的核组（如何组织多核）
 └─ blockNum                 ← 本 Cluster（本任务）配置的逻辑核数
     └─ block                ← 单个逻辑 AI Core 实例（GetBlockIdx）
         └─ Vec / Tile / Pipe   ← 核内 SIMD 与流水（非线程）
```

| 层 | 定义 | 典型 API / 配置 |
|----|------|-----------------|
| **Cluster** | 多核任务的顶层切分单元：一组逻辑核及其绑定关系（耦合/分离、AIC:AIV 比等） | 任务/核组配置；平台核资源与 `GetTaskRatio` 等 |
| **blockNum** | 该任务启动的 **逻辑 AI Core 个数** | Host：`<<<blockNum, ...>>>`；Device：`AscendC::GetBlockNum()` |
| **block** | 当前正在执行的那一颗逻辑核 | `AscendC::GetBlockIdx()` ∈ `[0, blockNum)`（融合启动时 AIV 范围可能按 ratio 扩展） |

### 3.2 与 SIMT / NV 的差异

| 维度 | SIMT / NV | AscendC SIMD |
|------|-----------|--------------|
| 顶层 | Grid（Block 网格） | **Cluster**（核组） |
| 「有多少并行实例」 | `gridDim`（× 可选 Cluster） | **`blockNum`** |
| 「我是谁」 | `blockIdx`（+ `threadIdx`） | **`GetBlockIdx()`**（无 threadIdx） |
| 块内并行 | 多 Thread SIMT | 向量 lane / Tile + MTE/Pipe |
| Cluster 角色 | Grid 与 Block 之间的可选层 | **顶层**；其下才是 blockNum → block |

关系式（单 Cluster 任务的常见情况）：

```
逻辑工作集
  → 按 Cluster 策略绑到一组 Core
  → 用 blockNum 展开为多个 block
  → 每个 block 用 GetBlockIdx 算 GM 偏移，核内再按 Tile 向量化
```

### 3.3 SIMD 多核切分（示意）

```cpp
// Host：配置逻辑核数
add_custom<<<blockNum, nullptr, stream>>>(x, y, z);

// Device（SIMD）
int64_t blockNum = AscendC::GetBlockNum();
int64_t blockIdx = AscendC::GetBlockIdx();
int64_t perBlock = totalLength / blockNum;

xGm.SetGlobalBuffer((__gm__ T*)x + blockIdx * perBlock, perBlock);
// 核内：Pipe + Vec/Tile 处理本 block 分片
```

**融合 AIC/AIV 启动时**（分离架构常见）：

| AIC:AIV | AIC 上 `GetBlockIdx` | AIV 上 `GetBlockIdx` |
|---------|----------------------|----------------------|
| 1:2 | `[0, blockNum)` | `[0, 2 * blockNum)` |
| 1:1 | `[0, blockNum)` | `[0, blockNum)` |

此时 **Cluster** 仍表示「这一组 AIC/AIV 如何组成可调度核组」；**blockNum** 是逻辑组合个数；**block** 是组合内具体核实例。

### 3.4 耦合 vs 分离（影响 Cluster 内 Core 组成）

| 架构 | Cluster / blockNum 含义要点 |
|------|-----------------------------|
| **耦合** | Vector 与 Cube 同核；`blockNum` 启多个 AI Core 实例，不区分 AIV/AIC |
| **分离** | 纯 Vec → 按 AIV 数设 `blockNum`；纯 Cube → 按 AIC；融合 → 按「组合」数设 `blockNum`（如 2 AIV + 1 AIC 为一物理组合） |

`GetCoreNumAiv` / `GetCoreNumAic` / `GetTaskRatio` 用于把 **Cluster 的物理组成** 落到具体 `blockNum`。

---

## 4. 双架构对照总表

```
SIMT（与 NV 一致）:
  Grid → [Cluster: 如何切分 Core] → Block → Warp → Thread
  Launch: <<<gridDim, blockDim, ...>>>
  索引:   blockIdx, threadIdx

SIMD:
  Cluster → blockNum → block → (Vec/Tile/Pipe)
  Launch: <<<blockNum, ...>>>
  索引:   GetBlockIdx(), GetBlockNum()
```

| | SIMT | SIMD |
|--|------|------|
| 是否引入 Thread | 是 | 否 |
| Cluster 位置 | Grid 与 Block **之间**（可选，对齐 NV） | **最顶层** |
| 「核数」参数名 | 由 Grid/Cluster 映射到 AIV | **`blockNum`** |
| 编程习惯 | 一线程一元素 / grid-stride | 一核一分片 + 向量 Tile |
| 核内同步 | 块内 barrier；Cluster 级共调度同步（若平台提供） | `PipeBarrier` / `SetFlag` / `WaitFlag` |

---

## 5. 模式选用建议

| 场景 | 建议层次 |
|------|----------|
| 从 CUDA SIMT 算子迁移、需要 threadIdx 表达 | **SIMT**：`grid → block`；用 Cluster 仅表达 Core 切分 |
| 经典 AscendC Vec 算子、多核 GM 分片 | **SIMD**：`cluster → blockNum → block` |
| 需要 AIC+AIV 融合流水 | **SIMD** Cluster 描述组合；`blockNum` = 组合数 |
| 需要跨多 Block 共驻与近端共享（对标 NV DSMEM） | **SIMT** 下启用 Cluster 共调度边界；或 SIMD 下用平台核组能力 |

---

## 6. 与本仓库其它文档

| 文档 | 关系 |
|------|------|
| [NV Cluster 三种调用模式](../cuda/cluster.md) | SIMT 侧语义与 Host 三种配置方式的参照 |
| [PyTorch 算子特征 / SIMT·SIMD 打分](../design/pytorch_op_feature_compete.md) | 算子更亲 SIMT 或 SIMD 的选型依据 |
| [CCU 指令格式](../ccu/01-underlying-instruction-format.md) | 通信微码；与计算 Cluster 分层正交 |

---

## 7. 参考

| 来源 | 内容 |
|------|------|
| Ascend C 编程模型（异构并行 / SPMD） | 多核 `block_idx`、数据按核切分 |
| Ascend C SIMT 线程架构 | `Grid → Block → Thread`，AIV 上 Block/Warp 调度 |
| `GetBlockNum` / `GetBlockIdx` | SIMD 路径逻辑核数与核索引 |
| Kernel 直调 `blockDim`（逻辑核） | 耦合/分离下 `blockNum` 设置规则 |
| NVIDIA Thread Block Cluster | SIMT 下 Cluster = 共调度/切分边界的业界对齐点 |
