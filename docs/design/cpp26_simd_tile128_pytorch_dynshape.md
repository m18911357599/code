# SIMD-Tile 编程模型（TILE = 128B）：问题与解法、PyTorch Pattern 写法、易用性对比（GPU / AscendC）

> **TILE 定义（修正）**：`TILE = 128` 指 **128 字节（128B）**，不是 128 个 element。  
> 每 tile 的 **元素个数**随 dtype 变化：`lanes = 128 / sizeof(T)`（要求 `sizeof(T)` 整除 128；子字节类型另议）。

---

## 0. 文档目标与结论

### 0.1 目标

1. 采用 **simd-tile** 编程模型时，会遇到哪些问题，如何解决  
2. 分析 PyTorch 各类 pattern 在 simd-tile 下的编程方式  
3. 度量 simd-tile 的编程易用性，并与 **GPU（CUDA）**、**AscendC** 对比  

### 0.2 结论摘要

| 问题域 | 结论 |
|---|---|
| TILE=128B | 向量/搬运以 **字节块** 为量子；`float`→32 lane，`half/bf16`→64，`int8`→128 |
| 核心难点 | dtype 变 lane、动态 shape 尾块、broadcast/reduce 轴、跨步与对齐、专用算子不适合裸 tile 循环 |
| 解法主线 | **字节域切块 + dtype 域运算**；整块满 128B、尾块 mask/epilogue；Broadcast/Reduce 用 plan（collapse） |
| Pattern 写法 | Elementwise / Broadcast / Reduce / Layout 可统一 simd-tile；GEMM/Conv/Attn/DynOut 走专用 API，仅复用 128B 分块约定 |
| 易用性 | 相对手写 AVX/NEON：**明显更高**；相对 CUDA elementwise：**接近或略低**；相对 AscendC 矢量 API：抽象更短，但 **缺管道/多缓冲/内存层级显式控制** 时性能表达力偏弱 |

---

## 1. SIMD-Tile 编程模型

### 1.1 定义

```text
TILE_BYTES = 128
lanes<T>   = TILE_BYTES / sizeof(T)     // 例: f32→32, f16→64, i8→128
```

```cpp
inline constexpr std::size_t TILE_BYTES = 128;

template<class T>
inline constexpr std::size_t lanes_v = TILE_BYTES / sizeof(T);

template<class T>
using tile_t = /* C++26 simd 或后端向量类型，宽度 = lanes_v<T> */;

template<class T>
using mask_t = /* 与 tile_t<T> 同宽的谓词 */;
```

| 概念 | 含义 |
|---|---|
| **simd-tile** | 一次处理 **最多 128B** 有效载荷的向量步进单位 |
| **整块** | 连续 `TILE_BYTES` 对齐（或逻辑满宽）的一步 |
| **尾块** | 剩余 `< 128B`（按字节）或不足 `lanes<T>` 个元素的一步 |
| **GPU-like** | grid-stride 迭代多个 tile；尾块用 mask/predication |

与「固定 128 个 element」的差异：同一套循环在 `f32` 与 `f16` 下 **迭代次数、mask 宽度、归约树深度都不同**，必须以 **字节预算** 为第一性，再映射到 lane。

### 1.2 最小循环骨架（elementwise，连续）

```cpp
// L、指针按「元素」计；切块边界按字节预算换算
const int64_t lanes = lanes_v<T>;                 // 128/sizeof(T)
int64_t i = 0;
for (; i + lanes <= n; i += lanes) {              // 整块：满 128B
  tile_t<T> x = load_full(in + i);
  store_full(out + i, map(x));
}
if (i < n) {                                      // 尾块：<128B
  mask_t<T> m = lane_lt(n - i);                   // 有效 element 数
  store_masked(out + i, m, map(load_masked(in + i, m)));
}
```

---

## 2. 采用 SIMD-Tile 时遇到的问题与解决方案

### 2.1 问题总表

| ID | 问题 | 表现 | 解决方案（摘要） |
|---|---|---|---|
| P1 | **dtype → lane 数变化** | 同 `n` 元素，f32/f16 步长不同 | API 以元素语义书写；lowering 用 `lanes_v<T>`；禁止写死 `128` 当 element 数 |
| P2 | **动态 shape / 非对齐长度** | `n` 或行长不能整除 `lanes` | 整块满 128B + **至多一次尾块**（mask 或窄向量 epilogue） |
| P3 | **尾块按字节还是按元素** | 混合 dtype、结构化类型易错 | **对外元素语义**；对内 `tail_elems = n % lanes`，等价剩余字节 `< 128` |
| P4 | **Broadcast 多输入步长** | 各输入 `stride_eff` 不同 | 输出主导切块；每输入独立 address 生成；先 **dim collapse** |
| P5 | **Reduce 轴方向与短内维** | 内维 `< lanes` 时几乎全是尾块 | 自动拆 `outer/reduce/inner`；短内维 **多行打包** 进 128B |
| P6 | **跨步 / 非 contiguous** | 无法 unit-stride 满 128B load | 能则 `contiguous`；热路径 strided load；否则 gather；度量带宽 |
| P7 | **对齐与安全 load** | 128B 向量要求对齐或用 unaligned 指令 | 契约：支持 unaligned；对齐走快路径 |
| P8 | **谓词拖累整块** | 统一 mask 循环使满块也付 α_pred | **整块/尾块分流**：满 128B 无谓词 |
| P9 | **多 dtype / 类型提升** | binary op 提升后 tile 宽度变化 | 提升后的 compute dtype 决定 `lanes`；load 时 cast 进 tile |
| P10 | **子字节 / 非 2 幂 sizeof** | 不整除 128B | 不进通用 tile；专用路径或打包到字节容器 |
| P11 | **专用算子（GEMM 等）** | 裸 tile map 无法达峰值 | 领域 API；仅复用 128B 分块与尾块原语（§4.6） |
| P12 | **与硬件物理 VL 不一致** | CPU VL=32B/64B，NPU 向量单元另有宽度 | 128B 为 **软件逻辑量子**；后端再切 physical VL |

### 2.2 关键问题展开与解法

#### P1 — 128B 与 lane 的换算

| dtype | `sizeof` | `lanes` / tile | 备注 |
|---|---|---|---|
| `int8` / `uint8` | 1 | 128 | |
| `float16` / `bfloat16` | 2 | 64 | |
| `float` / `int32` | 4 | 32 | |
| `double` / `int64` | 8 | 16 | |
| `bool`（实现相关） | 常按 1B 存 | 视后端 | 位打包需专用 |

**解法**：所有用户可见 API 用「元素下标」；框架内：

```text
step_elems(T) = 128 / sizeof(T)
tail_elems    = n % step_elems(T)
tail_bytes    = tail_elems * sizeof(T)   // < 128
```

#### P2/P3/P8 — 尾块（128B 语义下）

| 策略 | 做法 | 何时用 |
|---|---|---|
| **T-Mask** | `mask = lane_id < tail_elems` | 默认；硬件谓词强 |
| **T-Split** | 尾块再按物理 VL 切；最后标量/窄向量 | 谓词贵 |
| **T-Pad（仅内部）** | scratch pad 到 128B，算完写回有效前缀 | 小尾且有廉价缓冲；**禁止作 API 契约** |

推荐热路径：**整块无 mask，尾块单独一次**（解决 P8）。

#### P4 — Broadcast

- 输出线性域按 `lanes_out`（由 **compute/output dtype** 的 128B 决定）切块  
- 每输入：`offset = Σ coord[d] * stride_eff[d]`  
- **先 collapse** 可合并维，把复杂广播降为标量/末维广播  

#### P5 — Reduce

- 运行时 plan：`{outer, reduce_len, inner}`  
- 优先 **内维归约**（连续 128B load + 水平 reduce）  
- `inner * sizeof(T) < 128`：标记 `ShortInner`，多行拼满 128B，或转置后再归约  
- 尾块无效 lane 填 **中性元**（0、+inf、true…）  

#### P6/P7 — 跨步与对齐

```text
if unit-stride && (offset % align_ok):  full 128B vector load
else if low-stride:                     strided / 多次窄 load 拼 tile
else:                                   gather
```

#### P9 — 类型提升

例：`int32 + float` → compute `float`，tile 按 **4B lane×32**；整数输入 load 后 cast。避免「按输入 dtype 各切各的 128B」导致 lane 对不齐。

#### P11 — 边界：何时退出通用 simd-tile

见 §4.6；原则：**能 map/reduce/broadcast 表达的留在通用模型；要 MMA/多缓冲/间接压缩的进专用 API**。

---

## 3. PyTorch 各类 Pattern 在 SIMD-Tile 下的编程方式

术语：ATen `pointwise` ≡ 本文 **elementwise**。切块量子均为 **128B**。

### 3.1 Pattern 一览

| Pattern | PyTorch 代表 | simd-tile 写法纲要 | 整块 / 尾块 |
|---|---|---|---|
| **E0 Elementwise 对齐** | `add/mul/relu` 同形连续 | 满 128B map 循环 + 尾块 mask | 按输出 dtype 的 `lanes` |
| **E1 Elementwise + where** | `clamp/where` | map + blend；尾块同 E0 | 同上 |
| **B0 标量广播** | `x + scalar`、`x + bias[1]` | splat 进 tile，再 E0 | 同上 |
| **B1 末维/可 collapse 广播** | `[B,1,K]+[B,H,K]` 可合并 | collapse → E0/B0 | 合并后一次尾块 |
| **B2 通用广播** | 任意 `stride_eff` | 输出主导；每 lane 算多输入地址（或先 materialize） | 输出 `numel` 尾块 |
| **R0 全维 Reduce** | `sum()` | 128B 累加 → 水平 reduce → 跨 tile 合并 | 输入侧尾块 + 中性元 |
| **R1 内维 Reduce** | `sum(-1)` | 每行：整块 128B + 行尾 | **每行**一个尾块 |
| **R2 外维/中间维** | `sum(0)` / `sum(1)` | 拆 outer×reduce×inner；能合并则转 R1 | 见 plan |
| **R3 短内维 Reduce** | 特征维很小 | 多行打包填满 128B 再归约 | 打包后尾块按字节 |
| **L0 Layout copy** | `clone/contiguous/cat` 段 | 按 **128B** 搬运（甚至与 dtype 无关的 memcpy 向量化） | 剩余 `<128B` 窄拷贝 |
| **L1 转置/ permute** | `transpose` | 128B×128B 块转置或 gather；边缘块专用 | 2D 边缘尾块 |
| **X\* 专用** | `mm/conv/softmax/nonzero` | §4.6 领域 API | 边缘块 epilogue |

### 3.2 Elementwise（E0/E1）

```text
elementwise(out, inputs..., op):
  T = compute_dtype(...)
  lanes = 128 / sizeof(T)
  FullTiles: load_full → op → store_full
  Tail:      mask(lane < rem) → load/op/store masked
```

**编程要点**：用户只写标量 `op`；不出现 `128` 字面量（除非读文档约定）。

### 3.3 Broadcast（B0–B2）

| 场景 | 编程方式 |
|---|---|
| B0 | `tile = simd(scalar)` 或首 lane broadcast |
| B1 | `BroadcastPlan.collapse()` 后同 E0 |
| B2 | `for each out_tile: for each input: gather_or_stride_load(addr(lane))`；代价高时对小输入先 expand |

动态 shape：每次 launch 重算 `stride_eff` 与 collapse，不换 kernel 族。

### 3.4 Reduce（R0–R3）

| 场景 | 编程方式 |
|---|---|
| R0 | 全局 `acc`；整块 `acc = combine(acc, reduce_tile(load_full))`；尾块中性元 |
| R1 | `for outer: acc=neutral; for tiles along K; write acc` |
| R2 | 索引重排或临时缓冲；优先降维到 R1 |
| R3 | `pack rows into 128B tile → partial reduce → 写回各 row` |

**Arg-reduce**（`argmax`）：tile 内持 `(val,idx)`，尾块禁用无效 idx。

### 3.5 Layout（L0/L1）

- **L0**：可按纯字节循环（`bytes` 步长 128），与 element 无关，尾块 `<128B`  
- **L1**：块内转置以 128B 为边；不完全块走 masked/scalar  

### 3.6 组合模式（PyTorch 常见）

| 组合 | 例 | simd-tile 组织 |
|---|---|---|
| Broadcast → Elementwise | `x + bias` | B\* 降到 E0 |
| Reduce → Broadcast → EW | `LayerNorm` 简化版 | R1 两遍 + E0；完整 Norm 见专用 |
| Cat 多段 copy | `torch.cat` | 每段独立 L0（各自尾块） |

### 3.7 与 TensorIterator 的对应

| TensorIterator 概念 | simd-tile |
|---|---|
| 输出线性迭代 | 按 `lanes(compute_dtype)` 步进 |
| operand stride | `stride_eff` |
| 2D / collapse | `BroadcastPlan` / `ReducePlan` |
| vectorized kernel | 整块 128B |
| scalar / unrolled 尾 | 尾块 mask / epilogue |

---

## 4. 专用 Pattern（不进通用 tile map）

这些仍可遵守「边缘按 128B / 不足则 epilogue」，但 **编程入口是领域 API**。

| 类别 | 代表 | simd-tile 角色 | 推荐写法 |
|---|---|---|---|
| GEMM / BMM | `addmm` | 分块边界、epilogue 融合 | `gemm_tiled`（MMA），非三层 simd 累加 |
| Conv | `conv2d` | im2col 缓冲按 128B；或专用滑窗 | `conv` 模板 |
| Attention | SDPA | 序列块对齐到 128B 载荷 | 融合 kernel |
| Softmax / Norm | `softmax` | 内维按 `lanes` 切；复用 R1 尾块 | `softmax_inner` |
| Gather/Scatter | `index_add` | 每步最多 `lanes` 个索引 | `gather/scatter` 引擎 |
| DynOut | `nonzero` | mask → compact；末块 mask | 二阶段 count + write |
| Sort / 全局 Scan | `sort/cumsum` | 局部 128B 网络 + 跨块 | 算法库 |

---

## 5. 易用性度量：SIMD-Tile vs GPU vs AscendC

### 5.1 评分维度（1–5，越高越易用）

| 维度 | 含义 |
|---|---|
| D1 心智负担 | 是否需理解硬件层级 / 管道 / mask |
| D2 动态 shape | 任意长度、广播、轴是否自然 |
| D3 尾块 / 对齐 | 是否需手写 epilogue、对齐约束 |
| D4 表达长度 | 完成同算子的代码量 / API 层数 |
| D5 可移植性 | 一处编写多后端 |
| D6 性能表达力 | 易用 API 是否仍能打满硬件（易用且不「写了也慢」） |

### 5.2 模型对照（写作模型）

| 模型 | 典型写法 | 并行量子 |
|---|---|---|
| **SIMD-Tile（本文）** | `elementwise` / `reduce` + 内部 128B tile | **128B** / dtype→lanes |
| **GPU (CUDA)** | `grid-stride` + 标量 thread；或 CUB/Thrust；或编译器向量化 | thread；warp=32；向量 load 常 16B |
| **AscendC** | `LocalTensor` + `Add/Mul/Reduce` 等 API；`DataCopy`；Pipe 多缓冲 | 矢量指令 + UB/L1 层级；实现侧常按块/重复态 |

### 5.3 分 Pattern 易用性对比

| Pattern | SIMD-Tile | GPU (CUDA) | AscendC | 说明 |
|---|---|---|---|---|
| Elementwise 对齐 | **5** | 5（或 4 手写 kernel） | 4 | Tile/CUDA 均可标量语义；AscendC 需 Tensor/API 形态 |
| Broadcast | **4–5** | 5（TensorIterator/ATen） | 3–4 | AscendC 常显式扩或依赖高阶 API |
| 内维 Reduce | **4** | 4–5（CUB） | 4 | 三者都有库；短内维均需技巧 |
| 外维/多维 Reduce | **3–4** | 4 | 3–4 | GPU 生态更成熟（CUB/block reduce） |
| Layout copy | **5** | 5 | 4 | 128B 与 DMA/DataCopy 同构 |
| Gather/Scatter | **3** | 3–4 | 3 | 均要间接指令；易用性接近 |
| Softmax/Norm | **3**（专用 API） | 3–4 | 3–4 | 都应领域核，不宜裸循环 |
| GEMM/Conv | **2** | 2（写 kernel）/5（cuBLAS） | 2–3（Cube API） | 比较的是「手写」；库级另计 |
| DynOut (`nonzero`) | **2–3** | 3 | 2–3 | 皆二阶段 |
| 管道 / 多缓冲重叠 | **2**（模型默认不暴露） | 3–4（stream/async） | **5** | AscendC 显式 Pipe 更强 |
| 多级存储 (UB/L1/GM) | **2**（抽象掉） | 3（shared/global） | **5** | AscendC 控制力最强，易用代价是概念多 |

### 5.4 综合对比（编程易用性）

| 维度 | SIMD-Tile (128B) | GPU (CUDA 手写) | GPU (+ATen/库) | AscendC 基础矢量 API |
|---|---|---|---|---|
| D1 心智负担 | **4–5** | 2–3 | 5 | 3 |
| D2 动态 shape | **4–5**（尾块内建） | 3–4（需自写） | 5 | 3–4（常要对齐/块规划） |
| D3 尾块/对齐 | **4–5**（框架分流） | 3 | 5 | 3（重复态/尾块模式需熟） |
| D4 表达长度 | **5**（L0 lambda） | 2–3 | 5 | 3–4 |
| D5 可移植 | **4–5**（逻辑 128B） | 2 | 3–4 | 2（绑 NPU） |
| D6 性能表达力 | **3** | 4–5 | 4–5（库） | **5**（Pipe+层级+Cube） |
| **综合（通用 EW/Bcast/Reduce）** | **高** | 中（手写）/高（库） | **高** | **中高** |
| **综合（要打满峰值的融合核）** | 中低 | 高 | 高（库/编译器） | **高** |

### 5.5 解读（给选型）

1. **SIMD-Tile 的定位**：把「128B 向量步进 + 尾块」做成与 GPU grid-stride 同级的 **可移植编写模型**，专门抬高 elementwise/broadcast/reduce/layout 的易用性与动态 shape 正确性。  
2. **相对 GPU**：写通用算子时，simd-tile ≈ 良好封装的 CUDA 辅助库；不比 ATen 更易，但比手写 kernel 易。GPU 在 warp 原语、生态库（CUB/cuBLAS）上更完整。  
3. **相对 AscendC**：simd-tile 更短、更少硬件概念（无强制 Pipe/UB/L1）；AscendC 在 **性能表达力与硬件发挥** 上更强，学习与代码结构成本更高。  
4. **128B 带来的独特心智点**：必须记住 **lane 数随 dtype 变**——这是相对「按 element 定宽」或 CUDA「一线程一元素」多出的一点；应用 `lanes_v<T>` 隐藏后，用户层可忽略。  
5. **不宜用 simd-tile 硬写的**：GEMM/Conv/FlashAttn/深度管道融合——三方都应走专用 API/库；比较易用性时应比领域 API，而不是比三层循环。

### 5.6 量化度量建议（工程落地时）

| 度量 | 方法 |
|---|---|
| 代码行数 / API 调用数 | 同算子三种模型实现对比（EW、broadcast add、sum dim=-1） |
| 首次正确时间 | 新手写动态 `n`、含尾块用例 |
| 性能比 | 相对手写峰值或库：`eff = achieved / peak`；看尾块占比 `r_tail_bytes` |
| 抽象泄漏次数 | 用户代码中出现 `128`、对齐、`DataCopy`、`__syncthreads`、`Pipe` 的次数 |

```text
r_tail_bytes ≈ (n % lanes) * sizeof(T) / (n * sizeof(T)) = (n % lanes) / n
η_vec        ≈ n / (ceil(n/lanes)*lanes)     // 按 element lane 槽位
```

---

## 6. 推荐软件表达（绑定解法）

```text
// L0：用户 —— 元素语义，无 128、无 mask
elementwise(out, a, b, [](auto x, auto y){ return x * y; });
reduce(out, x, dims={-1}, op=sum);

// L1：框架 plan —— 处理 broadcast/reduce/collapse / dtype→lanes
BroadcastPlan / ReducePlan / TensorDesc

// L2：后端 —— 128B 整块 + 尾块；physical VL 再切；AscendC/GPU/CPU 映射
```

**硬规则**

1. API 契约不要求 `nbytes % 128 == 0`  
2. 用户代码不写死 element 数 `128`  
3. 热路径整块无谓词；尾块单独  
4. 专用算子用领域名（`gemm`/`softmax_inner`），不伪装成 `elementwise`  

---

## 7. 附录

### 7.1 术语

| 用语 | 含义 |
|---|---|
| **TILE / 128B** | 128 字节逻辑向量量子 |
| **lanes** | `128 / sizeof(T)` 个元素 |
| **elementwise** | 逐元素；ATen tag 名 `pointwise` |
| **simd-tile 模型** | 以 128B tile 步进的编程与 lowering 约定 |

### 7.2 参考

- C++26 `std::datapar::simd`（P1928）  
- PyTorch TensorIterator / ATen tags（`pointwise`/`reduction`/`dynamic_output_shape`）  
- CUDA grid-stride loop；CUB reductions  
- AscendC 矢量 API / DataCopy / Pipe（与逻辑 128B 分块的映射需按产品向量宽度再切）  

---

*本版修正：TILE=128B（非 element 个数）；目标收敛为「问题与解法 / Pattern 写法 / 对 GPU·AscendC 易用性对比」。*
