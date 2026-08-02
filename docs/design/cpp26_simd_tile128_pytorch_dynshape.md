# C++26 SIMD（tile_len=128）× GPU-like 编程模型：PyTorch 算子 Pattern 分类与全量动态 Shape 易用性设计

> 目标：在 **固定 SIMD 宽度 `tile_len = 128`**（元素数或等价 lane 数，见 §1）的前提下，用接近 GPU 的编程抽象（block/tile/mask）覆盖 PyTorch/ATen 现有算子 pattern，实现 **全量动态 shape** 的易用编写与可移植映射，并给出指令分类与易用性评估。

---

## 0. 结论摘要

| 维度 | 结论 |
|---|---|
| 编程模型 | 以 C++26 `std::datapar::simd` / `simd_mask` 为 **lane 原语**；之上固定 `TILE=128`，用 GPU-like 的 `blockIdx × tile` + **尾块 mask** 表达动态长度 |
| 动态 shape 关键 | 不要求编译期 shape；运行时仅依赖 `numel / strides / sizes`；所有 kernel 统一 `for (i = tid*TILE; i < n; i += grid*TILE)` + `mask = iota < remain` |
| PyTorch 覆盖 | 按 shape 行为 + 计算结构双轴分类；**~70%+ 算子（TensorIterator/pointwise/reduction/view）** 可落入少数通用 pattern；Fixed/Batched/Matmul/Conv 需专用 tile 模板 |
| 指令需求 | 八大类：算术、谓词/mask、归约、gather/scatter、广播/shuffle、内存（连续/跨步/间接）、同步、特殊数学 |
| 易用性 | Pointwise / 简单 Reduce：**高**；Broadcast+Strided：**中高**；Indexing / Unique 等 data-dependent：**中低**；Matmul/Conv/Attention：**低（需专用 DSL/模板，而非裸 simd）** |

**一句话**：`tile_len=128` 把硬件 SIMD/向量机抽象成「固定宽 GPU warp」；动态 shape 的易用性来自 **统一尾块 mask + 运行时线性化索引**，而不是为每个 shape 特化 kernel。

---

## 1. 基线假设与抽象

### 1.1 C++26 SIMD 角色

C++26 引入 data-parallel types（P1928，`std::datapar::simd` / `basic_simd`）：

- **元素级并行**：`simd` 上运算默认 element-wise、lane 间无序
- **谓词**：`simd_mask` + `where(mask, v) = expr`
- **水平操作**：reduction（sum/min/max…）单独标注
- **宽度**：标准允许实现相关 ABI；本设计 **强制固定逻辑宽度** `TILE = 128`

```cpp
// 逻辑约定（示意，非标准强制 API）
inline constexpr std::size_t TILE = 128;
template<class T>
using tile_t = std::datapar::simd<T, std::datapar::simd_abi::fixed_size<TILE>>;
template<class T>
using mask_t = std::datapar::simd_mask<T, std::datapar::simd_abi::fixed_size<TILE>>;
```

> 说明：物理寄存器可能是 4/8/16/32 lane（AVX/NEON）或可伸缩（SVE）。`TILE=128` 是 **软件逻辑 tile**：编译器可拆成多条物理向量指令，或映射到 NPU/GPU 的 128-lane 向量单元。对 Ascend/类 GPU 后端，这对应「一次处理 128 个元素的向量指令宽度」。

### 1.2 GPU-like 执行模型（固定 tile）

| GPU 概念 | 本模型对应 |
|---|---|
| warp / wavefront | 一个 `tile_t<T>`（128 lanes） |
| threadIdx.x | lane id ∈ `[0,128)` |
| block | 多个 tile 的协作组（可选，用于 shared reduce / transpose） |
| grid-stride loop | `for (base = gid*TILE; base < n; base += grid*TILE)` |
| predicated exec | `mask_t` / `where` |
| shared memory | `tile_shared[TILE]` 或 block-shared buffer（后端提供） |
| global load | `simd::copy_from` / gather；跨步用 strided load 或标量拼装 |

```text
                 numel = N (runtime dynamic)
  ┌──────────────┬──────────────┬─────┬──────────┐
  │ tile 0 (128) │ tile 1 (128) │ ... │ tile k   │  ← last tile masked
  └──────────────┴──────────────┴─────┴──────────┘
  grid-stride: each "core/block" owns a stream of tiles
```

### 1.3 「全量动态 Shape」定义

本设计中的 **全量动态 shape** 指：

1. **Rank / sizes / strides 仅运行时可知**（含 symbolic / 导出后的动态维）
2. Kernel **不** 为具体 `[B,H,W]` 特化代码路径（除可选 JIT 特化）
3. 尾部 `N % 128 ≠ 0` 必须正确（mask），禁止「要求对齐到 128」作为 API 契约
4. Broadcast、非连续 stride、任意 reduce 轴在 **统一运行时描述符** 下工作
5. Data-dependent 输出 shape（`unique`、`nonzero`）允许二阶段，但不退出统一编程模型

---

## 2. PyTorch 现有算子 Pattern 分类

结合 ezyang *shape taxonomy*（~1364 op 变体统计）与 ATen `tags.yaml`（`pointwise` / `reduction` / `dynamic_output_shape` / `core`），整理为 **双轴分类**：

- **轴 A：Shape 行为**（决定索引与输出分配）
- **轴 B：计算结构**（决定指令与 tile 模板）

### 2.1 轴 A — Shape 行为（来自 PyTorch 生态）

| ID | Pattern | 约占比（历史统计） | 代表算子 | 动态 Shape 要点 |
|---|---|---|---|---|
| A1 | **TensorIterator / Pointwise** | ~505（最大头） | `add/mul/relu/where` | 输出 = broadcast(inputs)；element 独立 |
| A2 | **Reduction** | 含于 TI + tag:reduction | `sum/mean/amax/prod` | 轴运行时指定；keepdim 元数据 |
| A3 | **Fixed-rank** | ~273 | `conv2d/addmm` | 固定 2D/3D 循环；batch 维动态 |
| A4 | **N-D generic** | ~107 | `squeeze/index_add/tensordot` | 任意 rank；需 list 级 shape 规则 |
| A5 | **Identity / View** | ~42 + view 族 | `clone/view/reshape/permute` | 多为元数据；或纯 copy tile |
| A6 | **Flatten-as-1D** | ~11 | `take/bucketize` | 忽略高维，按 1D 动态长度 |
| A7 | **Batched / FeatureBatched** | ~94 / ~19 | `nll_loss/batch_norm` | 前缀 batch 或后缀 feature 动态 |
| A8 | **Composite** | ~95 | `kl_div/isfinite` | 分解到 A1–A7，不单独映射指令 |
| A9 | **Factory** | ~90 | `empty/arange/randn` | 无输入 tensor；按运行时 size 填 |
| A10 | **Variadic** | ~14 | `cat/stack` | 输入个数/各维动态 |
| A11 | **Dynamic output shape** | ~15 + tag | `unique/nonzero/masked_select` | 输出长度依赖数据 → 两阶段 |
| A12 | **Sparse / Trivial** | ~40 / ~59 | sparse ops / `size` | 特殊路径或 host 侧 |

### 2.2 轴 B — 计算结构（面向 tile=128 实现）

| ID | 计算 Pattern | 典型 ATen | Tile 算法骨架 |
|---|---|---|---|
| B1 | **Elementwise unary/binary/ternary** | pointwise | map / zip_with + mask |
| B2 | **Broadcast elementwise** | `add` 带广播 | 每 lane 多维索引 → 输入偏移 |
| B3 | **Horizontal / dimensional reduce** | `sum(dim)` | tile 内 reduce + block 原子/树归约 |
| B4 | **Scan / prefix** | `cumsum` | tile scan + 跨 tile 前缀传递 |
| B5 | **Compare-select / 谓词** | `where/clamp/maximum` | mask 生成 + blend |
| B6 | **Gather / Index** | `index/gather/embedding` | 间接 load（gather 指令） |
| B7 | **Scatter / Atomic** | `index_add/scatter` | 间接 store / atomic |
| B8 | **Sort / TopK 局部** | `sort/topk` | tile 内网络 + 跨 tile 归并（重） |
| B9 | **Matmul-like** | `mm/bmm/addmm` | 128×K 外积/分块（专用） |
| B10 | **Convolution-like** | `conv1d/2d/3d` | im2col+B9 或专用滑窗 tile |
| B11 | **Normalize / Softmax 族** | `layer_norm/softmax` | 两/三遍 reduce + map |
| B12 | **RNG** | `rand/dropout` | 计数器 RNG 每 lane 独立 |
| B13 | **Memory / layout** | `copy/transpose/pad` | 连续 copy 或 gather-scatter 重排 |
| B14 | **Data-dependent compact** | `nonzero/unique` | 谓词 → ballot/scan → 写回 |

### 2.3 交叉矩阵：哪些 A×B 必须优先支持

|  | B1 Map | B2 Bcast | B3 Reduce | B4 Scan | B6/B7 Index | B9/B10 重计算 | B14 Compact |
|---|---|---|---|---|---|---|---|
| **A1 Pointwise** | ★必选 | ★必选 | — | — | — | — | — |
| **A2 Reduction** | 预处理 | 可选 | ★必选 | 少见 | — | — | — |
| **A3 Fixed** | 局部 | 局部 | 局部 | — | — | ★专用模板 | — |
| **A4/A7 N-D/Batch** | ★ | ★ | ★ | 可选 | ★ | 分解 | — |
| **A5 View/Copy** | — | — | — | — | — | — | copy 模板 |
| **A10 Cat** | — | — | — | — | — | — | 分段 copy |
| **A11 DynOut** | — | — | — | ★ | ★ | — | ★必选 |

★ = 全量动态 shape 下的 **第一优先** 通用内核。

### 2.4 覆盖策略：「少量 Pattern 模板」吃掉长尾

```text
Composite (A8) ──decompose──► A1/A2/A5/...
     │
     ▼
统一 Runtime Descriptor (TensorDesc{sizes,strides,dtype})
     │
     ├── Pattern PE (Pointwise Engine)     ← B1+B2
     ├── Pattern RE (Reduction Engine)     ← B3 (+B11 多遍)
     ├── Pattern SE (Scan/Compact Engine)  ← B4+B14
     ├── Pattern IE (Indexing Engine)      ← B6+B7
     ├── Pattern ME (Memory/Layout Engine) ← B13+A5+A10
     └── Pattern XE (Expert: GEMM/Conv/Attn)← B9+B10 手写/生成
```

经验比例（实现投入 vs 算子覆盖）：

| 引擎 | 估计覆盖 ATen 变体 | 对动态 shape 的完成度 |
|---|---|---|
| PE + RE + ME | ~55–65% | **可全动态** |
| + IE + SE | ~70–80% | 全动态（含二阶段） |
| + XE（GEMM/Conv/Attn/Norm 融合） | ~90%+ 性能关键路径 | 动态 batch/seq；算法 tile 固定 128 |

---

## 3. 全量动态 Shape 的易用性设计

### 3.1 统一运行时描述符

```cpp
struct TensorDesc {
  void*       data;
  int64_t     numel;          // 动态
  int         rank;           // 动态 0..MAX_RANK
  int64_t     sizes[MAX_RANK];
  int64_t     strides[MAX_RANK];
  DType       dtype;
};
```

编写者不碰静态 shape；只写 **lane 纯函数** 或 **带 desc 的索引函数**。

### 3.2 三种易用 API 层级

#### L0 — Scalar lambda（最高易用，覆盖 A1）

```cpp
// 用户只写标量语义；框架负责 tile=128、尾 mask、broadcast、dtype dispatch
pointwise(out, a, b, [](auto x, auto y) { return x * y + T(1); });
```

内部：

```cpp
for (int64_t base = gid * TILE; base < n; base += grid * TILE) {
  mask_t m = lane_id < (n - base);          // 动态尾块
  auto xa = load_bcast(a, base, m);         // 按尺寸广播
  auto ya = load_bcast(b, base, m);
  auto zo = map(m, xa, ya, op);
  store(out, base, m, zo);
}
```

#### L1 — Index lambda（覆盖跨步 / 非 element 对齐）

```cpp
transform_indexed(out, in, [](int64_t i, auto v) {
  // i 为逻辑线性下标，运行时从动态 sizes 解码多维坐标
  return v * v;
});
```

#### L2 — Tile 专家（覆盖 B9/B10/B11）

```cpp
tile_kernel(desc, [](TileContext ctx) {
  auto x = ctx.load_tile<float>();          // 128 lanes
  auto m = ctx.mask();                      // 动态 valid lanes
  auto s = reduce_sum(where(m, x, 0));      // tile 内
  ctx.store_tile(x / s);
});
```

### 3.3 动态 Shape 关键机制对照

| 机制 | 作用 | 对应 GPU |
|---|---|---|
| **Grid-stride + mask** | 任意 `numel`，无对齐要求 | 标准 CUDA grid-stride |
| **运行时 broadcast 偏移** | `sizes` 不一致时每 lane 算地址 | TensorIterator |
| **线性化 ↔ 多维坐标** | 任意 rank 的 reduce/transpose | 通用 ND 索引 |
| **两阶段 compact** | `nonzero`：先 count 再 write | thrust copy_if |
| **动态轴 reduce** | `dim` 运行时；inner/outer 分解 | CUB BlockReduce |
| **可选 shape 特化 JIT** | 热路径静态化 strides | 性能优化，非正确性必需 |

### 3.4 「全量」边界（诚实清单）

| 可全动态且易用 | 可全动态但难写 | 需放宽/二阶段 |
|---|---|---|
| pointwise、激活、逐点比较 | 任意轴 softmax/layer_norm | `unique` / `nonzero` 输出长度 |
| copy/contiguous/cat | 高维 transpose 融合 | data-dependent 控制流 |
| sum/mean 全维或单维 | sort 全局 | 稀疏变长索引 |
| gather/scatter 常规 | 动态 rank > MAX_RANK | host 同步取 shape |

`MAX_RANK`（建议 8 或 16）是工程折中，与 PyTorch 常见上限一致；超过则走慢路径。

---

## 4. 需要采用的指令分类（面向 tile=128）

将硬件/后端指令按 **算子 pattern 消费关系** 分为 8 类。每类给出：语义、C++26/`tile` 映射、主要服务的 B-pattern、动态 shape 注意点。

### 4.1 I1 算术与逻辑（Arithmetic / Logic）

| 子类 | 示例 | C++26 / tile API | 服务 |
|---|---|---|---|
| 二元算术 | add/sub/mul/div/fma | `simd` 运算符 / `fma` | B1/B2 |
| 比较 | eq/lt/gt | → `simd_mask` | B5 |
| 位运算 | and/or/xor/shift | `simd` 位运算 | B1、量化 |
| 类型转换 | cast、bf16↔f32 | `simd_cast` / 扩展 | 全 pattern |

**动态 shape**：与长度无关；只作用在有效 mask 的 lane。

### 4.2 I2 谓词与混合（Mask / Predication / Blend）

| 子类 | 示例 | 映射 | 服务 |
|---|---|---|---|
| 尾块 mask | `iota < remain` | `simd_mask` | **所有动态长度** |
| 条件写 | `where(m, a, b)` | `where` | B5、`clamp` |
| 谓词存储 | masked store | 扩展 / 后端 | 防写越界 |
| ballot / popcount | 有效 lane 计数 | 后端扩展 | B14 |

**动态 shape 关键指令**：没有 I2，就没有安全的全动态 `numel`。

### 4.3 I3 归约与扫描（Reduce / Scan）

| 子类 | 示例 | 映射 | 服务 |
|---|---|---|---|
| tile 水平归约 | sum/min/max/and | `reduce` | B3/B11 |
| 跨 tile 原子归约 | atomicAdd | 后端 | 全局 reduce |
| 前缀和 | inclusive/exclusive scan | 扩展库 | B4/B14 |
| arg-reduce | argmax | 成对 (val,idx) | `amax`+index |

**动态 shape**：reduce 轴长度运行时变化 → 必须支持「任意次 tile 迭代 + 中性元填无效 lane」。

### 4.4 I4 内存访问（Memory）

| 子类 | 示例 | 映射 | 服务 |
|---|---|---|---|
| 连续向量 load/store | unit-stride | `copy_from/to` | B1/B13 |
| 掩码 load/store | masked | 扩展 | 尾块 |
| 跨步 load/store | stride = S | 循环标量或 strided 指令 | 非 contiguous |
| 广播 load | 标量→ tile | `simd(value)` | B2 |
| 异步拷贝 / DMA | TMA-like | 后端 | 大块 ME/XE |

### 4.5 I5 间接访问（Gather / Scatter）

| 子类 | 示例 | 映射 | 服务 |
|---|---|---|---|
| gather | `x[idx[lane]]` | gather 指令或标量回退 | B6、embedding |
| scatter | `y[idx[lane]] = v` | scatter | B7 |
| atomic scatter | conflict 安全 | atomic | `index_add` |

**动态 shape**：索引范围运行时检查；冲突频率决定是否需要排序优化路径。

### 4.6 I6 通道重排（Permute / Shuffle / Broadcast lane）

| 子类 | 示例 | 服务 |
|---|---|---|
| lane shuffle / broadcast lane0 | tile 内通信 | B3 树归约、B4 scan |
| interleave / deinterleave | 转置碎片 | B13、`transpose` |
| compress / expand | 按 mask 压缩 | B14 |

C++26 标准 simd **弱于** ISPC/GPU 的 lane shuffle；实现全量动态 compact **必须** 补 I6 扩展或 lib 仿真。

### 4.7 I7 同步与协作（Sync）

| 子类 | 示例 | 服务 |
|---|---|---|
| tile 内隐式同步 | 单 simd 操作 | 默认 |
| block barrier | 多 tile 共享缓冲 | B3/B11/B9 |
| grid / device sync | 多 kernel 两阶段 | A11 |

### 4.8 I8 特殊函数与矩阵（Special / Tensor Core-like）

| 子类 | 示例 | 服务 |
|---|---|---|
| 初等函数 | exp/log/sin/rsqrt | 激活、softmax |
| RNG | philox 每 lane | B12 |
| 矩阵块 MMA | 128×tile 外积 | B9/B10 |
| 直方图 / sort 网络 | 局部 | B8 |

### 4.9 Pattern → 指令需求映射表

| Pattern | I1 | I2 | I3 | I4 | I5 | I6 | I7 | I8 |
|---|---|---|---|---|---|---|---|---|
| B1 Pointwise | ● | ● | | ● | | | | ○ |
| B2 Broadcast EW | ● | ● | | ● | | | | |
| B3 Reduce | ● | ● | ● | ● | | ○ | ○ | |
| B4 Scan | ● | ● | ● | ● | | ● | ○ | |
| B5 CmpSel | ● | ● | | ● | | | | |
| B6/B7 Index | ● | ● | | ○ | ● | | ○ | |
| B9/B10 GEMM/Conv | ● | ○ | ○ | ● | ○ | ○ | ● | ● |
| B11 Norm/Softmax | ● | ● | ● | ● | | ○ | ● | ● |
| B13 Layout | | ● | | ● | ○ | ● | | |
| B14 Compact | ● | ● | ● | ● | | ● | ● | |

● 必需　○ 强烈建议 / 性能路径

### 4.10 最小指令集建议（MVP → 完整）

**MVP（先打通全动态易用）**：I1 + I2 + I4(连续+mask) + I3(tile reduce)  
→ 可实现绝大多数 pointwise / 全维 reduce / contiguous copy。

**完整动态 shape**：MVP + I4 跨步 + I5 + I6(compress) + I3 跨 tile  
→ gather/scatter/nonzero/cumsum/任意轴 reduce。

**性能完备**：完整 + I7 block + I8 MMA/特殊函数  
→ matmul/conv/softmax 训练推理主路径。

---

## 5. 易用性评估

### 5.1 评分标准（1–5）

| 分 | 含义 |
|---|---|
| 5 | 用户写标量/一行 lambda；动态 shape 零心智负担 |
| 4 | 需选引擎 API（reduce dim 等），仍无手写 mask |
| 3 | 需理解 tile/mask 或二阶段；有样板可抄 |
| 2 | 需专家 tile kernel；动态维易踩坑 |
| 1 | 几乎必须专用内核生成器 / 图编译 |

### 5.2 分类评估表

| 算子类别（A×B） | 表达易用性 | 动态 Shape 正确性难度 | 性能可达性 | 综合 | 说明 |
|---|---|---|---|---|---|
| Pointwise 无广播 (A1×B1) | 5 | 1（易） | 5 | **优秀** | L0 API 即可 |
| Pointwise 广播 (A1×B2) | 5 | 2 | 4 | **优秀** | 框架藏 broadcast |
| 激活 / 逐点三元 (B5) | 5 | 1 | 5 | **优秀** | mask blend |
| 全维 / 简单维 reduce (A2×B3) | 4 | 2 | 4 | **良好** | 中性元+多 tile |
| Softmax / LayerNorm (B11) | 3 | 3 | 3–4 | **中等** | 多遍；数值稳定 |
| Contiguous copy / cat (B13) | 5 | 1 | 5 | **优秀** | ME |
| Transpose / permute 高维 | 3 | 3 | 2–3 | **中等** | 依赖 I6/跨步 |
| Gather/Embedding (B6) | 4 | 2 | 3 | **良好** | 需 I5 |
| Scatter/index_add (B7) | 3 | 3 | 2–3 | **中等** | 冲突与原子 |
| Cumsum (B4) | 3 | 3 | 3 | **中等** | 需 scan |
| Nonzero/Unique (A11×B14) | 3 | 4 | 2–3 | **偏难** | 二阶段+同步 |
| Matmul/BMM (B9) | 2 | 2（含动态 MNK） | 5（有 MMA） | **专家** | 不用裸 simd 手写 |
| Conv (B10) | 2 | 3 | 4–5 | **专家** | 专用模板 |
| Sort/TopK 全局 (B8) | 2 | 3 | 2–3 | **专家** | 库级算法 |
| Composite (A8) | 5（分解后） | 1 | — | **优秀** | 不直写 |

### 5.3 「全量动态 Shape 易用性」总评

| 目标 | 评估 |
|---|---|
| API 是否能让用户忘记 `TILE=128` | **能**（L0/L1）；L2 才暴露 |
| 是否要求 `numel % 128 == 0` | **否**（I2 尾 mask 为硬需求） |
| 任意 rank / 动态维 broadcast | **能**（运行时 desc + 线性索引） |
| 一次编写、多后端（CPU simd / GPU / NPU） | **中高**：L0/L1 可移植；I5/I6/I8 需后端能力探测 |
| 覆盖「写得出」vs「跑得快」 | 写得出：~80% 算子；跑得快：还需 XE 与融合 |
| 相对手写 CUDA / 手写 AVX | 点对点算子 **明显更易**；GEMM 级 **不易替代** cuBLAS/专用库 |

**综合结论**：

- 在 `tile_len=128` + GPU-like mask 模型下，对 PyTorch **主体算子（pointwise / reduction / memory）实现全量动态 shape 的易用性为「高」**。
- 对 **indexing / compact / scan** 为「中」，取决于 I5/I6 是否一等公民。
- 对 **GEMM/Conv/Attention** 不应期待 C++26 simd 裸写达到易用；应提供 **同 tile 约束下的专家模板**，对外仍是动态 `M,N,K`。

### 5.4 风险与缓解

| 风险 | 影响 | 缓解 |
|---|---|---|
| C++26 `where` 弱于真实谓词执行 | 分支算子性能差 | 后端 lowering 到原生 predication |
| 固定 128 与物理宽度不整除 | 多余拆分开销 | 逻辑 tile；后端再切 physical VL |
| 高 rank 索引计算重 | 广播点对点变慢 | 短路径：可合并维 / 运行时 collapse |
| data-dependent shape 与图编译 | export/trace 失败 | 保留 ATen `dynamic_output_shape` 语义；二阶段 host 可见 size |
| 过度统一导致 GEMM 退化 | 训练不可用 | Pattern XE 旁路，不走 PE |

---

## 6. 推荐落地路线（与易用性对齐）

| Phase | 交付 | 解锁的 PyTorch 类 | 易用性目标 |
|---|---|---|---|
| **P0** | `tile_t`/`mask_t` + L0 `pointwise` + 尾 mask | A1、A8 分解后的点对点 | 评分 5 |
| **P1** | 运行时 broadcast + strided load | 非连续 / 广播 A1 | 评分 5 |
| **P2** | Reduction Engine（含动态 dim） | A2、部分 B11 | 评分 4 |
| **P3** | Gather/Scatter + Compact/Scan | A4 indexing、A11 | 评分 3–4 |
| **P4** | Layout/transpose/cat | A5/A10 | 评分 4–5 |
| **P5** | Expert GEMM/Conv/Norm tile 模板 | A3/B9/B10/B11 | 对外动态，对内专家 |

---

## 7. 附录

### 7.1 与标准 / 生态的关系

| 项目 | 关系 |
|---|---|
| C++26 `std::datapar::simd` | Lane 级可移植原语 |
| P0350 `execution::simd` | 可作 host 侧算法策略；设备侧仍用 tile kernel |
| PyTorch TensorIterator | L0/L1 的语义对标 |
| ISPC / CUDA / Triton | 易用性上限参考；本设计用库+固定 TILE 逼近 |
| AscendC / NPU `kernel_operator_vec_*` | I1–I4 可直接映射；I5/I6 对齐 gather/scatter/brcb |

### 7.2 `tile_len=128` 的含义澄清

| 解释 | 何时采用 |
|---|---|
| 128 个 **元素**（与 dtype 无关） | 软件逻辑 tile（推荐默认） |
| 128 **字节** | 映射 cache line / UB 块时 |
| 128 个 **bit lanes** | 与 mask 寄存器讨论时需换算 |

本文默认：**128 个元素 lanes**；`float` tile = 512B 逻辑载荷，后端可再切。

### 7.3 参考

- P1928R15：`std::simd` 并入 C++26  
- cppreference：Data-parallel types (SIMD)  
- ezyang：*A brief taxonomy of PyTorch operators by shape behavior*（2020）  
- PyTorch `aten/src/ATen/native/tags.yaml`：`pointwise` / `reduction` / `dynamic_output_shape`  

---

*文档性质：设计分析；不绑定具体仓库实现。可与 `npu_arch_kernel_operator_feature_simplify.md` 对照：后者是 AscendC 多 arch 实现收敛，本文是上层「固定 tile + 动态 shape」算子编程模型。*
