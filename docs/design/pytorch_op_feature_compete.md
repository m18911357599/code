# PyTorch 算子特征分类、API 特征与竞品分析

> 范围：以 **PyTorch ATen / TensorIterator / native_functions** 为基准，对算子计算特征、API 形态、动态 shape / 非对齐 / padding 处理特征做分类，并与 **CUDA 手写、Triton、AscendC、SIMD-Tile(128B)** 做竞品对比（竞分）。  
> 相关文档：[SIMD-Tile × PyTorch Pattern](../../docs/design/cpp26_simd_tile128_pytorch_dynshape.md)（若已合入）、[AscendC npu_arch Feature](../../docs/design/npu_arch_kernel_operator_feature_simplify.md)（若已合入）。

---

## 0. 结论摘要

| 维度 | 结论 |
|---|---|
| **算子分类主轴** | 按 **数据依赖图 + 访存/并行模式** 分 8 大类：Elemwise / Broadcast / Reduce / Layout(Concat·Transpose·Copy) / Index / Matmul / Conv / DynOut；复合算子（Norm/Softmax/SDPA）视为组合或专用 |
| **API 特征主轴** | 按 **谁负责 shape/stride/尾块** 分 4 层：用户 Python API → Schema/Dispatch → Iterator/Plan → Backend Kernel；竞分关键在 **中间层是否吸收动态与非对齐** |
| **动态 shape** | PyTorch 热路径默认 **运行时 shape + 一次 plan**；编译路径（Inductor/Export）再加符号约束；与 AscendC「静态 tiling 优先」形成最大差异 |
| **非对齐 / 尾块** | PyTorch 用 **vectorized 整块 + scalar/unrolled 尾**；Triton 用 **mask tile**；AscendC 常受 **32B/512B 对齐与 Pad** 约束；SIMD-Tile 以 **128B 整块 + 至多一次尾块** 对齐 PyTorch 语义 |
| **Padding** | 框架层多为 **逻辑广播/对齐（不物化）**；硬件层（NPU UB/Cube、部分 Triton 后端）易出现 **隐式 pad → 带宽膨胀**；竞分要区分「语义 pad」与「硬件 pad」 |
| **选型建议** | 通用 EW/Bcast/Reduce：**跟 PyTorch TensorIterator 语义**；峰值 GEMM/Conv/Attn：**领域库**；NPU 落地：**上层 PyTorch 友好 + 下层 AscendC 能力显式** |

---

## 1. 分析框架

### 1.1 三个正交轴

```text
轴 A  计算/访存 Pattern   →  算子分类（§2）
轴 B  API / 责任边界      →  API 特征分类（§3）
轴 C  形状·对齐·填充策略  →  DynShape / Align / Pad（§4）
```

竞分（§5）在同一用例上对 A/B/C 打分，避免「拿手写 CUDA GEMM 比 AscendC Add」这类错位对比。

### 1.2 与官方 ATen tags 的关系

`aten/src/ATen/native/tags.yaml` 给出的是 **编译/导出语义标签**，不是完整计算分类：

| 官方 tag | 本文映射 |
|---|---|
| `pointwise` | Elemwise（含隐式 Broadcast） |
| `reduction` | Reduce |
| `dynamic_output_shape` | DynOut（输出 shape 依赖 **数据值**） |
| `data_dependent_output` | 非 Tensor 输出依赖数据（如 `item`） |
| `inplace` / `out` / `view_copy` / `inplace_view` | API 形态（§3），不是计算 Pattern |
| `needs_*_strides` / `flexible_layout` | 布局契约（§4.2） |

**缺口**：官方 tags **没有** 单独的 `broadcast` / `concat` / `transpose` / `conv` / `matmul`。Broadcast 被折叠进 `pointwise`；Layout/Matmul/Conv 靠 schema 与实现路径识别。本文补全工程分类。

---

## 2. 算子特征分析与分类

### 2.1 分类总表

| 类别 ID | 名称 | 定义（数据依赖） | 访存特征 | 并行特征 | PyTorch 代表 |
|---|---|---|---|---|---|
| **E** | Elemwise / Pointwise | 输出 `[i]` 只依赖各输入广播后的同址元素 | 流式、可 coalesced | 完全数据并行 | `add/mul/relu/clamp/where` |
| **B** | Broadcast（显式刻画） | 同 E，但输入 `stride_eff` 含 0 | 多输入步长不一致 | 输出主导迭代 | `x+bias`、`[B,1,K]+[B,H,K]` |
| **R** | Reduce | 多输入元素 → 少输出；含结合律聚合 | 写冲突 / 局部累加 | 树归约 / 分块归约 | `sum/mean/amax/argmax` |
| **L** | Layout | 不改（或仅拷贝）数值，改 shape/stride/拼接 | 搬移带宽 bound | 可按段并行 | `reshape/transpose/permute/cat/stack/contiguous/copy_` |
| **I** | Index / Indirect | 地址由索引张量决定 | gather/scatter、不规则 | 冲突敏感 | `index_select/gather/scatter/index_add` |
| **M** | Matmul / 线性代数 | `(i,k)×(k,j)` 收缩 | 分块复用、高算术密度 | MMA / 库 | `mm/bmm/addmm/linear` |
| **C** | Conv / 滑窗 | 局部邻域加权 | im2col 或专用滑窗 | 通道/空间并行 | `conv1d/2d/3d`、`avg_pool` |
| **D** | DynOut | 输出 shape/长度依赖 **输入数值** | 两阶段（count→write） | 前缀和/compact | `nonzero/unique/masked_select` |

复合算子（不单列新「第一类」，但实现常专用）：

| 复合 | 分解 | 为何常专用 |
|---|---|---|
| Softmax / LogSoftmax | R(max) → E(sub/exp) → R(sum) → E(div) | 数值稳定 + 融合带宽 |
| LayerNorm / RMSNorm | R → E | 多遍归约与广播融合 |
| SDPA / FlashAttn | 分块 M + Softmax + 在线合并 | IO 感知算法 |
| EmbeddingBag | I + 可选 R | 稀疏写冲突 |

### 2.2 子类细化

#### 2.2.1 Elemwise（E）

| 子类 | 特征 | 例 |
|---|---|---|
| E0 同形连续 | 全 operand unit-stride，同 numel | `a+b` contiguous |
| E1 谓词/选择 | 额外 mask 或 blend | `where`、`clamp` |
| E2 类型提升 | compute dtype ≠ 某输入 dtype | `int32 + float` |
| E3 就地 | 写回 `self` | `add_` |

**关键特征**：无跨元素依赖 → TensorIterator 可任意切块与重排维；尾块只需一次。

#### 2.2.2 Broadcast（B）

| 子类 | 特征 | 例 |
|---|---|---|
| B0 标量 / 0-stride | 一侧为 scalar 或 `numel=1` | `x + 1.0` |
| B1 可 collapse | 相邻维合并后降为 E0/B0 | `[B,1,K]+[B,H,K]` 合并 |
| B2 通用 | 任意 `stride_eff`，含中间维广播 | 任意 `expand` 后运算 |

**关键特征**：输出 shape = broadcast(inputs)；输入 **不物化 expand**（stride=0），除非后端强制 materialize。

#### 2.2.3 Reduce（R）

| 子类 | 特征 | 例 |
|---|---|---|
| R0 全维 | 输出标量或空 shape | `x.sum()` |
| R1 内维（连续轴） | 沿最快维归约 | `sum(-1)` |
| R2 外/中间维 | 需重排或跨步累加 | `sum(0)` |
| R3 短内维 | `inner * sizeof(T)` 小于向量宽 | 特征维=3 的 sum |
| R4 Arg-reduce | 携带 index | `argmax` |
| R5 扫描型 | 有序前缀依赖 | `cumsum`（半 reduce） |

**关键特征**：需 **中性元**（0、+inf…）填无效 lane；结合律决定可否乱序并行。

#### 2.2.4 Layout（L）

| 子类 | 特征 | 例 |
|---|---|---|
| L0 View / 元数据 | 不搬数据 | `view`、`transpose`（可 view 时） |
| L1 Materialize copy | 必须搬数据 | `contiguous`、`clone`、`permute` 打断时 |
| L2 Concat / Split | 多段沿轴拼接/切开 | `cat`、`split`、`chunk` |
| L3 Pack / Pad（语义） | 显式填充到目标 shape | `nn.utils.rnn.pad_sequence`、`F.pad` |

**关键特征**：`cat` = 多段独立 copy（每段各自尾块）；`transpose` 热路径常走 **分块转置**，边缘用 gather/scalar。

#### 2.2.5 Index（I）

| 子类 | 特征 | 例 |
|---|---|---|
| I0 Gather | 读间接 | `gather`、`index_select` |
| I1 Scatter | 写间接；冲突策略 | `scatter`、`index_add` |
| I2 Masked | 布尔压缩/填充 | `masked_fill`、`masked_select`(→D) |

#### 2.2.6 Matmul（M） / Conv（C） / DynOut（D）

| 类 | 子特征 | 对中间层含义 |
|---|---|---|
| M | 分块 (BM,BN,BK)、epilogue 融合 bias/act | **不宜** 走通用 TensorIterator map |
| C | stride/pad/dilation、groups、im2col vs 直接卷积 | pad 是 **算法参数**，不是向量尾块 |
| D | 输出长度未知 → 两阶段或上限缓冲 | 与「静态 shape 假设」冲突最大 |

### 2.3 特征维度矩阵（用于实现选型）

| 特征维度 | E | B | R | L | I | M | C | D |
|---|---|---|---|---|---|---|---|---|
| 输出 shape 由输入 shape 决定 | Y | Y | Y | Y | 部分 | Y | Y | **N（由数据）** |
| 可无序并行 | Y | Y | 部分 | Y | 写冲突时 N | 分块内有序 | 分块内有序 | 二阶段 |
| 适合通用向量 map | **Y** | **Y** | 部分 | copy 时 Y | N | N | N | N |
| 对对齐敏感度 | 中 | 中 | 高（短内维） | 低–中 | 高 | **很高** | **很高** | 中 |
| 典型瓶颈 | 带宽 | 带宽+地址 | 带宽/同步 | 带宽 | 延迟/冲突 | 算力 | 算力+带宽 | 同步+原子 |

### 2.4 Pattern 组合（真实模型里的算子）

| 组合 | 例 | 实现策略 |
|---|---|---|
| B→E | `Linear` bias、`x+γ` | BroadcastPlan → elementwise |
| R→B→E | LayerNorm 简化 | 两遍 R + E；或专用核 |
| L→E | channels_last 后 EW | 先 layout 契约，再 E |
| M→E | GEMM + bias + GELU | epilogue 融合 |
| I→R | embedding bag | 专用稀疏核 |
| Mask→D | `masked_select` | DynOut |

---

## 3. API 特征分析与分类

### 3.1 四层 API 栈（PyTorch）

```text
L3  用户 API          torch.add / Tensor.add / F.conv2d / nn.Linear
L2  Schema + Dispatch  native_functions.yaml → 按 device/dtype/layout 分发
L1  迭代与计划         TensorIterator / Structured kernels / meta
L0  Backend 核         CPU vec / CUDA / XPU / 私有 NPU / cuBLAS…
```

**竞分核心**：哪一层吃掉「动态 shape、广播、尾块、对齐」。PyTorch 的优势在 **L1 默认吃掉**；AscendC 更多暴露在 **L0**；Triton 把 L1 的一部分编译进 mask/tile。

### 3.2 按责任边界分类

| API 类 | 谁算 shape | 谁处理广播 | 谁处理尾块/对齐 | 代表 |
|---|---|---|---|---|
| **A1 声明式算子** | 框架 | 框架 | 框架 | `torch.add`、`torch.sum` |
| **A2 Out 变体** | 调用方预分配 out | 框架校验 | 框架 | `torch.add(a,b,out=c)` |
| **A3 Inplace** | 不变 | 受限广播 | 框架 | `a.add_(b)` |
| **A4 Structured / meta** | meta 函数 | schema | 不进核 | PT2 / Export |
| **A5 Iterator 核 API** | Iterator build | Iterator | `cpu_kernel_vec` 整块+尾 | ATen native |
| **A6 领域库 API** | 调用方/包装 | 包装层 | 库内部 tiling | `addmm`→GEMM、SDPA |
| **A7 可编程核 DSL** | 用户写 tile | 用户 mask | 用户/编译器 | Triton、手写 CUDA、AscendC |

### 3.3 按编程模型分类（跨产品）

| 模型 | 抽象粒度 | 动态 shape 写法 | 典型产品 |
|---|---|---|---|
| **标量语义 + 运行时 plan** | 每元素 lambda | 自然：任意 `n` | PyTorch TensorIterator、SIMD-Tile L0 |
| **线程网格** | thread/warp | `if (i < n)` | CUDA |
| **Tile + mask** | 程序实例 × BLOCK | `tl.load(..., mask=)` | Triton |
| **显式缓冲 + 指令 API** | LocalTensor / DataCopy / Pipe | 常需 tiling 规划 | AscendC |
| **逻辑向量量子** | 128B tile + 尾块 | 框架分流整块/尾块 | SIMD-Tile |

### 3.4 PyTorch 算子 API 形态特征

| 特征 | 说明 | 对后端含义 |
|---|---|---|
| **重载族** | 函数式 / 方法 / inplace / out | 同一计算核，多入口 |
| **类型提升** | 二元 op 的 dtype promote | compute dtype 决定向量 lane |
| **广播隐式** | 用户不写 expand | 后端必须支持 0-stride 或先 materialize |
| **dim 规范** | 负维、多维 reduce、keepdim | ReducePlan 规范化 |
| **布局宽容** | 多数 EW 接受非 contiguous | Iterator reorder/coalesce |
| **Factory + 计算分离** | `empty` 再 `copy_` / out | 便于内存池 |
| **复合模块** | `nn.Linear` = M + B + 可选 E | 融合机会在编译器 |

### 3.5 API 易用性相关的可度量特征

| 度量 | 含义 | PyTorch 倾向 |
|---|---|---|
| 用户可见对齐约束 | 是否要求 `n % VL == 0` | **无**（API 契约） |
| 尾块是否手写 | 用户是否写 epilogue | **否** |
| Broadcast 是否显式 | 是否先 `expand` | **否**（可隐式） |
| 动态 rank 支持 | 任意维数 | EW/Reduce：**是**（Iterator）；Triton 核：**难** |
| 专用算子入口 | GEMM 是否伪装成 EW | **否**（`addmm` 等） |

---

## 4. 动态 Shape、非对齐、Padding 处理特征

### 4.1 动态 Shape：三层含义（必须拆开）

| 层级 | 含义 | PyTorch 行为 | 竞品常见行为 |
|---|---|---|---|
| **S1 运行时可变长度** | 每次 forward `N/H/W` 不同 | Eager：每次 build Iterator；无重新 codegen | CUDA：grid 随 `N`；Triton：常要 constexpr tile + mask；AscendC：tiling 重算或走动态模板 |
| **S2 符号动态（编译）** | `SymInt` / 守卫 | PT2：符号化 + 守卫失败再编译 | Inductor/Triton 特化；导出需约束 |
| **S3 数据依赖动态** | 输出长度看数值 | `dynamic_output_shape`；两阶段或同步 | 多数 DSL 不友好，需 host 协作 |

**特征标签建议（实现侧）**：

```text
DynLen      : 长度运行时可知，不依赖元素值          → E/B/R/L/M/C 主体
DynRank     : 维数不固定                            → Iterator 强；DSL 弱
DynOutData  : 输出 shape 依赖数据                   → D 类
SymShape    : 编译期符号维                          → PT2 / Export
```

### 4.2 非对齐：三类「不对齐」

| 类型 | 定义 | PyTorch 处理 | 风险 |
|---|---|---|---|
| **U1 长度非向量整除** | `n % lanes != 0` | 整块向量核 + 标量/窄向量尾循环 | 尾块占比高时效率降 |
| **U2 地址未对齐** | ptr % align ≠ 0 | CPU：unaligned load 或先标量对齐到边界；GPU：多数自然宽 load 可容忍 | 部分 NPU 指令硬要求对齐 |
| **U3 步长非连续** | `stride != 1`（元素） | reorder + coalesce；否则 strided loop / 先 contiguous | 隐式 `contiguous()` 引入额外拷贝 |

TensorIterator 关键手段：

1. `compute_shape` → broadcast 后的计算 shape  
2. `compute_strides` → 字节步长  
3. `reorder_dimensions` → 快维优先  
4. `coalesce_dimensions` → 合并可合并维（降维到接近 1D）  
5. 向量核处理主循环；剩余走 scalar

### 4.3 Padding：语义 Pad vs 硬件 Pad

| 种类 | 谁引入 | 是否改变数值语义 | 例 |
|---|---|---|---|
| **P-Sem 语义填充** | 用户 / 算法 API | **是**（或明确忽略区） | `F.pad`、conv 的 padding、sequence pad |
| **P-Bcast 逻辑拉伸** | 广播规则 | 否（不物化） | size=1 维 stride=0 |
| **P-Vec 向量尾填充** | 向量化实现 | 否（写回时 mask 掉） | 尾块中性元；禁止泄漏到输出 |
| **P-HW 硬件对齐填充** | 后端/编译器 | 否，但 **带宽/算力膨胀** | Ascend UB 尾轴 pad 到 32B/512B；部分 Triton-Ascend 自动 pad |

**竞分要点**：PyTorch API **不把 P-HW 暴露给用户**；NPU 栈若在 lowering 中自动 P-HW，短尾轴（如 `..., 3`）会出现「逻辑很小、物理很大」的性能坑——需 transpose / 借轴 / 改 layout 规避。

### 4.4 各类算子的 Dyn / Align / Pad 处方

| 类别 | 动态 shape | 非对齐 | Padding |
|---|---|---|---|
| E | 每次按 `numel` 切块 | U1 尾块；U2 unaligned 快/慢路径 | 仅 P-Vec |
| B | 重算 `stride_eff` + collapse | 同 E；广播维 stride=0 | P-Bcast；忌过早 materialize |
| R | plan `{outer,reduce,inner}` | R3 短内维：多行打包或转置 | 尾 lane 中性元（P-Vec） |
| L-cat | 每段独立长度 | 每段自有尾块 | 段间无强制 pad |
| L-transpose | 任意 M×N | 分块；边缘块 | 块内可 P-Vec |
| I | 索引长度动态 | gather 难向量化 | 一般无 pad |
| M/C | 动态 MNK / NHW | 库内 tiling；要求 L1/L0 对齐 | Conv：**P-Sem**；库内 **P-HW** |
| D | **S3** | compact 尾块 | 上界缓冲 ≈ 软 pad |

### 4.5 推荐契约（对接自研后端 / SIMD-Tile）

```text
1. 对外 API：不要求 nbytes % 128 == 0，不要求用户写 mask
2. 对内 lowering：整块满向量量子 + 至多一次尾块（T-Mask / T-Split）
3. P-HW 不得泄漏为 Python 可见的数值改变
4. 短内维 Reduce / 短尾轴 EW：优先 layout 变换，再考虑硬件 pad
5. DynOut：显式两阶段 API，不伪装成 pointwise
6. GEMM/Conv：领域 API；只复用「分块边界 / epilogue / 尾块」约定
```

---

## 5. 竞品分析（竞分）

### 5.1 对比对象

| 对象 | 定位 |
|---|---|
| **PyTorch ATen + TensorIterator** | 行业参照：声明式 + 运行时 plan |
| **CUDA 手写** | 性能上限与控制力参照 |
| **Triton** | Tile DSL；Inductor 默认 codegen 之一 |
| **AscendC Basic API** | NPU 显式存储层级与向量/Cube API |
| **SIMD-Tile (128B)** | 以 128B 为逻辑量子的可移植编写模型 |

### 5.2 评分标准（1–5）

| 分项 | 含义 |
|---|---|
| C1 算子覆盖表达力 | 8 大类是否都能自然表达 |
| C2 API 心智负担 | 用户要懂多少硬件/尾块/对齐 |
| C3 动态 shape（S1） | 任意长度是否少特化 |
| C4 非对齐/尾块 | 框架是否默认正确且不太慢 |
| C5 Pad 可控性 | 能否避免隐式 P-HW 坑 |
| C6 性能表达力 | 打满硬件的能力（管道/MMA/多缓冲） |
| C7 与 PyTorch 语义对齐度 | 广播/类型提升/inplace 等 |

### 5.3 总评矩阵

| 分项 | PyTorch TI | CUDA 手写 | Triton | AscendC | SIMD-Tile |
|---|---|---|---|---|---|
| C1 表达力 | **5** | 5 | 3–4（DynOut/任意 stride 弱） | 4 | 4（M/C/D 走专用） |
| C2 心智负担 | **5** | 2 | 3–4 | 2–3 | **4–5** |
| C3 动态 shape | **5** | 4 | 3–4（tile/mask） | 3（tiling/对齐） | **4–5** |
| C4 尾块/对齐 | **5** | 3 | 4（mask） | 2–3（32B/512B） | **4–5** |
| C5 Pad 可控 | **5**（少隐式 P-HW） | 4 | 3–4（后端相关） | **2–3**（易自动 pad） | 4（禁止 API 级 pad） |
| C6 性能表达 | 3–4（靠后端） | **5** | 4（规则核强） | **5** | 3（通用核） |
| C7 PT 语义对齐 | **5** | 2（需自建） | 3 | 2–3 | **4–5**（目标对齐 TI） |

### 5.4 分 Pattern 竞分

| Pattern | 最易用 | 性能上限常见归属 | 备注 |
|---|---|---|---|
| E0/E1 Elemwise | PyTorch / SIMD-Tile | CUDA / AscendC 管道 | Triton 对任意 rank/stride 不友好 |
| B Broadcast | PyTorch | 同 E；小输入可先 materialize | AscendC 常显式扩或高阶 API |
| R1 内维 Reduce | PyTorch / CUB | CUB / AscendC Reduce | 短内维三者都要技巧 |
| R2/R3 | PyTorch plan | 手写/转置后 R1 | AscendC 短轴 pad 风险高 |
| L2 Concat | PyTorch | memcpy 带宽 | 各段独立尾块 |
| L1 Transpose | 库 / 专用 | 手写分块 | 边缘块决定复杂度 |
| I Gather/Scatter | 接近 | 硬件原子/冲突 | DSL 表达接近，优化难 |
| M/C | **库 API**（三方皆然） | cuBLAS / Cube | 不要用通用 EW 模型硬写 |
| D DynOut | PyTorch 两阶段 | 手写 compact | Triton/AscendC 均别扭 |
| 融合 Norm/SDPA | 专用 API | Flash/厂商库 | 竞分应比领域 API |

### 5.5 动态 Shape × 尾块：同一用例对照

用例：`out = a + b`，`a,b` 长度 `N` 运行时可变，且 `N % lanes != 0`。

| 栈 | 写法要点 | 尾块 |
|---|---|---|
| PyTorch | 用户无感知；Iterator 建 plan | `cpu_kernel_vec`：向量主循环 + 标量尾 |
| CUDA | `grid-stride` + `if (i < N)` | 线程谓词；或向量 load + 边界标量 |
| Triton | `for` 超 tile + `mask = offs < N` | **整块也可能带 mask**（除非特化末 tile） |
| AscendC | Tiling 算 `repeat`/`mask`；注意 UB 对齐 | 常配合 `SetMask` / 尾轴 pad |
| SIMD-Tile | `lanes=128/sizeof(T)`；满 128B 无谓词 | **至多一次** mask/epilogue |

用例：`sum(dim=-1)`，`K=3`（短内维）。

| 栈 | 风险 | 较优策略 |
|---|---|---|
| PyTorch | 向量利用率低 | 多行打包 / 换轴 |
| Triton | BLOCK 过大浪费 | 调 BLOCK、多行 |
| AscendC | **P-HW 把 K pad 到 32B/512B** | 借轴转置、避免尾轴为 3 |
| SIMD-Tile | 几乎全尾块 | `ShortInner` 打包进 128B |

### 5.6 API 特征竞分（责任落点）

```text
用户要写的「额外概念」越多，C2 越低：

PyTorch:   几乎 0（shape/broadcast/尾块全隐式）
SIMD-Tile: 0 于 L0 API；框架知 128B
Triton:    BLOCK、mask、constexpr 特化
CUDA:      grid/block、索引、同步、对齐
AscendC:   Pipe、LocalTensor、DataCopy、对齐、多级存储
```

### 5.7 对自研/NPU 落地的竞分结论

1. **对齐 PyTorch 的是「语义与中间层」**，不是对齐 CUDA 线程模型：优先具备 Iterator/Plan 级的 Broadcast·Reduce·尾块处理。  
2. **AscendC 的优势在 C6**，短板在 C3/C4/C5；应用层应提供 PyTorch 风格包装，把 P-HW 与 tiling 关在 lowering 内，并治理短尾轴。  
3. **Triton 适合规则 tile 核与融合**；不适合作为「任意 stride / 任意 rank」的 ATen 通用后端唯一方案。  
4. **SIMD-Tile** 适合作为 CPU/多后端的 **E/B/R/L 编写契约**，与 PyTorch Pattern 同构；M/C/D 保持领域 API。  
5. 竞分表上的「赢」应分场景：  
   - 通用算子正确性与动态 shape → **PyTorch / SIMD-Tile**  
   - 峰值矩阵与管道 → **CUDA 库 / AscendC Cube+Pipe**  
   - 快速融合实验 → **Triton**

---

## 6. 附录

### 6.1 术语

| 术语 | 含义 |
|---|---|
| Elemwise / Pointwise | 逐元素；ATen tag 名 `pointwise` |
| stride_eff | 广播后有效步长（广播维为 0） |
| lanes | 一向量步的元素数；SIMD-Tile 下为 `128/sizeof(T)` |
| P-HW | 硬件对齐引起的隐式 padding |
| DynOut | 输出 shape 依赖输入数据值 |

### 6.2 参考

- PyTorch `aten/src/ATen/native/tags.yaml`（`pointwise` / `reduction` / `dynamic_output_shape` 等）  
- PyTorch TensorIterator（`TensorIterator.h` / Wiki: How to use TensorIterator）  
- native_functions.yaml schema 与 Dispatch  
- Triton programming model（tile + mask）；Triton-Ascend 对齐/自动 pad 说明  
- AscendC Basic API（`kernel_operator_*`、DataCopy、向量 Mask、Cube）  
- 本仓库相关设计：SIMD-Tile 128B Pattern；npu_arch feature 简化  

### 6.3 文档维护

| 变更 | 动作 |
|---|---|
| 新增算子大类 | 更新 §2.1 表与 §2.3 矩阵 |
| 后端对齐策略变化 | 更新 §4.3 / §5.5 |
| 合入 SIMD-Tile / AscendC 文档 | 修正文首相对链接 |

---

*文档目标：给出可落地的算子/API/动态形状特征分类，以及与主流编程栈的竞品对照，供算子库与 NPU 中间层选型使用。*
