# PyTorch 内置底层算子：按核心功能分类、API 特征与同步特征

> 范围：以 ATen `native_functions.yaml` 及量化/稀疏/分布式等命名空间为对象，覆盖业界常称的 **约三千量级** 内置底层算子（见 §1.1）。  
> 本文按 **核心功能** 重新分类；Pattern 统一用 **两位数字编号**；每类给出典型算子与备注。  
> 对照维度：API 形态、动态 shape / 非对齐 / padding，以及 **同步算子特征提取（Pattern 29）**。

---

## 0. 结论摘要

| 维度 | 结论 |
|---|---|
| **分类主轴** | 按 **核心功能** 分为 **Pattern 01–29**；含 matmul 的标为 **Cube 类**（09/10/12.2/19.4） |
| **规模** | schema 约 **2500+**，唯一基名约 **1500+**；计入 inplace / out / 重载 / quantized·sparse·foreach / c10d 后常称 **~3000–3500** |
| **实现主路径** | **01–06、14–16** 多走 TensorIterator；**07–12、19、22–23** 多为固定维/领域核；**25–26** 含数据依赖动态输出；**29** 走运行时/通信后端 |
| **API** | 用户声明式 API → Schema/Dispatch → Plan/Iterator → Backend；中间层吸收动态与非对齐 |
| **Dyn / Align / Pad** | 语义 Pad（07）≠ 向量尾填充 ≠ 硬件对齐 Pad；短尾轴在 NPU 上易被隐式 pad |
| **同步** | 同步不是计算 Pattern，而是 **执行序约束**；需按阻塞域（Host/Device/Stream/Rank）与序关系（happens-before）提取特征 |
| **SIMT/SIMD** | 基线 Score_SIMT≈1.86、Score_SIMD≈1.25；**特殊指令后 Score_SIMD′≈1.86**（含 `vreduce`/`vsort`/`vmergesort` 等，见总表 s_SIMD′） |

---

## 1. 规模与编号约定

### 1.1 「约 3500 个」怎么理解

| 统计口径 | 数量级（main 线量级） | 说明 |
|---|---|---|
| `native_functions.yaml` 的 `- func:` schema | **~2500+** | 含 overload（如 `add.Tensor` / `add.Scalar`） |
| 唯一算子基名 | **~1500+** | 去掉 `.overload` 后的名字 |
| + inplace / out / 复合变体观感 | **~2000–3000** | 同一功能多入口 |
| + `quantized` / `sparse` / `_foreach_*` 等 | 常称 **~3000–3500** | 本文「三千量级」所指 |

分类对象是 **功能族**，不是把 3500 个 schema 逐条枚举；每类用 **典型算子 + 备注** 代表该族。

### 1.2 Pattern 编号规则

```text
Pattern NN   = 核心功能大类（两位数字，稳定编号）
NN.M         = 可选子类（一位小数编号）
```

示例：`07` = 填充类；`07.1` = 常数填充；`05.2` = Arg 规约。

后文 API / 动态 shape / 同步特征均引用这些数字编号。

### 1.3 与官方 ATen tags 的关系

| 官方 tag | 对应 Pattern |
|---|---|
| `pointwise` | 01、02、03（含隐式广播） |
| `reduction` | 05（及部分 11/22 内规约） |
| `dynamic_output_shape` | 26（及 25 中部分） |
| `inplace` / `out` / `view_copy` | API 形态（§3），不是功能类 |

官方 tags **没有** 填充 / 量化 / 卷积 / 矩阵乘等功能标签，需本文补全。

---

## 2. 按核心功能的 Pattern 分类总表

### 2.0 标记与打分约定

| 术语 | 含义 |
|---|---|
| **Cube 类** | 热路径含 **matmul / 等价矩阵收缩**（GEMM、卷积 Cube、SDPA 中 QKᵀ/PV、量化 Linear/Conv 等），峰值走 **Cube/MMA**，**不参与** SIMT/SIMD 打分 |
| **Vec 类** | 通用向量/线程并行可覆盖的主路径 |
| **Ctrl / Lib** | 控制面或专用库（打分记 **难=0**，或 Ctrl 直接剔出分母） |
| **规格数 N** | 以 ATen 唯一基名近似统计（`native_functions` 启发式归类；量级示意，非官方 census） |
| **亲 / 偏向 / 难** | 计分：`亲=2`；**偏向=2（对方模式记 1）**；`难=0` |

**加权分（剔除 Cube 类；Ctrl 28/29 不入分母）**：

```text
Score_mode = Σ_i ( N_i × s_mode(i) ) / Σ_i N_i

计分规则：
  双亲（两列均为「亲」）     → SIMT=2, SIMD=2
  偏 SIMT（「偏向」主 SIMT） → SIMT=2, SIMD=1（对方）
  偏 SIMD（「偏向」主 SIMD） → SIMT=1, SIMD=2（对方）
  某侧「难」                 → 该侧=0
  亲 + 对方                  → 亲侧=2, 对方侧=1

s_SIMD  = 基线（仅通用算术/load-store）
s_SIMD′ = 采用「特殊指令」列硬件原语之后的同分规则重估
Score_SIMD′ 同上公式，用 s_SIMD′
```

### 2.1 总表

> **s_SIMD′**：在采用「特殊指令」列所列硬件原语后的 SIMD 计分（同一套 亲=2 / 偏向主侧=2 / 对方=1 / 难=0）。基线 `s_SIMD` 假设仅有通用算术+load/store，无定长 reduce/scan/sort/compress/gather 等。

| Pattern | 核心功能 | 算子类 | N | SIMT档 | SIMD档 | s_SIMT | s_SIMD | 特殊指令（升档关键） | s_SIMD′ | 备注 |
|---|---|---|---|---|---|---|---|---|---|---|
| **01** | 逐元素算术 | Vec | 227 | 亲 | 亲 | 2 | 2 | `vloadm`/`vstorem` | **2** | 已亲；谓词尾块保档 |
| **02** | 比较/逻辑/选择 | Vec | 39 | 亲 | 亲 | 2 | 2 | `vcmp`/`vblend` | **2** | |
| **03** | 激活与非线性 | Vec | 79 | 亲 | 对方 | 2 | 1 | `vexp`/`vtanh` 等近似 | **2** | 有向量超越→升亲 |
| **04** | 广播与维扩展 | Vec | 35 | 偏向 | 对方 | 2 | 1 | `vsplat`/`vgather` | **2** | |
| **05** | 规约 | Vec | 47 | 偏向 | 对方 | 2 | 1 | **`vreduce_*`+masked** | **2** | 定长 reduce→亲 |
| **06** | 扫描与累积 | Vec | 12 | 偏向 | 难 | 2 | 0 | **`vscan`/`vprefix`** | **2** | 难→亲 |
| **07** | 填充 | Vec | 29 | 偏向 | 对方 | 2 | 1 | `vsplat`/`vgather` | **2** | |
| **08** | 池化 | Vec | 42 | 偏向 | 对方 | 2 | 1 | `vreduce`/`vmax` | **2** | |
| **09** | 卷积 | **Cube** | 38 | — | — | — | — | Cube/MMA | — | 剔出打分 |
| **10** | 矩阵乘/线性代数 | **Cube** | 45 | — | — | — | — | Cube/MMA | — | 剔出打分 |
| **11** | 归一化 | Vec | 32 | 偏向 | 对方 | 2 | 1 | **`vreduce_add`** | **2** | |
| **12.1** | Softmax | Vec | 16 | 偏向 | 对方 | 2 | 1 | **`vreduce_max`+`add`** | **2** | |
| **12.2/12.3** | SDPA/MHA | **Cube** | 21 | — | — | — | — | Cube+epilogue | — | 剔出打分 |
| **13** | 索引/散射/Embedding | Vec | 55 | 偏向 | 难 | 2 | 0 | **`vgather`/`vscatter`** | **2** | 难→亲 |
| **14** | 布局变换 | Vec | 85 | 对方 | 偏向 | 1 | 2 | `vtranspose`/`vshuf` | **2** | 已偏 SIMD |
| **15** | 拼接与分割 | Vec | 26 | 亲 | 亲 | 2 | 2 | `vload`/`vstore` | **2** | |
| **16** | 拷贝与类型转换 | Vec | 26 | 亲 | 亲 | 2 | 2 | `vload`/`vcvt` | **2** | |
| **17** | 工厂与创建 | Vec | 33 | 亲 | 亲 | 2 | 2 | `vsplat` | **2** | |
| **18** | 随机与 Dropout | Vec | 34 | 偏向 | 对方 | 2 | 1 | **`vrng`** | **2** | |
| **19** QDQ | 量化变换 | Vec | 27 | 亲 | 亲 | 2 | 2 | `vcvt` | **2** | |
| **19.4** | 量化 Conv/Linear | **Cube** | ~1+ | — | — | — | — | 整数 Cube | — | 剔出打分 |
| **20** | 稀疏（非 MM） | Vec | 43 | 偏向 | 难 | 2 | 0 | `vgather` | **1** | 仍不规则→对方 |
| **21** | FFT / 信号 | Lib | 31 | 难 | 难 | 0 | 0 | 专用 FFT 库 | **0** | 不走通用 SIMD |
| **22** | 损失函数 | Vec | 35 | 偏向 | 对方 | 2 | 1 | `vreduce` | **2** | |
| **23** | 上采样与插值 | Vec | 39 | 偏向 | 难 | 2 | 0 | `vgather` | **1** | 邻域仍别扭→对方 |
| **24** | Nested / Jagged | Vec | 14 | 偏向 | 对方 | 2 | 1 | 段内 `vload` | **2** | |
| **25** | 排序 / TopK / Unique | Vec | 10 | 偏向 | 难 | 2 | 0 | **`vsort`/`vmergesort`** | **2** | **难→亲**（见下） |
| **26** | 数据依赖动态输出 | Vec | 6 | 偏向 | 难 | 2 | 0 | **`vcompress`+`vprefix`** | **2** | 难→亲 |
| **27** | 特殊函数 | Vec | 37 | 偏向 | 对方 | 2 | 1 | 向量特殊函数近似 | **2** | |
| **28** | 元信息 / 控制 | Ctrl | 23 | — | — | — | — | — | — | 不入分母 |
| **29** | 同步 / 序约束 | Ctrl | 4 | — | — | — | — | — | — | 不入分母 |

Cube 类合计 **N_cube ≈ 105**；不参与 Score。

**Pattern 25 升档说明**：基线手写 bitonic/`vcmp`+`vblend` 网络 → SIMD **难(0)**；若 ISA 提供定长 **`vsort`（单向量内排序）** 与跨向量 **`vmergesort`/`vmerge_odd_even`（归并网络积木）**，则 `sort`/`topk`/`argsort` 可拼装为库级原语，SIMD′=**亲(2)**。原型见 §8.4。

### 2.2 剔除 Cube 后的 SIMT / SIMD 打分

参与集合：**Vec + Lib**（不含 Cube、Ctrl），`Σ N ≈ 1059`。

| 模式 | Σ(N·s) | Score=Σ(N·s)/ΣN | /满分2 | 解读 |
|---|---|---|---|---|
| **SIMT** | 1971 | **1.86** | **93%** | 非 Cube 整体强烈亲 SIMT |
| **SIMD（基线）** | 1326 | **1.25** | **63%** | 无特殊原语；gather/sort/scan/reduce 拖累 |
| **SIMD′（特殊指令后）** | 1974 | **1.86** | **93%** | 与 SIMT 持平；升档来自 reduce/scan/sort/gather/compress 等 |

```text
ΔScore_SIMD = Score_SIMD′ − Score_SIMD ≈ 1.86 − 1.25 = +0.61
```

升档贡献最大的规格块（`s: 0/1 → 2`）：**05/06/11/12.1/13/18/22/25/26/03/04/07/08/27** 等；**20/23** 仅抬到对方(1)；**21** 仍为 0。

### 2.3 建议（已剔除 Cube）

1. **默认仍可用 SIMT**（基线 Score 1.86）；若目标 ISA **承诺 §2.1 特殊指令集**，则 SIMD′≈SIMT，可 **SIMD 优先实现非 Cube**。  
2. **基线 SIMD（1.25）**：先覆盖已是 s=2 的双亲族 `01/02/15–17/19QDQ` 与 `14`。  
3. **指令投资优先级（按抬升 ΣN·Δs）**：  
   - P0：`vreduce_*`(+masked) → 05/08/11/12.1/22  
   - P0：`vgather`/`vscatter` → 13（及 04/07/23）  
   - P1：`vscan`/`vcompress` → 06/26  
   - P1：**`vsort`/`vmergesort`** → **25**  
   - P2：`vrng`、向量超越/特殊函数 → 18/03/27  
4. **Cube** 仍独立：`09/10/12.2/19.4`。  
5. **度量口径**：对外报告 SIMD 易用性时须声明是 **基线** 还是 **SIMD′（指令优化后）**，避免混比。

复合模块（`nn.Linear`、`nn.MultiheadAttention`）= **Cube + Vec epilogue**，编号仍落在 09/10/12 组合，不单开。

---

## 3. 各类详解：子类、典型算子、备注、SIMT/SIMD

### 3.0 列约定：SIMT / SIMD 易用性与 SIMD 指令特征

详表在原有列之外增加三列：

| 列名 | 含义 |
|---|---|
| **SIMT 易用** | 一线程一元素（CUDA grid-stride / warp 协作）写出正确实现的难度；**高=易写** |
| **SIMD 易用** | 一指令多 lane / tile（AVX·NEON·RVV·AscendC Vec·SIMD-Tile）写出正确实现的难度；**高=易写**；**随硬件原语有无可升降档** |
| **SIMD 典型指令特征** | 热路径所需可移植向量指令族（含硬件原语需求）；原型见 **§8.4** |

易用性取值：`高` / `中` / `低` / `N/A`（可带简短括注）。

#### 3.0.1 硬件指令支持 → 抬升 SIMD 易用性

SIMD 易用性取决于算法 **与** ISA 原语。缺原语时需手写 shuffle 树 / 标量收尾，易用性降 1～2 档。

| 硬件能力 | 主要抬升 Pattern | 缺省代价 | 易用性（有 → 无） |
|---|---|---|---|
| **定长水平规约** `vreduce_*`（固定 VL→标量/偏量） | 05、11、12.1、08、22 | 蝶式 `vhadd`+多轮 `vshuf` | 中/高 → 低 |
| **掩码规约** `vreduce_*_masked` | 05 尾块、短内维 | 先 blend 中性元 | 中 → 低 |
| **定长前缀** `vscan`/`vprefix` | 06、26 | 多轮扫描网络 | 中 → 低 |
| **compress/expand** | 13.3、26 | 前缀和+scatter | 中 → 低 |
| **定长排序/归并** `vsort`/`vmergesort` | **25** | bitonic 手写网络 | **难 → 亲** |
| **gather/scatter** | 04、07、13、23 | 标量间接 | 中 → 低 |
| **谓词访存** `vloadm`/`vstorem` | 01–03 尾块 | 标量 epilogue | 高 → 中 |
| **向量 RNG** `vrng` | 18 | 标量填 lane | 中 → 低 |
| **Cube/MMA** | 09/10/12.2 | 禁止通用 SIMD 冒充 | （Cube，不入 SIMD 分） |

**约定**：各 Pattern「备注」写清 **硬件依赖**；「SIMD 典型指令特征」列写助记，**完整原型见 §8.4**。

**SIMD 指令助记**：

| 助记 | 特征含义 | 硬件要点 |
|---|---|---|
| `vload`/`vstore` | 连续向量读写（含 unaligned） | 对齐快路径可选 |
| `vloadm`/`vstorem` | mask/谓词读写（尾块） | 谓词/mask 寄存器 |
| `vadd`/`vmul`/`vfma`/`vdiv`… | 逐 lane 算术 | 标准 |
| `vcmp`/`vblend`/`vwhere` | 比较与选择 | |
| `vsplat`/`vbroadcast` | 标量扩到全宽 | |
| `vreduce_add/max/min/...` | **定长**水平规约（VL→标量） | **关键抬升 05/11/12** |
| `vreduce_*_masked` | 忽略无效 lane 的规约 | 尾块/短内维 |
| `vhadd`/`vhmax` | 成对水平加（无整宽 reduce 时的积木） | 可软件拼 `vreduce` |
| `vgather`/`vscatter` | 间接读写 | |
| `vshuf`/`vtranspose` | 重排 / 分块转置 | |
| `vcvt` | dtype 转换 | 饱和/舍入 |
| `vprefix`/`vscan` | **定长**有序前缀 | 抬升 06/26 |
| `vcompress`/`vexpand` | 按 mask 压缩/展开 | 抬升 26 |
| `vsort`/`vmergesort` | 定长向量排序 / 归并网络积木 | **抬升 25** |
| `vatomic_*` | 地址原子 | scatter 冲突 |
| `vrng` | 向量随机 | |
| `MMA/Cube` | Cube 类矩阵加速 | 不计入 §2 SIMD 分 |

---

### Pattern 01 — 逐元素算术

| 子类 | 含义 | 典型算子 | 备注 | SIMT 易用 | SIMD 易用 | SIMD 典型指令特征 |
|---|---|---|---|---|---|---|
| **01.1** 二元算术 | `+/-/×/÷` 等 | `add`、`sub`、`mul`、`div`、`remainder`、`pow`、`atan2`、`hypot` | 默认支持广播与类型提升；TensorIterator 主力 | 高（`i<n`） | 高（整块+尾 mask） | `vload/vstore`、`vloadm/vstorem`、`vadd`/`vmul`/`vdiv`/`vfma`、`vcvt`（提升时） |
| **01.2** 一元算术 | 逐点变换 | `neg`、`abs`、`reciprocal`、`sqrt`、`rsqrt`、`exp`、`log`、`sin`/`cos` | 与 27 有重叠；实现仍按 pointwise | 高 | 高～中（超越函数看 ISA 是否有向量 `vexp`/`vlog`/`vsin`，否则标量近似） | 同 01.1 + `vabs`/`vsqrt`/`vexp`… |
| **01.3** 三元组合 | 乘加类 | `addcmul`、`addcdiv`、`lerp` | 易融合进 10 的 epilogue | 高 | 高 | `vfma`、`vload×3`、`vblend`（lerp） |
| **01.4** Foreach 变体 | 列表逐元素 | `_foreach_add`、`_foreach_mul`、… | 优化器步进；并行提交多 tensor | 高（多 stream/多核） | 高（每 tensor 同 01.1） | 同 01.1；调度非 SIMD |

**实现备注**：热路径 = 整块向量 + 尾块；**有 `vloadm`/`vstorem` 则尾块不降到标量**（SIMD 保持高）。非 contiguous 先 reorder/coalesce。指令原型 §8.4。

---

### Pattern 02 — 比较 / 逻辑 / 选择

| 子类 | 典型算子 | 备注 | SIMT 易用 | SIMD 易用 | SIMD 典型指令特征 |
|---|---|---|---|---|---|
| **02.1** 比较 | `eq`、`ne`、`lt`、`le`、`gt`、`ge`、`isnan`、`isinf`、`isfinite` | 输出常为 bool | 高 | 高 | `vcmp`、`vstore`（mask→bool 打包视后端） |
| **02.2** 位/逻辑 | `bitwise_and`/`or`/`xor`、`logical_and`、`__lshift__` | 整数/bool 路径 | 高 | 高 | `vand`/`vor`/`vxor`/`vshl`/`vshr` |
| **02.3** 选择 | `where`、`clamp`、`clamp_min`/`max`、`nan_to_num`、`maximum`、`minimum` | `where` = 谓词 blend；尾块需 mask | 高 | 高 | `vblend`/`vwhere`、`vmin`/`vmax`、`vcmp` |

---

### Pattern 03 — 激活与非线性

| 子类 | 典型算子 | 备注 | SIMT 易用 | SIMD 易用 | SIMD 典型指令特征 |
|---|---|---|---|---|---|
| **03.1** 经典激活 | `relu`、`leaky_relu`、`elu`、`selu`、`celu`、`threshold` | 常与 09/10 融合（Conv-ReLU） | 高 | 高 | `vmax`/`vblend`、`vfma`（leaky）；可作 10 epilogue |
| **03.2** 现代激活 | `gelu`、`silu`/`swish`、`mish`、`hardswish`、`hardsigmoid`、`hardtanh` | Transformer / CNN 高频 | 高 | 中（多项式/近似） | `vmul`/`vfma`、`vtanh`/`vexp` 近似、`vmin`/`vmax` |
| **03.3** 饱和/收缩 | `sigmoid`、`tanh`、`softplus`、`softshrink`、`hardshrink`、`glu` | Softmax 前常用 sigmoid/tanh | 高 | 中 | `vexp`/`vtanh` 近似、`vdiv`、`vblend` |
| **03.4** 带参数 | `prelu`、`rrelu` | 额外权重；非纯 01 | 高 | 高～中（通道广播） | `vbroadcast` + `vblend`/`vmul` |

---

### Pattern 04 — 广播与维扩展

| 子类 | 典型算子 | 备注 | SIMT 易用 | SIMD 易用 | SIMD 典型指令特征 |
|---|---|---|---|---|---|
| **04.1** 逻辑广播 | `expand`、`expand_as`、`broadcast_to`、`broadcast_tensors`、`broadcast_shapes` | **默认不物化**；stride=0 | 高（按输出下标算源址） | 中（0-stride 用 `vsplat`/`vbroadcast`，通用广播要 `vgather`） | `vsplat`、`vbroadcast`、必要时 `vgather` |
| **04.2** 重复物化 | `repeat`、`tile`、`repeat_interleave` | 真正拷贝；与 04.1 成本不同 | 高 | 中 | `vload`/`vstore` 循环；`repeat_interleave` 常要 `vgather` |
| **04.3** 隐式广播 | （无独立 API） | 发生在 01/02/03 的 TI build 中 | 见组合 | 见组合 | 同 04.1，并入 01 的地址生成 |

**备注**：实现选型时常把 04 从 01 拆出——后端是否支持 0-stride 决定要不要先 materialize。

---

### Pattern 05 — 规约

| 子类 | 典型算子 | 备注（含硬件依赖） | SIMT 易用 | SIMD 易用 | SIMD 典型指令特征 |
|---|---|---|---|---|---|
| **05.1** 数值规约 | `sum`、`prod`、`mean`、`nansum`、`nanmean`、`norm`、`linalg.vector_norm`、`logsumexp`、`count_nonzero` | keepdim/dim；**若 ISA 提供定长 `vreduce_add/max/...`，SIMD 升至高**；否则手写水平树 | 中（warp/block reduce） | **高（有定长 reduce）/ 中～低（无）** | `vload`/`vloadm`、`vadd`/`vmul`、**`vreduce_*` / `vreduce_*_masked`**（§8.4）；缺则 `vhadd`+`vshuf` |
| **05.2** Arg 规约 | `argmax`、`argmin`、`amax`、`amin`、`aminmax` | 需 **值+index 对规约**；硬件若有 `vreduce_max_with_index` 则升档 | 中 | 中（有 index-reduce）/ 低（无） | `vreduce_max_arg`/`vmin_arg` 或 `vmax`+idx 伴随、`vblend`；尾 lane 禁无效 idx |
| **05.3** 逻辑规约 | `all`、`any` | bool 中性元；`vreduce_and/or` 定长则升档 | 中 | 高（有）/ 中（无） | `vcmp`、`vreduce_and`/`vreduce_or`（或 `vand`/`vor` 树） |
| **05.4** 统计规约 | `std`、`var`、`median`、`quantile`、`mode` | 两遍 `vreduce`；quantile 仍近排序 | 中～低 | 低 | 两遍 `vreduce_add`；或走 25 |
| **05.5** 全局标量 | `trace`、无 dim `sum()` | 同 05.1 | 中 | 同 05.1 | 同 05.1；对角可 `vgather` |

**子特征备注**：

- **05.a 内维规约**：连续 load + **定长 `vreduce`** → SIMD 易用性最高  
- **05.b 外/中维**：先 `vtranspose` 或跨步累加；无 transpose 则偏 SIMT  
- **05.c 短内维**：多行打包填满 VL 后再 `vreduce`；依赖 **`vreduce_*_masked`** 或中性元  
- **硬件建议**：优先实现 **固定 VL 的 `vreduce_{add,max,min,and,or}` + masked 变体**（原型 §8.4），可将 05 的 SIMD 档从「对方/中」抬向「亲/高」 

---

### Pattern 06 — 扫描与累积

| 典型算子 | 备注 | SIMT 易用 | SIMD 易用 | SIMD 典型指令特征 |
|---|---|---|---|---|
| `cumsum`、`cumprod`、`cummax`、`cummin`、`logcumsumexp` | 有序前缀；**定长 `vscan`/`vprefix` 硬件可把 SIMD 从低抬到中**；缺则多轮 `vshuf`（§8.4） | 中（Hillis-Steele/Blelloch） | **中（有 vscan）/ 低（无）** | **`vscan_add`/`vscan_max`**、`vshuf`、`vload`/`vstore` |

---

### Pattern 07 — 填充（Pad）

| 子类 | 典型算子 | 备注 | SIMT 易用 | SIMD 易用 | SIMD 典型指令特征 |
|---|---|---|---|---|---|
| **07.1** 常值填充 | `constant_pad_nd`、`F.pad(..., mode='constant')`、`zero_`、`fill_`、`masked_fill` | **语义 Pad**：改变有效区域外的值 | 高 | 高～中（边界分支） | `vsplat`+`vstore`；`masked_fill` 用 `vblend`/`vstorem` |
| **07.2** 边界反射/复制 | `reflection_pad1d/2d/3d`、`replication_pad*`、`circular_pad*` | 卷积/分割任务常用 | 高 | 中～低（地址折叠） | 常 `vgather`；或分段 `vload` |
| **07.3** 序列填充 | `nn.utils.rnn.pad_sequence`、`pad_packed_sequence`、`_pad_packed_sequence` | 与 24 Nested 交互 | 高 | 中（变长段） | 段内 `vload`/`vstore`/`vsplat` |
| **07.4** 矩阵三角填充 | `tril`、`triu`、`tril_`、`triu_` | 按三角掩码写 | 高 | 中 | lane 坐标生成 mask + `vblend` |

**重要区分（实现与对齐常用）**：

| 名称 | 是否改数值语义 | 例 |
|---|---|---|
| 语义 Pad（本 Pattern） | 是 | `F.pad`、conv 的 padding 参数 |
| 广播拉伸 | 否（逻辑） | 04 + 01 |
| 向量尾填充 | 否（写回 mask） | 实现细节 |
| **硬件对齐 Pad** | 否，但耗带宽 | NPU UB 尾轴 pad 到 32B/512B |

---

### Pattern 08 — 池化

| 子类 | 典型算子 | 备注 | SIMT 易用 | SIMD 易用 | SIMD 典型指令特征 |
|---|---|---|---|---|---|
| **08.1** Max/Avg/Lp | `max_pool2d`、`avg_pool2d`、`lp_pool2d`、`*_pool1d/3d` | 固定 1–3D；含 indices 的 max_pool | 高～中（窗内循环） | 中（沿 W 向量化或窗内 `vmax`） | `vload`、`vmax`/`vadd`、`vreduce_*`；带 indices 时存 arg |
| **08.2** Adaptive | `adaptive_avg_pool2d`、`adaptive_max_pool2d` | 输出尺寸指定；实现与普通池化不同 | 中 | 中～低（窗边界不规则） | 同 08.1 + 动态窗索引 |
| **08.3** 反池化 | `max_unpool2d` | 依赖 08.1 的 indices | 中 | 低 | `vscatter` |
| **08.4** Fractional | `fractional_max_pool2d` | 随机/分数步长 | 中 | 低 | 不规则窗 + `vmax` |

---

### Pattern 09 — 卷积（**Cube 类**）

> 热路径为 **Cube/MMA 矩阵收缩**（或 im2col→GEMM），**含 matmul 等价计算**；§2 打分已剔除。

| 子类 | 典型算子 | 备注 | SIMT 易用 | SIMD 易用 | SIMD/Cube 指令特征 |
|---|---|---|---|---|---|
| **09.1** 标准卷积 | `convolution`、`conv1d/2d/3d`、`_convolution` | 统一入口；padding 为算法参数 | 中 | 低 | **Cube/MMA**；后备 `vfma` 滑窗 / im2col |
| **09.2** 转置/深度可分 | `conv_transpose*`、`_conv_depthwise2d` | 与 23 不同 | 中 | 低～中 | 主路径 **Cube**；depthwise 可 `vfma`+`vload` |
| **09.3** 后端专用 | `cudnn_convolution`、`miopen_convolution`、`slow_conv*` | 分发到库 | N/A（库） | N/A（库） | **Cube/库** |
| **09.4** im2col | `im2col`、`col2im` | 折叠到 10（Cube） | 高 | 中 | `vload`/`vstore`/`vgather`；`col2im`：`vscatter` |

**备注**：Cube 类；勿用通用 01 模型硬写。

---

### Pattern 10 — 矩阵乘与线性代数（**Cube 类**）

> **matmul / GEMM 本体**；峰值 **Cube/MMA**。§2 打分已剔除。

| 子类 | 典型算子 | 备注 | SIMT 易用 | SIMD 易用 | SIMD/Cube 指令特征 |
|---|---|---|---|---|---|
| **10.1** GEMM 族 | `mm`、`bmm`、`addmm`、`baddbmm`、`addbmm`、`matmul`、`tensordot`、`einsum` | 算力主力；epilogue 可融 01/03 | 低 | 低 | **Cube/MMA**；epilogue `vfma`/`vadd` |
| **10.2** 向量积 | `dot`、`vdot`、`ger`、`inner`、`outer`、`addr` | 小规模构建块 | 中 | 高～中 | 可 Vec：`vfma`+`vreduce_add`；`vsplat`×`vmul` |
| **10.3** 分解/求解 | `linalg.svd`、`linalg.qr`、`linalg.cholesky`、`linalg.solve`、`triangular_solve`、`lu_*` | 数值库 | 低 | 低 | 库（内部或调 Cube） |
| **10.4** 量化/低比特 GEMM | `_int_mm`、`_dyn_quant_matmul_4bit`、`_weight_int4pack_mm` | 与 19.4 交界 | 低 | 低～中 | **整数 Cube/MMA**、`vcvt`、scale `vmul` |
| **10.5** 分组/稀疏 MM | `_grouped_mm`、`_cslt_sparse_mm`、`_sparse_semi_structured_linear` | 与 20 交界 | 中～低 | 低 | **Cube** 或 `vgather` |

---

### Pattern 11 — 归一化

| 典型算子 | 备注 | SIMT 易用 | SIMD 易用 | SIMD 典型指令特征 |
|---|---|---|---|---|
| `native_batch_norm`、`batch_norm`、`native_layer_norm`、`layer_norm`、`native_group_norm`、`group_norm`、`instance_norm`、`rms_norm`、`normalize` | 内部=05+01；**定长 `vreduce_add` 是 SIMD 易用关键**（有则中→偏高） | 中（两遍/Welford） | **中～高（有 vreduce）/ 低（无）** | **`vreduce_add`**、`vsplat`、`vrsqrt`/`vdiv`、`vfma`、`vload`（§8.4） |

---

### Pattern 12 — Softmax / Attention

| 子类 | 典型算子 | 备注 | SIMT 易用 | SIMD 易用 | SIMD/Cube 指令特征 |
|---|---|---|---|---|---|
| **12.1** Softmax | `softmax`、`log_softmax`、`_softmax`、`_safe_softmax` | Vec；依赖 **定长 `vreduce_max`+`vreduce_add`**，有则 SIMD 升档 | 中 | **中～高（有双 reduce）/ 低（无）** | **`vreduce_max`/`vreduce_add`**、`vsub`、`vexp`、`vdiv`（§8.4） |
| **12.2** SDPA / Flash（**Cube**） | `scaled_dot_product_attention`、`_flash_attention_*`、`_efficient_attention_*`、`_cudnn_attention_*` | **内含 matmul**（QKᵀ/PV）；§2 已剔除 | 低 | 低～中 | **Cube/MMA** + 12.1 epilogue |
| **12.3** MHA 包装（**Cube**） | `_native_multi_head_attention` | 组合 10+12+13；含 matmul | 低 | 低 | **Cube** + Vec epilogue |

---

### Pattern 13 — 索引 / 散射 / Embedding

| 子类 | 典型算子 | 备注 | SIMT 易用 | SIMD 易用 | SIMD 典型指令特征 |
|---|---|---|---|---|---|
| **13.1** Gather | `gather`、`index_select`、`take`、`take_along_dim`、`index` | 读间接；难向量化 | 高 | 中～低 | `vgather`；连续 index 可退化 `vload` |
| **13.2** Scatter | `scatter`、`scatter_add`、`scatter_reduce`、`index_add`、`index_put`、`put_` | 写冲突；需原子或确定性策略 | 中（原子） | 低 | `vscatter`、`vatomic_add`；冲突时标量回退 |
| **13.3** Masked | `masked_select`、`masked_scatter`、`masked_fill` | `masked_select` → 亦属 26 | 中 | 中～低 | `vcompress`/`vexpand`（若有）、`vblend`；否则两阶段 |
| **13.4** Embedding | `embedding`、`embedding_bag`、`_embedding_bag` | 查表 + 可选 05；稀疏梯度 | 高～中 | 中（行 gather） | `vgather` 行；bag 再 `vreduce` |
| **13.5** 搜索桶 | `bucketize`、`searchsorted`、`one_hot` | 半有序索引 | 高 | 中～低 | 向量二分或 `vcmp` 边界；`one_hot` 用 `vscatter` |

---

### Pattern 14 — 布局变换

| 子类 | 典型算子 | 备注 | SIMT 易用 | SIMD 易用 | SIMD 典型指令特征 |
|---|---|---|---|---|---|
| **14.1** 元数据 View | `view`、`reshape`（可 view 时）、`expand`、`transpose`/`permute`（可 view）、`squeeze`/`unsqueeze`、`as_strided`、`select`、`narrow`、`diagonal` | `inplace_view` / view 语义；尽量零拷贝 | N/A（无向量核） | N/A（无向量核） | 零指令数据面 |
| **14.2** 物化转置/重排 | `contiguous`、`permute` 打断时、`transpose` 拷贝路径、`movedim` | 分块转置；边缘 gather | 中 | 中 | `vload`/`vstore`、`vtranspose`/`vshuf`；边缘 `vgather`/`vloadm` |
| **14.3** 展平/折叠 | `flatten`、`unflatten`、`unfold` | `unfold` 有重叠窗，偏 08/09 预备 | 高 | 中（`unfold` 重叠） | copy 用 `vload`/`vstore`；`unfold` 常 `vgather` |
| **14.4** view_copy 族 | `*_copy`（如 `transpose_copy`、`permute_copy`） | 功能化 IR / Export 用 | 中 | 中 | 同 14.2：`vload`/`vstore`、`vtranspose`/`vshuf` |

---

### Pattern 15 — 拼接与分割

| 子类 | 典型算子 | 备注 | SIMT 易用 | SIMD 易用 | SIMD 典型指令特征 |
|---|---|---|---|---|---|
| **15.1** 拼接 | `cat`、`concat`、`stack`、`hstack`/`vstack`/`dstack`、`column_stack` | **分段 copy**；每段独立尾块 | 高 | 高 | 每段 `vload`/`vstore`/`vloadm`/`vstorem` |
| **15.2** 分割 | `split`、`split_with_sizes`、`chunk`、`tensor_split`、`unbind`、`hsplit`/`vsplit` | 多为 view 或分段 view | N/A（view）/ 高（物化） | N/A（view）/ 高（物化） | view 无指令；物化同 15.1：`vload`/`vstore` |
| **15.3** 块对角等 | `block_diag` | 隐式零填充区域 ↔ 与 07 相关 | 高 | 中 | 块内 copy + `vsplat(0)` 填空隙 |

---

### Pattern 16 — 拷贝与类型转换

| 子类 | 典型算子 | 备注 | SIMT 易用 | SIMD 易用 | SIMD 典型指令特征 |
|---|---|---|---|---|---|
| **16.1** 拷贝 | `clone`、`copy_`、`_to_copy`、`_copy_from` | 保 dtype 或随 `to` | 高 | 高 | `vload`/`vstore`（可按字节宽向量） |
| **16.2** dtype/device | `to`、`type_as`、`_autocast_to_*_precision`、历史 `_cast_*` | 涉及拷贝+转换 | 高 | 高～中 | `vcvt` + `vload`/`vstore`；跨设备非 SIMD |
| **16.3** 别名 | `alias`、`detach`、`detach_` | 元数据；非数据面 | N/A | N/A | — |

---

### Pattern 17 — 工厂与创建

| 子类 | 典型算子 | 备注 | SIMT 易用 | SIMD 易用 | SIMD 典型指令特征 |
|---|---|---|---|---|---|
| **17.1** 空/常量 | `empty`、`zeros`、`ones`、`full`、`empty_like`、`zeros_like` | 无 tensor 输入或 like | 高 | 高 | `vsplat`+`vstore`；`empty` 可无写 |
| **17.2** 序列 | `arange`、`linspace`、`logspace`、`eye`、`range`（legacy） | 索引生成常用 | 高 | 高～中 | lane id → `vfma` 生成序列；`eye` 用 mask 对角 |
| **17.3** 窗函数 | `bartlett_window`、`hann_window`、`hamming_window`、`kaiser_window` | 与 21 配合 | 高 | 中 | `vcos`/`vmul` 近似或标量表 |
| **17.4** 量化工厂 | `_empty_affine_quantized`、`_empty_per_channel_affine_quantized` | 归属 17 创建、服务 19 | 见组合 | 见组合 | 同 17.1 + 元数据；数据面可选 `vsplat` |

---

### Pattern 18 — 随机与 Dropout

| 子类 | 典型算子 | 备注 | SIMT 易用 | SIMD 易用 | SIMD 典型指令特征 |
|---|---|---|---|---|---|
| **18.1** 分布采样 | `rand`、`randn`、`randint`、`normal`、`uniform_`、`bernoulli`、`poisson`、`multinomial` | `Generator` 控制；nondeterministic 标签 | 高～中（每线程 RNG） | 中（向量 Philox/xorshift） | `vrng`/`vphilox`、`vcvt`、Box-Muller 用 `vlog`/`vsin` |
| **18.2** Dropout | `dropout`、`native_dropout`、`alpha_dropout`、`feature_dropout` | 训练图高频；与 01 融合可能 | 高 | 高～中 | `vrng`+`vcmp`+`vblend`/`vmul` |

---

### Pattern 19 — 量化

| 子类 | 典型算子 | 备注 | SIMT 易用 | SIMD 易用 | SIMD 典型指令特征 |
|---|---|---|---|---|---|
| **19.1** 量/反量 | `quantize_per_tensor`、`quantize_per_channel`、`dequantize` | 进出整型表示 | 高 | 高～中 | `vmul`/`vfma`、`vcvt`（sat）、`vrnd`；per-channel 要 `vbroadcast` 轴 |
| **19.2** Fake quant | `fake_quantize_per_tensor_affine`、`_fake_quantize_learnable_*`、`choose_qparams_*` | QAT；可反传 | 高 | 中 | 同 19.1 + 反传 STE（`vblend`） |
| **19.3** Q 张量元数据 | `q_scale`、`q_zero_point`、`q_per_channel_scales`、`int_repr` | 读量化参数 | N/A（标量元数据） | N/A / 高（`int_repr`） | `int_repr` 同 16：`vload`/`vstore` |
| **19.4** 量化计算核（**Cube**） | `quantized::conv2d`、`quantized::linear`、`_int_mm` 等 | **含 matmul**；与 09/10 整数 Cube 对应；§2 已剔除 | 低 | 低 | **整数 Cube/MMA** + scale epilogue |

**备注**：功能上「量化」既是 **dtype/数值域变换（19.1–19.3）**，也是 **同构算子的整数后端（19.4）**。

---

### Pattern 20 — 稀疏

| 典型算子 | 备注 | SIMT 易用 | SIMD 易用 | SIMD 典型指令特征 |
|---|---|---|---|---|
| `to_sparse`、`to_dense`、`sparse_coo_tensor`、`_to_sparse_csr/csc/bsr/bsc`、稀疏 `mm`/`add`、`_sparse_semi_structured_*` | shape 规则含 sparse/dense 维；与 10/13 交界 | 中 | 低（不规则） | `vgather`/`vscatter`；稠密块内可 `vfma`；semi-structured 走专用 MMA |

---

### Pattern 21 — FFT / 信号

| 典型算子 | 备注 | SIMT 易用 | SIMD 易用 | SIMD 典型指令特征 |
|---|---|---|---|---|
| `_fft_c2c`、`_fft_r2c`、`_fft_c2r`、`fft.*`、`stft`、`istft`、`fftfreq` | 多调 MKL / cuFFT / pocketfft；计划缓存（`_cufft_*`）属辅助 | 低（专用库） | 低（专用库） | 库内自选 / MMA |

---

### Pattern 22 — 损失函数

| 典型算子 | 备注 | SIMT 易用 | SIMD 易用 | SIMD 典型指令特征 |
|---|---|---|---|---|
| `nll_loss`、`cross_entropy`、`mse_loss`、`l1_loss`、`smooth_l1_loss`、`binary_cross_entropy`、`kl_div`、`ctc_loss`、`triplet_margin_loss`、`hinge_embedding_loss` | 内部常 **01/05/13 组合**；CTC 等更专用 | 高～中 | 中（先 01 再 05） | `vsub`/`vmul`/`vmax`/`vlog` + `vreduce`；CTC **低**（动态规划，非 SIMD map） |

---

### Pattern 23 — 上采样与插值

| 典型算子 | 备注 | SIMT 易用 | SIMD 易用 | SIMD 典型指令特征 |
|---|---|---|---|---|
| `upsample_nearest2d`、`upsample_bilinear2d`、`upsample_bicubic2d`、`upsample_trilinear3d`、`_upsample_*_aa`、`grid_sampler_2d/3d`、`affine_grid` | 固定维；与 09.2 转置卷积不同路径 | 高～中 | 中～低（邻域 gather） | nearest=`vgather`/复制；bilinear=`vload` 邻域 + `vfma` 权重；grid_sample 几乎必 `vgather` |

---

### Pattern 24 — Nested / Jagged

| 典型算子 | 备注 | SIMT 易用 | SIMD 易用 | SIMD 典型指令特征 |
|---|---|---|---|---|
| `_nested_from_padded`、`_nested_tensor_*`、`_jagged_to_padded_dense_forward`、`_pack_padded_sequence` | **变长** 与 07 填充互转；动态 shape 友好表示 | 高～中（按段） | 中（段内向量，段间标量调度） | 段内同 15/16；padding 区 `vsplat` |

---

### Pattern 25 — 排序 / TopK / Unique

| 子类 | 典型算子 | 备注（含硬件依赖） | SIMT 易用 | SIMD 易用 | SIMD 典型指令特征 |
|---|---|---|---|---|---|
| **25.1** 排序选择 | `sort`、`argsort`、`topk`、`kthvalue`、`msort` | 基线手写 bitonic；**有定长 `vsort`+跨块 `vmergesort` 则 SIMD′=亲(2)**（总表） | 中 | **亲（有 vsort/mergesort）/ 难～低（无）** | **`vsort`**、**`vmergesort`/`vmerge_odd_even`**、`vcmp`/`vblend`；无则 `vshuf` 网络（§8.4） |
| **25.2** Unique / 成员 | `unique`、`unique_consecutive`、`_unique2`、`isin` | 常先排序；依赖 25.1 + `vcompress` | 中 | 中（有 sort+compress）/ 低（无） | `vsort` 后 `vcmp` 邻差 + **`vcompress`** |

---

### Pattern 26 — 数据依赖动态输出

| 典型算子 | 备注 | SIMT 易用 | SIMD 易用 | SIMD 典型指令特征 |
|---|---|---|---|---|
| `nonzero`、`masked_select`、`unique`（长度）、同步 compact | `dynamic_output_shape`；两阶段；**有 `vcompress`+`vprefix` 则 SIMD 升档** | 中 | **中（有 compress）/ 低（无）** | `vcmp`、**`vprefix`/`vreduce`**、**`vcompress`**/`vscatter`（§8.4） |

---

### Pattern 27 — 特殊函数

| 典型算子 | 备注 | SIMT 易用 | SIMD 易用 | SIMD 典型指令特征 |
|---|---|---|---|---|
| `special.erf`、`erfc`、`erfinv`、`lgamma`、`digamma`、`polygamma`、`i0`、`sinc`、`xlogy` | 数学特殊函数；实现形态同 01，单独成类便于库映射 | 高 | 中（多项式/有理近似） | 同 01 + `vfma` 多级近似；缺向量超越函数时 SIMT 更易 |

---

### Pattern 28 — 元信息 / 控制 / 辅助

| 子类 | 典型算子 | 备注 | SIMT 易用 | SIMD 易用 | SIMD 典型指令特征 |
|---|---|---|---|---|---|
| **28.1** 查询 | `size`、`stride`、`dim`、`numel`、`is_floating_point`、`device`、`layout` | 非数据面 | N/A | N/A | — |
| **28.2** 断言/调试 | `_assert_async`、`_assert_tensor_metadata`、`_foobar` | 图安全/测试；`_assert_async` 亦带 **29** 序依赖语义 | N/A / 设备侧标量比较 | N/A / 设备侧标量比较 | N/A / 设备侧标量比较；非 SIMD 主路径 |
| **28.3** Autograd 辅助 | `_backward`、`requires_grad_`、AMP `_amp_*` | 训练基础设施 | N/A（控制面） | N/A / 高（AMP unscale） | AMP unscale 可同 01：`vmul`/`vcmp` |

---

### Pattern 29 — 同步 / 序约束

> 同步算子 **通常不改变张量数值**，而是建立 **happens-before**。特征提取见 §7。

| 子类 | 典型 API / 算子 | 备注 | SIMT 易用 | SIMD 易用 | SIMD 典型指令特征 |
|---|---|---|---|---|---|
| **29.1** Host↔Device 全同步 | `torch.cuda.synchronize` / `torch.npu.synchronize`、`Stream.synchronize`、`.item()` / `.cpu()` 隐式同步 | CPU 线程阻塞到指定 device/stream 完成 | N/A（运行时） | N/A | — |
| **29.2** 跨 Stream 序 | `Event.record` / `Event.wait` / `Stream.wait_event` / `Stream.wait_stream` | 设备侧序；默认 **不** 阻塞 Host | N/A | N/A | N/A；对应硬件 event，非 SIMD |
| **29.3** 张量–流绑定 | `record_stream` | ATen schema 有显式条目；缓存分配器跨流复用安全 | N/A | N/A | — |
| **29.4** 图/编译依赖令牌 | `_make_dep_token`、`_functional_assert_async`、带 `dep_token` 的 constrain | 把副作用变成 SSA 边，供 Inductor/Export | N/A（IR 边） | N/A | — |
| **29.5** 集合通信完成 | `Work.wait` / `Work.synchronize`、`barrier`、`async_op=False` 路径 | **Host-block** vs **stream-order** 必须拆开（见 §7.3） | N/A | N/A | —（通信完成协议；核内 reduce 见 05/29.8） |
| **29.6** P2P 通信序 | `send`/`recv`（同步语义）、`isend`/`irecv` + `wait` | CUDA+NCCL 上常为 stream 序，易被误读成 Host 阻塞 | N/A | N/A | — |
| **29.7** 隐式同步点 | 非 `non_blocking` 的 D2H、跨设备 `copy_`、部分 view 失败路径 | 无独立算子名，但特征上等同 29.1 | N/A | N/A | —（伴随 copy 数据面见 16） |
| **29.8** 核内同步（后端） | `__syncthreads`、AscendC `PipeBarrier`/`SetFlag`/`WaitFlag` | 不在 Python ATen 面；lowering 时与 29.2 对照 | 必需（block barrier） | 模型通常隐式（tile 边界） | `bar.sync` / `PipeBarrier`，**不是**算术 SIMD |

---

## 4. Pattern 交叉与组合（真实模型）

| 组合 | 例 | Pattern 链 |
|---|---|---|
| Linear | `F.linear` | **Cube 10** →（可选 Vec 01 bias）→（可选 03） |
| Conv-BN-ReLU | CNN 块 | **Cube 09** → 11 → 03 |
| LayerNorm | Transformer | 11（内含 05+01） |
| SDPA | Attention | **Cube 12.2**（内含 matmul）+ Vec 12.1 epilogue |
| Embedding + sum | `embedding_bag` | 13 → 05 |
| Pad + Pack | RNN batch | 07 ↔ 24 |
| QDQ + Conv | PTQ/QAT | 19 QDQ → **Cube 19.4/09** |
| 通信–计算重叠 | async allreduce + compute + wait | Cube/Vec ∥ **29.5** → wait |
| 多流通用 | copy 到侧流 + event 回主流 | 16 + **29.2/29.3** |

---

## 5. API 特征分类（仍用数字编号引用）

### 5.1 四层责任栈

```text
L3  用户 API         torch.*/Tensor.*/F.*/nn.*
L2  Schema+Dispatch  native_functions.yaml
L1  Plan/Iterator    TensorIterator / meta / structured
L0  Backend          CPU vec / CUDA / 库 / NPU / Triton…
```

| API 类 | 编号 | 谁处理 shape/广播/尾块 | 覆盖 Pattern |
|---|---|---|---|
| 声明式算子 | **A1** | 框架 L1 | 01–06、14–16 为主 |
| out 变体 | **A2** | 调用方备缓冲 + 框架校验 | 多数计算类 |
| inplace | **A3** | 受限广播 | 01–03、07.1 等 |
| 领域库入口 | **A4** | 库内 tiling | 09、10、12、21 |
| 可编程 DSL | **A5** | 用户 mask/tile | 自研核；不限 |
| 运行时同步/通信 | **A6** | 用户或框架显式 wait | **29**（Stream/Event/c10d） |

### 5.2 API 形态 × Pattern

| 特征 | 强相关 Pattern | 备注 |
|---|---|---|
| 隐式广播 | 01–03、04 | 用户不写 expand |
| dim/keepdim | 05、06、11、25 | 负维规范化 |
| padding 参数 | 07、09、08 | 语义 pad |
| scale/zero_point | 19 | 量化契约 |
| Generator | 18 | 随机可复现 |
| 动态输出 | 26、25.2、13.3 | 导出需小心 |
| Stream / Event / Work | 29 | 序约束；常无 Tensor 输出 |

---

## 6. 动态 Shape、非对齐、Padding（按 Pattern）

### 6.1 动态 Shape 三层

| 层级 | 含义 | 高发 Pattern |
|---|---|---|
| **S1** 运行时长度可变 | 每次 forward 的 N/H/W 变 | 01–16、09–12 |
| **S2** 符号维（编译） | SymInt / 守卫 | PT2 全图；09/10 特化敏感 |
| **S3** 数据依赖输出 | 输出长度看数值 | **26**、25.2、13.3 |

### 6.2 非对齐三类

| 类型 | 含义 | 高发 Pattern |
|---|---|---|
| **U1** 长度非向量整除 | `n % lanes != 0` | 01–03、05、16 |
| **U2** 地址未对齐 | ptr 不对齐 | 01、16；部分 NPU 硬约束 |
| **U3** 非连续 stride | 需 strided 或先 contiguous | 14 之后接 01；13 |

### 6.3 各 Pattern 处方（摘要）

| Pattern | Dyn | Align / 尾块 | Pad |
|---|---|---|---|
| 01–03 | S1 每次按 numel 切 | U1 尾块；热路径无谓词整块 | 仅向量尾（非语义） |
| 04 | 重算 stride_eff | 0-stride 地址生成 | 逻辑拉伸，勿过早物化 |
| 05 | plan {outer,reduce,inner} | 05.c 短内维打包 | 尾 lane 中性元 |
| 07 | pad 宽度可动态 | 边界写 | **语义 Pad** |
| 08–09、23 | 空间尺寸 S1/S2 | 库内对齐 | 算法 padding ≠ 硬件 pad |
| 10 | MNK 动态 | 分块对齐到 MMA | 库内 P-HW |
| 12 | 序列长动态 | tile 对齐 | mask 或 pad 到 tile（注意语义） |
| 13 | 索引长动态 | gather 难对齐 | 一般无 |
| 15 | 段长各异 | **每段** 自有尾块 | 段间不强制 pad |
| 19 | 同对应浮点算子 | 整型向量宽可能不同 | qparams 与对齐解耦 |
| 24 | 变长本质 | 与 07 互转 | padded ↔ nested |
| 26 | **S3** | compact 尾块 | 上界缓冲≈软 pad |
| 29 | 一般不改 shape | 与对齐无关 | 不引入语义 Pad；通信缓冲对齐另计 |

### 6.4 推荐契约

```text
1. API 不要求用户保证 nbytes % VL == 0
2. 07 语义 Pad 与硬件对齐 Pad 必须在文档/下层分开
3. 05.c / 短尾轴：优先 layout 变换，再考虑硬件 pad
4. 26：显式两阶段，不伪装成 01
5. 09/10/12：领域 API；只复用分块与尾块约定
6. 29：文档与实现必须标明阻塞域（Host / Stream / Rank），禁止把 stream-order 写成 Host 同步
```

---

## 7. 同步算子特征提取

同步类（Pattern **29**）与 01–28 正交：**不描述“算什么”，而描述“何时可见/何时可继续”**。特征提取按下表维度做，便于中间层路由、图捕获与 NPU Pipe 映射。

### 7.1 特征维度（提取模板）

| 特征 ID | 名称 | 取值空间 | 提取问题 |
|---|---|---|---|
| **F1** | 阻塞域 | `Host` / `Device` / `Stream` / `Rank` / `Graph` | 谁在等待？ |
| **F2** | 作用域 | `current_stream` / `named_stream` / `device_all` / `process_group` / `p2p_peer` | 等的范围有多大？ |
| **F3** | 序关系 | `HB`（happens-before）/ `barrier`（多方汇合）/ `token`（SSA 依赖） | 建立何种偏序？ |
| **F4** | 同步粒度 | `full_device` / `stream` / `event` / `work_item` / `kernel_block` | 粒度越粗气泡越大 |
| **F5** | 可见性 | `memory_visible` / `progress_only` | 是否保证内存对等待方可见（通常 HB ⇒ 可见） |
| **F6** | 异步可分性 | `fused`（调用即等）/ `split`（发起 + 显式 wait） | 能否与计算重叠 |
| **F7** | 超时/可查询 | `blocking` / `timed_wait` / `query` | 是否可轮询或带 timeout |
| **F8** | 图捕获兼容 | `allowed` / `forbidden` / `record_only` | CUDA Graph / 图模式能否记录 |
| **F9** | 数值副作用 | `none` / `comm_buffer` / `assert_only` | 是否改用户张量（纯同步应为 none） |
| **F10** | 隐式触发 | `explicit` / `implicit_on_copy` / `implicit_on_read` | API 是否在名字里暴露同步 |

### 7.2 子类 × 特征矩阵

| 子类 | F1 阻塞域 | F2 作用域 | F3 序 | F4 粒度 | F6 可分 | F8 图捕获 | 典型入口 |
|---|---|---|---|---|---|---|---|
| **29.1** Host↔Device | Host | device / stream | HB | device 或 stream | fused | 常 forbidden（host sync） | `cuda.synchronize`、`stream.synchronize` |
| **29.2** 跨 Stream | Device（流） | event / stream | HB | event | split（record/wait） | record/wait 节点可进图 | `Event`、`wait_stream` |
| **29.3** record_stream | —（注解） | tensor↔stream | 分配器序 | tensor | — | 需一致记录 | `record_stream` |
| **29.4** dep_token | Graph/编译 | token 边 | token HB | op 边 | split（token 传递） | **为图设计** | `_make_dep_token`、functional assert |
| **29.5** 集合通信 wait | Host 或 Stream | process_group | barrier / HB | work | split（async_op） | capture 时禁 host sync | `Work.wait`、`barrier` |
| **29.6** P2P | Host 或 Stream | peer rank | HB | work | `isend/irecv` 可分 | 同 29.5 | `send/recv`、`isend/irecv` |
| **29.7** 隐式同步 | Host | 触发该次拷贝的流 | HB | stream/device | fused（藏在 copy） | 图内需改成显式 | `.item()`、同步 `copy_` |
| **29.8** 核内同步 | Device 线程块/Pipe | block / pipe | HB / barrier | block 或硬件 pipe | 核内 fused | 核内原语 | `__syncthreads`、`PipeBarrier` |

### 7.3 关键语义：Host-block vs Stream-order

同一 API 名在不同后端可表示两种完全不同的 F1：

| 语义 | 含义 | 后果 |
|---|---|---|
| **Host-block** | CPU 线程等到 GPU/通信完成 | Python 后续代码所见内存已就绪；易串行化 |
| **Stream-order** | 仅当前 CUDA/NPU stream 等待某 event/work | CPU 可继续；必须在同设备流序上消费结果 |

**提取规则**：

1. 标注每个 29.x API 的默认语义（NCCL 上 `Work.wait` / 部分 P2P 常为 stream-order）。  
2. `barrier(async_op=False)` 路径若声称同步，应核对其是否真正 Host-block。  
3. 图捕获期间：Host-block（`cudaStreamSynchronize` / `Event.synchronize`）通常 **非法或无意义** → F8=`forbidden`。  
4. 通信–计算重叠：发起侧 F6=`split`，消费前必须有匹配的 wait（29.5/29.2）。

### 7.4 ATen / 运行时中的显式锚点

| 锚点 | 所在层 | 提取要点 |
|---|---|---|
| `record_stream(Tensor, Stream)` | ATen schema | 唯一直接进 `native_functions` 的流注解算子；F9=`none`，服务分配器 |
| `_assert_async` / `_functional_assert_async` | ATen | 设备侧断言；functional 版用 `dep_token` 把断言纳入 29.4 |
| `_make_dep_token` | ATen | 纯序载体；无数值 |
| `torch.cuda.Event/Stream`、`torch.npu.*` | 运行时绑定 | 不在 yaml 功能表，但属 Pattern 29 用户面 |
| `torch.distributed.*` + `Work` | c10d | 集体/P2P；完成协议走 29.5/29.6 |
| `non_blocking=True` 的 `copy_` / `to` | 16 + 29 | **缺少** 显式 wait 时，后续 Host 读触发 29.7 |

### 7.5 与计算 Pattern 的耦合点

| 计算侧 | 同步特征如何挂接 |
|---|---|
| 01–16 单流默认 | 依赖默认流隐式序；多流时需补 29.2/29.3 |
| 10/12 长核 | 侧流通信用 29.5 split，主流算完再 wait |
| 16 D2H | 默认同刻 29.7；profiling 时应标为隐式同步 |
| 26 DynOut | 常被迫 Host 同步拿长度（F1=Host，F10=implicit） |
| 19 量化参数读取 | `q_scale` 等若落 Host，可能隐式同步 |

### 7.6 特征提取检查清单（落地）

对每个同步相关 API / IR 节点填写：

```text
name:
subclass: 29.x
F1_block_domain: Host | Stream | Device | Rank | Graph
F2_scope: ...
F3_relation: HB | barrier | token
F4_granularity: ...
F6_async: fused | split
F8_graph: allowed | forbidden | record_only
F9_side_effect: none | ...
F10_implicit: explicit | implicit_on_copy | implicit_on_read
notes:  # 如 “NCCL wait 默认为 stream-order，非 Host-block”
```

映射建议：

- 中间层 IR：显式节点 `Sync` / `Wait` / `RecordEvent` / `CommWork`，禁止靠隐式 D2H 表达序。  
- NPU lowering：29.2 ↔ Event/Stream；29.8 ↔ `PipeBarrier` / `SetFlag`/`WaitFlag`；29.5 ↔ HCCL/HCOMM work + 明确 Host vs 设备序。  
- 与 CCU / fullmesh 文档对照时：旗标 poll、RDMA Complete 属于 **29.5/29.6** 的设备侧完成协议，不是 05 规约。

---

## 8. 附录

### 8.1 术语

| 术语 | 含义 |
|---|---|
| Pattern NN | 本文核心功能编号 |
| schema | `native_functions.yaml` 中一条 `- func:` |
| 语义 Pad | Pattern 07；改变填充区数值约定 |
| 硬件 Pad | 后端对齐引入的额外元素，不改变对外语义但影响性能 |
| DynOut | Pattern 26 |
| Host-block | CPU 线程等待设备/通信完成 |
| Stream-order | 仅流队列上的 happens-before，不阻塞 Host |
| dep_token | 编译图中的显式序依赖载体（29.4） |

### 8.2 参考

- `aten/src/ATen/native/native_functions.yaml`、`tags.yaml`（含 `record_stream`、`_assert_async`、`_make_dep_token`）  
- TensorIterator 文档与实现  
- ezyang: *A brief taxonomy of PyTorch operators by shape behavior*（按 shape 行为的另一正交分类）  
- PyTorch CUDA Stream/Event 文档；c10d `Work.wait` / `barrier` Host vs stream 语义讨论  
- PyTorch Quantization / Sparse / NestedTensor 文档  
- AscendC `PipeBarrier` / `SetFlag` / `WaitFlag`；集合通信完成旗标  

### 8.3 维护

| 变更 | 动作 |
|---|---|
| 新增功能族 | 追加 Pattern 30+；不复用已有数字 |
| 算子迁类 | 改 §3 典型列表与备注，更新交叉表 §4 |
| 同步语义变更 | 更新 §7 特征矩阵与 Host-block / stream-order 注记 |
| SIMT/SIMD 映射变更 | 更新 §3.0 助记表与各 Pattern 末列 |
| **SIMD 硬件原语变更** | 更新 §3.0.1、相关 Pattern 备注、**§8.4 原型**、总表「特殊指令」与 **s_SIMD′** |
| 统计口径变化 | 更新 §1.1 |

### 8.4 SIMD 指令原型（可移植伪接口）

> 下列为 **逻辑原型**（非某一 ISA 汇编）。`VL` = 固定向量 lane 数（或实现定义的最大定长）；`mask_t` 与 `vec<T,VL>` 同宽。后端映射：AVX-512 / NEON / RVV / AscendC Vec / C++26 `simd` 等。  
> **定长**意指单次操作作用在编译期可知或 ABI 固定的 `VL` 上（相对“软件循环拼规约”）。

```cpp
// ---- 类型 ----
template<class T, int VL> struct vec;      // VL 个 T 的定长向量
template<int VL>          struct mask_t;   // VL 位谓词
enum class reduce_op { add, mul, max, min, and_, or_ };
enum class scan_op   { add, max, min, mul };

// ---- 访存（Pattern 01–03 / 15–17 尾块关键）----
template<class T, int VL>
vec<T,VL> vload(const T* p);                          // 连续；可有 unaligned 变体

template<class T, int VL>
vec<T,VL> vloadm(const T* p, mask_t<VL> m);           // 无效 lane 不产生故障读

template<class T, int VL>
void vstore(T* p, vec<T,VL> v);

template<class T, int VL>
void vstorem(T* p, mask_t<VL> m, vec<T,VL> v);

template<class T, int VL>
vec<T,VL> vgather(const T* base, vec<int,VL> idx);

template<class T, int VL>
void vscatter(T* base, vec<int,VL> idx, vec<T,VL> v);
// 冲突：确定性顺序或要求无冲突；否则 vatomic_add

// ---- 逐 lane 算术 / 选择 ----
template<class T, int VL>
vec<T,VL> vadd(vec<T,VL> a, vec<T,VL> b);
template<class T, int VL>
vec<T,VL> vmul(vec<T,VL> a, vec<T,VL> b);
template<class T, int VL>
vec<T,VL> vfma(vec<T,VL> a, vec<T,VL> b, vec<T,VL> c); // a*b+c

template<class T, int VL>
mask_t<VL> vcmp_lt(vec<T,VL> a, vec<T,VL> b);         // 及 eq/ne/le/...

template<class T, int VL>
vec<T,VL> vblend(mask_t<VL> m, vec<T,VL> t, vec<T,VL> f); // m?t:f

template<class T, int VL>
vec<T,VL> vsplat(T x);                                // 广播到 VL

template<class To, class From, int VL>
vec<To,VL> vcvt(vec<From,VL> x);                      // 类型转换

// ---- 定长水平规约（抬升 Pattern 05/11/12.1；硬件优先提供）----
// 语义：对 v 的全部 VL 个 lane 做结合运算，返回标量（或广播回 vec，由后端定）
template<class T, int VL>
T vreduce(reduce_op op, vec<T,VL> v);
// 等价别名（便于 codegen）：
//   T vreduce_add(vec<T,VL>);
//   T vreduce_max(vec<T,VL>);
//   T vreduce_min(vec<T,VL>);
//   T vreduce_and(vec<T,VL>); // 逻辑/按位
//   T vreduce_or (vec<T,VL>);

// 掩码规约：m=false 的 lane 视作 op 的中性元（+0 / +inf / true…），不参与
template<class T, int VL>
T vreduce_masked(reduce_op op, vec<T,VL> v, mask_t<VL> m);

// Arg-reduce：返回 (值, lane_index)；无效 lane 禁用
template<class T, int VL>
struct arg_t { T val; int idx; };
template<class T, int VL>
arg_t<T> vreduce_max_arg(vec<T,VL> v, mask_t<VL> m);
template<class T, int VL>
arg_t<T> vreduce_min_arg(vec<T,VL> v, mask_t<VL> m);

// 无整宽 reduce 时的积木（软件模拟定长 reduce）：
template<class T, int VL>
vec<T,VL> vhadd(vec<T,VL> v);   // 相邻对水平加，宽度折半语义由实现定义
template<class T, int VL>
vec<T,VL> vshuf(vec<T,VL> v, vec<int,VL> perm);

// ---- 定长前缀 / 扫描（抬升 Pattern 06/26）----
template<class T, int VL>
vec<T,VL> vscan(scan_op op, vec<T,VL> v, bool inclusive = true);
// 别名：vscan_add / vscan_max / …

template<class T, int VL>
vec<T,VL> vprefix_sum(vec<T,VL> v);   // 常用；可与 vscan(add) 同构

// ---- 压缩 / 展开（抬升 Pattern 26 / masked_select）----
template<class T, int VL>
int vcompress_store(T* out, mask_t<VL> m, vec<T,VL> v);
// 将 m=true 的 lane 按序写入 out，返回写入个数（≤VL）

template<class T, int VL>
vec<T,VL> vexpand_load(const T* in, mask_t<VL> m);
// 按 m 把紧凑输入展开到向量（其余 lane 未定义或零）

// ---- 定长排序 / 归并（抬升 Pattern 25：sort / topk / argsort）----
// 语义：单向量内全序排序；stable 可选。返回值向量；argsort 另返索引。
enum class sort_order { asc, desc };

template<class T, int VL>
vec<T,VL> vsort(vec<T,VL> v, sort_order ord = sort_order::asc);

template<class T, int VL>
struct sorted_t { vec<T,VL> keys; vec<int,VL> idx; };
template<class T, int VL>
sorted_t<T,VL> vargsort(vec<T,VL> v, sort_order ord = sort_order::asc);

// 跨向量归并积木：输入 a、b 各自已有序，输出长度为 2·VL 的有序对（或写回两个 vec）
template<class T, int VL>
void vmergesort(vec<T,VL> a, vec<T,VL> b, vec<T,VL>& lo, vec<T,VL>& hi,
                sort_order ord = sort_order::asc);
// 等价别名 / 网络积木：
//   vmerge_odd_even(a,b)  — odd-even 归并一层
//   vbitonic_merge(a,b)   — bitonic 归并一层
// 大数组 = 分块 vsort + 多轮 vmergesort（库级 mergesort / bitonic）

// 掩码部分排序（TopK 热路径）：仅对 m=true 的有效 lane 排序，无效 lane 沉底/置哨兵
template<class T, int VL>
vec<T,VL> vsort_masked(vec<T,VL> v, mask_t<VL> m, sort_order ord = sort_order::asc);

// ---- 其它 ----
template<class T, int VL>
vec<T,VL> vrng(rng_state& st);        // 向量随机（Pattern 18）

template<class T, int VL>
void vatomic_add(T* base, vec<int,VL> idx, vec<T,VL> v); // scatter-add

// Cube（不属通用 SIMD；Pattern 09/10/12.2）
// mma_sync(acc, a, b) / CubeLoad / CubeExecute … 见各 NPU/GPU MMA ABI
```

**与 Pattern 的最小硬件清单（建议实现优先级）**：

| 优先级 | 原语 | 解锁的 SIMD 易用性 |
|---|---|---|
| P0 | `vload/vstore`、`vloadm/vstorem`、`vadd/vmul/vfma`、`vcmp/vblend` | 01–03、15–17 双亲 |
| P0 | **`vreduce_{add,max,min}` + `_masked`** | **05/11/12.1 从中/对方 → 高** |
| P1 | `vscan`/`vprefix`、`vcompress` | 06、26 |
| P1 | `vgather`/`vscatter`、`vtranspose` | 13、14、23 |
| P1 | **`vsort` / `vmergesort`（+ `vargsort`/`vsort_masked`）** | **25 难→亲（SIMD′=2）** |
| P2 | `vrng`、`vreduce_*_arg`、超越函数近似 | 18、05.2、03 |
| — | Cube/MMA | 09/10/12.2（独立路由） |

**缺省回退**：无 `vreduce` 时用 `log2(VL)` 级 `vshuf`+`vhadd` 软件树；无 `vsort`/`vmergesort` 时用 `vcmp`+`vblend`/`vshuf` 手写 bitonic，并把 Pattern **25** 的 SIMD 易用性标为 **难**（基线 `s_SIMD=0`，与总表一致）。

---

*本文以核心功能 + 数字 Pattern 组织 PyTorch 三千量级底层算子；同步特征见 Pattern 29；SIMD 易用性分 **基线** 与 **特殊指令后（SIMD′，含 `vreduce`/`vsort`/`vmergesort` 等）** 两档，原型见 §8.4。*
