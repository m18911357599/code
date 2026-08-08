# PyTorch 内置底层算子：按核心功能分类、API 特征与同步特征

> 范围：以 ATen `native_functions.yaml` 及量化/稀疏/分布式等命名空间为对象，覆盖业界常称的 **约三千量级** 内置底层算子（见 §1.1）。  
> 本文按 **核心功能** 重新分类；Pattern 统一用 **两位数字编号**；每类给出典型算子与备注。  
> 对照维度：API 形态、动态 shape / 非对齐 / padding，以及 **同步算子特征提取（Pattern 29）**。

---

## 0. 结论摘要

| 维度 | 结论 |
|---|---|
| **分类主轴** | 按 **核心功能** 分为 **Pattern 01–29**（逐元素、规约、填充、量化、卷积、矩阵乘、**同步** 等），不以字母代号 |
| **规模** | schema 约 **2500+**，唯一基名约 **1500+**；计入 inplace / out / 重载 / quantized·sparse·foreach / c10d 后常称 **~3000–3500** |
| **实现主路径** | **01–06、14–16** 多走 TensorIterator；**07–12、19、22–23** 多为固定维/领域核；**25–26** 含数据依赖动态输出；**29** 走运行时/通信后端 |
| **API** | 用户声明式 API → Schema/Dispatch → Plan/Iterator → Backend；中间层吸收动态与非对齐 |
| **Dyn / Align / Pad** | 语义 Pad（07）≠ 向量尾填充 ≠ 硬件对齐 Pad；短尾轴在 NPU 上易被隐式 pad |
| **同步** | 同步不是计算 Pattern，而是 **执行序约束**；需按阻塞域（Host/Device/Stream/Rank）与序关系（happens-before）提取特征 |
| **SIMT/SIMD** | Pattern 详解含易用性与所需向量指令；**01–03/15–16 双高**，**09–10/12 走 MMA**，**13/25–26 SIMD 偏低** |

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

> **亲 SIMT/SIMD**：该类更贴哪套并行模型（详解见 §3）。`双亲`=两者都易写；`偏 SIMT/偏 SIMD`=一方明显更顺；`亲 MMA`=峰值靠矩阵加速单元而非通用 SIMD；`N/A`=无数据面向量化。

| Pattern | 核心功能 | 定义（一句话） | 规模感 | 典型访存/并行 | 亲 SIMT/SIMD |
|---|---|---|---|---|---|
| **01** | 逐元素算术 | 同址（广播后）二元/一元数值运算 | 很大 | 带宽；完全并行 | **双亲** |
| **02** | 比较 / 逻辑 / 选择 | 比较、位运算、`where` 类 | 大 | 同 01 | **双亲** |
| **03** | 激活与非线性 | 逐点非线性；常可与 01/09/10 融合 | 中 | 同 01 | **双亲**（超越函数 SIMD 略降） |
| **04** | 广播与维扩展 | 逻辑扩展 shape，默认不拷贝 | 中 | 0-stride / 物化 | **偏 SIMT**（通用广播）；标量广播 SIMD 亦易 |
| **05** | 规约 | 多→少；沿 dim 聚合 | 大 | 树归约 / 分块 | **双亲中**（SIMT 靠 warp reduce；SIMD 靠 `vreduce`） |
| **06** | 扫描与累积 | 有序前缀依赖 | 小 | 难乱序 | **偏 SIMT**（扫描原语）；SIMD 需 `vscan` |
| **07** | 填充 | 边界/长度语义 Pad | 中 | 搬移 + 边界写 | **偏 SIMT**（边界分支）；常值填充 SIMD 易 |
| **08** | 池化 | 滑窗降采样 / 反池化 | 中 | 固定维；空间并行 | **偏 SIMT**；沿轴可 SIMD |
| **09** | 卷积 | 局部加权；含转置卷积 | 中 | 高算术密度 | **亲 MMA**（非裸 SIMD） |
| **10** | 矩阵乘与线性代数 | GEMM / 分解 / 求解 | 大 | MMA / 库 | **亲 MMA**；小向量积可 SIMD |
| **11** | 归一化 | BN / LN / GN / RMSNorm 等 | 中 | 规约+广播融合 | **双亲中**（05+01 组合） |
| **12** | Softmax / Attention | Softmax、SDPA、MHA | 中 | 融合 IO 感知 | Softmax **双亲中**；SDPA **亲 MMA** |
| **13** | 索引 / 散射 / Embedding | 间接读写 | 大 | gather/scatter | **偏 SIMT**；SIMD 需 gather/scatter |
| **14** | 布局变换 | view / transpose / reshape 等 | 很大 | 元数据或搬移 | view **N/A**；物化转置 **偏 SIMD**（`vtranspose`） |
| **15** | 拼接与分割 | cat / stack / split | 中 | 分段 copy | **双亲**（分段 copy） |
| **16** | 拷贝与类型转换 | clone / to / cast | 中 | 带宽 | **双亲** |
| **17** | 工厂与创建 | 无输入或仅 shape 造张量 | 中 | 写填充 | **双亲** |
| **18** | 随机与 Dropout | 采样、噪声、dropout | 中 | RNG + 逐点 | **偏 SIMT**；有向量 RNG 则双亲 |
| **19** | 量化 | quantize / dequant / fake_quant | 中 | 量纲变换+整数核 | QDQ **双亲**；Q-GEMM **亲 MMA** |
| **20** | 稀疏 | COO/CSR 等稀疏格式与算子 | 中 | 间接、压缩 | **偏 SIMT** |
| **21** | FFT / 信号 | 频域变换 | 小–中 | 专用库 | **亲库**（非通用 SIMT/SIMD map） |
| **22** | 损失函数 | 训练目标 | 中 | 常含规约 | **偏 SIMT/双亲中**（01+05）；CTC 低 |
| **23** | 上采样与插值 | upsample / grid_sample | 中 | 固定维插值 | **偏 SIMT**（邻域 gather） |
| **24** | Nested / Jagged | 变长序列结构 | 小–中 | 偏移+填充交互 | **偏 SIMT**（按段）；段内可 SIMD |
| **25** | 排序 / TopK / Unique | 比较网络、选择 | 小–中 | 不规则 | **偏 SIMT**；小块可 SIMD 比较网络 |
| **26** | 数据依赖动态输出 | 输出 shape 依赖 **数值** | 小 | 两阶段 | **偏 SIMT**；有 `vcompress` 则 SIMD 中 |
| **27** | 特殊函数 | `special.*`、高阶数学 | 中 | 多为 01 变体 | **偏 SIMT**（近似）；有向量库则双亲 |
| **28** | 元信息 / 控制 / 辅助 | size、device、assert、autograd 钩子 | 中 | 非计算主路径 | **N/A** |
| **29** | 同步 / 序约束 | Host·Device·Stream·跨 Rank 等待与屏障 | 中 | 不改数值；约束执行序 | **N/A**（29.8 核内 barrier 属 SIMT 原语） |

复合模块（`nn.Linear`、`nn.MultiheadAttention`）由上表 Pattern **组合** 而成，不单开编号。

---

## 3. 各类详解：子类、典型算子、备注、SIMT/SIMD

### 3.0 列约定：SIMT / SIMD 易用性与 SIMD 指令特征

详表在原有列之外增加三列：

| 列名 | 含义 |
|---|---|
| **SIMT 易用** | 一线程一元素（CUDA grid-stride / warp 协作）写出正确实现的难度；**高=易写** |
| **SIMD 易用** | 一指令多 lane / tile（AVX·NEON·RVV·AscendC Vec·SIMD-Tile）写出正确实现的难度；**高=易写** |
| **SIMD 典型指令特征** | 该类热路径需要的可移植向量指令族（后端再映射到具体 ISA） |

易用性取值：`高` / `中` / `低` / `N/A`（可带简短括注）。

**SIMD 指令助记**：

| 助记 | 特征含义 |
|---|---|
| `vload`/`vstore` | 连续向量读写（含 unaligned） |
| `vloadm`/`vstorem` | mask/谓词读写（尾块） |
| `vadd`/`vmul`/`vfma`/`vdiv`… | 逐 lane 算术 |
| `vcmp`/`vblend`/`vwhere` | 比较与选择 |
| `vsplat`/`vbroadcast` | 标量/短向量扩到全宽 |
| `vhadd`/`vhmax`/`vreduce_*` | 水平/树规约 |
| `vgather`/`vscatter` | 间接读写 |
| `vshuf`/`vtranspose` | 重排 / 分块转置 |
| `vcvt` | dtype 转换 |
| `vprefix`/`vscan` | 有序前缀 |
| `vcompress`/`vexpand` | 按 mask 压缩/展开 |
| `vatomic_*` | lane/地址原子（scatter 冲突） |
| `MMA/*` | 非通用 SIMD，矩阵加速单元 |

---

### Pattern 01 — 逐元素算术

| 子类 | 含义 | 典型算子 | 备注 | SIMT 易用 | SIMD 易用 | SIMD 典型指令特征 |
|---|---|---|---|---|---|---|
| **01.1** 二元算术 | `+/-/×/÷` 等 | `add`、`sub`、`mul`、`div`、`remainder`、`pow`、`atan2`、`hypot` | 默认支持广播与类型提升；TensorIterator 主力 | 高（`i<n`） | 高（整块+尾 mask） | `vload/vstore`、`vloadm/vstorem`、`vadd`/`vmul`/`vdiv`/`vfma`、`vcvt`（提升时） |
| **01.2** 一元算术 | 逐点变换 | `neg`、`abs`、`reciprocal`、`sqrt`、`rsqrt`、`exp`、`log`、`sin`/`cos` | 与 27 有重叠；实现仍按 pointwise | 高 | 高～中（超越函数看 ISA 是否有向量 `vexp`/`vlog`/`vsin`，否则标量近似） | 同 01.1 + `vabs`/`vsqrt`/`vexp`… |
| **01.3** 三元组合 | 乘加类 | `addcmul`、`addcdiv`、`lerp` | 易融合进 10 的 epilogue | 高 | 高 | `vfma`、`vload×3`、`vblend`（lerp） |
| **01.4** Foreach 变体 | 列表逐元素 | `_foreach_add`、`_foreach_mul`、… | 优化器步进；并行提交多 tensor | 高（多 stream/多核） | 高（每 tensor 同 01.1） | 同 01.1；调度非 SIMD |

**实现备注**：热路径 = 整块向量 + 标量/窄向量尾；非 contiguous 先 reorder/coalesce。

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

| 子类 | 典型算子 | 备注 | SIMT 易用 | SIMD 易用 | SIMD 典型指令特征 |
|---|---|---|---|---|---|
| **05.1** 数值规约 | `sum`、`prod`、`mean`、`nansum`、`nanmean`、`norm`、`linalg.vector_norm`、`logsumexp`、`count_nonzero` | keepdim / dim / 多维 reduce | 中（需 warp/block reduce） | 中（水平规约+跨 tile 合并） | `vload`/`vloadm`、`vadd`/`vmul`、`vhadd`/`vreduce_*`；尾 lane 填中性元 |
| **05.2** Arg 规约 | `argmax`、`argmin`、`amax`、`amin`、`aminmax` | 携带 index 或值；尾块禁无效 idx | 中 | 中～低（值+索引对） | `vmax`/`vmin` + index 伴随、`vblend`；禁用无效 lane |
| **05.3** 逻辑规约 | `all`、`any` | bool 中性元不同 | 中 | 高～中 | `vand`/`vor` 规约、`vcmp` |
| **05.4** 统计规约 | `std`、`var`、`median`、`quantile`、`mode` | 有的两遍扫描；`quantile` 更复杂 | 中～低 | 低（`quantile`/`median` 近排序） | 两遍 `vreduce`；或走 25 |
| **05.5** 全局标量 | `trace`（方阵）、无 dim 的 `sum()` | 可看作全维 05.1 | 见组合 | 见组合 | 同 05.1；对角线另要跨步 `vgather` 或标量 |

**子特征备注**：

- **05.a 内维规约**：`sum(-1)`，连续 load 友好 → SIMD 易用性升高  
- **05.b 外/中维**：`sum(0)`，常需换轴或跨步累加 → 倾向 SIMT 或先 `vtranspose`  
- **05.c 短内维**：如 K=3 → SIMD 利用率低；多行打包进向量宽  

---

### Pattern 06 — 扫描与累积

| 典型算子 | 备注 | SIMT 易用 | SIMD 易用 | SIMD 典型指令特征 |
|---|---|---|---|---|
| `cumsum`、`cumprod`、`cummax`、`cummin`、`logcumsumexp` | 沿 dim **有序**；不可随意乱序并行；与 05 不同 | 中（Hillis-Steele / Blelloch） | 中～低（块内 `vprefix`/`vscan` + 块间回写） | `vscan_add`/`vscan_max`、`vshuf`（蝶式）、`vload`/`vstore` |

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

### Pattern 09 — 卷积

| 子类 | 典型算子 | 备注 | SIMT 易用 | SIMD 易用 | SIMD 典型指令特征 |
|---|---|---|---|---|---|
| **09.1** 标准卷积 | `convolution`、`conv1d/2d/3d`、`_convolution` | 统一入口；padding 为 **算法参数**（语义） | 中（手写难打满） | 低（应走 im2col+10 或专用 MMA/Winograd，而非裸 SIMD map） | 若 SIMD 路径则 `vfma` 滑窗或 `vload`→im2col；峰值用 **MMA/Cube** 非通用 SIMD |
| **09.2** 转置/深度可分 | `conv_transpose*`、`_conv_depthwise2d` | 上采样通路；与 23 不同 | 中 | 低～中（depthwise 可通道/空间向量化） | depthwise：`vfma`+`vload`；transpose：`vscatter`/`vatomic_add` 或 col2im |
| **09.3** 后端专用 | `cudnn_convolution`、`miopen_convolution`、`slow_conv*` | 分发到库；用户通常不直接调 | N/A（库） | N/A（库） | 库内自选 / MMA |
| **09.4** im2col | `im2col`、`col2im` | 卷积折叠到 10；调试/后备路径 | 高 | 中 | `vload`/`vstore`、`vgather`（展开）；`col2im` 用 `vscatter`/`vatomic_add` |

**备注**：不宜用通用 01 模型硬写；与 07 的 pad、08 的窗口模式相关但实现独立。

---

### Pattern 10 — 矩阵乘与线性代数

| 子类 | 典型算子 | 备注 | SIMT 易用 | SIMD 易用 | SIMD 典型指令特征 |
|---|---|---|---|---|---|
| **10.1** GEMM 族 | `mm`、`bmm`、`addmm`、`baddbmm`、`addbmm`、`matmul`、`tensordot`、`einsum` | 训练/推理算力主力；epilogue 可融 01/03 | 低（手写 tile） | 低（应 MMA/WMMA/Cube/AMX，外加 epilogue `vfma`/`vadd`） | 通用 SIMD 仅作小规模/后备 |
| **10.2** 向量积 | `dot`、`vdot`、`ger`、`inner`、`outer`、`addr` | 小规模或构建块 | 中 | 高～中 | `vfma` + `vreduce_add`；`outer`/`ger` 用 `vsplat`×`vmul` |
| **10.3** 分解/求解 | `linalg.svd`、`linalg.qr`、`linalg.cholesky`、`linalg.solve`、`triangular_solve`、`lu_*` | 数值库路径 | 低（数值库） | 低（数值库） | 非通用向量 map |
| **10.4** 量化/低比特 GEMM | `_int_mm`、`_dyn_quant_matmul_4bit`、`_weight_int4pack_mm` | 与 19 交界；专用核 | 低 | 低～中（`vpdpbusd`/int8 MMA、unpack） | 整数 MMA、`vcvt`、scale `vmul` |
| **10.5** 分组/稀疏 MM | `_grouped_mm`、`_cslt_sparse_mm`、`_sparse_semi_structured_linear` | 与 20 交界 | 中～低 | 低 | 结构化稀疏 MMA 或 `vgather` |

---

### Pattern 11 — 归一化

| 典型算子 | 备注 | SIMT 易用 | SIMD 易用 | SIMD 典型指令特征 |
|---|---|---|---|---|
| `native_batch_norm`、`batch_norm`、`native_layer_norm`、`layer_norm`、`native_group_norm`、`group_norm`、`instance_norm`、`rms_norm`、`normalize` | 内部 = **05 规约 + 01/04 仿射**；实现几乎总是融合专用核 | 中（两遍/Welford） | 中（内维 `vreduce` + `vsplat` 广播 + `vfma`） | `vload`、`vreduce_add`/`vhadd`、`vsplat`、`vrsqrt`/`vdiv`、`vfma` |

---

### Pattern 12 — Softmax / Attention

| 子类 | 典型算子 | 备注 | SIMT 易用 | SIMD 易用 | SIMD 典型指令特征 |
|---|---|---|---|---|---|
| **12.1** Softmax | `softmax`、`log_softmax`、`_softmax`、`_safe_softmax` | 数值稳定：max-sub → exp → sum → div | 中 | 中 | `vmax`/`vreduce_max`、`vsub`、`vexp`、`vreduce_add`、`vdiv` |
| **12.2** SDPA / Flash | `scaled_dot_product_attention`、`_flash_attention_forward`、`_efficient_attention_forward`、`_cudnn_attention_forward` | IO 感知；动态序列长度敏感 | 低（专用算法） | 低～中（块内 GEMM+在线 softmax） | MMA + 12.1 向量 epilogue；**非**纯 SIMD map |
| **12.3** MHA 包装 | `_native_multi_head_attention` | 组合 10+12+13 | 见组合 | 见组合 | 同 10+12+13 组合 |

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
| **19.4** 量化计算核 | `quantized::conv2d`、`quantized::linear`、`_int_mm` 等 | 常挂独立 namespace / DispatchKey；与 09/10 功能对应的整数实现 | 见组合 | 见组合 | 同 09/10 的 **整数 MMA + scale epilogue**；非通用 SIMD 主路径 |

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

| 子类 | 典型算子 | 备注 | SIMT 易用 | SIMD 易用 | SIMD 典型指令特征 |
|---|---|---|---|---|---|
| **25.1** 排序选择 | `sort`、`argsort`、`topk`、`kthvalue`、`msort` | 比较网络；dim 指定 | 中 | 中～低（bitonic/`vcmp`+`vblend` 网络） | `vcmp`、`vblend`、`vmin`/`vmax`、`vshuf`；大 topk 用堆（标量友好） |
| **25.2** Unique / 成员 | `unique`、`unique_consecutive`、`_unique2`、`isin` | `unique` 输出长度数据依赖 → 亦标 26 | 中 | 低 | 排序后 `vcmp` 邻差 + `vcompress`；或哈希（SIMT 更自然） |

---

### Pattern 26 — 数据依赖动态输出

| 典型算子 | 备注 | SIMT 易用 | SIMD 易用 | SIMD 典型指令特征 |
|---|---|---|---|---|
| `nonzero`、`masked_select`、`unique`（长度）、`syncronized` 类 compact | 官方 tag：`dynamic_output_shape`；实现常 **两阶段 count → write**；meta/导出受限 | 中（compact 扫描） | 中～低 | 阶段1 `vcmp`+`vreduce`/`vprefix` 计数；阶段2 `vcompress`/`vscatter`；无 compress 则 SIMT 更易 |

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
| Linear | `F.linear` | 10 →（可选 01 bias）→（可选 03） |
| Conv-BN-ReLU | CNN 块 | 09 → 11 → 03 |
| LayerNorm | Transformer | 11（内含 05+01） |
| SDPA | Attention | 10 + 12（+ 18 dropout） |
| Embedding + sum | `embedding_bag` | 13 → 05 |
| Pad + Pack | RNN batch | 07 ↔ 24 |
| QDQ + Conv | PTQ/QAT | 19 → 09/19.4 |
| 通信–计算重叠 | async allreduce + compute + wait | 10/01 ∥ **29.5** → **29.5 wait** |
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
| 统计口径变化 | 更新 §1.1 |

---

*本文以核心功能 + 数字 Pattern 组织 PyTorch 三千量级底层算子，并提取同步算子（Pattern 29）的阻塞域与序关系特征，供算子库与 NPU 中间层对照使用。*
