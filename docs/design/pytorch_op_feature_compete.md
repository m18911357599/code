# PyTorch 内置底层算子：按核心功能分类、API 特征与竞品分析

> 范围：以 ATen `native_functions.yaml` 及量化/稀疏等命名空间为对象，覆盖业界常称的 **约三千量级** 内置底层算子（见 §1.1）。  
> 本文按 **核心功能** 重新分类；Pattern 统一用 **两位数字编号**；每类给出典型算子与备注。  
> 对照维度：API 形态、动态 shape / 非对齐 / padding，以及 CUDA / Triton / AscendC / SIMD-Tile(128B) 竞分。

---

## 0. 结论摘要

| 维度 | 结论 |
|---|---|
| **分类主轴** | 按 **核心功能** 分为 **Pattern 01–28**（逐元素、规约、填充、量化、卷积、矩阵乘等），不以字母代号 |
| **规模** | schema 约 **2500+**，唯一基名约 **1500+**；计入 inplace / out / 重载 / quantized·sparse·foreach 后常称 **~3000–3500** |
| **实现主路径** | **01–06、14–16** 多走 TensorIterator；**07–12、19、22–23** 多为固定维/领域核；**25–26** 含数据依赖动态输出 |
| **API** | 用户声明式 API → Schema/Dispatch → Plan/Iterator → Backend；竞分关键在中间层是否吸收动态与非对齐 |
| **Dyn / Align / Pad** | 语义 Pad（07）≠ 向量尾填充 ≠ 硬件对齐 Pad；短尾轴在 NPU 上易被隐式 pad |
| **选型** | 通用 01/05/14：**跟 PyTorch 语义**；09/10/12 峰值：**领域库**；NPU：**上层 PT 友好 + 下层能力显式** |

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

后文 API / 动态 shape / 竞分均引用这些数字编号。

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

| Pattern | 核心功能 | 定义（一句话） | 规模感（基名量级） | 典型访存/并行 |
|---|---|---|---|---|
| **01** | 逐元素算术 | 同址（广播后）二元/一元数值运算 | 很大 | 带宽；完全并行 |
| **02** | 比较 / 逻辑 / 选择 | 比较、位运算、`where` 类 | 大 | 同 01 |
| **03** | 激活与非线性 | 逐点非线性；常可与 01/09/10 融合 | 中 | 同 01 |
| **04** | 广播与维扩展 | 逻辑扩展 shape，默认不拷贝 | 中 | 0-stride / 物化 |
| **05** | 规约 | 多→少；沿 dim 聚合 | 大 | 树归约 / 分块 |
| **06** | 扫描与累积 | 有序前缀依赖 | 小 | 难乱序 |
| **07** | 填充 | 边界/长度语义 Pad | 中 | 搬移 + 边界写 |
| **08** | 池化 | 滑窗降采样 / 反池化 | 中 | 固定维；空间并行 |
| **09** | 卷积 | 局部加权；含转置卷积 | 中 | 高算术密度 |
| **10** | 矩阵乘与线性代数 | GEMM / 分解 / 求解 | 大 | MMA / 库 |
| **11** | 归一化 | BN / LN / GN / RMSNorm 等 | 中 | 规约+广播融合 |
| **12** | Softmax / Attention | Softmax、SDPA、MHA | 中 | 融合 IO 感知 |
| **13** | 索引 / 散射 / Embedding | 间接读写 | 大 | gather/scatter |
| **14** | 布局变换 | view / transpose / reshape 等 | 很大 | 元数据或搬移 |
| **15** | 拼接与分割 | cat / stack / split | 中 | 分段 copy |
| **16** | 拷贝与类型转换 | clone / to / cast | 中 | 带宽 |
| **17** | 工厂与创建 | 无输入或仅 shape 造张量 | 中 | 写填充 |
| **18** | 随机与 Dropout | 采样、噪声、dropout | 中 | RNG + 逐点 |
| **19** | 量化 | quantize / dequant / fake_quant | 中 | 量纲变换+整数核 |
| **20** | 稀疏 | COO/CSR 等稀疏格式与算子 | 中 | 间接、压缩 |
| **21** | FFT / 信号 | 频域变换 | 小–中 | 专用库 |
| **22** | 损失函数 | 训练目标 | 中 | 常含规约 |
| **23** | 上采样与插值 | upsample / grid_sample | 中 | 固定维插值 |
| **24** | Nested / Jagged | 变长序列结构 | 小–中 | 偏移+填充交互 |
| **25** | 排序 / TopK / Unique | 比较网络、选择 | 小–中 | 不规则 |
| **26** | 数据依赖动态输出 | 输出 shape 依赖 **数值** | 小 | 两阶段 |
| **27** | 特殊函数 | `special.*`、高阶数学 | 中 | 多为 01 变体 |
| **28** | 元信息 / 控制 / 辅助 | size、device、assert、autograd 钩子 | 中 | 非计算主路径 |

复合模块（`nn.Linear`、`nn.MultiheadAttention`）由上表 Pattern **组合** 而成，不单开编号。

---

## 3. 各类详解：子类、典型算子、备注

### Pattern 01 — 逐元素算术

| 子类 | 含义 | 典型算子 | 备注 |
|---|---|---|---|
| **01.1** 二元算术 | `+/-/×/÷` 等 | `add`、`sub`、`mul`、`div`、`remainder`、`pow`、`atan2`、`hypot` | 默认支持广播与类型提升；TensorIterator 主力 |
| **01.2** 一元算术 | 逐点变换 | `neg`、`abs`、`reciprocal`、`sqrt`、`rsqrt`、`exp`、`log`、`sin`/`cos` | 与 27 有重叠；实现仍按 pointwise |
| **01.3** 三元组合 | 乘加类 | `addcmul`、`addcdiv`、`lerp` | 易融合进 10 的 epilogue |
| **01.4** Foreach 变体 | 列表逐元素 | `_foreach_add`、`_foreach_mul`、… | 优化器步进；并行提交多 tensor |

**实现备注**：热路径 = 整块向量 + 标量/窄向量尾；非 contiguous 先 reorder/coalesce。

---

### Pattern 02 — 比较 / 逻辑 / 选择

| 子类 | 典型算子 | 备注 |
|---|---|---|
| **02.1** 比较 | `eq`、`ne`、`lt`、`le`、`gt`、`ge`、`isnan`、`isinf`、`isfinite` | 输出常为 bool |
| **02.2** 位/逻辑 | `bitwise_and`/`or`/`xor`、`logical_and`、`__lshift__` | 整数/bool 路径 |
| **02.3** 选择 | `where`、`clamp`、`clamp_min`/`max`、`nan_to_num`、`maximum`、`minimum` | `where` = 谓词 blend；尾块需 mask |

---

### Pattern 03 — 激活与非线性

| 子类 | 典型算子 | 备注 |
|---|---|---|
| **03.1** 经典激活 | `relu`、`leaky_relu`、`elu`、`selu`、`celu`、`threshold` | 常与 09/10 融合（Conv-ReLU） |
| **03.2** 现代激活 | `gelu`、`silu`/`swish`、`mish`、`hardswish`、`hardsigmoid`、`hardtanh` | Transformer / CNN 高频 |
| **03.3** 饱和/收缩 | `sigmoid`、`tanh`、`softplus`、`softshrink`、`hardshrink`、`glu` | Softmax 前常用 sigmoid/tanh |
| **03.4** 带参数 | `prelu`、`rrelu` | 额外权重；非纯 01 |

---

### Pattern 04 — 广播与维扩展

| 子类 | 典型算子 | 备注 |
|---|---|---|
| **04.1** 逻辑广播 | `expand`、`expand_as`、`broadcast_to`、`broadcast_tensors`、`broadcast_shapes` | **默认不物化**；stride=0 |
| **04.2** 重复物化 | `repeat`、`tile`、`repeat_interleave` | 真正拷贝；与 04.1 成本不同 |
| **04.3** 隐式广播 | （无独立 API） | 发生在 01/02/03 的 TI build 中 |

**备注**：竞分时常把 04 从 01 拆出——后端是否支持 0-stride 决定要不要先 materialize。

---

### Pattern 05 — 规约

| 子类 | 典型算子 | 备注 |
|---|---|---|
| **05.1** 数值规约 | `sum`、`prod`、`mean`、`nansum`、`nanmean`、`norm`、`linalg.vector_norm`、`logsumexp`、`count_nonzero` | keepdim / dim / 多维 reduce |
| **05.2** Arg 规约 | `argmax`、`argmin`、`amax`、`amin`、`aminmax` | 携带 index 或值；尾块禁无效 idx |
| **05.3** 逻辑规约 | `all`、`any` | bool 中性元不同 |
| **05.4** 统计规约 | `std`、`var`、`median`、`quantile`、`mode` | 有的两遍扫描；`quantile` 更复杂 |
| **05.5** 全局标量 | `trace`（方阵）、无 dim 的 `sum()` | 可看作全维 05.1 |

**子特征备注**：

- **05.a 内维规约**：`sum(-1)`，连续 load 友好  
- **05.b 外/中维**：`sum(0)`，常需换轴或跨步累加  
- **05.c 短内维**：如 K=3，向量利用率低；宜多行打包，忌硬件盲目 pad  

---

### Pattern 06 — 扫描与累积

| 典型算子 | 备注 |
|---|---|
| `cumsum`、`cumprod`、`cummax`、`cummin`、`logcumsumexp` | 沿 dim **有序**；不可随意乱序并行；与 05 不同 |

---

### Pattern 07 — 填充（Pad）

| 子类 | 典型算子 | 备注 |
|---|---|---|
| **07.1** 常值填充 | `constant_pad_nd`、`F.pad(..., mode='constant')`、`zero_`、`fill_`、`masked_fill` | **语义 Pad**：改变有效区域外的值 |
| **07.2** 边界反射/复制 | `reflection_pad1d/2d/3d`、`replication_pad*`、`circular_pad*` | 卷积/分割任务常用 |
| **07.3** 序列填充 | `nn.utils.rnn.pad_sequence`、`pad_packed_sequence`、`_pad_packed_sequence` | 与 24 Nested 交互 |
| **07.4** 矩阵三角填充 | `tril`、`triu`、`tril_`、`triu_` | 按三角掩码写 |

**重要区分（竞分常用）**：

| 名称 | 是否改数值语义 | 例 |
|---|---|---|
| 语义 Pad（本 Pattern） | 是 | `F.pad`、conv 的 padding 参数 |
| 广播拉伸 | 否（逻辑） | 04 + 01 |
| 向量尾填充 | 否（写回 mask） | 实现细节 |
| **硬件对齐 Pad** | 否，但耗带宽 | NPU UB 尾轴 pad 到 32B/512B |

---

### Pattern 08 — 池化

| 子类 | 典型算子 | 备注 |
|---|---|---|
| **08.1** Max/Avg/Lp | `max_pool2d`、`avg_pool2d`、`lp_pool2d`、`*_pool1d/3d` | 固定 1–3D；含 indices 的 max_pool |
| **08.2** Adaptive | `adaptive_avg_pool2d`、`adaptive_max_pool2d` | 输出尺寸指定；实现与普通池化不同 |
| **08.3** 反池化 | `max_unpool2d` | 依赖 08.1 的 indices |
| **08.4** Fractional | `fractional_max_pool2d` | 随机/分数步长 |

---

### Pattern 09 — 卷积

| 子类 | 典型算子 | 备注 |
|---|---|---|
| **09.1** 标准卷积 | `convolution`、`conv1d/2d/3d`、`_convolution` | 统一入口；padding 为 **算法参数**（语义） |
| **09.2** 转置/深度可分 | `conv_transpose*`、`_conv_depthwise2d` | 上采样通路；与 23 不同 |
| **09.3** 后端专用 | `cudnn_convolution`、`miopen_convolution`、`slow_conv*` | 分发到库；用户通常不直接调 |
| **09.4** im2col | `im2col`、`col2im` | 卷积折叠到 10；调试/后备路径 |

**备注**：不宜用通用 01 模型硬写；与 07 的 pad、08 的窗口模式相关但实现独立。

---

### Pattern 10 — 矩阵乘与线性代数

| 子类 | 典型算子 | 备注 |
|---|---|---|
| **10.1** GEMM 族 | `mm`、`bmm`、`addmm`、`baddbmm`、`addbmm`、`matmul`、`tensordot`、`einsum` | 训练/推理算力主力；epilogue 可融 01/03 |
| **10.2** 向量积 | `dot`、`vdot`、`ger`、`inner`、`outer`、`addr` | 小规模或构建块 |
| **10.3** 分解/求解 | `linalg.svd`、`linalg.qr`、`linalg.cholesky`、`linalg.solve`、`triangular_solve`、`lu_*` | 数值库路径 |
| **10.4** 量化/低比特 GEMM | `_int_mm`、`_dyn_quant_matmul_4bit`、`_weight_int4pack_mm` | 与 19 交界；专用核 |
| **10.5** 分组/稀疏 MM | `_grouped_mm`、`_cslt_sparse_mm`、`_sparse_semi_structured_linear` | 与 20 交界 |

---

### Pattern 11 — 归一化

| 典型算子 | 备注 |
|---|---|
| `native_batch_norm`、`batch_norm`、`native_layer_norm`、`layer_norm`、`native_group_norm`、`group_norm`、`instance_norm`、`rms_norm`、`normalize` | 内部 = **05 规约 + 01/04 仿射**；实现几乎总是融合专用核 |

---

### Pattern 12 — Softmax / Attention

| 子类 | 典型算子 | 备注 |
|---|---|---|
| **12.1** Softmax | `softmax`、`log_softmax`、`_softmax`、`_safe_softmax` | 数值稳定：max-sub → exp → sum → div |
| **12.2** SDPA / Flash | `scaled_dot_product_attention`、`_flash_attention_forward`、`_efficient_attention_forward`、`_cudnn_attention_forward` | IO 感知；动态序列长度敏感 |
| **12.3** MHA 包装 | `_native_multi_head_attention` | 组合 10+12+13 |

---

### Pattern 13 — 索引 / 散射 / Embedding

| 子类 | 典型算子 | 备注 |
|---|---|---|
| **13.1** Gather | `gather`、`index_select`、`take`、`take_along_dim`、`index` | 读间接；难向量化 |
| **13.2** Scatter | `scatter`、`scatter_add`、`scatter_reduce`、`index_add`、`index_put`、`put_` | 写冲突；需原子或确定性策略 |
| **13.3** Masked | `masked_select`、`masked_scatter`、`masked_fill` | `masked_select` → 亦属 26 |
| **13.4** Embedding | `embedding`、`embedding_bag`、`_embedding_bag` | 查表 + 可选 05；稀疏梯度 |
| **13.5** 搜索桶 | `bucketize`、`searchsorted`、`one_hot` | 半有序索引 |

---

### Pattern 14 — 布局变换

| 子类 | 典型算子 | 备注 |
|---|---|---|
| **14.1** 元数据 View | `view`、`reshape`（可 view 时）、`expand`、`transpose`/`permute`（可 view）、`squeeze`/`unsqueeze`、`as_strided`、`select`、`narrow`、`diagonal` | `inplace_view` / view 语义；尽量零拷贝 |
| **14.2** 物化转置/重排 | `contiguous`、`permute` 打断时、`transpose` 拷贝路径、`movedim` | 分块转置；边缘 gather |
| **14.3** 展平/折叠 | `flatten`、`unflatten`、`unfold` | `unfold` 有重叠窗，偏 08/09 预备 |
| **14.4** view_copy 族 | `*_copy`（如 `transpose_copy`、`permute_copy`） | 功能化 IR / Export 用 |

---

### Pattern 15 — 拼接与分割

| 子类 | 典型算子 | 备注 |
|---|---|---|
| **15.1** 拼接 | `cat`、`concat`、`stack`、`hstack`/`vstack`/`dstack`、`column_stack` | **分段 copy**；每段独立尾块 |
| **15.2** 分割 | `split`、`split_with_sizes`、`chunk`、`tensor_split`、`unbind`、`hsplit`/`vsplit` | 多为 view 或分段 view |
| **15.3** 块对角等 | `block_diag` | 隐式零填充区域 ↔ 与 07 相关 |

---

### Pattern 16 — 拷贝与类型转换

| 子类 | 典型算子 | 备注 |
|---|---|---|
| **16.1** 拷贝 | `clone`、`copy_`、`_to_copy`、`_copy_from` | 保 dtype 或随 `to` |
| **16.2** dtype/device | `to`、`type_as`、`_autocast_to_*_precision`、历史 `_cast_*` | 涉及拷贝+转换 |
| **16.3** 别名 | `alias`、`detach`、`detach_` | 元数据；非数据面 |

---

### Pattern 17 — 工厂与创建

| 子类 | 典型算子 | 备注 |
|---|---|---|
| **17.1** 空/常量 | `empty`、`zeros`、`ones`、`full`、`empty_like`、`zeros_like` | 无 tensor 输入或 like |
| **17.2** 序列 | `arange`、`linspace`、`logspace`、`eye`、`range`（legacy） | 索引生成常用 |
| **17.3** 窗函数 | `bartlett_window`、`hann_window`、`hamming_window`、`kaiser_window` | 与 21 配合 |
| **17.4** 量化工厂 | `_empty_affine_quantized`、`_empty_per_channel_affine_quantized` | 归属 17 创建、服务 19 |

---

### Pattern 18 — 随机与 Dropout

| 子类 | 典型算子 | 备注 |
|---|---|---|
| **18.1** 分布采样 | `rand`、`randn`、`randint`、`normal`、`uniform_`、`bernoulli`、`poisson`、`multinomial` | `Generator` 控制；nondeterministic 标签 |
| **18.2** Dropout | `dropout`、`native_dropout`、`alpha_dropout`、`feature_dropout` | 训练图高频；与 01 融合可能 |

---

### Pattern 19 — 量化

| 子类 | 典型算子 | 备注 |
|---|---|---|
| **19.1** 量/反量 | `quantize_per_tensor`、`quantize_per_channel`、`dequantize` | 进出整型表示 |
| **19.2** Fake quant | `fake_quantize_per_tensor_affine`、`_fake_quantize_learnable_*`、`choose_qparams_*` | QAT；可反传 |
| **19.3** Q 张量元数据 | `q_scale`、`q_zero_point`、`q_per_channel_scales`、`int_repr` | 读量化参数 |
| **19.4** 量化计算核 | `quantized::conv2d`、`quantized::linear`、`_int_mm` 等 | 常挂独立 namespace / DispatchKey；与 09/10 功能对应的整数实现 |

**备注**：功能上「量化」既是 **dtype/数值域变换（19.1–19.3）**，也是 **同构算子的整数后端（19.4）**。

---

### Pattern 20 — 稀疏

| 典型算子 | 备注 |
|---|---|
| `to_sparse`、`to_dense`、`sparse_coo_tensor`、`_to_sparse_csr/csc/bsr/bsc`、稀疏 `mm`/`add`、`_sparse_semi_structured_*` | shape 规则含 sparse/dense 维；与 10/13 交界 |

---

### Pattern 21 — FFT / 信号

| 典型算子 | 备注 |
|---|---|
| `_fft_c2c`、`_fft_r2c`、`_fft_c2r`、`fft.*`、`stft`、`istft`、`fftfreq` | 多调 MKL / cuFFT / pocketfft；计划缓存（`_cufft_*`）属辅助 |

---

### Pattern 22 — 损失函数

| 典型算子 | 备注 |
|---|---|
| `nll_loss`、`cross_entropy`、`mse_loss`、`l1_loss`、`smooth_l1_loss`、`binary_cross_entropy`、`kl_div`、`ctc_loss`、`triplet_margin_loss`、`hinge_embedding_loss` | 内部常 **01/05/13 组合**；CTC 等更专用 |

---

### Pattern 23 — 上采样与插值

| 典型算子 | 备注 |
|---|---|
| `upsample_nearest2d`、`upsample_bilinear2d`、`upsample_bicubic2d`、`upsample_trilinear3d`、`_upsample_*_aa`、`grid_sampler_2d/3d`、`affine_grid` | 固定维；与 09.2 转置卷积不同路径 |

---

### Pattern 24 — Nested / Jagged

| 典型算子 | 备注 |
|---|---|
| `_nested_from_padded`、`_nested_tensor_*`、`_jagged_to_padded_dense_forward`、`_pack_padded_sequence` | **变长** 与 07 填充互转；动态 shape 友好表示 |

---

### Pattern 25 — 排序 / TopK / Unique

| 子类 | 典型算子 | 备注 |
|---|---|---|
| **25.1** 排序选择 | `sort`、`argsort`、`topk`、`kthvalue`、`msort` | 比较网络；dim 指定 |
| **25.2** Unique / 成员 | `unique`、`unique_consecutive`、`_unique2`、`isin` | `unique` 输出长度数据依赖 → 亦标 26 |

---

### Pattern 26 — 数据依赖动态输出

| 典型算子 | 备注 |
|---|---|
| `nonzero`、`masked_select`、`unique`（长度）、`syncronized` 类 compact | 官方 tag：`dynamic_output_shape`；实现常 **两阶段 count → write**；meta/导出受限 |

---

### Pattern 27 — 特殊函数

| 典型算子 | 备注 |
|---|---|
| `special.erf`、`erfc`、`erfinv`、`lgamma`、`digamma`、`polygamma`、`i0`、`sinc`、`xlogy` | 数学特殊函数；实现形态同 01，单独成类便于库映射 |

---

### Pattern 28 — 元信息 / 控制 / 辅助

| 子类 | 典型算子 | 备注 |
|---|---|---|
| **28.1** 查询 | `size`、`stride`、`dim`、`numel`、`is_floating_point`、`device`、`layout` | 非数据面 |
| **28.2** 断言/调试 | `_assert_async`、`_assert_tensor_metadata`、`_foobar` | 图安全/测试 |
| **28.3** Autograd 辅助 | `_backward`、`requires_grad_`、AMP `_amp_*` | 训练基础设施 |

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

### 5.2 API 形态 × Pattern

| 特征 | 强相关 Pattern | 备注 |
|---|---|---|
| 隐式广播 | 01–03、04 | 用户不写 expand |
| dim/keepdim | 05、06、11、25 | 负维规范化 |
| padding 参数 | 07、09、08 | 语义 pad |
| scale/zero_point | 19 | 量化契约 |
| Generator | 18 | 随机可复现 |
| 动态输出 | 26、25.2、13.3 | 导出需小心 |

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

### 6.4 推荐契约

```text
1. API 不要求用户保证 nbytes % VL == 0
2. 07 语义 Pad 与硬件对齐 Pad 必须在文档/下层分开
3. 05.c / 短尾轴：优先 layout 变换，再考虑硬件 pad
4. 26：显式两阶段，不伪装成 01
5. 09/10/12：领域 API；只复用分块与尾块约定
```

---

## 7. 竞品分析（竞分）

### 7.1 对象

| 对象 | 定位 |
|---|---|
| PyTorch ATen + TensorIterator | 功能覆盖与语义参照 |
| CUDA 手写 | 性能与控制力上限 |
| Triton | Tile + mask DSL |
| AscendC | NPU 显式存储/向量/Cube |
| SIMD-Tile (128B) | 逻辑 128B 量子的可移植模型 |

### 7.2 分项（1–5）

| 分项 | 含义 |
|---|---|
| C1 | Pattern 01–28 表达覆盖 |
| C2 | API 心智负担 |
| C3 | S1 动态长度 |
| C4 | U1/U2 尾块与对齐 |
| C5 | Pad 可控（尤其反硬件隐式 pad） |
| C6 | 性能表达力 |
| C7 | 与 PyTorch 功能语义对齐 |

### 7.3 总评

| 分项 | PyTorch | CUDA 手写 | Triton | AscendC | SIMD-Tile |
|---|---|---|---|---|---|
| C1 | **5** | 5 | 3–4 | 4 | 4（09/10/12/26 走专用） |
| C2 | **5** | 2 | 3–4 | 2–3 | **4–5** |
| C3 | **5** | 4 | 3–4 | 3 | **4–5** |
| C4 | **5** | 3 | 4 | 2–3 | **4–5** |
| C5 | **5** | 4 | 3–4 | 2–3 | 4 |
| C6 | 3–4 | **5** | 4 | **5** | 3 |
| C7 | **5** | 2 | 3 | 2–3 | **4–5** |

### 7.4 分 Pattern 竞分要点

| Pattern | 最易用 | 性能常归属 | 备注 |
|---|---|---|---|
| 01–03 | PyTorch / SIMD-Tile | CUDA / AscendC 管道 | Triton 任意 rank/stride 弱 |
| 05 | PyTorch | CUB / AscendC Reduce | 05.c 三者都要技巧 |
| 07 | PyTorch | 带宽 | 与硬件 pad 勿混 |
| 08–09、23 | 库 API | cuDNN / Cube | 固定维友好 |
| 10 | **库** | cuBLAS / Cube | 禁止用 01 硬写打满 |
| 12 | 专用 API | Flash / 厂商库 | 比领域 API，不比裸循环 |
| 13 | 接近 | 原子/冲突硬件 | DSL 表达接近 |
| 15 | PyTorch | memcpy | 分段尾块 |
| 19 | PyTorch 量化栈 | oneDNN / 厂商 Q 核 | 19.4 强绑定后端 |
| 24 | PyTorch Nested | 定制 | 与 07 互转成本 |
| 26 | PyTorch 两阶段 | 手写 compact | Triton/AscendC 均别扭 |

### 7.5 同一用例

**用例 A**：`a+b`，长度 `N` 动态且 `N % lanes != 0`（Pattern **01**）

| 栈 | 尾块策略 |
|---|---|
| PyTorch | 向量主循环 + 标量尾 |
| CUDA | `i < N` 或向量+边界 |
| Triton | tile + `mask` |
| AscendC | repeat/mask；注意 UB 对齐 |
| SIMD-Tile | 满 128B 无谓词 + **至多一次** 尾块 |

**用例 B**：`sum(dim=-1)`，`K=3`（Pattern **05**，05.c）

| 栈 | 风险 | 较优 |
|---|---|---|
| AscendC | 尾轴硬件 pad 膨胀 | 借轴/转置 |
| SIMD-Tile / PT | 利用率低 | 多行打包 |

### 7.6 落地建议

1. 中间层按 **Pattern 01–28** 建路由，而不是按产品名 `#ifdef`。  
2. **01/04/05/14/15/16** 对齐 TensorIterator 语义；**09/10/12/21** 对齐领域库。  
3. **07 vs 硬件 pad**、**19 量化**、**26 DynOut** 是 NPU 竞分差异最大的三类，需单独能力开关与测试集。  
4. SIMD-Tile 适合作为 01–06、14–16 的编写契约；与本文数字 Pattern 一一对应即可落地。

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

### 8.2 参考

- `aten/src/ATen/native/native_functions.yaml`、`tags.yaml`  
- TensorIterator 文档与实现  
- ezyang: *A brief taxonomy of PyTorch operators by shape behavior*（按 shape 行为的另一正交分类）  
- PyTorch Quantization / Sparse / NestedTensor 文档  
- Triton、AscendC Basic API、SIMD-Tile(128B) 设计  

### 8.3 维护

| 变更 | 动作 |
|---|---|
| 新增功能族 | 追加 Pattern 29+；不复用已有数字 |
| 算子迁类 | 改 §3 典型列表与备注，更新交叉表 §4 |
| 统计口径变化 | 更新 §1.1 |

---

*本文以核心功能 + 数字 Pattern 重新组织 PyTorch 三千量级底层算子，并保留 API、动态形状与竞分结论，供算子库与 NPU 中间层对照使用。*
