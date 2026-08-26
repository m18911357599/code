# 常用网络算子：调用次数、耗时与汇总

> 对齐 [PyTorch 算子 Pattern 分类](pytorch_op_feature_compete.md)（01–29；Cube = 09/10/12.2）。  
> 对 9 个常用网络做 **ATen 叶子调用次数**、**排他 CPU self-time**、**Cube FLOPs（MAC×2）** 统计，并对照公开 GPU/加速器文献。  
> 复现：`python tools/net_op_profile/profile_common_nets.py`；明细见 `tools/net_op_profile/results/`。

---

## 0. 结论摘要

| 命题 | 证据 |
|---|---|
| **FLOPs 几乎全是 Cube** | 除检测后处理外，Cube（卷积 / GEMM / 融合 MHA）占 **95–99.9% FLOPs** |
| **次数远不等于耗时** | Cube 只占 **约 27–36% 叶子次数**，却占 **75–94% CPU 时间**；布局/激活次数多、时间少 |
| **CNN 热路径 = Conv+BN** | ResNet-50：`conv2d` 53 次 / 78.6% 时间 / 99.3% FLOPs；BN 同样 53 次但仅 10% 时间 |
| **Transformer 热路径 = Linear/GEMM** | BERT/GPT prefill：`linear` 73 次约 87% 时间；`matmul` 24 次仅 3%；Softmax/GELU/LN 合计约 5% |
| **Decode ≠ Prefill** | GPT-2 同结构 97 次 GEMM，prefill 22.5 GFLOPs vs decode 0.25 GFLOPs（**约 89×**）；CPU 墙钟只降约 **3×**（带宽/启动开销） |
| **检测被后处理淹没** | SSDLite：1866 次叶子调用，卷积仅 98 次（5.3%），却贡献 95.8% FLOPs；NMS/index/where/topk 合计 **>50% CPU 时间** |
| **融合改变“看起来的”次数** | ViT 的 QKV+Attn 收成 12 次 `_native_multi_head_attention`；BERT 显式图是 73×Linear + 24×matmul + 12×softmax |
| **加速器风险（Amdahl）** | Cube 变快后，Vec（LN/Softmax/GELU/Add）与 **13/25 后处理** 的时间份额会上升；文献中 BERT 非 GEMM 可达 30–40%（训练，低精度更明显） |

**覆盖优先级（按跨网络热度）：**

| 优先级 | Pattern | 角色 |
|---|---|---|
| P0 | **09 卷积 / 10 GEMM / 12.2 SDPA** | 几乎全部 FLOPs |
| P0 | **11 归一化 / 03 激活 / 01 Add** | 次数高、易成融合 epilogue |
| P1 | **12.1 Softmax / 14 布局 / 15 cat** | Transformer / U-Net / decode KV |
| P1 | **13 gather / 25 TopK·NMS / 02 where** | 检测、Embedding、动态索引 |
| P2 | **08 池化 / 07 mask / 18 Dropout / 23 上采样** | 场景性 |

---

## 1. 范围与方法

### 1.1 网络清单（推理，batch=1，随机权重，结构与量级同预训练）

| 网络 | 族 | 输入 | 代表场景 |
|---|---|---|---|
| ResNet-50 | CNN | 1×3×224×224 | ImageNet 分类 |
| MobileNetV3-Large | 轻量 CNN | 1×3×224×224 | 端侧分类 |
| ViT-B/16 | Vision Transformer | 1×3×224×224 | 视觉 Transformer（融合 MHA） |
| BERT-Base | Encoder | seq=128 | NLP 理解；**显式** QKV+matmul+softmax |
| GPT-2-Small-prefill | Decoder | seq=128 | 生成预填充 |
| GPT-2-Small-decode | Decoder | 1 token，KV=127 | 自回归一步 |
| SSDLite320 | 检测 | 3×320×320 | 骨干 + 多尺度头 + NMS |
| U-Net | 分割 | 1×3×128×128 | 编码-解码 + skip |
| LSTM-2L | RNN | seq=128，hidden=512 | 融合 `aten::lstm` |

BERT/GPT 不用 HuggingFace 权重，按 BERT-Base / GPT-2 Small 超参手写，避免下载；**调用次数由结构决定，与权重无关**。

### 1.2 三个口径（不要混比）

| 口径 | 含义 | 跨设备稳定性 |
|---|---|---|
| **叶子调用次数** | `TorchDispatchMode` 拦截 ATen；同 Pattern 的包装子调用不重复计（避免 `conv2d→convolution→mkldnn` 计 3 次） | **高**（结构不变量） |
| **FLOPs** | 卷积/GEMM/SDPA/融合 MHA/LSTM 按 MAC×2 | **高** |
| **CPU self-time %** | 本机 PyTorch 2.13 CPU、4 线程、oneDNN | **低**（GPU/NPU 上 Cube 更快，Vec/后处理份额上升） |

墙钟含 dispatcher 开销，只作相对参考；分网络 ATen 明细以 `tools/net_op_profile/results/tables.md` 为准（与 `summary.json` 同一次跑）。GPU 份额引用 §5 文献，不把 CPU 毫秒当器件性能。

Pattern 编号与 Cube/Vec 划分见 `pytorch_op_feature_compete.md`。

---

## 2. 跨网络总表

实测环境：PyTorch `2.13.0+cpu` / torchvision `0.28.0+cpu` / 4 线程 / `eval` + `inference_mode`。

| 网络 | 叶子次数 | ATen 种类 | 墙钟 ms | 估算 FLOPs | Cube 次数% | Cube 耗时% | Cube FLOPs% |
|---|---:|---:|---:|---:|---:|---:|---:|
| ResNet-50 | 175 | 8 | 26.2 | 8.23e9 | 30.9 | 81.3 | 99.3 |
| MobileNetV3-Large | 187 | 12 | 11.0 | 4.56e8 | 34.2 | 76.6 | 95.0 |
| ViT-B/16 | 142 | 12 | 57.5 | 3.52e10 | 26.8 | 94.5 | 99.9 |
| BERT-Base | 312 | 16 | 45.3 | 2.24e10 | 31.1 | 91.2 | 99.8 |
| GPT-2-Small-prefill | 345 | 18 | 49.9 | 2.25e10 | 28.1 | 89.6 | 99.8 |
| GPT-2-Small-decode | 357 | 18 | 16.2 | 2.52e8 | 27.2 | 78.8 | 99.9 |
| SSDLite320 | **1866** | **46** | 64.1 | 1.22e9 | **5.3** | **29.9** | 95.8 |
| U-Net | 63 | 5 | 14.1 | 6.86e9 | 36.5 | 91.1 | 99.7 |
| LSTM-2L | 3 | 2 | 4.7 | 1.07e9 | 33.3 | 99.3 | 100 |

读表：

1. **分类/分割/语言模型**：种类少（5–18），图干净；检测种类 46、次数 1866，后处理把图打散。  
2. **LSTM 次数=1 是假象**：`aten::lstm` 把 128 步×2 层×4 门收成一次启动。  
3. **MobileNet FLOPs 只有 ResNet 的 ~1/18**，但叶子次数相当（187 vs 175）——轻量网更吃带宽与小核启动。  
4. **GPT decode FLOPs 约为 prefill 的 1/89**，叶子次数几乎相同（357 vs 345）——decode 是 **同等次数的瘦 GEMM + KV cat**。

### 2.1 Pattern 耗时份额（CPU %）

| 网络 | 09 Conv | 10 GEMM | 12.2 MHA | 11 Norm | 03 Act | 12.1 Softmax | 14 View | 13 Index | 25 Sort/NMS | 其它要点 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| ResNet-50 | **78.6** | 0.8 | — | 10.2 | 2.4 | — | — | — | — | 池化 6.2 |
| MobileNetV3 | **73.9** | 1.5 | — | 11.5 | 6.1 | — | — | — | — | h-swish/SE |
| ViT-B/16 | 0.9 | **56.7** | **36.8** | 2.0 | 2.0 | — | 0.1 | — | — | Cube 合计 94.3 |
| BERT-Base | — | **90.6** | — | 1.5 | 2.3 | 1.0 | 1.9 | 0.1 | — | Linear 为主 |
| GPT prefill | — | **89.6** | — | 1.5 | 2.1 | 0.9 | 1.4 | 0.1 | — | 因果 mask 1.9 |
| GPT decode | — | **78.0** | — | 2.4 | 4.1 | 0.9 | 3.1 | 0.2 | — | **cat 5.0** |
| SSDLite | **29.8** | — | — | 4.7 | 2.1 | 0.5 | 1.9 | **20.5** | **18.4** | where 等 15.0 |
| U-Net | **91.8** | — | — | 4.5 | 1.9 | — | — | — | — | 反卷积含在 09；cat 1.7 |
| LSTM-2L | — | **99.3** | — | — | — | — | — | — | — | 融合核 |

### 2.2 Pattern 叶子调用次数

| 网络 | 09 | 10 | 12.2 | 11 | 03 | 12.1 | 14 | 01 | 13 | 25 | 15 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ResNet-50 | 53 | 1 | — | 53 | 49 | — | 1 | 16 | — | — | — |
| MobileNetV3 | 62 | 2 | — | 46 | 48 | — | 1 | 18 | — | — | — |
| ViT-B/16 | 1 | 25 | 12 | 25 | 12 | — | 3 | 25 | — | — | 1 |
| BERT-Base | — | 97 | — | 25 | 13 | 12 | **122** | 38 | 3 | — | — |
| GPT prefill | — | 97 | — | 25 | 12 | 12 | **122** | 37 | 2 | — | — |
| GPT decode | — | 97 | — | 25 | 12 | 12 | 110 | 37 | 2 | — | **24** |
| SSDLite | 98 | — | — | 70 | 71 | 1 | 176 | 80 | **635** | **182** | 26 |
| U-Net | 23 | — | — | 18 | 18 | — | — | — | — | — | 4 |
| LSTM-2L | — | 1 | — | — | — | — | — | — | — | — | — |

**次数冠军不是 Cube：** BERT/GPT 里 Pattern **14 布局** 占叶子次数 35–39%，耗时 <2%。检测里 **13 索引 635 次**、比较 281 次、TopK/NMS 182 次。

---

## 3. 分网络说明

### 3.1 ResNet-50 — 教科书 CNN

结构不变量（与实测一致）：

| 算子 | 期望次数 | 实测 | Pattern |
|---|---:|---:|---|
| `conv2d` | 53 | 53 | 09 |
| `batch_norm` | 53 | 53 | 11 |
| `relu_` | 49 | 49 | 03 |
| `add_`（残差） | 16 | 16 | 01 |
| `max_pool2d` + `adaptive_avg_pool2d` | 1+1 | 1+1 | 08 |
| `linear`（FC） | 1 | 1 | 10 |

FLOPs 8.23e9，其中卷积 99.3%。BN 与 Conv **等次数**，时间约 1:8。推理期 BN 可折入卷积（文献中融合后算术强度约从 67 提到 121），折后 Pattern 11 次数可降到 0。

### 3.2 MobileNetV3-Large — 小核、多次、低 FLOPs

62 次卷积（含 depthwise + 1×1）、46 次 BN、h-swish 21 + ReLU 11+8 + h-sigmoid 8。FLOPs 仅 0.46e9，CPU 上 Conv 仍 74%，但 BN+激活合计 **17.6%**，高于 ResNet 的 12.6%。Depthwise 算术强度低，NPU 上更容易从 Cube-bound 变成 **搬运/Vec-bound**。

### 3.3 ViT-B/16 vs BERT-Base — 融合改变次数账本

| 项 | ViT-B/16（融合 MHA） | BERT-Base seq=128（显式图） |
|---|---|---|
| 叶子次数 | 142 | 312 |
| `linear` | 25（MLP×2×12 + head 等） | **73**（每层 Q/K/V/O/FFN1/FFN2 + pooler） |
| Attention 核 | 12× `_native_multi_head_attention` | 24× `matmul` + 12× `softmax` |
| `layer_norm` | 25 | 25 |
| `gelu` | 12 | 12 |
| Cube 时间 | 56.7% Linear + 36.8% MHA = **94.3%** | Linear+matmul = **90.6%** |
| FLOPs | 35.2e9 | 22.4e9 |

BERT 每层 6 次 Linear + 2 次 act-to-act matmul，12 层 → 72+24=96，加 pooler → **97**，与实测一致。`linear` 占 87.3% 时间，`matmul`（QKᵀ/PV）只 3.4%——seq=128 时投影 GEMM 比 attention 矩阵更大。序列变长后 24 次 `matmul` 的 FLOPs 按 \(O(S^2)\) 涨，份额会上升。

布局：BERT `view` 48 + `transpose` 60 + `contiguous` 12 = **122**，全是为了把 QKV 切成 head。融合 MHA / SDPA 后这些次数可抹掉（ViT 仅 3 次布局）。

### 3.4 GPT-2-Small：prefill vs decode

同一套 12 层、768 维、73×Linear + 24×matmul + 12×softmax：

| | Prefill S=128 | Decode S_q=1, KV=127 | 比 |
|---|---:|---:|---|
| 叶子次数 | 345 | 357 | ≈1× |
| GEMM 次数 | 97 | 97 | 1× |
| FLOPs | 2.25e10 | 2.52e8 | **89×** |
| 墙钟 | 50.5 ms | 16.7 ms | 3.0× |
| Cube 时间% | 89.6 | 78.0 | decode Vec/搬运更显眼 |
| 多出来的算子 | 因果 `triu`+`masked_fill` | **`cat`×24（KV cache）** | decode 特有 Pattern 15 |

Decode 的 Linear 是 **GEMV**（M=1），算术强度随 hidden 变，不随 seq 变（权重每步重读）。文献中 GPT 解码常为 **memory-bound**；本 CPU 账上 Cube 仍 78%，是因为 oneDNN GEMV 仍贵，且 KV 很短。更长上下文时 `cat`/KV 读写与 softmax 的 \(O(T)\) 会继续抬高非 Cube 份额。

LM head（`linear` vocab=50257）在 decode 一步里仍是大矩阵-向量乘，计入 73 次 Linear 之一。

### 3.5 SSDLite320 — 次数爆炸在后处理

骨干仍是卷积（98 次，95.8% FLOPs），但 **eval 完整前向含 NMS**：

| 算子 | 次数 | CPU 时间% | Pattern |
|---|---:|---:|---|
| `conv2d` | 98 | 29.8 | 09 |
| `index` | 545 | 19.5 | 13 |
| `nms` | 90 | 13.5 | 25 |
| `where` / `eq` / `gt` | 91+90+90 | ~15 | 02 |
| `topk` | 90 | 3.9 | 25 |
| `batch_norm` | 70 | 4.7 | 11 |

叶子 1866 次、46 种 ATen。这是检测/分割头 + NMS 的典型形态：**FLOPs 看骨干，时延看后处理**（尤其小 batch、CPU 或弱 Cube 时）。YOLO 系同类：decode 框、TopK、NMS、gather。

### 3.6 U-Net — 反卷积与 skip

23 次卷积类（19×`conv2d` + 4×`conv_transpose2d`），18×BN，18×ReLU，4×`cat`。反卷积 4 次却占 **12.3% 时间**（相对次数被放大）。Skip `cat` 次数少、带宽集中。分割网优化要同时看 09.2 转置卷积与 Pattern 15。

### 3.7 LSTM-2L — 融合掩盖次数

`aten::lstm` **1 次**启动，内部是 2 层 × 128 步 × 4 门 GEMM（估算 1.07e9 FLOPs）。若按未融合展开，Linear/add/sigmoid/tanh 次数为 \(O(\text{layers} \times S)\)。RNN 报“调用次数”必须声明是否融合；NPU 上常见选择是保持融合核，或展开成 Cube+Vec 流水。

---

## 4. 汇总：谁该先做、做多深

### 4.1 按 FLOPs（器件算力规划）

几乎所有常用网：**>95% FLOPs ∈ {09 Conv, 10 GEMM, 12.2 融合 Attention}**。  
Cube 单元、tiling、FIXPIPE/epilogue 融合决定峰值。Vec 不决定 FLOPs 账，但决定 **达不到峰值的那一截**。

### 4.2 按调用次数（启动、图编译、融合机会）

| 场景 | 高频 Pattern | 含义 |
|---|---|---|
| CNN | 09/11/03 接近 1:1:1 | Conv-BN-ReLU 应成固定融合组 |
| Transformer 显式图 | 14 布局 > 10 GEMM > 01 Add | 先消 view/transpose，再切 Cube |
| Transformer 融合图 | 10 MLP Linear + 12.2 MHA | 两次 Cube 家族启动/层 |
| 生成 decode | 10 次数不变 + 15 cat | KV 增量写，忌每步全量 copy |
| 检测 | 13/02/25 ≫ 09 | 后处理要单独内核/排序原语（Pattern 25 的 `vsort`） |

### 4.3 按 CPU 时间（本机；GPU/NPU 需再加权）

| 族 | 时间结构 |
|---|---|
| CNN / U-Net | Conv 74–92%，其余 BN/激活/池化 |
| Encoder / Prefill / ViT | GEMM/MHA 90–94%，LN+GELU+Softmax+Add ≈ 5–8% |
| Decode | GEMM 仍最大，但 cat/GELU/LN/布局合计 >20% |
| 检测 | Conv 不到 1/3，index+NMS+比较过半 |

Cube 在 GPU/NPU 上加速比通常高于 LN/Softmax/NMS。用 Amdahl：若器件把 Cube 做快 \(k\) 倍、Vec 不变，则非 Cube 份额 \(\approx \frac{1-c}{c/k + (1-c)}\)。以 BERT CPU 的 \(c=0.91\) 为例，Cube 再快 10× 后非 Cube 时间份额从 9% 升到约 **50%**。这与文献中「CNN 加速器跑 BERT 时非 matmul 占墙钟」一致。

### 4.4 与 Pattern 文档的对应

| 实测热点 | Pattern 文档中的位置 | 硬件含义 |
|---|---|---|
| `conv2d` / `conv_transpose2d` | 09 Cube | MMA/im2col |
| `linear` / `matmul` / `lstm` | 10 Cube | 投影 vs act-to-act 两种形状 |
| `_native_multi_head_attention` | 12.2 Cube | 内含 matmul + softmax epilogue |
| `softmax` | 12.1 Vec | 要 `vreduce_max`+`vreduce_add` |
| `layer_norm` / `batch_norm` | 11 | BN 可折入；LN 要运行时 reduce |
| `gelu` / `relu` / `hardswish` | 03 | 作 Cube epilogue |
| `add` 残差 | 01 | 随路 add |
| `embedding` / `index` / `cat` | 13 / 15 | gather、KV 拼接 |
| `topk` / `nms` / `_unique2` | 25 | `vsort`/`vmergesort` 投资点 |
| `view`/`transpose` | 14 | 编译期消掉，不应下到核 |

---

## 5. 文献中的器件时间（对照，非本次实测）

CPU 份额不能当 GPU/NPU 份额。下列为公开剖析的量级：

| 来源 | 负载 | 要点 |
|---|---|---|
| Kim et al., *Full Stack Optimization of Transformer Inference*, 2023 | BERT / GPT-2 / ResNet-50 | Transformer **~99% FLOPs 是 matmul**；非线性 FLOPs 可忽略但 MOPs 可观。ResNet 非线性同样低 FLOPs、高 MOP；**BN 可折入卷积、ReLU 可随路**，算术强度 66.9→121.4。CNN 加速器跑 BERT 时，非 matmul + 量化往返可把利用率打到极低。 |
| Ham et al., *Demystifying BERT*, 2021 | BERT 训练（GPU） | GEMM 仍最大，但 **data-intensive 约 34% 时间**；FP32 下 Linear+FC 约 57%，混合精度降到约 40%（GEMM 受益更大）。Attention 的 BGEMM 形状更瘦，不一定喂饱阵列。 |
| Zafiri et al., EuroMLSys 2021 | BERT **推理、CPU** | matmul（Linear+bmm）**66–91%** 时间；线程越多、序列越短，LN/Softmax/transpose 份额越高（库内 GEMM 更可扩展）。 |
| 常识 / 多篇 decode 剖析 | GPT decode | 逐步 GEMV + 重读权重，**memory-bound**；KV cache 带宽随上下文线性涨。 |

与本次 CPU 推理的关系：BERT/GPT prefill 的 90% Cube 时间落在文献区间上沿（单线程级 GEMM 更重、未开 CUDA Core）。检测后处理、decode 的 cat、ViT 融合 MHA，文献较少按 ATen 次数拆，本次补的是 **次数账**。

---

## 6. 方法边界

- 无 GPU：时间份额仅 CPU。次数与 FLOPs 可直接用于 NPU 规划。  
- BERT/GPT 为结构等价实现，非 HuggingFace 逐算子对齐（如 bias、type embedding 细节）。次数与官方层公式一致。  
- `eval` 下 Dropout 仍可能作为空算子出现（ViT 37 次），时间可忽略。  
- SSDLite 的 NMS 次数随候选框走，随机权重下框分布与预训练不同，**后处理次数是量级示意**。  
- 未覆盖：扩散 U-Net 大分辨率、推荐 DLRM（Embedding 主导）、训练反传（约 2× GEMM + 优化器）、分布式通信（Pattern 29）。

复现：

```bash
python tools/net_op_profile/profile_common_nets.py
python tools/net_op_profile/profile_common_nets.py --only ResNet-50 BERT-Base
```

---

## 7. 维护

| 变更 | 动作 |
|---|---|
| 新增常用网 | 在 `tools/net_op_profile/models.py` 加 `Workload`，重跑脚本，更新 §2 总表 |
| ATen 改名 / 新融合核 | 更新 `pattern_map.py`（如本次 `_native_multi_head_attention` → 12.2） |
| 要 GPU 份额 | 同脚本可扩 `ProfilerActivity.CUDA`；次数表不必重做 |
| Pattern 编号变更 | 与 `pytorch_op_feature_compete.md` 同步 |
