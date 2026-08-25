# 微架构易用性度量

> 整改思路：① 先固定 **26 个 Pattern**，每个 Pattern 在 **7 个度量项**下各有难度权重 $w_{c,j}$，据此度量 **该 Pattern 的易用性** $\eta_j$；② 再单独定义 **7 个度量项的权重 $\Omega_c$** 与 **每项怎么度量**（难/中/易的判定方法），据此度量 **该度量项的易用性** $\eta_c$。  
> 难度档：$w\in\{5,3,1\}$（难 / 中 / 易）。第 6、7 列 Excel 尚未打分，公式保留，计算时只对已填列求和。

---

## 0. 两层输出

| 层 | 对象 | 输入 | 输出 |
|---|---|---|---|
| **Pattern 层** | 26 个算子范式 | 该行的 7 个 $w_{c,j}$ | Pattern 易用性 $\eta_j$ |
| **度量项层** | 7 个能力域 | 该列的 26 个 $w_{c,j}$，以及该项的度量方法 | 度量项权重 $\Omega_c$、度量项易用性 $\eta_c$ |

矩阵 $W=(w_{c,j})$ 只打一次。行聚合得 Pattern 画像，列聚合得度量项画像。等权时 $\frac{1}{26}\sum_j\eta_j=\frac{1}{C}\sum_c\eta_c$（$C$ 为已填列数）。

---

## 1. 26 个 Pattern：各自易用性

### 1.1 Pattern 清单

每个 Pattern 选一个代表算子（Excel「算子名」列）。`补！` 表示套件里还缺独立样本。

| $j$ | Pattern | 代表算子 |
|---|---|---|
| 1 | Elementwise | silu / gelu / add |
| 2 | Broadcast | bias_add（可由 Norm 覆盖） |
| 3 | Reduction | sum / mean / max（可由 Norm 覆盖） |
| 4 | Contraction | Matmul, QuantMatmul |
| 5 | ArgReduce | argmax（补！） |
| 6 | Layout Transform | ViewCopy |
| 7 | Padding | pad（补！） |
| 8 | IndexGather | GatherV2 |
| 9 | ScatterUpdate | scatter（可由 ViewCopy 覆盖） |
| 10 | AtomicUpdate | scatter_add（可由 ViewCopy 覆盖） |
| 11 | Interpolation | ResizeBilinearV2（补！） |
| 12 | MaskPredicate | where / masked_fill |
| 13 | SortSelect | topk / sort |
| 14 | Histogram | bincount / histc |
| 15 | SlidingWindow | Conv2dV2 |
| 16 | ControlFlow | cond / while / ctcloss（补！） |
| 17 | Recurrence | cumsum（补！） / lstm |
| 18 | MegaKernel | MegaMoe |
| 19 | Spectral | fft / dct |
| 20 | DynamicShape / VariableOutput | 动态 batch、nonzero / unique |
| 21 | RandomSampling | dropout / multinomial |
| 22 | Norm | RmsNorm, GroupNormGrad |
| 23 | Sparse | sparse.mm |
| 24 | Quantization | DynamicMxQuant |
| 25 | CollectiveCommunication | dispatch + combine |
| 26 | FusedComposite | QSMLA, QLI |

### 1.2 单元格怎么打分

对固定 Pattern $j$、固定度量项 $c$，只打一档：

| 档 | $w_{c,j}$ | 含义 |
|---|---|---|
| 易 | 1 | 该项上几乎无额外软件代价，或硬件/编译器已免除 |
| 中 | 3 | 有可管理的软件代价（改参数、少量分支、单次同步） |
| 难 | 5 | 显著软件代价（该项度量方法中的「难」判定成立） |

判定规则按度量项写在 **§2.2**，不在 Pattern 侧另发明一套。同一单元格禁止同时用两套标准。

### 1.3 权重矩阵 $w_{c,j}$（Excel 现状）

列：1 系统规格 · 2 内存模型 · 3 计算模型 · 4 控制模型 · 5 访存模型 · 6 调试调优 · 7 Agent 友好。

| $j$ | Pattern | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
|---|---|---|---|---|---|---|---|---|
| 1 | Elementwise | 1 | 1 | 1 | 1 | 1 | — | — |
| 2 | Broadcast | 1 | 1 | 1 | 1 | 5 | — | — |
| 3 | Reduction | 3 | 5 | 3 | 1 | 5 | — | — |
| 4 | Contraction | 3 | 5 | 3 | 3 | 5 | — | — |
| 5 | ArgReduce | 3 | 5 | 5 | 5 | 5 | — | — |
| 6 | Layout Transform | 5 | 3 | 3 | 1 | 5 | — | — |
| 7 | Padding | 5 | 5 | 5 | 5 | 5 | — | — |
| 8 | IndexGather | 3 | 3 | 3 | 1 | 3 | — | — |
| 9 | ScatterUpdate | 3 | 3 | 3 | 1 | 3 | — | — |
| 10 | AtomicUpdate | 5 | 1 | 1 | 5 | 1 | — | — |
| 11 | Interpolation | 3 | 3 | 5 | 1 | 5 | — | — |
| 12 | MaskPredicate | 1 | 1 | 5 | 5 | 1 | — | — |
| 13 | SortSelect | 5 | 5 | 5 | 5 | 5 | — | — |
| 14 | Histogram | 3 | 3 | 3 | 1 | 3 | — | — |
| 15 | SlidingWindow | 5 | 5 | 5 | 5 | 5 | — | — |
| 16 | ControlFlow | 1 | 1 | 1 | 3 | 1 | — | — |
| 17 | Recurrence | 3 | 5 | 5 | 5 | 5 | — | — |
| 18 | MegaKernel | 1 | 5 | 3 | 5 | 3 | — | — |
| 19 | Spectral | 5 | 5 | 5 | 5 | 5 | — | — |
| 20 | DynamicShape | 5 | 5 | 5 | 3 | 5 | — | — |
| 21 | RandomSampling | 1 | 1 | 5 | 1 | 1 | — | — |
| 22 | Norm | 3 | 5 | 5 | 5 | 5 | — | — |
| 23 | Sparse | 5 | 5 | 5 | 5 | 5 | — | — |
| 24 | Quantization | 1 | 1 | 5 | 3 | 3 | — | — |
| 25 | Collective | 3 | 1 | 1 | 5 | 3 | — | — |
| 26 | FusedComposite | 5 | 5 | 5 | 5 | 5 | — | — |
| | **列和 $\Sigma_j w_{c,j}$** | **82** | **88** | **96** | **86** | **98** | 待填 | 待填 |

### 1.4 Pattern 易用性 $\eta_j$

已填度量项集合为 $\mathcal{C}$（当前 $\mathcal{C}=\{1,2,3,4,5\}$，$C=|\mathcal{C}|=5$）。

单格免除率（越大越好）：

$$
u_{c,j}=\frac{5-w_{c,j}}{4}\in\{0,\ 0.5,\ 1\}
$$

**等权 Pattern 易用性**（默认，不依赖 $\Omega_c$）：

$$
\eta_j=\frac{1}{C}\sum_{c\in\mathcal{C}} u_{c,j}
=\frac{5C-\sum_{c\in\mathcal{C}} w_{c,j}}{4C}
$$

当前：$C=5$，故 $\eta_j=(25-\sum_{c=1}^{5}w_{c,j})/20$。全易 → $100\%$；全难 → $0\%$。

**加权 Pattern 易用性**（启用 §2.1 的 $\Omega_c$ 时）：

$$
\eta_j^{\Omega}=\frac{\sum_{c\in\mathcal{C}}\Omega_c\,u_{c,j}}{\sum_{c\in\mathcal{C}}\Omega_c}
$$

等权 $\Omega$ 时 $\eta_j^{\Omega}=\eta_j$。第 6、7 列补齐后 $C\leftarrow 7$，分母改为 $4\times 7=28$，旧 $\eta_j$ 作废重算。

### 1.5 26 个 Pattern 实测（等权，$\mathcal{C}=\{1..5\}$）

| $j$ | Pattern | $\sum_{c=1}^{5}w_{c,j}$ | 难列数 | $\eta_j$ |
|---|---|---|---|---|
| 1 | Elementwise | 5 | 0 | **100%** |
| 16 | ControlFlow | 7 | 0 | **90%** |
| 2 | Broadcast | 9 | 1 | **80%** |
| 21 | RandomSampling | 9 | 1 | **80%** |
| 8 | IndexGather | 13 | 0 | **60%** |
| 9 | ScatterUpdate | 13 | 0 | **60%** |
| 10 | AtomicUpdate | 13 | 2 | **60%** |
| 12 | MaskPredicate | 13 | 2 | **60%** |
| 14 | Histogram | 13 | 0 | **60%** |
| 24 | Quantization | 13 | 1 | **60%** |
| 25 | Collective | 13 | 1 | **60%** |
| 3 | Reduction | 17 | 2 | **40%** |
| 6 | Layout Transform | 17 | 2 | **40%** |
| 11 | Interpolation | 17 | 2 | **40%** |
| 18 | MegaKernel | 17 | 2 | **40%** |
| 4 | Contraction | 19 | 2 | **30%** |
| 5 | ArgReduce | 23 | 4 | **10%** |
| 17 | Recurrence | 23 | 4 | **10%** |
| 20 | DynamicShape | 23 | 4 | **10%** |
| 22 | Norm | 23 | 4 | **10%** |
| 7 | Padding | 25 | 5 | **0%** |
| 13 | SortSelect | 25 | 5 | **0%** |
| 15 | SlidingWindow | 25 | 5 | **0%** |
| 19 | Spectral | 25 | 5 | **0%** |
| 23 | Sparse | 25 | 5 | **0%** |
| 26 | FusedComposite | 25 | 5 | **0%** |
| | **平均** $\bar\eta=\frac{1}{26}\sum\eta_j$ | 450 / 26 | — | **38.46%** |

优先改 $\eta_j=0$ 的 6 个 Pattern（五行全难），其次 $\eta_j=10\%$ 的 ArgReduce / Recurrence / DynamicShape / Norm。

---

## 2. 7 个度量项：权重与度量方法

### 2.1 度量项权重 $\Omega_c$

$\Omega_c$ 回答「这一项在综合分里占多重」，**必须独立于矩阵打分**。禁止再用列和 $\Sigma_j w_{c,j}$ 当 $\Omega_c$：列和已经是「该项有多难」，再拿来当权重会把已很难的列再放大一次。

| 规则 | 取值 |
|---|---|
| 未启用的列 | $\Omega_c=0$（当前 $c=6,7$） |
| 已启用列的默认 | 等权：$\Omega_c=1/C$（当前 $C=5$，故 $\Omega_{1..5}=0.20$） |
| 改权重 | 只改先验表并 bump `omega_ver`，不得用当次 $\Sigma w$ 反推 |

业务若要强调访存，可把先验改成例如 $(0.15,0.15,0.15,0.15,0.40,0,0)$，在 `omega_ver` 里写死。本文实例用默认等权。

7 列都打分后：默认 $\Omega_c=1/7$，先前 5 列等权结果作废重算。

### 2.2 每个度量项怎么度量（难 / 中 / 易）

对 **某一个 Pattern 的代表算子**，按下列操作定义打 $w_{c,j}$。先看该项「难」是否成立，再看「中」，否则为「易」。

#### $c=1$ 系统规格

**度量什么**：带宽、对齐、Tile、核数等规格，是否迫使该 Pattern 多切搬运。

| 档 | 判定 |
|---|---|
| 易(1) | 一次搬运即可打到目标带宽；无额外切分 |
| 中(3) | 只需改对齐 / BurstLen / Tile，不必拆成两次业务搬运 |
| 难(5) | **跳搬运中**：必须 2 次搬运切分、2 次搬运 |

**做法**：对该 Pattern 做访存 μ-bench（copy / 该算子热路径），扫描是否出现第二次 DMA/copy 切分；对照手册对齐与 Burst 约束。

#### $c=2$ 内存模型

**度量什么**：地址空间与访问是否连续；离散访问是否要软件处理。

| 档 | 判定 |
|---|---|
| 易(1) | 连续访问，硬件视图一致 |
| 中(3) | 规则跨步 / 有限 bank 约束，可用公式表达 |
| 难(5) | **离散**：间接、随机、多视图别名，必须软件拼地址或回避 bank |

**做法**：把热路径访存标成连续 / 跨步 / 间接；间接或必须手写离散重排则打难。

#### $c=3$ 计算模型

**度量什么**：该 Pattern 的计算是否有原语，还是靠偏移计算拼出来。

| 档 | 判定 |
|---|---|
| 易(1) | ISA / intrinsic 直接覆盖主计算 |
| 中(3) | 用现有算术拼，无需复杂索引场 |
| 难(5) | **偏移计算**：主代价在 index / 水平树 / 软件模拟原语 |

**做法**：对照指令清单。缺 `vreduce`/`vsort`/`vscan` 等而手写树或标量尾，打难。

#### $c=4$ 控制模型

**度量什么**：流水、同步、生产-消费是否要手写。

| 档 | 判定 |
|---|---|
| 易(1) | 单流水，无显式多引擎同步 |
| 中(3) | 少量同步或单一控制流（如 cond） |
| 难(5) | **多流水**：必须手写多队列 / SetFlag-WaitFlag / 核间握手 |

**做法**：数该代表实现里以建立序为目的的同步语句；≥2 条独立流水且手写依赖则打难。

#### $c=5$ 访存模型

**度量什么**：buffer 管理、布局/指令变形是否落在用户代码。

| 档 | 判定 |
|---|---|
| 易(1) | 搬移与布局由硬件或 API 隐式完成 |
| 中(3) | 通过参数选择布局 / 一次 format 转换 |
| 难(5) | **内存管理 + 指令变形**：手管多级 buffer，或 nz/pad/swizzle 必须手写 |

**做法**：统计仅为搬移、分配、转置、padding 而存在的语句；有多级显式 workspace 则打难。

#### $c=6$ 调试调优（列待填）

**度量什么**：该 Pattern 出问题后能否看见、复现、归因。

| 档 | 判定 |
|---|---|
| 易(1) | 关键事件可采集，同输入可复现，工具能指到源 |
| 中(3) | 有计数器但 Stall / 首因不全 |
| 难(5) | 静默失败、不可复现、或无法定位到本 Pattern 代码 |

**做法**：对该代表算子跑一次：PMU/trace 是否覆盖热路径；故意注入错精度或漏同步，看能否归因。26 个 $w_{6,j}$ 填完后再算 $\eta_6$ 与各 $\eta_j$。

#### $c=7$ Agent 友好（列待填）

**度量什么**：Agent 能否按规格生成正确实现，以及要改多少。

| 档 | 判定 |
|---|---|
| 易(1) | 一次生成，正确性 oracle 通过 |
| 中(3) | 需少量改写（参数、一处同步）后通过 |
| 难(5) | 生成失败，或必须重写热路径 |

**做法**：冻结提示词与 oracle，对 26 个代表算子各跑固定次数，按成功率与改写行数套上表。

### 2.3 度量项易用性 $\eta_c$

$$
\eta_c=\frac{1}{N}\sum_{j=1}^{N}u_{c,j}
=\frac{5N-\sum_{j=1}^{N}w_{c,j}}{4N}
=\frac{130-\Sigma_j w_{c,j}}{104},\quad N=26
$$

语义：该项上 26 个 Pattern 的平均免除率。全易 → $100\%$；全难 → $0\%$。

### 2.4 五项实测（$N=26$）

| $c$ | 度量项 | $\Sigma_j w_{c,j}$ | 难/中/易 | $\eta_c$ | 默认 $\Omega_c$ |
|---|---|---|---|---|---|
| 1 | 系统规格 | 82 | 9 / 10 / 7 | 46.15% | 0.20 |
| 2 | 内存模型 | 88 | 13 / 5 / 8 | 40.38% | 0.20 |
| 3 | 计算模型 | 96 | 14 / 7 / 5 | 32.69% | 0.20 |
| 4 | 控制模型 | 86 | 13 / 4 / 9 | 42.31% | 0.20 |
| 5 | 访存模型 | 98 | 15 / 6 / 5 | 30.77% | 0.20 |
| 6 | 调试调优 | 待填 | — | — | 0 |
| 7 | Agent 友好 | 待填 | — | — | 0 |

访存（30.77%）与计算（32.69%）是当前度量项短板。

---

## 3. 综合分

等权、已填 5 列时，行平均与列平均相同：

$$
U=\frac{1}{26}\sum_{j=1}^{26}\eta_j
=\frac{1}{5}\sum_{c=1}^{5}\eta_c
=38.46\%
$$

启用先验 $\Omega$ 后只走度量项侧：

$$
U^{\Omega}=\sum_{c\in\mathcal{C}}\Omega_c\,\eta_c
$$

（此时一般 $U^{\Omega}\ne\bar\eta_j$，这是刻意的：综合分改跟「项有多重要」，Pattern 表仍报 $\eta_j$。）

等级只是对 $U$ 的注释，**不替换**线性 $U$：

| $U$ | $[90,100]$ | $[80,90)$ | $[70,80)$ | $[60,70)$ | $[0,60)$ |
|---|---|---|---|---|---|
| 档 | S | A | B | C | D |

当前 $U=38.46\%$，档 D。不要把未过 $50\%$ 的 $\eta$ 先映射成 0 再加权——那样五项全变成 0，Pattern 之间的 100%～0% 差异会被抹掉。

---

## 4. 补列与重算

1. 按 §2.2 给 26 个 Pattern 填 $w_{6,j}$、$w_{7,j}$。  
2. $C\leftarrow 7$，重算全部 $\eta_j$（分母 28）与 $\eta_6,\eta_7$。  
3. 默认 $\Omega_c=1/7$，或写入新的 `omega_ver`。  
4. 重算 $U$ 与 $U^{\Omega}$。  
5. 矩阵 $W$ 的打分规则版本与 $\Omega$ 版本分开记；改判定表必须重打受影响列。
