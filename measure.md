# 微架构易用性度量

## 0. 体系总览

### 0.1 三层结构

| 层级 | 含义 | 示例 |
|---|---|---|
| **度量项** | 顶层能力域，对齐 Excel 7 列 | 系统规格易用性、内存模型易用性、计算模型易用性、控制模型易用性、访存模型易用性、调试调优易用性、Agent 友好易用性 |
| **Pattern 维度** | 26 个静态算子范式，按难/中/易加权 | Contraction、ArgReduce、SlidingWindow、FusedComposite … |
| **度量指标** | Pattern 加权后的易用性免除率 $\eta_c \in [0,100\%]$ | 系统规格 Pattern 免除率、内存模型 Pattern 免除率 … |

### 0.2 度量分类（对齐 Excel 7 列）

| 分类 $c$ | 度量项（Excel 列） | Pattern 权重和 $\Sigma w_c$ | 难(5)/中(3)/易(1) 分布 | 基础权重 $\Omega_c$ |
|---|---|---|---|---|
| 1 | 系统规格易用性 | $82$ | 9 / 10 / 7 | $0.18$ |
| 2 | 内存模型易用性 | $88$ | 13 / 5 / 8 | $0.20$ |
| 3 | 计算模型易用性 | $96$ | 14 / 7 / 5 | $0.22$ |
| 4 | 控制模型易用性 | $86$ | 13 / 4 / 9 | $0.19$ |
| 5 | 访存模型易用性 | $98$ | 15 / 6 / 5 | $0.21$ |
| 6 | 调试调优易用性 | 待填 | — | $0.00$（占位） |
| 7 | Agent 友好易用性 | 待填 | — | $0.00$（占位） |

> $\Sigma w_c$ 越大表示该度量项下 Pattern 总难度越高、易用性越低。基础权重 $\Omega_c$ 按 $\Sigma w_c$ 归一化（5 个有数据列之和 $=450$，第 6、7 列暂无数据，权重置 0 占位）。
> 待 Excel 第 6、7 列补齐后，$\Omega_c$ 按 $\Sigma w_c / \sum_{c'} \Sigma w_{c'}$ 重新归一化即可，公式体系不变。

---

## 1. Pattern 权重与微架构易用性

### 1.1 26 个 Pattern 难度权重

> **权重定义**（来源 Excel R30–R33）：
> - **难 = 5**：该 Pattern 在该度量项下需付出显著软件代价（如跳搬运中、偏移计算、多流水、内存管理/指令变形等）。
> - **中 = 3**：需付出中等软件代价。
> - **易 = 1**：基本无软件代价或硬件全自动。

| # | Pattern 范式 | 算子名 | 1 系统规格 | 2 内存模型 | 3 计算模型 | 4 控制模型 | 5 访存模型 | 6 调试调优 | 7 Agent 友好 |
|---|---|---|---|---|---|---|---|---|---|
| 1 | Elementwise · silu/gelu/add | NA, 太简单 | 易(1) | 易(1) | 易(1) | 易(1) | 易(1) | — | — |
| 2 | Broadcast · bias_add | NA, Norm 包含 | 易(1) | 易(1) | 易(1) | 易(1) | 难(5) | — | — |
| 3 | Reduction · sum/mean/max | NA, Norm 包含 | 中(3) | 难(5) | 中(3) | 易(1) | 难(5) | — | — |
| 4 | Contraction · matmul/linear | Matmul, QuantMatmul | 中(3) | 难(5) | 中(3) | 中(3) | 难(5) | — | — |
| 5 | ArgReduce · argmax/argmin | 补！argmax | 中(3) | 难(5) | 难(5) | 难(5) | 难(5) | — | — |
| 6 | Layout Transform · View | ViewCopy | 难(5) | 中(3) | 中(3) | 易(1) | 难(5) | — | — |
| 7 | Padding · pad | 补！padding | 难(5) | 难(5) | 难(5) | 难(5) | 难(5) | — | — |
| 8 | IndexGather · gather/embedding | GatherV2 | 中(3) | 中(3) | 中(3) | 易(1) | 中(3) | — | — |
| 9 | ScatterUpdate · scatter | NA, ViewCopy 包含 | 中(3) | 中(3) | 中(3) | 易(1) | 中(3) | — | — |
| 10 | AtomicUpdate · scatter_add | NA, ViewCopy 包含 | 难(5) | 易(1) | 易(1) | 难(5) | 易(1) | — | — |
| 11 | Interpolation · grid_sample/roi_align | 补！ResizeBilinearV2 | 中(3) | 中(3) | 难(5) | 易(1) | 难(5) | — | — |
| 12 | MaskPredicate · where/masked_fill | NA, NPU 不太关注 | 易(1) | 易(1) | 难(5) | 难(5) | 易(1) | — | — |
| 13 | SortSelect · topk/sort | NA, (QLI 覆盖) | 难(5) | 难(5) | 难(5) | 难(5) | 难(5) | — | — |
| 14 | Histogram · bincount/histc | NA, NPU 不太关注 | 中(3) | 中(3) | 中(3) | 易(1) | 中(3) | — | — |
| 15 | SlidingWindow · conv2d/pool | Conv2dV2 | 难(5) | 难(5) | 难(5) | 难(5) | 难(5) | — | — |
| 16 | ControlFlow · cond/while_loop/hash | 补！ctcloss | 易(1) | 易(1) | 易(1) | 中(3) | 易(1) | — | — |
| 17 | Recurrence · cumsum/lstm | 补！cumsum | 中(3) | 难(5) | 难(5) | 难(5) | 难(5) | — | — |
| 18 | MegaKernel(片段) PTO | MegaMoe | 易(1) | 难(5) | 中(3) | 难(5) | 中(3) | — | — |
| 19 | Spectral · fft/dct | NA, NPU 不太关注 | 难(5) | 难(5) | 难(5) | 难(5) | 难(5) | — | — |
| 20 | DynamicShape · 动态 batch/序列；VariableOutput · nonzero/unique | NA, 非本次验证重点 | 难(5) | 难(5) | 难(5) | 中(3) | 难(5) | — | — |
| 21 | RandomSampling · dropout/multinomial | NA, 非本次验证重点 | 易(1) | 易(1) | 难(5) | 易(1) | 易(1) | — | — |
| 22 | Norm 类 | RmsNorm, GroupNormGrad | 中(3) | 难(5) | 难(5) | 难(5) | 难(5) | — | — |
| 23 | Sparse · sparse.mm | NA, 非本次验证重点 | 难(5) | 难(5) | 难(5) | 难(5) | 难(5) | — | — |
| 24 | Quantization · int8 GEMM/fp8 | DynamicMxQuant | 易(1) | 易(1) | 难(5) | 中(3) | 中(3) | — | — |
| 25 | CollectiveCommunication · all_reduce/all_gather | dispatch+combine | 中(3) | 易(1) | 易(1) | 难(5) | 中(3) | — | — |
| 26 | FusedComposite · softmax/flash_attention | QSMLA, QLI | 难(5) | 难(5) | 难(5) | 难(5) | 难(5) | — | — |
| — | **列权重和 $\Sigma w_c$** | | **82** | **88** | **96** | **86** | **98** | 待填 | 待填 |
| — | **难/中/易 分布** | | 9/10/7 | 13/5/8 | 14/7/5 | 13/4/9 | 15/6/5 | — | — |

> **难代价说明**（Excel R30 备注）：
> - 系统规格「难」= 跳搬运中（需要 2 次搬运切分、2 次搬运）。
> - 内存模型「难」= 离散的难。
> - 计算模型「难」= 偏移计算。
> - 控制模型「难」= 多流水。
> - 访存模型「难」= 内存管理，指令变形。
> - 「连续的易」：连续访问为易，离散访问为难。

### 1.2 单度量项 Pattern 易用性（线性免除率）

设度量项 $c$ 下第 $j$ 个 Pattern 的难度权重为 $w_{c,j} \in \{5,3,1\}$（难=5、中=3、易=1），共 $N=26$ 个 Pattern。定义难度上下界：

$$
D_{\max} = N \cdot 5 = 130 \quad (\text{全难上界}), \qquad
D_{\min} = N \cdot 1 = 26 \quad (\text{全易下界})
$$

则度量项 $c$ 的 **Pattern 易用性免除率** $\eta_c \in [0, 100\%]$：

$$
\boxed{\;
\eta_c = \frac{D_{\max} - \sum_{j=1}^{N} w_{c,j}}{D_{\max} - D_{\min}} \times 100\%
       = \frac{130 - \Sigma w_c}{104} \times 100\%
\;}
$$

> **语义**：$\Sigma w_c$ 越大（难 Pattern 越多）→ $\eta_c$ 越低 → 易用性越差。$\eta_c = 100\%$ 表示全部 Pattern 为「易」；$\eta_c = 0\%$ 表示全部 Pattern 为「难」。

### 1.3 五度量项易用性计算实例

| 度量项 $c$ | $\Sigma w_c$ | $\eta_c = \dfrac{130 - \Sigma w_c}{104}$ | 难度档位 |
|---|---|---|---|
| 1 系统规格 | $82$ | $\dfrac{48}{104} = 46.15\%$ | 中 |
| 2 内存模型 | $88$ | $\dfrac{42}{104} = 40.38\%$ | 中偏难 |
| 3 计算模型 | $96$ | $\dfrac{34}{104} = 32.69\%$ | 难 |
| 4 控制模型 | $86$ | $\dfrac{44}{104} = 42.31\%$ | 中 |
| 5 访存模型 | $98$ | $\dfrac{32}{104} = 30.77\%$ | 难 |
| 6 调试调优 | 待填 | 待计算 | — |
| 7 Agent 友好 | 待填 | 待计算 | — |

> **解读**：访存模型（30.77%）与计算模型（32.69%）易用性最低，是当前微架构最大短板；系统规格（46.15%）相对最优但仍未过半，说明 26 个典型 Pattern 下整体微架构对软件仍不友好。

### 1.4 微架构综合易用性

5 个有数据度量项按 $\Sigma w_c$ 归一化得分类权重 $\Omega_c$：

$$
\Omega_c = \frac{\Sigma w_c}{\sum_{c'=1}^{5} \Sigma w_{c'}} = \frac{\Sigma w_c}{450}, \qquad \sum_{c=1}^{5} \Omega_c = 1
$$

| $c$ | $\Omega_c$ |
|---|---|
| 1 系统规格 | $82/450 = 0.182$ |
| 2 内存模型 | $88/450 = 0.196$ |
| 3 计算模型 | $96/450 = 0.213$ |
| 4 控制模型 | $86/450 = 0.191$ |
| 5 访存模型 | $98/450 = 0.218$ |

**微架构综合易用性**（按难度权重加权，难项影响大）：

$$
\boxed{\;
U_{\text{微架构}} = \sum_{c=1}^{5} \Omega_c \cdot \eta_c
\;}
$$

代入实例值：

$$
U_{\text{微架构}} = 0.182 \times 46.15\% + 0.196 \times 40.38\% + 0.213 \times 32.69\% + 0.191 \times 42.31\% + 0.218 \times 30.77\% \approx 37.86\%
$$

> 当前 26 Pattern 加权后微架构综合易用性约 **37.86%**，处于「难」档（< 60%），主要短板在访存模型与计算模型。

---

## 2. 评分模型（锚点得分原则）

> **统一原则**：所有度量指标统一为「越大越好」形式后，按「四锚点得分原则」归一化到 $s \in \{100, 80, 60, 0\}$。本章 $\eta_c$ 已是 $[0,100\%]$ 越大越好量，直接套用四锚点公式。

### 2.1 单度量项得分 $s_c$

设度量项 $c$ 的 Pattern 易用性免除率 $\eta_c \in [0, 100\%]$，四个锚点阈值满足 $\tau_{100} \ge \tau_{80} \ge \tau_{60} \ge \tau_{0}$：

$$
s_c =
\begin{cases}
100, & \eta_c \ge \tau_{100} \\
80,  & \tau_{80} \le \eta_c < \tau_{100} \\
60,  & \tau_{60} \le \eta_c < \tau_{80} \\
0,   & \eta_c \le \tau_{0}
\end{cases}
$$

**锚点定义**（对齐 §1.1 难度权重语义）：

| 锚点分 | 含义 | $\eta_c$ 阈值 | 对应 Pattern 分布 |
|---|---|---|---|
| $\tau_{100}=100\%$ | 硬件全自动，用户零感知 | 全部 Pattern 为「易」 | 26 易 / 0 中 / 0 难 |
| $\tau_{80}=75\%$ | 自动为主，少量手动 | 多数易、少量中、无难 | 约 20 易 / 6 中 / 0 难 |
| $\tau_{60}=50\%$ | 手动为主但可管理 | 易难各半 | 约 13 易 / 0 中 / 13 难 |
| $\tau_{0}=0\%$ | 完全手动，无自动化 | 全部 Pattern 为「难」 | 0 易 / 0 中 / 26 难 |

> 锚点间线性插值（可选）：$\eta_c \in (\tau_0, \tau_{60})$ 时，$s_c = 60 \cdot \dfrac{\eta_c - \tau_0}{\tau_{60} - \tau_0}$。

### 2.2 度量项得分 $S_c$

度量项 $c$ 的得分即其 Pattern 易用性免除率的锚点映射：

$$
S_c = s_c = \text{AnchorMap}(\eta_c)
$$

> 本体系中每个度量项只有一个聚合指标 $\eta_c$，故 $S_c = s_c$，无指标级二次加权。

### 2.3 综合易用性 $U$

$$
U = \sum_{c=1}^{7} \Omega_c \cdot S_c, \qquad \sum_{c} \Omega_c = 1
$$

> 第 6、7 度量项数据待填前，$\Omega_6 = \Omega_7 = 0$，仅在 5 项内归一化（见 §1.4）。

### 2.4 等级映射

$$
\text{Grade}(U) =
\begin{cases}
S, & U \in [90, 100] \\
A, & U \in [80, 90) \\
B, & U \in [70, 80) \\
C, & U \in [60, 70) \\
D, & U \in [0, 60)
\end{cases}
$$

---

## 3. 7 个度量项易用性详细定义

> **风格说明**：每个度量项独立成节，给出（1）Pattern 加权公式 $\eta_c$；（2）26 Pattern 权重明细表；（3）锚点；（4）当前实测值；（5）短板 Pattern 与改进建议。
> 第 6、7 度量项 Excel 数据待填，给出公式模板与待填表头。

### 3.1 系统规格易用性（$\Omega_1 = 0.182$）

> **来源**：Excel「系统规格」列。度量目标：Pattern 在系统规格层面（带宽、对齐、搬运切分）的软件代价；「难」= 跳搬运中（需 2 次搬运切分、2 次搬运）。

#### 3.1.1 Pattern 易用性公式

$$
\eta_1 = \frac{130 - \sum_{j=1}^{26} w_{1,j}}{104} \times 100\%
$$

#### 3.1.2 26 Pattern 权重明细

| # | Pattern | $w_{1,j}$ | | # | Pattern | $w_{1,j}$ | | # | Pattern | $w_{1,j}$ |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | Elementwise | 1 | | 10 | AtomicUpdate | 5 | | 19 | Spectral | 5 |
| 2 | Broadcast | 1 | | 11 | Interpolation | 3 | | 20 | DynamicShape | 5 |
| 3 | Reduction | 3 | | 12 | MaskPredicate | 1 | | 21 | RandomSampling | 1 |
| 4 | Contraction | 3 | | 13 | SortSelect | 5 | | 22 | Norm | 3 |
| 5 | ArgReduce | 3 | | 14 | Histogram | 3 | | 23 | Sparse | 5 |
| 6 | Layout Transform | 5 | | 15 | SlidingWindow | 5 | | 24 | Quantization | 1 |
| 7 | Padding | 5 | | 16 | ControlFlow | 1 | | 25 | Collective | 3 |
| 8 | IndexGather | 3 | | 17 | Recurrence | 3 | | 26 | FusedComposite | 5 |
| 9 | ScatterUpdate | 3 | | 18 | MegaKernel | 1 | | — | **$\Sigma = 82$** | |

- **编码**：M1-1 ｜ **锚点**：$\tau_{100}=100\%,\ \tau_{80}=75\%,\ \tau_{60}=50\%,\ \tau_0=0\%$
- **当前实测**：$\eta_1 = 46.15\%$ → $S_1 = 0$（D 档，未过及格线）
- **短板 Pattern**（难=5）：Layout Transform、Padding、AtomicUpdate、SortSelect、SlidingWindow、Spectral、DynamicShape、Sparse、FusedComposite（9 个）
- **改进建议**：硬件自动 padding/对齐；编译器自动 layout 推导消除跳搬运中

#### 3.1.3 度量项得分

$$
S_1 = \text{AnchorMap}\!\left(\frac{130 - 82}{104}\right) = \text{AnchorMap}(46.15\%) = 0
$$

---

### 3.2 内存模型易用性（$\Omega_2 = 0.196$）

> **来源**：Excel「内存模型」列。度量目标：Pattern 在内存模型层面（离散访问、bank 冲突、层级感知）的软件代价；「难」= 离散的难。

#### 3.2.1 Pattern 易用性公式

$$
\eta_2 = \frac{130 - \sum_{j=1}^{26} w_{2,j}}{104} \times 100\%
$$

#### 3.2.2 26 Pattern 权重明细

| # | Pattern | $w_{2,j}$ | | # | Pattern | $w_{2,j}$ | | # | Pattern | $w_{2,j}$ |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | Elementwise | 1 | | 10 | AtomicUpdate | 1 | | 19 | Spectral | 5 |
| 2 | Broadcast | 1 | | 11 | Interpolation | 3 | | 20 | DynamicShape | 5 |
| 3 | Reduction | 5 | | 12 | MaskPredicate | 1 | | 21 | RandomSampling | 1 |
| 4 | Contraction | 5 | | 13 | SortSelect | 5 | | 22 | Norm | 5 |
| 5 | ArgReduce | 5 | | 14 | Histogram | 3 | | 23 | Sparse | 5 |
| 6 | Layout Transform | 3 | | 15 | SlidingWindow | 5 | | 24 | Quantization | 1 |
| 7 | Padding | 5 | | 16 | ControlFlow | 1 | | 25 | Collective | 1 |
| 8 | IndexGather | 3 | | 17 | Recurrence | 5 | | 26 | FusedComposite | 5 |
| 9 | ScatterUpdate | 3 | | 18 | MegaKernel | 5 | | — | **$\Sigma = 88$** | |

- **编码**：M2-1 ｜ **锚点**：$\tau_{100}=100\%,\ \tau_{80}=75\%,\ \tau_{60}=50\%,\ \tau_0=0\%$
- **当前实测**：$\eta_2 = 40.38\%$ → $S_2 = 0$
- **短板 Pattern**（难=5，共 13 个）：Reduction、Contraction、ArgReduce、Padding、SortSelect、SlidingWindow、Recurrence、MegaKernel、Spectral、DynamicShape、Norm、Sparse、FusedComposite
- **改进建议**：硬件离散访问优化；bank 冲突自动规避；层级屏蔽

#### 3.2.3 度量项得分

$$
S_2 = \text{AnchorMap}\!\left(\frac{130 - 88}{104}\right) = \text{AnchorMap}(40.38\%) = 0
$$

---

### 3.3 计算模型易用性（$\Omega_3 = 0.213$）

> **来源**：Excel「计算模型」列。度量目标：Pattern 在计算模型层面（指令完备、偏移计算、原子性）的软件代价；「难」= 偏移计算。

#### 3.3.1 Pattern 易用性公式

$$
\eta_3 = \frac{130 - \sum_{j=1}^{26} w_{3,j}}{104} \times 100\%
$$

#### 3.3.2 26 Pattern 权重明细

| # | Pattern | $w_{3,j}$ | | # | Pattern | $w_{3,j}$ | | # | Pattern | $w_{3,j}$ |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | Elementwise | 1 | | 10 | AtomicUpdate | 1 | | 19 | Spectral | 5 |
| 2 | Broadcast | 1 | | 11 | Interpolation | 5 | | 20 | DynamicShape | 5 |
| 3 | Reduction | 3 | | 12 | MaskPredicate | 5 | | 21 | RandomSampling | 5 |
| 4 | Contraction | 3 | | 13 | SortSelect | 5 | | 22 | Norm | 5 |
| 5 | ArgReduce | 5 | | 14 | Histogram | 3 | | 23 | Sparse | 5 |
| 6 | Layout Transform | 3 | | 15 | SlidingWindow | 5 | | 24 | Quantization | 5 |
| 7 | Padding | 5 | | 16 | ControlFlow | 1 | | 25 | Collective | 1 |
| 8 | IndexGather | 3 | | 17 | Recurrence | 5 | | 26 | FusedComposite | 5 |
| 9 | ScatterUpdate | 3 | | 18 | MegaKernel | 3 | | — | **$\Sigma = 96$** | |

- **编码**：M3-1 ｜ **锚点**：$\tau_{100}=100\%,\ \tau_{80}=75\%,\ \tau_{60}=50\%,\ \tau_0=0\%$
- **当前实测**：$\eta_3 = 32.69\%$ → $S_3 = 0$
- **短板 Pattern**（难=5，共 14 个）：ArgReduce、Padding、Interpolation、MaskPredicate、SortSelect、SlidingWindow、Recurrence、Spectral、DynamicShape、RandomSampling、Norm、Sparse、Quantization、FusedComposite
- **改进建议**：补齐离散/偏移计算指令；指令原子化；减少状态残留

#### 3.3.3 度量项得分

$$
S_3 = \text{AnchorMap}\!\left(\frac{130 - 96}{104}\right) = \text{AnchorMap}(32.69\%) = 0
$$

---

### 3.4 控制模型易用性（$\Omega_4 = 0.191$）

> **来源**：Excel「控制模型」列。度量目标：Pattern 在控制模型层面（多流水、同步、生产消费）的软件代价；「难」= 多流水。

#### 3.4.1 Pattern 易用性公式

$$
\eta_4 = \frac{130 - \sum_{j=1}^{26} w_{4,j}}{104} \times 100\%
$$

#### 3.4.2 26 Pattern 权重明细

| # | Pattern | $w_{4,j}$ | | # | Pattern | $w_{4,j}$ | | # | Pattern | $w_{4,j}$ |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | Elementwise | 1 | | 10 | AtomicUpdate | 5 | | 19 | Spectral | 5 |
| 2 | Broadcast | 1 | | 11 | Interpolation | 1 | | 20 | DynamicShape | 3 |
| 3 | Reduction | 1 | | 12 | MaskPredicate | 5 | | 21 | RandomSampling | 1 |
| 4 | Contraction | 3 | | 13 | SortSelect | 5 | | 22 | Norm | 5 |
| 5 | ArgReduce | 5 | | 14 | Histogram | 1 | | 23 | Sparse | 5 |
| 6 | Layout Transform | 1 | | 15 | SlidingWindow | 5 | | 24 | Quantization | 3 |
| 7 | Padding | 5 | | 16 | ControlFlow | 3 | | 25 | Collective | 5 |
| 8 | IndexGather | 1 | | 17 | Recurrence | 5 | | 26 | FusedComposite | 5 |
| 9 | ScatterUpdate | 1 | | 18 | MegaKernel | 5 | | — | **$\Sigma = 86$** | |

- **编码**：M4-1 ｜ **锚点**：$\tau_{100}=100\%,\ \tau_{80}=75\%,\ \tau_{60}=50\%,\ \tau_0=0\%$
- **当前实测**：$\eta_4 = 42.31\%$ → $S_4 = 0$
- **短板 Pattern**（难=5，共 13 个）：ArgReduce、Padding、AtomicUpdate、MaskPredicate、SortSelect、SlidingWindow、Recurrence、MegaKernel、Spectral、Norm、Sparse、Collective、FusedComposite
- **改进建议**：单逻辑单线程编程模型；多流水自动编排；同步免除

#### 3.4.3 度量项得分

$$
S_4 = \text{AnchorMap}\!\left(\frac{130 - 86}{104}\right) = \text{AnchorMap}(42.31\%) = 0
$$

---

### 3.5 访存模型易用性（$\Omega_5 = 0.218$）

> **来源**：Excel「访存模型」列。度量目标：Pattern 在访存模型层面（内存管理、指令变形、搬运）的软件代价；「难」= 内存管理，指令变形。

#### 3.5.1 Pattern 易用性公式

$$
\eta_5 = \frac{130 - \sum_{j=1}^{26} w_{5,j}}{104} \times 100\%
$$

#### 3.5.2 26 Pattern 权重明细

| # | Pattern | $w_{5,j}$ | | # | Pattern | $w_{5,j}$ | | # | Pattern | $w_{5,j}$ |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | Elementwise | 1 | | 10 | AtomicUpdate | 1 | | 19 | Spectral | 5 |
| 2 | Broadcast | 5 | | 11 | Interpolation | 5 | | 20 | DynamicShape | 5 |
| 3 | Reduction | 5 | | 12 | MaskPredicate | 1 | | 21 | RandomSampling | 1 |
| 4 | Contraction | 5 | | 13 | SortSelect | 5 | | 22 | Norm | 5 |
| 5 | ArgReduce | 5 | | 14 | Histogram | 3 | | 23 | Sparse | 5 |
| 6 | Layout Transform | 5 | | 15 | SlidingWindow | 5 | | 24 | Quantization | 3 |
| 7 | Padding | 5 | | 16 | ControlFlow | 1 | | 25 | Collective | 3 |
| 8 | IndexGather | 3 | | 17 | Recurrence | 5 | | 26 | FusedComposite | 5 |
| 9 | ScatterUpdate | 3 | | 18 | MegaKernel | 3 | | — | **$\Sigma = 98$** | |

- **编码**：M5-1 ｜ **锚点**：$\tau_{100}=100\%,\ \tau_{80}=75\%,\ \tau_{60}=50\%,\ \tau_0=0\%$
- **当前实测**：$\eta_5 = 30.77\%$ → $S_5 = 0$（5 项中最低）
- **短板 Pattern**（难=5，共 15 个）：Broadcast、Reduction、Contraction、ArgReduce、Layout Transform、Padding、Interpolation、SortSelect、SlidingWindow、Recurrence、Spectral、DynamicShape、Norm、Sparse、FusedComposite
- **改进建议**：自动内存管理；指令变形自动化；离散访问偏移免除

#### 3.5.3 度量项得分

$$
S_5 = \text{AnchorMap}\!\left(\frac{130 - 98}{104}\right) = \text{AnchorMap}(30.77\%) = 0
$$

---

### 3.6 调试调优易用性（$\Omega_6 = 0.00$，待填）

> **来源**：Excel「调试调优」列（当前为空）。度量目标：Pattern 在调试调优层面（异常处理、断点、可观测、性能度量）的软件代价。
> **状态**：Excel 数据待填，本节给出公式模板与待填表头。

#### 3.6.1 Pattern 易用性公式（模板）

$$
\eta_6 = \frac{130 - \sum_{j=1}^{26} w_{6,j}}{104} \times 100\%
$$

#### 3.6.2 26 Pattern 权重明细（待填）

| # | Pattern | $w_{6,j}$ | | # | Pattern | $w_{6,j}$ | | # | Pattern | $w_{6,j}$ |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | Elementwise | — | | 10 | AtomicUpdate | — | | 19 | Spectral | — |
| 2 | Broadcast | — | | 11 | Interpolation | — | | 20 | DynamicShape | — |
| 3 | Reduction | — | | 12 | MaskPredicate | — | | 21 | RandomSampling | — |
| 4 | Contraction | — | | 13 | SortSelect | — | | 22 | Norm | — |
| 5 | ArgReduce | — | | 14 | Histogram | — | | 23 | Sparse | — |
| 6 | Layout Transform | — | | 15 | SlidingWindow | — | | 24 | Quantization | — |
| 7 | Padding | — | | 16 | ControlFlow | — | | 25 | Collective | — |
| 8 | IndexGather | — | | 17 | Recurrence | — | | 26 | FusedComposite | — |
| 9 | ScatterUpdate | — | | 18 | MegaKernel | — | | — | **$\Sigma = ?$** | |

- **编码**：M6-1 ｜ **锚点**：$\tau_{100}=100\%,\ \tau_{80}=75\%,\ \tau_{60}=50\%,\ \tau_0=0\%$
- **当前实测**：待 Excel 补齐后计算
- **权重补齐后**：$\Omega_6 = \Sigma w_6 / (\sum_{c=1}^{7} \Sigma w_c)$，并重新归一化所有 $\Omega_c$

#### 3.6.3 度量项得分（模板）

$$
S_6 = \text{AnchorMap}\!\left(\frac{130 - \Sigma w_6}{104}\right)
$$

---

### 3.7 Agent 友好易用性（$\Omega_7 = 0.00$，待填）

> **来源**：Excel「Agent 友好」列（当前为空）。度量目标：Pattern 对 Agent 自动生成/自动调优代码的友好程度。
> **状态**：Excel 数据待填，本节给出公式模板与待填表头。

#### 3.7.1 Pattern 易用性公式（模板）

$$
\eta_7 = \frac{130 - \sum_{j=1}^{26} w_{7,j}}{104} \times 100\%
$$

#### 3.7.2 26 Pattern 权重明细（待填）

| # | Pattern | $w_{7,j}$ | | # | Pattern | $w_{7,j}$ | | # | Pattern | $w_{7,j}$ |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | Elementwise | — | | 10 | AtomicUpdate | — | | 19 | Spectral | — |
| 2 | Broadcast | — | | 11 | Interpolation | — | | 20 | DynamicShape | — |
| 3 | Reduction | — | | 12 | MaskPredicate | — | | 21 | RandomSampling | — |
| 4 | Contraction | — | | 13 | SortSelect | — | | 22 | Norm | — |
| 5 | ArgReduce | — | | 14 | Histogram | — | | 23 | Sparse | — |
| 6 | Layout Transform | — | | 15 | SlidingWindow | — | | 24 | Quantization | — |
| 7 | Padding | — | | 16 | ControlFlow | — | | 25 | Collective | — |
| 8 | IndexGather | — | | 17 | Recurrence | — | | 26 | FusedComposite | — |
| 9 | ScatterUpdate | — | | 18 | MegaKernel | — | | — | **$\Sigma = ?$** | |

- **编码**：M7-1 ｜ **锚点**：$\tau_{100}=100\%,\ \tau_{80}=75\%,\ \tau_{60}=50\%,\ \tau_0=0\%$
- **当前实测**：待 Excel 补齐后计算
- **权重补齐后**：$\Omega_7 = \Sigma w_7 / (\sum_{c=1}^{7} \Sigma w_c)$，并重新归一化所有 $\Omega_c$

#### 3.7.3 度量项得分（模板）

$$
S_7 = \text{AnchorMap}\!\left(\frac{130 - \Sigma w_7}{104}\right)
$$

---

## 4. 综合易用性与短板分析

### 4.1 微架构综合易用性公式

$$
\boxed{\;
U_{\text{微架构}} = \sum_{c=1}^{7} \Omega_c \cdot S_c, \qquad
\Omega_c = \frac{\Sigma w_c}{\sum_{c'=1}^{7} \Sigma w_{c'}}
\;}
$$

> 当前第 6、7 列待填，退化为 5 项归一化（见 §1.4）。

### 4.2 当前实测（5 项）

| 度量项 | $\Sigma w_c$ | $\Omega_c$ | $\eta_c$ | $S_c$ | 等级 |
|---|---|---|---|---|---|
| 1 系统规格 | 82 | 0.182 | 46.15% | 0 | D |
| 2 内存模型 | 88 | 0.196 | 40.38% | 0 | D |
| 3 计算模型 | 96 | 0.213 | 32.69% | 0 | D |
| 4 控制模型 | 86 | 0.191 | 42.31% | 0 | D |
| 5 访存模型 | 98 | 0.218 | 30.77% | 0 | D |
| **综合** | 450 | 1.000 | — | **0** | **D** |

> 5 项 $\eta_c$ 均低于 $\tau_{60}=50\%$，综合易用性 $U = 0$（D 档）。微架构对 26 个典型 Pattern 的整体易用性存在系统性短板。

### 4.3 跨度量项短板 Pattern（多列同时为「难」）

下表列出在 ≥3 个度量项同时为「难(5)」的 Pattern，为最高优先级改进对象：

| # | Pattern | 难列数 | 难列明细 |
|---|---|---|---|
| 7 | Padding | 5 | 全部 5 列 |
| 13 | SortSelect | 5 | 全部 5 列 |
| 15 | SlidingWindow | 5 | 全部 5 列 |
| 19 | Spectral | 5 | 全部 5 列 |
| 23 | Sparse | 5 | 全部 5 列 |
| 26 | FusedComposite | 5 | 全部 5 列 |
| 5 | ArgReduce | 4 | 内存/计算/控制/访存 |
| 17 | Recurrence | 4 | 内存/计算/控制/访存 |
| 20 | DynamicShape | 4 | 系统规格/计算/控制/访存 |
| 22 | Norm | 4 | 内存/计算/控制/访存 |

> **Padding / SortSelect / SlidingWindow / Spectral / Sparse / FusedComposite** 在 5 个度量项全部为「难」，是微架构易用性的核心瓶颈，建议优先投入硬件自动化与编译器特化 path。

---

## 5. 锚点提取方法汇总

| 数据来源 | 提取方法 | 适用度量项 |
|---|---|---|
| Excel 静态 Pattern 评分 | 26 Pattern × 7 列难度权重（难5/中3/易1） | 1–7 全部 |
| 算子源码 AST 扫描 | 解析 AscendC 算子源码，统计代码行/参数数/分支数/同步调用数 | 校验 Pattern 难度打分 |
| 硬件规格文档解析 | 解析指令集手册/target spec，提取能力位/容量/对齐粒度 | 校验系统规格/访存模型 |
| PMU 事件采集 | 运行时采集 PMU 计数器 | 校验调试调优列（待填） |
| μ-bench 对比 | 编写最小用例，对比有/无优化性能差异 | 校验计算/访存模型 |
| Agent 代码生成测试 | Agent 自动生成 26 Pattern 算子代码，统计成功率/改写行数 | 校验 Agent 友好列（待填） |

---

## 6. 待填数据与扩展流程

1. **补齐 Excel 第 6 列（调试调优）**：对 26 Pattern 逐一打分难/中/易，填入 $w_{6,j}$，计算 $\Sigma w_6$、$\eta_6$、$S_6$。
2. **补齐 Excel 第 7 列（Agent 友好）**：同上。
3. **重新归一化 $\Omega_c$**：$\Omega_c = \Sigma w_c / \sum_{c=1}^{7} \Sigma w_c$，覆盖所有 7 项。
4. **重算综合易用性**：$U = \sum_{c=1}^{7} \Omega_c \cdot S_c$。
5. **公式体系不变**：$\eta_c$ 线性免除率公式、四锚点评分、综合加权公式对任意列数自适应。
