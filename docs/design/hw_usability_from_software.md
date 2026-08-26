# 从软件度量硬件易用性：客观维度框架

> 目标：用 **软件制品上可计数的量** 度量硬件易用性，禁止问卷、访谈、Likert、专家「高/中/低」作为主指标。  
> 典型例子：为正确同步而必须插入的代码行（S-LOC）。  
> 对照：本文把 [`pytorch_op_feature_compete.md`](pytorch_op_feature_compete.md) 中的 SIMT/SIMD 档位，落到可复现的计数口径；同步税与 Pattern **29** / §7 对齐。

---

## 0. 定义与禁区

**硬件易用性（软件口径）**：在 **固定工作负载套件** 与 **固定正确性契约** 下，为让该硬件产出与参考实现等价的结果，软件侧必须额外付出的、可观测工作量。可选地再单独报一份「为达到参考性能契约」的额外工作量，二者不得混加。

| 允许 | 禁止（主观） |
|---|---|
| 源码 / AST / IR / ISA / 微码 / 运行轨迹上的计数 | 问卷、SUS、NPS、访谈满意度 |
| 相对 **同一基线实现** 的差量 | 无基线的「感觉好写」 |
| 事先写清的计数规则 | 专家打「高/中/低」且无法复现 |
| 静态制品 + 可选动态事件计数 | 把峰值 FLOPS、功耗、面积当成易用性 |

性能、正确性、易用性正交：气泡周期衡量效率，不衡量「好不好写」。动态轨迹只用来 **数事件**（几次 Host-block、几次隐式 sync），不把 latency 当作易用性分数。

---

## 1. 度量项的维度（坐标系）

每一个度量项是下面 **六轴** 上的一个点。缺任一轴则该项不可比。

```text
度量项 = (税种, 观测层, 计量种类, 义务, 可见性, 作用域)
         × 工作负载套件 × 参考基线 × 计数规则版本
```

### 1.1 D1 税种 — 额外工作的类别

| ID | 税种 | 软件上在数什么 | 硬件根因（只作解释，不计入分） |
|---|---|---|---|
| **T-Sync** | 同步税 | barrier / event / wait / fence / flag / CKE set·wait / PipeBarrier | 多流水、多核、Host–Device、多 Rank 无统一序 |
| **T-Mem** | 存储税 | 显式 copy/DMA、多地址空间搬移、workspace 管理 | 多级存储、不能 cache-coherent 的视图 |
| **T-Layout** | 布局税 | pad / align / 转置 / 打包 / 尾块标量收尾 | 向量宽、Bank、MMA tile、地址对齐硬约束 |
| **T-Ctrl** | 控制税 | 手工流水发射、特殊分支、双缓冲状态机 | 软件管线、无硬件调度器或调度器能力弱 |
| **T-Type** | 类型税 | 显式 dtype 转换、量化 QDQ、饱和/舍入插桩 | 计算单元 dtype 与 API dtype 不一致 |
| **T-Res** | 资源税 | 手工切块、占位、多核划分、L2 window 配置 | 寄存器/UB/L2 容量与绑定需软件声明 |
| **T-Err** | 约束税 | 为避开 UB/竞态而写的检查、注释契约、静态断言 | 弱内存模型、未定义对齐、隐式同步 |
| **T-Port** | 专用税 | 仅该硬件出现的 API/内建/编译指示 | ISA 原语或运行时与基线模型不对齐 |
| **T-Gap** | 原语缺口税 | 缺硬件原语时的软件模拟（reduce 树、bitonic） | 无 `vreduce`/`vscan`/`vsort` 等 |

同一段代码只归入 **一个主税种**；若一行同时同步+搬移，按 **主目的** 归类（copy 上的隐式 sync 记 T-Sync 的隐式子项，copy 本身记 T-Mem）。

### 1.2 D2 观测层 — 在哪计数

与现有 API 责任栈对齐，并向下延伸到 ISA 与轨迹。

| 层 | 制品 | 典型计数对象 |
|---|---|---|
| **L3** 用户 API | Python / C++ 调用点 | `synchronize`、`.item()`、手工 Event |
| **L2** Schema / 运行时 | ATen / c10d / Stream | 隐式 D2H、`Work.wait` |
| **L1** 编译 / IR | Graph、Inductor、CCU translator | `Wait`/`RecordEvent` 节点、dep_token |
| **L0** 设备核 | CUDA / AscendC / 手写 kernel | `__syncthreads`、`SetFlag`/`WaitFlag` |
| **L−1** ISA / 微码 | SASS / CCU `CcuInstr` | `bar.sync`、CKE set/wait 字段 |
| **L−2** 动态轨迹 | nsys / NPU profiler / 日志 | 实际发生的 Host-block、隐式 sync 次数 |

**规则**：跨层比较必须声明层；L3 低税 + L0 高税 = 框架吸收，不是硬件消失。报告时写 `T-Sync@L0` 这种限定名。

### 1.3 D3 计量种类 — 如何计数

| ID | 种类 | 单位 | 适用 |
|---|---|---|---|
| **Q-Vol** | 体积 | 行 / token / AST 节点 / 指令条数 | 「为同步插入的代码行」属此类 |
| **Q-Card** | 基数 | 次、个（独立 API、独立 barrier、地址空间数） | 去重后的概念负担 |
| **Q-Ratio** | 比率 | 税代码 / 有效载荷代码 | 消除内核大小差异 |
| **Q-Delta** | 差量 | 相对基线的 ΔLOC、Δ指令 | 跨硬件唯一公平口径 |
| **Q-Freq** | 频次 | 每算子 / 每迭代 / 每 kernel | 静态体积相同但热路径不同 |
| **Q-Depth** | 深度 | 流水级数、必须同时记住的序域个数 | 认知负担的客观代理（仍非问卷） |
| **Q-Pair** | 配对完备性 | wait/set 是否成对、未匹配数 | 同步协议完整度 |

体积默认用 **AST 语句** 而非物理换行（避免格式噪声）。若只能用物理行，必须同一 formatter。

### 1.4 D4 义务 — 为什么必须写

| ID | 义务 | 计入易用性？ |
|---|---|---|
| **O-Corr** | 正确性必需（不写则结果错或死锁） | **主指标** |
| **O-UB** | 为使行为被定义（不写则竞态/UB） | **主指标**（与 O-Corr 可合并报，但须可拆） |
| **O-Perf** | 仅为性能（不写仍正确但慢） | **副指标**，单独成表 |
| **O-Style** | 风格/可读性 | **不计** |

「为同步插入的代码行」默认只计 **O-Corr ∪ O-UB**。双缓冲、软件流水、L2 Persistent window 默认 O-Perf，除非无该配置则功能不可用。

### 1.5 D5 可见性 — 谁看见这段税

| ID | 可见性 | 含义 |
|---|---|---|
| **V-User** | 用户必须写 | 出现在业务/算子作者代码 |
| **V-Fw** | 框架吸收 | 用户 API 无，L1/L2 实现有 |
| **V-Impl** | 硬件隐式 | 代码无对应语句，轨迹上仍发生（如隐式 D2H） |

V-Impl 用 Q-Card/Q-Freq 在 L−2 计数，不用 LOC（没有行可插）。V-Impl 高 = 硬件/运行时把序藏起来，**易用性不一定更好**：漏标的隐式同步是正确性陷阱，应作为独立差指标。

### 1.6 D6 作用域 — 序/数据作用范围

| ID | 范围 | 同步税上的对应（Pattern 29） |
|---|---|---|
| lane / warp | 向量/束内 | 通常由 ISA 隐式 |
| block / pipe | `__syncthreads` / `PipeBarrier` | 29.8 |
| stream / event | 设备队列序 | 29.2 |
| device | 全设备 | 29.1 |
| host | CPU 线程阻塞 | 29.1 / 29.7 |
| rank / mesh | 通信域 | 29.5 / 29.6 |

粒度越粗，同样一次同步的 **性能** 代价越大；易用性上则数 **程序员必须区分的范围种类数**（Q-Depth@D6），而不是气泡时间。

---

## 2. 基线、套件、计数规则（使数字客观）

无这三项，任何 LOC 都不可比。

### 2.1 参考基线（必须三选一并写死）

| 基线 ID | 含义 | 用途 |
|---|---|---|
| **B-Seq** | 正确的单线程 C/C++ 参考 | 绝对下限（含所有并行税） |
| **B-SIMT** | 同一算子的 CUDA SIMT 规范写法（grid-stride，无手工流水） | 与现有「SIMT 易用」对齐 |
| **B-ATen** | 只调 PyTorch 声明式 API、不写核 | 测「硬件是否被框架藏住」 |

跨芯片比较推荐 **B-SIMT**：差的是相对 CUDA 编程模型的税，而不是相对串行 C 的并行税。

### 2.2 工作负载套件

固定、版本化、按 Pattern 抽样，禁止用「随便一个 kernel」。

| 套件槽 | 来源 | 最少覆盖 |
|---|---|---|
| W1 计算核 | Pattern 01–03、05、11 | 点对、规约 |
| W2 布局敏感 | 14、16、U1/U2 尾块与非对齐 | 布局税 |
| W3 领域核 | 09/10/12（Cube）与 Vec epilogue | 资源/流水税 |
| W4 同步核 | 29.2 多流、29.5 通信重叠、29.8 核内 barrier | 同步税 |
| W5 缺口核 | 06/25/26（scan/sort/compact） | 原语缺口税 |

每个样本附带：**输入契约**（shape、dtype、contiguous 否）、**正确性 oracle**、**是否要求性能契约**。

### 2.3 计数规则（S-LOC 操作定义）

**同步语句（计入 S-LOC / S-AST）** 当且仅当该 AST 语句的 **主语义** 是建立序，而不是计算或搬数：

```text
INCLUDE:
  __syncthreads / syncthreads_count
  cudaDeviceSynchronize / stream.synchronize / event.record|wait
  atomic 栅栏、threadfence*、membar、fence
  PipeBarrier / SetFlag / WaitFlag / Notify / Wait
  CCU：独立 Sync* 指令；以及指令尾部 setCKE*/waitCKE* 各计 1 次序操作（L−1 用 Q-Card 而非行）
  Work.wait / barrier / irecv+wait
  为等待完成而写的 poll 循环（整段计 1 个控制结构 + 其体内行）

EXCLUDE:
  纯计算、纯 load/store（即使有副作用）
  注释、空行、花括号、日志
  性能用 prefetch / L2 AccessPolicyWindow（改记 T-Res、O-Perf）
  框架生成代码与手写代码必须分表，禁止加总后当「硬件税」
```

**差量**：

```text
S-LOC_hw     = count(sync AST, impl_hw, W, layer)
S-LOC_base   = count(sync AST, impl_base, W, layer)
ΔS-LOC       = S-LOC_hw - S-LOC_base          # 主报
S-Ratio      = S-LOC_hw / max(Payload-AST, 1)
Payload-AST  = 非税种语句（计算+必要访存）
```

Δ < 0 合法：硬件若把 barrier 收进指令隐式序，L0 的 ΔS-LOC 可为负；此时必须同报 L−2 隐式同步次数，避免把「藏起来的序」报成更易用。

---

## 3. 核心度量项（由维度实例化）

名称采用 `税种.计量@层`。下表为建议最小集；扩展时先填六轴再命名。

### 3.1 同步税（用户例子所在族）

| 度量项 | 六轴 | 操作定义 | 方向 |
|---|---|---|---|
| **S-LOC** | T-Sync, L0 或 L3, Q-Vol, O-Corr, V-User, 按语句作用域 | 同步 AST 语句数 | ↓ 更好 |
| **ΔS-LOC** | 同上 + Q-Delta vs B-SIMT | 相对基线多写的同步语句 | ↓ 更好 |
| **S-Ratio** | T-Sync, 同层, Q-Ratio | 同步语句 / 载荷语句 | ↓ 更好 |
| **S-N** | T-Sync, 同层, Q-Card | 动态或静态同步操作次数（去重规则：同一宏展开计 1） | ↓ 更好 |
| **S-PairGap** | T-Sync, L0/L−1, Q-Pair | `set` 无匹配 `wait` 或反向的静态计数 | =0 为合格 |
| **S-DomainN** | T-Sync, L0+L3, Q-Depth, D6 | 该样本必须同时正确使用的阻塞域种类数（Host/Stream/Block/Rank…） | ↓ 更好 |
| **S-HostRT** | T-Sync, L2/L−2, Q-Freq, host | Host-block 次数 / 迭代（含 `.item()`、同步 D2H） | ↓ 更好 |
| **S-Implicit** | T-Sync, L−2, Q-Freq, V-Impl | 轨迹中无对应源语句的同步事件 | ↓ 更好（陷阱更少） |

CCU 特例：大量传输指令尾部带 CKE set/wait（见 `docs/ccu/01-underlying-instruction-format.md`）。L0 源若无独立 sync 行，应在 **L−1** 报 `S-N@L−1`（含尾部 CKE 字段非空的指令条数），否则会低估同步税。

### 3.2 存储 / 布局税

| 度量项 | 六轴 | 操作定义 | 方向 |
|---|---|---|---|
| **C-N** | T-Mem, L0/L3, Q-Card | 显式 copy/DMA/Move 调用次数 | ↓ |
| **C-LOC** | T-Mem, 同层, Q-Vol | 仅为搬移而存在的语句 | ↓ |
| **Space-N** | T-Mem, L0, Q-Card | 程序员必须命名的地址空间数（GM/UB/L1/寄存器/MS…） | ↓ |
| **Pad-B** | T-Layout, L0, Q-Vol（字节） | `allocated − logical_nbytes`（硬件 pad，不含 Pattern 07 语义 pad） | ↓ |
| **Tail-LOC** | T-Layout, L0, Q-Vol, O-Corr | 仅为 `n % VL != 0` 或未对齐而写的 epilogue | ↓ |
| **Align-Br** | T-Layout, L0, Q-Card | 对齐快路径 vs 慢路径分支条数 | ↓ |

### 3.3 控制 / 资源 / 类型税

| 度量项 | 六轴 | 操作定义 | 方向 |
|---|---|---|---|
| **Pipe-Stage** | T-Ctrl, L0, Q-Depth | 必须由软件推进的流水级数（如 MTE/VEC/CUBE 队列） | ↓ |
| **Flag-SM** | T-Ctrl, L0, Q-Card | 双缓冲/多缓冲状态变量与 flag 对数 | ↓ |
| **CC** | T-Ctrl, L0, Q-Vol | 核函数 McCabe 圈复杂度（工具固定版本） | ↓ |
| **Tile-ParamN** | T-Res, L0/L3, Q-Card | 必须由作者给出的硬件块大小/核数/workspace 参数个数 | ↓ |
| **Cvt-N** | T-Type, L0, Q-Card | 热路径显式 dtype convert 次数（非 API 提升） | ↓ |

L2 Persistent 的 `SetLimit` + `AccessPolicyWindow`（`docs/cuda/a.md`）记 **T-Res、O-Perf、V-User@L3**，不计入 S-LOC。

### 3.4 原语缺口税（替代主观 SIMD 档）

现有文档用「亲/偏向/难」再映射 2/1/0。客观替换：

```text
Gap-LOC(P)  = LOC(软件模拟实现) - LOC(有目标原语的实现)
Gap-I(P)    = 热路径指令条数(无原语) / 指令条数(有原语)
Cover(P)    = 1 若套件所需原型在 ISA 中存在，否则 0
```

| Pattern | 目标原语（见 §8.4） | 无原语时 Gap 的主要形态 |
|---|---|---|
| 05/11 | `vreduce_*` + masked | `log2(VL)` 级 shuffle 树 → Gap-I ≈ Θ(log VL) |
| 06/26 | `vscan` / `vcompress` | 扫描网络 + scatter |
| 25 | `vsort` / `vmergesort` | bitonic 手写 |
| 01 尾块 | `vloadm`/`vstorem` | 标量 epilogue → Tail-LOC |

**禁止**再用「高/中/低」作为对外数字；若需与旧表对照，只允许：`Cover=1 ∧ Gap-I≤1.2 → 旧「亲」` 这种 **事后映射**，映射不得参与计算。

### 3.5 专用税与框架吸收

| 度量项 | 六轴 | 操作定义 |
|---|---|---|
| **API-Min** | T-Port, L3, Q-Card | 完成 W 中一样本所需最少调用次数（含创建 Stream/Event/Workspace） |
| **HW-ParamN** | T-Port, L3, Q-Card | 参数中无对应 B-SIMT/B-ATen 等价物的个数 |
| **Absorb** | 任意税种, Q-Delta 跨层 | `tax@L3 / tax@L0`；→0 表示框架吸税，用户易用但后端仍税 |

Absorb 用于拆开「芯片易用」与「栈易用」。评硬件后端时看 L0/L−1；评产品栈时看 L3。

---

## 4. 一张样本如何填（模板）

```text
workload:     Pattern-05.sum  n=4096, dtype=fp32, contiguous
baseline:     B-SIMT
layer:        L0
rule_ver:     1.0
impl:         <git rev + 文件>

# 每项：六轴 + 数
S-LOC:        2          # __syncthreads x2
ΔS-LOC:       0          # 与 CUDA 参考相同
S-Ratio:      2/40
S-DomainN:    1          # block
S-Implicit:   0
Tail-LOC:     8          # 无 vloadm 的标量尾
Gap-I:        4.0        # 无 vreduce，shuffle 树
Cover:        0
Space-N:      2          # global + shared
Pipe-Stage:   1
O-Perf 另表:  (空)
```

套件总分只允许：

- **微平均**：Σ 计数 / Σ 载荷（体积类）
- **宏平均**：各 Pattern 先内部平均再等权（避免 01 点对淹没 25 sort）

Cube 类（09/10/12.2）与 Vec 类分表，沿用现有「Cube 不入 SIMD 分母」的口径。

---

## 5. 与现有文档的关系

| 旧物 | 本文处理 |
|---|---|
| SIMT/SIMD「高/中/低」、Score_SIMT/SIMD | 退役为主指标；改报 Cover、Gap-I、ΔS-LOC、Tail-LOC |
| Pattern 29 / §7 F1–F10 | F1/F4 → D6；F6 async fused/split → V-Fw vs V-User；F10 隐式 → V-Impl + S-Implicit |
| CCU 指令尾 CKE | 强制 L−1 `S-N`，禁止只看 L0 源码行 |
| CUDA L2 Persistent | T-Res + O-Perf，不是同步税 |
| 硬件 pad vs 语义 pad | Pad-B 只计硬件 pad；07 不计入布局税 |

---

## 6. 最小落地清单

1. 冻结套件 W1–W5 的样本列表与 oracle（git 跟踪）。  
2. 选基线 B-SIMT，固定 formatter / AST 工具版本。  
3. 先实现 **S-LOC / ΔS-LOC / S-Ratio / S-N / S-Implicit / Tail-LOC / Cover / Gap-I / Space-N / Pipe-Stage** 十项。  
4. 每个后端一张表：行=样本，列=度量项，页脚=宏/微平均；O-Perf 另页。  
5. 变更计数规则必须 bump `rule_ver`，旧数字作废。

---

## 7. 反例：看起来客观、实为主观

| 做法 | 问题 |
|---|---|
| 专家把 kernel 标「中等难写」再赋 1.5 分 | 不可复现 |
| 用运行时间或占用率当易用性 | 测的是效率 |
| 只数物理行、不固定格式化 | 换 clang-format 分数就变 |
| 把 codegen 行数加进手写税 | 测的是生成器啰嗦，不是硬件 |
| 只报 L3 的 0 行同步 | 忽略 L0/L−1/L−2 的税与隐式序 |

---

*度量项必须落在 (税种 × 观测层 × 计量 × 义务 × 可见性 × 作用域) 上；「为同步插入的代码行」= T-Sync × L0/L3 × Q-Vol × O-Corr × V-User。跨硬件只比较同轴、同套件、同基线、同 rule_ver 的 Δ。*
