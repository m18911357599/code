# Ascend 950 多 Pipeline 编程分析

> 基于昇腾社区（CANN / Ascend C 文档）与 [asc-devkit](https://gitcode.com/cann/asc-devkit) 仓资料整理。  
> 适用芯片：Ascend 950PR / Ascend 950DT（`__NPU_ARCH__` = 3510）。

---

## 1. 编程模型

### 1.1 编程者视角的硬件模型：4 类并行单元构成多 Pipeline

从编程者角度，AI Core 可以简化为 **4 类执行单元**，它们各自持有独立的指令队列，**异步并行**推进——这就是“多 Pipeline”的全部含义：

| 单元 | 角色 | 对应流水 |
|------|------|----------|
| **Scalar** | “指挥官”：取指/译码，把指令**顺序发射**到各单元的指令队列；自身也执行标量计算（GetValue 等） | PIPE_S |
| **Vector** | 矢量计算，读写 UB | PIPE_V |
| **Cube** | 矩阵计算，读写 L0A/L0B/L0C | PIPE_M |
| **MTE** | 搬运引擎，细分为：MTE2（GM→UB/L1，搬入）、MTE3（UB→GM，搬出）、MTE1（L1→L0A/L0B）、FixPipe（L0C→GM/L1/UB） | PIPE_MTE2 / MTE3 / MTE1 / FIX |

关键心智模型：

```
                 ┌── Vector 指令队列 ──→ Vector 单元 ──┐
Scalar 顺序发射 ─┼── Cube   指令队列 ──→ Cube   单元 ──┼── 各队列内部顺序执行
                 └── MTE    指令队列 ──→ MTE    单元 ──┘    队列之间完全并行
```

- **发射是串行的，执行是并行的**。Scalar 按程序顺序把指令丢进各队列后立即继续，不等待执行完成。
- **同一队列内指令顺序执行**（但“开始执行”不等于“前一条已完成读写”，见 PipeBarrier）。
- **不同队列之间没有任何默认顺序保证**——这既是并行性能的来源，也是所有同步问题的来源。

典型 Vector 数据路径（三个单元接力访问同一块 UB）：

```
GM --MTE2--> UB(VECIN) --Vector--> UB(VECOUT) --MTE3--> GM
```

### 1.2 两个执行单元访问同一块 Buffer：必须同步

由于队列间无顺序保证，只要 **2 个执行单元先后访问同一块 Buffer**，就存在数据竞争，必须由同步原语强制时序。两种依赖：

- **写后读（先写完才能读）**：MTE2 写完 UB，Vector 才能读——否则读到半截数据。
- **读后写（先读完才能覆写）**：Vector 还没读完，MTE2 不能往同一块 UB 写新数据——否则未读数据被冲掉。

编程者可用的同步方式，按“谁访问同一块 Buffer”归类：

| # | 同步方式 | 适用场景 | 谁来插 |
|---|----------|----------|--------|
| 1 | **TQue `EnQue`/`DeQue` + `AllocTensor`/`FreeTensor`** | 两个不同单元（不同流水）接力访问同一 Buffer；框架把 Set/Wait 藏在队列操作里 | 框架（推荐） |
| 2 | **编译器自动同步** | LocalTensor 依赖关系清晰的部分场景 | 编译器 |
| 3 | **SetFlag/WaitFlag（事件对）** | 两个不同流水 + 一块 Buffer，手动点对点同步；`HardEvent::MTE2_V` 即“V 等 MTE2” | 开发者（ISASI）/ TQueSync（跨代） |
| 4 | **Mutex Lock/Unlock（950 新增）** | 把“这块 Buffer”抽象成一把锁，多个流水依次 Lock/Unlock 同一 MutexID | 开发者 |
| 5 | **PipeBarrier** | **同一个单元**先后两条指令访问同一 Buffer（如 MTE2 两次搬运地址重叠）——注意同队列顺序执行≠读写已完成 | 开发者 |
| 6 | **CrossCoreSetFlag/WaitFlag** | 访问同一块 Buffer 的两个单元不在同一个核（AIC 与 AIV 经 GM/直连通路交接） | 开发者/框架 |

一句话概括编程者视角：**写算子 = 给 4 类并行单元排流水 + 在每一处“两个单元共享一块 Buffer”的交接点选一种同步方式**。方式 1（TQue）是默认答案，3/4/5 是手动兜底，6 处理核间。

### 1.3 Ascend C 推荐范式：TPipe + TQue

Ascend C 把核内计算拆成可并行 Stage（典型三段：`CopyIn` / `Compute` / `CopyOut`），用队列完成 Stage 间通信与同步：

| 组件 | 职责 |
|------|------|
| **TPipe** | 统一管理片上内存、EventID / MutexID 等同步资源 |
| **TQue** | Stage 间通信与同步；`EnQue`/`DeQue`、`AllocTensor`/`FreeTensor` |
| **TPosition** | 逻辑位置（VECIN/VECOUT/A1/B1/CO1…），屏蔽物理 Buffer 差异 |

标准化四步：

1. **内存获取**：`AllocTensor` 或上游 `DeQue`
2. **计算执行**：搬运 / 计算 / 格式转换
3. **数据传递**：`EnQue` 推入下游并触发同步
4. **内存释放**：`FreeTensor`，通知可覆写

矢量伪代码骨架（来自编程指南）：

```cpp
AscendC::TPipe pipe;
AscendC::TQue<AscendC::TPosition::VecIn, 1> queIn;
AscendC::TQue<AscendC::TPosition::VecOut, 1> queOut;
pipe.InitBuffer(queIn, 2, 1024);   // num=2 开启 Double Buffer
pipe.InitBuffer(queOut, 2, 1024);

for (...) {
    { auto t = queIn.AllocTensor<T>(); DataCopy(t, gm, n); queIn.EnQue(t); }
    { auto t = queIn.DeQue<T>(); auto o = queOut.AllocTensor<T>();
      Abs(o, t, n); queIn.FreeTensor(t); queOut.EnQue(o); }
    { auto o = queOut.DeQue<T>(); DataCopy(gmOut, o, n); queOut.FreeTensor(o); }
}
```

### 1.4 Ascend 950（3510）对编程模型的扩展

相对 2201，950 在编程面上的关键变化：

| 能力 | 含义 | 对多 Pipeline 的影响 |
|------|------|----------------------|
| RegBase VF | Vector 从 MemBase 切到寄存器级计算 | Compute Stage 内部中间结果可少回写 UB |
| C-V 直连 | L0C→UB、UB→L1 | 融合算子少走 GM，流水更紧 |
| Fixpipe 增强 + DB | L0C 双目标搬运、L0C/UB 独立 Double Buffer | Cube/Vector 后处理流水更密 |
| Mutex | 核内锁式同步 | 替代部分 SetFlag/WaitFlag 反向同步 |
| CrossCore 模式 4 | AIV0/AIV1 可单独触发 AIC | CV 融合核间握手更细粒度 |
| SIMT / 混合编程 | 新增线程级并行 | 与 SIMD 流水并存时同步更复杂 |

官方仍推荐：**优先 TPipe/TQue 范式，让框架插同步**；静态 Tensor / 手写 SetFlag 仅在性能极限或框架覆盖不到时使用。

---

## 2. 多 Pipeline 与同步

### 2.1 流水类型

| 流水 | 典型用途 |
|------|----------|
| PIPE_S | Scalar（GetValue/SetValue 等） |
| PIPE_V | Vector 计算；部分架构下 L0C→UB |
| PIPE_M | Cube 矩阵乘 |
| PIPE_MTE1 | L1→L0A/L0B |
| PIPE_MTE2 | GM→L1 / GM→UB |
| PIPE_MTE3 | UB→GM |
| PIPE_FIX | L0C→GM / L0C→L1 |

### 2.2 同步分层

```
┌─────────────────────────────────────────────┐
│ 应用层：TQue EnQue/DeQue / Alloc/FreeTensor │  ← 推荐
├─────────────────────────────────────────────┤
│ 兼容层：TQueSync::SetFlag/WaitFlag          │  ← 跨代兼容
├─────────────────────────────────────────────┤
│ 硬件层：SetFlag/WaitFlag(ISASI)             │  ← event 点对点
│         Mutex Lock/Unlock(ISASI, 950+)      │  ← buffer/锁语义
│         PipeBarrier(ISASI)                  │  ← 单流水内
├─────────────────────────────────────────────┤
│ 核间层：CrossCoreSetFlag/WaitFlag           │
└─────────────────────────────────────────────┘
```

**多流水同步**：不同 PIPE 间有数据依赖时，用 SetFlag/WaitFlag 或 Mutex。  
**单流水同步**：同 PIPE 内地址重叠、乱序完成时用 `PipeBarrier`。

### 2.3 谁插同步？

| 场景 | 同步责任 |
|------|----------|
| TPipe/TQue 范式 | 框架在 EnQue/DeQue、Alloc/Free 中自动插 |
| LocalTensor 依赖清晰 | 编译器可自动插部分同步 |
| Cube 侧多数路径 | Ascend C 框架处理 |
| 静态 Tensor / 地址别名 / Scalar 原子与 MTE 争用 GM | **开发者手动插** |

Vector 典型依赖链：

1. MTE2 写完 UB → V 才能读（`MTE2_V`）
2. V 算完 → MTE3 才能搬出（`V_MTE3`）
3. 复用 Buffer：消费者读完 → 生产者才能再写（`V_MTE2` / `MTE3_V`）

Double Buffer 通过 `InitBuffer(que, 2, len)` 让“当前算 Buffer0、同时搬 Buffer1”重叠，隐藏搬运时延。

---

## 3. SetFlag / WaitFlag 与 BufferId 同步

### 3.1 SetFlag / WaitFlag（事件 / EventID）

语义：源流水与目的流水之间的一对硬件“锁”。

- **SetFlag\<Src_Dst\>(id)**：源流水前序读写完成后，把标志位置 1；**不阻塞**源流水后续指令。
- **WaitFlag\<Src_Dst\>(id)**：目的流水看到标志为 0 则阻塞；为 1 则清 0 并放行。

命名：`HardEvent::MTE2_V` 表示 **PIPE_V 等待 PIPE_MTE2**。

硬约束：

- 必须成对，且模板参数 + eventID 完全一致
- 同流水同 ID **禁止连续两次 SetFlag**（未定义行为，配合 `PipeBarrier<PIPE_ALL>` 可能卡死）
- 950 EventID 范围：**0–7**；TPipe 场景用 `AllocEventID` / `FetchEventID`，禁止手写冲突 ID
- 静态 Tensor：建议 0–5；6/7 接近系统预留（7 与自动同步相关）
- ISASI 接口**不保证跨硬件版本兼容**；跨代用 `TQueSync`

```cpp
int32_t eid = static_cast<int32_t>(
    GetTPipePtr()->FetchEventID(AscendC::HardEvent::S_MTE3));
AscendC::SetFlag<AscendC::HardEvent::S_MTE3>(eid);
AscendC::WaitFlag<AscendC::HardEvent::S_MTE3>(eid);
AscendC::DataCopy(dstGlobal, dstLocal, dataSize);
```

官方说明：**每个 EventID 对应一块存储数据的搬运/就绪状态**。  
因此在工程语义上，EventID 常按 **Buffer 维度** 分配——这就是“BufferId 式”使用 Event 的常见做法。

### 3.2 BufferId 同步：两种落地形态

#### A. TQue 内隐式 BufferId（框架路径）

`InitBuffer(que, num, len)` 创建 `num` 个 Buffer。每个 Buffer 槽位在 `TBuf` 上挂：

- `enQueEvtID`：生产者 `EnQue` → `SetFlag`；消费者 `DeQue` → `WaitFlag` + `ReleaseEventID`
- `freeBufEvtID`：消费者 `FreeTensor` → `SetFlag`；下次 `AllocTensor` → `WaitFlag`

即：**Buffer 槽位 ID ↔ EventID 绑定**，开发者不直接摸 ID，但流水正确性依赖“每块 Buffer 一条就绪事件”。

#### B. Ascend 950 MutexID（显式 Buffer/锁路径）

3510 新增 Mutex，类似 CPU 锁，按 **MutexID（可视为 Buffer 绑定的锁 ID）** 同步：

```cpp
uint8_t mutexId0 = AscendC::AllocMutexID();
uint8_t mutexId1 = AscendC::AllocMutexID();
// 双缓冲：偶数 tile 用 mutexId0，奇数用 mutexId1
AscendC::Mutex::Lock<PIPE_MTE2>(mutexId);
DataCopy(...);
AscendC::Mutex::Unlock<PIPE_MTE2>(mutexId);

AscendC::Mutex::Lock<PIPE_V>(mutexId);
Add(...);
AscendC::Mutex::Unlock<PIPE_V>(mutexId);

AscendC::Mutex::Lock<PIPE_MTE3>(mutexId);
DataCopy(...);
AscendC::Mutex::Unlock<PIPE_MTE3>(mutexId);
```

相对 SetFlag/WaitFlag 的官方优势：

1. **内聚性更强**：同步对象是 MutexID，不必为每对方向手写 `Src_Dst` 与反向事件
2. **可简化反向同步**：Buffer 复用时不必显式再插一套反向 HardEvent
3. **信号量更多**：MutexID 建议 0–27（28–31 系统预留），比 EventID 0–7 宽松

| 维度 | SetFlag/WaitFlag + EventID | Mutex Lock/Unlock + MutexID |
|------|---------------------------|-----------------------------|
| 绑定对象 | 源流水→目的流水 + eventID | MutexID + 被锁流水 |
| 语义 | 点对点就绪脉冲（set/clear） | 锁占用/释放 |
| ID 数量（950） | Event 0–7 | Mutex 约 0–27 |
| 反向同步 | 常需再插一对反向事件 | 同一 ID 上 Lock/Unlock 即可串起复用 |
| 兼容性 | ISASI；跨代用 TQueSync | 950/3510 新能力 |
| 典型用法 | 静态 Tensor、细粒度事件 | 双缓冲手写流水、与 Buffer 槽位 1:1 |

ATVOSS 等模板库侧的 **bufferid 静态分配**，是在更高层把 UB Buffer 槽位与调度路径在编译期定死，减少运行时动态抢占，使流水调度更确定——与“Buffer ↔ 同步原语绑定”是同一设计哲学的上浮。

---

## 4. 多 Pipeline 演进与易用性问题

### 4.1 演进脉络

```
手工指令 + 裸同步
    → 静态 Tensor + 手写 SetFlag/WaitFlag/PipeBarrier
        → TPipe/TQue：EnQue/DeQue 隐藏事件
            → 编译器自动同步（LocalTensor 依赖）
                → 950：Mutex、更细 CrossCore、C-V 直连、RegBase
                    → 模板库：CATLASS / ATVOSS（流水与 bufferid 下沉）
                        → 声明式后处理 EVG、算子直调 <<<>>>、对话式仿真
```

| 阶段 | 同步表达 | 易用性 |
|------|----------|--------|
| 裸事件 | HardEvent + 手管 EventID | 灵活，极易漏同步/ID 冲突 |
| TQue 范式 | 队列生命周期 | 主流推荐，正确性门槛下降 |
| 自动同步 | 编译器插桩 | 进一步减负，但依赖分析有边界 |
| 950 Mutex | Lock/Unlock(MutexID) | 双缓冲手写更直观，反向同步更简单 |
| 模板库 | 数学/图表达，隐藏流水 | 多数场景零手写同步；极限性能仍要下钻 |

### 4.2 易用性痛点（仍在）

1. **心智负担**：流水类型 × 事件方向 × ID 生命周期；双缓冲还要维护反向可覆写事件。  
2. **ID 稀缺与冲突**：EventID 仅 0–7；与框架/Matmul/SyncAll/自动同步预留冲突会导致**卡死/timeout**。  
3. **成对与顺序约束**：漏 Wait、连续 Set、模板参数不一致 → 难查的挂死。  
4. **抽象泄漏**：地址重叠、Scalar 原子 vs MTE、别名、RegBase/SIMT 混用时，框架/编译器不再包办。  
5. **跨代兼容**：ISASI（SetFlag、Mutex）绑定架构；950 新通路（L0C↔UB）改写融合数据流。  
6. **调试成本**：同步错误常表现为整卡 timeout，需依赖 cannsim 流水图 / msprof，而非普通断言。  
7. **模板库两面性**：ATVOSS/CATLASS 提升易用，但黑盒调度在 Bound 场景仍可能要回到手写 Pipeline。

### 4.3 实践建议（面向 950）

1. **默认 TPipe/TQue + Double Buffer**，不要先手写事件。  
2. 必须手写时：优先 **Mutex + 每 Buffer 一个 MutexID**（950）；跨代代码用 **TQueSync**。  
3. EventID/MutexID **一律 Alloc/Fetch**，用完 Release；禁止魔法数字踩 6/7 或系统预留。  
4. 融合算子优先走 **C-V 直连 + 模板（CATLASS/ATVOSS）**，核间用文档约定的 CrossCore 模式与 flagId 分段。  
5. 用 **仿真流水图** 验证气泡与同步，再上板。

---

## 5. 对比 GPU：多流水在性能发挥与硬件设计上的好处

先说清两种延迟隐藏思路的本质差别：

| | GPU（SIMT） | Ascend 多流水（DSA） |
|---|---|---|
| 隐藏访存延迟的手段 | 海量线程超额订阅：一个 warp 等访存，调度器切换到别的 warp | 独立异步引擎：搬运（MTE）与计算（V/M）本来就是不同流水，靠 Double Buffer 重叠 |
| 依赖管理 | 硬件 scoreboard / 调度器动态解决 | 软件（框架/编译器/开发者）显式插 Set/Wait、Mutex |
| 片上存储 | 硬件 Cache（L1/L2，带 tag 和一致性）+ 软件 Shared Memory | 纯软件管理的多级 Scratchpad（UB/L1/L0A/L0B/L0C） |
| 调度确定性 | 动态、不可精确预测 | 静态、指令队列内顺序执行，可预测 |

### 5.1 性能发挥上的好处

1. **延迟隐藏不依赖线程规模。** GPU 要把访存延迟藏住，需要足够多的活跃 warp（高 occupancy），寄存器压力大时 occupancy 掉、延迟就露出来。多流水只需要 2 个 Buffer（Double Buffer）就能把 MTE 搬运与 Vector/Cube 计算完全重叠——用**空间（一块额外 Buffer）**替代 GPU 的**并发线程数**，对片上资源占用的方式更直接、可控。

2. **搬运与计算是物理上独立的指令队列。** MTE2/MTE3/V/M/FIX 各自有队列深度（950 上 AIV 的 Vector 队列深 32、MTE2/MTE3 各 16），Scalar 发射后各单元独立推进。计算指令永远不会被搬运指令占住发射槽，反之亦然；GPU 上访存指令与计算指令共享 warp 的发射带宽，访存密集段会挤压计算发射。

3. **确定性调度让性能可预测、可逼近 roofline。** 指令在队列内顺序执行、同步点显式可见，配合仿真流水图可以静态推算每一拍的气泡在哪，规则负载（GEMM、Attention、逐元素）容易调到接近硬件上限。GPU 动态调度下同一 kernel 的重叠行为受 occupancy、cache 命中等运行时因素影响，调优更依赖统计画像。

4. **同步代价低且粒度精准。** SetFlag/WaitFlag 是一对硬件标志位指令，只约束**指定两条流水**，其他流水不受影响；Ascend 950 的 Mutex 进一步细化到 Buffer 粒度。GPU 内 block 级 `__syncthreads()` 是全 block barrier，粒度粗；跨 SM 同步要走 L2/全局内存原子操作，代价高得多。

5. **数据通路专用化，减少无谓搬运。** 950 的 L0C→UB、UB→L1 直连、Fixpipe 双目标搬运是为“矩阵结果 → 向量后处理”这条固定数据流专门修的路，Cube 输出不用绕 GM/共享内存即可进 Vector。GPU 上 Tensor Core 结果通常要经寄存器/Shared Memory 中转再做 epilogue。

6. **佐证：GPU 自身也在向显式多流水演进。** NVIDIA Ampere 引入 `cp.async`（异步拷贝绕过寄存器），Hopper 引入 TMA（专职搬运引擎）+ 异步 wgmma + mbarrier，CUTLASS/FlashAttention-3 用 warp specialization 把 warp 分成“搬运组/计算组”手工搭生产者-消费者流水——本质是在 SIMT 之上重建“MTE + 计算流水 + 事件同步”。这说明在规则的 AI 负载上，**显式解耦的搬运/计算流水是公认的高效结构**；Ascend 是把它直接做进指令集和硬件，GPU 是用软件在通用架构上模拟。

### 5.2 硬件设计上的好处

1. **面积和功耗花在算力而非调度上。** GPU 为支撑 SIMT 需要巨大的寄存器堆（每 SM 256KB 级）、warp 调度器、记分板、以及维持数万线程上下文的开销。多流水架构不需要为延迟隐藏保存海量线程状态，省下的面积/功耗可投给 Cube MAC 阵列与 SRAM，**同等工艺下计算密度和能效比更容易做高**——这是 NPU/DSA 路线的核心论据。

2. **控制逻辑简单，验证与时序收敛容易。** 各流水是顺序执行的指令队列 + 少量事件标志位，没有乱序、没有推测执行、没有复杂仲裁。相比动态调度器，这类控制路径面积小、频率好收、功能验证空间小。

3. **Scratchpad 代替 Cache：无 tag、无一致性协议、延迟确定。** UB/L1/L0 由软件显式编址，硬件不需要 tag 阵列、替换逻辑、跨核一致性（MESI 类）协议；同容量下有效存储密度更高，访问延迟固定（这也是确定性调度成立的前提）。GPU 的 L1/L2 cache 与一致性开销在规则 AI 负载上很多时候是“为不需要的灵活性买单”。

4. **专用引擎可以按数据流形状定制。** MTE 支持 ND-DMA、随路格式转换（NZ2DN）、随路量化等“搬运即变换”能力，Fixpipe 在搬出路径上做后处理；这些功能塞进通用 load/store 单元很困难。硬件按“GM→L1→L0→Cube→L0C→UB→GM”这条已知数据流分级建引擎，每级带宽/容量可以精确配比。

5. **同步资源是廉价的硬件原语。** 8 个 EventID + 一组 Mutex 标志位的硬件成本，远低于 GPU 为支持任意线程间同步所需的原子单元、内存序保障和 cache 一致性机制。

### 5.3 客观代价（对照）

好处不是免费的，反方向的代价同样明确：

1. **复杂度转移给软件**：同步正确性、tiling、Buffer 编排全部落在编译器/框架/开发者头上（即第 4 节的易用性问题）；GPU 的动态调度把这些藏进硬件，开发者写错的空间小。
2. **不规则负载吃亏**：动态形状、稀疏、数据依赖分支等场景，SIMT 的动态调度和 cache 天然适配；静态多流水容易出气泡，且 Scratchpad 需要软件预知访问模式。
3. **跨代兼容成本**：流水结构、Buffer 层次是 ISA 可见的（ISASI 接口不保证跨代兼容），硬件演进会传导为软件迁移工作；GPU 用 PTX/SASS 分层把这层震动挡掉了大半。
4. **生态门槛**：确定性性能的前提是工具链（自动同步、模板库、仿真）足够成熟，这正是 CANN 持续投入 CATLASS/ATVOSS/CANNSIM 的原因。

**小结**：在数据流规则、可静态规划的 AI 核心负载上，多流水以更少的硬件调度开销换取更高的计算密度、能效与可预测性能，且 GPU 近年（cp.async/TMA/warp specialization）的演进方向印证了这一结构的价值；其代价是把调度复杂度上移到软件栈，在不规则负载与生态成熟度上弱于 SIMT。

---

## 参考资料

- asc-devkit：`docs/zh/guide/编程指南/编程模型/.../TPipe-TQue框架编程原理.md`  
- asc-devkit：`docs/zh/api/.../SetFlag_WaitFlag_ISASI.md`、`Mutex_ISASI.md`、`intra_core_sync_overview.md`  
- asc-devkit：`docs/zh/asc_950_feature_guide.md`、`2201到3510架构变更.md`  
- asc-devkit 样例：`examples/.../05_sync_control/mutex`  
- 昇腾社区：同步控制简介、基于 TPipe/TQue 编程、Double Buffer 专题  
- CANN 社区文：《升级开发利器，释放 Ascend 950 算力》（CATLASS/ATVOSS/bufferid 静态分配）  
- GPU 对照：NVIDIA Ampere `cp.async`、Hopper TMA / `wgmma` / mbarrier（CUDA Programming Guide / PTX ISA）；CUTLASS 与 FlashAttention-3 的 warp specialization 生产者-消费者流水实践

---

## 一句话结论

Ascend 950 多 Pipeline 的本质是 **解耦的异步执行单元 + 显式同步**：SetFlag/WaitFlag 用 EventID 表达点对点就绪，BufferId（TQue 槽位事件或 950 MutexID）把同步钉在具体 Buffer 上；相比 GPU 用海量线程动态隐藏延迟，多流水以确定性调度和专用引擎换取更高计算密度与能效，代价是调度复杂度上移到软件栈——这正是 TQue、自动同步与模板库持续演进的原因。
