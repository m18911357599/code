# Ascend 950 多 Pipeline 编程分析

> 基于昇腾社区（CANN / Ascend C 文档）与 [asc-devkit](https://gitcode.com/cann/asc-devkit) 仓资料整理。  
> 适用芯片：Ascend 950PR / Ascend 950DT（`__NPU_ARCH__` = 3510）。

---

## 1. 编程模型

### 1.1 硬件前提：异步多流水

AI Core 内 Scalar 负责取指/译码/发射，把指令分发到各自独立的指令队列；Vector、Cube、MTE2/MTE3/MTE1、FixPipe 等单元**异步并行**执行。这是多 Pipeline 编程的硬件根基。

典型 Vector 数据路径：

```
GM --MTE2--> UB(VECIN) --PIPE_V--> UB(VECOUT) --MTE3--> GM
```

同一片 Local Memory 上若存在写后读 / 读后写依赖，就必须用同步约束时序。

### 1.2 Ascend C 推荐范式：TPipe + TQue

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

### 1.3 Ascend 950（3510）对编程模型的扩展

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

## 4. 生产者与消费者模型解读

### 4.1 两对依赖，两套 API

| 依赖类型 | 含义 | Ascend C 封装 | 底层同步 |
|----------|------|---------------|----------|
| **写后读（RAW）** | 生产者写完，消费者才能读 | `EnQue` / `DeQue` | Set / Wait（就绪） |
| **读后写（WAR，复用）** | 消费者读完，才能覆写 | `FreeTensor` / `AllocTensor` | Set / Wait（可覆写） |

解读要点：

- **EnQue = 生产者信号**：上游 Stage 写完后 `Set`，唤醒下游。
- **DeQue = 消费者等待**：下游 `Wait` 到就绪再读。
- **FreeTensor = 释放通知**：读侧用完后 `Set`“可覆写”。
- **AllocTensor = 申请等待**：写侧 `Wait` 到可覆写再占用 Buffer。

这把“硬件事件编排”翻译成了队列 + 内存生命周期，心智模型接近经典 **Queue Pipeline / 有界缓冲生产者-消费者**。

### 4.2 Double Buffer 下的环形缓冲

`BUFFER_NUM=2` 时，队列深度与 Buffer 数构成有界缓冲：

```
进度 i=0: 写 Buf0 → 算 Buf0 → ...
进度 i=1: 写 Buf1（可与算 Buf0 重叠）→ 算 Buf1
进度 i=2: Alloc Buf0 时 Wait“i=0 已 Free” → 再写 Buf0
```

同一数据分片上 CopyIn→Compute→CopyOut **串行**；不同分片、不同 Stage **并行**。生产者-消费者关系同时存在于：

1. **Stage 维**：MTE2 生产、V 消费；V 生产、MTE3 消费  
2. **Buffer 维**：偶数/奇数槽位各自持有独立 EventID 或 MutexID，避免乒乓冲突

### 4.3 融合与核间生产者-消费者（950）

CV 融合中常见：

- AIC（Cube）生产 L0C/GM 结果 → AIV（Vector）消费  
- 950：优先 L0C→UB 直连；仍需 `CrossCoreSetFlag/WaitFlag`（模式 2 为 1:2，模式 4 为 1:1）  
- flagId 是计数器语义（+1/-1），与核内 Event 的 0/1 脉冲不同；需成对、防冲突、注意计数上限（同 ID 最多约 15 次 Set）

---

## 5. 多 Pipeline 演进与易用性问题

### 5.1 演进脉络

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

### 5.2 易用性痛点（仍在）

1. **心智负担**：流水类型 × 事件方向 × ID 生命周期；双缓冲还要维护反向可覆写事件。  
2. **ID 稀缺与冲突**：EventID 仅 0–7；与框架/Matmul/SyncAll/自动同步预留冲突会导致**卡死/timeout**。  
3. **成对与顺序约束**：漏 Wait、连续 Set、模板参数不一致 → 难查的挂死。  
4. **抽象泄漏**：地址重叠、Scalar 原子 vs MTE、别名、RegBase/SIMT 混用时，框架/编译器不再包办。  
5. **跨代兼容**：ISASI（SetFlag、Mutex）绑定架构；950 新通路（L0C↔UB）改写融合数据流。  
6. **调试成本**：同步错误常表现为整卡 timeout，需依赖 cannsim 流水图 / msprof，而非普通断言。  
7. **模板库两面性**：ATVOSS/CATLASS 提升易用，但黑盒调度在 Bound 场景仍可能要回到手写 Pipeline。

### 5.3 实践建议（面向 950）

1. **默认 TPipe/TQue + Double Buffer**，不要先手写事件。  
2. 必须手写时：优先 **Mutex + 每 Buffer 一个 MutexID**（950）；跨代代码用 **TQueSync**。  
3. EventID/MutexID **一律 Alloc/Fetch**，用完 Release；禁止魔法数字踩 6/7 或系统预留。  
4. 融合算子优先走 **C-V 直连 + 模板（CATLASS/ATVOSS）**，核间用文档约定的 CrossCore 模式与 flagId 分段。  
5. 用 **仿真流水图** 验证气泡与同步，再上板。

---

## 参考资料

- asc-devkit：`docs/zh/guide/编程指南/编程模型/.../TPipe-TQue框架编程原理.md`  
- asc-devkit：`docs/zh/api/.../SetFlag_WaitFlag_ISASI.md`、`Mutex_ISASI.md`、`intra_core_sync_overview.md`  
- asc-devkit：`docs/zh/asc_950_feature_guide.md`、`2201到3510架构变更.md`  
- asc-devkit 样例：`examples/.../05_sync_control/mutex`  
- 昇腾社区：同步控制简介、基于 TPipe/TQue 编程、Double Buffer 专题  
- CANN 社区文：《升级开发利器，释放 Ascend 950 算力》（CATLASS/ATVOSS/bufferid 静态分配）

---

## 一句话结论

Ascend 950 多 Pipeline 的本质仍是 **异步执行单元 + 生产者/消费者有界缓冲**；SetFlag/WaitFlag 用 EventID 表达点对点就绪，BufferId（TQue 槽位事件或 950 MutexID）把同步钉在具体 Buffer 上；软件演进方向是把这些细节沉到 TQue、自动同步与模板库，但极限性能与特殊依赖场景仍要求理解并正确编排底层同步。
