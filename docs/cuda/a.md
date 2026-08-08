# CUDA L2 Persistent 综合分析（Host / Stream / Kernel）

> 对应本地分析稿：`d:/github/a.md`  
> 依据：[CUDA Programming Guide §4.13 L2 Cache Control](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/l2-cache-control.html)、Runtime API `cudaAccessPolicyWindow`、NVIDIA 论坛与 StackOverflow 上 Robert Crovella 的澄清。  
> 适用：Compute Capability ≥ 8.0，CUDA Runtime ≥ 11.0（kernel 侧 `annotated_ptr` 需 ≥ 11.5）。

---

## 0. 一句话结论

L2 Persistent **不是把数据钉死在 L2**，而是：

1. Host 先从 L2 划出一块 **set-aside（专留区）**；
2. Stream（或 Graph Kernel Node）声明一个 **Access Policy Window**（地址范围 + hitRatio + hit/miss 属性）；
3. 之后在该 Stream 上执行的 Kernel，访问窗口内地址时，按策略获得 **更高/更低的保留优先级**；
4. Kernel 侧还可用 `cuda::annotated_ptr` / PTX `createpolicy` 做 **单次访存级** 细化。

**Stream 是策略附着的主通道：策略跟 stream 走，命中效果落在 L2 全局 set-aside 上。**

---

## 1. 硬件语义：set-aside 与三种 Access Property

### 1.1 set-aside

| 概念 | 含义 |
|------|------|
| 普通 L2 | 所有访问按常规 LRU/替换策略竞争 |
| set-aside | 为 Persisting 访问优先保留的 L2 子集 |
| Persisting | 优先占用 set-aside，不易被挤出 |
| Normal / Streaming | 只能用 set-aside 中「未被 Persisting 占用」的空间；Streaming 更易被驱逐 |

查询：

- `cudaDeviceProp::l2CacheSize`
- `cudaDeviceProp::persistingL2CacheMaxSize`
- `cudaDeviceProp::accessPolicyMaxWindowSize`

限制：

- MIG：set-aside 功能禁用  
- MPS：运行时不能 `cudaDeviceSetLimit` 改大小，只能用环境变量  
  `CUDA_DEVICE_DEFAULT_PERSISTING_L2_CACHE_PERCENTAGE_LIMIT`

### 1.2 三种属性

| 属性 | 行为 |
|------|------|
| `cudaAccessPropertyPersisting` | 优先留在 set-aside |
| `cudaAccessPropertyStreaming` | 优先被驱逐（一次性流式读友好） |
| `cudaAccessPropertyNormal` | **强制清除**该窗口上先前的 Persisting 状态 |

---

## 2. 三层控制总览

```
┌─────────────────────────────────────────────────────────────┐
│ Host / Device / Context                                     │
│  · cudaDeviceSetLimit(PersistingL2CacheSize)  → 划容量      │
│  · cudaCtxResetPersistingL2Cache()            → 全局清线    │
│  · cudaDeviceGetLimit / GetDeviceProperties   → 查询        │
└────────────────────────────┬────────────────────────────────┘
                             │ 配额已就绪
┌────────────────────────────▼────────────────────────────────┐
│ Stream（主路径） / Graph Kernel Node                        │
│  · cudaStreamSetAttribute(..., AccessPolicyWindow, ...)     │
│  · 字段：base_ptr, num_bytes, hitRatio, hitProp, missProp   │
│  · 生效对象：随后在该 stream 上 launch 的 kernel             │
└────────────────────────────┬────────────────────────────────┘
                             │ launch 时携带/继承策略
┌────────────────────────────▼────────────────────────────────┐
│ Kernel                                                      │
│  · 被动：窗口内访存自动套用 stream/node 策略                │
│  · 主动：annotated_ptr / PTX createpolicy + L2::cache_hint  │
└─────────────────────────────────────────────────────────────┘
```

| 层级 | 控制什么 | 不控制什么 |
|------|----------|------------|
| **Host** | set-aside 大小、全局 reset、能力查询 | 具体哪段地址、哪次访问 |
| **Stream** | 地址窗口、hit/miss 比例与属性、对本 stream 后续 work 生效 | L2 物理替换细节；不独占 set-aside |
| **Kernel** | 真正产生带属性的访存；可 per-access 再标注 | 通常不负责划容量（那是 Host） |

---

## 3. Stream 模式深度分析（重点）

### 3.1 机制：策略挂在 Stream 上，不是挂在 Kernel 符号上

```cpp
cudaStreamAttrValue attr{};
attr.accessPolicyWindow.base_ptr  = data1;
attr.accessPolicyWindow.num_bytes = window_size;   // ≤ accessPolicyMaxWindowSize
attr.accessPolicyWindow.hitRatio  = 0.6f;
attr.accessPolicyWindow.hitProp   = cudaAccessPropertyPersisting;
attr.accessPolicyWindow.missProp  = cudaAccessPropertyStreaming;

cudaStreamSetAttribute(stream,
                       cudaStreamAttributeAccessPolicyWindow,
                       &attr);

// 之后：凡在该 stream 上执行的 kernel，访问 [data1, data1+window_size)
// 都按上述 window 获得 hit/miss 属性
kernelA<<<g, b, 0, stream>>>(data1);
kernelB<<<g, b, 0, stream>>>(data1);  // 同 stream，同样受益
```

关键语义：

1. **生效时机**：`SetAttribute` 之后、**随后**在该 stream 上执行的 kernel。不是“设完立刻把数据搬进 L2”。
2. **继承范围**：同 stream 上后续多个不同 kernel 都继承当前 window（直到被覆盖或 `num_bytes=0` 关闭）。
3. **关闭 window**：`num_bytes = 0` 再 `SetAttribute` 一次即可禁用。
4. **真正进 cache**：仍要靠 kernel 去 **touch** 那些地址；策略只改 eviction priority。

### 3.2 Access Policy Window 字段语义

| 字段 | 含义 | 注意 |
|------|------|------|
| `base_ptr` | 窗口起始全局地址 | Driver 可能对齐 |
| `num_bytes` | 窗口字节数 | 受 `accessPolicyMaxWindowSize` 限制；0 = 关闭 |
| `hitRatio` | 约该比例的段走 `hitProp`，其余走 `missProp` | 近似概率/分段，非精确逐地址公式 |
| `hitProp` | hit 段属性，常用 Persisting | — |
| `missProp` | miss 段属性，须为 Normal 或 Streaming | — |

文档对分段的描述：把窗口切成若干 segment，使  
`hit_segments / window ≈ hitRatio`，  
`miss_segments / window ≈ 1 - hitRatio`；  
具体切法“fitted to architecture”。业界共识（Crovella）：

- 粒度至少是 L2 line（长期为 32B，也可能更粗）；
- 哪些地址进 hit 段近似 **随机选定**，**不随访问模式动态改划分**；
- 因此 `hitRatio` 是静态分区 hint，不是运行时热度计数器。

### 3.3 hitRatio 怎么选（Stream 调参核心）

记：

- `S` = set-aside 大小  
- `W` = window `num_bytes`  
- `R` = `hitRatio`  
- 期望“有资格进 set-aside 的数据量” ≈ `W × R`

| 场景 | 建议 | 原因 |
|------|------|------|
| 热数据恰好 ≈ S | `W≈S, R=1` | 整窗保护，避免被其它流量挤掉 |
| 热数据 > S，且 A/B 循环扫 | `W>S, R≈S/W` | 防止整窗 thrashing；保住一部分持久命中 |
| 热数据 > S，但阶段内时间局部性很强 | 可试 `R=1` | 行为接近普通 cache，但窗口由用户限定 |
| **多 stream 并发各挂 window** | 各 stream 的 `W_i×R_i` 之和 ≲ S | set-aside **共享**；总和超容量则互相驱逐 |

官方例子（S=16KB）：

- 两 stream 各 W=16KB、R=1 → 易互相踢  
- 两 stream 各 W=16KB、R=0.5 → 更不易互踢

### 3.4 单 Stream 标准生命周期（推荐模板）

```text
Host:  SetLimit(PersistingL2CacheSize)
  │
Stream: SetAttribute(window on data1)          ← 打开策略
  │
Kernel×N on stream: 反复访问 data1             ← 填满并复用 set-aside
  │
Stream: num_bytes=0 → SetAttribute             ← 关闭策略窗口
Host:   cudaCtxResetPersistingL2Cache()        ← 清掉仍挂着的 persistent 线
  │
Kernel: 访问 data2（希望用满正常 L2）
```

对应官方示例骨架：

```cpp
// 1) 划容量
size_t size = min((size_t)(prop.l2CacheSize * 0.75), prop.persistingL2CacheMaxSize);
cudaDeviceSetLimit(cudaLimitPersistingL2CacheSize, size);

// 2) 挂 stream window
cudaStreamSetAttribute(stream, cudaStreamAttributeAccessPolicyWindow, &attr);

// 3) 同 stream 多轮复用
for (int i = 0; i < 10; ++i)
    kernelA<<<g, b, 0, stream>>>(data1);
kernelB<<<g, b, 0, stream>>>(data1);

// 4) 关 window + 清 L2 persistent 线
attr.accessPolicyWindow.num_bytes = 0;
cudaStreamSetAttribute(stream, cudaStreamAttributeAccessPolicyWindow, &attr);
cudaCtxResetPersistingL2Cache();

// 5) 后续 normal/streaming 工作
kernelC<<<g, b, 0, stream>>>(data2);
```

**为何必须 reset：** Persisting 线在 kernel 结束后仍可能占着 set-aside（persistence-after-use）。不 reset，后续 normal 访问等于少了一块 L2。

三种复位方式：

1. 对该区域再设 `hitProp = Normal`（定向清除）  
2. `cudaCtxResetPersistingL2Cache()`（上下文级一锅端）  
3. 长期不碰自动降级（时间不确定，**强烈不推荐依赖**）

### 3.5 多 Stream：策略独立，容量共享

这是 Stream 模式最容易误解的地方。

```text
Stream A: window(dataA, WA, RA) ──┐
                                  ├──► 共享同一块 set-aside S
Stream B: window(dataB, WB, RB) ──┘
```

要点：

| 问题 | 结论 |
|------|------|
| 每个 stream 能否有自己的 window？ | 能 |
| set-aside 是否按 stream 隔离？ | **否，全局共享** |
| 净占用如何估？ | ≈ Σ (并发 kernel 各自 W×R) |
| 超过 S 会怎样？ | Persisting 收益下降，互相驱逐 |
| Stream B 没设 window，读 Stream A 已驻留的数据？ | **可能命中**（线已在 L2），但 B **不会主动按 Persisting 属性把新数据推进 set-aside**；要稳定受益应给 B 也设 policy，或保证数据已由 A touch 且未 reset |
| 两 stream 对**重叠地址**设不同 policy？ | 设备侧可见顺序受各 stream 独立推进影响，存在“谁后到达谁覆盖”的竞态；**同一地址范围同一时刻只有一种有效 persistence 行为**（论坛澄清，非逐访问精确定义） |

实践建议：

1. 先列出可能 **并发** 的 stream/kernel 集合；  
2. 给每个分配预算 `budget_i`，使 `Σ budget_i ≤ S`；  
3. 令 `W_i × R_i ≈ budget_i`；  
4. 生命周期边界用 event 对齐后再统一 reset，避免一边还在用 Persisting、另一边已 Normal 扫大块。

### 3.6 Stream 与 default stream / 优先级 / 图

| 话题 | 说明 |
|------|------|
| 非 default stream | 常规用法；每个 stream 独立 attribute |
| default stream | 也可设 attribute，但与其它同步语义耦合，多 stream 场景更难推理 |
| Stream priority | 调度优先级 ≠ L2 驻留优先级；两者正交 |
| CUDA Graph | 用 `cudaGraphKernelNodeSetAttribute(..., AccessPolicyWindow)`，策略挂在 **节点** 上，适合固定拓扑里给特定 kernel 节点单独 window |
| 捕获进 Graph 的 stream attribute | 以 Graph/Node API 显式设置为准，不要假设“捕获时 stream 上的 window 一定完整迁移”（实现细节以当前 Toolkit 文档为准；稳妥做法是对 node 再设一次） |

### 3.7 Stream 模式处理流程（实现者视角）

可按驱动/运行时逻辑理解（概念模型，非公开源码）：

```text
cudaStreamSetAttribute(AccessPolicyWindow):
  将 window 描述符写入该 stream 的属性槽
  （base/size/ratio/props；可能做对齐与上限裁剪）

kernel<<<..., stream>>>:
  打包 launch 时附带“当前 stream 的 access policy”
  （或令 SM/MMU 侧在该 grid 执行期间启用该 window）

device 侧访存:
  if addr ∈ window:
      按预划分 hit/miss 段选择 Persisting 或 Streaming/Normal
      Persisting → 优先分配/保留 set-aside 行
  else:
      普通 L2 路径

其它 stream 的 grid:
  使用各自 stream 属性槽中的 window
  但争用同一 set-aside 容量与 tag

cudaCtxResetPersistingL2Cache:
  将仍标记为 persisting 的 L2 行降为 normal
```

因此调试时应区分三类问题：

1. **策略没挂上**：SetAttribute 时机错、launch 进了别的 stream、`num_bytes=0`  
2. **容量打爆**：多 stream `Σ W·R > S`  
3. **清场不净**：用完未 reset，拖慢后续阶段  

---

## 4. Host 层控制细节

```cpp
cudaDeviceSetLimit(cudaLimitPersistingL2CacheSize, size);
cudaDeviceGetLimit(&cur, cudaLimitPersistingL2CacheSize);
cudaCtxResetPersistingL2Cache();   // Driver: cuCtxResetPersistingL2Cache
```

Host 职责清单：

1. 查询能力与上限  
2. 按工作负载划 `S`（常用 ≤ 0.75 × L2，且 ≤ `persistingL2CacheMaxSize`）  
3. 编排各 stream 的 window 预算  
4. 在阶段切换点做 reset  
5. 处理 MIG/MPS 特殊约束  

Host **不能**单靠 SetLimit 指定“哪块全局内存常驻”；没有 window/注解，set-aside 只是空配额。

---

## 5. Kernel 层控制细节

### 5.1 被动模式（最常见）

Kernel 代码无改。只要：

- launch 使用挂了 window 的 stream；  
- 访问落在 `[base_ptr, base_ptr+num_bytes)`；  

硬件侧就会套用 hit/miss 属性。

### 5.2 主动模式（细粒度）

| 方式 | 粒度 | 说明 |
|------|------|------|
| `cuda::annotated_ptr` + `access_property` | 指针/访问 | libcu++，CUDA 11.5+ |
| PTX `createpolicy` + `ld/st ... L2::cache_hint` | 单条访存 | 可 fractional / range |
| PTX `applypriority` / `discard` | 已缓存行 | 改优先级或丢弃 |

与 Stream window 的关系：

- Stream window：适合“整段 buffer、整段 pipeline 阶段”  
- Kernel 注解：适合“同一 kernel 内表 A 要驻留、表 B 要 streaming”  
- 可组合，但重叠策略时以更具体的访存 hint / 设备可见顺序为准；生产代码应避免互相打架

---

## 6. Stream vs Graph Node vs Kernel 注解对照

| 维度 | Stream Attribute | Graph Kernel Node Attr | Kernel annotated_ptr / PTX |
|------|------------------|------------------------|----------------------------|
| 配置位置 | Host | Host（建图时） | Device 代码 |
| 作用域 | 该 stream 后续 kernels | 单个 graph node | 带注解的访问 |
| 改动成本 | 低（host 几行） | 中（图节点） | 高（改 kernel） |
| 多 kernel 复用同一窗口 | 天然适合 | 每节点设或共享参数 | 每处访问自己标 |
| 动态换窗口 | `SetAttribute` 覆盖即可 | 需更新 node / 重建 | 改指针属性或 policy 寄存器 |
| 典型场景 | 多迭代读同一权重/LUT | 固定 DAG 中热点节点 | 混合访问模式的复杂 kernel |

---

## 7. 场景化建议

### 7.1 推理：权重 / Embedding 表反复读

- Host：按表大小划 `S`  
- 单计算 stream：`W=表大小（或热点切片）, R=1`（若 `W≤S`）  
- 多 batch kernel 同 stream 连跑，吃驻留  
- 换模型或大阶段结束 → 关 window + reset  

### 7.2 训练：Producer–Consumer 链

- Producer stream 写出激活，Consumer stream 立刻读  
- **两 stream 都应对该缓冲设合理 window**，或合并到同一 stream 保证顺序与策略一致  
- 仅 producer 设 window、consumer 不设：consumer 可能碰巧命中，但不可靠  

### 7.3 多 Stream 流水（拷贝 ∥ 计算）

- H2D 流通常 **Streaming**（或根本不设 Persisting）  
- Compute 流对重用缓冲设 Persisting  
- 注意 DMA/拷贝与 compute 并发时不要把 set-aside 预算全打满在无复用数据上  

### 7.4 Histogram / 小表随机更新

- A100 白皮书用例：小表进 L2 Persistent 可显著降 DRAM 往返  
- `W` 对齐表大小，`R=1`，单 stream 即可  

---

## 8. 调试与验证清单

1. 确认 CC ≥ 8.0，且非 MIG。  
2. 打印 `l2CacheSize / persistingL2CacheMaxSize / accessPolicyMaxWindowSize`。  
3. 确认 kernel 的 **第 4 个 launch 参数** 真是挂了 attribute 的那个 stream。  
4. 用 Nsight Compute 看 L2 hit rate / DRAM 流量，对比开关 window。  
5. 多 stream 时单独测：只开 A、只开 B、A+B 并发，观察是否互踢。  
6. 阶段切换后若性能回退，检查是否忘记 `ResetPersistingL2Cache`。  
7. `hitRatio<1` 时不要期望“某固定地址一定常驻”——划分近似随机。  

---

## 9. 常见误区

| 误区 | 纠正 |
|------|------|
| SetAttribute 后数据已在 L2 | 否；要 kernel 访问才会装入 |
| 每个 stream 独占一块 L2 | 否；set-aside 共享 |
| 其它 stream 自动继承 window | 否；attribute 按 stream；已缓存行或许可读到 |
| `R=1` 且 `W>S` 一定最好 | 可能 thrashing；循环扫描时宜降低 R |
| 不 reset 也没关系 | persistence-after-use 会长期占坑 |
| 依赖自动降级 | 官方强烈不推荐 |
| 这是正确性功能 | 否，纯性能 hint |

---

## 10. 极简决策树（Stream 优先）

```text
数据是否被同一阶段内多次复用？
  ├─ 否 → 不要 Persisting；考虑 Streaming
  └─ 是 → Host 划 S
           │
           是否主要在一个 stream 内复用？
             ├─ 是 → 该 stream 设 window；W×R ≲ S
             └─ 否（多 stream 并发）→ 为每个并发 stream 分配预算
                                      Σ(W_i×R_i) ≲ S
           │
           同 kernel 内是否冷热访问混杂？
             ├─ 是 → 叠加 annotated_ptr / createpolicy
             └─ 否 → 仅 stream window 即可
           │
           阶段结束 → num_bytes=0 + CtxResetPersistingL2Cache
```

---

## 11. 参考

1. CUDA Programming Guide — [4.13 L2 Cache Control](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/l2-cache-control.html)  
2. CUDA Runtime — [`cudaAccessPolicyWindow`](https://docs.nvidia.com/cuda/cuda-runtime-api/structcudaAccessPolicyWindow.html)  
3. libcu++ — [`cuda::annotated_ptr` / access properties](https://nvidia.github.io/cccl/libcudacxx/extended_api/memory_access_properties/annotated_ptr.html)  
4. PTX ISA — `createpolicy` / `applypriority` / `discard` / `L2::cache_hint`  
5. Forum: [L2 persistence clarifications](https://forums.developer.nvidia.com/t/l2-persistence-clarifications/281031)  
6. Forum: [Persistent L2 API restrict in stream, what if in other stream?](https://forums.developer.nvidia.com/t/persistant-l2-api-restrict-in-stream-what-if-in-other-stream/278887)  
7. SO: [What is the L2 cache accessPolicyWindow introduced in CUDA 11](https://stackoverflow.com/questions/68359654/what-is-the-l2-cache-accesspolicywindow-introduced-in-cuda-11)  

---

## 12. Stream 模式小结

- **控制面在 Stream**：`AccessPolicyWindow` 是 stream 属性，决定后续 kernel 的地址窗口策略。  
- **数据面在 L2 全局**：set-aside 不按 stream 分片，多 stream 是容量复用与竞争关系。  
- **执行面在 Kernel**：不 touch 就不填充；可用注解做比 window 更细的控制。  
- **生命周期要闭环**：SetLimit → SetAttribute → 复用 → 关 window → Reset。  
- **调参抓手是 `W×R` 与并发集合**：先定预算，再设 window，而不是先 `R=1` 再到处挂。
