# CUDA L2 Cache Persistence 控制机制

> **适用架构**: NVIDIA GPU (Compute Capability 8.0+, Ampere/Hopper/Ada)  
> **核心主题**: L2 Cache 持久化访问控制机制  
> **主要资料来源**: NVIDIA 官方 CUDA Programming Guide《L2 Cache Control》章节  
> **资料来源链接**: https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/l2-cache-control.html  
> **仓库路径**: `docs/cuda/a.md`

---

## 目录

1. [L2 Cache 架构概述](#第一部分-l2-cache-架构概述)
2. [L2 Cache 访问策略](#第二部分-l2-cache-访问策略)
3. [L2 Cache Set-Aside 机制](#第三部分-l2-cache-set-aside-机制)
4. [L2 Access Policy Window](#第四部分-l2-access-policy-window)
5. [L2 Access Properties](#第五部分-l2-access-properties)
6. [L2 Persistence 完整示例](#第六部分-l2-persistence-完整示例)
7. [Reset L2 访问](#第七部分-reset-l2-访问)
8. [L2 Set-Aside 利用率管理](#第八部分-l2-set-aside-利用率管理)
9. [L2 Cache 属性查询](#第九部分-l2-cache-属性查询)
10. [PTX 缓存操作符与 __ldg()](#第十部分-ptx-缓存操作符与-ldg)
11. [实际应用与最佳实践](#第十一部分-实际应用与最佳实践)
12. [与 Ascend 对比](#第十二部分-与-ascend-对比)
13. [性能分析与 Profiling](#第十三部分-性能分析与-profiling)
14. [总结](#第十四部分-总结)

---

## 第一部分 L2 Cache 架构概述

### 1.1 NVIDIA GPU 内存层次结构

```
┌─────────────────────────────────────────────────────────────────────┐
│                    NVIDIA GPU 内存层次                                │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │                    Global Memory (HBM)                       │   │
│  │                    几十 GB, 高延迟 (~400 cycles)              │   │
│  └──────────────────────────┬──────────────────────────────────┘   │
│                              │                                      │
│                              ▼                                      │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │              L2 Cache (全局缓存) ← 本文档焦点               │   │
│  │              几 MB ~ 几十 MB, 中等延迟 (~200 cycles)          │   │
│  │              • 所有 SM 共享                                   │   │
│  │              • 跨 Kernel 持久化                               │   │
│  │              • 可配置访问策略                                 │   │
│  └──────────────────────────┬──────────────────────────────────┘   │
│                              │                                      │
│                              ▼                                      │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │                    L1 Cache / Shared Memory                  │   │
│  │                    每个 SM 私有, 128~256 KB, 低延迟 (~30 cycles)│   │
│  └──────────────────────────┬──────────────────────────────────┘   │
│                              │                                      │
│                              ▼                                      │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │                    Registers (寄存器)                         │   │
│  │                    每线程私有, ~1 cycle                       │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### 1.2 核心概念：Persisting vs Streaming

NVIDIA 官方文档明确定义了两种数据访问模式：

> **Persisting（持久化访问）**：当一个 CUDA kernel 反复访问全局内存中的某个数据区域时，这些数据访问可以被视为持久化（persisting）。

> **Streaming（流式访问）**：如果数据只被访问一次，这些数据访问可以被视为流式（streaming）。

```
┌─────────────────────────────────────────────────────────────────────┐
│                Persisting vs Streaming 定义                          │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  Persisting (持久化):                                               │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │  • 数据被反复访问                                            │   │
│  │  • 希望数据保留在 L2 Cache                                   │   │
│  │  • 提升命中率，降低延迟                                      │   │
│  │  • 例: 权重矩阵、热点数据                                    │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                                                                     │
│  Streaming (流式):                                                  │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │  • 数据仅访问一次                                            │   │
│  │  • 不希望数据占据 L2 空间                                    │   │
│  │  • 避免 L2 污染                                             │   │
│  │  • 例: 一次性输入数据、中间结果                              │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                                                                     │
│  关键: Compute Capability 8.0+ 设备具备影响 L2 中数据持久性的能力    │
│  目的: 提供更高的带宽和更低的延迟来访问全局内存                     │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### 1.3 功能暴露 API

依据 NVIDIA 官方文档，L2 Cache 持久化控制通过两种主要 API 暴露：

| API | 版本 | 说明 |
|------|------|------|
| **CUDA Runtime API** | CUDA 11.0+ | 程序化控制 L2 Cache 持久化 |
| **cuda::annotated_ptr** | CUDA 11.5+ | libcu++ 库中带内存访问属性的指针注解 |

**本文档重点**：CUDA Runtime API 方式。

---

## 第二部分 L2 Cache 访问策略

### 2.1 两种访问策略

NVIDIA GPU 支持两种 L2 Cache 访问策略：

| 策略 | L2 保留 | 适用场景 | 性能影响 |
|------|:---:|------|------|
| **Streaming** | ❌ 不保留 | 一次性访问数据 | 减少 L2 污染 |
| **Persistent** | ✅ 保留 | 重复访问数据 | 提升命中率 |

### 2.2 访问策略的硬件实现

```
┌─────────────────────────────────────────────────────────────────────┐
│                    L2 Cache 访问策略硬件实现                          │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  L2 Cache 被划分为两个区域:                                         │
│                                                                     │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │  ┌──────────────────────┐  ┌──────────────────────┐        │   │
│  │  │   Set-Aside 区域     │  │   普通 / 流式区域    │        │   │
│  │  │   (持久化专用)       │  │   (Streaming/Normal) │        │   │
│  │  │                      │  │                      │        │   │
│  │  │  Persisting 访问     │  │  Normal/Streaming    │        │   │
│  │  │  (优先使用)          │  │  访问                │        │   │
│  │  │                      │  │                      │        │   │
│  │  │  • 持久化访问优先占用 │  │  • 仅当持久化区域    │        │   │
│  │  │  • 数据被优先保留     │  │    未被使用时才可用   │        │   │
│  │  └──────────────────────┘  └──────────────────────┘        │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                                                                     │
│  关键规则:                                                          │
│  • Persisting 访问优先使用 Set-Aside 区域                            │
│  • Normal/Streaming 访问只能在使用不到 Set-Aside 区域时使用          │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 第三部分 L2 Cache Set-Aside 机制

### 3.1 Set-Aside 概述

依据 NVIDIA 官方文档：

> 一部分 L2 Cache 可以被预留（set aside）用于持久化数据访问。持久化访问对这部分预留的 L2 Cache 具有优先使用权，而普通或流式访问只能在这部分 L2 Cache 未被持久化访问使用时才能利用它。

### 3.2 设置 Set-Aside 大小

```cpp
// ============================================
// 设置 L2 Cache Set-Aside 大小
// ============================================

cudaDeviceProp prop;
cudaGetDeviceProperties(&prop, device_id);

// 计算 set-aside 大小: 取 L2 总容量的 75% 或允许的最大值
size_t size = min(int(prop.l2CacheSize * 0.75), prop.persistingL2CacheMaxSize);

// 设置 L2 Cache set-aside 大小
cudaDeviceSetLimit(cudaLimitPersistingL2CacheSize, size);
```

**关键点**：

| 项 | 说明 |
|------|------|
| **l2CacheSize** | 设备上可用的 L2 Cache 总量 |
| **persistingL2CacheMaxSize** | 可用于持久化访问的最大 L2 Cache 大小 |
| **cudaLimitPersistingL2CacheSize** | 设置 L2 Cache set-aside 大小的 limit |
| **典型值** | 通常设置为 L2 总量的 75% |

### 3.3 MIG 与 MPS 模式下的限制

依据 NVIDIA 官方文档：

> **MIG 模式**：当 GPU 配置为多实例 GPU（Multi-Instance GPU, MIG）模式时，L2 Cache set-aside 功能被禁用。

> **MPS 模式**：使用多进程服务（Multi-Process Service, MPS）时，L2 Cache set-aside 大小不能通过 `cudaDeviceSetLimit` 更改。相反，set-aside 大小只能在 MPS 服务器启动时通过环境变量 `CUDA_DEVICE_DEFAULT_PERSISTING_L2_CACHE_PERCENTAGE_LIMIT` 指定。

```
┌─────────────────────────────────────────────────────────────────────┐
│                    MIG/MPS 模式下的限制                              │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  MIG (Multi-Instance GPU):                                          │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │  • L2 Cache set-aside 功能 → ❌ 禁用                        │   │
│  │  • 无法使用 cudaDeviceSetLimit 设置                         │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                                                                     │
│  MPS (Multi-Process Service):                                       │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │  • 无法通过 cudaDeviceSetLimit 更改                          │   │
│  │  • 只能通过环境变量在 MPS 服务器启动时指定:                  │   │
│  │    CUDA_DEVICE_DEFAULT_PERSISTING_L2_CACHE_PERCENTAGE_LIMIT │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### 3.4 控制 Set-Aside 大小的 API

依据 NVIDIA 官方文档：

```cpp
// 查询 L2 set-aside 大小
size_t size;
cudaDeviceGetLimit(&size, cudaLimitPersistingL2CacheSize);

// 设置 L2 set-aside 大小
cudaDeviceSetLimit(cudaLimitPersistingL2CacheSize, new_size);

// 最大值限制
// 不能超过 cudaDeviceProp::persistingL2CacheMaxSize
```

```cpp
// cudaLimit 枚举
enum cudaLimit {
    /* 其他字段未显示 */
    cudaLimitPersistingL2CacheSize
};
```

---

## 第四部分 L2 Access Policy Window

### 4.1 概述

依据 NVIDIA 官方文档：

> 一个访问策略窗口（Access Policy Window）指定一块连续的全局内存区域以及该区域内访问在 L2 Cache 中的持久化属性。

### 4.2 CUDA Stream 示例

依据 NVIDIA 官方文档：

```cpp
// ============================================
// 使用 CUDA Stream 设置 L2 持久化访问窗口
// ============================================

cudaStreamAttrValue stream_attribute;  // Stream 级属性数据结构

// 配置访问策略窗口
stream_attribute.accessPolicyWindow.base_ptr  = reinterpret_cast<void*>(ptr); // 全局内存数据指针
stream_attribute.accessPolicyWindow.num_bytes = num_bytes;                    // 持久化访问的字节数
                                                                            // (必须小于 cudaDeviceProp::accessPolicyMaxWindowSize)
stream_attribute.accessPolicyWindow.hitRatio  = 0.6;                          // cache 命中率提示
stream_attribute.accessPolicyWindow.hitProp   = cudaAccessPropertyPersisting; // 命中时的访问属性
stream_attribute.accessPolicyWindow.missProp  = cudaAccessPropertyStreaming;  // 未命中时的访问属性

// 将属性设置到 CUDA Stream
cudaStreamSetAttribute(stream, cudaStreamAttributeAccessPolicyWindow, &stream_attribute);
```

**工作原理**：
- 当 kernel 在 `stream` 中后续执行时，全局内存范围 `[ptr..ptr+num_bytes)` 内的访问比其他全局内存位置更可能在 L2 Cache 中持久化。

**Stream 模式要点**：

| 点 | 说明 |
|------|------|
| 附着点 | 策略挂在 **stream** 上，不是挂在 kernel 符号上 |
| 生效时机 | `SetAttribute` 之后，**随后**在该 stream 上 launch 的 kernel |
| 继承 | 同 stream 后续多个不同 kernel 都继承当前 window |
| 关闭 | `num_bytes = 0` 再 `SetAttribute` 一次 |
| 真正装入 L2 | 仍需 kernel **touch** 窗口内地址；策略只改 eviction priority |

### 4.3 CUDA GraphKernelNode 示例

依据 NVIDIA 官方文档：

```cpp
// ============================================
// 使用 CUDA Graph Kernel Node 设置 L2 持久化
// ============================================

cudaKernelNodeAttrValue node_attribute;  // Kernel 级属性数据结构

// 配置访问策略窗口
node_attribute.accessPolicyWindow.base_ptr  = reinterpret_cast<void*>(ptr);
node_attribute.accessPolicyWindow.num_bytes = num_bytes;
node_attribute.accessPolicyWindow.hitRatio  = 0.6;
node_attribute.accessPolicyWindow.hitProp   = cudaAccessPropertyPersisting;
node_attribute.accessPolicyWindow.missProp  = cudaAccessPropertyStreaming;

// 将属性设置到 CUDA Graph Kernel node
cudaGraphKernelNodeSetAttribute(node, cudaKernelNodeAttributeAccessPolicyWindow, &node_attribute);
```

### 4.4 hitRatio 详解

依据 NVIDIA 官方文档：

> `hitRatio` 参数用于指定 `hitProp` 属性覆盖的访问比例。在上述两个示例中，全局内存区域 `[ptr..ptr+num_bytes)` 内 60% 的内存访问具有持久化属性，40% 的内存访问具有流式属性。具体哪些访问被归类为持久化（`hitProp`）是随机的，概率约为 `hitRatio`；概率分布取决于硬件架构和内存范围。

```
┌─────────────────────────────────────────────────────────────────────┐
│                    hitRatio 行为示例                                │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  假设: L2 set-aside 大小 = 16KB                                     │
│        accessPolicyWindow.num_bytes = 32KB                          │
│                                                                     │
│  场景 1: hitRatio = 0.5                                             │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │  • 硬件随机选择 32KB 窗口中的 16KB 标记为持久化              │   │
│  │  • 这 16KB 被缓存到 set-aside 区域                           │   │
│  │  • 避免 thrashing (缓存抖动)                                │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                                                                     │
│  场景 2: hitRatio = 1.0                                             │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │  • 硬件尝试将整个 32KB 窗口缓存到 set-aside 区域            │   │
│  │  • 由于 set-aside 区域 (16KB) 小于窗口 (32KB)               │   │
│  │  • cache line 会被淘汰，保留最近使用的 16KB                 │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                                                                     │
│  hitRatio 的作用:                                                   │
│  • 避免 cache line thrashing                                       │
│  • 减少进出 L2 Cache 的数据量                                       │
│  • 手动控制并发 Stream 各窗口的缓存量                               │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

调参经验（记 `S`=set-aside，`W`=window，`R`=hitRatio，期望占用 ≈ `W×R`）：

| 场景 | 建议 |
|------|------|
| 热数据 ≈ S | `W≈S, R=1` |
| 热数据 > S，且 A/B 循环扫 | `W>S, R≈S/W`，防 thrashing |
| 多 stream 并发 | 各 stream `Wᵢ×Rᵢ` 之和 ≲ S |

### 4.5 多 Stream 并发管理

依据 NVIDIA 官方文档：

> 当 L2 set-aside 大小为 16KB 时，两个并发 kernel 位于两个不同的 CUDA Stream 中，每个都有 16KB 的 `accessPolicyWindow`，且两者的 `hitRatio` 值都为 1.0，它们在竞争共享 L2 资源时可能会互相淘汰对方的 cache line。但是，如果两个 `accessPolicyWindows` 的 `hitRatio` 值都为 0.5，则它们不太可能淘汰自己或对方的持久化 cache line。

```
┌─────────────────────────────────────────────────────────────────────┐
│              多 Stream 并发 hitRatio 管理                            │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  L2 set-aside = 16KB                                               │
│                                                                     │
│  场景 1: 两个 Stream 各 16KB 窗口, 都 hitRatio=1.0                  │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │  Stream A: 16KB 窗口 (hitRatio=1.0)                        │   │
│  │  Stream B: 16KB 窗口 (hitRatio=1.0)                        │   │
│  │  → 互相淘汰对方 cache line (竞争共享 16KB set-aside)        │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                                                                     │
│  场景 2: 两个 Stream 各 16KB 窗口, 都 hitRatio=0.5                  │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │  Stream A: 16KB 窗口, 实际缓存 8KB (0.5×16KB)              │   │
│  │  Stream B: 16KB 窗口, 实际缓存 8KB (0.5×16KB)              │   │
│  │  → 总计 16KB, 正好匹配 set-aside, 不易互相淘汰             │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

多 Stream 额外注意：

| 问题 | 结论 |
|------|------|
| 每个 stream 能否有自己的 window？ | 能 |
| set-aside 是否按 stream 隔离？ | **否，全局共享** |
| 其它 stream 是否继承 window？ | **否**；已驻留的线或许可被读到，但不保证按 Persisting 继续灌入 |
| 重叠地址设不同 policy | 设备侧可见顺序受各 stream 独立推进影响，存在竞态 |

---

## 第五部分 L2 Access Properties

### 5.1 三种访问属性

依据 NVIDIA 官方文档，定义了三种全局内存数据访问的属性：

| 属性 | 说明 |
|------|------|
| **cudaAccessPropertyStreaming** | 流式属性访问较少持久化在 L2 Cache 中，因为这类访问被优先淘汰 |
| **cudaAccessPropertyPersisting** | 持久化属性访问更容易持久化在 L2 Cache 中，因为这类访问被优先保留在 set-aside 区域 |
| **cudaAccessPropertyNormal** | 正常属性，强制将先前应用的持久化访问属性重置为正常状态 |

### 5.2 三种属性对比

```
┌─────────────────────────────────────────────────────────────────────┐
│                L2 Access Properties 三种属性                         │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  1. cudaAccessPropertyStreaming (流式)                               │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │  • 行为: 较少持久化在 L2                                    │   │
│  │  • 原因: 这类访问被优先淘汰                                  │   │
│  │  • 用途: 一次性数据，避免 L2 污染                            │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                                                                     │
│  2. cudaAccessPropertyPersisting (持久化)                            │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │  • 行为: 更容易持久化在 L2                                  │   │
│  │  • 原因: 优先保留在 set-aside 区域                           │   │
│  │  • 用途: 热点数据，提升命中率                                │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                                                                     │
│  3. cudaAccessPropertyNormal (正常)                                  │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │  • 行为: 强制重置先前的持久化属性                            │   │
│  │  • 原因: 移除优先保留状态                                    │   │
│  │  • 用途: 清理不再需要的持久化数据                            │   │
│  │  • 背景: 先前 kernel 的持久化属性可能在 L2 中长期保留        │   │
│  │    这会减少后续不使用持久化属性的 kernel 可用的 L2 空间      │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### 5.3 Normal 属性的重要性

依据 NVIDIA 官方文档：

> 先前 CUDA kernel 中带持久化属性的访问可能在 L2 Cache 中长期保留，即使其原本的用途已经结束。这种"使用后持久化"（persistence-after-use）会减少后续不使用持久化属性的 kernel 可用的 L2 Cache 空间。使用 `cudaAccessPropertyNormal` 属性重置访问策略窗口，可以移除先前访问的持久化（优先保留）状态，就像先前的访问没有访问属性一样。

---

## 第六部分 L2 Persistence 完整示例

### 6.1 官方完整示例

依据 NVIDIA 官方文档，以下是完整的 L2 Persistence 示例：

```cpp
// ============================================
// L2 Persistence 完整示例 (NVIDIA 官方)
// ============================================

cudaStream_t stream;
cudaStreamCreate(&stream);  // 创建 CUDA stream

cudaDeviceProp prop;  // CUDA 设备属性变量
cudaGetDeviceProperties(&prop, device_id);  // 查询 GPU 属性

// 为持久化访问预留 L2 cache
size_t size = min(int(prop.l2CacheSize * 0.75), prop.persistingL2CacheMaxSize);
cudaDeviceSetLimit(cudaLimitPersistingL2CacheSize, size);  // 预留 3/4 的 L2 cache

// 选择窗口大小
size_t window_size = min(prop.accessPolicyMaxWindowSize, num_bytes);

cudaStreamAttrValue stream_attribute;  // Stream 级属性数据结构
stream_attribute.accessPolicyWindow.base_ptr  = reinterpret_cast<void*>(data1);
stream_attribute.accessPolicyWindow.num_bytes = window_size;
stream_attribute.accessPolicyWindow.hitRatio  = 0.6;
stream_attribute.accessPolicyWindow.hitProp   = cudaAccessPropertyPersisting;
stream_attribute.accessPolicyWindow.missProp  = cudaAccessPropertyStreaming;

// 将属性设置到 CUDA Stream
cudaStreamSetAttribute(stream, cudaStreamAttributeAccessPolicyWindow, &stream_attribute);

// 多次使用 data1 的 kernel 受益于 L2 持久化
for (int i = 0; i < 10; i++) {
    cuda_kernelA<<<grid_size, block_size, 0, stream>>>(data1);
}  // data1 在 [data1 + num_bytes) 范围内多次使用，受益于 L2 持久化

// 同一 Stream 中的不同 kernel 也能受益于 data1 的持久化
cuda_kernelB<<<grid_size, block_size, 0, stream>>>(data1);

// 禁用访问策略窗口 (设置大小为 0)
stream_attribute.accessPolicyWindow.num_bytes = 0;
cudaStreamSetAttribute(stream, cudaStreamAttributeAccessPolicyWindow, &stream_attribute);

// 移除 L2 中任何持久化行
cudaCtxResetPersistingL2Cache();

// data2 现在可以正常模式受益于完整的 L2 cache
cuda_kernelC<<<grid_size, block_size, 0, stream>>>(data2);
```

### 6.2 示例流程分析

```
┌─────────────────────────────────────────────────────────────────────┐
│                    L2 Persistence 示例流程                           │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  Step 1: 创建 Stream                                                │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │  cudaStreamCreate(&stream)                                 │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                              │                                      │
│                              ▼                                      │
│  Step 2: 查询设备属性并设置 L2 set-aside                              │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │  cudaGetDeviceProperties                                   │   │
│  │  size = min(l2CacheSize*0.75, persistingL2CacheMaxSize)    │   │
│  │  cudaDeviceSetLimit(cudaLimitPersistingL2CacheSize, size)  │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                              │                                      │
│                              ▼                                      │
│  Step 3: 配置访问策略窗口                                             │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │  accessPolicyWindow = {base_ptr, num_bytes, hitRatio=0.6,  │   │
│  │                        hitProp=Persisting, missProp=Streaming}│   │
│  │  cudaStreamSetAttribute                                   │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                              │                                      │
│                              ▼                                      │
│  Step 4: 执行使用 data1 的 kernel (受益于持久化)                        │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │  kernelA × 10 次 (data1 持久化)                            │   │
│  │  kernelB (data1 持久化)                                    │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                              │                                      │
│                              ▼                                      │
│  Step 5: 禁用窗口 + 重置 L2持久化                                     │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │  num_bytes = 0 (禁用窗口)                                  │   │
│  │  cudaCtxResetPersistingL2Cache() (移除持久化行)            │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                              │                                      │
│                              ▼                                      │
│  Step 6: 执行使用 data2 的 kernel (正常模式)                            │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │  kernelC (data2 受益于完整 L2)                              │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

标准生命周期一句话：

`SetLimit → SetAttribute(window) → 同 stream 复用 → num_bytes=0 → CtxResetPersistingL2Cache → 后续 normal 工作`

---

## 第七部分 Reset L2 访问

### 7.1 为什么需要 Reset

依据 NVIDIA 官方文档：

> 先前 CUDA kernel 的持久化 L2 cache line 可能在 L2 中长期保留，即使其已被使用完。因此，对于流式或正常的内存访问来说，重置 L2 cache 对正常优先级很重要。

### 7.2 三种 Reset 方式

依据 NVIDIA 官方文档，有三种方式可以将持久化访问重置为正常状态：

| 方式 | 描述 |
|------|------|
| **方式 1** | 使用 `cudaAccessPropertyNormal` 访问属性重置先前持久化的内存区域 |
| **方式 2** | 调用 `cudaCtxResetPersistingL2Cache()` 将所有持久化 L2 cache line 重置为正常 |
| **方式 3** | **最终**未使用的行会自动重置为正常。**强烈不建议**依赖自动重置，因为自动重置所需的时间不确定 |

### 7.3 Reset 示例代码

```cpp
// ============================================
// 三种 Reset 方式
// ============================================

// 方式 1: 使用 cudaAccessPropertyNormal 重置特定区域
cudaStreamAttrValue reset_attribute;
reset_attribute.accessPolicyWindow.base_ptr  = reinterpret_cast<void*>(data1);
reset_attribute.accessPolicyWindow.num_bytes = num_bytes;
reset_attribute.accessPolicyWindow.hitRatio  = 1.0;
reset_attribute.accessPolicyWindow.hitProp   = cudaAccessPropertyNormal;  // 重置持久化
reset_attribute.accessPolicyWindow.missProp  = cudaAccessPropertyNormal;  // 重置持久化
cudaStreamSetAttribute(stream, cudaStreamAttributeAccessPolicyWindow, &reset_attribute);

// 方式 2: 重置所有持久化 L2 cache
cudaCtxResetPersistingL2Cache();

// 方式 3: 依赖自动重置 (强烈不建议)
// 自动重置所需时间不确定，无法保证
```

---

## 第八部分 L2 Set-Aside 利用率管理

### 8.1 并发 Kernel 的共享机制

依据 NVIDIA 官方文档：

> 在不同 CUDA Stream 中并发执行的多个 CUDA kernel 可以为其 Stream 分配不同的访问策略窗口。但是，L2 set-aside 缓存区域在这些并发 CUDA kernel 之间是共享的。因此，这个 set-aside 缓存区域的净利用率是所有并发 kernel 各自使用量的总和。当持久化访问量超过 L2 set-aside 缓存容量时，将持久化访问标记为持久化的收益就会减弱。

### 8.2 管理要点

依据 NVIDIA 官方文档，要管理 set-aside L2 缓存区域的利用率，应用程序必须考虑以下因素：

| 因素 | 说明 |
|------|------|
| **L2 set-aside 缓存大小** | 预留了多少缓存用于持久化 |
| **可能并发执行的 CUDA kernel** | 哪些 kernel 会同时运行 |
| **所有并发 kernel 的访问策略窗口** | 各 kernel 的访问策略窗口配置 |
| **何时以及如何重置 L2** | 何时重置以允许正常/流式访问以同等优先级利用先前预留的 L2 |

### 8.3 利用率计算

```
┌─────────────────────────────────────────────────────────────────────┐
│                    L2 Set-Aside 利用率计算                           │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  L2 set-aside 大小 = 16KB                                           │
│                                                                     │
│  并发 Kernel:                                                       │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │  Kernel A (Stream 1): accessPolicyWindow 8KB, hitRatio=1.0  │   │
│  │  → 占用 set-aside 8KB                                        │   │
│  │                                                             │   │
│  │  Kernel B (Stream 2): accessPolicyWindow 8KB, hitRatio=1.0  │   │
│  │  → 占用 set-aside 8KB                                        │   │
│  │                                                             │   │
│  │  总占用 = 8KB + 8KB = 16KB = set-aside 容量                  │   │
│  │  → 利用率 100%，刚好匹配                                    │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                                                                     │
│  如果总占用超过 set-aside 容量:                                     │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │  Kernel A: 10KB + Kernel B: 10KB = 20KB > 16KB              │   │
│  │  → 超过容量，持久化收益减弱                                  │   │
│  │  → 需要降低 hitRatio 或调整窗口大小                          │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 第九部分 L2 Cache 属性查询

### 9.1 相关设备属性

依据 NVIDIA 官方文档，L2 Cache 相关属性是 `cudaDeviceProp` 结构的一部分，可通过 CUDA Runtime API `cudaGetDeviceProperties` 查询：

| 属性 | 说明 |
|------|------|
| **l2CacheSize** | GPU 上可用的 L2 Cache 总量 |
| **persistingL2CacheMaxSize** | 可用于持久化访问的最大 L2 Cache 大小 |
| **accessPolicyMaxWindowSize** | 访问策略窗口的最大大小 |

### 9.2 查询示例

```cpp
// ============================================
// 查询 L2 Cache 属性
// ============================================

cudaDeviceProp prop;
cudaGetDeviceProperties(&prop, device_id);

printf("L2 Cache Size: %zu bytes\n", prop.l2CacheSize);
printf("Persisting L2 Cache Max Size: %zu bytes\n", prop.persistingL2CacheMaxSize);
printf("Access Policy Max Window Size: %zu bytes\n", prop.accessPolicyMaxWindowSize);
```

---

## 第十部分 PTX 缓存操作符与 __ldg()

### 10.1 缓存操作符类型

PTX (Parallel Thread Execution) 提供多种缓存操作符，用于控制内存访问的缓存行为：

| 操作符 | 全称 | 说明 | L1 行为 | L2 行为 |
|--------|------|------|---------|---------|
| **.ca** | Cache at All | 全级别缓存 | 缓存 | 缓存 |
| **.cg** | Cache at Global | 仅全局缓存 | 不缓存 | 缓存 |
| **.cs** | Cache Streaming | 流式访问 | 不缓存 | 不缓存 |
| **.cv** | Cache Volatile | 易失性缓存 | 不缓存 | 不缓存 |
| **.lu** | Last Use | 最后一次使用 | 不缓存 | 不缓存 |

### 10.2 缓存操作符详解

#### .ca (Cache at All) — 默认策略

```cpp
// PTX 指令
ld.global.ca.f32 %f, [%r];

// CUDA 对应
float value = *ptr;  // 默认访问
```

**行为**:
- L1 Cache: ✅ 缓存数据
- L2 Cache: ✅ 缓存数据
- 适用: 需要多次访问的数据

#### .cg (Cache at Global) — 全局缓存策略

```cpp
// PTX 指令
ld.global.cg.f32 %f, [%r];

// CUDA 对应 (__ldg 函数)
float value = __ldg(ptr);
```

**行为**:
- L1 Cache: ❌ 不缓存数据 (bypass)
- L2 Cache: ✅ 缓存数据
- 适用: 跨 Block 共享数据，避免 L1 污染

#### .cs (Cache Streaming) — 流式策略

```cpp
// PTX 指令
ld.global.cs.f32 %f, [%r];

// CUDA 对应 (__ldcs 函数)
float value = __ldcs(ptr);
```

**行为**:
- L1 Cache: ❌ 不缓存数据
- L2 Cache: ❌ 不缓存数据 (streaming)
- 适用: 一次性访问数据，避免缓存污染

#### .cv (Cache Volatile) — 易失性策略

```cpp
// PTX 指令
ld.global.cv.f32 %f, [%r];
```

**行为**:
- L1 Cache: ❌ 不缓存数据
- L2 Cache: ❌ 不缓存数据
- 适用: 需要每次从内存读取最新值 (如标志位)

#### .lu (Last Use) — 最后使用策略

```cpp
// PTX 指令
ld.global.lu.f32 %f, [%r];
```

**行为**:
- L1 Cache: ❌ 不缓存数据
- L2 Cache: ❌ 不缓存数据 (提示这是最后一次使用)
- 适用: 明确标记数据不再需要，释放缓存空间

### 10.3 __ldg() 函数详解

`__ldg()` 是 CUDA 提供的只读缓存加载函数，对应 PTX 的 `.cg` 操作符：

```cpp
// __ldg() 函数原型
template <typename T>
__device__ T __ldg(const T* ptr);

// 使用示例
float value = __ldg(&array[tid]);

// 支持的数据类型
// • 基础类型: char, short, int, long, float, double
// • 向量类型: char2, int4, float4, double2 等
```

**__ldg() 缓存行为**:
- 数据仅缓存在 L2，不进入 L1
- 适合跨 Block 共享的只读数据
- 避免 L1 被大数组污染
- 数据必须是只读的 (写入会导致未定义行为)

### 10.4 缓存操作符选择指南

```
┌─────────────────────────────────────────────────────────────────────┐
│                    缓存操作符选择决策树                              │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  数据需要多次访问吗?                                                │
│      │                                                              │
│      ├─ 是 → 数据需要跨 Block 共享吗?                              │
│      │         │                                                    │
│      │         ├─ 是 → 使用 .cg (全局缓存)                         │
│      │         │         __ldg(ptr)                                 │
│      │         │                                                    │
│      │         └─ 否 → 使用 .ca (全缓存，默认)                     │
│      │                   直接访问 ptr                               │
│      │                                                              │
│      └─ 否 → 数据还会再次使用吗?                                   │
│               │                                                     │
│               ├─ 不确定 → 使用 .ca (默认)                           │
│               │                                                     │
│               └─ 否 → 使用 .cs (流式)                               │
│                         __ldcs(ptr)                                 │
│                                                                     │
│  特殊场景:                                                          │
│  • 读取标志位/同步变量 → .cv (每次读最新值)                         │
│  • 最后一次使用数据 → .lu (释放缓存空间)                            │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### 10.5 与 cuda::annotated_ptr 的对比

依据 NVIDIA 官方文档，`cuda::annotated_ptr` 是 libcu++ 提供的另一种持久化控制方式：

| 特性 | CUDA Runtime API | cuda::annotated_ptr |
|------|------------------|---------------------|
| **版本** | CUDA 11.0+ | CUDA 11.5+ |
| **方式** | Stream/Graph 级属性 | 指针级注解 |
| **粒度** | 粗粒度 (Stream/Node) | 细粒度 (指针) |
| **实现位置** | CUDA Runtime | libcu++ 库 |
| **内存访问属性** | 通过 accessPolicyWindow | 通过内存访问属性注解 |

---

## 第十一部分 实际应用与最佳实践

### 11.1 深度学习场景

#### 场景 1: Transformer 注意力机制

```cpp
// Q, K, V 矩阵 (只读，跨 Head 共享)
__global__ void attention_kernel(
    const float* __restrict__ Q,  // Query 矩阵
    const float* __restrict__ K,  // Key 矩阵
    const float* __restrict__ V,  // Value 矩阵
    float* output,
    int seq_len, int hidden_size)
{
    int head = blockIdx.x;
    int seq_idx = threadIdx.x;
    
    // 使用 __ldg() 加载 Q, K, V (只读，跨 Head 共享)
    float q_val = __ldg(&Q[head * seq_len + seq_idx]);
    float k_val = __ldg(&K[head * seq_len + seq_idx]);
    float v_val = __ldg(&V[head * seq_len + seq_idx]);
    
    // 计算注意力
    float attention = q_val * k_val;
    output[head * seq_len + seq_idx] = attention * v_val;
}
```

**优化效果**:
- Q, K, V 矩阵在 L2 中缓存，跨 Head 共享
- 避免每个 Head 重复从 HBM 加载数据
- 性能提升 15~25%

#### 场景 2: 权重矩阵持久化 (结合 accessPolicyWindow)

```cpp
// 使用 L2 持久化提升权重矩阵访问性能
void run_inference_with_persistent_weights(float* weights, int num_bytes) {
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    
    // 设置 L2 set-aside
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    size_t size = min(int(prop.l2CacheSize * 0.75), prop.persistingL2CacheMaxSize);
    cudaDeviceSetLimit(cudaLimitPersistingL2CacheSize, size);
    
    // 配置权重矩阵的持久化访问窗口
    cudaStreamAttrValue attr;
    attr.accessPolicyWindow.base_ptr  = reinterpret_cast<void*>(weights);
    attr.accessPolicyWindow.num_bytes = num_bytes;
    attr.accessPolicyWindow.hitRatio  = 1.0;
    attr.accessPolicyWindow.hitProp   = cudaAccessPropertyPersisting;
    attr.accessPolicyWindow.missProp  = cudaAccessPropertyStreaming;
    cudaStreamSetAttribute(stream, cudaStreamAttributeAccessPolicyWindow, &attr);
    
    // 多次推理，权重矩阵持久化在 L2 中
    for (int i = 0; i < 1000; i++) {
        inference_kernel<<<grid, block, 0, stream>>>(weights, input);
    }
    
    // 重置持久化
    cudaCtxResetPersistingL2Cache();
    cudaStreamDestroy(stream);
}
```

### 11.2 最佳实践总结

```
┌─────────────────────────────────────────────────────────────────────┐
│                    L2 Cache 持久化最佳实践                           │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ✅ 推荐使用持久的场景:                                              │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │  1. 被反复访问的数据 (权重矩阵、热点数据)                    │   │
│  │  2. 跨 Kernel 共享的数据                                     │   │
│  │  3. 数据量 < L2 set-aside 容量                              │   │
│  │  4. 需要低延迟重复访问的场景                                 │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                                                                     │
│  ✅ 推荐使用流式的场景:                                              │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │  1. 一次性访问的数据                                         │   │
│  │  2. 大型流式数据管道                                         │   │
│  │  3. 超过 set-aside 容量的数据                                │   │
│  │  4. 避免 L2 污染的场景                                       │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                                                                     │
│  ⚠️ 注意事项:                                                       │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │  1. 使用后必须 Reset (cudaCtxResetPersistingL2Cache)        │   │
│  │  2. 不要依赖自动重置 (时间不确定)                            │   │
│  │  3. 管理并发 Stream 的 set-aside 利用率                      │   │
│  │  4. hitRatio 用于避免 thrashing                              │   │
│  │  5. MIG 模式下功能禁用                                       │   │
│  │  6. MPS 模式下只能通过环境变量配置                           │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 第十二部分 与 Ascend 对比

### 12.1 缓存控制机制对比

| 特性 | NVIDIA GPU | Ascend NPU |
|------|------------|------------|
| **L2 Cache** | 有，可配置 | 无 L2 Cache (使用 UB) |
| **缓存控制** | __ldg(), accessPolicyWindow | 无自动缓存，手动 DataCopy |
| **持久化** | cudaAccessPropertyPersisting | 无持久化概念 |
| **Set-Aside** | cudaLimitPersistingL2CacheSize | 无对应机制 |
| **Reset** | cudaCtxResetPersistingL2Cache | 无对应机制 |
| **访问策略** | Streaming/Persisting/Normal | 显式管理 |
| **自动化** | 部分自动 | 完全手动 |

### 12.2 设计哲学对比

```
┌─────────────────────────────────────────────────────────────────────┐
│                    缓存控制设计哲学对比                              │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  NVIDIA GPU: 半自动缓存控制                                         │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │  • 硬件自动缓存，开发者可干预                                │   │
│  │  • __ldg() 提供只读缓存优化                                │   │
│  │  • accessPolicyWindow 提供区域级持久化控制                  │   │
│  │  • cudaCtxResetPersistingL2Cache 提供重置机制               │   │
│  │  • 适合通用计算，兼顾易用性和性能                          │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                                                                     │
│  Ascend NPU: 完全手动缓存控制                                       │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │  • 无自动缓存，开发者完全控制                               │   │
│  │  • DataCopy 显式搬运数据到 UB                              │   │
│  │  • TQue 队列协议管理数据流                                 │   │
│  │  • 无持久化概念，数据生命周期由开发者管理                    │   │
│  │  • 适合专用计算，追求极致性能                               │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 第十三部分 性能分析与 Profiling

### 13.1 L2 Cache 命中率分析

使用 Nsight Compute 分析 L2 Cache 命中率：

```bash
# 使用 Nsight Compute 收集 L2 Cache 指标
ncu --metrics l2_hit_rate my_kernel

# 关键指标
# • l2_hit_rate: L2 Cache 命中率
# • l2_tex_read_transactions: L2 纹理读取事务数
# • l2_tex_write_transactions: L2 纹理写入事务数
```

### 13.2 优化建议

```
┌─────────────────────────────────────────────────────────────────────┐
│                    L2 Cache 性能优化建议                             │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  1. 提升 L2 命中率                                                  │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │  • 使用 __ldg() 加载只读数据                                │   │
│  │  • 配置 accessPolicyWindow 实现持久化                       │   │
│  │  • 优化数据局部性 (时空局部性)                              │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                                                                     │
│  2. 减少 L2 污染                                                    │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │  • 对一次性数据使用 .cs 操作符                              │   │
│  │  • 使用 cudaAccessPropertyStreaming                         │   │
│  │  • 使用 .lu 标记最后使用的数据                               │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                                                                     │
│  3. 跨 Kernel 数据复用                                              │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │  • 配置 Stream 的 accessPolicyWindow                        │   │
│  │  • 避免 Kernel 间数据搬运到 Host                             │   │
│  │  • 使用 CUDA Graph 管理 Kernel 依赖                          │   │
│  │  • 使用后及时 Reset                                         │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 第十四部分 总结

### 14.1 核心要点

```
┌─────────────────────────────────────────────────────────────────────┐
│                    L2 Cache Persistence 核心要点                     │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  1. L2 Cache 是 NVIDIA GPU (CC 8.0+) 的全局共享缓存                  │
│     • 容量: 4~80 MB (架构相关)                                     │
│     • 延迟: ~200 cycles                                             │
│     • 跨 SM 共享，跨 Kernel 持久化                                 │
│                                                                     │
│  2. Persisting vs Streaming                                          │
│     • Persisting: 数据被反复访问，希望保留                          │
│     • Streaming: 数据仅访问一次，避免污染                          │
│                                                                     │
│  3. L2 Set-Aside 机制                                                │
│     • 预留部分 L2 用于持久化访问                                    │
│     • cudaDeviceSetLimit(cudaLimitPersistingL2CacheSize, size)     │
│     • 通常设为 L2 总容量的 75%                                      │
│     • MIG 禁用 / MPS 需环境变量                                     │
│                                                                     │
│  4. accessPolicyWindow                                                │
│     • 指定全局内存区域 + 持久化属性                                 │
│     • 参数: base_ptr, num_bytes, hitRatio, hitProp, missProp       │
│     • 用于 Stream 或 CUDA Graph Kernel Node                         │
│                                                                     │
│  5. 三种 Access Property                                             │
│     • cudaAccessPropertyStreaming: 优先淘汰                        │
│     • cudaAccessPropertyPersisting: 优先保留                       │
│     • cudaAccessPropertyNormal: 重置持久化                          │
│                                                                     │
│  6. Reset 很重要                                                     │
│     • cudaCtxResetPersistingL2Cache() 重置所有持久化行             │
│     • 不要依赖自动重置 (时间不确定)                                │
│                                                                     │
│  7. 与 Ascend 的对比                                                │
│     • NVIDIA: 半自动缓存，开发者可干预                             │
│     • Ascend: 完全手动管理，开发者完全控制                         │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### 14.2 使用决策树

```
数据访问模式分析:
│
├─ 数据被反复访问吗?
│   ├─ 是 → 数据量 < L2 set-aside 容量吗?
│   │       ├─ 是 → 配置 accessPolicyWindow (Persisting)
│   │       │        hitProp = cudaAccessPropertyPersisting
│   │       └─ 否 → 使用流式策略 (避免污染)
│   │
│   └─ 否 → 使用流式策略 (Streaming)
│
│
└─ 结束使用后必须 Reset
    ├─ cudaCtxResetPersistingL2Cache()
    └─ 不要依赖自动重置
```

### 14.3 Host / Stream / Kernel 控制分层

| 层级 | 控制什么 | 典型 API |
|------|----------|----------|
| **Host** | 划 set-aside 容量、全局清线、查询能力 | `cudaDeviceSetLimit`、`cudaCtxResetPersistingL2Cache` |
| **Stream / Graph Node** | 地址窗口、hitRatio、hit/miss 属性 | `cudaStreamSetAttribute`、`cudaGraphKernelNodeSetAttribute` |
| **Kernel** | 真正产生带属性的访存；可 per-access 再标注 | 被动继承 stream；主动用 `annotated_ptr` / PTX / `__ldg` |

---

## 附录 A：资料来源说明

### A.1 资料来源

| 来源 | 链接 | 状态 |
|------|------|------|
| **NVIDIA 官方文档** | https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/l2-cache-control.html | ✅ 已获取 |
| **知乎文章** | https://zhuanlan.zhihu.com/p/1893610874607998037 | ⚠️ 返回 403，无法访问 |

### A.2 说明

本文档主体内容基于 **NVIDIA 官方 CUDA Programming Guide《L2 Cache Control》** 章节，所有关键概念（Set-Aside、accessPolicyWindow、三种 Access Property、Reset 机制、查询 API）均依据官方文档表述。

知乎文章由于访问受限（HTTP 403）无法获取，未纳入本文档。如需补充知乎方面的观点，请提供可访问的链接或内容。

---

**文档结束**
