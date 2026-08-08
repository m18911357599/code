# CUDA L2 Persistent：Stream 控制面配置

> CC ≥ 8.0 · CUDA 11.0+ · [官方 L2 Cache Control](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/l2-cache-control.html)

**结论**：Persistent 不钉死数据，只改 L2 eviction 优先级。控制面分两层——Host 划 set-aside；Stream（或 Graph Node）挂 Access Policy Window。Kernel 被动继承 stream 策略。

---

## 1. 控制面分层

| 层 | 职责 | API |
|----|------|-----|
| **Host / Device** | 划 L2 set-aside；全局清 persistent 线；查能力 | `cudaDeviceSetLimit` / `GetLimit`、`cudaCtxResetPersistingL2Cache`、`cudaGetDeviceProperties` |
| **Stream**（主路径） | 绑地址窗口 + hitRatio + hit/miss 属性 | `cudaStreamSetAttribute(..., AccessPolicyWindow, ...)` |
| **Graph Node** | 与 stream 等价，粒度是 kernel node | `cudaGraphKernelNodeSetAttribute(..., AccessPolicyWindow, ...)` |
| **Kernel** | 执行时 touch 窗口地址才真正装入 L2 | 通常无改代码；可选 `annotated_ptr`（本文从略） |

```
Host: SetLimit(PersistingL2CacheSize)
  → Stream: SetAttribute(AccessPolicyWindow)
    → Kernel<<<..., stream>>> 访问 [base, base+num_bytes)
  → Stream: num_bytes=0 关窗口
  → Host: CtxResetPersistingL2Cache()
```

---

## 2. Persistent 容量：Set-Aside

Persisting 访问优先用 set-aside；Normal/Streaming 只能用其中空闲部分。

```cpp
cudaDeviceProp prop;
cudaGetDeviceProperties(&prop, device_id);
size_t size = min((size_t)(prop.l2CacheSize * 0.75), prop.persistingL2CacheMaxSize);
cudaDeviceSetLimit(cudaLimitPersistingL2CacheSize, size);

size_t cur;
cudaDeviceGetLimit(&cur, cudaLimitPersistingL2CacheSize);  // 查询
```

| 属性 / 限制 | 含义 |
|-------------|------|
| `l2CacheSize` | L2 总量 |
| `persistingL2CacheMaxSize` | set-aside 上限 |
| `accessPolicyMaxWindowSize` | 单窗口上限 |
| MIG | set-aside **禁用** |
| MPS | 不能 `SetLimit`；用 `CUDA_DEVICE_DEFAULT_PERSISTING_L2_CACHE_PERCENTAGE_LIMIT` |

---

## 3. Stream 控制面：Access Policy Window

### 3.1 配置

```cpp
cudaStreamAttrValue attr{};
attr.accessPolicyWindow.base_ptr  = ptr;          // 全局地址起点（driver 可对齐）
attr.accessPolicyWindow.num_bytes = window_size;  // ≤ accessPolicyMaxWindowSize；0=关闭
attr.accessPolicyWindow.hitRatio  = 0.6f;         // ≈该比例走 hitProp
attr.accessPolicyWindow.hitProp   = cudaAccessPropertyPersisting;
attr.accessPolicyWindow.missProp  = cudaAccessPropertyStreaming;  // 仅 Normal 或 Streaming

cudaStreamSetAttribute(stream, cudaStreamAttributeAccessPolicyWindow, &attr);
```

Graph 版：把 `cudaStreamAttrValue` / `cudaStreamSetAttribute` 换成  
`cudaKernelNodeAttrValue` / `cudaGraphKernelNodeSetAttribute(..., cudaKernelNodeAttributeAccessPolicyWindow, ...)`。

### 3.2 Stream 语义（重点）

| 点 | 行为 |
|----|------|
| 附着点 | 策略挂在 **stream**，不挂在 kernel 符号 |
| 生效时机 | `SetAttribute` **之后**、该 stream 上随后 launch 的 kernel |
| 继承 | 同 stream 后续多个 kernel 共用当前 window |
| 装入条件 | kernel 必须 **访问** 窗口内地址；SetAttribute 本身不预取 |
| 关闭 | `num_bytes = 0` 再 SetAttribute |
| 多 stream | 各有独立 window，但 **set-aside 全局共享** |
| 跨 stream 读 | 不继承对方 attribute；已驻留线或许可命中，但不保证按 Persisting 继续灌入 |

### 3.3 三种 Access Property

| 属性 | 作用 |
|------|------|
| `Persisting` | 优先留在 set-aside |
| `Streaming` | 优先被驱逐（一次性数据） |
| `Normal` | **强制清除**该窗口先前的 Persisting 状态 |

### 3.4 hitRatio 与预算

硬件把窗口切成 hit/miss 段，约 `hitRatio` 比例走 `hitProp`（具体地址近似随机，非运行时热度计数）。

记 `S`=set-aside，`W`=`num_bytes`，`R`=`hitRatio`，期望占用 ≈ `W×R`：

| 场景 | 配法 |
|------|------|
| 热数据 ≈ S | `W≈S, R=1` |
| 热数据 > S 且循环扫 | `R≈S/W`，防 thrashing |
| N 个并发 stream | `Σ(Wᵢ×Rᵢ) ≲ S` |

例：S=16KB，两 stream 各 W=16KB、R=1 → 互踢；改 R=0.5 → 各约 8KB，不易互踢。

---

## 4. 标准配置闭环

```cpp
cudaStream_t stream;
cudaStreamCreate(&stream);

// ① Host：划容量
cudaDeviceProp prop;
cudaGetDeviceProperties(&prop, device_id);
size_t aside = min((size_t)(prop.l2CacheSize * 0.75), prop.persistingL2CacheMaxSize);
cudaDeviceSetLimit(cudaLimitPersistingL2CacheSize, aside);

size_t window_size = min(prop.accessPolicyMaxWindowSize, num_bytes);

// ② Stream：开 Persistent 窗口
cudaStreamAttrValue attr{};
attr.accessPolicyWindow.base_ptr  = reinterpret_cast<void*>(data1);
attr.accessPolicyWindow.num_bytes = window_size;
attr.accessPolicyWindow.hitRatio  = 0.6f;
attr.accessPolicyWindow.hitProp   = cudaAccessPropertyPersisting;
attr.accessPolicyWindow.missProp  = cudaAccessPropertyStreaming;
cudaStreamSetAttribute(stream, cudaStreamAttributeAccessPolicyWindow, &attr);

// ③ 同 stream 复用
for (int i = 0; i < 10; ++i)
    kernelA<<<grid, block, 0, stream>>>(data1);
kernelB<<<grid, block, 0, stream>>>(data1);

// ④ 关窗口
attr.accessPolicyWindow.num_bytes = 0;
cudaStreamSetAttribute(stream, cudaStreamAttributeAccessPolicyWindow, &attr);

// ⑤ Host：清 persistent 线（persistence-after-use）
cudaCtxResetPersistingL2Cache();

kernelC<<<grid, block, 0, stream>>>(data2);  // 恢复正常 L2 竞争
```

### Reset 三种方式

1. 窗口上设 `hitProp/missProp = Normal`（定向清）  
2. `cudaCtxResetPersistingL2Cache()`（上下文级一锅端）  
3. 长期不碰自动降级——**时间不确定，勿依赖**

---

## 5. 多 Stream 控制面注意

- 先列出**可能并发**的 stream/kernel，再按 `Σ(W×R) ≲ S` 分预算。  
- 阶段切换用 event 对齐后统一关窗口 + reset，避免一边还 Persisting、一边 Normal 扫大块。  
- Producer/Consumer 跨 stream：两边都应对共享缓冲设合理 window，或合并到同一 stream。

---

## 6. 速查

| 要做什么 | 调用 |
|----------|------|
| 划 Persistent 容量 | `cudaDeviceSetLimit(cudaLimitPersistingL2CacheSize, size)` |
| Stream 开窗口 | `cudaStreamSetAttribute(stream, cudaStreamAttributeAccessPolicyWindow, &attr)` |
| Stream 关窗口 | `attr.accessPolicyWindow.num_bytes = 0` 后再次 SetAttribute |
| 清所有 persistent 线 | `cudaCtxResetPersistingL2Cache()` |
| Graph 节点挂窗口 | `cudaGraphKernelNodeSetAttribute(node, cudaKernelNodeAttributeAccessPolicyWindow, &attr)` |
| 查能力 | `prop.l2CacheSize` / `persistingL2CacheMaxSize` / `accessPolicyMaxWindowSize` |

**误区**：SetAttribute ≠ 数据已在 L2；每 stream 不独占 set-aside；用完不 reset 会长期占坑。
