# ACLSHMEM：AIV 直驱 SDMA 流程、SQE / Doorbell 字段与性能改进点

> 基于 CANN / ACLSHMEM（`cann/shmem`）开源实现梳理。核心路径：
> - Host：`SdmaTransportManager`
> - Device：`aclshmemi_sdma_post_send` / `aclshmemi_fill_sdma_sqe`
> - 平台：Atlas A2/A3（910B/910_93）；Ascend950 的 SDMA put/get 路径当前未开放

---

## 1. 背景：为什么是 AIV 直驱 SDMA

传统路径是 Host/AICPU 组 SQE、敲 doorbell，AIV 只能被动等结果。  
ACLSHMEM 的 **AIV 直驱 SDMA** 把数据面下沉到 AIV：

1. Host 只做控制面：建 STARS stream / SQ、分配 workspace、创建 Notify。
2. AIV 在 kernel 内直接：
   - 读 `stars_channel_info_t`
   - 组装 64B SDMA SQE 到 SQ
   - `dcci` 刷 cache
   - 写 SQ Tail doorbell
3. STARS/SDMA 引擎 DMA 搬数，AIV 用 **flag SQE + poll** 或 **Notify Record** 做完成同步。

收益：去掉 Host 往返；SDMA 不占 MTE 通道，可与 MTE 构成双平面。

---

## 2. 整体架构

```text
┌──────────────── Host / Init ────────────────┐
│ SdmaTransportManager::OpenDevice            │
│  1) CreateStarsStreams(MAX_AIV=48)          │
│  2) MallocSdmaWorkspace(16KB)               │
│  3) CreateNotifyIds → workspace+14KB        │
│  4) Launch AICPU: AclnnShmemSdmaStarsQuery  │
│     → 回填 sq_base / sq_reg_base / sq_depth │
└─────────────────────┬───────────────────────┘
                      │ device_state.sdma_workspace_addr
                      ▼
┌──────────────── AIV Kernel ─────────────────┐
│ aclshmemx_sdma_put/get_nbi                  │
│   └─ aclshmemi_sdma_post_send               │
│        ├─ 读 channel.sq_tail (dcci 后)      │
│        ├─ fill SDMA SQE(s) 到 sq_base       │
│        ├─ DataCacheCleanAndInvalid          │
│        └─ Ring DB: *(sq_reg_base+offset)=tail│
│                                             │
│ 完成同步二选一：                              │
│  A) aclshmemx_sdma_quiet  (flag SQE + poll) │
│  B) aclshmemx_sdma_notify_record + Host wait│
└─────────────────────────────────────────────┘
```

Workspace 布局（逻辑）：

```text
sdma_workspace (16KB)
├─ [0, 64)          stars_channel_flag_info_t
├─ [64, ...)        stars_channel_info_t × 48   ← channel_base
├─ flag workspace   send/recv/remote flag(64B 对齐)
└─ +14KB            notify_id[48]
```

每个 AIV（`GetBlockIdx()`）绑定一条 channel：`channel_base + block_idx`。

---

## 3. AIV 驱动 SDMA 详细流程

### 3.1 Host 控制面（一次性）

| 步骤 | 动作 | 说明 |
| --- | --- | --- |
| 1 | `CreateStarsStreams(48)` | 每 AIV 一条 `ACL_STREAM_DEVICE_USE_ONLY` stream，取 `sq_id/cq_id/stream_id` |
| 2 | `MallocSdmaWorkspace(16KB)` | AIV 与 AICPU 共享上下文 |
| 3 | `CreateNotifyIds` | `aclrtCreateNotify`，id 写入 workspace+14KB |
| 4 | AICPU `AclnnShmemSdmaStarsQuery` | 查询 STARS SQ 基址/寄存器基址/深度，写入 `stars_channel_info_t` |

`stars_channel_info_t`（64B）关键字段：

| 字段 | 含义 |
| --- | --- |
| `sq_head` | SQ 消费头；task_id = sq_tail - sq_head，应保持单调（不要随意回写 head） |
| `sq_tail` | 软件维护的提交尾；offset +4 |
| `sq_base` | SQ 缓冲区基址（HBM），每槽 64B SQE |
| `sq_reg_base` | SQ 门铃寄存器基址（MMIO） |
| `sq_depth` | SQ 深度；tail 取模用 |
| `stream_id` / `sq_id` / `cq_id` | STARS 绑定信息 |
| `dev_id` | die id |

### 3.2 Device 数据面：`aclshmemi_sdma_post_send`

入口：`aclshmemx_sdma_put_nbi` / `get_nbi` → `aclshmem_ptr` 取远端 VA → `aclshmemi_sdma_post_send`。

**步骤：**

1. **取 channel**  
   `channel_base = sdma_workspace + sizeof(flag_info)`  
   `channel = channel_base + GetBlockIdx()`

2. **分片配置**  
   - `queue_num = 1`（当前每核一队列）  
   - `block_bytes = 1MB`（单条 SQE 最大搬运）  
   - `iter_num = ceil(message_len / 1MB)`

3. **读 sq_tail**  
   对 `channel_info+4` 做 `dcci`，再读 `sq_tail`（避免读到 stale cache）。

4. **填数据 SQE**  
   每个 1MB（末片取余）调用 `aclshmemi_fill_sdma_sqe`：  
   `sqe = sq_base + (sq_tail % sq_depth)`  
   `task_id = sq_tail - sq_head`  
   写完后 `sq_tail = (sq_tail + 1) % sq_depth`

5. **刷 cache**  
   `DataCacheCleanAndInvalid` 覆盖本批 SQE，保证 STARS 从 HBM 读到新 SQE。

6. **敲 doorbell**  
   - `*(sq_reg_base + SQ_TAIL_OFFSET) = new_sq_tail`  
   - 软件侧同步写回 `channel_info.sq_tail`

7. **返回**（非阻塞）  
   不等 DMA 完成；完成用 quiet / notify。

### 3.3 完成同步

#### 方式 A：`aclshmemx_sdma_quiet`（AIV 自旋）

1. 再提交一条 **flag SQE**：把 send flag（8B）DMA 到 `remote_recv_workspace`。  
2. AIV poll：把 remote flag 拷到 local，直到非 0（上限约 1e6 次）。  
3. 清零 flag。

特点：算子内本地完成语义清晰；**占住 AIV**。

#### 方式 B：`aclshmemx_sdma_notify_record` + Host `aclrtWaitAndResetNotify`

1. 填 `NOTIFY_RECORD` SQE（type=6），`notify_id` 来自 workspace。  
2. 敲 DB。  
3. Host 在另一 stream 上 `aclrtWaitAndResetNotify`，再 launch 下游 kernel。

特点：**及时释放 AIV**，适合跨 stream 依赖。

---

## 4. SQE 字段说明

SQE 固定 **64 字节**。数据搬运用 `stars_sdma_sqe_t`；完成通知用 `stars_notify_sqe_t`；CMO/prefetch 用 CMO SQE（A2 与 Ascend950/v2 layout 不同）。

### 4.1 公共 Header（`stars_sqe_header_t`，前 8B）

| 字段 | 位宽 | 数据 SQE 填法 | 说明 |
| --- | --- | --- | --- |
| `type` | 6 | **11** (`ACLSHMEM_SQE_TYPE_SDMA`) | Notify Record 为 **6** |
| `res1` | 10 | 0 | 保留 |
| `block_dim` | 16 | **0** | AIV 直驱不走 kernel dim |
| `rt_streamid` | 16 | `channel_info->stream_id` | 绑定 STARS stream |
| `task_id` | 16 | `sq_tail - sq_head` | 标识 SQ 内位置，须随 tail 单调 |

### 4.2 SDMA 数据 SQE（`stars_sdma_sqe_t`）

`aclshmemi_fill_sdma_sqe` 实际写入：

| 偏移/区域 | 字段 | 填值 | 含义 |
| --- | --- | --- | --- |
| 0–7 | header | 见上 | type=11 |
| 8–11 | `res3` | 0 | 保留 |
| 12–13 | `res4` | 0 | 保留 |
| 14 | `kernel_credit` | **240** | STARS 信用/超时相关 |
| 15 | `ptr_mode` | **0** | 地址直接模式（非指针间接） |
| 16 | `opcode` | **0** | 普通 SDMA copy |
| | `ie2` | 0 | 中断使能扩展 |
| | `sssv` / `dssv` | **1** | src/dst stream 有效 |
| | `sns` / `dns` | **1** | src/dst non-secure |
| | `qos` | **6** | HCCL QoS |
| | `sro` / `dro` | 0 | 不读-only / 不写旁路特殊 |
| | `partid` / `mpam` | 0 | MPAM 分区默认关 |
| 20–23 | `src_streamid` / `src_sub_streamid` | 0 | 由 SQ 绑定推断时可留 0 |
| 24–27 | `dst_streamid` / `dst_sub_streamid` | 0 | 同上 |
| 28–31 | `length` | 本次字节数 | ≤1MB/SQE |
| 32–35 | `src_addr_low` | src & 0xFFFFFFFF | 源 VA 低 32 |
| 36–39 | `src_addr_high` | src >> 32 | 源 VA 高 32 |
| 40–43 | `dst_addr_low` | dst & 0xFFFFFFFF | 目的 VA 低 32 |
| 44–47 | `dst_addr_high` | dst >> 32 | 目的 VA 高 32 |
| 48 | `link_type` | **255** | 链路类型（实现写满值） |
| 49–63 | reserved | 0 | 对齐到 64B |

**put / get 差异**：同一套 SQE，仅 src/dst 对调语义：

- put：本地 `src` → `aclshmem_ptr(dst, pe)`  
- get：`aclshmem_ptr(src, pe)` → 本地 `dst`

### 4.3 Notify Record SQE（`stars_notify_sqe_t`）

| 字段 | 填值 | 说明 |
| --- | --- | --- |
| `header.type` | **6** | NOTIFY_RECORD |
| `header.rt_streamid` | stream_id | 同 channel |
| `header.task_id` | sq_tail - sq_head | 同数据 SQE |
| `notify_id` | 13bit，来自 notify 表 | Host CreateNotify 的 id |
| `kernel_credit` | **254** | 默认 credit |
| 其余 | 0 | timeout 等保留 |

### 4.4 CMO / Prefetch SQE（简述）

`aclshmemx_cmo_nbi` 当前仅支持 `CMO_TYPE_PREFETCH`（opcode=6）：

- A2 layout：`stars_sdma_cmo_sqe_t`，`qos=6`，`partid=63`  
- Ascend950 / STARS v2：`stars_v2_sdma_cmo_sqe_t`，`wr_cqe=1`，doorbell offset 变为 `0x0`

---

## 5. Doorbell（DB）字段说明

### 5.1 敲铃动作

```text
地址:  channel_info->sq_reg_base + ACLSHMEM_STARS_SQ_TAIL_OFFSET
值:    新的 sq_tail（已取模 sq_depth）
写路径: UB staging → DataCopyPad 到 MMIO（aclshmemi_set_value）
```

| 平台 | `ACLSHMEM_STARS_SQ_TAIL_OFFSET` |
| --- | --- |
| A2/A3（非 3510） | **0x8** |
| Ascend950（`__NPU_ARCH__ == 3510`） | **0x0** |

### 5.2 软件侧同步写

敲硬件 DB 后，同步更新：

```text
channel_info->sq_tail  (offset +4)
```

后续 post/quiet 都依赖该软件 tail；读前必须 `dcci`。

### 5.3 正确顺序（硬约束）

```text
写 SQE → PipeBarrier/顺序保证 → DCCI 刷 SQE → 写 DB(tail) → 更新软件 sq_tail
```

错误顺序会导致：

| 错误 | 现象 |
| --- | --- |
| 先敲 DB 再写 SQE / 未 DCCI | 引擎读到旧 SQE → 错数据 / hang |
| 读 sq_tail 未 DCCI | 复用过期 tail → slot 覆盖或永不消费 |
| Wait 后回写 sq_head=completedTail | task_id 回绕，flag SQE 被忽略 |
| 多 AIV 共用同一 send_workspace 填 flag | flag 互相覆盖 → quiet 失败 |

---

## 6. 性能改进点

### 6.1 引擎选择（收益最大）

| 数据规模 | 推荐 | 原因 |
| --- | --- | --- |
| &lt; 2MB | **MTE** put/get | setup 小、延迟低 |
| ≥ 2MB，节点内 P2P | **SDMA** | 带宽高，不占 MTE |
| 跨节点 | **RDMA/RoCE** | 唯一路径 |
| MoE 倾斜大包 | **MTE+SDMA 双平面** | 小段 MTE、大段 SDMA |

双平面阈值经验（dispatch_doubleplane）：

```text
use_sdma = remote && segment_bytes >= 2MB
           && segment_bytes > 远端平均 segment
```

控制面（ready/count）始终走 MTE，避免 “ready 已见、payload 未到”。

### 6.2 提交与同步

| 改进 | 做法 | 效果 |
| --- | --- | --- |
| 批量 issue | 连续 `*_nbi`，按批 quiet | 摊薄 SQE/DB 开销 |
| outstanding 限流 | 每约 **256** 次 SDMA issue 一次 quiet | 防 SQ 积压/尾延迟 |
| 跨 stream 完成 | 用 **notify_record** 替代 quiet | 释放 AIV，降低占用 |
| 避免全局 barrier | producer signal / consumer wait | 去掉最慢者拖累 |
| 读 channel 前 dcci | 每次取 sq_tail 先失效 cache | 正确性 + 少超时重试 |

### 6.3 分片与分核

| 项 | 现状 / 建议 |
| --- | --- |
| 单 SQE 分片 | 实现固定 **1MB/SQE**；超大消息多 SQE 一轮敲一次 DB |
| 每核队列 | 当前 `queue_num=1`；可评估多 queue 轮转提升并行度 |
| block_dim | 小消息少核（如 8）；大消息按 PE/数据维扩展，≤ AIV 上限（910B2=48） |
| 消除 AIV 内串行 peer 循环 | 按源 PE 分核并行，或 sender-put + 本地 reduce |

### 6.4 带宽相关细节

- 地址 **512B 对齐** 更接近峰值（相对 32B 可有显著差距）。  
- 单次有效载荷建议 ≥ **16KB**，过小则 setup 占比过高。  
- 禁止双拷贝 staging：`sendbuf → local_symm → remote` 会把有效带宽打到约一半；应直接 put 到远端。  
- SDMA 高阶接口需预留 UB ≥ **64B**（默认约 191KB 处）；与 RDMA/UDMA 默认 UB 区段错开。  
- SDMA 与 MTE 双平面时：计算/控制走 MTE，大 payload 走 SDMA，真正吃满 HCCS。

### 6.5 正确性相关的“性能坑”

这些不是调参，但会表现为超时、带宽抖动：

1. **sq_head 保持 Host 初值**，不要按完成推进（task_id 语义依赖 head 固定）。  
2. **多核勿共享 flag staging**；每核独立 send/recv flag 槽。  
3. **数据 SQE 与 flag/notify SQE 同队列有序**，保证 “数据完成后才置位”。  
4. quiet 超时（1e6）要结合业务加监控，避免静默失败。

### 6.6 优化优先级（实操）

1. 选引擎：&lt;2MB MTE / ≥2MB SDMA / 跨节点 RDMA  
2. 拆掉 AIV 内逐 peer 串行 wait  
3. MTE+SDMA 双平面（MoE 大段）  
4. 批量 nbi + 限频 quiet；跨 stream 用 notify  
5. 对齐、chunk、block_dim、负载均衡  

---

## 7. API 速查

| API | 语义 |
| --- | --- |
| `aclshmemx_set_sdma_config(offset, ub_size, sync_id)` | 配置 UB scratch（≥64B）与 event |
| `aclshmemx_sdma_put_nbi` / `get_nbi` | 非阻塞提交 SDMA SQE + DB |
| `aclshmemx_sdma_quiet` | flag SQE + AIV poll 完成 |
| `aclshmemx_sdma_notify_record` | 下发 Notify Record SQE |
| Host `aclrtWaitAndResetNotify` | 等待 Notify，驱动下游 stream |

初始化需打开：`ACLSHMEM_DATA_OP_SDMA`（双平面再或上 `ACLSHMEM_DATA_OP_MTE`）。

---

## 8. 参考源码路径（cann/shmem）

| 模块 | 路径 |
| --- | --- |
| Device SQE/DB | `src/device/gm2gm/engine/shmem_device_sdma.hpp` |
| SQE 结构体 | `src/device/gm2gm/engine/shmemi_device_sdma.h` |
| Host 建链 | `src/host/transport/device_sdma/device_sdma_transport_manager.cpp` |
| 公共宏 | `include/host_device/shmem_common_types.h` |
| SDMA demo | `examples/sdma/` |
| NotifyWait | `examples/notifywait/` |
| 双平面 MoE | `examples/dispatch/dispatch_doubleplane/` |

---

## 9. 一句话总结

**AIV 直驱 SDMA = Host 建好 STARS channel → AIV 写 64B SQE 到 `sq_base` → DCCI → 写 `sq_reg_base+offset` 更新 tail（DB）→ SDMA 搬数 → quiet(flag) 或 notify_record 收尾。**  
性能上优先做引擎分流与双平面，其次控 outstanding/同步方式，再调对齐与分核。
