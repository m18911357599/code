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

## 8. 参考源码与核心代码（cann/shmem）

| 模块 | 路径 |
| --- | --- |
| Device SQE/DB | `src/device/gm2gm/engine/shmem_device_sdma.hpp` |
| SQE 结构体 | `src/device/gm2gm/engine/shmemi_device_sdma.h` |
| Host 建链 | `src/host/transport/device_sdma/device_sdma_transport_manager.cpp` |
| 公共宏 | `include/host_device/shmem_common_types.h` |
| SDMA demo | `examples/sdma/` |
| NotifyWait | `examples/notifywait/` |
| 双平面 MoE | `examples/dispatch/dispatch_doubleplane/` |

### 8.1 Channel / SQE 结构体

来源：`shmemi_device_sdma.h`

```cpp
struct stars_channel_info_t {
    uint32_t sq_head;
    uint32_t sq_tail;        // 软件维护的提交尾（offset +4）
    uint64_t sq_base;        // SQ 缓冲区基地址（HBM）
    uint64_t sq_reg_base;    // SQ 寄存器基地址（门铃 MMIO）
    uint32_t sq_depth;
    uint32_t sq_id;
    uint32_t cq_id;
    uint32_t logic_cq_id;
    uint64_t cqe_addr;
    uint32_t report_cqe_num;
    uint32_t stream_id;
    uint32_t dev_id;
    uint8_t reserved[4];     // 对齐到 64 字节
};

struct stars_sqe_header_t {
    uint8_t type : 6;        // SDMA=11, NOTIFY_RECORD=6
    uint16_t res1 : 10;
    uint16_t block_dim;
    uint16_t rt_streamid;
    uint16_t task_id;        // = sq_tail - sq_head
};

struct stars_sdma_sqe_t {
    stars_sqe_header_t header;   // 0~7
    uint32_t res3;               // 8~11
    uint16_t res4;
    uint8_t kernel_credit;       // 14: 数据 SQE 填 240
    uint8_t ptr_mode : 1;        // 15: 0=直接地址
    uint8_t res5 : 7;
    uint32_t opcode : 8;         // 16: copy=0
    uint32_t ie2 : 1;
    uint32_t sssv : 1;           // src stream valid
    uint32_t dssv : 1;           // dst stream valid
    uint32_t sns : 1;
    uint32_t dns : 1;
    uint32_t qos : 4;            // HCCL QoS=6
    uint32_t sro : 1;
    uint32_t dro : 1;
    uint32_t partid : 8;
    uint32_t mpam : 1;
    uint32_t res6 : 4;
    uint16_t src_streamid;
    uint16_t src_sub_streamid;
    uint16_t dst_streamid;
    uint16_t dst_sub_streamid;
    uint32_t length;
    uint32_t src_addr_low;
    uint32_t src_addr_high;
    uint32_t dst_addr_low;
    uint32_t dst_addr_high;
    uint8_t link_type;           // 填 255
    uint8_t resvered[3];
    uint32_t reslast[3];         // 对齐到 64B
};

// Doorbell 偏移：A2/A3=0x8，Ascend950(3510)=0x0
#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3510)
constexpr uint32_t ACLSHMEM_STARS_SQ_TAIL_OFFSET = 0x0;
#else
constexpr uint32_t ACLSHMEM_STARS_SQ_TAIL_OFFSET = 0x8;
#endif
```

### 8.2 填充 SDMA SQE

来源：`shmem_device_sdma.hpp` → `aclshmemi_fill_sdma_sqe`

```cpp
ACLSHMEM_DEVICE void aclshmemi_fill_sdma_sqe(__gm__ stars_channel_info_t* channel_info,
                                             __gm__ uint8_t* src, __gm__ uint8_t* dst,
                                             uint32_t length, uint32_t sq_tail, uint32_t task_id)
{
    __gm__ stars_sdma_sqe_t *sqe =
        (__gm__ stars_sdma_sqe_t *)(channel_info->sq_base);
    sqe += (sq_tail % channel_info->sq_depth);

    sqe->header.type = ACLSHMEM_SQE_TYPE_SDMA;   // 11
    sqe->header.block_dim = 0;
    sqe->header.rt_streamid = channel_info->stream_id;
    sqe->header.task_id = task_id;

    sqe->kernel_credit = ACLSHMEM_STARS_DEFAULT_KERNEL_CREDIT; // 240
    sqe->ptr_mode = 0;

    sqe->opcode = 0;
    sqe->ie2 = 0;
    sqe->sssv = 1U;
    sqe->dssv = 1U;
    sqe->sns = 1U;
    sqe->dns = 1U;
    sqe->qos = 6;   // HCCL QoS
    sqe->partid = 0U;
    sqe->mpam = 0;
    sqe->length = length;

    uint64_t src_addr = reinterpret_cast<uint64_t>(src);
    uint64_t dst_addr = reinterpret_cast<uint64_t>(dst);
    sqe->src_addr_low  = static_cast<uint32_t>(src_addr & 0xFFFFFFFF);
    sqe->src_addr_high = static_cast<uint32_t>((src_addr >> 32) & 0xFFFFFFFF);
    sqe->dst_addr_low  = static_cast<uint32_t>(dst_addr & 0xFFFFFFFF);
    sqe->dst_addr_high = static_cast<uint32_t>((dst_addr >> 32) & 0xFFFFFFFF);
    sqe->link_type = static_cast<uint8_t>(255U);
}
```

### 8.3 Post Send：组 SQE → DCCI → 敲 DB

来源：`aclshmemi_sdma_post_send`（核心片段）

```cpp
ACLSHMEM_DEVICE void aclshmemi_sdma_post_send(__gm__ uint8_t *recv_buffer,
                                              __gm__ uint8_t *send_buffer,
                                              uint64_t message_len,
                                              AscendC::LocalTensor<uint32_t> &tmp_local,
                                              uint32_t sync_id)
{
    __gm__ uint8_t *channel_base = aclshmemi_sdma_get_channel_base();
    const auto cur_block_idx = AscendC::GetBlockIdx();

    sdma_config_t config;
    config.queue_num = 1;
    config.block_bytes = 1024 * 1024;           // 1MB / SQE
    config.per_core_bytes = message_len;
    config.iter_num = (config.per_core_bytes + config.block_bytes - 1) / config.block_bytes;

    __gm__ stars_channel_info_t *batch_write_channel_info =
        (__gm__ stars_channel_info_t *)(channel_base) + cur_block_idx * config.queue_num;

    // 1) 读软件 sq_tail（先 dcci，避免 stale）
    uint32_t sq_tail[ACLSHMEM_MAX_AIV_PER_NPU] = {0};
    for (uint32_t queue_id = 0U; queue_id < config.queue_num; ++queue_id) {
        __gm__ stars_channel_info_t *channel_info = batch_write_channel_info + queue_id;
        dcci_cacheline(((__gm__ uint8_t *)channel_info) + 4);
        sq_tail[queue_id] = *((__gm__ uint32_t *)(((__gm__ uint8_t *)channel_info) + 4));
    }

    // 2) 按 1MB 分片填数据 SQE
    aclshmemi_sdma_submit_data_sqes(batch_write_channel_info, send_buffer, recv_buffer,
                                    config, sq_tail);

    // 3) DCCI 刷 SQE 到 HBM，再敲 doorbell
    auto item_size = config.iter_num * sizeof(stars_sdma_sqe_t);
    for (uint8_t queue_id = 0; queue_id < config.queue_num; queue_id++) {
        __gm__ stars_channel_info_t *channel_info = batch_write_channel_info + queue_id;

        AscendC::GlobalTensor<uint8_t> write_info;
        write_info.SetGlobalBuffer((__gm__ uint8_t *)(channel_info->sq_base), item_size);
        AscendC::DataCacheCleanAndInvalid<uint8_t, AscendC::CacheLine::ENTIRE_DATA_CACHE,
            AscendC::DcciDst::CACHELINE_OUT>(write_info);

        // Ring Doorbell：写硬件 SQ Tail
        aclshmemi_set_value<uint32_t>(
            (__gm__ uint8_t *)(channel_info->sq_reg_base) + ACLSHMEM_STARS_SQ_TAIL_OFFSET,
            sq_tail[queue_id], tmp_local, sync_id);
        // 同步软件侧 sq_tail
        aclshmemi_set_value<uint32_t>(
            ((__gm__ uint8_t *)channel_info) + 4, sq_tail[queue_id], tmp_local, sync_id);
    }
    AscendC::PipeBarrier<PIPE_ALL>();
}
```

`submit_data_sqes` 内每次：

```cpp
aclshmemi_fill_sdma_sqe(channel_info, src_addr, dst_addr, transfer_bytes,
                        sq_tail[queue_idx],
                        sq_tail[queue_idx] - channel_info->sq_head);
sq_tail[queue_idx] = (sq_tail[queue_idx] + 1) % (channel_info->sq_depth);
```

### 8.4 Quiet：Flag SQE + Poll

来源：`aclshmemi_sdma_submit_flag_sqes` / `aclshmemi_sdma_poll_for_completion`

```cpp
// 再提交一条 SDMA SQE：把 send flag(8B) DMA 到 remote_recv_workspace
aclshmemi_fill_sdma_sqe(channel_info,
    layout.send_workspace,
    layout.remote_recv_workspace + queue_id * ACLSHMEM_SDMA_FLAG_LENGTH,
    /*flag_size=*/8, sq_tail, sq_tail - channel_info->sq_head);

sq_tail = (sq_tail + 1) % (channel_info->sq_depth);

// DCCI + Ring Doorbell（同 post_send）
aclshmemi_set_value<uint32_t>(
    (__gm__ uint8_t *)(channel_info->sq_reg_base) + ACLSHMEM_STARS_SQ_TAIL_OFFSET,
    sq_tail, tmp_local, sync_id);
aclshmemi_set_value<uint32_t>(((__gm__ uint8_t *)channel_info) + 4, sq_tail, tmp_local, sync_id);

// AIV 自旋等待 flag 非 0
while (send_value == 0 && times < max_times /*1e6*/) {
    copy_gm_to_gm<uint32_t>(local_recv_workspace, remote_recv_workspace, 1, tmp_local, sync_id);
    dcci_cacheline(local_recv_workspace);
    send_value = *((__gm__ uint32_t *)local_recv_workspace);
    times++;
}
```

### 8.5 Notify Record SQE

来源：`aclshmemi_fill_notify_record_sqe` / `aclshmemi_stars_submit_notify_record`

```cpp
ACLSHMEM_DEVICE void aclshmemi_fill_notify_record_sqe(__gm__ stars_channel_info_t *channel_info,
                                                      uint32_t sq_tail, uint32_t task_id,
                                                      uint32_t notify_id)
{
    __gm__ stars_notify_sqe_t *sqe =
        (__gm__ stars_notify_sqe_t *)(channel_info->sq_base);
    sqe += (sq_tail % channel_info->sq_depth);

    sqe->header.type = ACLSHMEM_SQE_TYPE_NOTIFY_RECORD; // 6
    sqe->header.block_dim = 0;
    sqe->header.rt_streamid = channel_info->stream_id;
    sqe->header.task_id = task_id;
    sqe->notify_id = notify_id;
    sqe->kernel_credit = ACLSHMEM_DEFAULT_KERNEL_CREDIT; // 254
}

// 提交时同样：填 SQE → DCCI → 写 sq_reg_base+offset → 更新软件 sq_tail
aclshmemi_set_value<uint32_t>(
    (__gm__ uint8_t *)(channel_info->sq_reg_base) + ACLSHMEM_STARS_SQ_TAIL_OFFSET,
    sq_tail, tmp_local, sync_id);
```

### 8.6 Host 控制面建链

来源：`device_sdma_transport_manager.cpp` → `OpenDevice`

```cpp
Result SdmaTransportManager::OpenDevice(const TransportOptions& options)
{
    // 1) 按最大 AIV 数建 STARS stream（取 sq_id/cq_id/stream_id）
    ACLSHMEM_CHECK_RET(CreateStarsStreams(ACLSHMEM_MAX_AIV_PER_NPU)); // 48

    // 2) AIV/AICPU 共享 workspace
    constexpr size_t workspace_size = 16 * 1024;
    ACLSHMEM_CHECK_RET(MallocSdmaWorkspace(workspace_size));

    // 3) Notify id 写入 workspace + 14KB
    CreateNotifyIds();

    // 4) 资源 H2D，并 launch AICPU 查询 STARS SQ 基址/门铃基址
    ACLSHMEM_CHECK_RET(CopyHostOpResToDevice());
    ACLSHMEM_CHECK_RET(
        LaunchSdmaAicpuKernel(reinterpret_cast<uint64_t>(op_res_info_device_ptr_),
                              op_res_info_.workspace_addr));
    return ACLSHMEM_SUCCESS;
}
```

`CreateStarsStreams` 关键片段：

```cpp
ACLSHMEM_CHECK_RET(aclrtCreateStreamWithConfig(&stream, 0, ACL_STREAM_DEVICE_USE_ONLY));
aclrtStreamGetId(stream, &stream_id);
DlRtApi::RtStreamGetSqid(stream, &sq_id);
DlRtApi::RtStreamGetCqid(stream, &cq_id, &logic_cq_id);
streams_[i].stream_id = stream_id;
streams_[i].sq_id = sq_id;
streams_[i].cq_id = cq_id;
streams_[i].dev_id = die_id;
```

### 8.7 高阶 put 入口

来源：`aclshmemx_sdma_put_nbi`

```cpp
template <typename T>
ACLSHMEM_DEVICE void aclshmemx_sdma_put_nbi(__gm__ T *dst, __gm__ T *src, __ubuf__ T *buf,
                                            uint32_t ub_size, uint32_t elem_size,
                                            int pe, uint32_t sync_id)
{
    auto ptr = aclshmem_ptr(dst, pe);   // 对称地址 → 远端 VA

    AscendC::LocalTensor<uint32_t> ub_tensor;
    ub_tensor.address_.logicPos = static_cast<uint8_t>(AscendC::TPosition::VECOUT);
    ub_tensor.address_.bufferAddr = reinterpret_cast<uint64_t>(buf);
    ub_tensor.address_.dataLen = ub_size;

    // put: 本地 src → 远端 ptr
    aclshmemi_sdma_post_send((__gm__ uint8_t *)ptr, (__gm__ uint8_t *)src,
                             elem_size * sizeof(T), ub_tensor, sync_id);
}
```

---

## 9. 一句话总结

**AIV 直驱 SDMA = Host 建好 STARS channel → AIV 写 64B SQE 到 `sq_base` → DCCI → 写 `sq_reg_base+offset` 更新 tail（DB）→ SDMA 搬数 → quiet(flag) 或 notify_record 收尾。**  
性能上优先做引擎分流与双平面，其次控 outstanding/同步方式，再调对齐与分核。
