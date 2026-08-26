# ACLSHMEM：AIV 直驱 URMA（UDMA）与 RoCE 对比学习

> 基于 CANN / ACLSHMEM（`cann/shmem`）开源实现梳理。  
> 核心对照：
> - **URMA / UDMA**：Ascend950 上 AIV 直驱 UB（灵衢）远程内存访问
> - **RoCE / RDMA**：AIV 直驱 RoCEv2 QP（跨节点以太网 RDMA）
>
> 本文侧重“控制面谁建链、数据面 AIV 怎么敲铃、语义与选型差异”。

---

## 0. 一分钟结论

| 问题 | 答案 |
| --- | --- |
| AIV 如何直驱 URMA？ | Host 用 HCOMM 建 **UB endpoint + jetty（JFS/JFC）**，把 SQ/CQ/DB/token/EID 表拷到 Device；AIV 在 kernel 内组装 **SQE+SGE**，`st_dev` 敲 **SW doorbell**，再 poll **JFC CQE** |
| 与 RoCE 的本质差别？ | 同一套“AIV 组 WQE → 敲 DB → 轮询 CQE”范式，但底层是 **两套协议栈与对象模型**：URMA/UB（EID、jetty、token） vs RoCEv2（QP、GID、lkey/rkey） |
| 何时用谁？ | **机内 / UB 域**优先 UDMA；**跨节点以太网**走 RoCE；高阶 RMA 路由优先级：SDMA → UDMA → MTE → RoCE |

术语对应（项目 glossary）：

| 缩写 | 含义 | 在 shmem 中的角色 |
| --- | --- | --- |
| **URMA** | Unified Remote Memory Access | UB 上的远程内存访问协议/语义 |
| **UDMA** | Unified Direct Memory Access | SHMEM 引擎名与公开 API（`aclshmemx_udma_*`），实现上消费 URMA jetty |
| **RoCE** | RDMA over Converged Ethernet | RDMA 引擎使用的网络协议；API 为 `aclshmemx_roce_*` |
| **Jetty / JFS / JFC** | URMA 队列对象 | JFS=发送 jetty（SQ），JFC=完成 jetty（CQ） |
| **QP** | Queue Pair | RoCE 侧 SQ/RQ/SCQ/RCQ 配对 |
| **AIV** | Vector Core | 数据面真正写 WQE、敲 DB、poll CQE 的执行核 |

---

## 1. 背景：为什么是“AIV 直驱”

传统路径：Host / AICPU 组 WQE、敲 doorbell，AIV 只能等结果。  
ACLSHMEM 把 **数据面下沉到 AIV**：

1. Host 只做控制面：建链、注册 MR、把队列上下文发布到 Device HBM。
2. AIV 在 kernel 内直接：
   - 读队列上下文（SQ/CQ/DB、远端寻址信息）
   - 组装 WQE 到 SQ ring
   - cache 维护（`dcci` / `DataCopyPad`）
   - 写 doorbell
   - poll CQE 完成 quiet

URMA 与 RoCE **共享这一范式**，差别在 Host 建的是什么对象、WQE 字段语义、以及网络/总线路径。

---

## 2. 整体架构对照

### 2.1 AIV 直驱 URMA（UDMA）

```text
┌──────────────── Host / Init（UdmaTransportManager）──────────────┐
│ 1) HcommEndpointCreate(COMM_PROTOCOL_UBC_CTP, EID)               │
│ 2) HcommMemReg（对称堆 → tokenId / tokenValue）                   │
│ 3) HcommChannelCreate(..., COMM_ENGINE_AIV, ...)                 │
│    → 分配 jetty：ubJfs(SQ+DB) / ubJfc(CQ+DB)                     │
│ 4) D2H 读 ChannelEntity → FillWqCtx / FillCqCtx / FillMemInfo    │
│ 5) H2D 发布 aclshmemi_aiv_udma_info_t（SQ/CQ/mem/EID/AMO）        │
└───────────────────────────────┬──────────────────────────────────┘
                                │ Device VA: jetty rings + SW DB
                                ▼
┌──────────────── URMA / UB 硬件 ──────────────────────────────────┐
│ JFS: SQ ring + SQ DB     JFC: CQ ring + CQ DB                    │
│ 远端段：tokenId / tokenValue / remote EID                        │
└───────────────────────────────▲──────────────────────────────────┘
                                │ AIV: fill SQE+SGE → st_dev(SQ DB)
┌──────────────── AIV Kernel（数据面）─────────────────────────────┐
│ aclshmemx_udma_{put,get}_nbi → aclshmemi_udma_post_send[_mte3]   │
│ aclshmemx_udma_quiet         → poll JFC CQE → st_dev(CQ DB)      │
└──────────────────────────────────────────────────────────────────┘
```

### 2.2 AIV 直驱 RoCE（RDMA）

```text
┌──────────────── Host / Init ─────────────────────────────────────┐
│ 910B/C: RdmaTransportManager     （HCCP RA / RaAiQpCreate）       │
│ 950:    RdmaTransportManagerV2   （HCOMM + COMM_PROTOCOL_ROCE）   │
│                                                                  │
│ 1) 建 endpoint / RA 设备                                         │
│ 2) 注册 MR → lkey / rkey                                         │
│ 3) 建 RC QP（CONN_QP_AI_CORE / COMM_ENGINE_AIV）                  │
│ 4) 发布 AiQpRMAQueueInfo ↔ aclshmemi_rdma_info 到 Device         │
└───────────────────────────────┬──────────────────────────────────┘
                                │ Device VA: SQ/CQ + HW/SW DB
                                ▼
┌──────────────── RoCEv2 NIC / 网卡数据面 ─────────────────────────┐
│ RC QP: SQ + SCQ（AIV 驱动）；RQ/RCQ 按后端保留                    │
│ 寻址：remote VA + rkey；路径：GID / sgid_index / TC / SL         │
└───────────────────────────────▲──────────────────────────────────┘
                                │ AIV: fill WQE → ring SQ DB
┌──────────────── AIV Kernel（数据面）─────────────────────────────┐
│ aclshmemx_roce_{put,get}_nbi → post_send<BACKEND, WRITE/READ>    │
│ aclshmemx_roce_quiet         → poll SCQ（必要时敲 CQ DB）         │
│ Backend 编译期选择：IN_DIE / XSCALE / HNS_1825                    │
└──────────────────────────────────────────────────────────────────┘
```

### 2.3 侧车对照（同一范式，不同对象）

```text
          Host 控制面                 Device 数据面              底层
UDMA  HCOMM + UBC_CTP + Jetty   SQE/SGE + SW DB + JFC CQE   URMA over UB
RoCE  HCCP/HCOMM + RoCE + QP    WQE + HW/SW DB + SCQ CQE    RoCEv2 Ethernet
```

---

## 3. AIV 直驱 URMA 详细流程

### 3.1 Host 控制面（一次性）

| 步骤 | 动作 | 说明 |
| --- | --- | --- |
| 1 | `OpenDevice` / `CreateEndpoint` | `COMM_PROTOCOL_UBC_CTP`，按 Clos/EID 建本地 endpoint |
| 2 | `RegisterMemoryRegion` | 对每个 endpoint `HcommMemReg`；得到 token 保护信息 |
| 3 | `Connect` / `AsyncConnect` | allgather `EndpointDesc`；按 peer（或 relay slot）建 channel |
| 4 | `HcommChannelCreate(..., COMM_ENGINE_AIV, ...)` | **明确把数据面引擎交给 AIV**；HCOMM 内部分配 jetty |
| 5 | `WaitHcommChannelReady` | 等 channel READY |
| 6 | `BuildUdmaInfo` | 从 `SqContext.ubJfs` / `CqContext.ubJfc` / remote MR 填设备可见表 |
| 7 | H2D 发布 | `udmaInfoAddress` → device meta，kernel 经 `aclshmemi_get_udma_info_address` 取用 |

Jetty 创建发生在 **HCOMM 内部**；SHMEM 读回的关键字段：

| 来源 | 填到 | 含义 |
| --- | --- | --- |
| `ubJfs.jfsID` | `wq.wqn` | 发送 jetty 号 |
| `ubJfs.sqVa` | `wq.buf_addr` | SQ ring 基址 |
| `ubJfs.wqeSize` | `wq.wqe_size` | 单 WQE 字节数 |
| `ubJfs.dbVa` | `wq.db_addr` | **SW doorbell** VA |
| `ubJfc.jfcID` | `cq.cqn` | 完成 jetty 号 |
| `ubJfc.scqVa` | `cq.buf_addr` | CQ ring 基址 |
| `ubJfc.dbVa` | `cq.db_addr` | CQ SW doorbell |
| remote `tokenId/Value` + EID | `ubmem_info` | SQE 远端寻址与鉴权 |

`db_mode` 固定为 **`SW_DB`**：AIV 用 `st_dev` 直接写 DB 内存/寄存器镜像。

### 3.2 Device 数据面：`aclshmemi_udma_post_send`

入口：`aclshmemx_udma_put/get_nbi` → `aclshmem_ptr` 取远端 VA → `post_send`。

**步骤：**

1. **算 slot**  
   - direct：`slot = pe`  
   - relay：`slot = pe * rank_count + relay_pe`
2. **取 WQ 上下文**  
   `sq_ptr[slot][qp_idx]` → `head / buf_addr / wqe_size / db_addr / wqe_cnt`
3. **SQ 将满则先 poll CQ**（防 ring 覆盖）
4. **读远端 `ubmem_info`**（token、tid、tpn、eid_addr、rmt_jetty_type=1）
5. **组装 WQE**
   - **PIPE_S**：标量直接写 SQ GM + `dcci_cachelines`
   - **PIPE_MTE3**（公开 API 默认）：在 UB scratch 组 SQE+SGE，再 `DataCopyPad` 到 SQ
6. **推进 PI / 敲 SQ DB**  
   `st_dev(cur_head, db_addr)`；更新 `head`、`wqe_cnt`
7. **返回**（NBI）；完成靠 `aclshmemx_udma_quiet(pe)`

put/get 差异主要是 opcode：`WRITE` vs `READ`；远端 VA / 本地 VA 角色对调。

### 3.3 Quiet / CQE

`aclshmemx_udma_quiet(pe)`：

1. 对目标 slot（relay 时对所有相关 slot）poll SCQ，直到 `tail == wqe_cnt`
2. 等待 CQE **owner bit** 翻转；检查 `status/substatus == 0`
3. `st_dev(cur_tail & 0xFFFFFF, cq_db)` —— 注释明确 **reference URMA**
4. 同步写回 WQ `tail`

语义：`*_nbi` 只保证 WQE 已提交；读 get 目的地或复用 put 源缓冲前必须 quiet（或 signal 协议）。

### 3.4 UDMA SQE 关键字段（学习用）

`aclshmemi_sqe_ctx_t`（再加 `aclshmemi_sge_ctx_t`；数据搬路径约 64B 一块 WQEBB）：

| 字段 | 来源 / 填法 | 含义 |
| --- | --- | --- |
| `token_en` / `rmt_token_value` | remote MR token | UB 段访问鉴权 |
| `rmt_jetty_type` | 固定 **1**（peer jetty） | 远端对象类型 |
| `opcode` | WRITE / READ / WRITE_WITH_NOTIFY / FAA / CAS… | 操作类型 |
| `tp_id` | `ubJfs.tpID` | 传输层实例 |
| `rmt_jetty_or_seg_id` | `tokenId`（tid） | 远端段/jetty 标识 |
| `rmt_eid_{l,h}` | 16B URMA EID | 远端 endpoint 身份 |
| `rmt_addr_*` | `aclshmem_ptr` 结果 | 远端 VA |
| SGE `va/len/token_id` | 本地缓冲 | 本端 payload（或 AMO scratch） |

---

## 4. RoCE 路径要点（对照学习）

### 4.1 Host 控制面差异

| 项 | UDMA / URMA | RoCE / RDMA |
| --- | --- | --- |
| 传输管理器 | `UdmaTransportManager` | v1 `RdmaTransportManager`；950 上 v2 `RdmaTransportManagerV2` |
| 控制面 API | **HCOMM only** | 910：`HCCP RA`（`RaAiQpCreate` 等）；950：`HCOMM` |
| 协议 | `COMM_PROTOCOL_UBC_CTP` | `COMM_PROTOCOL_ROCE` / RoCEv2 |
| 队列对象 | Jetty（JFS/JFC） | RC **QP**（SQ/RQ/SCQ/RCQ） |
| 寻址身份 | **EID**（16B） | **GID / sgid_index** + IP（V2 endpoint） |
| 内存钥匙 | **tokenId / tokenValue** | **lkey / rkey** |
| AIV 标记 | `COMM_ENGINE_AIV` | 同：`COMM_ENGINE_AIV` 或 `CONN_QP_AI_CORE` + `cq_cstm=1`（AIV poll CQ） |

### 4.2 Device 数据面差异

| 项 | UDMA | RoCE |
| --- | --- | --- |
| 公开 API | `aclshmemx_udma_*` | `aclshmemx_roce_*` |
| WQE 布局 | URMA SQE+SGE（jetty/EID/token） | 后端相关：IN_DIE 32B+16B；XSCALE 128B；HNS_1825 64B BE |
| Doorbell | **统一 SW_DB**，`st_dev(PI)` | IN_DIE/XSCALE/HNS 各有 HW/SW/双写模型；常打包 QPN、PI、SL/COS |
| 完成队列 | JFC CQE（`is_jetty` 等） | SCQ CQE（owner / wqe_id / status） |
| Backend | 单套 UDMA 实现 | **编译期三后端**：`IN_DIE`（910）、`XSCALE` / `HNS_1825`（950） |
| 平台 | **仅 Ascend950**（`__NPU_ARCH__==3510`） | 910B/C + Ascend950 |
| 原子 | 内建 FAA/CAS/…；float add 走 reduce-write | 主要 **XSCALE** 支持；IN_DIE/HNS 基本不支持 |
| 单次上限 | put 请求 ≤ **256MB** | 受 WQE/后端与 MR 约束（实现按消息长度切分/限流） |

### 4.3 RoCE WQE/DB 直觉（与 UDMA 对照）

- **IN_DIE（910）**：WQE 像经典 HNS RoCEv2——ctrl（opcode/owner/rkey/remote VA）+ SGE；SQ DB 打包 QPN + `HNS_ROCE_V2_SQ_DB` + PI + SL。
- **XSCALE（950）**：更大 WQE（128B），Diamond send doorbell（`qp_id` + `next_pid`）；支持 batch defer/submit 与 atomics。
- **HNS_1825（950）**：64B WQEBB、大端字段；SW PI 镜像 + HW DB（含 `sgid_index`）。

共同点：都是 AIV 写 ring → 刷可见性 → 敲铃 → poll CQ。  
不同点：铃里装的是 **URMA jetty PI**，还是 **RoCE QPN/PI/SL**。

---

## 5. 对比总表（建议背下来）

| 维度 | AIV→URMA（UDMA） | AIV→RoCE（RDMA） |
| --- | --- | --- |
| 互联 | 灵衢 **UB** | **以太网 RoCEv2** |
| SHMEM 引擎名 | UDMA | RDMA / ROCE |
| Host 协议枚举 | `UBC_CTP` | `ROCE` |
| 队列抽象 | Jetty（JFS/JFC） | QP |
| 远端身份 | EID | GID / IP |
| 内存保护 | token | lkey/rkey |
| DB 模型 | SW_DB 为主 | 后端相关 HW/SW |
| 典型场景 | 950 节点内/UB 域低时延 RMA | 跨节点、无 UB 或需以太网 |
| 高阶路由位 | `ACLSHMEM_TRANSPORT_UDMA` | `ACLSHMEM_TRANSPORT_ROCE` |
| Init 开关 | `ACLSHMEM_DATA_OP_UDMA` | `ACLSHMEM_DATA_OP_ROCE` |
| 与 HCCP 关系 | **不用 HCCP 数据面** | 910 控制面核心是 HCCP；950 迁到 HCOMM |

高阶 `aclshmem_get/put` 路由（同 PE 多引擎时）：

```text
SDMA → UDMA → MTE → RoCE
```

即：能走机内 SDMA/UB UDMA 时，不会落到 RoCE。

---

## 6. 学习路径：如何读代码

建议按“先 Host 建链，再 Device 敲铃，最后对照字段”阅读：

### 6.1 URMA / UDMA

| 顺序 | 文件 | 看什么 |
| --- | --- | --- |
| 1 | `src/host/transport/device_udma/device_udma_transport_manager.cpp` | Endpoint / MemReg / ChannelCreate(`COMM_ENGINE_AIV`) / FillWqCtx |
| 2 | `src/host/transport/device_udma/device_udma_def.h` | Host/Device 共布局：wq/cq/ubmem/info |
| 3 | `src/device/gm2gm/engine/shmemi_device_udma.h` | SQE/SGE/CQE 位域、opcode |
| 4 | `src/device/gm2gm/engine/shmem_device_udma.hpp` | `post_send`、`poll_cq`、quiet、MTE3 组装 |
| 5 | `include/device/gm2gm/engine/shmem_device_udma.h` | 公开 API 契约（256MB、quiet、并发限制） |
| 6 | `examples/udma_demo/`、`examples/udma_atomic_add/` | 端到端用法 |

### 6.2 RoCE / RDMA

| 顺序 | 文件 | 看什么 |
| --- | --- | --- |
| 1 | `device_rdma_transport_manager.cpp` / `_v2.cpp` | HCCP vs HCOMM 建 QP |
| 2 | `fixed_ranks_qp_manager.cpp` | `RaQpAiCreate`、`cq_cstm=1`、CopyAiWQInfo |
| 3 | `shmemi_device_rdma.h` | `aclshmemi_rdma_info` / sq_ctx |
| 4 | `rdma_backends/rdma_device_backend_{in_die,xscale,hns_1825}.hpp` | WQE/DB/CQE 差异 |
| 5 | `shmem_device_rdma.hpp` | `aclshmemx_roce_*`、quiet/sync |
| 6 | `examples/rdma_demo/` | 端到端用法 |

### 6.3 汇合点

| 文件 | 作用 |
| --- | --- |
| `src/device/gm2gm/shmem_device_rma.hpp` | 高阶 RMA 按 topo 位选择 SDMA/UDMA/MTE/RoCE |
| `src/host/transport/composite_transport_manager.*` | 多引擎组合打开 |
| `docs/glossary.md` | URMA / UDMA / RoCE / AIV 官方释义 |

---

## 7. 端到端时序（对照）

### 7.1 UDMA put

```text
Host: OpenDevice → MemReg → Connect(AIV jetty) → publish udma_info

AIV:  aclshmemx_udma_put_nbi(dst, src, ub, n, pe, sync_id)
        aclshmem_ptr(dst, pe)
        slot = pe（或 relay 映射）
        读 wq_ctx / ubmem_info
        填 SQE(EID,token,tid,opcode=WRITE,rmt_addr) + SGE(local)
        dcci 或 MTE3 DataCopyPad
        st_dev(new_head, sq_db)          ← 直驱 URMA
      aclshmemx_udma_quiet(pe)
        poll JFC until owner flip
        st_dev(tail, cq_db)
```

### 7.2 RoCE put

```text
Host: OpenDevice → MemReg → Connect(AIV QP) → publish qp_info

AIV:  aclshmemx_roce_put_nbi(dst, src, ub, n, pe, sync_id)
        aclshmem_ptr(dst, pe)
        读 sq_ctx[pe] / rkey,lkey
        fill WQE(WRITE, remote VA, rkey, local SGE)
        dcci / DataCopyPad
        ring_sq_doorbell<BACKEND>()      ← 直驱 RoCE NIC
      aclshmemx_roce_quiet(pe, ub, sync_id)
        poll SCQ until CI catches PI
        （按后端）ring CQ DB
```

---

## 8. 选型与常见误区

### 8.1 选型

| 场景 | 建议 |
| --- | --- |
| Ascend950，UB 可达 | **UDMA**（低时延内存语义） |
| 910B/C 节点内大包 | **SDMA / MTE**（见 AIV-SDMA 文档）；跨节点 **RoCE** |
| 跨机架 / 仅以太网 | **RoCE** |
| 需要远端原子（950） | UDMA 全套 AMO；RoCE 优先确认 **XSCALE** 后端 |
| 不清楚拓扑 | 用高阶 `aclshmem_put/get`，让 topo 路由自动选 |

### 8.2 误区

1. **把 UDMA 当成“另一种 RoCE”**  
   API 都像 RDMA put/get，但对象是 jetty/EID/token，不是 QP/GID/rkey。
2. **以为 Host 在 runtime 敲 UDMA DB**  
   Host 只建链；runtime DB 由 AIV `st_dev` 完成。
3. **混淆 URMA 与 UDMA**  
   URMA = 协议/语义；UDMA = SHMEM 引擎与 API 名。
4. **quiet 前读 get 缓冲**  
   NBI 只保证提交；完成以 quiet/signal 为准。
5. **同 PE 并发 RMA/AMO**  
   UDMA/RoCE 文档均声明：**到同一 PE 的并发 RMA/AMO 不支持**。
6. **950 上 RoCE 后端选错**  
   XSCALE 与 HNS_1825 的 WQE/DB 布局不兼容，需与 `-rdma_backend` 一致。

---

## 9. 核心代码索引（cann/shmem）

### 9.1 UDMA：AIV channel + 填 jetty 上下文

```cpp
// HcommChannelCreate(..., COMM_ENGINE_AIV, ...)
// src/host/transport/device_udma/device_udma_transport_manager.cpp

void UdmaTransportManager::FillWqCtx(
    const SqContext& sq_context, uint32_t dst_pe, aclshmemi_udma_wq_ctx_t& dst_wq) const
{
    const auto& ubJfs = sq_context.contextInfo.ubJfs;
    dst_wq.wqn = ubJfs.jfsID;
    dst_wq.buf_addr = ubJfs.sqVa;
    dst_wq.wqe_size = ubJfs.wqeSize;
    dst_wq.db_mode = aclshmemi_udma_db_mode_t::SW_DB;
    dst_wq.db_addr = ubJfs.dbVa;
    // ...
}
```

### 9.2 UDMA：AIV 敲 SQ / CQ doorbell

```cpp
// src/device/gm2gm/engine/shmem_device_udma.hpp

// Ring SQ Doorbell
st_dev(cur_head, (__gm__ uint32_t*)qp_ctx_entry->db_addr, 0);

// Ring CQ Doorbell (reference URMA implementation)
st_dev((uint32_t)(cur_tail & 0xFFFFFF), (__gm__ uint32_t*)cq_ctx_entry->db_addr, 0);
```

### 9.3 RoCE：AIV QP 创建（910 HCCP 示例）

```cpp
// fixed_ranks_qp_manager.cpp（CONN_QP_AI_CORE）
attr.qp_attr.qp_type = IBV_QPT_RC;
attr.data_plane_flag.bs.cq_cstm = 1;  // AIV 自己 poll CQ
RaQpAiCreate(rdmaHandle_, attr, channel.aiQpInfo, channel.qpHandles[qpType]);
```

### 9.4 RoCE：950 HCOMM 建链

```cpp
// device_rdma_transport_manager_v2.cpp
HcommChannelCreate(endpointHandle_, COMM_ENGINE_AIV, channelDescs.data(), ...);
// channelDescs[].protocol / roceAttr = COMM_PROTOCOL_ROCE
```

---

## 10. 与 SDMA 文档的关系

同仓还有 **AIV 直驱 SDMA** 路径（STARS SQE + SQ Tail DB），面向 910 机内 DMA，不走 URMA/RoCE。  
三者可记为：

```text
SDMA  —— 机内 STARS/SDMA 引擎（SQE type=11）
UDMA  —— UB 上的 URMA jetty（SQE+SGE，EID/token）
RoCE  —— 以太网上的 RDMA QP（WQE，rkey/GID）
```

都是 **Host 建资源、AIV 跑数据面**；差别在“铃敲给谁、信封里写什么地址”。

---

## 11. 参考

- 上游仓库：<https://gitcode.com/cann/shmem>
- Issue：[SHMEM 支持 AIV 直驱 UDMA](https://gitcode.com/cann/shmem/issues/161)（控制面 jetty/HCCP 封装 + 数据面 put/get：组 WQE、敲 DB、轮询 CQE）
- 项目术语：`docs/glossary.md`（URMA / UDMA / RoCE / AIV）
- Device API：`docs/api/device_api.rst`（`shmem_device_udma.h` / `shmem_device_rdma.h`）
- 构建：Ascend950 需 `-soc_type Ascend950`；RoCE 另需 `-enable_rdma`，950 指定 `-rdma_backend XSCALE|...`
