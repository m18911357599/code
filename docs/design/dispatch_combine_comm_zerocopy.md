# `dispatch.combine`：RDMA fullmesh 的 Write / Read 语义

> 范围：仅 `comm_alg=fullmesh`；跨卡通信只用 **RDMA Write** 与 **RDMA Read**。  
> 不含 hierarchy / HCCS 中转。宿端只做控制面（初始化 / 绑 Tensor / 下发 / 等待），**不拼 RDMA 子图、不搬 token**。  
> 记号：`BS` 本卡 token 数，`H` hidden，`K` topK，`E` 全局专家数，`R` EP world size，`L=E/R` 每卡本地专家数。

---

## 0. fullmesh 通信模型

每对 rank 之间有直连 QP；每卡预注册对称窗（send/recv + flag）。  
token 只发往 **目标专家所在卡**，无中间跳。

```text
Dispatch：源卡  --RDMA Write-->  专家卡 recv 窗
Combine ：专家卡 --RDMA Write-->  源卡   combine 窗
           或源卡 --RDMA Read -->  专家卡 send 窗（见 §5）
```

本卡专家：本地拷贝，不进 RDMA。

---

## 1. 宿端处理逻辑（Host 控制面）

宿端 = CPU Host。数据面全程在 Device↔NIC；宿端只负责 **一次初始化** 与 **每步轻量调度**。

### 1.1 职责边界

```text
┌──────────────────────────── 宿端 Host ────────────────────────────┐
│  初始化：EP 域 / QP / 对称窗注册 / 上界 A / stream                  │
│  每步：绑 Tensor 指针 → launch Dispatch/Combine → event/同步       │
│  禁止：算路由表、拼 AllToAllV 子图、Host DRAM 中转 token、按 rank 循环提 RDMA │
└───────────────────────────────┬───────────────────────────────────┘
                                │ launch（描述符 + 指针，无大块拷贝）
                                ▼
┌──────────────────────────── Device ───────────────────────────────┐
│  Matmul → TopK → map/dedup/pack → RDMA Write/Read → FFN → reduce   │
└───────────────────────────────────────────────────────────────────┘
```

| 做 | 不做 |
|---|---|
| 创建/加入 EP 通信域，配置 `comm_alg=fullmesh` | 扫描 `expert_ids` 算 send/recv count |
| 按 `max_BS,H,K,R` 注册对称窗，设够 `HCCL_BUFFSIZE` | 在 Host 拼 per-rank RDMA WQE 列表 |
| 分配/复用 Device Tensor（`x`,`expert_ids`,`expand_x`,`assist`,…） | `cuda/h2d` 搬 token 或路由大表 |
| `aclnn*` / `torch_npu` 下发 + stream/event | 每步动态改 QP 拓扑 |
| 读标量上界 / 可选 D2H 少量 debug count | 参与 flag 轮询与加权归约 |

### 1.2 初始化（进程 / EP 组生命周期，一次）

```text
Host:
  1) 解析并行配置: R, E, L, K, max_BS, H, dtype
  2) 建立 group_ep；fullmesh：保证 rank 两两可达
  3) 计算对称窗字节数（dispatch_recv / combine_* / flag）
     要求 ≥ 2*(BS*R*min(L,K)*H*elem + 余量) 量级（以实际 HCCL 公式为准）
  4) Device 侧注册内存（rkey 交换在通信库内完成，Host 只触发）
  5) 创建 stream_comm / stream_compute（可同一 stream）
  6) 预分配常驻 Tensor 壳：expand_x[A,H]、assist、ep_recv_counts…
```

初始化后 Host **缓存** buffer handle；后续 step 不再重新注册（除非 shape 上界变化）。

### 1.3 每步调度（一层 MoE）

```text
Host 伪码（单 rank 视角）:

# —— 前置：业务已把 x 放在 Device ——
x, W_gate = ...                    # Device 指针，Host 只持引用

# ① 下发 Gate + TopK（可融合成一算子，或两段）
launch_gate_topk(x, W_gate, out={logits, expert_ids, expert_scales})
# Host 不等待中间结果；或仅 wait 若框架强制同步（应避免）

# ② 下发 Dispatch（内含 map/dedup/pack + RDMA Write）
launch_moe_dispatch(
  x, expert_ids, expert_scales,
  group_ep, ep_rank, moe_expert_num,
  comm_alg="fullmesh",
  out={expand_x, assist_info, expert_token_nums, ep_recv_counts, ...})

# ③ 本卡专家 FFN（消费 expand_x / expert_token_nums）
launch_expert_ffn(expand_x, expert_token_nums, out=expert_out)

# ④ 下发 Combine（RDMA Write 回传 或 Read 拉取 + 加权）
launch_moe_combine(
  expert_out, expert_ids, assist_info, ep_recv_counts, expert_scales,
  group_ep, ...,
  out=x_out)

stream_wait / event_sync_if_needed()
# Host 此时才消费 x_out（若后续仍在 Device，可继续不下到 Host）
```

时序（宿端视角）：

```text
Host timeline
 |--init--|---- step n ----|---- step n+1 ----|
          |                |
          | launch Gate    |
          | launch Dispatch|   ← 不等待路由表回 Host
          | launch FFN     |   ← 依赖 Device 侧 flag/event
          | launch Combine |
          |  (optional sync before 下一段非 MoE)
```

### 1.4 绑参与 workspace

每步 Host 传给执行器的是 **句柄**，不是载荷拷贝：

```text
args:
  ptr_x, ptr_expert_ids, ptr_scales,
  ptr_expand_x, ptr_assist, ptr_counts,
  ep_world_size, ep_rank_id, moe_expert_num,
  global_bs / max_bs, quant_mode, comm_alg=fullmesh
workspace:
  通信库内部临时（仍在 Device）；Host 只问 GetWorkspaceSize 一次并复用
```

shape 约定由 Host 在 launch 前写死上界 `A`；真实 `n_recv` 写在 Device 的 `expert_token_nums`，**默认不 D2H**。若框架 tiling 需要 Host 知真实长度，只 D2H 该小 Tensor（KB 级），仍不算数据面破零拷贝。

### 1.5 同步与重叠

| 模式 | 宿端行为 | 说明 |
|---|---|---|
| 默认串流 | 四段依次 enqueue 到同一 stream | 最简单；Device 内核间自同步 |
| 双流重叠 | Dispatch/Combine 走 `stream_comm`，FFN/`Gate` 走 `stream_compute`，用 Device event 交接 | Host 只插 event record/wait，不轮询 RDMA CQ |
| 异步返回 | launch 后立即返回 PyTorch/ME 后续节点 | 依赖图上的隐式依赖；Host 禁止提前读 `x_out` |

宿端 **不** 轮询 `dispatch_flag` / CQ；完成语义留在 Device 算子内部或通信库。

### 1.6 错误与保底

```text
Host:
  - 启动前校验：HCCL_BUFFSIZE、A ≥ 理论上界、各卡 R/E/K 一致
  - launch 失败 / 超时：报错并可选 dump ep_recv_counts（此时才允许 D2H）
  - 禁止回退路径：把 token gather 到 Host 再 socket 发送（会破坏 fullmesh 零拷贝契约）
```

### 1.7 与源端 / RDMA 的分工一览

```text
宿端 Host          源端 Device              远端 Device
─────────          ────────────             ────────────
init 窗/QP
launch Gate   →    Matmul+TopK
launch Dispatch→   map/dedup/pack
                   RDMA Write ───────────►  poll + 整理 + FFN
launch Combine →                   ◄─────  Write 回传 / 等 Read
                   reduce → x_out
(optional sync)
```

---

## 2. 源端：Matmul → 选专家 → token 转换

源卡在发起任何 RDMA Write 之前，完成本地三条链：**门控 Matmul、选专家、token 布局转换**。  
后两步把“自然序 token”变成“可 Write 的 (dst_rank, slot) 载荷”。

### 2.1 总示意

```text
源卡 Device
┌──────────────────────────────────────────────────────────────────────────┐
│                                                                          │
│  x[BS, H]                                                                │
│     │                                                                    │
│     │  ① Gate Matmul                                                     │
│     ▼                                                                    │
│  logits = x @ W_gateᵀ     →  [BS, E]                                     │
│     │                                                                    │
│     │  ② 选专家 (softmax + TopK)                                          │
│     ▼                                                                    │
│  expert_ids[BS, K]   expert_scales[BS, K]                                │
│     │                      │                                             │
│     │  ③ token 转换        │  (权重随路由走，Combine 再用)                 │
│     │  map / dedup / pack  │                                             │
│     ▼                      ▼                                             │
│  send_buf[n_write, H]   meta(dst_rank, slot, src_tok, scales…)           │
│     │                                                                    │
│     │  ④ RDMA Write (fullmesh)                                           │
│     ▼                                                                    │
│  专家卡.dispatch_recv[src][slot]                                         │
└──────────────────────────────────────────────────────────────────────────┘
```

### 2.2 ① Gate Matmul

```text
                    W_gate[E, H]
                         │
  token0 ──┐             │
  token1 ──┼─ x[BS,H] ──►│◄── Matmul ──► logits[BS, E]
  token2 ──┘             │                 │
                         │            每行 E 个专家分
```

- 输入：本卡自然序 hidden `x[i,:] = token_i`。  
- 输出：每个 token 对全部专家的分数；**仍按 token 行序**，未打乱。  
- 通信无关：Matmul 纯本地，不产生 RDMA。

### 2.3 ② 选专家

```text
logits[BS, E]
    │  softmax / sigmoid（按实现）
    │  TopK → 每行取 K 个最大
    ▼
token0:  expert_ids[0]=[e2, e5, …]   scales[0]=[w2, w5, …]
token1:  expert_ids[1]=[e0, e2, …]   scales[1]=[w0, w2, …]
   …                 │
                     ▼
            e → (dst_rank, local_eid) = (e/L, e%L)
```

小例子（`R=4, L=2, K=2`）：

```text
token  expert_ids   dst_ranks(去重前)   dst_ranks(去重后)
  t0    [0, 1]       [0, 0]              [0]        ← 同卡两专家，只 Write 1 次
  t1    [2, 5]       [1, 2]              [1, 2]     ← 两次 Write
  t2    [3, 4]       [1, 2]              [1, 2]
```

选专家输出是 **索引 + 权重**，不是已重排的激活；激活仍停在 `x`。

### 2.4 ③ token 转换（自然序 → Write 载荷）

把 `(x, expert_ids)` 转成 fullmesh Write 所需的 **紧凑 send 布局**：

```text
自然序 (token-major)              通信序 (rank-major, 已去重)
─────────────────────             ─────────────────────────────
x[0] ──► 专家 0,1 @rank0    \\
x[1] ──► 专家 2@r1, 5@r2     }──►  pack
x[2] ──► 专家 3@r1, 4@r2    /

send_buf / WQE 列表（示意）:
  Write#0: payload=x[0] → rank0.slot_a
  Write#1: payload=x[1] → rank1.slot_b
  Write#2: payload=x[1] → rank2.slot_c
  Write#3: payload=x[2] → rank1.slot_d
  Write#4: payload=x[2] → rank2.slot_e
  （t0→rank0 仅一条；t0 在 rank0 上再 fan-out 到 local e0,e1）
```

转换三步（均在源卡）：

| 步 | 做什么 | 输入 → 输出 |
|---|---|---|
| map | `expert_id → dst_rank / local_eid / remote_slot` | `expert_ids` → 描述符表 |
| dedup | 同一 `(token, dst_rank)` 合并 | K 条路由 → ≤K 次 Write |
| pack | gather `x[token]` 到注册窗 / send 视图 | `x[BS,H]` → `send_buf[n_write,H]` |

```text
         expert_ids                    x[BS,H]
              │                           │
              ▼                           │
         ┌─────────┐                      │
         │  map    │── dest 表 ──┐        │
         └─────────┘             │        │
              │                  ▼        ▼
              ▼            ┌──────────────────┐
         ┌─────────┐       │ gather / pack    │
         │  dedup  │──────►│ send_buf + WQE   │──► RDMA Write
         └─────────┘       └──────────────────┘
              │
              └──► assist / counts（给 Combine，不经 Host）
```

要点：

- **Matmul 不改变 token 布局**；**选专家只产索引**；**转换才搬激活**。  
- pack 目标应是 RDMA 已注册窗（或 view），以便 Write 零 staging。  
- `expert_scales` 可随 meta 走或随 token 一并 Write，供远端/回程加权。

### 2.5 与后续 RDMA 的衔接

```text
① Matmul     ② TopK        ③ 转换           ④ Write
x → logits → ids/scales → send_buf+meta → 专家卡.dispatch_recv
     本地         本地          本地            fullmesh
```

---

## 3. 选专家 → Write/Read 描述符

§2.3–2.4 的 map 结果直接落到 WQE：

```text
e → dst_rank = e / L
    local_eid = e % L
    slot      = 远端 recv 窗内偏移（由 src_rank、token、去重序决定）
```

| 阶段 | 操作 | 本地地址 | 远端地址 |
|---|---|---|---|
| Dispatch | Write | `send_buf` 中该 token 行 | `dst_rank.dispatch_recv[src][slot]` |
| Combine | Write 或 Read | 专家输出 / 源卡输出槽 | 对端 combine 窗对应 slot |

同行 K 个 `expert_ids` 不重复；可选 mask 关掉无效槽，避免空 Write。

---

## 4. 去重（减少 Write 次数）

同一 token 的多个 expert 落在同一 `dst_rank` 时：

```text
ranks = unique(dst_rank(expert_ids[t,*]))
for r in ranks:
    RDMA Write token_t → r.dispatch_recv  一次
# r 卡内再 fan-out 到各 local_eid 输入区
```

- Write 键：`(src_rank, token, dst_rank)`，不是 `(…, dst_expert)`。  
- Combine 对称：同卡多专家输出可先本地加权，再 **一次** Write/Read 回源卡。

---

## 5. Write / Read 语义（fullmesh 唯一数据面）

### 5.1 对称 Buffer

```text
rank r:
  dispatch_recv[src][slot][H]   # 被源卡 Write
  dispatch_flag[src]            # Write 完成通知（或写后原子加）
  combine_send[dst][slot][H]    # 专家结果待取/待写
  combine_recv[src][slot][H]    # 源卡收结果（Write 回传时）
  combine_flag[…]
```

### 5.2 Dispatch = RDMA Write

```text
源卡:
  1) §2 token 转换得到 send_buf / WQE
  2) RDMA Write(local_token, remote=r.dispatch_recv[src][slot], len=H')
  3) 写完后通知：Write flag 或带 immediate / 单独小 Write

专家卡:
  poll dispatch_flag 收齐预期 src
  按 local_eid 整理 → expand_x → FFN
```

语义：**推模型（push）**。发送方主动把数据推进远端 recv 窗；接收方只轮询完成位，不发起数据面 Read。

### 5.3 Combine = RDMA Write（回写）或 RDMA Read（拉取）

**Write 回传（常用 push）：**

```text
专家卡:  (可选本地加权) → RDMA Write → 源卡.combine_recv[expert_rank][slot]
源卡:    poll combine_flag → Σ w·y → x_out[BS,H]
```

**Read 拉取（pull，语义等价、发起端在源卡）：**

```text
专家卡:  把 y 放进本卡 combine_send[src][slot]，置 ready flag
源卡:    poll 对端 ready → RDMA Read(remote=combine_send, local=combine_recv)
         → Σ w·y → x_out
```

| | RDMA Write（Combine） | RDMA Read（Combine） |
|---|---|---|
| 数据面发起方 | 专家卡 | 源卡 |
| 远端窗角色 | 源卡 `combine_recv` 被写 | 专家卡 `combine_send` 被读 |
| 完成通知 | 写 flag / 写后通知 | Read 完成 CQ + 对端 ready |
| fullmesh 约束 | 直连到源卡，无中转 | 直连读专家卡，无中转 |

Dispatch 固定 **Write**；Combine 在 Write/Read 二选一。

### 5.4 与零拷贝的交界

- §2 pack 的 `send_buf` 必须是 **已注册** 窗（或 view）。  
- FFN 若直接写 `combine_send`，Combine 的 Write/Read 无 Device staging。  
- Host 不参与数据面。

---

## 6. 远端收齐 → 整理 → 计算 → 回送（仅 Write/Read）

```text
① 收齐：poll 各 src 的 dispatch_flag（数据已由对端 Write 进本卡窗）
② 整理：dispatch_recv → 按 local_eid 展开 expand_x（去重 token 本地 fan-out）
③ 计算：FFN(expand_x 切片)
④ 回送：
     Write 路径: Write(y) → 源卡 combine_recv + flag
     Read  路径: 发布 combine_send + ready，等源卡 Read
⑤ 源卡: 对各专家回程槽做加权归约 → x_out
```

---

## 7. 端到端（单层）

```text
Host:      init 窗/QP → 每步 launch Gate / Dispatch / FFN / Combine
Src Dev:   Matmul → TopK → map/dedup/pack
Dispatch:  RDMA Write → (专家卡 poll)
Expert:    pack → FFN
Combine:   RDMA Write 回源  或  源卡 RDMA Read
Src Dev:   reduce → x_out
Host:      optional stream sync（仍不搬 token）
```

---

## 8. 小结

| 步骤 | fullmesh RDMA 语义 |
|---|---|
| 宿端控制面 | 初始化 EP/对称窗；每步绑指针并 launch；不拼子图、不搬 token、不轮询 CQ |
| 源端 Matmul | `x @ W_gate` → `logits[BS,E]`，布局仍为 token 自然序 |
| 选专家 | TopK → `expert_ids/scales`；再 `e→(dst_rank,local_eid)` |
| token 转换 | map + dedup + gather/pack → `send_buf` + WQE |
| 写远端 buffer | Dispatch：**RDMA Write** 进专家卡 `dispatch_recv` |
| 收齐整理计算回送 | poll → 整理 → FFN → Combine：**Write** 或 **Read** |
| 宿端零拷贝 | 数据面只有 NIC↔注册窗；Host 仅控制面 |
