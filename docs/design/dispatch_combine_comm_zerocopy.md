# `dispatch.combine` 通信实现分析：选专家 · 去重 · 远端 Buffer · 回送 · 宿端零拷贝

> 视角：通信路径（EP AllToAllV / RDMA），而非纯计算语义。  
> 对照实现：昇腾 `MoeDistributeDispatch` / `MoeDistributeCombine`（及 v2），并与 DeepEP 类低时延零拷贝路径对照。  
> 记号：`BS`=本卡 token 数，`H`=hidden，`K`=topK，`E`=全局专家数，`R`=EP world size，`L=E/R`=每卡本地专家数。

---

## 0. 通信视角总览

```text
[宿端 Host] 只做：构图 / 下发一次融合算子 / 可选异步 event
        │  （无路由表 AllGather、无 Host 拼 RDMA 子图、无 Host 侧 token 拷贝）
        ▼
[本卡 Device · Dispatch]
  ① 选专家：读 expert_ids[BS,K] → rank/expert 映射
  ② 去重：同 rank 多专家只发 1 次 token
  ③ 写远端 buffer：AIV(+AICPU) 直驱 RDMA → 对端对称/共享内存槽位
        ▼
[远端 Device · 收齐 → 整理 → FFN → Combine]
  ④ 按专家整理 expand_x；算 expert；局部加权；RDMA 回写源卡
        ▼
[源卡 · Combine 后处理]
  加权归约到原始 token 序 → 输出 x_out[BS,H]
```

一对算子必须成对、共享 EP 域与元数据（v2 用 `assist_info_for_combine` / `ep_recv_counts` 等）。通信算法分：

| `comm_alg` | 路径 | 通信量特征 |
|---|---|---|
| `fullmesh` | token 直发目标专家所在卡（RDMA） | 接近 `BS·H·K`（本卡专家可免跨卡） |
| `hierarchy` | 跨机同号卡 RDMA + 机内 HCCS/转发 | 利用 Node-Limited Routing，跨机量可显著下降 |

---

## 1. 选专家（通信侧如何落地）

### 1.1 输入与映射

业务侧 Gating 已算出 `expert_ids[BS,K]`（及可选 `expert_scales`）。Dispatch **不重算分数**，只做通信路由：

```text
expert_id e ∈ [0, E)
  → owning_rank  = shared_offset + e / L     # 共享专家卡排布另计
  → local_eid    = e % L
```

约束（通信正确性前提）：

- 同行 K 个 `expert_ids` **不重复**（API 约束），否则去重语义与 expand 索引会歧义。
- 可选 `x_active_mask`：关掉无效 token / 无效 (token,k) 槽，避免空发。

### 1.2 Device 内“选专家”流水（替代 Host 路由表）

传统路径：Host/另一算子 AllGather 路由表 → Host 算 send/recv count → 再下 AllToAllV。  
融合路径把下列全部沉到 Device：

1. **扫描** `expert_ids`，按目标 rank / 本地 expert 原子计数（notify / count 阶段）。
2. **前缀和** 得到每目的地的 send offset、每专家 recv 容量（`expert_token_nums` / `ep_recv_counts`）。
3. **重排**：把发往同一目标的 token 在本地 send 窗内聚簇（AICPU 时代按 rank 汇聚；AIV 直驱后可降到 token 粒度发 WQE）。
4. **预计算 Combine 元数据**：源 token 下标、槽位、权重搬运信息 → v2 的 `assist_info_for_combine`（`A×128` 打包），避免 Combine 再扫一遍全局路由。

通信语义：**选专家 = 把 (token → expert) 变成 (token → rank, slot) 的可 RDMA 写描述符**。

---

## 2. 去重（减冗余跨卡传输）

### 2.1 问题

TopK 常使同一 token 的多个 expert 落在 **同一 rank**。若按 “每个 (token, expert) 各发一次”，跨卡流量 ≈ `BS·H·K`，其中大量是同卡重复载荷。

### 2.2 Dispatch 去重

```text
对每个 token t:
  ranks = unique({ owning_rank(expert_ids[t,k]) | k < K })
  for r in ranks:
      RDMA 写 token_t 到 r 的 recv buffer 一次
  # 远端 rank 内再 fan-out 到各 local expert 输入槽
```

要点：

- **跨卡键**：`(src_rank, token_id, dst_rank)` 唯一；**不是** `(…, dst_expert)`。
- 远端后处理按 `assist_info` / expand 索引，把同一份 payload **本地拷贝或视图** 到多个 expert 的连续区间（`do_expand` 语义）。
- 前缀和统计里会出现两类计数：
  - **去重后 per-rank recv count**（决定 RDMA 载荷条数）；
  - **按 expert 展开后的 token 数**（决定 FFN 输入长度 / `expert_token_nums`）。

### 2.3 Combine 侧对称去重（局部归约）

Combine 反向优化：

```text
本卡上属于同一源 token 的多个 local expert 输出
  → 先按 expert_scales 做局部加权求和
  → 再 RDMA 回源卡（或 hierarchy 先机内再跨机）
源卡再与其它 rank 回传结果做最终 reduce
```

hierarchy 在 Combine 上常体现为：**server 内先汇总同一 token，再跨机**，进一步砍跨机字节数。

通信量直觉（忽略本卡与对齐填充）：

| 策略 | Dispatch 跨卡量级 | Combine 跨卡量级 |
|---|---|---|
| 无去重 | ~`BS·H·K` | ~`BS·H·K` |
| 仅 Dispatch 去重 | ~`BS·H·|#unique ranks|` | 仍可 ~`BS·H·K` |
| Dispatch 去重 + Combine 局部 reduce | 同上 | ~`BS·H·|#unique ranks|` |

---

## 3. 写远端 Buffer

### 3.1 Buffer 模型

EP 域预先登记 **对称 / 共享通信窗**（HCCL buffer，`HCCL_BUFFSIZE` 需按 `comm_alg` 与 shape 配足）：

```text
每卡 recv 窗（示意）:
  [header: counts / flags / offsets]
  [payload slots: 按 src_rank × token 或 expert × rank×token 排布]
  [combine 回传窗 / 双缓冲]
```

低时延路径常见布局（与 DeepEP LL 对照）：

```text
recv_x: [L, R * max_tokens_per_rank, H]   # 或扁平 (A, H)，A 为上界
flag  : 每 src 完成标记 / 原子计数
```

### 3.2 写路径（谁写、写什么）

1. **本卡预处理**：量化（可选 int8）、拼 send 描述符（地址、长度、dst rkey/slot）。
2. **下发 RDMA Write**：
   - 阶段一：AIV → GM 消息区 → **AICPU** 双核驱动 RDMA；
   - 阶段三： **AIV 直驱 RDMA**，多核并行下 WQE，可按 token 粒度发送。
3. **写完成通知**：对端 flag / 计数原子加；接收侧 AIV **轮询 flag**，替代传统 Host 侧 RDMA 前后同步 RTT。
4. **本卡专家**：可走本地拷贝，不进 RDMA。

`fullmesh`：源卡直接 Write 到目标专家卡的 slot。  
`hierarchy`：跨机只打到目标 server 的同号卡；同号卡再 HCCS/机内转到真正 expert 卡——**远端 buffer 可能是“中转窗”而非最终 expert 窗**。

### 3.3 与业务 Tensor 的边界

- 非零拷贝：先 `Device memcpy` 业务 `x` → 通信 registered buffer，再 RDMA。
- 零拷贝方向：业务输出 **直接写在** 已注册的 combine/dispatch buffer 视图上（见 §5），RDMA 只搬该窗，不再二次拷。

---

## 4. 远端：收齐 → 整理 → 计算 → 回送

### 4.1 收齐

```text
while recv_flags from all expected src ranks not ready:
    poll / wait on Device
整理 ep_recv_counts / expert_token_nums
```

全卡信息不必回 Host：counts 与 `assist_info` 留在 Device，供 FFN tiling 与 Combine 使用。

### 4.2 整理（按专家）

- 将各 src 写入的 payload **按 local expert 重排** 为 `expand_x[A,H]`（或 `[L, ·, H]`）。
- 应用 alignment（如 512B/专家维度对齐）与 padding；无效 pad 不进有效 `expert_token_nums`。
- Dispatch 去重后的单份 token，在此 fan-out 到多个 expert 输入区。

### 4.3 计算

各 local expert 对属于自己的 token 切片跑 FFN（可与其它卡通信重叠，若拆成预/后处理算子释放 AIV）。

### 4.4 回送（Combine）

```text
expert_out → (可选本地同 token 加权) → RDMA Write 回 src 的 combine recv 窗
src: 按 assist_info / expand 映射做最终 Σ w_k * y_k → x_out[BS,H]
```

元数据闭环：

| Dispatch 产出 | Combine 消费 |
|---|---|
| `assist_info_for_combine` | 还原 token 归属与槽位 |
| `ep_recv_counts` | 作为对端 `ep_send_counts` 控制回传长度 |
| `expand_scales` | 与回传结果加权 |
| `expert_ids`（原张量） | 路由一致性校验 / 权重对齐 |

---

## 5. 宿端零拷贝分析

此处 **宿端 = Host（CPU）侧控制面 + 可能经 Host 的数据面**。零拷贝目标：数据面与路由控制面都不经过 Host 内存，且尽量少 Host↔Device 同步。

### 5.1 传统路径的宿端拷贝 / 同步点

```text
① Host 拼路由 / AllGather 收齐 expert 表     ← Host 参与元数据
② Host 算 AllToAllV split，下发通信子图       ← Host 构造 RDMA 任务
③ Device → Host staging → NIC（若无 GPU-Direct）← 数据经宿端
④ RDMA 前后 Host 同步、再下下一个算子           ← Stream 同步放大 EP
```

小包推理下，②④ 的下发与同步往往比纯带宽更伤。

### 5.2 融合算子如何逼近“宿端零拷贝”

| 层次 | 做法 | 消掉的宿端成本 |
|---|---|---|
| 控制面下沉 | 路由计数、前缀和、重排、flag 轮询全在 AIV | Host 不再算 send/recv 表，无路由 AllGather 同步 |
| RDMA 下沉 | AICPU / **AIV 直驱** 发 WQE | Host 不再构造通信子图 |
| 数据面 | NIC DMA 读写 Device 对称内存（GDR/IBGDA 类） | 载荷不进 Host DRAM |
| 元数据常驻 Device | `assist_info`、counts 留 GM | Combine 无需 Host 回灌索引 |
| 拆分预/后处理 | RDMA 传输期释放 AIV | Host 可并行下发其它 Vector 算子（控制面重叠） |

**严格意义**：Host 仍发起一次 `aclnn*` / `torch_npu` 下发，但不形成“宿端数据拷贝”；是 **宿端旁路（host-bypass）的零拷贝数据面**。

### 5.3 Device 内侧“通信 Buffer 零拷贝”（与宿端正交）

对标 DeepEP `get_next_low_latency_combine_buffer` + `zero_copy=True`：

```text
expert GEMM 输出地址 == RDMA registered combine send buffer 的 view
⇒ Combine 内核跳过 “expert_out → comm_buf” 的 Device memcpy
⇒ 只剩 RDMA + 远端加权
```

Dispatch 侧对称优化：若 `x` 已在可注册窗或 kernel 内 fused cast→write，可去掉 send staging。

分层看零拷贝：

```text
L0 宿端零拷贝：Host DRAM 不出现 token/路由大块拷贝（融合算子已做到）
L1 控制同步零开销：无 Host 侧 AllGather 路由屏障（Device notify + flag）
L2 Device 通信窗零拷贝：业务输出直接落在 RDMA 窗（需框架配合 buffer 租约）
L3 去重零冗余：同 rank 多专家不重复过 NIC（§2）
```

实现 checklist（宿端视角）：

1. 单次下发 Dispatch/Combine，禁止 Host 循环按 rank 提交小 RDMA。  
2. 所有 count / index 输出落 Device Tensor，Host 只读标量上界（如 `A`）做内存规划。  
3. `HCCL_BUFFSIZE` 预留足够对称窗，避免运行时回落 Host staging。  
4. 推理固定 BS 时预注册双缓冲，Combine 输出 in-place 写用户 `out`。  
5. 进一步抠 L2：给 FFN 提供 `combine_send_view`，声明生命周期与 EP handle 绑定。

### 5.4 残余非零拷贝点（需诚实保留）

- Host 仍拷贝极小 launch 参数（group、world size 等）。  
- 未走直驱 / 未注册的 Tensor 仍会 Device↔通信窗 memcpy（L2 未打通）。  
- hierarchy 中转在 **中间卡** 多一次机内搬移（换跨机字节，不是宿端拷贝）。  
- 动态 shape 导致 `A` 上界过大时，窗内 padding 浪费带宽但不经过 Host。

---

## 6. 端到端时序（单层 MoE）

```text
Host: launch Dispatch
Device: map expert_ids → ranks → dedup → RDMA Write remote slots
Remote: poll flags → pack expand_x by expert → FFN
Remote: local weighted reduce (optional) → RDMA Write back
Host: launch Combine (或与 FFN 流水重叠的后半段)
Src: poll combine flags → final weighted sum → x_out
```

关键延迟项：WQE 下发、跨机 RTT、flag 轮询、整理拷贝、FFN、回传。  
优化主轴：**去重减字节（§2）→ 直驱减下发（§3）→ 宿端旁路（§5 L0/L1）→ 通信窗零拷贝（§5 L2）→ hierarchy 减跨机（§0）**。

---

## 7. 小结

| 步骤 | 通信实现要点 |
|---|---|
| 1 选专家 | Device 扫描 `expert_ids`→rank/slot；产出 Combine 用 assist 元数据 |
| 2 去重 | 按目的 **rank** 唯一发送；远端 fan-out；Combine 本地先 reduce |
| 3 写远端 buffer | 对称窗 + RDMA Write + flag；AIV 直驱提高并行度 |
| 4 远端收齐整理计算回送 | flag 收齐→按专家整理→FFN→（局部加权）→RDMA 回源→最终归约 |
| 5 宿端零拷贝 | 数据与路由均不经 Host；Host 仅 launch；再向上打通 Device 通信窗零拷贝与去重 |

---

## 参考

- 昇腾社区：MoeDistributeDispatch / Combine 通算融合与分层通信（AIV+AICPU → AIV 直驱 RDMA）
- CANN：Dispatch/Combine 去重传输、调度与预/后处理拆分
- `torch_npu.npu_moe_distribute_dispatch_v2` / `combine_v2`：`comm_alg`、`assist_info_for_combine`、`HCCL_BUFFSIZE`
- DeepEP：low-latency combine `zero_copy` + registered RDMA buffer view（L2 对照）
