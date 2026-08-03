# `dispatch.combine`：RDMA fullmesh 的 Write / Read 语义

> 范围：仅 `comm_alg=fullmesh`；跨卡通信只用 **RDMA Write** 与 **RDMA Read**。  
> 不含 hierarchy / HCCS 中转、不含 Host 拼子图细节。  
> 记号：`BS` 本卡 token 数，`H` hidden，`K` topK，`E` 全局专家数，`R` EP world size，`L=E/R` 每卡本地专家数。

---

## 0. fullmesh 通信模型

每对 rank 之间有直连 QP；每卡预注册对称窗（send/recv + flag）。  
token 只发往 **目标专家所在卡**，无中间跳。

```text
Dispatch：源卡  --RDMA Write-->  专家卡 recv 窗
Combine ：专家卡 --RDMA Write-->  源卡   combine 窗
           或源卡 --RDMA Read -->  专家卡 send 窗（见 §3）
```

本卡专家：本地拷贝，不进 RDMA。

---

## 1. 选专家 → Write/Read 描述符

`expert_ids[BS,K]` → 通信描述符（不重算 gating）：

```text
e → dst_rank = e / L
    local_eid = e % L
    slot      = 远端 recv 窗内偏移（由 src_rank、token、去重序决定）
```

产出两类可下发的 RDMA 操作描述：

| 阶段 | 操作 | 本地地址 | 远端地址 |
|---|---|---|---|
| Dispatch | Write | 本卡 token payload | `dst_rank` 的 dispatch recv slot |
| Combine | Write 或 Read | 专家输出 / 源卡输出槽 | 对端 combine 窗对应 slot |

同行 K 个 `expert_ids` 不重复；可选 mask 关掉无效槽，避免空 Write。

---

## 2. 去重（减少 Write 次数）

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

## 3. Write / Read 语义（fullmesh 唯一数据面）

### 3.1 对称 Buffer

```text
rank r:
  dispatch_recv[src][slot][H]   # 被源卡 Write
  dispatch_flag[src]            # Write 完成通知（或写后原子加）
  combine_send[dst][slot][H]    # 专家结果待取/待写
  combine_recv[src][slot][H]    # 源卡收结果（Write 回传时）
  combine_flag[…]
```

### 3.2 Dispatch = RDMA Write

```text
源卡:
  1) 按去重后的 (t → r) 填本地 send 视图
  2) RDMA Write(local_token, remote=r.dispatch_recv[src][slot], len=H')
  3) 写完后通知：Write flag 或带 immediate / 单独小 Write

专家卡:
  poll dispatch_flag 收齐预期 src
  按 local_eid 整理 → expand_x → FFN
```

语义：**推模型（push）**。发送方主动把数据推进远端 recv 窗；接收方只轮询完成位，不发起数据面 Read。

### 3.3 Combine = RDMA Write（回写）或 RDMA Read（拉取）

**Write 回传（常用 push）：**

```text
专家卡:  (可选本地加权) → RDMA Write → 源卡.combine_recv[expert_rank][slot]
源卡:    poll combine_flag → Σ w·y → x_out[BS,H]
```

**Read 拉取（pull，语义等价、发起端在源卡）：**

```text
专家卡:  把 y 放进本卡 combine_send[src][slot]，置 ready flag
源卡:    poll 对端 ready（或本地侧约定）→ RDMA Read(remote=combine_send, local=combine_recv)
         → Σ w·y → x_out
```

| | RDMA Write（Combine） | RDMA Read（Combine） |
|---|---|---|
| 数据面发起方 | 专家卡 | 源卡 |
| 远端窗角色 | 源卡 `combine_recv` 被写 | 专家卡 `combine_send` 被读 |
| 完成通知 | 写 flag / 写后通知 | Read 完成 CQ + 对端 ready |
| fullmesh 约束 | 直连到源卡，无中转 | 直连读专家卡，无中转 |

Dispatch 阶段固定用 **Write**（专家卡被动收）；Combine 在 Write/Read 二选一，fullmesh 下都是点对点、无中间 hop。

### 3.4 与零拷贝的交界

- payload 必须落在 **已注册** 的对称窗（或窗上的 view）。  
- FFN 若直接写 `combine_send`，则 Combine 的 Write/Read **不再** Device staging 拷贝。  
- Host 不参与数据面：无 Host DRAM 中转，只有 launch。

---

## 4. 远端收齐 → 整理 → 计算 → 回送（仅 Write/Read）

```text
① 收齐：poll 各 src 的 dispatch_flag（数据已由对端 Write 进本卡窗）
② 整理：dispatch_recv → 按 local_eid 展开 expand_x（去重 token 本地 fan-out）
③ 计算：FFN(expand_x 切片)
④ 回送：
     Write 路径: Write(y) → 源卡 combine_recv + flag
     Read  路径: 发布 combine_send + ready，等源卡 Read
⑤ 源卡: 对各专家回程槽做加权归约 → x_out
```

元数据（counts、slot 映射）留在 Device，只服务于算 Write/Read 的地址与长度，不经 Host。

---

## 5. 端到端（单层）

```text
Dispatch:  map → dedup → RDMA Write → (专家卡 poll)
Expert:    pack → FFN
Combine:   RDMA Write 回源  或  源卡 RDMA Read
Src:       reduce → x_out
```

---

## 6. 小结

| 步骤 | fullmesh RDMA 语义 |
|---|---|
| 选专家 | `expert_ids` → `(dst_rank, slot)`，生成 Write/Read WQE 地址 |
| 去重 | 每 `(token, dst_rank)` 至多一次跨卡 Write；远端本地展开 |
| 写远端 buffer | Dispatch：**RDMA Write** 进专家卡 `dispatch_recv` |
| 收齐整理计算回送 | poll flag → 整理 → FFN → Combine：**Write 回传** 或 **Read 拉取** |
| 宿端零拷贝 | 数据面只有 NIC↔注册窗；Host 不下数据、不中转 |
