# `dispatch.combine`：RDMA fullmesh Write / Read

> 范围：仅 `comm_alg=fullmesh`；跨卡只用 **RDMA Write / Read**。不含 hierarchy。  
> 结构：**Host 预置 → 源端 → 宿端 → 规格原因**。  
> 术语：
>
> | 名称 | 含义 |
> |---|---|
> | **Host 预置** | 进程侧按公式备好对称窗与 `expand_x` 上界（只定容量，不跑路由） |
> | **源端** | 持有自然序 token 的 Device rank（Gate / Dispatch 发出 / Combine 归约） |
> | **宿端** | 持有目标专家的 Device rank（收 Write / 整理 / FFN / 回传） |

记号：`BS`/`max_bs`，`H`，`K`，`E`，`R=ep_world_size`，`L=local_expert_num`，`A`=本卡 recv 行上界。

```text
Align32(x)     = ((x + 31)  / 32)  * 32
Align512(x)    = ((x + 511) / 512) * 512
480Align512(x) = ((x + 479) / 480) * 512
```

总览：

```text
Host 预置:  算 A、配 HCCL_BUFFSIZE、注册对称窗
源端:       Matmul → TopK → map/dedup/pack → RDMA Write
宿端:       poll → 按专家整理 → FFN → Write 回源 / 待 Read
源端:       收齐回程 → Σ w·y → x_out
```

---

## 1. Host 预置

只做数据面容量与窗布局；不扫 `expert_ids`、不拼 WQE。

### 1.1 逻辑窗

```text
每卡对称通信窗
├── dispatch_recv[src][slot][H]   # 宿端被 Write
├── dispatch_flag[src]
├── combine_send[dst][slot][H]    # 宿端放专家输出（Read 源 或 Write 源）
├── combine_recv[src][slot][H]    # 源端收回传（Write 回传时）
└── combine_flag[…]
```

```text
RDMA Write(Dispatch) → 对端 dispatch_recv
RDMA Write(Combine)  → 对端 combine_recv
RDMA Read (Combine)  → 读宿端 combine_send
```

### 1.2 `expand_x` 行上界 `A`

```text
共享专家卡:  A = BS * shared_expert_num / shared_expert_rank_num

MoE 专家卡:
  global_bs == 0:  A >= BS * R * min(L, K)
  global_bs != 0:  A >= global_bs * min(L, K)
```

`expand_x`：`(A, H)`。

### 1.3 通信窗公式（`HCCL_BUFFSIZE` / `hccl_buffer_size`，MB）

**Atlas A2 · fullmesh**

```text
>= 2 * ( BS * R * min(L, K) * H * sizeof(uint16) + 2MB )
```

**Atlas A3 · fullmesh_v1**（`""` 默认）

```text
>= 2 * (
      L * max_bs * R * Align512( Align32(2*H) + 64 )
    + (K + shared_expert_num) * max_bs * Align512(2*H)
   )
```

**Atlas A3 · fullmesh_v2**（需 `tp_world_size=1`）

```text
>= 2 * (
      L * max_bs * R * 480Align512( Align32(2*H) + 64 )
    + (K + shared_expert_num) * max_bs * Align512(2*H)
   )
```

| 公式块 | 窗 |
|---|---|
| `L·max_bs·R·Align…(2H+64)` / A2 的 `BS·R·min(L,K)·H·2` | Dispatch 收窗 |
| `(K+shared)·max_bs·Align512(2H)` / A2 括号内同源量级 | Combine 回程窗 |
| 外层 `2` | 双相或 Dispatch+Combine |

各 EP rank 入参一致；默认未配时常按 200MB，过小则运行期失败。理由见 §4。

---

## 2. 源端

持有 `x[BS,H]` 的 rank：本地出路由，再 **RDMA Write** 到宿端；回程后加权归约。

### 2.1 Matmul → 选专家 → token 转换

```text
x[BS,H]
  │ ① Gate Matmul:  logits = x @ W_gateᵀ → [BS,E]     （布局仍为 token 序）
  │ ② TopK:         expert_ids[BS,K], expert_scales[BS,K]
  │                 e → (dst_rank, local_eid)=(e/L, e%L)
  │ ③ 转换:         map → dedup → gather/pack
  ▼
send_buf[n_write,H] + meta(dst_rank, slot, src_tok, scales…)
  │ ④ RDMA Write (fullmesh)
  ▼
宿端.dispatch_recv[src][slot]
```

```text
                    W_gate[E,H]
token0..BS-1 ─x──► Matmul ──► logits[BS,E] ──TopK──► ids/scales
                                              │
                    x[BS,H] ──────────────────┤
                                              ▼
                                         map / dedup / pack
                                              │
                                              ▼
                                         RDMA Write × n_write
```

小例（`R=4,L=2,K=2`）：

```text
token  expert_ids  dst 去重前   dst 去重后
 t0    [0,1]       [0,0]       [0]      → Write ×1
 t1    [2,5]       [1,2]       [1,2]    → Write ×2
 t2    [3,4]       [1,2]       [1,2]    → Write ×2
```

| 步 | 输入 → 输出 |
|---|---|
| map | `expert_ids` → `(dst_rank, local_eid, slot)` |
| dedup | 同 `(token,dst_rank)` 合并 → Write 次数 ≤K |
| pack | `x[token]` → 注册窗内 `send_buf` |

要点：Matmul 不改布局；选专家只产索引/权重；**转换才搬激活**。  
Write 键：`(src_rank, token, dst_rank)`，不是 per-expert。

### 2.2 Dispatch：RDMA Write

```text
for each deduped (t → r):
  RDMA Write(send_buf[t], remote=r.dispatch_recv[src][slot])
  通知: flag / immediate
# 本卡专家: 本地拷贝，不进 RDMA
```

推模型：源端主动 Write；宿端只 poll，不发数据面 Read。

### 2.3 Combine 归约（回程落源端）

宿端回传完成后（§3.3）：

```text
poll combine_flag  (Write 回传)
  或  RDMA Read(宿端.combine_send) 后本地完成
Σ_k w_k * y_k  →  x_out[BS,H]
```

| Combine 模式 | 源端角色 |
|---|---|
| Write 回传 | 被动收 `combine_recv`，再 reduce |
| Read 拉取 | 主动 Read 宿端 `combine_send`，再 reduce |

---

## 3. 宿端

持有目标专家的 rank：收齐源端 Write → 按专家整理 → 计算 → 回送。

### 3.1 收齐

```text
while 未收齐预期 src 的 dispatch_flag:
  poll
# 载荷已在本卡 dispatch_recv（由源端 Write 填入）
```

### 3.2 整理 + 计算

```text
dispatch_recv
  → 按 local_eid 展开 / fan-out（去重 token 在本卡复制到多专家槽）
  → expand_x[A,H]（有效行由 expert_token_nums 给出）
  → 各 local expert FFN(expand_x 切片)
```

### 3.3 回送（Write 或待 Read）

**Write 回传（push）**

```text
(可选) 同 token 多本地专家输出先按 scales 局部加权
RDMA Write(y) → 源端.combine_recv[本卡][slot] + flag
```

**Read 拉取（pull，发起在源端）**

```text
把 y 写入本卡 combine_send[src][slot]，置 ready
# 源端随后 RDMA Read 该窗
```

```text
源端 ──Write──► 宿端.dispatch_recv ──整理/FFN──► combine_send/recv 路径
                                              │
                 ┌── Write 回传 ───────────────┘
                 └── 或源端 Read(combine_send)
```

---

## 4. 规格原因

说明 §1 公式为何如此，而非调参经验堆砌。

### 4.1 为何 `A ≥ BS·R·min(L,K)`（`global_bs=0`）

- 最坏：每个 src 的每个 token，最多命中本卡 `min(L,K)` 个本地专家槽（TopK 与本地专家数取小）。  
- `R` 个 src 都打满 → 行数上界 `BS·R·min(L,K)`。  
- `global_bs≠0` 时用全局 token 上界替换 `BS·R`，避免各卡 BS 不齐时低估。  
- 共享专家卡流量模型不同，故单独 `A=BS·shared/shared_ranks`。

### 4.2 为何出现 `min(L,K)` 而非 `K` 或 `L`

- 单 token 对本卡：选中专家数 ≤K，且本卡只有 L 个专家 → ≤`min(L,K)`。  
- 用 `K` 会在 `L<K` 时高估；用 `L` 会在 `K<L` 时高估。  
- A2 通信窗与 `A` 同用该因子，使 **业务 expand 与 RDMA 收窗同一悲观界**。

### 4.3 为何 A2 用 `sizeof(uint16)` 且 `+2MB`，外层再 `×2`

- fp16/bf16 载荷按 2B 计；公式写成 `uint16` 即此。  
- `+2MB`：flag、小头、通信库对齐，避免纯 payload 估满后无余量。  
- 外层 `×2`：Dispatch 与 Combine（或 ping-pong）各一份，防止同窗覆写。

### 4.4 为何 A3 要 `Align32(2H)+64` 再 `Align512` / `480Align512`

```text
token 记录 ≈ Align32(2·H)     # 2B·H 载荷，32B 对齐
           + 64               # per-token 头（索引/scale/flag 槽位）
再 Align512 / 480Align512     # 匹配 RDMA/HCCL 块与多平面搬运粒度
```

- v1 用 `Align512`；v2 用 `480Align512` 适配另一套平面拼块，减小内部碎片（约束更严，如 K≤12、`tp=1`）。  
- Combine 项 `(K+shared)·max_bs·Align512(2H)`：回程按「每 token 最多 K 路 + 共享专家」计，与源端归约扇入一致；局部 reduce 只减实际流量，**预置仍按上界**。

### 4.5 为何预置按上界、运行按去重

| | 预置（Host） | 运行（源/宿端） |
|---|---|---|
| 目标 | 永不因窗小失败 | 少 Write、少字节 |
| 计数 | `min(L,K)` 满扇出 | 按 `(token,dst_rank)` 去重 |
| 结果 | `A` 与 `HCCL_BUFFSIZE` 偏大但稳定 | `n_write ≤ BS·K`，常远小于上界 |

去重不能缩小预置公式：最坏路由仍可能接近满上界（专家均匀打到各卡且少碰撞）。

### 4.6 为何 Dispatch 只用 Write、Combine 允许 Read

- Dispatch：宿端在收齐前无有效数据可被 Read；Write 推送与 flag 轮询最简单。  
- Combine：专家输出已在宿端，Write 回源或源端 Read 语义等价；Read 便于源端调度拉取节奏，Write 便于宿端算完即推。  
- 两种 Combine 复用同一量级回程窗（§1.3），故规格不因选 Write/Read 再翻倍，但实现须保证 send/recv 生命周期不双开满额。

---

## 5. 对照小结

| 段落 | 做什么 |
|---|---|
| **Host 预置** | 定 `A`、配 fullmesh 窗公式、注册 `dispatch_*` / `combine_*` |
| **源端** | Matmul → TopK → map/dedup/pack → **Write**；回程 reduce → `x_out` |
| **宿端** | poll 收齐 → 按专家整理 → FFN → **Write 回传** 或 **待 Read** |
| **规格原因** | `min(L,K)`、Align、`×2`、上界预置 vs 去重运行 |
