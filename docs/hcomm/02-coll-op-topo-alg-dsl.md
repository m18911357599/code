# 通信算子：从 RankTable 到「拓扑模型 × 算法模板 × 算子」

> 目标：通信算子实现不绑死某张 rank table JSON，而是  
> **RankTable DSL → 拓扑模型 → 算法模板（可匹配）→ 算子 + 模板参数 → 可执行 schedule**。  
> 工具：[`tools/coll_op_dsl/`](../../tools/coll_op_dsl/README.md)。

---

## 1. 问题

集合通信算子（AllReduce / ReduceScatter / AllGather / Broadcast）的正确性与性能都依赖 **rank 之间怎么连**：同 Server 直连、跨 Server CLOS、超节点分层。hcomm 里这条链已经存在，但是散落在三处：

| 层次 | hcomm 实体 | 现状 |
|------|------------|------|
| 成员描述 | rank table JSON | 字段多、V1/V2 方言混用 |
| 拓扑查询 | `RankGraph` / `NetInstance` | C++ 控制面 API，算子里直接调 |
| 算法实现 | `CollAlgFactory` + 各 `topo_match_*` | 算法与拓扑匹配、流水段、切分绑在同一份 C++ 里 |

结果是：换一张表、换一种组网，就要改算子实现。要做的是把三层 **切开**，用 DSL 把「诉求」写出来，再用模板参数把算子实例化出来。

---

## 2. 三层切分

```
┌─────────────────────────────────────────────────────────┐
│ ① RankTable DSL                                          │
│    集群成员：谁是 rank、在哪台 Server、哪层、什么地址     │
└───────────────────────────┬─────────────────────────────┘
                            │ 派生（不依赖物理 topo 文件）
                            ▼
┌─────────────────────────────────────────────────────────┐
│ ② Topology Model（RankGraph 的语言面）                    │
│    layer → instance → ranks[]                            │
│    查询：instances(L)、ranks_in(L, id)、slot 转置        │
└───────────────────────────┬─────────────────────────────┘
                            │ match + lower
                            ▼
┌─────────────────────────────────────────────────────────┐
│ ③ Algorithm template                                     │
│    ring / recursive_hd / hierarchical                    │
│    声明：match 约束 + 在哪一层用什么 pattern              │
└───────────────────────────┬─────────────────────────────┘
                            │ 绑定算子 + 模板参数
                            ▼
┌─────────────────────────────────────────────────────────┐
│ ④ Operator instance                                      │
│    AllReduce { reduce, dtype, count, use <algo> }        │
│    → schedule（逐步 send/recv/reduce）→ 模拟执行         │
└─────────────────────────────────────────────────────────┘
```

对应关系（hcomm → 本方案）：

| 诉求 | 本方案 | hcomm 对照 |
|------|--------|------------|
| 表怎么写 | RankTable DSL | `rank_table.json` / `RankTableInfo` |
| 拓扑怎么问 | Topology Model | `HcclRankGraphGetRanksByLayer` 等 |
| 算法怎么选 | `algo` + `match` | `topo_match_mesh` / Clos / 1DMesh |
| 算子怎么做 | `operator` + 模板参数 | `AllReduce` + 算法实现 + chunk/pipeline |
| 实现是否对 | schedule 模拟 | 真机 HCCL 跑数 |

---

## 3. ① RankTable DSL

仍用 [`tools/ranktable_dsl`](../../tools/ranktable_dsl/README.md) 描述 **成员**。V2 两层样例：

```
ranktable 2.0 {
  rank 0 {
    local 0
    device 0
    level 0 "az0-rack0-pod0" TOPO_FILE_DESC { addr IPV4 172.16.0.10 ports [1/0] }
    level 1 "az0" CLOS { addr IPV4 172.16.0.15 ports [0/4, 0/5] plane plane0 }
  }
  ...
}
```

这里 **不写算法**。表只回答：4 个 rank、layer0 两个 instance、layer1 一个 CLOS。

---

## 4. ② 拓扑模型（从 rank table 派生）

把 `level_list` 按 `(net_layer, net_instance_id)` 聚类，得到 RankGraph 的语言面。不读驱动 topo 文件时：

- `TOPO_FILE_DESC` / `1DMESH`：instance 内按 `rank_id` 排序，视为环/直连组
- `CLOS`：instance 内全互通，算法仍可在其上走 ring / recursive_hd
- V1 `server_list`：每个 Server → layer0 instance；`super_pod_list` → layer1 instance

导出形态：

```
topo {
  rank_count 4
  layer 0 {
    instance "az0-rack0-pod0" TOPO_FILE_DESC ranks [0, 1]
    instance "az0-rack0-pod1" TOPO_FILE_DESC ranks [2, 3]
  }
  layer 1 {
    instance "az0" CLOS ranks [0, 1, 2, 3]
  }
}
```

算子侧只用这组查询，不再解析 JSON 字段：

| 查询 | 含义 |
|------|------|
| `layers()` | 有哪些 `net_layer` |
| `instances(L)` | 该层所有 NetInstance |
| `ranks_in(L, id)` | instance 内 rank 列表（有序） |
| `slot_groups(L)` | 转置：各 instance 第 i 号位组成跨 instance 组（分层算法用） |

`slot_groups(0)` 在上面的 2×2 上得到 `[0,2]` 与 `[1,3]`：同 local 槽位、跨 Server。这就是分层 AllReduce 的 inter 组。

---

## 5. ③ 算法模板（独立于算子）

算法是 **pattern + 拓扑约束**，不是 AllReduce 的一份拷贝。三种内置模板：

| 模板 | `match` | 在拓扑上的落点 | 适合 |
|------|---------|----------------|------|
| `ring` | `ranks >= 2` | 某个 instance / 某一层的每个 instance | 任意 N，带宽最优 RS+AG |
| `recursive_hd` | 组大小为 2 的幂 | 同上 | 延迟更敏感、N=2^k |
| `hierarchical` | `layers >= 2` 且 layer0 各 instance 等宽 | intra = 模板@L0，inter = 模板@slot 组 | 两层组网（Server 内 + CLOS） |

DSL：

```
algo ring {
  match ranks >= 2
  pattern ring
}

algo recursive_hd {
  match power_of_two
  pattern recursive_hd
}

algo hierarchical {
  match layers >= 2
  intra ring on layer 0
  inter ring on layer 1
}
```

`match` 失败则该模板不能实例化（对应 hcomm 里 topo_match 返回不匹配）。  
**同一套 ring 模板**可以降 AllReduce、ReduceScatter、AllGather、Broadcast，差别只在算子怎么组合这些原语。

原语（算法真正实现的东西）：

| 原语 | 语义 |
|------|------|
| Ring ReduceScatter | N−1 步，每步向 next 发一块、从 prev 收一块并 reduce；结束后 rank i 持有块 `(i+1)%N` |
| Ring AllGather | N−1 步，把已有块沿环转一圈 |
| Ring AllReduce | RS + AG |
| Recursive doubling AllReduce | `partner = i XOR 2^k`，整缓冲交换后 reduce |
| Broadcast / Reduce | 环上转发或向 root 归约 |

分层 AllReduce 固定为 NCC 式二维：

1. 每个 L0 instance 做 **intra ReduceScatter**
2. 每个 `slot_group` 对持有切片做 **inter AllReduce**
3. 每个 L0 instance 做 **intra AllGather**

换 `intra`/`inter` 的 pattern（ring ↔ recursive_hd）不换算子名。

---

## 6. ④ 算子 + 模板参数

算子只声明 **集合语义 + 数据参数 + 选用哪份算法模板**：

```
topo from "v2_two_layer.rt"

operator AllReduce {
  reduce SUM
  dtype i32
  count 8
  use hierarchical
}
```

模板参数：

| 参数 | 作用 | 换它会变什么 |
|------|------|----------------|
| `use` | 选算法模板 | schedule 形状（步数、peer） |
| `reduce` | SUM / MAX / MIN | 归约函数，不改通信图 |
| `count` | 每 rank 元素数 | 切块大小（须能被组大小整除） |
| `dtype` | 占位，模拟器用 i32 | 真机才影响宽度 |
| `on layer K` | 平坦算法落在哪一层 | 通信域（Server 内 vs 全局） |
| `root` | Broadcast / Reduce 的根 | 数据源/汇 |

**算子独立、算法独立、参数独立**：AllReduce 可以挂 ring 或 hierarchical；ring 也可以挂 AllGather。这就是「抽象不同算法，独立出算子 + 模板参数」。

---

## 7. Lower：诉求 → schedule

`lower` 不生成 CCU 二进制，只生成逐步动作，用来描述 **算子实现诉求**：

```
phase reducescatter step 0
  rank 0 send chunk0 -> 1   recv chunk1 <- 1   reduce SUM
...
phase allgather step 0
  rank 0 send chunk1 -> 1   recv chunk0 <- 1   copy
```

这与 hcomm 数据面「先问 RankGraph 要 peer/link，再下发 Channel 传输」同构；本仓库用模拟器执行这份 schedule，检查集合语义。

---

## 8. 验证思路（脚本）

`tools/coll_op_dsl/verify.py` 跑通整条链：

1. RankTable DSL → 打印 Topology Model
2. 同一 AllReduce、两套算法（L1 ring vs hierarchical）→ **结果相同、schedule 不同**
3. `match` 拒绝：`recursive_hd` 用在 3 卡组上失败
4. ReduceScatter / AllGather / Broadcast 各自语义
5. V1 server 表同样能派生拓扑并 AllReduce

集合语义金标：

- AllReduce SUM：每 rank 输出 `sum_r input_r`
- ReduceScatter：rank 只持有全局和的一块
- AllGather：按组序拼回
- Broadcast：全体等于 root 的输入

---

## 9. 以后接到 hcomm / CCU

| 本方案产物 | 下游 |
|------------|------|
| Topology Model 查询 | 直接对应 RankGraph 控制面 |
| schedule 的 send/recv/reduce | CCU `Trans*` + MS Reduce，或 AI CPU 传输原语 |
| `chunk` / pipeline | 现有 HCCL 切分与流水段 |
| `algo.match` | `CollAlgFactory` 的 topo_match |

当前验证停在 **正确的集合语义 + 可打印的实现诉求**，不模拟链路协议。
