# C++26 SIMD（tile_len=128）× GPU-like：Elementwise / Reduce / Broadcast 分类、尾块处理与性能发挥

> 目标：固定逻辑 SIMD 宽度 `TILE = 128`，以 GPU-like 的 tile + mask 模型覆盖 PyTorch/ATen 主体算子；强调 **全量动态 shape**、**尾块处理**、**硬件性能发挥度量** 与 **软件表达建议**。需专用处理的类别单独归入第 6 章。

---

## 0. 结论摘要

| 维度 | 结论 |
|---|---|
| 术语 | 统一使用 **elementwise**（逐元素）；ATen 文档中的 `pointwise` tag 视为同义别名 |
| 通用路径 | **Elementwise / Broadcast / Reduce / Layout-copy** 走统一 tile 循环 + 尾块策略，可全动态 shape |
| 尾块 | 不以 `numel % 128 == 0` 为契约；整块用满宽向量，尾块用 **masked / 拆分 / 标量回退** 三选一（见 §4） |
| 专用路径 | GEMM、Conv、Attention、全局 Sort/TopK、data-dependent 输出等 **集中到第 6 章**，不塞进通用 elementwise 引擎 |
| 性能发挥 | 关键看 **向量利用率、尾块占比、带宽效率、谓词开销、广播/归约轴是否可向量化**；软件表达应让整块路径零分支、尾块路径可预测 |
| 易用性 | Elementwise / 简单 Broadcast+Reduce：**高**；复杂 Broadcast+多维 Reduce：**中高**；第 6 章专用类：**模板/库级表达** |

**一句话**：把动态长度拆成「满 128 的整块 + 不足 128 的尾块」；整块追求峰值吞吐，尾块保证正确且可控开销。

---

## 1. 基线：固定 TILE=128 的 GPU-like 抽象

### 1.1 逻辑 tile（非物理寄存器宽度）

```cpp
inline constexpr std::size_t TILE = 128;  // 128 个元素 lane（与 dtype 无关的逻辑宽度）
template<class T>
using tile_t = std::datapar::simd<T, /* fixed_size<TILE> 或后端等价 */>;
template<class T>
using mask_t = typename tile_t<T>::mask_type;  // 或 simd_mask 等价物
```

| GPU 概念 | 本模型 |
|---|---|
| warp | 一个 `tile_t`（128 lanes） |
| predicated exec | `mask_t` / masked load-store |
| grid-stride | 多个满宽 tile 迭代；最后一截为尾块 |

物理后端（AVX/NEON/SVE/NPU）可将 128 再切为多条物理向量指令；对软件表达而言 **只暴露 TILE=128**。

### 1.2 全量动态 Shape（通用路径契约）

1. `rank / sizes / strides / numel` 仅运行时可知  
2. 不要求编译期固定 `[B,H,W]`，也不要求 `numel` 对齐到 128  
3. Broadcast 与 Reduce 的轴集合运行时给定  
4. 专用类（第 6 章）可另定契约，但对外尺寸仍动态  

---

## 2. PyTorch 算子 Pattern 分类（通用路径）

双轴：**Shape 行为（A）** × **计算结构（B）**。ATen `tags.yaml` 中的 `pointwise` 在本文一律写作 **elementwise**。

### 2.1 Shape 行为（A）

| ID | Pattern | 代表 | 动态 Shape 要点 |
|---|---|---|---|
| A1 | **Elementwise**（原 pointwise / TensorIterator 主体） | `add/mul/relu/where` | 输出形状 = broadcast(inputs) |
| A2 | **Reduction** | `sum/mean/amax` | `dim` / `keepdim` 运行时 |
| A3 | **N-D / Batched** | `index_add`、带 batch 的 loss | 任意前缀 batch 或任意 rank |
| A4 | **View / Identity / Flatten-1D** | `view/clone/take` | 元数据或按 1D `numel` |
| A5 | **Variadic layout** | `cat/stack` | 段长动态 |
| A6 | **Composite** | `kl_div` | 分解到 A1/A2，不直映射指令 |
| A7 | **Factory** | `arange/empty` | 按运行时 size 填充 |

> Fixed-rank 重计算（conv/mm）、data-dependent 输出（unique/nonzero）、稀疏等 → **第 6 章**。

### 2.2 计算结构（B）— 通用

| ID | Pattern | 代表 | 说明 |
|---|---|---|---|
| B1 | **Elementwise map** | unary/binary/ternary | 无广播或已对齐 |
| B2 | **Broadcast elementwise** | 见 §3.1 多场景 | 运行时算每 lane 源地址 |
| B3 | **Reduce** | 见 §3.2 多场景 | tile 内归约 + 跨 tile 合并 |
| B4 | **Compare-select** | `where/clamp` | mask + blend（可并入 B1） |
| B5 | **Layout copy** | `copy/contiguous` 段 | 连续或可合并跨步 |

Scan / Gather-Scatter / Compact 等偏专用或半专用，主述见第 6 章；通用引擎仅在需要时薄封装。

### 2.3 覆盖关系（通用）

```text
Composite ──decompose──► Elementwise / Broadcast / Reduce / Layout
                              │
                              ▼
              统一 TensorDesc + 整块循环 + 尾块策略（§4）
```

| 组合引擎 | 估计覆盖 | 动态 shape |
|---|---|---|
| Elementwise + Broadcast + Reduce + Layout | ~55–70% 变体 | 全动态 |
| + 第 6 章专用模板 | 性能关键长尾 | 对外动态 |

---

## 3. Broadcast 与 Reduce 场景（展开）

### 3.1 Broadcast 场景

约定：输出逻辑形状为 `out_sizes`；每个输入通过 `expand` 规则对齐。实现时对每个输入维护 `stride_eff[d] = (sizes[d]==1 ? 0 : strides[d])`。

| 场景 ID | 名称 | 形状例 | 访问特征 | 尾块关系 | 软件表达建议 |
|---|---|---|---|---|---|
| BC0 | **无广播 / 已对齐** | `[N]+[N]→[N]` | 连续 unit-stride | 仅 `numel` 尾块 | `elementwise(out,a,b,op)` |
| BC1 | **标量广播** | `[]+[N]` / `[1]+[N]` | 一端 splat 到 tile | 随 `N` 尾块 | `broadcast_scalar` 或自动识别 |
| BC2 | **末维对齐广播** | `[B,1]+[B,K]` | 内维连续，外维扩 | 按行/`numel` 切尾 | collapse 后当 BC0/BC1 |
| BC3 | **中间维广播** | `[B,1,K]+[B,H,K]` | 非单调跨步 | 每行/每 tile 独立尾块 | `BroadcastPlan` + 运行时偏移 |
| BC4 | **多输入异形广播** | 三输入各不同 | 每输入一套 `stride_eff` | 以 **输出 numel** 为准切块 | TensorIterator 式输出主导 |
| BC5 | **高维可合并广播** | 多维但可 `collapse` | 合并后降为 BC0–2 | 合并后一次尾块 | **优先 runtime dim collapse** |
| BC6 | **非连续 + 广播** | broadcast 且 stride≠1 | 跨步 load 或 gather | 尾块 mask 必须配合跨步 | 能 `contiguous` 则拷贝；热路径保留跨步 |
| BC7 | **行广播 / 列广播** | `[M,1]+[1,N]→[M,N]` | 外积式扩展 | 按输出行主序切 128 | 专用 `Broadcast2D` 短路径（仍通用引擎内） |

**动态 shape 要点**：广播维集合运行时变化时，只更新 `stride_eff` 与 `collapse` 结果，不换 kernel 族。

### 3.2 Reduce 场景

| 场景 ID | 名称 | 形状例 | 归约结构 | 尾块落点 | 软件表达建议 |
|---|---|---|---|---|---|
| RD0 | **全维归约** | `[…]→[]` | 全局一值 | **输入**侧按 `numel` 切块；无效 lane 填中性元 | `reduce_all(op, neutral)` |
| RD1 | **最内维归约** | `[B,K]→[B]`，`dim=-1` | 每行独立 | **每行长度 K** 各自有尾块 | `reduce_inner`（向量友好） |
| RD2 | **最外维归约** | `[B,K]→[K]`，`dim=0` | 沿外维累加到长为 K 的缓冲 | 输出/`K` 维切块；外维循环可无 mask | `reduce_outer` |
| RD3 | **中间维归约** | `[B,H,K]→[B,K]`，`dim=1` | 重排为 outer×reduce×inner | reduce 长度上的尾块 | 运行时拆 `outer/reduce/inner` |
| RD4 | **多维同时归约** | `dim=(1,2)` | 合并 reduce 轴 | 合并后同 RD0/RD1 | **先合并轴再归约** |
| RD5 | **keepdim** | 同上但保留 1 | 仅元数据差 | 同对应 RD* | 输出 view 包一层 |
| RD6 | **分段 / 部分归约** | 每段长动态 | 段内 RD0/RD1 | **每段各自尾块** | 传入 `segment_offset/length` |
| RD7 | **Arg-reduce** | `argmax` | (val,idx) 对 | 同 RD1/RD0；无效 lane 禁用 idx | `reduce_argmax` |
| RD8 | **两阶段统计** | `softmax`/`layer_norm` 的 max/sum | 多遍 RD1 + elementwise | 每遍独立尾块 | 分解为 Reduce 遍 + Elementwise 遍（完整算子见 §6） |
| RD9 | **布尔 / 比特归约** | `any/all` | and/or 归约 | 中性元 `true/false` | 同 RD0/RD1，换 op/neutral |

**向量化友好序**：`RD1 ≈ RD4(合并后内维) > RD0 > RD2 > RD3`。软件侧应 **自动把可合并轴变成内维归约**。

### 3.3 Broadcast × Reduce 组合（常见）

| 组合 | 例 | 处理顺序 | 尾块 |
|---|---|---|---|
| 先 Broadcast 再 Elementwise | `x + bias` | BC* → B1 | 输出主导 |
| 先 Reduce 再 Broadcast | `x - x.mean(...)` | RD* → BC1/BC2 | 两阶段各管各的尾块 |
| Broadcast 输入上的 Reduce | `sum(a + b)` 且 a、b 异形 | 融合则按输出扩展域归约；否则物化 | 以扩展后线性域切块 |

---

## 4. 尾块处理（重点）

动态 shape 下，任意长度 `L`（`numel`、行长 `K`、段长等）都拆成：

```text
n_full = L / TILE          // 整块个数
n_tail = L % TILE          // 尾块有效元素数，∈ [0, TILE)
```

整块：`mask = all_true`，走满宽向量指令。  
尾块：`0 < n_tail < TILE`，必须显式策略，禁止读/写越界。

### 4.1 三种尾块策略

| 策略 | 做法 | 正确性 | 性能特征 | 适用 |
|---|---|---|---|---|
| **T-Mask（推荐默认）** | `mask = (lane_id < n_tail)`；masked load/compute/store | 强 | 一条向量路径；依赖硬件谓词效率 | 通用 elementwise / 多数 reduce |
| **T-Split** | 尾块再切成物理 VL 的满块 + 更小尾块；或整块循环后单独标量/窄向量 epilogue | 强 | 整块峰值不受损；epilogue 指令开销固定 | 谓词弱或代价高的后端 |
| **T-Pad（受限）** | 分配/临时缓冲 pad 到 128，无效 lane 填中性元，最后写回有效前缀 | 需控制写回范围 | 计算满宽，但多拷贝/多带宽 | 小尾块且有廉价 scratch；**不可**作为对外 API 契约 |

> API **不得**要求调用方保证 `L % 128 == 0`。T-Pad 只能是实现内部优化。

### 4.2 尾块在各 Pattern 中的落点

| Pattern | 切块长度 `L` | 整块 | 尾块要点 |
|---|---|---|---|
| Elementwise / BC0–BC4 | 输出 `numel` | 满宽 map | T-Mask store，防止写穿 |
| BC7 行主序 | 每行 `N` 或全局 `M*N` | 同行连续优先 | 行长不足 128：行尾 T-Mask；跨行拼 tile 需慎用（边界行） |
| RD0 全维 | 输入 `numel` | 累加到寄存器/共享 | 无效 lane ← **中性元**（0 / +inf / …） |
| RD1 内维 | 每行 `K` | 行内满块 | **每个 outer 一条尾块**；短行（`K<128`）整行即尾块 |
| RD2 外维 | 输出向量长 | 沿外维迭代可无 mask | 输出侧若向量写，仍按输出长切尾 |
| RD3/RD4 | `reduce_len` 或合并后长度 | 同 RD0/1 | 先合并轴，减少「每段都是短尾块」 |
| Layout copy | 字节或元素长度 | 连续 DMA/向量搬 | 尾块 T-Mask 或窄拷贝 |

### 4.3 尾块控制流（软件侧推荐形态）

**形态 A — 统一 mask（易用优先）**

```cpp
for (int64_t base = 0; base < L; base += TILE) {
  const int32_t valid = (int32_t)min<int64_t>(TILE, L - base);
  mask_t m = lane_lt(valid);          // lane_id < valid
  auto x = load(in + base, m);
  store(out + base, m, map(x));
}
```

整块时 `valid==TILE`，后端应常量折叠掉谓词（见 §5）。

**形态 B — 整块 / 尾块分流（性能优先，推荐热路径）**

```cpp
int64_t base = 0;
for (; base + TILE <= L; base += TILE) {   // 无 mask 快路径
  store(out + base, map(load_full(in + base)));
}
if (base < L) {                            // 至多一次尾块
  mask_t m = lane_lt((int32_t)(L - base));
  store(out + base, m, map(load(in + base, m)));
}
```

**形态 C — Reduce 中性元**

```cpp
acc = neutral;
for (full tiles) acc = combine(acc, reduce_tile(load_full(...)));
if (tail) acc = combine(acc, reduce_tile(where(m, load(...), neutral)));
```

### 4.4 短尾块与超短问题

| 情况 | 现象 | 建议 |
|---|---|---|
| `L ≫ 128` 且均匀 | 尾块占比 → 0 | 形态 B，整块打满 |
| 大量 `K < 128` 的短行（RD1） | **每次都是尾块**，向量利用率低 | 转置/打包多行拼 tile，或换 RD 算法；软件暴露 `reduce_inner_packed` |
| `L` 分布动态（训练动态 batch） | 尾块占比波动 | 度量 `tail_ratio`（§5）；必要时 pad batch（实现内） |

---

## 5. 硬件性能发挥：度量与软件表达

### 5.1 度量指标

| 指标 | 定义 | 健康方向 | 主要影响因素 |
|---|---|---|---|
| **向量利用率 η_vec** | 有效元素运算 / 发出的 lane 槽位 | → 1 | 尾块策略、短行、无效谓词 lane |
| **尾块占比 r_tail** | 尾块处理元素 / 总元素 ≈ `(L%TILE)/L`（多段则按段平均） | → 0（长轴） | 动态 shape、是否多短段 |
| **满宽指令比 ρ_full** | 无谓词满宽向量指令 / 全部向量指令 | 高 | 形态 B 分流、collapse |
| **谓词开销因子 α_pred** | 同计算在 masked vs full 下的耗时比 | → 1 | 硬件 predication 质量 |
| **带宽效率 η_bw** | 有效业务字节 / 实际搬运字节 | → 1 | 广播重复读、pad、非连续 |
| **归约并行效率 η_red** | 相对峰值归约吞吐 | 高 | 内维 vs 外维、跨 tile 原子 |
| **占用 / 延迟隐藏** | pipeline busy | 高 | 整块循环展开、多缓冲 |

实用近似（单次 elementwise 连续）：

```text
η_vec ≈ 1 - r_tail * (1 - n_tail/TILE)/ (L/TILE 相关项)
      ≈ L / (ceil(L/TILE) * TILE)
```

即：**ceil 对齐浪费的 lane 比例**。`L=129` → η_vec≈129/256≈0.50；`L=12800` → ≈0.999。

### 5.2 场景对性能发挥的影响（定性）

| 场景 | η_vec | η_bw | 说明 |
|---|---|---|---|
| BC0 长尾 `L≫128` | 高 | 高 | 峰值主战场 |
| BC1 标量广播 | 高 | 中高 | splat，少带宽 |
| BC3/BC6 跨步广播 | 中高 | **低** | 易变成 gather |
| BC5 先 collapse | **升高** | 升高 | **软件必做** |
| RD1 长内维 | 高 | 高 | 最佳 reduce |
| RD1 大量短行 | **很低** | 中 | 需打包/转置 |
| RD2 外维 | 中 | 中 | 写冲突/原子可能降 η_red |
| RD3 中间维未合并 | 低–中 | 低 | 先合并到 RD1/RD4 |
| 统一 T-Mask 不分流 | 名义满宽但 α_pred>1 | — | 整块也被拖慢 |
| 形态 B 分流 | 整块 α_pred≈1 | — | **推荐热路径表达** |

### 5.3 软件表达建议（让硬件打满）

#### （1）分层 API：用户不见 TILE，热路径可分流

| 层级 | 表达 | 谁处理尾块 |
|---|---|---|
| **L0 声明式** | `elementwise(out, a, b, op)` / `reduce(out, in, dims, op)` | 框架 |
| **L1 计划式** | `BroadcastPlan` / `ReducePlan`（collapse 后的 inner/outer） | 框架按 plan 选 BC*/RD* |
| **L2 分流行** | 框架生成「整块循环 + 单次尾块」IR | 显式形态 B |

用户默认 L0；编译/运行时 lowering 到 L1/L2。

#### （2）表达原则

1. **输出主导切块**：elementwise/broadcast 以输出线性域为 `L`，避免多输入各切各的。  
2. **先 collapse，再发向量**：动态 shape 下每次 launch 做 dim 合并，把 BC3→BC0/2、RD3→RD1。  
3. **整块无谓词**：`valid==TILE` 走 `load_full/store_full`；禁止在热循环里每次算 `lane_lt`。  
4. **尾块至多一次（单段）**：形态 B；多段 reduce 则「每段至多一次尾块」。  
5. **Reduce 中性元显式**：`neutral_of<Op,T>()`，尾块 `where(m,x,neutral)`，避免脏 lane 污染。  
6. **短内维申报**：当 `inner < TILE` 占主导，plan 标记 `ShortInner`，换打包策略，而不是沉默低 η_vec。  
7. **能力探测**：无硬件 mask store 则 T-Split；有 MMA 则进第 6 章，不走 elementwise 仿真。

#### （3）推荐 IR 片段（示意）

```text
ElementwiseOp {
  plan: collapse(out, inputs) -> {L, loads[]}
  body:  FullTiles(L) { map(load_full*) -> store_full }
         Tail(L)      { m=lane_lt(n_tail); map(load_m*) -> store_m }
}

ReduceOp {
  plan: dims -> {outer, reduce_len, inner}  // 动态
  body:  prefer inner-reduce (RD1);
         each segment: FullTiles(reduce_len) + Tail with neutral
}
```

#### （4）与易用性的折中

| 表达选择 | 易用 | 性能发挥 |
|---|---|---|
| 全程统一 T-Mask 循环 | 最高 | 整块受 α_pred 拖累 |
| L0 API + 内部形态 B | 高 | **佳平衡** |
| 要求用户手写整块/尾块 | 低 | 高但不稳 |
| 强制输入 pad 到 128 | 假易用 | 带宽与生态差，**禁止作契约** |

---

## 6. 需专用处理的类别（专章）

以下 **不** 塞进通用 elementwise/broadcast/reduce 引擎；对外仍支持动态尺寸，对内独立模板或库。

### 6.1 清单与理由

| 类别 | 代表算子 | 为何专用 | 与 TILE=128 的关系 | 动态 shape | 建议表达 |
|---|---|---|---|---|---|
| **GEMM / 批量 GEMM** | `mm/bmm/addmm` | 需 MMA/分块；非 lane map | 块尺寸取 128 的因子 | M/N/K 动态 | `gemm_tiled` 库；禁止 simd 三层循环 |
| **Convolution** | `conv1d/2d/3d` | 滑窗/im2col+GEMM | 同左 | NCHW 动态 | `conv` 模板 / winograd 等 |
| **Attention 族** | SDPA / FlashAttn | 块状 softmax+GEMM 融合 | 序列维按 128 分块 | `seq` 动态 | 融合 kernel，不拆成通用 RD+GEMM naive |
| **Normalize 融合** | `softmax/layer_norm/rms_norm` | 多遍 + 数值稳定 | 内维按 128 切；短内维专用 | 特征维动态 | `normalize_inner` 专用；可复用 RD1 尾块策略 |
| **全局 Sort / TopK** | `sort/topk` | 跨 tile 归并网络 | 局部 128 位序网络 | `n` 动态 | 算法库 |
| **Scan 全局** | `cumsum` 任意维 | 跨 tile 前缀依赖 | tile scan + 前缀传递 | 轴长动态 | `scan` 库；尾块用 mask+中性元 |
| **Gather / Scatter / Index** | `gather/index_add/embedding` | 间接寻址、冲突 | 每 tile 128 个索引 | 索引长动态 | `gather/scatter` 引擎；尾块 T-Mask |
| **Data-dependent 输出** | `nonzero/unique/masked_select` | 输出长度依赖数据 | compact 按 mask | 二阶段 | `count_then_write`；ballot/scan 扩展 |
| **RNG 质量路径** | `randn/dropout` 大批量 | 计数器 RNG 对齐 | 128 lane 独立流 | shape 动态 | `rng_philox_tile` |
| **Sparse / 混合布局** | sparse mm 等 | 索引结构特殊 | 视格式而定 | 动态nnz | 独立后端 |
| **Fixed-rank 尚不可分解者** | 部分 legacy op | 维度语义绑死 | 仅 batch 动态 | 有限动态 | 逐 op 模板，能分解则降级到 §2 |

### 6.2 专用类的尾块与性能

| 类别 | 尾块策略 | 性能要点 |
|---|---|---|
| GEMM/Conv/Attn | 边缘块（M/N/K 尾）独立 epilogue | 整块 MMA 满吞吐；边缘块允许较低 η_vec |
| Softmax/Norm | 内维 T-Mask + 中性元 | 短特征维时 η_vec 差 → 专用打包 |
| Sort/Scan | 末 tile mask | 延迟绑定，带宽次之 |
| Gather/Scatter | 索引尾块 mask | 随机访问主导，η_bw 先天低 |
| DynOut compact | 末 tile mask + 全局 sync | 两 kernel；测的是 scan/atomic 而非向量峰值 |

### 6.3 专用类软件表达（建议）

```text
// 通用：声明式，框架负责整块/尾块
elementwise(out, x, y, [](auto a, auto b){ return a+b; });
reduce(out, x, dims={1}, op=sum);

// 专用：显式领域 API，不伪装成 elementwise
gemm(C, A, B, {.tile = 128});
softmax_inner(out, in, /*dim=*/-1);
nonzero_async(count, out, pred);   // 二阶段
```

原则：**专用类暴露领域名**；内部可复用 §4 尾块原语，但 **调度与寄存器/共享内存规划独立**。

---

## 7. 指令分类（面向通用路径 + 专用衔接）

| 类 | 名称 | 通用路径用途 | 尾块相关 |
|---|---|---|---|
| **I1** | 算术/逻辑/cast | elementwise | 仅作用于有效 lane |
| **I2** | 谓词 / blend / masked 访存 | 尾块与 `where` | **T-Mask 核心** |
| **I3** | 水平归约 / 跨 tile 合并 | RD* | 中性元 + 尾块 |
| **I4** | 连续 / 跨步 访存 | BC* / layout | masked load/store |
| **I5** | Gather/Scatter/Atomic | 第 6 章索引 | 索引尾块 |
| **I6** | Shuffle / compress | 短行打包、compact | 提升短尾 η_vec |
| **I7** | Block/grid 同步 | 多遍 reduce、专用融合 | 二阶段 dynout |
| **I8** | 初等函数 / MMA / RNG | 激活；第 6 章 GEMM | MMA 边缘块 |

**MVP（通用动态 shape）**：I1 + I2 + I4(连续+mask) + I3(tile reduce)。  
**性能增强**：形态 B 分流、I4 跨步、I6 打包。  
**专用完备**：I5 + I7 + I8。

---

## 8. 易用性评估（修订）

评分：5 = 只写标量语义；4 = 选 dim/plan；3 = 需理解尾块/二阶段；2 = 领域专家 API；1 = 生成器/图编译。

| 类别 | 表达易用 | 动态 shape | 性能发挥（配合 §5 表达） | 综合 |
|---|---|---|---|---|
| Elementwise 对齐 (BC0) | 5 | 易 | 高（形态 B） | **优秀** |
| 标量/末维广播 (BC1/2/5) | 5 | 易 | 高 | **优秀** |
| 复杂广播 (BC3/4/6/7) | 4 | 中 | 中–高（取决于 collapse） | **良好** |
| 内维/合并归约 (RD1/4/5) | 4 | 易 | 高 | **良好** |
| 全维/外维/中间维 (RD0/2/3) | 4 | 中 | 中–高 | **良好** |
| 短内维大量尾块 (RD1 短行) | 3 | 易 | **低**（需打包） | **中等** |
| 多遍统计 (RD8→§6 Norm) | 3 | 中 | 中–高 | **中等** |
| Layout copy | 5 | 易 | 高 | **优秀** |
| 第 6 章 GEMM/Conv/Attn | 2 | 中（尺寸动态） | 很高（专用） | **专家** |
| 第 6 章 DynOut / Sort | 2–3 | 难 | 中 | **专用** |

**总评**：通用路径在「L0 API + 内部整块/尾块分流 + collapse」下，**全量动态 shape 易用性高，且不牺牲整块峰值**；性能风险集中在 **短尾块密集** 与 **未合并的广播/归约轴**，应用 plan 与第 6 章打包/专用核化解。

---

## 9. 落地顺序（简）

| Phase | 内容 |
|---|---|
| P0 | Elementwise + 形态 B 尾块 + 度量 `η_vec/r_tail` |
| P1 | Broadcast 场景 BC0–BC5（含 collapse） |
| P2 | Reduce RD0/RD1/RD4/RD5 + 中性元尾块 |
| P3 | RD2/RD3/RD6、BC6/BC7 短路径 |
| P4 | 第 6 章：GEMM/Norm/Gather 等按需接入，复用尾块原语 |

---

## 10. 附录

### 10.1 术语

| 本文 | ATen / 文献别名 |
|---|---|
| **elementwise** | `pointwise` tag、逐元素 |
| **tile / TILE** | 逻辑 SIMD 宽度 128 lanes |
| **整块 / 尾块** | full tile / remainder epilogue |
| **collapse** | 相邻维合并（TensorIterator 同类） |

### 10.2 参考

- C++26 data-parallel types（P1928）  
- ezyang：PyTorch operators by shape behavior  
- ATen `tags.yaml`（`pointwise`≡本文 elementwise；`reduction`；`dynamic_output_shape`）  

---

*修订要点：pointwise→elementwise；Broadcast/Reduce 多场景化；压缩算法叙述；尾块专章；性能度量与软件表达；专用类归入第 6 章。*
