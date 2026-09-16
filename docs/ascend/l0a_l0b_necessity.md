# 昇腾 Cube 引入 L0A / L0B 的必要性评估

规格锚定公开 Ascend 910B1：L1=512KB，L0A=L0B=64KB，L0C=128KB，Cube 16×16×16，L1→L0A=512 B/cyc，L1→L0B=256 B/cyc，核分摊 HBM 32 B/cyc，入元 FP16 \(s=2\) B。  
可运行模型：`python3 tools/l0_model/matmul_hierarchy_model.py --out tools/l0_model/out`

---

## 1. 问题背景

昇腾 AI Core 的 Cube 不做通用 Load，只从 L0 取数：

```
GM --MTE2--> L1(m×ka 的 A, kb×n 的 B) --MTE1--> L0A / L0B --> Cube --> L0C
```

工程上已经用 L1 缓存每次搬入的 `m×ka`、`kb×n`。要回答的是：**在 L1 已经存在的前提下，是否还要引入 L0A、L0B；L0 与 L1 的容量比取多少；性能收益从哪来。**

约束来自三类软件行为：

1. **多个切分 Matmul**。大 GEMM 拆成核上 tile，计算多次、搬入多次。切完以后算术强度下降，容易变成 **搬入 Bound**。
2. **`blockNum` 核切 A 或切 B**。只切 A(\(M×K\)) 则核上 \(m=M/\textit{blockNum}\)、\(n=N\)、走满 \(K\)；只切 B(\(K×N\)) 则 \(n\) 变薄。\(m\) 或 \(n\) 过小，对侧矩阵几乎不复用。
3. **每次搬入 \(m·k_a\)、\(k_b·n\)，落在 L1**。L1 的职责是抗 HBM 延迟、跨 \(K\) 复用。Cube 每个 pulse 只要 16×16。若 Cube 直接打 L1，L1 从「大块搬入缓存」退化成「512 B 碎读源」，和搬入通路抢端口。

因此评估对象不是「L0 能不能少搬 HBM」，而是「L1 专管搬入、L0 专管供数」这一分层是否必要，以及 \(S_{L0A}/S_{L1}\)、\(S_{L0B}/S_{L1}\) 落在哪一档。

---

## 2. 评估公式

符号：核上形状 \((m,n,K)\)；Cube 峰值 \(P=4096\) MAC/cyc；Cube 边长 \(C=16\)；L1→L0A / L0B 带宽 \(B_A=512\)、\(B_B=256\) B/cyc；HBM 核分摊 \(B_{GM}=32\) B/cyc。

### 2.1 搬入 Bound

$$
T_{\text{cube}}=\frac{mnK}{P},\qquad
T_{GM}=\frac{(m+n)Ks}{B_{GM}}
$$

搬入 Bound 当且仅当 \(T_{GM}>T_{\text{cube}}\)：

$$
\frac{mn}{m+n} < \frac{Ps}{B_{GM}}=256
\tag{1}
$$

方阵需 \(m=n>512\) 才从 HBM 侧计算 Bound。`blockNum=32` 只切 A 时，1024³ 核上 32×1024，\(\frac{mn}{m+n}=31\ll 256\)，必搬入 Bound。

### 2.2 L1 工作集（每次搬入）

双缓冲：

$$
2\cdot k_{l1}\cdot(m+n)\cdot s \le S_{L1}
\tag{2}
$$

即每次缓存到 L1 的 \(m·k_a\) 与 \(k_b·n\)（取 \(k_a=k_b=k_{l1}\)）必须装进半片 L1。

### 2.3 无 L0 时的重载放大

Cube 原生复用宽度为 \(C=16\)。无 L0A 时 A 在 L1 上被读 \(\lceil n/C\rceil\) 次；无 L0B 时 B 被读 \(\lceil m/C\rceil\) 次：

$$
\textit{Bytes}_{L1,A}=mKs\cdot\frac{n}{C},\qquad
\textit{Bytes}_{L1,B}=nKs\cdot\frac{m}{C}
\tag{3}
$$

有 L0 且 A-stationary：A 从 L1 读 1 次，B 按 \(m/m_{l0}\) 次（B-stationary 对偶）。

### 2.4 Cube 口带宽与 L0B 隐藏填入

$$
BW_{\text{Cube},A}=\frac{Ps}{C}=512\ \text{B/cyc}=B_A
\tag{4}
$$

$$
BW_{\text{Cube},B}=\frac{Ps}{C}=512\ \text{B/cyc},\quad B_B=256
$$

L0B 用更大的 \(n_{l0}\) 沿 M 复用 B，填入被计算藏住的条件：

$$
n_{l0}\ \ge\ \frac{Ps}{B_B}=32
\tag{5}
$$

无 L0B 时，B 口有效供数只有 \(B_B\)，Cube 利用率上限 \(B_B/BW_{\text{Cube},B}=50\%\)。

### 2.5 L0 与 L1 工作集比

$$
\rho_A=\frac{m_{l0}\,k_{l0}}{k_{l1}(m+n)},\qquad
\rho_B=\frac{k_{l0}\,n_{l0}}{k_{l1}(m+n)}
\tag{6}
$$

规格比 \(S_{L0A}/S_{L1}=64/512=1/8\)。式 (6) 给出的是 **一次计算用到的工作集比**，用来核对 1/8 是否落在合理区间，而不是要求 L0 做成 L1 的缩小拷贝。

### 2.6 周期屋顶与加速比

$$
T=\max(T_{\text{cube}},\,T_{GM},\,T_{MTE1})+T_{\text{prologue}}
\tag{7}
$$

有 L0A+L0B：\(T_{MTE1}=\max(T_A,T_B)\)（A/B 通道并发）。无 L0：A/B 在 L1 读口串行，且 16×16 碎读另付 L1 气泡。加速比：

$$
\textit{Speedup}=\frac{T(\text{L1-only})}{T(\text{L0A+L0B})}
\tag{8}
$$

闭式若只计 HBM 字节，搬入 Bound 下式 (8) \(=1\)（L0 不改 GM 流量）。把式 (3) 的 L1 重载算进 \(T_{MTE1}\) 后，L1-only 可能先于 GM 顶死，式 (8) \(>1\)。

---

## 3. 数据验证

模型：`tools/l0_model/matmul_hierarchy_model.py`。自检：Cube A 口 = L1→L0A = 512 B/cyc；L0B 隐藏填入 \(n_{l0}\ge 32\)；32 核 A-split 核上 32×1024 满足式 (1)；1024³ 上 L0A+L0B 快于 L1-only。

### 3.1 闭式带宽（不计 L1 气泡）

| 场景 | 核上形状 | \(\frac{mn}{m+n}\) | Bound | 无 L0 重载 (A,B) | 闭式加速比 |
|---|---|---|---|---|---|
| 单核 1024³ | 1024×1024×1024 | 512 | 计算 | 64×, 64× | **3.00×** |
| 32 核 A-split | 32×1024×1024 | 31 | 搬入 | 64×, 2× | **1.00×** |
| 32 核 2D-split | 256×128×1024 | 85 | 搬入 | 8×, 16× | **1.00×** |
| 宽 N | 1024×4096×1024 | 819 | 计算 | 256×, 64× | **3.00×** |
| decode 形 | 16×512×4096 | 16 | 搬入 | 32×, 1× | **1.00×** |

闭式结论：计算 Bound 上无 L0 的 L1 流量按式 (3) 放大，屋顶从 Cube 变成 MTE1，约 3×；搬入 Bound 上屋顶是 GM，L0 不省字节，加速比 1。

### 3.2 层次消融（仿真，含 L1 碎读）

| 场景 | 层次 | Cube% | Bound | vs L1-only |
|---|---|---|---|---|
| 单核 1024³ | L1-only | 30.4% | MTE1 | 1.00× |
| 单核 1024³ | 仅 L0A | 49.2% | MTE1（B 口 256） | 1.62× |
| 单核 1024³ | 仅 L0B | **96.9%** | Cube | **3.18×** |
| 单核 1024³ | L0A+L0B | **96.9%** | Cube | **3.18×** |
| 32 核 A-split | L0A+L0B | 10.7% | GM | **2.17×** |
| 32 核 2D-split | L0A+L0B | 24.6% | GM | **5.42×** |
| 32 核 B-split | L0A+L0B | 10.7% | GM | **3.00×** |
| decode 16×512 | L0A+L0B | 5.6% | GM | 1.47× |

与式 (4)(5) 一致：只加 L0A，Cube 被 \(B_B=256\) 卡在约一半；**L0B 才把利用率拉到 97%**。搬入场景仿真加速比 > 闭式 1.0，因为 L1-only 的 16×16 碎读先把 \(T_{MTE1}\) 抬过 \(T_{GM}\)，L0 把核拉回 GM 屋顶。

### 3.3 L0 / L1 容量比（L1 固定 512KB）

| L0 KB | \(S_{L0}/S_{L1}\) | 单核 L0A+L0B Cube% | 32 核 A-split Cube% |
|---|---|---|---|
| 4 | 1/128 | 78.0% | 10.7% |
| 8 | 1/64 | **96.9%** | 10.7% |
| 16～64 | 1/32～**1/8** | 96.9% | 10.7% |
| 128～256 | 1/4～1/2 | 96.9% | 10.7% |

计算 Bound：8KB 已饱和；4KB 双 L0 ping-pong 不够。搬入 Bound：Cube% 由 GM 决定，加大 L0 无收益。

默认 64KB 工作集（式 (6)）：

| 场景 | L1 ws | L0A ws | L0B ws | \(\rho_A\) | \(\rho_B\) | \(\rho_A+\rho_B\) |
|---|---|---|---|---|---|---|
| 单核 1024³ | 256KB | 2KB | 32KB | 0.008 | **0.125** | 0.13 |
| 32 核 A-split | 231KB | 0.5KB | 32KB | 0.002 | 0.14 | 0.14 |
| 32 核 2D | 252KB | 4KB | 4KB | 0.016 | 0.016 | 0.03 |

规格 64/512=1/8 与单核 \(\rho_B=32\text{KB}/256\text{KB}=1/8\) 对齐。L0A 在 B-stationary 时可以很小；转置/切 N 后角色对调，规格上必须 L0A=L0B。

### 3.4 `blockNum` 扫描（1024³）

| blockNum | 切分 | 式 (1) pair | 搬入? | L0A+L0B vs L1-only |
|---|---|---|---|---|
| 1 | A | 512 | 否 | 3.18× |
| 4 | A | 205 | 是 | 8.09× |
| 32 | A | 31 | 是 | 2.17× |
| 32 | 2D | 85 | 是 | 5.42× |

1D 切薄后必搬入 Bound；2D 把 pair 从 31 抬到 85，是软件侧缓解搬入的办法。2D 后两侧 tile 都不够大，L1-only 双侧重载更狠，L0 收益反而更大。

复现命令：

```bash
python3 tools/l0_model/matmul_hierarchy_model.py --out tools/l0_model/out
```

原始表：`tools/l0_model/out/run_report.md`、`tools/l0_model/out/hierarchy_ablation.csv`。

---

## 4. 结论

1. **L1 不能替代 L0A/L0B。** L1 按式 (2) 缓存搬入块 \(m·k_a\)、\(k_b·n\)；Cube 按 16×16 供数。无 L0 则式 (3) 把 L1 流量放大 \(n/16\)、\(m/16\)，L1 从搬入缓存变成碎读源。
2. **L0B 比 L0A 更硬。** 式 (5)：\(n_{l0}\ge 32\) 才能用 256 B/cyc 填入跟上 512 B/cyc 的 B 口。仅 L0A 利用率 49%；有 L0B 后 **97%，计算 Bound 约 3.2×**（闭式 3.0×）。
3. **L0A 仍然必要，且须与 L0B 等容量。** 切 B / 宽 N / 转置后 A 侧出现同样的重载；Cube A 口与 layout（ZZ）只能接 L0A。规格对称 64KB+64KB。
4. **L0A、L0B 与 L1 的合理比例是 1/8。** 64KB/512KB=1/8，与工作集比 \(\rho_B\approx 1/8\) 一致。低于约 8KB（1/64）伤 ping-pong；高于 64KB 无算力收益。搬入 Bound 应加 L1（更长 \(k_{l1}\)），不应加 L0。
5. **搬入 Bound 下 L0 不省 HBM 字节，但保住搬入屋顶。** 闭式加速比 1.0；仿真因碎读先顶死 L1，32 核 A-split **2.2×**、2D-split **5.4×**。软件用 2D 切 \(M,N\) 提高式 (1) 的 pair；硬件用 L0 保证切完后 L1 仍能当搬入缓存用。
