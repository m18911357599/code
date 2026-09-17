# 昇腾 Cube 引入 L0A / L0B：切分场景下的公式化证明

规格：Ascend 910B1。\(P=4096\) MAC/cyc，\(C=16\)，\(s=2\) B，\(B_A=512\)、\(B_B=256\)、\(B_{GM}=32\) B/cyc，\(S_{L1}=512\)KB，\(S_{L0A}=S_{L0B}=64\)KB。  
默认 GEMM：\(M=N=K=1024\)，`blockNum` \(G=32\)。  
模型：`python3 tools/l0_model/matmul_hierarchy_model.py --out tools/l0_model/out`

---

## 1. 问题背景

Cube 只从 L0 取数；L1 已缓存每次搬入的 \(m·k_a\)、\(k_b·n\)：

```
GM --MTE2--> L1 --MTE1--> L0A / L0B --> Cube(mmad) --> L0C
```

软件把大 Matmul 切到 `blockNum` 核上，四种切法：

| 场景 | 核上 \((m,n,k)\) | 直观 |
|---|---|---|
| **SplitA** | \(m=M/G,\ n=N,\ k=K\) | 切薄 A 的 M，B 几乎不复用 |
| **SplitB** | \(m=M,\ n=N/G,\ k=K\) | 切薄 B 的 N，A 几乎不复用 |
| **SplitK** | \(m=M,\ n=N,\ k=K/G\) | 切 K，C 要核间规约 |
| **SplitA+B** | \(m=M/g_M,\ n=N/g_N,\ k=K\) | 二维切，两侧都变窄 |

要证明的是：在这四种切法下，**L0A、L0B 各自在容量和带宽上的收益是什么**。  
对照假设：**取消 L0A，mmad 直接从 L1 取 A，并把这一路带宽提高到 \(\mu B_A\)**，能否得到与 L0A 相同的时间/流量收益。

---

## 2. 评估公式

### 2.1 周期屋顶

$$
T_{\mathrm{cube}}=\frac{mnk}{P},\quad
T_{GM}=\frac{(m+n)ks}{B_{GM}},\quad
T=\max(T_{\mathrm{cube}},T_{GM},T_{MTE1})
\tag{1}
$$

SplitK 另加部分和写回：\(T_{GM}\leftarrow T_{GM}+mns/B_{GM}\)。

搬入 Bound：\(T_{GM}>T_{\mathrm{cube}}\) \(\Leftrightarrow\) \(mn/(m+n)<Ps/B_{GM}=256\)。

### 2.2 重载与 MTE1

无复用缓冲时 Cube 只按 \(C=16\) 复用：

$$
\eta_A=\frac{n}{C},\qquad \eta_B=\frac{m}{C}
\tag{2}
$$

$$
T_A=\frac{mks\cdot\eta_A}{B_A},\qquad
T_B=\frac{nks\cdot\eta_B}{B_B}
\tag{3}
$$

有 L0A 且 A-stationary：\(\eta_A=1\)。有 L0B 且 B-stationary：\(\eta_B=1\)。  
无 L0B 时 B 口还受 \(B_B=256\) 对 Cube 512 B/cyc 的端口上限：\(T_B\ge 2\,T_{\mathrm{cube}}\)。  
A、B 都从 L1 出：\(T_{MTE1}=T_A+T_B\)；否则 \(T_{MTE1}=\max(T_A,T_B)\)。

### 2.3 填入隐藏（容量下限的来源）

L1→L0A 填一块 \(m_0\times k_0\) 要藏进 Cube：

$$
\frac{m_0 k_0 s}{B_A}\le\frac{m_0 n_0 k_0}{P}
\Rightarrow n_0\ge\frac{Ps}{B_A}=16
\tag{4}
$$

L1→L0B 填一块 \(k_0\times n_0\)：

$$
\frac{k_0 n_0 s}{B_B}\le\frac{m_0 n_0 k_0}{P}
\Rightarrow m_0\ge\frac{Ps}{B_B}=32
\tag{5}
$$

### 2.4 容量（双缓冲、一块 \(k_0=C\)）

$$
S_{L0A}\ge 2\,m\,C\,s,\qquad S_{L0B}\ge 2\,C\,n\,s
\tag{6}
$$

装不下则再切 \(m\) 或 \(n\)，\(\eta\) 回升。64KB 刚好是 \(m=1024\) 或 \(n=1024\)、\(k_0=16\)、FP16 ping-pong 的上界：\(2\cdot1024\cdot16\cdot2=65536\)。

### 2.5 四场景代入（\(G=32,\ 1024^3\)）

| 场景 | \((m,n,k)\) | \(\eta_A=n/C\) | \(\eta_B=m/C\) | 式(6) \(S_{L0A}\) | 式(6) \(S_{L0B}\) | \(T_{\mathrm{cube}}\) | \(T_{GM}\) |
|---|---|---|---|---|---|---|---|
| SplitA | 32×1024×1024 | **64** | 2 | 2KB | **64KB** | 8.2k | 67.6k |
| SplitB | 1024×32×1024 | 2 | **64** | **64KB** | 2KB | 8.2k | 67.6k |
| SplitK | 1024×1024×32 | **64** | **64** | **64KB** | **64KB** | 8.2k | 69.6k（含 C 写） |
| SplitA+B | 256×128×1024 | 8 | 16 | 16KB | 8KB | 8.2k | 24.6k |

证明要点：

- **SplitA**：\(n\) 仍是 1024，无 L0A 则 A 路 L1 流量 ×64；L0A 容量只需 2KB。无 L0B 则 \(\eta_B=2\) 不大，但 L0B 要放下整行 \(n\)，容量 **64KB**。\(m=32\) 恰等于式 (5)，B 填入卡在隐藏边界。
- **SplitB**：与 SplitA 对偶。L0A 容量 64KB，L0B 2KB；无 L0B 时 \(\eta_B=64\)。
- **SplitK**：\(mn/(m+n)=512\) 仍是计算侧 pair，但 \(T_{GM}\) 多一项 \(mns/B_{GM}\)，屋顶变成 GM。\(\eta_A=\eta_B=64\)，两侧 L0 都要 64KB。K 切完不能降低 A/B 重载，只能少算一轮 K。
- **SplitA+B**：两侧 \(\eta\) 都降（8 与 16），容量降到 16KB+8KB，但仍 \(\gg 1\)，L1-only 双侧重载还在。

### 2.6 取消 L0A、mmad 直连 L1、A 路带宽 \(\mu B_A\)

A 仍按 pulse 从 L1 取（无复用缓冲），\(\eta_A=n/C\) 不变：

$$
\frac{T_A^{\mathrm{mmad}}}{T_{\mathrm{cube}}}
=\frac{mks\cdot(n/C)/(\mu B_A)}{mnk/P}
=\frac{Ps}{C\,\mu B_A}
=\frac{1}{\mu}
\quad(B_A=512)
\tag{7}
$$

式 (7) **不含 \(n\)**。\(\mu=1\) 时 \(T_A=T_{\mathrm{cube}}\)，A 路已经顶在 Cube 屋顶；**再加大 \(\mu\) 只让 \(T_A\) 小于 \(T_{\mathrm{cube}}\)，被 \(\max\) 吃掉，时间不变。**

若目标是让 \(T_A^{\mathrm{mmad}}=T_A^{L0A}=mks/B_A\)（流量也对齐，而不只是时间对齐）：

$$
\mu^\ast=\frac{n}{C}
\tag{8}
$$

| 场景 | \(\mu^\ast\) | 所需 \(B_{L1,A}\) |
|---|---|---|
| SplitA / SplitK | 64 | **32768 B/cyc**（64× 现网 L1→L0A） |
| SplitB | 2 | 1024 B/cyc |
| SplitA+B | 8 | 4096 B/cyc |

L1 按 pulse 付延迟 \(L\)（SRAM 未对 Cube 全流水）时：

$$
T_A\leftarrow T_A+N_{\mathrm{pulse}}L,\quad
N_{\mathrm{pulse}}=\frac{mnk}{C^3}=T_{\mathrm{cube}}
\tag{9}
$$

\(L=16\) 时 \(T_A\approx 17\,T_{\mathrm{cube}}\)，搬入屋顶也会被掀翻。L0A 把 \(N_{\mathrm{pulse}}\) 收成 \(N_{\mathrm{fill}}=(m/m_0)(k/k_0)\)，延迟按 DMA 突发付一次。

若 A 口与 MTE2 不能 1R1W 重叠：\(T\leftarrow T_{GM}+T_A\)，搬入更差。  
若把「A 驻留 L1、\(\eta_A=1\)、\(B_A=512\)」做进 L1，那就是把 L0A 的端口、双缓冲、分形做进大 SRAM，不是「加带宽」。

---

## 3. 数据验证

闭式，无 tiler 气泡。完整表见 `tools/l0_model/out/run_report.md`。

### 3.1 单核不切（计算 Bound 参照）

| 层次 | \(\eta_A,\eta_B\) | \(T_A\) | \(T_B\) | \(T\) | Bound | vs L1-only |
|---|---|---|---|---|---|---|
| L1-only | 64, 64 | 262k | 524k | 786k | MTE1 | 1.00× |
| 仅 L0A | 1, 64 | 4.1k | 524k | 524k | MTE1 | 1.50× |
| 仅 L0B | 64, 1 | 262k | 8.2k | **262k** | Cube | **3.00×** |
| L0A+L0B | 1, 1 | 4.1k | 8.2k | **262k** | Cube | **3.00×** |

**L0B 的时间收益是硬的**：无 L0B 时 \(T_B=2T_{\mathrm{cube}}\)，\(T=3T_{\mathrm{cube}}\)。仅 L0A 救不了 B 口。L0B-only 已回到 Cube 屋顶；L0A 在此把 \(T_A\) 从 262k 收到 4.1k，时间被 \(T_{\mathrm{cube}}\) 挡住，收益记在 **L1 占用 / 流量 \(\eta_A=64\)**。

### 3.2 四场景层次（\(G=32\)，屋顶都是 GM）

| 场景 | 层次 | \(T_A\) | \(T_B\) | \(T\) | Bound | 时间 vs L1-only | 容量 L0A / L0B |
|---|---|---|---|---|---|---|---|
| SplitA | L0A+L0B | 128 | 8.2k | 67.6k | GM | 1.00× | 2KB / **64KB** |
| SplitA | L1-only | 8.2k | 16.4k | 67.6k | GM | — | — |
| SplitB | L0A+L0B | 4.1k | 256 | 67.6k | GM | 1.00× | **64KB** / 2KB |
| SplitK | L0A+L0B | 128 | 256 | 69.6k | GM | 1.00× | **64KB / 64KB** |
| SplitA+B | L0A+L0B | 1.0k | 1.0k | 24.6k | GM | 1.00× | 16KB / 8KB |

闭式时间加速比全是 1：四场景都是 \(T_{GM}>T_{MTE1}\)。  
**容量收益不对偶**：SplitA 的面积在 L0B，SplitB 的面积在 L0A，SplitK 两侧都要满配，SplitA+B 两侧都缩小。  
**带宽流量收益**仍按式 (2)：SplitA 的 L0A 把 A 路 L1 流量压 64×（\(T_A\): 8.2k→128），SplitB 的 L0B 把 B 路压 64×（\(T_B\): 16.4k→256）。

带 L1 pulse 延迟 \(L=16\)（mmad 直连、留 L0B）后，时间才拉开：

| 场景 | \(L=0\) | \(L=1\) | \(L=16\) vs L0A+L0B |
|---|---|---|---|
| SplitA / SplitB | 1.00× | 1.00× | **2.06×** |
| SplitK | 1.00× | 1.00× | **2.00×** |
| SplitA+B | 1.00× | 1.00× | **5.67×**（\(T_{GM}\) 低，延迟先顶穿） |

仿真（碎读+端口）与 \(L=16\) 同方向：32 核 A-split 约 2.2×，2D 约 5.4×。

### 3.3 取消 L0A、mmad 从 L1、A 路 ×μ（保留 L0B）

四场景在 \(\mu=1\ldots64\) 下 **\(T\) 全部等于 L0A+L0B**（屋顶是 GM 或 Cube，\(T_A\) 已被挡住）。  
\(\mu=\mu^\ast\) 只把 \(T_A\) 收到与 L0A 相同（SplitA：128 cyc），**不改变 \(T\)**。

要在流量上也对齐 L0A，SplitA/SplitK 需要 **32768 B/cyc** 的 L1→Cube A 口，是现网 L1→L0A 的 64 倍，且 L1 每个 pulse 仍读 \(\eta_A\) 次，bank 翻转不降。  
把 \(\eta_A\) 做成 1 的唯一办法是 **复用缓冲**，容量仍是式 (6) 的 \(S_{L0A}\)，只是焊在 L1 边上，等价于 L0A。

---

## 4. 结论

1. **SplitA**：时间由 GM 定。L0A 容量只需 2KB，收益是 A 路流量 /64；L0B 必须 64KB 才能稳住 \(n=1024\)，且 \(m=32\) 卡在式 (5)。  
2. **SplitB**：与 SplitA 对偶。L0A 必须 64KB；L0B 2KB。无 L0B 时 \(\eta_B=64\)，单核计算 Bound 上这就是 3× 时间。  
3. **SplitK**：pair 不降，C 写让 \(T_{GM}\) 更大。\(\eta_A=\eta_B=64\)，L0A、L0B 都要 64KB。切 K 不能代替 L0。  
4. **SplitA+B**：容量降到 16KB+8KB，\(\eta\) 仍是 8 与 16。\(T_{GM}\) 最低，L1 pulse 延迟最容易先爆（\(L=16\) 时 5.7×）。  
5. **L0B 的带宽收益是时间刚需**（无 L0B ⇒ \(T_B\ge 2T_{\mathrm{cube}}\)）。**L0A 的带宽收益首先是流量 \(\eta_A=n/C\) 和 L1 占用**；时间收益出现在 L1 非全流水或与 MTE2 争口时。  
6. **取消 L0A、mmad 直连 L1、只加该路带宽，不能达成与 L0A 相同的收益。** 式 (7)：\(\mu=1\) 时 \(T_A\) 已等于 \(T_{\mathrm{cube}}\)，再加带宽时间为零。要对齐流量需 \(\mu^\ast=n/C\)（SplitA 为 64×、32768 B/cyc），代价是把 Cube 宽口做到大容量 L1 上，且不消除 \(\eta_A\) 次读。要 \(\eta_A=1\) 必须加一块 \(2mCs\) 的复用缓冲——那就是 L0A。
