# 昇腾 Cube 引入 L0A / L0B 的必要性与性能收益（软件模型）

> 对象：Da Vinci Cube 核内存储层次。规格锚定公开 **Ascend 910B1** `platform_config`：L1=512KB，L0A=L0B=64KB，L0C=128KB，Cube 16×16×16，L1→L0A=512 B/cyc，L1→L0B=256 B/cyc，核分摊 HBM `ddr_rate`=32 B/cyc。
> 可运行模型：`tools/l0_model/matmul_hierarchy_model.py`（tile 级流水，非 RTL）。

---

## 0. 结论

1. **L1 不能替代 L0A/L0B**。Cube 每个 16×16 pulse 需要 A/B 各 512 B/cyc；L1→L0A 刚好 512，L1→L0B 只有 256。没有 L0B，B 口顶死在 50% 峰值。没有 L0A，A 在 `n` 维上按 16 列反复从 L1 读，流量放大 `n/16` 倍。
2. **计算 Bound 收益约 3×**。1024³ FP16、单核：L1-only Cube 利用率 30%，L0A-only 49%（仍被 B 口卡住），L0B 或 L0A+L0B **97%、相对 L1-only 3.2×**。闭式带宽模型给出 3.0×，与仿真一致。
3. **搬入 Bound 不减少 HBM 字节，但避免 L1 被 Cube 撕碎**。`blockNum` 核沿 M 切 A 后，核上 `m` 变小，`mn/(m+n)` 低于 256 即搬入 Bound。此时 L0 不省 GM 流量；它把 L1 上的 16×16 碎读聚成 `m×ka` / `kb×n` 的 MTE 突发，并把 Cube 从 L1 读口卸掉，使 L1 专职 GM 填仓。仿真里 32 核 A-split 仍有 **2.2×**，2D-split 可达 **5.4×**（L1-only 变成 MTE1 Bound）。
4. **L0A/L1 容量比约 1/8 是工作集比，不是越大越好**。64KB/512KB=1/8。单核 1024³ 上 8KB 已回到 97% Cube；4KB（1/128）双缓冲不够，掉到 78%。搬入 Bound 更吃 L1（更长 `k_l1` 突发），再加大 L0 无收益。
5. **L0A 与 L0B 必须成对、容量对称**。卷积/GEMM 转置后 A/B 角色对调；L0B 还要 `n_l0≥32` 才能用 256 B/cyc 的填入跟上 512 B/cyc 的 Cube B 口。只做 L0A ≈ 只发挥一半。

---

## 1. 层次与数据路径

```
GM/HBM  --MTE2-->  L1 (m×ka 的 A, kb×n 的 B)
                   |  MTE1
                   +--> L0A (FRACTAL_ZZ) --Cube A 口 512 B/cyc-->  Cube 16³
                   +--> L0B (FRACTAL_ZN) --Cube B 口 512 B/cyc-->  Cube
                                                          |
                                                         L0C (累加) --> FIXPIPE
```

| 缓冲 | 容量 | 角色 | 软件看到的切分 |
|---|---|---|---|
| L1 | 512KB（双缓冲可用 256KB） | 缓存从 GM 搬入的 `m×ka`、`kb×n`，抗 HBM 延迟 | `k_l1`，以及核内 `m_blk/n_blk` |
| L0A | 64KB（双缓冲 32KB） | Cube 左矩阵，按 `n_l0` 复用 | `m_l0 × k_l0` |
| L0B | 64KB（双缓冲 32KB） | Cube 右矩阵，按 `m_l0` 复用 | `k_l0 × n_l0` |
| L0C | 128KB（双缓冲 64KB） | FP32 累加 `m_l0 × n_l0` | 限制 L0 面 |

L1 的职责是 **搬入与跨 K 复用**；L0 的职责是 **按 Cube pulse 供数、分形 layout、把 L1 读口还给 MTE2**。

---

## 2. 软件模型

### 2.1 多核切分

`blockNum` 个 Block 只切 A(`M×K`) 或只切 B(`K×N`)，K 不切（各核走满 K）：

| 模式 | 核上形状 | 典型后果 |
|---|---|---|
| A-split | `m=M/blockNum`, `n=N`, `k=K` | `m` 变小，B 几乎不复用，**搬入 Bound** |
| B-split | `m=M`, `n=N/blockNum` | `n` 变小，A 不复用 |
| 2D-split | `m=M/g_m`, `n=N/g_n` | `mn/(m+n)` 更大，更接近计算 Bound |

每次搬入 L1 的块是 **`m×ka` 与 `kb×n`**（模型里 `ka=kb=k_l1`），双缓冲约束：

$$
2\cdot k_{l1}\cdot(m+n)\cdot s \le S_{L1}
$$

### 2.2 搬入 Bound 判定

Cube 峰值 \(P=4096\) MAC/cyc，入元 \(s=2\) B，核分摊 HBM \(B_{GM}=32\) B/cyc：

$$
T_{\text{cube}}=\frac{mnK}{P},\quad
T_{GM}=\frac{(m+n)Ks}{B_{GM}}
$$

搬入 Bound 当且仅当 \(T_{GM}>T_{\text{cube}}\)，即

$$
\frac{mn}{m+n} < \frac{P s}{B_{GM}} = 256
$$

方阵要 \(m=n>512\) 才从 HBM 侧计算 Bound。32 核把 1024³ 沿 M 一切，核上 32×1024，\(\frac{mn}{m+n}=31\ll 256\)，必搬入 Bound。这就是「多个切分 Matmul、搬入 Bound、必须多核切分」的定量来源。

### 2.3 无 L0 时的 L1 重载

Cube 原生复用宽度只有 16。没有 L0A 时，A 的每个元素要在 L1 上被读 \(\lceil n/16\rceil\) 次；没有 L0B 时，B 被读 \(\lceil m/16\rceil\) 次：

$$
\text{Bytes}_{L1\leftarrow A}=mKs\cdot\frac{n}{16},\qquad
\text{Bytes}_{L1\leftarrow B}=nKs\cdot\frac{m}{16}
$$

有 L0 且 A-stationary 时，A 只从 L1 读 1 次，B 按 `m/m_l0` 次（对偶于 B-stationary）。

### 2.4 周期

$$
T=\max(T_{\text{cube}},\,T_{GM},\,T_{MTE1})+T_{\text{prologue}}
$$

- 有 L0A+L0B：MTE1 的 A/B 通道并发，\(T_{MTE1}=\max(T_A,T_B)\)。
- 无 L0：A/B 在 L1 读口上串行，且每个 16×16 pulse 另付 L1 访问气泡（模型取 16 cyc，对应「碎读无法把 L1 当成长突发」）。

---

## 3. 为什么必须有 L0A

### 3.1 带宽匹配

$$
BW_{\text{Cube},A}=\frac{P\cdot s}{n_{\text{pulse}}}=\frac{4096\times 2}{16}=512\ \text{B/cyc}
$$

与 L1→L0A=512 **相等**。这不是巧合：L0A 是按 Cube A 口宽度做的寄存器文件。若 Cube 直接打 L1，A 口还要和 B 口、MTE2 写抢同一套 L1 bank。

### 3.2 沿 N 的复用

A-stationary：`L0A` 保住 `m_l0×k_l0`，内层扫 `n`。无 L0A 时复用宽度掉到 16，L1 上 A 流量 ×`(n/16)`。1024³ 上这是 **64×**。

L0A 还有 layout 责任（L1 NZ → L0A ZZ），这不是容量问题，是 Cube 取数格式问题。

### 3.3 L0A 与 L1 的比例

双缓冲工作集：

$$
\rho_A=\frac{S_{L0A}/2}{S_{L1}/2}=\frac{m_0 k_0}{k_{l1}(m+n)}
$$

| 场景（仿真，64KB L0 / 512KB L1） | L1 工作集 | L0A | L0B | L0A/L1 | L0B/L1 | (A+B)/L1 |
|---|---|---|---|---|---|---|
| 单核 1024³（B-stationary） | 256KB | 2KB | 32KB | 0.008 | **0.125** | 0.13 |
| 32 核 A-split，核上 32×1024 | 231KB | 0.5KB | 32KB | 0.002 | 0.14 | 0.14 |
| 32 核 2D，核上 256×128 | 252KB | 4KB | 4KB | 0.016 | 0.016 | 0.03 |

读法：

- **容量规格比** \(64/512=1/8\)。与单核上 L0B 工作集 / L1 工作集 = 32KB/256KB = **1/8** 对齐。
- L0A 在 B-stationary 时可以很小（2KB 级，一个 `m0×k0`）；但 **转置/切 N 之后角色对调**，同一颗 SRAM 必须按 L0B 的 32KB 工作集来做，所以规格上 L0A=L0B=64KB（含 ping-pong）。
- 扫 L0 容量、L1 固定 512KB：单核 1024³ 上 **8KB（比 1/64）已回到 97% Cube**；4KB 双 L0 同时 ping-pong 不够，掉到 78%。64KB 相对 8KB 不再涨算力，留给更大 `n_l0`、转置和延迟隐藏。再加到 128/256KB **零收益**，搬入场景应把面积给 L1 而不是 L0。

**必要性下限**：ping-pong 各一块 Cube pulse 只需 \(2\times 16\times 16\times 2=1\)KB，但 \(T_{\text{cube}}(16^3)=1\) cyc 藏不住 L1 延迟。要 `n_l0≥32` 且一块 L0 填入 ≤ 一块 Cube 计算，工作集就到数 KB～32KB。64KB 是带双缓冲和转置余量的工程点，不是 512KB L1 的缩小版。

---

## 4. 为什么必须有 L0B（且往往比 L0A 更硬）

L1→L0B 只有 **256 B/cyc**（HotChips：feature 的 W×H ≫ Cout，A 侧带宽做宽、B 侧做窄）。Cube B 口仍要 512 B/cyc。用 L0B 把 `n_l0` 做大，B 在 L0 内沿 M 复用，填入被计算藏住的条件是：

$$
n_{l0}\ \ge\ \frac{P s}{BW_{L1\rightarrow L0B}}=\frac{4096\times 2}{256}=32
$$

因此 **L0B 至少要能放下 `k_l0×32` 的双缓冲**，而不是 16 列的单 pulse。

消融（1024³ 单核）：

| 层次 | Cube 利用率 | Bound | vs L1-only |
|---|---|---|---|
| L1-only | 30.4% | MTE1（A/B 各 64× 重载） | 1.0× |
| 仅 L0A | 49.2% | MTE1（B 口 256 vs 512） | 1.6× |
| 仅 L0B | **96.9%** | Cube | **3.2×** |
| L0A+L0B | **96.9%** | Cube | **3.2×** |

只加 L0A，Cube 被 L1→B 的 256 B/cyc 卡在约一半峰值；**L0B 才是打满 Cube 的那一层**。L0A 在「切 N / 宽 N / A-stationary」时补 A 侧重载，并与 L0B 对称以覆盖 NN/NT/TN。

宽 N=4096 时，64KB L0B 放不下整行 `n`（\(n\cdot k_0\cdot s\le 32\)KB 且 \(k_0\ge 16\) ⇒ \(n\le 1024\)），必须沿 N 再切；L0A 负责在这些 N-panel 之间保住 A。模型里 1024×4096×1024 仍是 L0B 把利用率从 31% 拉到 99%，L0A 在该形状上带宽收益接近 0，但切 B 的对偶形状 4096×1024 则反过来。

---

## 5. 搬入 Bound × 多核切分：L0 还赚什么

闭式只算 HBM 字节时，搬入 Bound 的 L0 加速比 = **1.0**（L0 不改 GM 流量）。仿真把 L1 碎读算进去之后：

| blockNum | 切分 | \(\frac{mn}{m+n}\) | 搬入? | L0A+L0B vs L1-only |
|---|---|---|---|---|
| 1 | A | 512 | 否 | **3.18×** |
| 4 | A | 205 | 是 | 8.1×（L1-only 掉进 MTE1） |
| 32 | A | 31 | 是 | **2.17×** |
| 32 | 2D | 85 | 是 | **5.42×** |

解释：

1. 1D A-split 核上 `m=32`，B 几乎不复用，真正屋顶是 GM。没有 L0 时 Cube 仍按 16 列去撕 L1，A 重载 64×，L1 从「给 MTE2 的大块缓存」变成「Cube 的 512B 随机源」，**搬入屋顶都达不到**。L0 把核拉回 GM Bound（Cube 10.7%，GM 90%），这是 2.2× 的来源。
2. 改 2D-split 是软件侧缓解搬入 Bound 的正确办法（`mn/(m+n)` 从 31 升到 85）。但 2D 让 A、B 两侧都不够大，L1-only 两侧重载（8× 与 16×）同时爆发，L0 收益反而更大。
3. 因此：**搬入 Bound 要靠切分策略 + 更大 L1 提高 `k_l1`；L0 负责在切完以后仍能把 Cube 喂饱、并把 L1 还给搬入通路。** 不是「搬入 Bound 所以 L0 没用」。

---

## 6. 性能收益汇总（910B1 参数，FP16）

闭式（只计带宽、不计 L1 气泡）——计算 Bound 3×，搬入 Bound 1×：

- 单核 1024³：无 L0 \(T\approx 786\)k（A/B 各 64×）vs 有 L0 \(T\approx 262\)k → **3.00×**
- 32 核 A-split：两边都是 \(T_{GM}\approx 68\)k → **1.00×**（字节相同）
- 宽 N 1024×4096×1024：无 L0 A 重载 256× → **3.00×**

带 L1 碎读 / 端口模型的仿真：

- 计算 Bound：L0B 或 L0A+L0B **3.2×**，Cube 97%；仅 L0A **1.6×**
- 搬入 Bound 32 核 A-split：**2.2×**（回到 GM 屋顶）
- 搬入 Bound 32 核 2D：**5.4×**
- decode 形 16×512×4096：**1.5×**（屋顶仍是 GM，Cube 仅 5.6%）

---

## 7. 设计含义

| 决策 | 建议 |
|---|---|
| 要不要 L0A | 要。否则 A 沿 N 放大 `n/16`，且 Cube 格式/A 口无处可接。容量规格与 L0B 对齐。 |
| 要不要 L0B | **更要。** 非对称 256 B/cyc 填入必须靠 `n_l0≥32` 的片上复用，否则峰值减半。 |
| L0A : L1 | **~1/8**（64KB : 512KB）。工作集比同样落在 1/8～1/16。低于 ~8KB/512KB 会伤 ping-pong；高于 64KB 无算力收益。 |
| L0B : L1 | 同 1/8；约束来自 `n_l0≥32` 与转置对称，不是来自「再缓存一份 B」。 |
| 搬入 Bound 优化 | 软件：2D 切 `M,N`，不要 1D 切薄；硬件：加 L1 / 加 HBM，而不是加 L0。L0 只保证切完后 L1 仍能当搬入缓存用。 |
| 若取消 L0、把端口做进 L1 | L1 必须提供 512+512 B/cyc 双流 + MTE2 写，面积/时序会逼 L1 变小，等于把 L0 做进 L1 并毁掉 `m×ka` 突发缓存。 |

复现：

```bash
python3 tools/l0_model/matmul_hierarchy_model.py --out tools/l0_model/out
```
