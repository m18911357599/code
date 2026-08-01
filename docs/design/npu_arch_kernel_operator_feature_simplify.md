# AscendC Basic API：基于 `kernel_operator_xx` 的 Feature 提取与 npu_arch 简化设计

> 分析对象：`cann/asc-devkit`（镜像：`mirror-cann/asc-devkit`）`impl/basic_api`  
> 特征单元：`kernel_operator_<feature>`  
> 目标：提取 feature、刻画跨 `npu_arch` 差异、给出渐进式简化方案

---

## 0. 结论摘要

当前 `impl/basic_api` 以 **`#if __NPU_ARCH__` → `dav_<code>/kernel_operator_*_impl.h`** 做编译期分发，共 **9** 套架构树、约 **37** 个公开 feature、约 **13 万行** arch 实现。跨代差异的本质不是“目录命名”，而是 **两套向量计算范式（MemBase vs RegBase）** 与 **不同存储通路图（L1↔GM / NDDMA / UB↔L1 等）**。

因此，npu_arch 简化应遵循：

1. **按 compute family 合并，而不是硬并 2201 与 3510 的同一 `*Impl` 文件**
2. **先抽 RegBase / MemBase 共享模板，再抽 DataCopy / MM 的 capability 路由表**
3. **近克隆优先（3510↔m510、l300↔l311），远差异保 backend**
4. **对外 API 面保持稳定；简化只动 `impl/` 与分发层**

---

## 1. 现状结构

### 1.1 三层组织

```text
include/basic_api/kernel_operator_<feat>_intf.h     # 公开 API
        ↓
impl/basic_api/kernel_operator_<feat>_intf_impl.h   # 包装层：校验 / pipe / debug / 分发
        ↓  #if __NPU_ARCH__ == NNNN
impl/basic_api/dav_<code>/kernel_operator_<feat>_impl.h  # 硬件实现
```

典型分发（`kernel_operator_vec_binary_intf_impl.h`）：

```cpp
#if __NPU_ARCH__ == 1001
#include "dav_c100/kernel_operator_vec_binary_impl.h"
#elif __NPU_ARCH__ == 2002
#include "dav_m200/kernel_operator_vec_binary_impl.h"
#elif __NPU_ARCH__ == 2201
#include "dav_c220/kernel_operator_vec_binary_impl.h"
// ... 3002 / 3102 / 3510 / 5102 / 3003 / 3113
#endif
```

### 1.2 架构映射（`dav_*` ↔ `__NPU_ARCH__`）

| `dav_*` 目录 | `__NPU_ARCH__` | 代表产品 / 备注 | LOC（kernel_operator\*） | 文件数 |
|---|---|---|---|---|
| `dav_c100` | 1001 | 早期训练线（ascend910） | 6.8k | 29 |
| `dav_m200` | 2002 | 推理线（310P / 610） | 9.3k | 29 |
| `dav_c220` | 2201 | Atlas A2/A3 | 13.1k | 36 |
| `dav_m300` | 3002 | 200I/500 A2（310B） | 17.9k | 30 |
| `dav_m310` | 3102 | 610Lite（最精简） | 12.9k | 26 |
| `dav_3510` | 3510 | Ascend 950PR/950DT（RegBase） | 22.2k | 38 |
| `dav_m510` | 5102 | 与 3510 向量侧高度同源 | 19.0k | 37 |
| `dav_l300` | 3003 | L 系列 | 15.2k | 32 |
| `dav_l311` | 3113 | L 系列姊妹 | 13.9k | 32 |

并行命名体系（简化时需统一）：

| 体系 | 用途 | 示例 |
|---|---|---|
| `dav_<code>` | basic_api 实现目录 | `dav_c220` |
| `npu_arch_<NNNN>` | c_api / utils/debug | `npu_arch_2201` |
| `__DAV_C220__` / `__DAV_C310__` | 编译器 / 测试宏 | AIC/AIV 细分 |

CMake **不在配置期** 裁剪 `dav_*`：全部安装，选择完全由预处理器 `__NPU_ARCH__` 完成。

### 1.3 计算范式族（简化的关键轴）

| Family | 代表 arch | 向量实现特征 |
|---|---|---|
| **MemBase** | c100 / m200 / c220 /（部分 m300） | `vadd` 等 intrinsic + `set_mask_*` / `set_vector_mask` |
| **RegBase** | 3510 / m510 /（l300/l311 部分） | `Reg::Add` + `VecBinaryImplTemplate` + MaskReg |
| **过渡 / 精简** | m300 / m310 | 部分 continuous 模板；m310 缺 brcb/gather/list_tensor |

---

## 2. Feature 提取

以公开头 `include/basic_api/kernel_operator_*_intf.h` 为 feature 边界；实现侧对应 `*_intf_impl.h` +（可选）`dav_*/ *_impl.h`。

### 2.1 Feature 清单与职责

| Feature ID | 公开接口文件 | 主要 API / 职责 | 实现形态 |
|---|---|---|---|
| `atomic` | `kernel_operator_atomic_intf.h` | AtomicAdd/Max/Min/Cas/Exch | 仅 3510/5102 |
| `block_sync` | `kernel_operator_block_sync_intf.h` | SetFlag/WaitFlag、PipeBarrier、SyncAll | 经 `sync_impl`（m300/m310 走 common） |
| `cache` | `kernel_operator_cache_intf.h` | DataCachePreload、ICache preload、clean/invalid | 全 arch |
| `common` | `kernel_operator_common_intf.h` | SoC 初始化、atomic store、ctrl SPR、饱和、任务交接 | 全 arch |
| `conv2d` | `kernel_operator_conv2d_intf.h` | Conv2D / tiling | **共享**（无 dav 树） |
| `data_copy` | `kernel_operator_data_copy_intf.h` | DataCopy / Pad / ND-DMA / L1↔UB | 全 arch + 肥 `*_intf_impl` |
| `determine_compute_sync` | `…determine_compute_sync_intf.h` | 块间 Wait/Notify workspace | 全 arch |
| `dump_tensor` | `kernel_operator_dump_tensor_intf.h` | DumpTensor / DumpAccChkPoint | 全 arch |
| `fixpipe` | `kernel_operator_fixpipe_intf.h` | Fixpipe、SetFixPipe\* | 全 arch |
| `gemm` | `kernel_operator_gemm_intf.h` | Gemm / tiling | **共享** |
| `limits` | `kernel_operator_limits_intf.h` | NumericLimits | 头文件内联 |
| `list_tensor` | `kernel_operator_list_tensor_intf.h` | ListTensorDesc | 缺 m300/m310 |
| `mm` | `kernel_operator_mm_intf.h` | LoadData\*、Mmad\*、sparse/MX | 全 arch |
| `mm_bitmode` | `kernel_operator_mm_bitmode_intf.h` | Load2D bit-mode 参数 | 仅 3510/5102 |
| `proposal` | `kernel_operator_proposal_intf.h` | Sort / MrgSort / ProposalConcat | 全 arch |
| `scalar` | `kernel_operator_scalar_intf.h` | CLZ、ScalarCast、GM bypass | **共享** |
| `set_atomic` | `kernel_operator_set_atomic_intf.h` | SetAtomicAdd/Max/Min/None/Type | 全 arch |
| `swap_mem` | `kernel_operator_swap_mem_intf.h` | sys workspace 指针 | 头文件内联 |
| `sys_var` | `kernel_operator_sys_var_intf.h` | BlockIdx/Num、arch version、UB size、cycle | 全 arch |
| `utils` | `kernel_operator_utils_intf.h` | Nop / Async | 头文件内联 |
| `vec_bilinearinterpolation` | `…vec_bilinearinterpolation_intf.h` | BilinearInterpolation | 全 arch |
| `vec_binary` | `kernel_operator_vec_binary_intf.h` | Add/Sub/Mul/Div/Max/Min/And/Or… | 全 arch |
| `vec_binary_scalar` | `…vec_binary_scalar_intf.h` | Adds/Muls/Maxs、LeakyRelu | 全 arch |
| `vec_brcb` | `kernel_operator_vec_brcb_intf.h` | Brcb 广播 | 缺 m310 |
| `vec_cmpsel` | `kernel_operator_vec_cmpsel_intf.h` | Compare\* / Select | 3510 拆为 cmp+sel |
| `vec_createvecindex` | `…vec_createvecindex_intf.h` | CreateVecIndex | 全 arch |
| `vec_duplicate` | `kernel_operator_vec_duplicate_intf.h` | Duplicate / Interleave | 全 arch |
| `vec_gather` | `kernel_operator_vec_gather_intf.h` | Gather / Gatherb | 缺 m310 |
| `vec_gather_mask` | `…vec_gather_mask_intf.h` | GatherMask | 全 arch |
| `vec_mulcast` | `kernel_operator_vec_mulcast_intf.h` | MulCast | 全 arch |
| `vec_reduce` | `kernel_operator_vec_reduce_intf.h` | ReduceMax/Min/Sum | 全 arch |
| `vec_scatter` | `kernel_operator_vec_scatter_intf.h` | Scatter | 全 arch |
| `vec_ternary_scalar` | `…vec_ternary_scalar_intf.h` | Axpy | 全 arch |
| `vec_transpose` | `kernel_operator_vec_transpose_intf.h` | Transpose / TransDataTo5HD | 全 arch |
| `vec_unary` | `kernel_operator_vec_unary_intf.h` | Relu/Exp/Ln/Abs/Sqrt… | 全 arch |
| `vec_vconv` | `kernel_operator_vec_vconv_intf.h` | Cast\* / DeqScale | 全 arch |
| `vec_vpadding` | `kernel_operator_vec_vpadding_intf.h` | VectorPadding | 全 arch |

### 2.2 Arch-only 辅助模块（无独立公开 intf）

| Helper | 出现 arch | 作用 |
|---|---|---|
| `sync` | 多数（非 m300/m310） | block_sync 后端 |
| `vec_template` | 3510 / m510 / l300 / l311 | RegBase 向量模板 |
| `vec_binary_continuous` | m300+ / RegBase 系 | 连续地址向量路径 |
| `vec_cmp` / `vec_sel` | 3510（及 c220 部分） | cmpsel 拆分 |
| `vec_compare_continuous` | 3510 / m510 | 连续 compare |
| `scm_data_copy` | c220 / 3510 / m510 | SCM 相关搬移 |
| `fixpipe_v2` | c220 / m300 | Fixpipe 扩展 |
| `set_spr` | c220 / m300 / l300 / l311 | SPR 配置 |
| `mm_bitmode` / `atomic` / `print` | 3510 / m510 | 新架构能力 |
| `cube_others` / `vec_others` / `reg_others` | c220 等 | 杂项 |

### 2.3 Feature × Arch 存在矩阵

图例：`Y`=有 arch impl；`S`=仅共享层；`-`=无；`†`=拆分实现；`~`=经 common 间接提供。

```text
feature                      c100  m200  c220  m300  m310  3510  m510  l300  l311
atomic                         -     -     -     -     -     Y     Y     -     -
block_sync                     Y     Y     Y     ~     ~     Y     Y     Y     Y
cache                          Y     Y     Y     Y     Y     Y     Y     Y     Y
common                         Y     Y     Y     Y     Y     Y     Y     Y     Y
conv2d                         S     S     S     S     S     S     S     S     S
data_copy                      Y     Y     Y     Y     Y     Y     Y     Y     Y
determine_compute_sync         Y     Y     Y     Y     Y     Y     Y     Y     Y
dump_tensor                    Y     Y     Y     Y     Y     Y     Y     Y     Y
fixpipe                        Y     Y     Y     Y     Y     Y     Y     Y     Y
gemm                           S     S     S     S     S     S     S     S     S
limits                         S     S     S     S     S     S     S     S     S
list_tensor                    Y     Y     Y     -     -     Y     Y     Y     Y
mm                             Y     Y     Y     Y     Y     Y     Y     Y     Y
mm_bitmode                     -     -     -     -     -     Y     Y     -     -
proposal                       Y     Y     Y     Y     Y     Y     Y     Y     Y
scalar                         S     S     S     S     S     S     S     S     S
set_atomic                     Y     Y     Y     Y     Y     Y     Y     Y     Y
swap_mem                       S     S     S     S     S     S     S     S     S
sys_var                        Y     Y     Y     Y     Y     Y     Y     Y     Y
utils                          S     S     S     S     S     S     S     S     S
vec_bilinearinterpolation      Y     Y     Y     Y     Y     Y     Y     Y     Y
vec_binary                     Y     Y     Y     Y     Y     Y     Y     Y     Y
vec_binary_scalar              Y     Y     Y     Y     Y     Y     Y     Y     Y
vec_brcb                       Y     Y     Y     Y     -     Y     Y     Y     Y
vec_cmpsel                     Y     Y     Y     Y     Y    Y†    Y     Y     Y
vec_createvecindex             Y     Y     Y     Y     Y     Y     Y     Y     Y
vec_duplicate                  Y     Y     Y     Y     Y     Y     Y     Y     Y
vec_gather                     Y     Y     Y     Y     -     Y     Y     Y     Y
vec_gather_mask                Y     Y     Y     Y     Y     Y     Y     Y     Y
vec_mulcast                    Y     Y     Y     Y     Y     Y     Y     Y     Y
vec_reduce                     Y     Y     Y     Y     Y     Y     Y     Y     Y
vec_scatter                    Y     Y     Y     Y     Y     Y     Y     Y     Y
vec_ternary_scalar             Y     Y     Y     Y     Y     Y     Y     Y     Y
vec_transpose                  Y     Y     Y     Y     Y     Y     Y     Y     Y
vec_unary                      Y     Y     Y     Y     Y     Y     Y     Y     Y
vec_vconv                      Y     Y     Y     Y     Y     Y     Y     Y     Y
vec_vpadding                   Y     Y     Y     Y     Y     Y     Y     Y     Y
```

### 2.4 Feature 分组（便于渐进改造）

| 组 | Features | 简化优先级 | 理由 |
|---|---|---|---|
| **G0 已共享** | conv2d, gemm, scalar, limits, swap_mem, utils | 低（保持） | 无/极少 arch 分叉 |
| **G1 RegBase 向量核** | vec_binary, vec_unary, vec_binary_scalar, vec_ternary_scalar, vec_template, vec_binary_continuous, vec_cmpsel, vec_reduce, vec_duplicate, vec_vconv | **P0** | 3510↔m510 Jaccard 最高（0.55–0.98） |
| **G2 MemBase 向量核** | 同上在 c100/m200/c220 | **P1** | 同范式可共享；不可与 RegBase 硬并 |
| **G3 轻量系统类** | cache, sys_var, set_atomic, common, dump_tensor, determine_compute_sync | **P1** | 差异小、收益稳 |
| **G4 Data 通路** | data_copy, scm_data_copy, fixpipe | **P2** | 通路图差异大，适合 capability 表 |
| **G5 Cube/MM** | mm, mm_bitmode, proposal, fixpipe | **P2–P3** | ISASI 强，先抽骨架再留 endpoint |
| **G6 新架构独有** | atomic, mm_bitmode, print, Mutex/CrossCore 相关 | **P3** | 保持 arch backend + 能力开关 |
| **G7 包装层卫生** | 所有 `*_intf_impl.h` 的 9 路 include 链 | 贯穿全程 | 宏/生成器消除重复 |

---

## 3. Feature 在 npu_arch 上的差异

### 3.1 差异维度（跨 feature 通用）

| 维度 | 说明 | 典型 feature |
|---|---|---|
| D1 指令 / Builtin 命名 | `vadd` vs `Reg::Add`；`copy_*_b16` vs `*_v2` | vec_\*, data_copy |
| D2 存储通路有无 | L1→GM、GM→L0、UB→L1、NDDMA、L0C↔UB | data_copy, mm, fixpipe |
| D3 Layout / 分形 | L0A ZZ↔ZN、ND2NZ / DN2NZ | mm, data_copy |
| D4 Mask 模型 | set_mask_count vs MaskReg/ZEROING | vec_\* |
| D5 数据类型 / 特性集 | s4、sparse、MX、bf16、Subnormal config | mm, vec_binary, vec_vconv |
| D6 同步原语 | 经典 PipeFlag vs Mutex/CrossCore | block_sync, common |
| D7 参数结构 | LoadData2D vs V2、cacheMode、FixpipeParams | mm, data_copy, fixpipe |
| D8 包装层 ifdef | `*_intf_impl` 内路径 / 校验分支密度 | data_copy, vec_reduce, proposal |

### 3.2 代码相似度（行集合 Jaccard，去空/注释）

| Feature | c220↔3510 | 3510↔m510 | l300↔l311 | c100↔m200 | c220↔m200 |
|---|---|---|---|---|---|
| `vec_binary` | 0.17 | **0.98** | 0.66 | 0.43 | 0.47 |
| `vec_unary` | 0.09 | **0.85** | 0.34 | 0.53 | 0.71 |
| `vec_reduce` | 0.05 | 0.55 | **0.99** | 0.82 | 0.77 |
| `vec_template` | — | **0.83** | 0.83 | — | — |
| `data_copy` | 0.20 | **0.70** | 0.46 | **0.74** | 0.20 |
| `mm` | 0.17 | 0.31 | 0.57 | 0.52 | 0.42 |
| `fixpipe` | 0.18 | 0.61 | 0.74 | **0.89** | 0.15 |
| `sys_var` | 0.63 | 0.69 | **0.93** | 0.56 | 0.55 |
| `set_atomic` | 0.44 | **0.97** | **0.95** | 0.77 | 0.42 |
| `cache` | **0.75** | **0.95** | **0.95** | 0.75 | 0.78 |
| `common` | 0.43 | 0.62 | 0.76 | 0.64 | 0.57 |

解读：

- **跨代（2201↔3510）向量/搬移几乎不可直接合并**（Jaccard < 0.2）
- **同族近克隆可立即共享**：3510↔m510、l300↔l311、c100↔m200（部分）
- **cache / sys_var / set_atomic** 是全族最容易抽 common 的系统类 feature

### 3.3 代表性 Feature 差异详解

#### 3.3.1 `vec_binary`（范式分裂的样板）

| 项 | MemBase（c220） | RegBase（3510 / m510） |
|---|---|---|
| 核心调用 | `vadd(dst, src0, src1, …)` | `Reg::Add` + `VecBinaryImplTemplate` |
| Mask | `set_mask_count` / `set_vector_mask` | `Reg::MaskReg` / `MaskMergeMode::ZEROING` |
| 文件形态 | 大体量 intrinsic 包装 | 薄包装 + `vec_template` / `continuous` |
| 类型 | half/float/int16/int32 为主 | 扩展 bf16、更多整数；Div 有 algo/config |
| 简化方向 | MemBase family 共享 `*IntrinsicsImpl` | 上提 template 到 `impl/basic_api/regbase/` |

#### 3.3.2 `data_copy`（通路图分裂的样板）

| 通路 / 能力 | c220 (2201) | 3510 | m200 (2002) |
|---|---|---|---|
| GM↔UB | Y | Y（含 align_v2 / cacheMode） | Y |
| GM↔L1 ND2NZ | Y | Y（+ DN2NZ / multi） | 有限 / 部分不支持上报 |
| L1→GM | **Y** | **无 / 需绕行** | 有限 |
| UB↔L1 | 有限 | **增强**（`copy_ubuf_to_cbuf` 等） | 常 `NOT_SUPPORT` |
| NDDMA | — | **Y** | — |
| L0C↔UB/GM | 矩阵搬出为主 | 更丰富 | 强调 L0C↔UB |

`*_intf_impl.h` 本身约 **1841** 行，含大量 `__NPU_ARCH__` 路径选择——这是包装层泄漏，应下沉为 **path → backend** 表。

#### 3.3.3 `mm`（ISASI / 特性集分裂）

| 能力 | c220 / m200 | 3510 |
|---|---|---|
| LoadData2D/3D + Mmad | Y | Y（参数 V2 / stride 变体） |
| Sparse / unzip-to-L0 / s4 | **保留** | **删除或迁移** |
| MX（`MmadMx` 等） | — | **新增** |
| GM→L0A/B 直达 | 部分存在 | **禁止**（须 GM→L1→LoadData） |
| SetLoadDataBoundary / L0 Fill | 有 | 删除/桩 |

跨代 mm 应共享 **编排骨架**，保留 **endpoint builtin**；不要追求单文件 `#ifdef` 全集。

#### 3.3.4 系统类（`cache` / `sys_var` / `set_atomic`）

- 相似度高、语义稳定，适合最早落地 `common + delta`
- `common` 在 l300/l311 已出现 **复用 3510 common_impl** 的苗头——应正规化为显式 shared 模块，而不是跨目录硬 include

---

## 4. 目标形态：简化后的 npu_arch

### 4.1 目标分层

```text
include/basic_api/kernel_operator_<feat>_intf.h          # 不变：稳定公开面
impl/basic_api/kernel_operator_<feat>_intf_impl.h        # 变薄：校验 + 调用 family API
impl/basic_api/
├── family/
│   ├── membase/          # 共享 MemBase 模板 / intrinsic 序列
│   ├── regbase/          # 共享 RegBase 模板（从 3510 上提）
│   └── policy/           # DataCopy / MM 路由与 capability
├── backend/
│   ├── npu_1001/ … npu_3113/   # 仅保留无法共享的 endpoint / 差异补丁
│   └── （过渡期可仍叫 dav_*，最后统一命名）
└── capability.h          # HAS_L1_TO_GM / VEC_REGBASE / HAS_NDDMA …
```

### 4.2 Capability 开关（替代散落 `#if __NPU_ARCH__`）

```cpp
// 示意
struct NpuCaps {
  static constexpr bool kVecRegBase = (__NPU_ARCH__ == 3510) || (__NPU_ARCH__ == 5102)
                                   || (__NPU_ARCH__ == 3003) || (__NPU_ARCH__ == 3113);
  static constexpr bool kHasL1ToGm  = (__NPU_ARCH__ == 2201) /* + 显式列表 */;
  static constexpr bool kHasNddma   = (__NPU_ARCH__ == 3510) || (__NPU_ARCH__ == 5102);
  static constexpr bool kHasMxMmad  = (__NPU_ARCH__ == 3510) || (__NPU_ARCH__ == 5102);
  // ...
};
```

原则：

- **业务代码问 capability，不问产品名**
- arch 枚举只出现在 `capability.h` 与 backend 注册处

### 4.3 命名统一（最后一步）

| 现状 | 目标建议 |
|---|---|
| `dav_c220` vs arch `2201` | 新代码统一 `backend/npu_2201`（或保留 dav 但生成映射表） |
| `npu_arch_*`（c_api）与 `dav_*`（basic）双轨 | basic 先 family 化；c_api 后续 mirror `common + npu_*` |
| `__DAV_C310__` vs dir `dav_3510` | 文档/脚本建立单向真源表，禁止第三套口语名扩散 |

---

## 5. 每个 Feature 的渐进式修改方案

### 5.1 总体阶段

| Phase | 目标 | 退出标准 |
|---|---|---|
| **P0** | RegBase 向量核共享化 | 3510/m510（及可接入的 l\*）共用同一 template TU；单测全绿 |
| **P1** | MemBase 向量核 + 系统类 common | c220/m200/c100 向量差异文件 < 阈值；cache/sys_var/set_atomic 合并 |
| **P2** | DataCopy / Fixpipe capability 路由 | `*_intf_impl` 中 `__NPU_ARCH__` 计数显著下降；通路表可单测 |
| **P3** | MM / Proposal 骨架共享 + 独有能力 backend | 跨代编排共用；sparse/MX 仍隔离 |
| **P4** | 命名 / c_api 对齐 / 生成分发 include | 单一 arch id；重复 9 路 include 由宏/生成消除 |

每阶段均要求：**公开 API 无行为回归**（header checker + 既有 api tests）。

### 5.2 分 Feature 方案

#### A. 已共享（G0）— 维持

| Feature | 动作 |
|---|---|
| `conv2d` / `gemm` / `scalar` / `limits` / `swap_mem` / `utils` | 不拆 arch；仅清理若存在的散落 `#if`；作为“共享层范本” |

#### B. RegBase 向量（G1）— Phase P0

| Feature | Step 1 | Step 2 | Step 3 |
|---|---|---|---|
| `vec_template` | 从 `dav_3510` 上提至 `family/regbase/` | m510/l300/l311 改为 include 共享 | 删除重复副本 |
| `vec_binary_continuous` / `vec_compare_continuous` | 同上 | 对齐 3510↔m510 差异补丁（`#if` 局部） | 单测回归 |
| `vec_binary` | 保留薄 `*Impl`（绑 `Reg::Op`） | 去掉与 template 重复的 glue | m510 与 3510 字节级趋同后只留一份 |
| `vec_unary` / `vec_binary_scalar` / `vec_ternary_scalar` | 同上模式 | 统一 dtype `static_assert` 策略 | — |
| `vec_cmpsel` | 统一 3510 的 cmp/sel 拆分或提供 facade | m510 走同一 facade | 旧 arch 仍走 MemBase |
| `vec_reduce` / `vec_duplicate` / `vec_vconv` / `vec_mulcast` / `vec_vpadding` | 评估是否已用 template | 能套 template 的迁入 | 不能的留 backend |
| `vec_gather` / `scatter` / `brcb` / `gather_mask` / `createvecindex` / `transpose` / `bilinear` | 次优先；先 capability（m310 缺 brcb/gather） | 共享编排 | endpoint 留 backend |

**验收：** RegBase 族每新增向量 API 只需改 `family/regbase` + 一处 op 绑定，而不是 4 个目录。

#### C. MemBase 向量（G2）— Phase P1

| Feature | 方案 |
|---|---|
| `vec_binary` 等 | 抽取 `family/membase/VecBinaryIntrinsics.h`：统一 mask save/set/restore + `vadd` 调用序列 |
| 与 RegBase 关系 | **禁止** 合入同一 `AddImpl`；在 `intf_impl` 用 `NpuCaps::kVecRegBase` 二选一 |
| c100 vs m200 | 优先合并高相似文件（如部分 data_copy/fixpipe/cache），差异用小补丁 |

#### D. 系统类（G3）— Phase P1

| Feature | 方案 |
|---|---|
| `cache` | 三路近乎相同 → `family/common/cache_impl.h` + 极少 SPR/地址差 |
| `sys_var` | 共享 GetBlockIdx/Num 等；arch version / UB size 用 constexpr 表 |
| `set_atomic` | 3510↔m510、l300↔l311 直接合一 |
| `common` | 废除 l\* → 硬 include 3510；改为正式 `family` 依赖 |
| `dump_tensor` / `determine_compute_sync` | 校验与格式共享；落盘/同步细节留 backend |
| `block_sync` | 统一经 `sync` backend；把 m300/m310 的“藏在 common”改为显式 stub/backend |

#### E. Data 通路（G4）— Phase P2

| Feature | 方案 |
|---|---|
| `data_copy` | 1）定义 path enum：`GM2UB/UB2GM/GM2L1/L12GM/UB2L1/L12UB/L0C2*/NDDMA…`<br>2）`capability` 描述每 arch 支持集合<br>3）`intf_impl` 只做 `(src,dst,params) → backend`<br>4）pad/slice/ND2NZ **编排** 进 `family/policy`；builtin 留 backend |
| `scm_data_copy` | 挂到同一 path 表的扩展槽 |
| `fixpipe` | 拆“参数正规化 / 通路选择 / 指令发射”；v1/v2 用 capability |

**反模式：** 在一个 `DataCopyGM2UBImpl` 里用巨型 `#if` 同时实现 2201 与 3510。

#### F. Cube / MM（G5）— Phase P2–P3

| Feature | 方案 |
|---|---|
| `mm` | 共享 Load/Mmad **循环与参数校验**；GM→L0 等缺失路径走文档化 workaround 策略对象 |
| `mm_bitmode` | 保持 3510/5102 backend；仅共享 struct 定义（已有 `*_struct.h`） |
| `proposal` | 排序网络共享；指令发射分 MemBase/RegBase |
| `conv2d`/`gemm` | 已共享，跟 MM 策略对象对齐即可 |

#### G. 新架构独有（G6）— Phase P3

| Feature | 方案 |
|---|---|
| `atomic` / `print` / MX / NDDMA / Mutex | `capability` 暴露；未支持 arch 保持编译期不可见或 `NOT_SUPPORT` |
| 不强制回移植到 2201 | 避免虚假“全 arch 实现” |

#### H. 分发卫生（G7）— 贯穿

| 动作 | 说明 |
|---|---|
| `ASC_INCLUDE_ARCH_IMPL(vec_binary)` 宏或 codegen | 消灭 ~28 处手写 9 路 `#elif` |
| 压缩 `*_intf_impl` 体积 | 先砍 data_copy / vec_binary_scalar / vec_binary / vec_reduce 中的 arch 业务分支 |
| 测试矩阵 | 至少覆盖：`2201`、`3510`、`5102`、`2002` + 一款 L 系 |

### 5.3 推荐落地顺序（可执行 backlog）

1. **引入 `capability.h` + `family/regbase/` 空壳**（不改行为）
2. **上提 `vec_template` + `vec_binary_continuous`**，切换 3510/m510
3. **合并 `vec_binary` / `vec_unary` RegBase 薄封装**
4. **合并 `cache` / `set_atomic` / `sys_var`**
5. **MemBase `vec_binary` family**
6. **DataCopy path table（先 GM↔UB + GM↔L1，再 NDDMA/L12GM）**
7. **MM 骨架 + MX/sparse 隔离**
8. **统一 backend 目录名 / 生成 include / 评估 c_api 同步**

### 5.4 风险与约束

| 风险 | 缓解 |
|---|---|
| RegBase/MemBase 误合并导致静默精度/性能回退 | family 门禁 + 分 arch CI |
| `*_intf_impl` 大重整影响 CPU debug 路径 | 保留 `ASCENDC_CPU_DEBUG` 钩子位置不变 |
| l300/l311 与 3510“看似相近实则 common 复用脏” | 先断跨目录硬 include，再谈合并 |
| 公开 API / ISASI 文档承诺 | 只动 impl；ISASI 行为变更需同步迁移指南 |
| 一次改全部 arch | 严格按 Phase；每 Phase 可独立回滚 |

---

## 6. 度量指标（简化是否成功）

| 指标 | 现状量级 | 目标方向 |
|---|---|---|
| arch 树总 LOC（basic_api `dav_*` kernel_operator） | ~130k | 明显下降（优先砍近克隆） |
| 3510↔m510 `vec_binary` 重复度 | Jaccard 0.98（双份维护） | → 单份 shared |
| `*_intf_impl` 内 `__NPU_ARCH__` 出现次数 | data_copy≈167、vec_reduce≈137 等 | 下沉到 capability/path 表后大幅减少 |
| 新增向量 API 需改动的目录数 | 最多 9 | RegBase≤2，MemBase≤2 |
| 命名体系数量 | 3（dav / npu_arch / \_\_DAV\_\_） | 对外文档 1 张真源表；新代码 1 套 backend id |

---

## 7. 附录

### 7.1 分析数据来源

- 源码树：`impl/basic_api/**`、`include/basic_api/kernel_operator_*`
- 文档：`docs/zh/contributing/directory-structure.md`、跨代迁移指南、`asc_950_feature_guide.md`
- 相似度：对 `*_impl.h` 非空非注释行做 Jaccard

### 7.2 与 adv_api / c_api 的关系

- **adv_api** 已有 `_common_impl` + `_3510_impl` 模式，是 basic_api 简化的对标形态
- **c_api** 的 `npu_arch_2201/3510` 重复更重，建议在 basic family 稳定后，按同样 `common + backend` 推进（不在本设计 P0 范围）

### 7.3 Feature ID 稳定约定

```text
feature_id == kernel_operator_<name> 中 <name>
公开头：    include/basic_api/kernel_operator_<name>_intf.h
包装实现：  impl/basic_api/kernel_operator_<name>_intf_impl.h
arch 实现： impl/basic_api/dav_*/kernel_operator_<name>_impl.h   # 过渡期
目标实现：  impl/basic_api/family/{membase|regbase|policy}/...
           impl/basic_api/backend/npu_<NNNN>/...
```

---

## 8. 下一步建议

1. 评审本文件的 Feature 分组与 Phase 切分是否与团队发布节奏一致  
2. 以 **`vec_template` + `vec_binary`（3510/m510）** 做第一个可合并 PR（行为零变更）  
3. 同步建立 `capability.h` 与 arch 真源映射表，供后续 DataCopy/MM 复用  

---

*文档生成说明：基于 asc-devkit `impl/basic_api` 静态结构与抽样 diff/相似度分析；未改动上游仓库代码。*
