# NVIDIA Thread Block Cluster 定义与三种调用模式

> CC ≥ 9.0（Hopper+）· [CUDA Programming Guide — Thread Block Clusters](https://docs.nvidia.com/cuda/cuda-c-programming-guide/) · [Advanced Host Programming](https://docs.nvidia.com/cuda/cuda-programming-guide/03-advanced/advanced-host-programming.html)

**结论**：Cluster 是 Grid 与 Block 之间的**可选**调度层——一组 Thread Block 保证同驻一个 GPC，可做 `cluster.sync()` 与 Distributed Shared Memory（DSMEM）。Host 侧有三种调用/配置模式。

---

## 1. 定义与分层

### 1.1 执行层次（含 Cluster）

```
Grid
 └─ Cluster          ← 可选；一组共调度 Thread Block
     └─ Thread Block
         └─ Warp
             └─ Thread
```

| 层 | 含义 | 调度/通信保证 |
|----|------|----------------|
| **Grid** | 一次 kernel launch 的全部 Thread Block | 块间默认无序；无硬件共驻保证 |
| **Cluster** | 用户指定的一组共调度 Block | 同 GPC 共驻；`cluster.sync()`；DSMEM |
| **Block** | 一组共调度 Thread | 同 SM；shared memory；`__syncthreads` |
| **Warp / Thread** | 执行与索引粒度 | SIMT 车道 |

兼容约定：带 Cluster 的 launch 中，`gridDim` **仍以 Thread Block 个数计**（除非启用「Blocks as Clusters」，见模式 C）。

### 1.2 Cluster 在硬件上代表什么

| 软件概念 | 硬件映射 |
|----------|----------|
| Cluster | 一组 Block **共调度到同一 GPC** |
| Cluster 尺寸 | 每 Cluster 含多少 Block（可移植上限通常为 **8**；部分机型可 opt-in 到 16） |
| DSMEM | Cluster 内各 Block 的 shared memory 互相可读写/原子 |

**约束摘要**：

- 各维 `gridDim.{x,y,z}` 必须能被对应 `clusterDim.{x,y,z}` 整除。
- 可移植 Cluster 大小 ≤ 8；更大尺寸需 `cudaFuncAttributeNonPortableClusterSizeAllowed` 等显式允许。
- Occupancy 建议用 `cudaOccupancyMaxActiveClusters` / `cudaOccupancyMaxPotentialClusterSize` 查询。

### 1.3 Device 侧常用查询

```cpp
#include <cooperative_groups.h>
namespace cg = cooperative_groups;

__global__ void k(...) {
    cg::cluster_group cluster = cg::this_cluster();
    unsigned rank  = cluster.block_rank();   // 本 Block 在 Cluster 内秩
    dim3     cdim  = cluster.dim_blocks();   // Cluster 内 Block 维
    cluster.sync();                          // Cluster 级屏障
    // DSMEM：cluster.map_shared_rank(local_smem, dst_block_rank)
}
```

---

## 2. 三种调用模式

| 模式 | 名称 | Cluster 尺寸何时定 | Launch 写法 | `<<<>>>` 第一参语义 |
|------|------|-------------------|-------------|---------------------|
| **A** | 编译期固定 Cluster | 编译期 `__cluster_dims__` | 经典 `<<<grid, block>>>` | **Thread Block 数** |
| **B** | 运行时 Cluster 维 | Launch 属性 `ClusterDimension`（可选 Preferred） | `cudaLaunchKernelEx` | **Thread Block 数**（`config.gridDim`） |
| **C** | Blocks as Clusters | 编译期 `__block_size__(block, cluster)` | `<<<numClusters, ...>>>` | **Cluster 数** |

三种模式互斥要点：`__cluster_dims__` 与 `__block_size__` 的第二元组**不可同时**指定；模式 C 启用后，`<<<>>>` 第一参按 **Cluster** 计数。

---

### 模式 A — 编译期 `__cluster_dims__` + 经典 Launch

**适用**：Cluster 形状固定、希望继续用 `<<<grid, block>>>`。

```cpp
// 编译期固定：每 Cluster 在 X 维 2 个 Block
__global__ void __cluster_dims__(2, 1, 1)
cluster_kernel(float *input, float *output)
{
    // ...
}

int main() {
    dim3 threadsPerBlock(256);
    dim3 numBlocks(/* 必须是 clusterDim 的整数倍 */);
    cluster_kernel<<<numBlocks, threadsPerBlock>>>(input, output);
}
```

| 点 | 说明 |
|----|------|
| Cluster 大小 | 写死在符号上，**launch 时不可改** |
| Grid 枚举 | 仍按 **Block** 数；Cluster 个数 = `grid / clusterDim`（各维相除） |
| 优点 | 写法简单，与旧代码路径接近 |
| 缺点 | 同一符号无法按负载动态改 Cluster 大小 |

---

### 模式 B — 运行时 `cudaLaunchKernelEx` + Launch Attribute

**适用**：同一 kernel 按直方图 bins、DSMEM 容量等**动态**选 Cluster 大小。

```cpp
__global__ void cluster_kernel(float *input, float *output)
{
    // 无编译期 __cluster_dims__；尺寸由 Host 属性给出
}

int main() {
    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks(N / threadsPerBlock.x, N / threadsPerBlock.y);

    cudaLaunchConfig_t config = {0};
    config.gridDim  = numBlocks;          // 仍按 Block 计
    config.blockDim = threadsPerBlock;

    cudaLaunchAttribute attribute[1];
    attribute[0].id = cudaLaunchAttributeClusterDimension;
    attribute[0].val.clusterDim.x = 2;
    attribute[0].val.clusterDim.y = 1;
    attribute[0].val.clusterDim.z = 1;
    config.attrs    = attribute;
    config.numAttrs = 1;

    cudaLaunchKernelEx(&config, cluster_kernel, input, output);
}
```

#### B.1 必选属性：`cudaLaunchAttributeClusterDimension`

- 三维 `clusterDim`：**最小/强制** Cluster 维。
- `gridDim` 各维必须可被对应 `clusterDim` 整除。
- 语义对齐模式 A 的 `__cluster_dims__`，但可 **per-launch** 修改。

#### B.2 可选属性（CC ≥ 10.0）：`cudaLaunchAttributePreferredClusterDimension`

| 规则 | 说明 |
|------|------|
| 前置条件 | 必须同时给出最小维（`__cluster_dims__` 或 `ClusterDimension`） |
| Preferred | 必须是最小维的**整数倍** |
| 运行保证 | 所有 Block 至少以最小维成 Cluster；**尽可能**用 Preferred 维，但不保证全部 |
| Kernel 要求 | 必须对「最小维 **或** Preferred 维」两种 Cluster 大小都正确 |

```cpp
// 示意：最小 2×1×1，偏好 4×1×1
attribute[0].id = cudaLaunchAttributeClusterDimension;
attribute[0].val.clusterDim = {2, 1, 1};
attribute[1].id = cudaLaunchAttributePreferredClusterDimension;
attribute[1].val.clusterDim = {4, 1, 1};
config.numAttrs = 2;
```

| 点 | 说明 |
|----|------|
| 优点 | 运行时可调；可配合 Preferred 做软目标 |
| 缺点 | Host 配置更重；需自行保证整除与占用 |

---

### 模式 C — `__block_size__`（Blocks as Clusters）

**适用**：希望 Host 直接按 **Cluster 个数** 描述 Grid，而把「每 Block 线程数 / 每 Cluster Block 数」收进 kernel 属性。

```cpp
// 第一元组：blockDim；第二元组：每 Cluster 的 Block 维
__block_size__((1024, 1, 1), (2, 2, 2))
__global__ void foo();

// <<<>>> 第一参 = Cluster 数 → 8×8×8 个 Cluster
// 等价于 16×16×16 个 Thread Block（因每 Cluster 为 2×2×2）
foo<<<dim3(8, 8, 8)>>>();
```

对比模式 A（同形状）：

```cpp
__cluster_dims__((2, 2, 2)) __global__ void foo_a();
// 第一参仍是 Block 数
foo_a<<<dim3(16, 16, 16), dim3(1024, 1, 1)>>>();
```

| 点 | 说明 |
|----|------|
| 第二元组缺省 | 视为 `(1,1,1)`（即每 Cluster 1 个 Block） |
| Stream | `<<<numClusters, 1, 0, stream>>>`（第二、三参固定约定；其它值 UB） |
| 互斥 | 不可与带第二维的 `__cluster_dims__` 同时使用 |
| 优点 | Grid 语义直接是「多少 Cluster」 |
| 缺点 | Launch 约定与经典 `<<<grid, block>>>` 不同，易混用 |

---

## 3. 模式对照（速查）

```
模式 A:  __cluster_dims__(Cx,Cy,Cz)
         kernel <<< numBlocks, threadsPerBlock >>>
         cluster 数 = numBlocks / clusterDim   （隐式）

模式 B:  cudaLaunchKernelEx
           + ClusterDimension[=最小维]
           + [PreferredClusterDimension]       （CC≥10）
         gridDim 仍是 Block 数

模式 C:  __block_size__((Bx,By,Bz), (Cx,Cy,Cz))
         kernel <<< numClusters >>>
         Block 总数 = numClusters * clusterDim （隐式）
```

| 维度 | A 编译期 | B 运行时 | C Blocks-as-Clusters |
|------|----------|----------|----------------------|
| 改 Cluster 大小 | 否 | 是 | 否（编译期） |
| Preferred 软目标 | — | 是（CC≥10） | — |
| Host 第一参 | Block | Block | **Cluster** |
| 典型场景 | 固定 DSMEM/同步拓扑 | 按数据选 Cluster | 以 Cluster 为调度单元建模 |

---

## 4. 与 AscendC 对照入口

NV 侧 **Cluster = 如何把 Block 共调度/切分到 GPC（一组 SM）**；编程主轴仍是 **Grid → Block（→ Thread）**。

AscendC 在不同架构语义下对齐方式见：

- [AscendC Cluster 模式定义](../ascendc/cluster.md)
  - **SIMT**：与 NV 一致，主轴 `grid → block`，Cluster 表示 **core 切分/共调度边界**
  - **SIMD**：显式分层 `cluster → blockNum → block`

---

## 5. 参考

| 文档 | 内容 |
|------|------|
| CUDA C++ Programming Guide § Thread Block Clusters | 层次、DSMEM、`cluster.sync` |
| Advanced Host Programming § Launching Clusters | 模式 B / Preferred / Blocks as Clusters |
| Hopper Tuning Guide | 可移植大小 8、非可移植 16、占用建议 |
| 本仓库 [L2 Persistent](a.md) | 同属 Host/Stream 控制面；与 Cluster 正交 |
