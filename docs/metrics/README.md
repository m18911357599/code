# 度量目录

后续度量文档一律以 **Markdown（`.md`）** 存储，不再使用 YAML / XLSX 作为目录源。

三级路径：**度量分类** → **度量项** → **度量子项**。

| 层级 | 数量 | 文件 |
|---|---|---|
| 度量分类 | 6 | 本目录 6 个 `F*.md` |
| 度量项 | 36 | 各类文件中的 `##` 节 |
| 度量子项 | 66 | 各度量项下的子节 / 表 |

总则（评分模型、三档权重、公式、亲和规则）：[下一代调试调优度量体系](../下一代调试调优度量体系.md)

## 分类

| 分类 | 负载轴 | 项 | 子项 | 文档 |
|---|---|---|---|---|
| [F1 互联带宽](F1-互联带宽.md) | 卡间 / 核间 / 引擎间交换是否墙 | 6 | 14 | `F1-互联带宽.md` |
| [F2 数据精度](F2-数据精度.md) | dtype / 量化 / 转换 / 舍入 | 7 | 16 | `F2-数据精度.md` |
| [F3 Batch 利用率](F3-Batch利用率.md) | 小 batch / 短序列时阵列与核是否吃不满 | 7 | 8 | `F3-Batch利用率.md` |
| [F4 每 token FLOPs](F4-每token-FLOPs.md) | 算术强度 / 每 token 运算量 | 9 | 19 | `F4-每token-FLOPs.md` |
| [F5 显存容量](F5-显存容量.md) | 工作集 / workspace / 片上缓冲能否放下 | 3 | 4 | `F5-显存容量.md` |
| [F6 显存敏感](F6-显存敏感.md) | 对带宽、层级、对齐、一致性敏感 | 4 | 5 | `F6-显存敏感.md` |

## 分类 → 度量项 → 度量子项

### [F1 互联带宽](F1-互联带宽.md)

- [E1 通算 overlap 线性可叠加度](F1-互联带宽.md#F1-E1)
  - [`η_ov` overlap 场景占比](F1-互联带宽.md#F1-E1-eta_ov)
  - [`η_lin` 叠加效率](F1-互联带宽.md#F1-E1-eta_lin)
  - [`L_blk` 手工同步阻塞代码行](F1-互联带宽.md#F1-E1-L_blk)
- [A3 同步暴露](F1-互联带宽.md#F1-A3)
  - [`n_intra` 核内同步调用数](F1-互联带宽.md#F1-A3-n_intra)
  - [`n_inter` 核间同步调用数](F1-互联带宽.md#F1-A3-n_inter)
  - [`η_xcore` 自动核间同步覆盖率](F1-互联带宽.md#F1-A3-eta_xcore)
- [A9 同步流水复杂度](F1-互联带宽.md#F1-A9)
  - [`n_pipe` 流水级数](F1-互联带宽.md#F1-A9-n_pipe)
  - [`n_ord` 序域个数](F1-互联带宽.md#F1-A9-n_ord)
- [A13 自动同步消除率](F1-互联带宽.md#F1-A13)
  - [`η_sync` 自动同步消除率](F1-互联带宽.md#F1-A13-eta_sync)
- [A11 ID 残留对消](F1-互联带宽.md#F1-A11)
  - [`L_cancel` ID 对消代码行](F1-互联带宽.md#F1-A11-L_cancel)
  - [`n_resid` 对消后残留次数](F1-互联带宽.md#F1-A11-n_resid)
- [C3 故障隔离与恢复](F1-互联带宽.md#F1-C3)
  - [`g_rst` 最小复位范围](F1-互联带宽.md#F1-C3-g_rst)
  - [`η_rst` 局部复位成功率](F1-互联带宽.md#F1-C3-eta_rst)
  - [`MTTR` 平均恢复时间](F1-互联带宽.md#F1-C3-MTTR)

### [F2 数据精度](F2-数据精度.md)

- [B3 数值/类型支持度](F2-数据精度.md#F2-B3)
  - [`h̄` 类型转换平均跳数](F2-数据精度.md#F2-B3-h_bar)
  - [`η_type` 类型完备度](F2-数据精度.md#F2-B3-eta_type)
- [B6 随路搬运](F2-数据精度.md#F2-B6)
  - [`n_fix` 随路互斥拆分步数](F2-数据精度.md#F2-B6-n_fix)
  - [`n_fmt` 格式转换调用数](F2-数据精度.md#F2-B6-n_fmt)
- [A15 自动格式转换覆盖率](F2-数据精度.md#F2-A15)
  - [`η_fmt` 自动格式转换覆盖率](F2-数据精度.md#F2-A15-eta_fmt)
- [C2 确定性与重放](F2-数据精度.md#F2-C2)
  - [`η_det` 同输入结果一致率](F2-数据精度.md#F2-C2-eta_det)
  - [`η_rep` 执行顺序可重放率](F2-数据精度.md#F2-C2-eta_rep)
- [C4 错误遥测](F2-数据精度.md#F2-C4)
  - [`η_ex` 异常类型覆盖率](F2-数据精度.md#F2-C4-eta_ex)
  - [`η_det` 异常检出率](F2-数据精度.md#F2-C4-eta_hit_fault)
  - [`ρ_sdc` 静默数据错误率](F2-数据精度.md#F2-C4-rho_sdc)
  - [`η_root` 首因识别率](F2-数据精度.md#F2-C4-eta_root)
  - [`η_snap` 现场保存成功率](F2-数据精度.md#F2-C4-eta_snap)
- [A8 多版本一致性](F2-数据精度.md#F2-A8)
  - [`η_ver` 跨版本行为一致率](F2-数据精度.md#F2-A8-eta_ver)
  - [`n_drift` 跨版本语义漂移数](F2-数据精度.md#F2-A8-n_drift)
- [D2 API 破坏性变更](F2-数据精度.md#F2-D2)
  - [`n_brk` 每代 API 破坏性变更数](F2-数据精度.md#F2-D2-n_brk)
  - [`η_det` 破坏性变更可检测率](F2-数据精度.md#F2-D2-eta_brk)

### [F3 Batch 利用率](F3-Batch利用率.md)

- [B1 搬运复杂度（tiling）](F3-Batch利用率.md#F3-B1)
  - [`n_til` Tiling 参数个数](F3-Batch利用率.md#F3-B1-n_til)
- [A14 自动 tiling 命中率](F3-Batch利用率.md#F3-A14)
  - [`η_til` 自动 tiling 命中率](F3-Batch利用率.md#F3-A14-eta_til)
- [A7 编译友好屏蔽率](F3-Batch利用率.md#F3-A7)
  - [`η_sh` 编译友好屏蔽率](F3-Batch利用率.md#F3-A7-eta_sh)
- [D4 形态配比保持度](F3-Batch利用率.md#F3-D4)
  - [`Δ_ratio` 形态比变更幅度](F3-Batch利用率.md#F3-D4-delta_ratio)
  - [`ρ_rw` 算子重编写比例](F3-Batch利用率.md#F3-D4-rho_rw)
- [D6 配比变化重写量](F3-Batch利用率.md#F3-D6)
  - [`L_rw` 配比变化重写代码行](F3-Batch利用率.md#F3-D6-L_rw)
- [D7 核映射自适应度](F3-Batch利用率.md#F3-D7)
  - [`η_map` 自动核映射命中率](F3-Batch利用率.md#F3-D7-eta_nmap)
- [A12 高层 API 替代覆盖率](F3-Batch利用率.md#F3-A12)
  - [`η_hl` 高层 API 覆盖率](F3-Batch利用率.md#F3-A12-eta_hl)

### [F4 每 token FLOPs](F4-每token-FLOPs.md)

- [B2 计算复杂度](F4-每token-FLOPs.md#F4-B2)
  - [`ρ_ins` 指令粒度比](F4-每token-FLOPs.md#F4-B2-rho_ins)
- [B7 计算行为表达](F4-每token-FLOPs.md#F4-B7)
  - [`n_beh` 显式饱和/舍入/特殊函数调用数](F4-每token-FLOPs.md#F4-B7-n_beh)
- [A2 多流水暴露](F4-每token-FLOPs.md#F4-A2)
  - [`r_sw` Pipeline 切换手动同步代码占比](F4-每token-FLOPs.md#F4-A2-r_sw)
  - [`η_auto` 自动同步覆盖率](F4-每token-FLOPs.md#F4-A2-eta_auto)
  - [`n_VF` VF 表达数](F4-每token-FLOPs.md#F4-A2-n_VF)
- [A4 API-指令耦合](F4-每token-FLOPs.md#F4-A4)
  - [`p̄` 入参平均数](F4-每token-FLOPs.md#F4-A4-p_bar)
  - [`n_reg` 寄存器配置数](F4-每token-FLOPs.md#F4-A4-n_reg)
  - [`Δ_sig` 跨版本签名差异数](F4-每token-FLOPs.md#F4-A4-delta_sig)
- [D5 算力配比冲击度](F4-每token-FLOPs.md#F4-D5)
  - [`Δ_pow` 配比变化幅度](F4-每token-FLOPs.md#F4-D5-delta_pow)
  - [`ρ_retune` tiling/流水重调比例](F4-每token-FLOPs.md#F4-D5-rho_retune)
- [D1 源码跨代可移植性](F4-每token-FLOPs.md#F4-D1)
  - [`η_port` 跨代零改动编译通过率](F4-每token-FLOPs.md#F4-D1-eta_port)
  - [`η_perf` 跨代性能达成率](F4-每token-FLOPs.md#F4-D1-eta_perf)
- [C1 性能度量覆盖](F4-每token-FLOPs.md#F4-C1)
  - [`η_evt` 关键性能事件覆盖率](F4-每token-FLOPs.md#F4-C1-eta_evt)
  - [`ε_met` 指标相对误差](F4-每token-FLOPs.md#F4-C1-eps_met)
  - [`ρ_loss` 数据丢失率](F4-每token-FLOPs.md#F4-C1-rho_loss)
  - [`η_stall` Stall 原因解释覆盖率](F4-每token-FLOPs.md#F4-C1-eta_stall)
- [C5 源码关联深度](F4-每token-FLOPs.md#F4-C5)
  - [`η_map` 模型→算子→kernel→指令映射完备度](F4-每token-FLOPs.md#F4-C5-eta_map)
  - [`η_hit` 硬件事件→可修改代码命中率](F4-每token-FLOPs.md#F4-C5-eta_hit)
- [A16 用户实际感知暴露度](F4-每token-FLOPs.md#F4-A16)
  - [`n_perc` 用户须感知暴露项数](F4-每token-FLOPs.md#F4-A16-n_perc)

### [F5 显存容量](F5-显存容量.md)

- [A5 资源/约束暴露](F5-显存容量.md#F5-A5)
  - [`L_mm` 内存管理代码行](F5-显存容量.md#F5-A5-L_mm)
  - [`L_id` ID 管理代码行](F5-显存容量.md#F5-A5-L_id)
- [A10 设备初始化](F5-显存容量.md#F5-A10)
  - [`L_init` 设备初始化代码行](F5-显存容量.md#F5-A10-L_init)
- [D3 二进制兼容性](F5-显存容量.md#F5-D3)
  - [`η_load` 跨代可加载率](F5-显存容量.md#F5-D3-eta_load)

### [F6 显存敏感](F6-显存敏感.md)

- [A1 内存层级暴露](F6-显存敏感.md#F6-A1)
  - [`ΔK` Tiling Key 增长量](F6-显存敏感.md#F6-A1-deltaK)
  - [`r_mv` 搬运代码行占比](F6-显存敏感.md#F6-A1-r_mv)
- [A6 硬件约束暴露](F6-显存敏感.md#F6-A6)
  - [`L_dcci` DCCI 代码行](F6-显存敏感.md#F6-A6-L_dcci)
- [B4 地址计算](F6-显存敏感.md#F6-B4)
  - [`L_addr` 地址计算代码行](F6-显存敏感.md#F6-B4-L_addr)
- [B5 地址对齐](F6-显存敏感.md#F6-B5)
  - [`n_pad` 对齐边界分支数](F6-显存敏感.md#F6-B5-n_pad)
