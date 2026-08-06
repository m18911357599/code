# HCOMM CCU 底层指令格式分析

> 基于 [cann/hcomm](https://gitcode.com/cann/hcomm) 源码与文档整理。  
> 主要定义位置：
> - `src/base_comm/resources/ccu/ccu_microcode/ccu_microcode_v1.h`
> - `src/base_comm/resources/ccu/ccu_microcode/ccu_microcode.cc`（V1 编解码）
> - `src/base_comm/resources/ccu/ccu_microcode/ccu_microcode_v2.cc`（V2 编解码）
> - legacy 镜像：`src/legacy/ascend950/unified_platform/ccu/ccu_microcode/`

---

## 1. 设计定位

CCU（Collective Communication Unit，集合通信加速单元）是位于 IO Die 的专用集合通信协处理器。Thread 抽象为 **Mission**，执行模型为：

1. Host 将 CCU 可识别指令序列下发到 CCU 指令空间，并提交 CCU Kernel 任务；
2. 调度器将 Kernel 发到 CCU 执行；
3. CCU 按指令流执行，经 URMA 完成数据搬运。

上层 CCU 数据面 API（Load/Store/Write/Loop/Notify 等）在注册阶段被记录，结束注册时由 Representation Translator **翻译为定长微码指令** `CcuInstr`。

仓库中并存两套指令集编码：

| 版本 | 用途概要 |
|------|----------|
| **CcuV1** | MS（Memory Slice）中转为主的传输/同步/Reduce 指令 |
| **CcuV2** | 寄存器/Xn 语义更完整，含算术逻辑、Load/Store、统一 TransMem、Wait/Fence |

---

## 2. 公共指令外壳（定长 32 Byte）

每条底层指令固定为 **32 字节**：

```
+--------+------------------------------+
| Header | Payload (union, 30 bytes)    |
| 2 Byte | 15 × uint16 等效字段         |
+--------+------------------------------+
```

### 2.1 指令头 `CcuInstrHeader`（16 bit）

```c
union CcuInstrHeader {
    struct {
        uint16_t code : 11;   // bit[10:0]  操作码（同类内编号）
        uint16_t type : 4;    // bit[14:11] 指令大类
        uint16_t reserved : 1;// bit[15]
    };
    uint16_t header;
};
```

### 2.2 指令大类 `type`

| type | 值 | 含义 |
|------|----|------|
| LOAD_TYPE | `0x0` | 加载 / 寄存器运算 / Load-Store |
| CTRL_TYPE | `0x1` | 流程控制（Loop / CKE / Jump / Wait / Fence） |
| TRANS_TYPE | `0x2` | 数据搬运与跨端同步 |
| REDUCE_TYPE | `0x3` | MS 上 Reduce（Add/Max/Min） |

### 2.3 整条指令载体

```c
struct CcuInstr {
    CcuInstrHeader header;
    union {
        CcuV1::CcuMicroCodeV1 v1;
        CcuV2::CcuMicroCodeV2 v2;
    };
};
```

Payload 按 `#pragma pack(1)` 打包，各子结构体统一占满 30 字节（不足用 `reserved` 填充）。

---

## 3. 关键概念（字段语义）

| 符号 | 含义 |
|------|------|
| **GSA** | Global/Shared Address 类地址寄存器（V1 主用） |
| **Xn / X** | 通用寄存器（存立即数、偏移、长度、循环参数等） |
| **MS** | Memory Slice，CCU 片上高速缓冲（默认 4KB，interleave 8） |
| **CKE** | Check/Complete Event 类同步事件位（set/wait/clear） |
| **Channel** | 通信通道号（跨 rank 传输） |
| **Token** | V2 中与地址配对的访存 token（`xstId`/`xdtId` 等） |
| **SQE Args** | 任务 SQE 参数区入口，用于把 Host 参数装入寄存器 |

多数传输类指令尾部都带 `setCKEId/setCKEMask`、`waitCKEId/waitCKEMask`，用于与事件位做依赖同步。

---

## 4. CcuV1 指令表

### 4.1 LOAD_TYPE (`type=0x0`)

| code | 名称 | Payload 结构 | 字段要点 |
|------|------|--------------|----------|
| `0x0` | LoadSqeArgsToGSA | `CcuInstrLoadSqeArgsToGSA` | `gsaId`, `sqeArgsId` |
| `0x1` | LoadSqeArgsToXn | `CcuInstrLoadSqeArgsToXn` | `xnId`, `sqeArgsId` |
| `0x2` | LoadImdToGSA | `CcuInstrLoadImdToGSA` | `gsaId`, `immediate(u64)` |
| `0x3` | LoadImdToXn | `CcuInstrLoadImdToXn` | `xnId`, `immediate(u64)`, `secFlag` |
| `0x4` | LoadGSAXn | `CcuInstrLoadGSAXn` | `gsAdId`, `gsAmId`, `xnId`（地址运算） |
| `0x5` | LoadGSAGSA | `CcuInstrLoadGSAGSA` | `gsAdId`, `gsAmId`, `gsAnId` |
| `0x6` | LoadXX | `CcuInstrLoadXX` | `xdId`, `xmId`, `xnId`（Xn 间运算/赋值） |

### 4.2 CTRL_TYPE (`type=0x1`)

| code | 名称 | Payload 结构 | 字段要点 |
|------|------|--------------|----------|
| `0x0` | Loop | `CcuInstrLoop` | `startInstrId`, `endInstrId`, `xnId`（打包 LoopCtxId/Offset/IterNum） |
| `0x1` | LoopGroup | `CcuInstrLoopGroup` | `startLoopInstrId`, `xnId`, `xmId`, `highPerfModeEn` |
| `0x2` | SetCKE | `CcuInstrSetCKE` | `clearType`, `setCKEId/Mask`, `waitCKEId/Mask` |
| `0x4` | ClearCKE | `CcuInstrClearCKE` | `clearType`, `clearCKEId/Mask`, `waitCKEId/Mask` |
| `0x5` | Jmp | `CcuInstrJmp` | `dstInstrXnId`, `conditionXnId`, `expectData(u64)` |

### 4.3 TRANS_TYPE (`type=0x2`) — 数据搬运

MS 相关通用尾部字段：`clearType`, `lengthEn`, `setCKE*`, `waitCKE*`。

| code | 名称 | 方向 | 关键字段 |
|------|------|------|----------|
| `0x0` | TransLocMemToLocMS | Loc Mem → Loc MS | `locMSId`, `locGSAId`, `locXnId`, `lengthXnId`, `channelId` |
| `0x1` | TransRmtMemToLocMS | Rmt Mem → Loc MS | `locMSId`, `rmtGSAId`, `rmtXnId`, `lengthXnId`, `channelId` |
| `0x2` | TransLocMSToLocMem | Loc MS → Loc Mem | `locGSAId`, `locXnId`, `locMSId`, ... |
| `0x3` | TransLocMSToRmtMem | Loc MS → Rmt Mem | `rmtGSAId`, `rmtXnId`, `locMSId`, ... |
| `0x4` | TransRmtMSToLocMem | Rmt MS → Loc Mem | `locGSAId`, `locXnId`, `rmtMSId`, ... |
| `0x5` | TransLocMSToLocMS | Loc MS → Loc MS | `dstMSId`, `srcMSId`, `lengthXnId`, `channelId` |
| `0x6` | TransRmtMSToLocMS | Rmt MS → Loc MS | `locMSId`, `rmtMSId`, ... |
| `0x7` | TransLocMSToRmtMS | Loc MS → Rmt MS | `rmtMSId`, `locMSId`, 另含 `setRmtCKEId/Mask` |
| `0x8` | TransRmtMemToLocMem | Rmt Mem → Loc Mem | GSA/Xn 对 + 可选 Reduce：`reduceEn`, `reduceDataType`, `reduceOpCode`, `udfType` |
| `0x9` | TransLocMemToRmtMem | Loc Mem → Rmt Mem | 同上，可带 Reduce |
| `0xa` | TransLocMemToLocMem | Loc Mem → Loc Mem | `dstGSAId/dstXnId`, `srcGSAId/srcXnId`, `lengthXnId`, `channelId` |

### 4.4 TRANS_TYPE — 同步（复用 TRANS 大类）

| code | 名称 | Payload | 要点 |
|------|------|---------|------|
| `0xb` | SyncCKE | `CcuInstrSyncCKE` | 远端 CKE ↔ 本端 CKE |
| `0xc` | SyncGSA | `CcuInstrSyncGSA` | 同步 GSA，可 `setRmtCKE*` |
| `0xd` | SyncXn | `CcuInstrSyncXn` | 同步 Xn 寄存器 |

### 4.5 REDUCE_TYPE (`type=0x3`)

| code | 名称 | Payload | 要点 |
|------|------|---------|------|
| `0x0` | Add | `CcuInstrAdd` | `msId[8]`, `count(2..8)`, `castEn`, `dataType`, `XnIdLength` |
| `0x1` | Max | `CcuInstrMax` | 同上，无 castEn |
| `0x2` | Min | `CcuInstrMin` | 同上 |

Reduce 常量：`CCU_REDUCE_SUM/MAX/MIN`，MS 参与数量范围 `CCU_REDUCE_MIN_MS=2` ~ `CCU_REDUCE_MAX_MS=8`。

---

## 5. CcuV2 指令表

V2 强化 Xn 寄存器与统一内存传输（`TransMem`），并增加算术/逻辑与 Wait/Fence。

### 5.1 LOAD_TYPE (`type=0x0`)

| code | 名称 | Payload | 说明 |
|------|------|---------|------|
| `0x1` | LoadSqeArgsToX | `CcuInstrLoadSqeArgsToX` | SQE 参数 → Xn，带 setCKE |
| `0x2` | LoadImdToX | `CcuInstrLoadImdToX` | 立即数 → Xn |
| `0x6` | LoadX | `CcuInstrLoadStoreX` | Xn 间 load/store 风格搬运（`xd/xs` + offset mode） |
| `0x7` | StoreX | `CcuInstrLoadStoreX` | 同上结构，store 语义 |
| `0x8` | ClearX | `CcuInstrClearX` | 清零/按 mode 清除 Xn |
| `0x9` | Nop | `CcuInstrNop` | 空操作 |
| `0xA` | Load | `CcuInstrLoad` | 从 Mem 加载到寄存器，含 cache hint |
| `0xB` | Store | `CcuInstrStore` | 寄存器写回 Mem；可 HSCB/broadcast |
| `0xD` | Add | `CcuInstrOperator` | `xd = xn + xm`（或立即数变体） |
| `0xE` | Sub | `CcuInstrOperator` | 减法 |
| `0xF` | Mul | `CcuInstrOperator` | 乘法 |
| `0x10` | And | `CcuInstrOperator` | 按位与 |
| `0x11` | Or | `CcuInstrOperator` | 按位或 |
| `0x12` | Not | `CcuInstrOperator` | 按位非 |
| `0x13` | Xor | `CcuInstrOperator` | 按位异或 |
| `0x14` | Shl | `CcuInstrOperator` | 左移（`shiftType`） |
| `0x15` | Shr | `CcuInstrOperator` | 右移（`shiftType`） |
| `0x16` | Popcnt | `CcuInstrOperator` | 弹计数 |

`CcuInstrOperator` 公共字段：`xdId`, `xnId`, `xmId`, `parMode`, `shiftType`, `setCKEId/Mask`。

### 5.2 CTRL_TYPE (`type=0x1`)

| code | 名称 | Payload | 要点 |
|------|------|---------|------|
| `0x0` | Loop | `CcuInstrLoop` | `start/endInstrId`, `xmId=IterNum`, `xnId=Offset`, `xpId=LoopCtxId`, `wishCKEBit`, `mode` |
| `0x1` | LoopGroup | `CcuInstrLoopGroup` | `startLoopInstrId`, `xnId`（Extend/Repeat/LoopNum）, `xmId`（资源 offset）, `xpId`（xnOffset） |
| `0x2` | SetCKBit | `CcuInstrSetCKE` | 另含 `userData(u64)` |
| `0x4` | ClearCKBit | `CcuInstrClearCKE` | clear + wait |
| `0x5` | Jmp | `CcuInstrJmp` | `expectedXnId`, `conditionXnId`, `relTarInstrXnId`, `conditionType`, `jumpMode` |
| `0x7` | Wait | `CcuInstrWait` | 条件等待：`conditionXnId` vs `expectedXnId` |
| `0x8` | Fence | `CcuInstrFence` | 屏障（payload 全 reserved） |

### 5.3 TRANS_TYPE (`type=0x2`)

| code | 名称 | Payload | 要点 |
|------|------|---------|------|
| `0x0` | TransLocMemToLocMS | `CcuInstrTransLocMemToLocMS` | `msId`, `xs/xst`, `xl`, `xo`, cache hint |
| `0x2` | TransLocMSToLocMem | `CcuInstrTransLocMSToLocMem` | 反向 |
| `0x5` | TransLocMSToLocMS | `CcuInstrTransLocMSToLocMS` | MS↔MS |
| `0x6` | TransLocMemToLocMem | `CcuInstrTransLocMemToLocMem` | 本端 Mem↔Mem，可指定 `usedMSId`/`msNum` |
| `0x10` | TransMem | `CcuInstrTransMem` | **统一远端传输**（见下） |
| `0xD` | SyncWtX | `CcuInstrSyncWtX` | 固定 8B 写同步 Xn，可带 notify |
| `0xE` | SyncAtX | `CcuInstrSyncAtX` | atomic store add 方式同步 Xn |

#### `CcuInstrTransMem`（V2 核心远程搬运格式）

关键字段：

- 地址：`xdId/xdtId`（dst）、`xsId/xstId`（src）、`xlId`（len）、`xcId`（channel）
- Notify/Atomic：`xnId/xntId`, `value(u32)`
- Reduce：`udfType`, `reduceDataType`, `reduceOpCode`, `udfEnable`
- DMA opcode（`dmaOpCode`，写入 WQE）：

| dmaOpCode | 语义 |
|-----------|------|
| `0x0` | Send |
| `0x1` | Send with immediate |
| `0x3` | Write |
| `0x5` | Write with Notify |
| `0x6` | Read |
| `0x70` | Write with atomic store add |

其它控制位：`order`, `fence`, `cqe`, `nf`(No Fragment), `splitMode`, `se`, `rmtJettyType`, `src_mode/dst_mode`, `msIdMode`, `targetHint`。

### 5.4 REDUCE_TYPE (`type=0x3`)

| code | 名称 | Payload |
|------|------|---------|
| `0x0` | Reduce Add | `CcuInstrReduce`（`castEn` 有效） |
| `0x1` | Reduce Max | `CcuInstrReduce` |
| `0x2` | Reduce Min | `CcuInstrReduce` |

---

## 6. 典型字段布局示意（V1 传输类）

以 `TransLocMemToLocMS` 为例（Header 之后 30B）：

```
[locMSId][locGSAId][locXnId][lengthXnId][channelId]
[reserved × 5]
[flags: clearType|lengthEn|reserved]
[setCKEId][setCKEMask][waitCKEId][waitCKEMask]
```

V2 同类指令则改为 `xsId/xstId/xlId/xoId + cache hint + setCKE*`，远端统一走 `TransMem`。

---

## 7. 与上层 API 的对应关系（概览）

| 上层 CCU API 分类 | 主要落到的底层 type |
|-------------------|---------------------|
| Variable / Address / LoadArg / Load / Store | LOAD |
| Loop / LoopGroup / IF-WHILE / Func | CTRL（Loop/Jmp/Wait 等） |
| LocalCopy / Read / Write / ReadReduce / WriteReduce | TRANS（V1 MS 路径或 V2 TransMem） |
| Event / Notify / LocalNotify | TRANS Sync* 或 CTRL Set/Clear CKE |
| LocalReduce | REDUCE |

上层接口文档见：`docs/zh/api_ref/comm_opdev/data_plane_api/ccu/`。

---

## 8. 源码索引

| 内容 | 路径 |
|------|------|
| 指令结构 / Header / V1+V2 payload | `src/base_comm/resources/ccu/ccu_microcode/ccu_microcode_v1.h` |
| V1 opcode 常量与编码 | `.../ccu_microcode.cc` 或 legacy `ccu_microcode.cpp` |
| V2 opcode 常量与编码 | `.../ccu_microcode_v2.cc` |
| 高层 API → 指令翻译 | `src/base_comm/resources/ccu/ccu_representation/` |
| 指令下发 / sizeof(CcuInstr) | `src/base_comm/resources/ccu/ccu_kernel/ccu_kernel_mgr.cc` |
| 架构简述 | `docs/zh/architecture/architecture-brief.md` §2.4.2 |

---

## 9. 小结

1. **定长 ISA**：每条指令 32B = 2B Header（`type[4]+code[11]`）+ 30B Payload。
2. **四大类**：Load / Ctrl / Trans / Reduce。
3. **双版本**：V1 以 GSA+MS 中转为主；V2 以 Xn + 统一 `TransMem`(URMA/WQE) 为主，并扩展 ALU 与 Wait/Fence。
4. **同步贯穿指令尾**：大量指令通过 CKE set/wait（及远端 Sync）表达依赖，而非仅靠独立同步指令。
