# CCU V1 汇编器 / 反汇编器（C 高性能实现）

基于 [cann/hcomm](https://gitcode.com/cann/hcomm) 的 **CcuV1** 定长微码（32B/`CcuInstr`）。

## 构建

```bash
cd tools/ccu_v1_asm/c
make -j && make test
```

## 命令

| 命令 | 作用 |
|------|------|
| `assemble` / `as` | 数值操作数汇编 → `.bin` |
| `disassemble` / `dis` | `.bin` → 汇编 |
| `verify` | 数值汇编往返语义验证 |
| `vasm` / `assemble-var` | **命名变量汇编** → `.bin` + **metainfo** |
| `verify-vasm` | 变量汇编验证（bin == assemble(lowered)） |

```bash
# 数值版
./build/ccu_v1_asm as ../examples/all_opcodes.s -o out.bin
./build/ccu_v1_asm verify ../examples/all_opcodes.s

# 变量版：自动分配/复用 ID，并写 metainfo
./build/ccu_v1_asm vasm ../examples/vars_reuse.s \
  -o out.bin -m out.meta.json --lowered out.lowered.s
./build/ccu_v1_asm verify-vasm ../examples/vars_reuse.s
```

## 汇编语法（位置操作数）

默认去掉 `field=`，按助记符规范顺序书写（更快解析）：

```asm
LOAD_IMD_TO_XN offset, 0x1000, 0
# 等价旧写法: LOAD_IMD_TO_XN xn=offset, imm=0x1000, sec=0
```

## 变量汇编

```asm
.xn  offset
.gsa src
.ms  slice0
.cke done
.ch  peer
.var xn pinned_reg = 3   # 固定物理 id

LOAD_IMD_TO_XN offset, 0x1000, 0
TRANS_LOC_MEM_TO_LOC_MS slice0, src, offset, offset, peer, 0, 1, done, 0x1, 0, 0
```

- 位置上的**标识符** → 变量（自动分配物理 id）；**数字** → 字面量
- 仍兼容 `name=value` 旧语法
- 操作数顺序见 `c/src/isa.c` 的 `k_ops_*`

### ID 生命周期与复用

对每个变量统计 `[live_start, live_end]`（首次引用～末次引用，指令下标）。按类型做 **线性扫描分配**：区间结束后 id 可被后续变量复用。

### metainfo（JSON）

描述 `name / type / id / pinned / live_start / live_end / use_count`，以及各资源 `limit` 与 `peak_used`。示例见 `make test` 生成的 `build/vars.meta.json`。

## 目录

```
c/include/ccu_v1_isa.h ccu_v1_asm.h ccu_v1_vasm.h
c/src/isa.c asm.c vasm.c cli.c
examples/all_opcodes.s vars_reuse.s
```
