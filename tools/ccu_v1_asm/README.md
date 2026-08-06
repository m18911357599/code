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

## 变量汇编语法

```asm
# 声明（可省略，首次使用时按操作数字段自动推断类型）
.xn  offset
.gsa src
.ms  slice0
.cke done
.ch  peer
.var xn pinned_reg = 3   # 固定物理 id（pinned，不参与复用抢占该 id 的冲突区间）

LOAD_IMD_TO_XN xn=offset, imm=0x1000, sec=0
TRANS_LOC_MEM_TO_LOC_MS ms=slice0, gsa=src, xn=offset, len_xn=offset, ch=peer, \
  clear=0, len_en=1, set_id=done, set_mask=0x1, wait_id=0, wait_mask=0
```

- 资源操作数字段中的**标识符** → 变量（自动分配物理 id）
- **数字字面量** → 直接编码（不进入分配器）
- 字段→类型：`xn/xd/xm/len_xn/...`→xn，`gsa/gsad/...`→gsa，`ms/...`→ms，`set_id/wait_id/...`→cke，`ch`→ch，`sqe`→sqe

### ID 生命周期与复用

对每个变量统计 `[live_start, live_end]`（首次引用～末次引用，指令下标）。按类型做 **线性扫描分配**：区间结束后 id 可被后续变量复用。

### metainfo（JSON）

描述 `name / type / id / pinned / live_start / live_end / use_count`，以及各资源 `limit` 与 `peak_used`。示例见 `make test` 生成的 `build/vars.meta.json`。

## 目录

```
c/include/ccu_v1_isa.h ccu_v1_asm.h ccu_v1_vasm.h
c/src/isa.c asm.c vasm.c cli.c
examples/all_opcodes.s vars_reuse.s
python_ref/   # 可选参考实现
```
