# CCU V1 汇编器 / 反汇编器

基于 [cann/hcomm](https://gitcode.com/cann/hcomm) 的 **CcuV1** 定长微码（32B/`CcuInstr`）。

高性能 C 实现：[`c/`](c/)（解析 / 编码 / 变量分配 / CLI）。

## 快速开始

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
./build/ccu_v1_asm as ../examples/all_opcodes.s -o out.bin
./build/ccu_v1_asm vasm ../examples/vars_reuse.s -o out.bin -m out.meta.json
```

## 汇编语法（位置操作数）

```asm
LOAD_IMD_TO_XN offset, 0x1000, 0
```

操作数顺序见 `c/src/isa.c` 的 `k_ops_*`。仍兼容 `name=value`。

## 变量汇编

```asm
.xn offset
LOAD_IMD_TO_XN offset, 0x1000, 0
```

标识符自动分配/复用物理 id；metainfo 描述 `name/type/id/live_*`。

## 目录

```
c/          C 核心 + CLI
examples/   all_opcodes.s  vars_reuse.s
```
