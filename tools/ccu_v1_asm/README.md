# CCU V1 汇编器 / 反汇编器

基于 [cann/hcomm](https://gitcode.com/cann/hcomm) 的 **CcuV1** 定长微码（32B/`CcuInstr`）。

高性能 C 实现：[`c/`](c/)（解析 / 编码 / 变量分配 / C 风格前端 / CLI）。

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
| `casm` / `assemble-c` | **C 风格源文件** → `.bin`（上下文内建填二进制） |
| `verify-casm` | C 风格汇编验证 |

```bash
./build/ccu_v1_asm as ../examples/all_opcodes.s -o out.bin
./build/ccu_v1_asm vasm ../examples/vars_reuse.s -o out.bin -m out.meta.json
./build/ccu_v1_asm casm ../examples/loop_main.c -o out.bin --lowered out.s
```

## C 风格汇编（推荐书写方式）

源文件像 C 一样写 `main`，汇编器维护 **上下文**；内建函数（如 `loop`）直接往上下文里填入 32B 指令二进制：

```c
void main()
{
    loop(0, 10, 11);              /* → LOOP start=0 end=10 xn=11 */
    load_imd_to_xn(6, 0x1000, 0);
}
```

完整覆盖全部 opcode 的例子：[`examples/loop_main.c`](examples/loop_main.c)（与 `all_opcodes.s` 操作数一致，`make test` 会 `cmp` 二者产出的 `.bin`）。
- 入口：`void main()` / `int main()`（可写 `void` 形参）
- 语句：`name(args...);`，`name` 为 ISA 助记符的小写形式（`LOOP` → `loop`）
- `loop(start, end, xn)`：在上下文中填写 `CcuV1Loop` 载荷二进制（见 `ccu_v1_casm_loop`）
- 其它 opcode 同样以函数调用形式发出
- 列表操作数用 `{...}` 或 `[...]`（如 reduce 的 `ms`）

API：`c/include/ccu_v1_casm.h`（`CcuV1CasmCtx` / `ccu_v1_casm_loop` / `ccu_v1_casm_compile`）。

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
examples/   all_opcodes.s  vars_reuse.s  loop_main.c
```
