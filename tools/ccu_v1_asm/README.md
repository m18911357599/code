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
| `disassemble` / `dis` | `.bin` → **C API** 源（`void _entry(void)`） |
| `verify` | 数值汇编往返语义验证 |
| `vasm` / `assemble-var` | **命名变量汇编** → `.bin` + **metainfo** |
| `verify-vasm` | 变量汇编验证（bin == assemble(lowered)） |
| `casm` / `assemble-c` | **C 风格文本**解释 → `.bin`（入口 `_entry` 或 `main`） |
| `verify-casm` | C 风格文本验证 |
| `casm_host` | **原生编译** `loop_main.c`：建上下文 → 调 `_entry` → 写 `.bin` |

```bash
./build/ccu_v1_asm as ../examples/all_opcodes.s -o out.bin
./build/ccu_v1_asm dis out.bin -o out.c          # C API / _entry
./build/ccu_v1_asm casm out.c -o out.re.bin
./build/casm_host -o out.bin                     # 推荐：真正调用 _entry()
```

## C 风格汇编（原生执行）

`loop_main.c` 是真实 C：指令函数往**当前上下文**填 32B 二进制。反汇编器输出同一风格，入口为 `_entry`。

生成流程（`casm_host`）：

1. **创建上下文** `CcuV1CasmCtx`
2. **`ccu_v1_casm_begin`**（调用 `_entry` 之前安装上下文）
3. **调用 `_entry()`** — `loop` / `load_*` / … 经 `ccu_v1_casm_emit` 得到 `ctx->inst`，直接写载荷字段
4. **`ccu_v1_casm_end`**
5. **`ccu_v1_casm_write_file`** — `fwrite` 连续 32B 指令到文件

无文本拼装 / `emit_scalars`；原生路径零助记符查找。
```c
#include "ccu_v1_casm_api.h"

void _entry(void)
{
    loop(0, 10, 11);   /* 向上下文填写 LOOP 载荷二进制 */
    load_imd_to_xn(6, 0x1000, 0);
    add(MS(0,1,2,0,0,0,0,0), 3, 1, 4, 13, 0, 1, 0x1, 2, 0x2);
}
```

构建：`loop_main.c` 编译为 `_entry`，链入 `casm_host`。

完整 29 条 opcode：[`examples/loop_main.c`](examples/loop_main.c)（与 `all_opcodes.s` 二进制 `cmp` 一致）。

API：`ccu_v1_casm.h` / `ccu_v1_casm_api.h`（指令接口均为 header `static inline`）。
`ccu_v1_format_instr_c` / `ccu_v1_disassemble_c_api` 生成上述调用形态。

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
c/          C 核心 + CLI + casm_host
examples/   all_opcodes.s  vars_reuse.s  loop_main.c
```
