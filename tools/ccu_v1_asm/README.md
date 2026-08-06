# CCU V1 汇编器 / 反汇编器（C 高性能实现）

基于 [cann/hcomm](https://gitcode.com/cann/hcomm) 的 **CcuV1** 定长微码（32B/`CcuInstr`）。

- **主实现**：`c/`（C11，`-O3 -flto`，packed struct 零拷贝编解码）
- **参考实现**：仓库内 Python 模块（便于对照；日常请用 C 版）

## 构建

```bash
cd tools/ccu_v1_asm/c
make -j
# 产物: build/ccu_v1_asm
make test   # 全 opcode 往返 + cmp 二进制
```

## 使用

```bash
./build/ccu_v1_asm assemble   ../examples/all_opcodes.s -o out.bin
./build/ccu_v1_asm disassemble out.bin -o out.dis.s
./build/ccu_v1_asm verify     ../examples/all_opcodes.s
```

`verify`：源 `.s` → `.bin` → `.dis.s`，再汇编；要求与源**语义一致**且二进制逐字节相同。

## 性能要点

| 手段 | 说明 |
|------|------|
| packed `CcuV1Instr` | 与硬件/软件布局一致，memcpy 即编码 |
| FNV-1a 助记符哈希 | O(1) mnemonic 查找 |
| 单遍解析 | 原地扫描 `name=value`，无正则 |
| mmap 读文件 | Linux 下减少 syscall |
| `-O3 -flto -march=native` | 默认发布优化 |

## 汇编语法

与先前约定相同（命名操作数，`#` 注释）。示例见 `examples/all_opcodes.s`（29 条 opcode 全覆盖）。

## 目录

```
c/
  include/ccu_v1_isa.h   # 指令布局 / opcode 表
  include/ccu_v1_asm.h   # 汇编 API
  src/isa.c asm.c cli.c
  Makefile
examples/all_opcodes.s
```
