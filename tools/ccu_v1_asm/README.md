# CCU V1 汇编器 / 反汇编器

基于 [cann/hcomm](https://gitcode.com/cann/hcomm) 的 **CcuV1** 定长微码（32B/`CcuInstr`），实现：

1. **汇编器**：汇编文本 → 二进制
2. **反汇编器**：二进制 → 汇编文本
3. **语义验证**：源文件汇编后再反汇编，与源文件对比功能语义无差异；并校验 `assemble(disasm) == assemble(source)` 二进制一致

## 快速使用

```bash
# 在仓库根目录，或将 tools 加入 PYTHONPATH
cd tools

# 汇编
python3 -m ccu_v1_asm assemble ccu_v1_asm/examples/all_opcodes.s -o /tmp/all_opcodes.bin

# 反汇编
python3 -m ccu_v1_asm disassemble /tmp/all_opcodes.bin -o /tmp/all_opcodes.dis.s

# 一键验证：源 asm → bin → dis.s，语义对比 + 再汇编二进制对比
python3 -m ccu_v1_asm verify ccu_v1_asm/examples/all_opcodes.s
```

## 汇编语法

```
# 注释
MNEMONIC name=value, name2=0x10, ms=[0,1,2,0,0,0,0,0]
```

- 每条指令一行；支持 `#` 行尾注释与空行
- 操作数均为 **命名操作数**（输入顺序无关；反汇编按规范顺序输出）
- 立即数支持十进制与 `0x` 十六进制
- `ms` 为 8 个 Memory Slice id 的列表（不足补 0）

完整 mnemonic 列表见 `isa.py` / 示例 `examples/all_opcodes.s`（覆盖全部 V1 opcode）。

## 二进制布局

与 hcomm `CcuInstr` 一致（little-endian，`#pragma pack(1)`）：

| 偏移 | 大小 | 内容 |
|------|------|------|
| 0 | 2B | Header：`code[10:0] \| type[14:11]` |
| 2 | 30B | Payload（bitfield 按 GCC LSB-first 打包） |

`type`：`0=LOAD, 1=CTRL, 2=TRANS, 3=REDUCE`

## 验证含义

`verify` 做两件事：

1. 解析源 `.s` 与反汇编 `.dis.s` 为语义 IR（mnemonic + 规范操作数），逐条比较
2. 将反汇编结果再汇编，要求二进制与首次汇编完全一致（reserved 位保持 0）

因此允许源文件与反汇编文本在空白、注释、操作数书写顺序上不同，但 **功能语义必须一致**。

## 测试

```bash
cd tools
python3 -m unittest ccu_v1_asm.tests.test_roundtrip -v
```
