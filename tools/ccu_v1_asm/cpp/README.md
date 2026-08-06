# CCU V1 C++ 汇编器

C++ 风格 API / CLI，**语义与二进制编码与 C 实现一致**（编译并链接 `c/src/{isa,asm,vasm}.c`）。  
文档按 C 代码结构对照给出。

## 构建与测试

```bash
cd tools/ccu_v1_asm/cpp
make -j && make test
# 产物: build/ccu_v1_asm_cpp
```

命令与 C 版相同：`assemble|as`、`disassemble|dis`、`verify`、`vasm`、`verify-vasm`。

## 与 C 的对应关系（文档按 C 代码）

详见 [docs/API_MAPPING.md](docs/API_MAPPING.md)。摘要：

| C (`c/include`, `c/src`) | C++ (`cpp/include/ccu/v1`) |
|--------------------------|----------------------------|
| `CcuV1Instr` / `ccu_v1_isa.h` | `ccu::v1::Instr` / `isa.hpp` |
| `ccu_v1_lookup_mnemonic` | `Isa::lookupMnemonic` |
| `CcuV1Program` + `ccu_v1_assemble_text` | `Program::assemble` |
| `ccu_v1_program_to_binary` | `Program::toBinary` |
| `ccu_v1_format_instr` | `Program::formatInstr` / `disassemble` |
| `CcuV1VasmResult` + `ccu_v1_vasm_assemble` | `VarAssembler::assemble` |
| `ccu_v1_vasm_write_metainfo` | `VarAssembler::metainfoJson` / `writeMetainfo` |
| `c/src/cli.c` | `cpp/src/main.cpp` |

语法、位置操作数、变量声明 / ID 复用 / metainfo 与 [C README](../README.md) 相同；操作数顺序以 `c/src/isa.c` 的 `k_ops_*` 为准。

## 目录

```
cpp/
  include/ccu/v1/{isa,assembler,vasm}.hpp
  src/{io,vasm,main}.cpp
  docs/API_MAPPING.md
  Makefile
```
