# CCU V1：C → C++ API 对照

本文档按 **C 代码**（`tools/ccu_v1_asm/c/`）结构说明 C++ 封装（`tools/ccu_v1_asm/cpp/`）。  
C++ 不重新实现编码逻辑，而是 **RAII / `std::string` / 异常** 封装同一套 C 核心（`cpp/Makefile` 将 `c/src/*.c` 编入 `cpp/build/c_*.o`），因此：

- 指令二进制布局 = `CcuV1Instr`（32B）
- 汇编语义 / 位置操作数 / 变量分配 = C 版行为
- `make test` 中会 `cmp` C++ 与 C 产出的 `.bin`

---

## 1. ISA（`c/include/ccu_v1_isa.h` + `c/src/isa.c`）

### C

```c
typedef struct { ... } CcuV1Instr;           /* 32B packed */
typedef struct {
    const char *mnemonic;
    uint8_t type;
    uint16_t code;
    CcuV1OpcodeId id;
    const char *const *operands;             /* 位置操作数表 */
    int nop;
} CcuV1OpcodeDesc;

const CcuV1OpcodeDesc *ccu_v1_opcodes(void);
const CcuV1OpcodeDesc *ccu_v1_lookup_mnemonic(const char *mnem);
const CcuV1OpcodeDesc *ccu_v1_lookup_opcode(uint8_t type, uint16_t code);
uint16_t ccu_v1_make_header(uint8_t type, uint16_t code);
```

位置操作数表：`k_ops_*`（如 `LOAD_IMD_TO_XN` → `xn, imm, sec`）。

### C++（`include/ccu/v1/isa.hpp`）

```cpp
namespace ccu::v1 {
  using Instr = CcuV1Instr;
  using OpcodeDesc = CcuV1OpcodeDesc;
  class Isa {
    static const OpcodeDesc *opcodes();
    static const OpcodeDesc *lookupMnemonic(std::string_view);
    static const OpcodeDesc *lookupOpcode(uint8_t type, uint16_t code);
    static uint16_t makeHeader(uint8_t type, uint16_t code);
  };
}
```

| C | C++ |
|---|-----|
| `CcuV1Instr` | `ccu::v1::Instr` |
| `ccu_v1_opcodes()` | `Isa::opcodes()` |
| `ccu_v1_lookup_mnemonic` | `Isa::lookupMnemonic` |
| `ccu_v1_lookup_opcode` | `Isa::lookupOpcode` |
| `ccu_v1_make_header` | `Isa::makeHeader` |
| `CCU_V1_INSTR_SIZE` | `kInstrSize` |

---

## 2. 数值汇编（`c/include/ccu_v1_asm.h` + `c/src/asm.c`）

### C

```c
typedef struct {
    CcuV1Instr *items;
    size_t count, capacity;
} CcuV1Program;

void ccu_v1_program_init/free(CcuV1Program *);
int  ccu_v1_assemble_text(const char *text, size_t len, CcuV1Program *out, char *errmsg, size_t);
int  ccu_v1_format_instr(const CcuV1Instr *, char *buf, size_t);
int  ccu_v1_disassemble_program(const CcuV1Program *, FILE *);
int  ccu_v1_program_from_binary(const uint8_t *, size_t, CcuV1Program *, char *, size_t);
int  ccu_v1_program_to_binary(const CcuV1Program *, uint8_t **out, size_t *len);
int  ccu_v1_program_semantic_eq(const CcuV1Program *, const CcuV1Program *);
```

默认语法：位置操作数 `MNEMONIC v0, v1, ...`；兼容 `name=value`。

### C++（`include/ccu/v1/assembler.hpp`）

```cpp
namespace ccu::v1 {
  class Program {
    static Program assemble(std::string_view text);          // ccu_v1_assemble_text
    static Program fromBinary(...);                          // ccu_v1_program_from_binary
    std::vector<uint8_t> toBinary() const;                   // ccu_v1_program_to_binary
    static std::string formatInstr(const Instr &);           // ccu_v1_format_instr
    std::string disassemble() const;                         // loop formatInstr
    bool semanticEq(const Program &) const;                  // ccu_v1_program_semantic_eq
    size_t size() const;
    const Instr &operator[](size_t) const;
  };
  class Io { /* readText / writeBinary / ... */ };
}
```

| C | C++ |
|---|-----|
| `ccu_v1_program_init/free` | `Program` 构造 / 析构（RAII） |
| `ccu_v1_assemble_text` | `Program::assemble`（失败抛 `Error`） |
| `ccu_v1_program_from_binary` | `Program::fromBinary` |
| `ccu_v1_program_to_binary` | `Program::toBinary` |
| `ccu_v1_format_instr` | `Program::formatInstr` |
| `ccu_v1_disassemble_program` | `Program::disassemble` |
| `ccu_v1_program_semantic_eq` | `Program::semanticEq` |
| `errmsg[]` 返回 -1 | `throw ccu::v1::Error` |

---

## 3. 变量汇编（`c/include/ccu_v1_vasm.h` + `c/src/vasm.c`）

### C

```c
typedef enum { CCU_V1_RES_XN, GSA, MS, CKE, CH, SQE, COUNT } CcuV1ResType;
typedef struct { uint16_t limits[COUNT]; } CcuV1VasmConfig;
typedef struct {
    char name[64]; CcuV1ResType type; int pinned; int16_t id;
    int live_start, live_end, use_count;
} CcuV1VarInfo;
typedef struct {
    CcuV1VarInfo *vars; size_t var_count;
    CcuV1Program program; char *lowered_asm; size_t instr_count;
    uint16_t used_peak[COUNT]; CcuV1VasmConfig cfg;
} CcuV1VasmResult;

int ccu_v1_vasm_assemble(const char *, size_t, const CcuV1VasmConfig *,
                         CcuV1VasmResult *, char *errmsg, size_t);
int ccu_v1_vasm_write_metainfo(const CcuV1VasmResult *, FILE *);
```

算法（C 实现）：声明 / 自动声明 → 收集 use → `[live_start,live_end]` → 线性扫描分配 → lower 为数值 asm → `ccu_v1_assemble_text`。

### C++（`include/ccu/v1/vasm.hpp`）

```cpp
namespace ccu::v1 {
  using ResType = CcuV1ResType;
  using VarInfo = CcuV1VarInfo;
  using VasmConfig = CcuV1VasmConfig;

  class VarAssembler {
    static VasmConfig defaultConfig();                 // ccu_v1_vasm_config_default
    static VarAssembler assemble(std::string_view, const VasmConfig * = nullptr);
    std::vector<uint8_t> toBinary() const;
    std::string loweredAsm() const;
    std::string metainfoJson() const;                  // ccu_v1_vasm_write_metainfo → string
    void writeMetainfo(const std::filesystem::path &) const;
    size_t instrCount() const;
    size_t varCount() const;
    const VarInfo &var(size_t) const;
    uint16_t peakUsed(ResType) const;
  };
}
```

| C | C++ |
|---|-----|
| `ccu_v1_vasm_result_init/free` | `VarAssembler` RAII |
| `ccu_v1_vasm_assemble` | `VarAssembler::assemble` |
| `result.program` + `to_binary` | `toBinary()` |
| `result.lowered_asm` | `loweredAsm()` |
| `ccu_v1_vasm_write_metainfo` | `metainfoJson` / `writeMetainfo` |
| `ccu_v1_res_type_name` | `resTypeName` |
| `used_peak[t]` | `peakUsed(t)` |

---

## 4. CLI（`c/src/cli.c` → `cpp/src/main.cpp`）

| C 命令 | C++ 命令 | 行为 |
|--------|----------|------|
| `assemble` / `as` | 同左 | 数值 asm → bin |
| `disassemble` / `dis` | 同左 | bin → 位置操作数 asm |
| `verify` | 同左 | 语义 + 再汇编二进制一致 |
| `vasm` / `assemble-var` | 同左 | 变量 asm → bin + metainfo |
| `verify-vasm` | 同左 | vasm bin == assemble(lowered) |

C 可执行文件：`c/build/ccu_v1_asm`  
C++ 可执行文件：`cpp/build/ccu_v1_asm_cpp`

---

## 5. 示例（共享）

与 C 共用：

- `examples/all_opcodes.s` — 全 opcode 位置语法
- `examples/vars_reuse.s` — 变量 + ID 复用

```bash
# C
cd c && ./build/ccu_v1_asm verify ../examples/all_opcodes.s

# C++
cd cpp && ./build/ccu_v1_asm_cpp verify ../examples/all_opcodes.s
```

---

## 6. 链接模型

```
ccu_v1_asm_cpp
  ├── cpp/src/{main,io,vasm}.cpp     # C++ CLI + RAII
  └── c/build/{isa,asm,vasm}.o       # C 核心（编码 / 解析 / 分配）
```

修改指令格式或分配算法时，改 **C 侧**；C++ 仅调整封装与文档对照表。
