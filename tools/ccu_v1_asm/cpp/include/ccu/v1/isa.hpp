#pragma once

/**
 * C++ façade over the C CCU V1 ISA (ccu_v1_isa.h / isa.c).
 * Binary layout is identical to the C `CcuV1Instr` (32 bytes).
 */

#include <cstdint>
#include <string>
#include <string_view>

extern "C" {
#include "ccu_v1_isa.h"
}

namespace ccu::v1 {

inline constexpr std::size_t kInstrSize = CCU_V1_INSTR_SIZE;
inline constexpr std::size_t kOpcodeCount = CCU_V1_OP_COUNT;

using Instr = CcuV1Instr;
using OpcodeId = CcuV1OpcodeId;
using OpcodeDesc = CcuV1OpcodeDesc;

/** Corresponds to C: ccu_v1_opcodes / ccu_v1_lookup_mnemonic / ccu_v1_lookup_opcode */
class Isa {
public:
    static const OpcodeDesc *opcodes() noexcept { return ccu_v1_opcodes(); }
    static std::size_t opcodeCount() noexcept { return ccu_v1_opcode_count(); }

    static const OpcodeDesc *lookupMnemonic(std::string_view mnem) {
        // C API expects NUL-terminated string
        std::string tmp(mnem);
        return ccu_v1_lookup_mnemonic(tmp.c_str());
    }

    static const OpcodeDesc *lookupOpcode(std::uint8_t type, std::uint16_t code) noexcept {
        return ccu_v1_lookup_opcode(type, code);
    }

    static std::uint16_t makeHeader(std::uint8_t type, std::uint16_t code) noexcept {
        return ccu_v1_make_header(type, code);
    }
};

} // namespace ccu::v1
