#pragma once

/**
 * C++ variable-aware assembler.
 *
 * Wraps C API in ccu_v1_vasm.h. Named resources, live-range analysis,
 * linear-scan ID reuse, metainfo JSON — same as C `vasm`.
 */

#include "ccu/v1/assembler.hpp"

#include <array>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <string>
#include <string_view>
#include <vector>

extern "C" {
#include "ccu_v1_vasm.h"
}

namespace ccu::v1 {

using ResType = CcuV1ResType;
using VasmConfig = CcuV1VasmConfig;
using VarInfo = CcuV1VarInfo;

inline constexpr int kResCount = CCU_V1_RES_COUNT;

/** Corresponds to C: ccu_v1_res_type_name / ccu_v1_res_type_parse */
inline const char *resTypeName(ResType t) noexcept { return ccu_v1_res_type_name(t); }

inline ResType parseResType(std::string_view s) {
    std::string tmp(s);
    ResType t{};
    if (ccu_v1_res_type_parse(tmp.c_str(), &t) != 0) {
        throw Error(std::string("unknown resource type: ") + tmp);
    }
    return t;
}

/**
 * Corresponds to C: CcuV1VasmResult + ccu_v1_vasm_assemble / write_metainfo.
 */
class VarAssembler {
public:
    VarAssembler() { ccu_v1_vasm_result_init(&raw_); }
    ~VarAssembler() { ccu_v1_vasm_result_free(&raw_); }

    VarAssembler(const VarAssembler &) = delete;
    VarAssembler &operator=(const VarAssembler &) = delete;

    VarAssembler(VarAssembler &&other) noexcept : raw_(other.raw_) {
        ccu_v1_vasm_result_init(&other.raw_);
    }
    VarAssembler &operator=(VarAssembler &&other) noexcept {
        if (this != &other) {
            ccu_v1_vasm_result_free(&raw_);
            raw_ = other.raw_;
            ccu_v1_vasm_result_init(&other.raw_);
        }
        return *this;
    }

    /** C: ccu_v1_vasm_config_default then optional overrides */
    static VasmConfig defaultConfig() {
        VasmConfig cfg{};
        ccu_v1_vasm_config_default(&cfg);
        return cfg;
    }

    /** C: ccu_v1_vasm_assemble */
    static VarAssembler assemble(std::string_view text, const VasmConfig *cfg = nullptr) {
        VarAssembler v;
        char err[256];
        if (ccu_v1_vasm_assemble(text.data(), text.size(), cfg, &v.raw_, err, sizeof(err)) != 0) {
            throw Error(err);
        }
        return v;
    }

    std::vector<std::uint8_t> toBinary() const {
        std::uint8_t *data = nullptr;
        std::size_t len = 0;
        if (ccu_v1_program_to_binary(&raw_.program, &data, &len) != 0) {
            throw Error("toBinary failed");
        }
        std::vector<std::uint8_t> out(data, data + len);
        free(data);
        return out;
    }

    std::string loweredAsm() const {
        return raw_.lowered_asm ? std::string(raw_.lowered_asm) : std::string{};
    }

    /** C: ccu_v1_vasm_write_metainfo */
    std::string metainfoJson() const;
    void writeMetainfo(const std::filesystem::path &path) const;

    std::size_t instrCount() const noexcept { return raw_.instr_count; }
    std::size_t varCount() const noexcept { return raw_.var_count; }

    const VarInfo &var(std::size_t i) const { return raw_.vars[i]; }

    std::uint16_t peakUsed(ResType t) const noexcept {
        return raw_.used_peak[static_cast<int>(t)];
    }

    const VasmConfig &config() const noexcept { return raw_.cfg; }
    const CcuV1VasmResult &raw() const noexcept { return raw_; }

private:
    CcuV1VasmResult raw_{};
};

} // namespace ccu::v1
