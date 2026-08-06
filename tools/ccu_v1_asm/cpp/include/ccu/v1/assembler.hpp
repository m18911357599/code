#pragma once

/**
 * C++ assembler / disassembler.
 *
 * Wraps C API in ccu_v1_asm.h with RAII and std::string / std::vector.
 * Semantics and binary encoding are identical to the C tool.
 */

#include "ccu/v1/isa.hpp"

#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

extern "C" {
#include "ccu_v1_asm.h"
}

namespace ccu::v1 {

class Error : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

/**
 * Corresponds to C: CcuV1Program + ccu_v1_assemble_text / from_binary / to_binary / format.
 */
class Program {
public:
    Program() { ccu_v1_program_init(&raw_); }
    ~Program() { ccu_v1_program_free(&raw_); }

    Program(const Program &) = delete;
    Program &operator=(const Program &) = delete;

    Program(Program &&other) noexcept : raw_(other.raw_) {
        ccu_v1_program_init(&other.raw_);
    }
    Program &operator=(Program &&other) noexcept {
        if (this != &other) {
            ccu_v1_program_free(&raw_);
            raw_ = other.raw_;
            ccu_v1_program_init(&other.raw_);
        }
        return *this;
    }

    /** C: ccu_v1_assemble_text */
    static Program assemble(std::string_view text) {
        Program p;
        char err[256];
        if (ccu_v1_assemble_text(text.data(), text.size(), &p.raw_, err, sizeof(err)) != 0) {
            throw Error(err);
        }
        return p;
    }

    /** C: ccu_v1_program_from_binary */
    static Program fromBinary(const std::uint8_t *data, std::size_t len) {
        Program p;
        char err[256];
        if (ccu_v1_program_from_binary(data, len, &p.raw_, err, sizeof(err)) != 0) {
            throw Error(err);
        }
        return p;
    }

    static Program fromBinary(const std::vector<std::uint8_t> &bin) {
        return fromBinary(bin.data(), bin.size());
    }

    /** C: ccu_v1_program_to_binary */
    std::vector<std::uint8_t> toBinary() const {
        std::uint8_t *data = nullptr;
        std::size_t len = 0;
        if (ccu_v1_program_to_binary(&raw_, &data, &len) != 0) {
            throw Error("toBinary failed");
        }
        std::vector<std::uint8_t> out(data, data + len);
        free(data);
        return out;
    }

    /** C: ccu_v1_format_instr (positional operands) */
    static std::string formatInstr(const Instr &instr) {
        char buf[1024];
        if (ccu_v1_format_instr(&instr, buf, sizeof(buf)) < 0) {
            throw Error("formatInstr failed");
        }
        return std::string(buf);
    }

    /** C: ccu_v1_disassemble_program → string */
    std::string disassemble() const {
        std::string out;
        out.reserve(raw_.count * 64);
        for (std::size_t i = 0; i < raw_.count; ++i) {
            out += formatInstr(raw_.items[i]);
            out.push_back('\n');
        }
        return out;
    }

    /** C: ccu_v1_program_semantic_eq (0 equal) */
    bool semanticEq(const Program &other) const {
        return ccu_v1_program_semantic_eq(&raw_, &other.raw_) == 0;
    }

    std::size_t size() const noexcept { return raw_.count; }
    const Instr &operator[](std::size_t i) const { return raw_.items[i]; }
    Instr &operator[](std::size_t i) { return raw_.items[i]; }

    const CcuV1Program &raw() const noexcept { return raw_; }
    CcuV1Program &raw() noexcept { return raw_; }

private:
    CcuV1Program raw_{};
};

/** File helpers (C CLI read_file / write_file equivalents). */
class Io {
public:
    static std::string readText(const std::filesystem::path &path);
    static std::vector<std::uint8_t> readBinary(const std::filesystem::path &path);
    static void writeBinary(const std::filesystem::path &path, const std::vector<std::uint8_t> &data);
    static void writeText(const std::filesystem::path &path, std::string_view text);
};

} // namespace ccu::v1
