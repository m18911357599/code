/**
 * C++ CLI for CCU V1 assembler — same commands as C ccu_v1_asm.
 *
 * Commands: assemble|as, disassemble|dis, verify, vasm|assemble-var, verify-vasm
 */
#include "ccu/v1/assembler.hpp"
#include "ccu/v1/vasm.hpp"

#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <string>
#include <string_view>
#include <vector>

namespace fs = std::filesystem;
using namespace ccu::v1;

static void usage(const char *argv0)
{
    std::cerr
        << "Usage:\n"
        << "  " << argv0 << " assemble    <in.s> -o <out.bin>\n"
        << "  " << argv0 << " disassemble <in.bin> -o <out.s>\n"
        << "  " << argv0 << " verify      <in.s> [-w workdir]\n"
        << "  " << argv0 << " vasm        <in.s> -o <out.bin> [-m out.meta.json] [--lowered out.s]\n"
        << "  " << argv0 << " verify-vasm <in.s> [-w workdir]\n"
        << "Aliases: as, dis, assemble-var\n";
}

static fs::path defaultWorkDir(const fs::path &in, const char *suffix)
{
    auto dir = in.parent_path();
    if (dir.empty()) {
        dir = ".";
    }
    return dir / suffix;
}

static int cmdAssemble(const fs::path &in, const fs::path &out)
{
    auto text = Io::readText(in);
    auto prog = Program::assemble(text);
    auto bin = prog.toBinary();
    Io::writeBinary(out, bin);
    std::cout << "assembled " << prog.size() << " instructions -> " << out.string() << " (" << bin.size()
              << " bytes)\n";
    return 0;
}

static int cmdDisassemble(const fs::path &in, const fs::path &out)
{
    auto bin = Io::readBinary(in);
    auto prog = Program::fromBinary(bin);
    auto text = prog.disassemble();
    Io::writeText(out, text);
    std::cout << "disassembled " << prog.size() << " instructions -> " << out.string() << "\n";
    return 0;
}

static int cmdVerify(const fs::path &in, fs::path work)
{
    fs::create_directories(work);
    auto stem = in.stem().string();
    auto binPath = work / (stem + ".bin");
    auto disPath = work / (stem + ".dis.s");
    auto rebinPath = work / (stem + ".re.bin");

    auto text = Io::readText(in);
    auto prog = Program::assemble(text);
    auto bin = prog.toBinary();
    Io::writeBinary(binPath, bin);

    auto fromBin = Program::fromBinary(bin);
    Io::writeText(disPath, fromBin.disassemble());

    auto fromDis = Program::assemble(Io::readText(disPath));
    auto rebin = fromDis.toBinary();
    Io::writeBinary(rebinPath, rebin);

    bool ok = prog.semanticEq(fromDis) && bin == rebin;
    if (!ok) {
        std::cerr << "FAIL: semantic/binary round-trip mismatch\n";
        return 1;
    }
    std::cout << "OK: " << prog.size() << " instructions\n"
              << "  source     : " << in << "\n"
              << "  binary     : " << binPath << " (" << bin.size() << " bytes)\n"
              << "  disasm     : " << disPath << "\n"
              << "  semantic   : source == disasm\n"
              << "  binary     : assemble(source) == assemble(disasm)\n";
    return 0;
}

static int cmdVasm(const fs::path &in, const fs::path &out, const fs::path *meta, const fs::path *lowered)
{
    auto text = Io::readText(in);
    auto vr = VarAssembler::assemble(text);
    auto bin = vr.toBinary();
    Io::writeBinary(out, bin);
    if (meta) {
        vr.writeMetainfo(*meta);
    }
    if (lowered) {
        Io::writeText(*lowered, vr.loweredAsm());
    }
    std::cout << "vasm: " << vr.instrCount() << " instructions, " << vr.varCount() << " variables -> " << out
              << "\n";
    if (meta) {
        std::cout << "  metainfo  : " << *meta << "\n";
    }
    for (int t = 0; t < kResCount; ++t) {
        auto peak = vr.peakUsed(static_cast<ResType>(t));
        if (peak) {
            std::cout << "  " << resTypeName(static_cast<ResType>(t)) << " peak: " << peak << " / "
                      << vr.config().limits[t] << "\n";
        }
    }
    return 0;
}

static int cmdVerifyVasm(const fs::path &in, fs::path work)
{
    fs::create_directories(work);
    auto stem = in.stem().string();
    auto binPath = work / (stem + ".bin");
    auto metaPath = work / (stem + ".meta.json");
    auto lowPath = work / (stem + ".lowered.s");
    auto rebinPath = work / (stem + ".re.bin");

    if (cmdVasm(in, binPath, &metaPath, &lowPath) != 0) {
        return 2;
    }
    if (cmdAssemble(lowPath, rebinPath) != 0) {
        return 2;
    }
    auto a = Io::readBinary(binPath);
    auto b = Io::readBinary(rebinPath);
    if (a != b) {
        std::cerr << "FAIL: vasm binary != assemble(lowered)\n";
        return 1;
    }
    std::cout << "OK vasm verify: binary == assemble(lowered), metainfo=" << metaPath << "\n";
    return 0;
}

int main(int argc, char **argv)
{
    try {
        if (argc < 2) {
            usage(argv[0]);
            return 2;
        }
        std::string_view cmd = argv[1];
        std::vector<std::string_view> args;
        for (int i = 2; i < argc; ++i) {
            args.emplace_back(argv[i]);
        }

        auto takeFlag = [&](std::string_view flag) -> const char * {
            for (std::size_t i = 0; i + 1 < args.size(); ++i) {
                if (args[i] == flag) {
                    return args[i + 1].data();
                }
            }
            return nullptr;
        };
        auto positional = [&]() -> std::vector<std::string_view> {
            std::vector<std::string_view> pos;
            for (std::size_t i = 0; i < args.size(); ++i) {
                if (args[i].size() && args[i][0] == '-') {
                    if (i + 1 < args.size()) {
                        ++i;
                    }
                    continue;
                }
                pos.push_back(args[i]);
            }
            return pos;
        };

        if (cmd == "assemble" || cmd == "as") {
            auto pos = positional();
            auto out = takeFlag("-o");
            if (!out) {
                out = takeFlag("--output");
            }
            if (pos.size() != 1 || !out) {
                usage(argv[0]);
                return 2;
            }
            return cmdAssemble(std::string(pos[0]), out);
        }
        if (cmd == "disassemble" || cmd == "dis") {
            auto pos = positional();
            auto out = takeFlag("-o");
            if (!out) {
                out = takeFlag("--output");
            }
            if (pos.size() != 1 || !out) {
                usage(argv[0]);
                return 2;
            }
            return cmdDisassemble(std::string(pos[0]), out);
        }
        if (cmd == "verify") {
            auto pos = positional();
            auto work = takeFlag("-w");
            if (!work) {
                work = takeFlag("--work-dir");
            }
            if (pos.size() != 1) {
                usage(argv[0]);
                return 2;
            }
            const std::string in_str(pos[0]);
            const fs::path in(in_str);
            fs::path w = work ? fs::path(work) : defaultWorkDir(in, ".ccu_v1_verify_cpp");
            return cmdVerify(in, w);
        }
        if (cmd == "vasm" || cmd == "assemble-var") {
            auto pos = positional();
            auto out = takeFlag("-o");
            if (!out) {
                out = takeFlag("--output");
            }
            auto meta = takeFlag("-m");
            if (!meta) {
                meta = takeFlag("--metainfo");
            }
            auto lowered = takeFlag("--lowered");
            if (pos.size() != 1 || !out) {
                usage(argv[0]);
                return 2;
            }
            fs::path metaPath, lowPath;
            const fs::path *mp = nullptr;
            const fs::path *lp = nullptr;
            if (meta) {
                metaPath = meta;
                mp = &metaPath;
            }
            if (lowered) {
                lowPath = lowered;
                lp = &lowPath;
            }
            return cmdVasm(std::string(pos[0]), out, mp, lp);
        }
        if (cmd == "verify-vasm") {
            auto pos = positional();
            auto work = takeFlag("-w");
            if (!work) {
                work = takeFlag("--work-dir");
            }
            if (pos.size() != 1) {
                usage(argv[0]);
                return 2;
            }
            const std::string in_str(pos[0]);
            const fs::path in(in_str);
            fs::path w = work ? fs::path(work) : defaultWorkDir(in, ".ccu_v1_vasm_verify_cpp");
            return cmdVerifyVasm(in, w);
        }
        usage(argv[0]);
        return 2;
    } catch (const std::exception &e) {
        std::cerr << "error: " << e.what() << "\n";
        return 2;
    }
}
