#include "ccu/v1/vasm.hpp"

#include <cstdio>
#include <fstream>
#include <sstream>

namespace ccu::v1 {

std::string VarAssembler::metainfoJson() const
{
    char *buf = nullptr;
    std::size_t sz = 0;
    FILE *mem = open_memstream(&buf, &sz);
    if (!mem) {
        throw Error("open_memstream failed");
    }
    if (ccu_v1_vasm_write_metainfo(&raw_, mem) != 0) {
        fclose(mem);
        free(buf);
        throw Error("writeMetainfo failed");
    }
    fclose(mem);
    std::string out(buf, sz);
    free(buf);
    return out;
}

void VarAssembler::writeMetainfo(const std::filesystem::path &path) const
{
    Io::writeText(path, metainfoJson());
}

} // namespace ccu::v1
