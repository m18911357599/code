#include "ccu/v1/assembler.hpp"

#include <fstream>
#include <iterator>
#include <sstream>

namespace ccu::v1 {

std::string Io::readText(const std::filesystem::path &path)
{
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        throw Error("cannot read " + path.string());
    }
    return std::string(std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>());
}

std::vector<std::uint8_t> Io::readBinary(const std::filesystem::path &path)
{
    auto text = readText(path);
    return std::vector<std::uint8_t>(text.begin(), text.end());
}

void Io::writeBinary(const std::filesystem::path &path, const std::vector<std::uint8_t> &data)
{
    std::ofstream out(path, std::ios::binary);
    if (!out) {
        throw Error("cannot write " + path.string());
    }
    out.write(reinterpret_cast<const char *>(data.data()), static_cast<std::streamsize>(data.size()));
}

void Io::writeText(const std::filesystem::path &path, std::string_view text)
{
    std::ofstream out(path, std::ios::binary);
    if (!out) {
        throw Error("cannot write " + path.string());
    }
    out.write(text.data(), static_cast<std::streamsize>(text.size()));
}

} // namespace ccu::v1
