#include <cstddef>
#include <fstream>
#include <iomanip>
#include <iostream>

int main(int argc, char **argv) {
  if (argc != 3) {
    std::cerr << "usage: embed_fatbin <input> <output>\n";
    return 1;
  }

  std::ifstream input(argv[1], std::ios::binary);
  std::ofstream output(argv[2], std::ios::binary | std::ios::trunc);
  if (!input || !output) {
    std::cerr << "failed to open fatbin input or generated source\n";
    return 1;
  }

  output << "#include <cstddef>\n\n"
            "alignas(8) extern const unsigned char "
            "lupine_smemcpy_fatbin[] = {\n";
  output << std::hex << std::setfill('0');
  std::size_t offset = 0;
  for (unsigned char byte = 0;
       input.read(reinterpret_cast<char *>(&byte), sizeof(byte)); ++offset) {
    if (offset % 16 == 0) {
      output << "  ";
    }
    output << "0x" << std::setw(2) << static_cast<unsigned int>(byte) << ',';
    output << (offset % 16 == 15 ? '\n' : ' ');
  }
  if (offset % 16 != 0) {
    output << '\n';
  }
  output << "};\n"
            "extern const std::size_t lupine_smemcpy_fatbin_size = "
            "sizeof(lupine_smemcpy_fatbin);\n";
  return input.eof() && output ? 0 : 1;
}
