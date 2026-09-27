#include "fatbin_filter.h"

#include <cstdlib>
#include <iostream>
#include <limits>

static void check(bool passed, const char *message) {
  if (!passed) {
    std::cerr << message << '\n';
    std::abort();
  }
}

struct member {
  uint16_t kind;
  uint32_t arch;
  uint32_t header_size = 64;
};

static std::vector<unsigned char>
fatbin(std::initializer_list<member> members) {
  std::vector<unsigned char> image(sizeof(lupine_fatbin_header));
  for (const auto &m : members) {
    lupine_fatbin_entry entry = {};
    entry.kind = m.kind;
    entry.version = LUPINE_FATBIN_ENTRY_VERSION;
    entry.header_size = m.header_size;
    entry.payload_size = 17;
    entry.arch = m.arch;
    const size_t at = image.size();
    image.resize(at + entry.header_size + entry.payload_size,
                 static_cast<unsigned char>(m.arch));
    std::memcpy(image.data() + at, &entry, sizeof(entry));
  }
  lupine_fatbin_header header{LUPINE_FATBIN_MAGIC, LUPINE_FATBIN_VERSION,
                              sizeof(lupine_fatbin_header),
                              image.size() - sizeof(lupine_fatbin_header)};
  std::memcpy(image.data(), &header, sizeof(header));
  return image;
}

int main() {
  const auto input = fatbin({{2, 75},
                             {2, 80},
                             {2, 86, 96},
                             {2, 90},
                             {2, 100},
                             {1, 70, 80},
                             {1, 120}});
  auto filter = [&](const std::vector<unsigned> &devices) {
    return lupine_filter_fatbin(input.data(), input.size(), devices);
  };
  check(filter({89}) == fatbin({{2, 80}, {2, 86, 96}, {1, 70, 80}}),
        "Ada keeps compatible cubins and PTX, with entry bytes unchanged");
  check(filter({75, 89}) ==
            fatbin({{2, 75}, {2, 80}, {2, 86, 96}, {1, 70, 80}}),
        "heterogeneous server keeps the union of device capabilities");
  check(filter({100}) == fatbin({{2, 100}, {1, 70, 80}}),
        "three-digit architectures are compared by major and minor");
  check(filter({}).empty() && filter({89, 0}).empty(),
        "missing or incomplete device inventory disables filtering");
  check(filter({75, 89, 90, 100, 120}).empty(),
        "do not copy an image whose entries are all needed");
  const auto incompatible = fatbin({{2, 90}, {1, 120}});
  check(lupine_filter_fatbin(incompatible.data(), incompatible.size(), {89})
            .empty(),
        "all-incompatible image must reach the driver unchanged");

  std::vector<unsigned char> unaligned(1, 0);
  unaligned.insert(unaligned.end(), input.begin(), input.end());
  check(lupine_filter_fatbin(unaligned.data() + 1, input.size(), {89}) ==
            filter({89}),
        "unaligned mapped images are supported");
  for (size_t size = 0; size < input.size(); ++size) {
    check(lupine_filter_fatbin(input.data(), size, {89}).empty(),
          "every truncated container is left unchanged");
  }
  auto reject = [](std::vector<unsigned char> image) {
    check(lupine_filter_fatbin(image.data(), image.size(), {89}).empty(),
          "unrecognized or malformed layout must be forwarded unchanged");
  };
  auto unknown = input;
  unknown[4] = 2;
  reject(unknown);
  unknown = input;
  unknown[6] = 24;
  reject(unknown);
  unknown = input;
  unknown[16] = 3;
  reject(unknown);
  unknown = input;
  unknown[18] = 2;
  reject(unknown);
  unknown = input;
  uint64_t overflow = std::numeric_limits<uint64_t>::max();
  std::memcpy(unknown.data() + 24, &overflow, sizeof(overflow));
  reject(unknown);
  unknown = input;
  uint32_t zero = 0;
  std::memcpy(unknown.data() + 44, &zero, sizeof(zero));
  reject(unknown);
  reject(std::vector<unsigned char>(128, 0));
  check(lupine_filter_fatbin(nullptr, 100, {89}).empty(), "null image");
  std::cout << "fatbin_filter_test: PASS\n";
}
