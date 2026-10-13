#include "macos_library_profile.h"

#include <cstdlib>
#include <dlfcn.h>
#include <fstream>
#include <iostream>
#include <vector>

static void check(bool passed, const char *message) {
  if (!passed) {
    std::cerr << message << '\n';
    std::abort();
  }
}

int main(int argc, char **argv) {
  check(argc == 5, "expected mode, profile, and two fixture paths");
  auto args = lupine_macos_profile::command_line();
  std::string expected;
  for (int i = 0; i < argc; ++i) {
    expected.append(argv[i]);
    expected.push_back('\0');
  }
  check(args == expected, "command line must retain argument boundaries");
  void *first = dlopen(argv[3], RTLD_NOW | RTLD_LOCAL);
  check(first, "load first fixture dylib");
  using bytes_fn = const unsigned char *(*)();
  const auto *bytes =
      reinterpret_cast<bytes_fn>(dlsym(first, "profile_bytes"))();
  std::string id, other_id;
  uint64_t offset = 0, other_offset = 0;
  check(lupine_macos_profile::image(bytes, 8, id, offset) == bytes,
        "locate embedded bytes");
  void *second = dlopen(argv[4], RTLD_NOW | RTLD_LOCAL);
  check(second, "load another dylib after registry initialization");
  const auto *other =
      reinterpret_cast<bytes_fn>(dlsym(second, "profile_bytes"))();
  check(lupine_macos_profile::image(other, 8, other_id, other_offset) == other,
        "locate second binary");
  check(id.size() == 16 && id != other_id, "UUID identifies the linked binary");
  check(lupine_macos_profile::image(nullptr, 8, id, offset) == bytes,
        "replay file offset using UUID");
  check(lupine_macos_profile::image(nullptr, 8, other_id, offset) != bytes,
        "another UUID cannot resolve the original binary");
  std::string unknown(16, '\0');
  check(!lupine_macos_profile::image(nullptr, 8, unknown, offset),
        "reject unknown UUID");
  std::vector<unsigned char> heap(bytes, bytes + 8);
  check(!lupine_macos_profile::image(heap.data(), 8, unknown, offset),
        "do not profile heap-owned copies");
  const auto *zero =
      reinterpret_cast<bytes_fn>(dlsym(first, "profile_zero_fill"))();
  check(!lupine_macos_profile::image(zero, 4096, unknown, offset),
        "do not profile zero-fill segments");
  check(!lupine_macos_profile::image(bytes, UINT64_MAX, unknown, offset),
        "reject an image extending past its segment");
  // Separate CTest invocations verify reconstruction after another exec/ASLR.
  // Relocate once more after negative checks, which may overwrite offset.
  check(lupine_macos_profile::image(bytes, 8, id, offset) == bytes,
        "locate again");
  if (std::string(argv[1]) == "record") {
    std::ofstream out(argv[2], std::ios::binary);
    out.write(id.data(), 16);
    out.write(reinterpret_cast<const char *>(&offset), sizeof(offset));
    out.close();
    check(bool(out), "write profile");
  } else {
    std::ifstream in(argv[2], std::ios::binary);
    std::string saved_id(16, '\0');
    uint64_t saved_offset = 0;
    in.read(saved_id.data(), 16);
    in.read(reinterpret_cast<char *>(&saved_offset), sizeof(saved_offset));
    check(bool(in), "read profile");
    check(lupine_macos_profile::image(nullptr, 8, saved_id, saved_offset) ==
              bytes,
          "replay profile in a new process");
  }
  dlclose(second);
  check(!lupine_macos_profile::image(nullptr, 8, other_id, other_offset),
        "forget unloaded dylibs");
  dlclose(first);
  check(!lupine_macos_profile::image(nullptr, 8, id, offset),
        "forget original unloaded dylib");
}
