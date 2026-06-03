#include "codegen/ptx_fatbin.hpp"

#include <fatbinary_section.h>

#include <cstddef>
#include <cstdint>
#include <iostream>
#include <type_traits>

static_assert(lupine::kFatbinWrapperMagic == FATBINC_MAGIC);
static_assert(lupine::kFatbinWrapperVersion == FATBINC_VERSION);
static_assert(lupine::kFatbinWrapperLinkVersion == FATBINC_LINK_VERSION);

static_assert(std::is_standard_layout<lupine::fatbin_wrapper>::value);
static_assert(sizeof(lupine::fatbin_wrapper) == sizeof(__fatBinC_Wrapper_t));
static_assert(alignof(lupine::fatbin_wrapper) == alignof(__fatBinC_Wrapper_t));
static_assert(offsetof(lupine::fatbin_wrapper, magic) ==
              offsetof(__fatBinC_Wrapper_t, magic));
static_assert(offsetof(lupine::fatbin_wrapper, version) ==
              offsetof(__fatBinC_Wrapper_t, version));
static_assert(offsetof(lupine::fatbin_wrapper, data) ==
              offsetof(__fatBinC_Wrapper_t, data));
static_assert(offsetof(lupine::fatbin_wrapper, filename_or_fatbins) ==
              offsetof(__fatBinC_Wrapper_t, filename_or_fatbins));

static_assert(std::is_standard_layout<lupine::fatbin_header>::value);
static_assert(sizeof(lupine::fatbin_header) == 16);
static_assert(offsetof(lupine::fatbin_header, magic) == 0);
static_assert(offsetof(lupine::fatbin_header, version) == 4);
static_assert(offsetof(lupine::fatbin_header, header_size) == 6);
static_assert(offsetof(lupine::fatbin_header, files_size) == 8);

int main() {
  alignas(8) unsigned long long fatbin[] = {
      0x00100001ba55ed50ULL,
      0x0000000000000030ULL,
  };
  __fatBinC_Wrapper_t nvidia_wrapper = {
      FATBINC_MAGIC,
      FATBINC_VERSION,
      fatbin,
      nullptr,
  };

  const auto *wrapper =
      reinterpret_cast<const lupine::fatbin_wrapper *>(&nvidia_wrapper);
  if (wrapper->magic != lupine::kFatbinWrapperMagic ||
      wrapper->version != lupine::kFatbinWrapperVersion ||
      wrapper->data != fatbin || wrapper->filename_or_fatbins != nullptr) {
    std::cerr << "fatbin wrapper layout does not match CUDA" << std::endl;
    return 1;
  }

  const auto *header =
      reinterpret_cast<const lupine::fatbin_header *>(wrapper->data);
  if (header->magic != lupine::kFatbinMagic || header->version != 1 ||
      header->header_size != sizeof(lupine::fatbin_header) ||
      header->files_size != 0x30) {
    std::cerr << "fatbin payload header layout is not understood" << std::endl;
    return 1;
  }

  return 0;
}
