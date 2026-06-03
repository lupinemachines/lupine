#ifndef LUPINE_PTX_FATBIN_HPP
#define LUPINE_PTX_FATBIN_HPP

#include <cstdint>

namespace lupine {

constexpr std::uint32_t kFatbinWrapperMagic = 0x466243b1;
constexpr std::uint32_t kFatbinWrapperVersion = 1;
constexpr std::uint32_t kFatbinWrapperLinkVersion = 2;
constexpr std::uint32_t kFatbinMagic = 0xba55ed50;

constexpr std::uint32_t kModuleImageFatbinWrapperV1 = 1;
constexpr std::uint32_t kModuleImageFatbinRaw = 2;
constexpr std::uint32_t kModuleImageFatbinWrapperV2 = 3;

// CUDA exposes this ABI in fatbinary_section.h as __fatBinC_Wrapper_t.
// Keep the layout compatible without depending on CUDA's private type names in
// runtime code.
struct fatbin_wrapper {
  std::uint32_t magic;
  std::uint32_t version;
  const void *data;
  const void *filename_or_fatbins;
};

// Header at the start of the fatbin payload referenced by fatbin_wrapper::data.
struct fatbin_header {
  std::uint32_t magic;
  std::uint16_t version;
  std::uint16_t header_size;
  std::uint64_t files_size;
};

} // namespace lupine

#endif
