#ifndef LUPINE_FATBIN_H
#define LUPINE_FATBIN_H

#include <cstdint>

// CUDA fatbin container layout, verified against nvcc 13.1 output. A fatbinC
// wrapper (__fatBinC_Wrapper_t) points at the fatbin proper: an outer header
// followed by `files_size` bytes of member entries, each a member header
// followed by `size` payload bytes. A compressed member carries a compression
// flag (0x2000 for LZ4 under --compress-mode=speed, 0x8000 for zstd under
// default/size/balance modes) and nonzero compressed_size/uncompressed_size
// fields, while an uncompressed member (--compress-mode=none, or SASS members
// that fatbinary left raw) has flags 0x11 and zero in both size fields.

struct lupine_fatbin_wrapper {
  uint32_t magic;
  uint32_t version;
  const void *data;
  const void *filename_or_fatbins;
};

struct lupine_fatbin_header {
  uint32_t magic;
  uint16_t version;
  uint16_t header_size;
  uint64_t files_size;
};

struct lupine_fatbin_member_header {
  uint16_t kind; // 1 = PTX, 2 = ELF (cubin)
  uint16_t version;
  uint32_t header_size;
  uint64_t size;            // stored payload bytes following this header
  uint32_t compressed_size; // unpadded compressed bytes; 0 if uncompressed
  uint32_t unknown;
  uint16_t minor;
  uint16_t major;
  uint32_t arch;
  uint32_t name_offset;
  uint32_t name_size;
  uint64_t flags;
  uint64_t reserved;
  uint64_t uncompressed_size; // 0 if uncompressed
};

static_assert(sizeof(lupine_fatbin_member_header) == 64,
              "fatbin member header layout must match CUDA's");

static constexpr uint32_t LUPINE_FATBINC_MAGIC = 0x466243b1;
static constexpr uint32_t LUPINE_FATBIN_MAGIC = 0xba55ed50;
static constexpr uint64_t LUPINE_FATBIN_MEMBER_LZ4 = 0x2000;
static constexpr uint64_t LUPINE_FATBIN_MEMBER_ZSTD = 0x8000;

#endif
