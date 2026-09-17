#ifndef LUPINE_FATBIN_H
#define LUPINE_FATBIN_H

#include <cstdint>

// CUDA fatbin container layout: a fatbinC wrapper (__fatBinC_Wrapper_t)
// points at the fatbin proper, an outer header followed by `files_size`
// bytes of member entries.

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

// The member entries follow the outer header back to back, each one a
// `header_size` byte entry header then `payload_size` bytes of image. Only
// this prefix is stable: the entry header is 64 bytes for the pre-10.x
// architectures and grows past it for the newer ones, and PTX entries size it
// differently again, so walk with `header_size` and never with sizeof.
struct lupine_fatbin_entry {
  uint16_t kind;
  uint16_t version;
  uint32_t header_size;
  uint64_t payload_size;
  uint32_t padded_payload_size;
  uint32_t reserved;
  uint16_t code_minor;
  uint16_t code_major;
  // Compute capability as major * 10 + minor, the same encoding sm_90 and
  // compute_90 spell; the arch-conditional and family flags that narrow an
  // image further live in a flags word past this prefix.
  uint32_t arch;
};
static_assert(sizeof(lupine_fatbin_entry) == 32,
              "fatbin entry prefix must match the on-disk layout");

static constexpr uint32_t LUPINE_FATBINC_MAGIC = 0x466243b1;
static constexpr uint32_t LUPINE_FATBIN_MAGIC = 0xba55ed50;
static constexpr uint16_t LUPINE_FATBIN_VERSION = 1;
static constexpr uint16_t LUPINE_FATBIN_ENTRY_VERSION = 0x0101;
static constexpr uint16_t LUPINE_FATBIN_ENTRY_PTX = 1;
static constexpr uint16_t LUPINE_FATBIN_ENTRY_CUBIN = 2;

#endif
