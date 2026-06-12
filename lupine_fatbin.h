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

static constexpr uint32_t LUPINE_FATBINC_MAGIC = 0x466243b1;
static constexpr uint32_t LUPINE_FATBIN_MAGIC = 0xba55ed50;

#endif
