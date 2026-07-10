#ifndef LUPINE_CACHE_H
#define LUPINE_CACHE_H

#include <cuda.h>

#include <cstddef>
#include <cstdint>

struct lupine_kernel_param_layout {
  uint32_t count = 0;
  size_t offsets[64] = {};
  size_t sizes[64] = {};
};

bool lupine_kernel_param_layout_cache_lookup(int route_id, CUfunction function,
                                             lupine_kernel_param_layout *layout,
                                             uint64_t *epoch);
void lupine_kernel_param_layout_cache_insert(
    int route_id, CUfunction function, const lupine_kernel_param_layout &layout,
    uint64_t epoch);
void lupine_kernel_param_layout_cache_invalidate();

#endif
