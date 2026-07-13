#ifndef LUPINE_CACHE_H
#define LUPINE_CACHE_H

#include <cuda.h>

#include <cstddef>
#include <cstdint>
#include <vector>

struct lupine_kernel_param_layout {
  uint32_t count = 0;
  std::vector<size_t> offsets;
  std::vector<size_t> sizes;
};

bool lupine_kernel_param_layout_cache_lookup(
    CUfunction function, lupine_kernel_param_layout *layout);
void lupine_kernel_param_layout_cache_insert(CUfunction function,
                                             lupine_kernel_param_layout layout);

bool lupine_current_context_device_cache_lookup(CUcontext context,
                                                CUdevice *device);
void lupine_current_context_device_cache_insert(CUcontext context,
                                                CUdevice device);
void lupine_current_context_device_cache_invalidate();

uint64_t lupine_lane_context_cache_epoch();
bool lupine_lane_context_cache_matches(int route_id, CUcontext context);
void lupine_lane_context_cache_update(int route_id, CUcontext context,
                                      uint64_t epoch, bool succeeded);
void lupine_lane_context_cache_store(int route_id, CUcontext context);

extern "C" void lupine_invalidate_current_context_cache();

#endif
