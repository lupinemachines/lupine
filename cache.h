#ifndef LUPINE_CACHE_H
#define LUPINE_CACHE_H

#include <cuda.h>

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

struct lupine_kernel_param_layout {
  uint32_t count = 0;
  std::vector<size_t> offsets;
  std::vector<size_t> sizes;
};

struct lupine_kernel_table_entry {
  std::string name;
  // One offset/size pair per kernel parameter.
  std::vector<uint64_t> params;
};

// A library image's kernel names and parameter layouts, persisted across runs
// so the next load can answer cuLibraryGetKernel and the parameter walk before
// the server's response arrives. Handles are per-load and never stored; only
// what the image itself fixes. The file is a hint that the response overrides,
// keyed by image bytes and the server's driver version.
bool lupine_kernel_table_cache_load(
    const unsigned char *image, size_t size, uint32_t driver_version,
    std::vector<lupine_kernel_table_entry> *out);
void lupine_kernel_table_cache_store(
    const unsigned char *image, size_t size, uint32_t driver_version,
    const std::vector<lupine_kernel_table_entry> &table);

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
