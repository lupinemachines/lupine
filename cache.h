#ifndef LUPINE_CACHE_H
#define LUPINE_CACHE_H

#include <cuda.h>

#include <cstddef>
#include <cstdint>

struct lupine_deviceptr_allocation_record {
  size_t size = 0;
  int route_id = -2;
  CUcontext context = nullptr;
};

// The caller holds the routing mutex. Lookup results remain valid until the
// next allocation cache mutation.
const lupine_deviceptr_allocation_record *
lupine_deviceptr_allocation_cache_lookup(CUdeviceptr ptr);
void lupine_deviceptr_allocation_cache_insert(CUdeviceptr base, size_t size,
                                             int route_id, CUcontext context);
void lupine_deviceptr_allocation_cache_erase(CUdeviceptr base);

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

// A driver context set on the lane bound to the calling thread, or a
// connection coming or going, moves that lane off what this thread believed
// was current there.
extern "C" void lupine_note_device_binding_changed();

#endif
