#ifndef LUPINE_CACHE_H
#define LUPINE_CACHE_H

#include <cuda.h>

#include <cstdint>

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

// The device a server answers a query with is a property of the lane bound to
// the calling thread. This token moves whenever anything can have rebound that
// lane: a runtime device selection, a driver context set on it, a context
// handle the server is free to hand out again, or a connection coming or
// going. A cached device answer is good for exactly as long as the token is.
extern "C" uint64_t lupine_device_binding_epoch();
extern "C" void lupine_note_device_binding_changed();

#endif
