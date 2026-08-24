#pragma once

#define LUPINE_CUDA_COMPAT_TYPES_ONLY
#include "cuda_compat.h"
#undef LUPINE_CUDA_COMPAT_TYPES_ONLY

extern "C" bool lupine_is_managed_host_alias(CUdeviceptr ptr);
extern "C" CUresult lupine_sync_mapped_device_to_host();
// Device work happened. A synchronization point uses this to skip a pass
// entirely when nothing has run since its last one.
extern "C" void lupine_note_device_work();
// Materializes every invalidated host allocation. Call before opening a stream
// capture so no fault can occur inside the capture window.
extern "C" void lupine_materialize_host_allocations();
