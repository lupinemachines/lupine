#pragma once

#define LUPINE_CUDA_COMPAT_TYPES_ONLY
#include "cuda_compat.h"
#undef LUPINE_CUDA_COMPAT_TYPES_ONLY

enum class lupine_copy_direction {
  host_to_host,
  host_to_device,
  device_to_host,
  device_to_device,
};
extern "C" bool lupine_copy_pointer_is_host(CUdeviceptr ptr);
extern "C" bool lupine_host_ptr_is_page_locked(const void *host);
extern "C" bool lupine_is_managed_host_alias(CUdeviceptr ptr);
extern "C" CUresult lupine_sync_mapped_device_to_host();
extern "C" void lupine_materialize_host_allocations();
