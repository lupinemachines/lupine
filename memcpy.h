#pragma once

#define LUPINE_CUDA_COMPAT_TYPES_ONLY
#include "cuda_compat.h"
#undef LUPINE_CUDA_COMPAT_TYPES_ONLY

extern "C" bool lupine_is_managed_host_alias(CUdeviceptr ptr);
extern "C" CUresult lupine_sync_mapped_device_to_host();
extern "C" void lupine_mark_kernel_param_mappings(void *const *kernel_params,
                                                  const size_t *sizes,
                                                  uint32_t count);
