#pragma once

#include <cstddef>
#include <cstdint>

#define LUPINE_CUDA_COMPAT_TYPES_ONLY
#include "cuda_compat.h"
#undef LUPINE_CUDA_COMPAT_TYPES_ONLY

struct rpc_write_cursor;

extern "C" bool lupine_translate_managed_host_ptr(CUdeviceptr ptr,
                                                  CUdeviceptr *translated);
extern "C" CUresult lupine_sync_mapped_device_to_host();

CUresult lupine_sync_mapped_host_to_device_for_launch(
    void *const *kernel_params, const size_t *sizes, uint32_t count,
    CUdeviceptr *translated_params, rpc_write_cursor *rpc_params,
    bool *used_managed_mapping = nullptr);
