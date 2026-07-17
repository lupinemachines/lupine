#pragma once

#include <cstddef>
#include <cstdint>

#define LUPINE_CUDA_COMPAT_TYPES_ONLY
#include "cuda_compat.h"
#undef LUPINE_CUDA_COMPAT_TYPES_ONLY

#ifdef cuMemPrefetchAsync
#undef cuMemPrefetchAsync
#endif
#ifdef cuMemPrefetchAsync_ptsz
#undef cuMemPrefetchAsync_ptsz
#endif

extern "C" CUresult cuMemPrefetchAsync(CUdeviceptr devPtr, size_t count,
                                       CUdevice dstDevice, CUstream hStream);
extern "C" CUresult cuMemPrefetchAsync_ptsz(CUdeviceptr devPtr, size_t count,
                                            CUdevice dstDevice,
                                            CUstream hStream);
extern "C" CUresult cuMemPrefetchAsync_v2(CUdeviceptr devPtr, size_t count,
                                          CUmemLocation location,
                                          unsigned int flags, CUstream hStream);

extern "C" void lupine_mark_host_range_clean(void *host, size_t size);
extern "C" void lupine_prepare_host_range_write(void *host, size_t size);
extern "C" CUresult lupine_flush_dirty_host_pages_to_server();
extern "C" bool lupine_translate_managed_host_ptr(CUdeviceptr ptr,
                                                  CUdeviceptr *translated);
extern "C" CUresult lupine_sync_mapped_device_to_host();

CUresult lupine_sync_mapped_host_to_device_for_launch(
    void *const *kernel_params, const size_t *sizes, uint32_t count,
    CUdeviceptr *translated_params, void **rpc_params,
    bool *used_managed_mapping = nullptr);
