#pragma once

#define LUPINE_CUDA_COMPAT_TYPES_ONLY
#include "cuda_compat.h"
#undef LUPINE_CUDA_COMPAT_TYPES_ONLY

typedef struct conn_t conn_t;

extern "C" bool lupine_copy_pointer_is_host(CUdeviceptr ptr);
extern "C" bool lupine_host_ptr_is_page_locked(const void *host);
extern "C" bool lupine_is_managed_host_alias(CUdeviceptr ptr);
extern "C" const void *lupine_mapped_host_read_source(const void *host,
                                                      size_t size);
extern "C" int lupine_write_cross_route_device_source(
    conn_t *destination_conn, int request_id, CUdeviceptr source, size_t width,
    size_t rows, size_t row_stride, size_t slices, size_t slice_stride);
extern "C" void
lupine_mark_mapped_host_kernel_params(void *const *kernel_params,
                                      const size_t *sizes, uint32_t count);
extern "C" CUresult lupine_sync_mapped_device_to_host();
extern "C" void lupine_materialize_host_allocations();
