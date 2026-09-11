#pragma once

#define LUPINE_CUDA_COMPAT_TYPES_ONLY
#include "cuda_compat.h"
#undef LUPINE_CUDA_COMPAT_TYPES_ONLY

typedef struct conn_t conn_t;

// Large managed allocations receive an aligned base that both processes can
// map directly, preserving base-pointer APIs such as stream attachment.
static constexpr size_t LUPINE_MANAGED_ALLOCATION_MIN_BYTES = 2 * 1024 * 1024;

// Shared client mappings and lifetime tracking for pinned host allocations.
// The caller owns the backend allocation and frees it if adoption fails.
extern "C" CUresult
lupine_adopt_host_allocation(conn_t *conn, void **host, void *remote_host,
                             CUdeviceptr device_ptr, size_t bytes,
                             unsigned int flags, bool managed = false);
using lupine_host_free_fn = CUresult (*)(conn_t *, void *);
extern "C" CUresult lupine_free_host_allocation(void *host,
                                                lupine_host_free_fn release);
using lupine_host_register_fn = CUresult (*)(conn_t *, void *, size_t,
                                             unsigned int, void **,
                                             CUdeviceptr *);
extern "C" CUresult lupine_register_host_allocation(
    conn_t *conn, void *host, size_t bytes, unsigned int flags,
    lupine_host_register_fn allocate, lupine_host_free_fn release);
extern "C" CUresult
lupine_unregister_host_allocation(void *host, lupine_host_free_fn release);
extern "C" void *lupine_host_pointer_for_rpc(void *host, conn_t **owner);
using lupine_device_free_fn = CUresult (*)(conn_t *, CUdeviceptr);
extern "C" CUresult
lupine_free_device_allocation(CUdeviceptr pointer,
                              lupine_device_free_fn release);

extern "C" bool lupine_copy_pointer_is_host(CUdeviceptr ptr);
extern "C" bool
lupine_device_attribute_is_virtualized(CUdevice_attribute attrib);
extern "C" bool lupine_host_ptr_is_page_locked(const void *host);
extern "C" bool lupine_is_managed_host_alias(CUdeviceptr ptr);
extern "C" const void *lupine_mapped_host_read_source(const void *host,
                                                      size_t size);
extern "C" int lupine_write_cross_route_device_source(conn_t *destination_conn,
                                                      int request_id,
                                                      CUdeviceptr source,
                                                      size_t bytes);
extern "C" void
lupine_mark_mapped_host_kernel_params(void *const *kernel_params,
                                      const size_t *sizes, uint32_t count);
extern "C" CUresult lupine_sync_mapped_device_to_host();
extern "C" int lupine_read_deferred_host_copy(conn_t *conn, void *destination,
                                              size_t bytes);
extern "C" CUresult lupine_invalidate_managed_allocations();
extern "C" void lupine_materialize_host_allocations();

extern "C" void lupine_stream_capture_begin();
extern "C" void lupine_stream_capture_begin_complete(bool started);
extern "C" CUresult lupine_complete_stream_end_capture(CUresult result);

class lupine_capture_begin_guard {
public:
  lupine_capture_begin_guard() { lupine_stream_capture_begin(); }
  ~lupine_capture_begin_guard() {
    if (!completed_) {
      lupine_stream_capture_begin_complete(false);
    }
  }

  CUresult complete(CUresult result) {
    if (!completed_) {
      completed_ = true;
      lupine_stream_capture_begin_complete(result == CUDA_SUCCESS);
    }
    return result;
  }

private:
  bool completed_ = false;
};
