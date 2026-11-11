#pragma once

#define LUPINE_CUDA_COMPAT_TYPES_ONLY
#include "cuda_compat.h"
#undef LUPINE_CUDA_COMPAT_TYPES_ONLY

typedef struct conn_t conn_t;
struct lupine_route;

// Large managed allocations receive an aligned base that both processes can
// map directly, preserving base-pointer APIs such as stream attachment.
static constexpr size_t LUPINE_MANAGED_ALLOCATION_MIN_BYTES = 2 * 1024 * 1024;

extern "C" bool
lupine_device_attribute_is_virtualized(conn_t *conn, CUdevice_attribute attrib);
bool lupine_host_range_is_protected(uintptr_t start, size_t size);
extern "C" bool lupine_is_managed_host_alias(CUdeviceptr ptr);
struct lupine_pointer_query_context {
  CUdeviceptr orig_ptr = 0;
  CUdeviceptr query_ptr = 0;
  CUdeviceptr managed_alias_base = 0;
  CUdeviceptr remote_host_base = 0;
  CUdeviceptr remote_device_base = 0;
  CUdeviceptr client_range_start = 0;
  size_t client_range_size = 0;
  bool managed_alias = false;
  bool remote_host_alias = false;
  bool tracked_host = false;
  bool has_client_range = false;
};
lupine_pointer_query_context lupine_resolve_pointer_query(CUdeviceptr ptr);
void lupine_fixup_pointer_attribute_value(
    const lupine_pointer_query_context &ctx, CUpointer_attribute attribute,
    void *data);
extern "C" CUdeviceptr lupine_resolve_pointer_attribute_target(CUdeviceptr ptr);
extern "C" CUresult lupine_managed_mem_advise(CUdeviceptr ptr, size_t count,
                                              CUmem_advise advice,
                                              CUdevice device);
extern "C" bool lupine_managed_mem_range_attribute(
    void *data, size_t data_size, CUmem_range_attribute attribute,
    CUdeviceptr ptr, size_t count, CUresult *result);

extern "C" const void *lupine_mapped_host_read_source(const void *host,
                                                      size_t size);
extern "C" int lupine_write_cross_route_device_source(conn_t *destination_conn,
                                                      int request_id,
                                                      CUdeviceptr source,
                                                      size_t bytes);
bool lupine_prepare_mapped_host_pointer(CUdeviceptr pointer, bool *portable);
CUresult lupine_prepare_portable_host_allocations(lupine_route route,
                                                  bool expose_portable);
extern "C" CUresult lupine_sync_mapped_device_to_host();
extern "C" void lupine_invalidate_launched_managed(CUdeviceptr pointer,
                                                   CUstream stream);
extern "C" void lupine_adopt_managed_global(lupine_route route,
                                            CUdeviceptr global, size_t bytes);
extern thread_local bool lupine_in_host_callback;
// Before a host callback's acknowledgement, which resumes its stream as soon as
// it arrives: waits until the host writes flushed so far, by this thread or
// another one that picked up its pages, are applied on the server. Only the
// connection's dispatch thread settles it.
int lupine_settle_host_flushes(conn_t *conn);
// Device-to-host copies into client memory arrive on a lane of their own,
// numbered (see cuda_server_memcpy.h). This starts the connection's receiver,
// which applies them in order.
void lupine_start_dtoh_receiver(conn_t *conn);
// Joins it, once the connection's transport has been shut down.
void lupine_join_dtoh_receiver(conn_t *conn);
// Reads the delivery number an answer or server callback carries, and waits
// until every delivery up to it has been applied.
extern "C" int lupine_read_dtoh_watermark(conn_t *conn);
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
