#ifndef LUPINE_CLIENT_HOOKS_H
#define LUPINE_CLIENT_HOOKS_H

// Declarations of the handwritten client helpers referenced by generated code
// in codegen/gen_client.cpp — both its marshalling plumbing and the
// @precall/@postcall/@onsuccess hook expressions in codegen/annotations.h.
// Definitions live in client.cpp, routing.cpp, memcpy.cpp, and cache.cpp.

#include <cuda.h>

#include "client_routing.h"
#include "rpc.h"

extern int rpc_size();
extern conn_t *rpc_client_get_connection(unsigned int index);
extern void rpc_close(conn_t *conn);

extern "C" void lupine_deep_cache_reset(const void *key);
extern "C" void *lupine_deep_cache_add(const void *key, size_t bytes);

extern "C" conn_t *lupine_rpc_conn_for_device(CUdevice *device);
extern "C" conn_t *lupine_rpc_conn_for_current_context();
extern "C" conn_t *lupine_rpc_conn_for_context(CUcontext ctx);
extern "C" conn_t *lupine_rpc_conn_for_module(CUmodule module);
extern "C" conn_t *lupine_rpc_conn_for_function(CUfunction function);
extern "C" conn_t *lupine_rpc_conn_for_stream(CUstream stream);
extern "C" conn_t *lupine_rpc_conn_for_event(CUevent event);
extern "C" conn_t *lupine_rpc_conn_for_deviceptr(CUdeviceptr ptr);
extern "C" CUfunction
lupine_translate_private_function_for_rpc(CUfunction function);

extern "C" void lupine_note_context_owner(CUcontext ctx, conn_t *conn);
extern "C" void lupine_note_module_owner(CUmodule module, conn_t *conn);
extern "C" void lupine_note_library_owner(CUlibrary library, conn_t *conn);
extern "C" void lupine_note_function_owner(CUfunction function, conn_t *conn);
extern "C" void lupine_note_stream_owner(CUstream stream, conn_t *conn);
extern "C" void lupine_note_event_owner(CUevent event, conn_t *conn);
extern "C" void lupine_note_memory_pool_owner(CUmemoryPool pool, conn_t *conn);
extern "C" void lupine_note_graph_owner(CUgraph graph, conn_t *conn);
extern "C" void lupine_note_graph_node_owner(CUgraphNode node, conn_t *conn);
extern "C" void lupine_note_graph_exec_owner(CUgraphExec exec, conn_t *conn);
extern "C" void lupine_note_deviceptr_owner(CUdeviceptr ptr, conn_t *conn);
extern "C" void lupine_note_deviceptr_allocation(CUdeviceptr ptr, size_t size,
                                                 conn_t *conn);
extern "C" void lupine_note_deviceptr_allocation_out(CUdeviceptr *dptr,
                                                     size_t size,
                                                     lupine_route route);
extern "C" void lupine_note_pitch_allocation_out(CUdeviceptr *dptr,
                                                 size_t *pPitch,
                                                 size_t WidthInBytes,
                                                 size_t Height,
                                                 lupine_route route);

extern "C" CUresult lupine_record_library_kernel(CUkernel kernel,
                                                 CUlibrary library,
                                                 const char *name,
                                                 lupine_route route);
extern "C" CUresult lupine_record_module_function(CUfunction function,
                                                  CUmodule module,
                                                  const char *name,
                                                  lupine_route route);
extern "C" CUresult lupine_record_module_function_out(CUfunction *hfunc,
                                                      CUmodule hmod,
                                                      const char *name,
                                                      lupine_route route);
extern "C" CUresult lupine_record_library_kernel_out(CUkernel *pKernel,
                                                     CUlibrary library,
                                                     const char *name,
                                                     lupine_route route);

extern "C" void lupine_prepare_host_range_write(void *host, size_t size);
extern "C" void lupine_mark_host_range_clean(void *host, size_t size);
extern "C" bool lupine_deviceptrs_share_route(CUdeviceptr first,
                                              CUdeviceptr second);
extern "C" bool lupine_translate_managed_host_ptr(CUdeviceptr ptr,
                                                  CUdeviceptr *translated);
extern "C" CUresult
lupine_cuMemcpyDtoD_via_client(CUdeviceptr dstDevice, CUdeviceptr srcDevice,
                               size_t ByteCount, CUstream hStream, bool async);

extern "C" void lupine_invalidate_current_context_cache();
extern "C" void lupine_forget_destroyed_context(CUcontext ctx);
extern "C" void lupine_pop_context_if_current(CUcontext ctx);
extern "C" void lupine_invalidate_function_caches();
extern "C" void lupine_note_primary_context_active(CUdevice dev);
extern "C" void lupine_note_primary_context_flags(CUdevice dev,
                                                  unsigned int flags);
extern "C" void lupine_invalidate_primary_context_state(CUdevice dev);
extern "C" void lupine_invalidate_kernel_attribute_cache();
extern "C" void lupine_kernel_attribute_cache_erase(int route_id,
                                                    CUkernel kernel, int attrib,
                                                    int dev);
extern "C" CUresult lupine_flush_dirty_host_pages_to_server();

extern "C" void lupine_apply_driver_version_override(int *driverVersion);

extern "C" int lupine_read_deferred_dtoh_copies(conn_t *conn);
extern "C" int lupine_forward_remote_stdout(conn_t *conn);
extern "C" CUresult lupine_sync_mapped_device_to_host();

#endif
