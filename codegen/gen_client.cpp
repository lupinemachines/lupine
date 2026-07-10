#include <cuda.h>

#define LUPINE_CUDA_COMPAT_TYPES_ONLY
#include "cuda_compat.h"
#undef LUPINE_CUDA_COMPAT_TYPES_ONLY

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <unordered_map>
#include <vector>

#include "gen_api.h"

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

extern "C" void lupine_forget_deviceptr_owner(CUdeviceptr ptr);

extern "C" void lupine_record_library_kernel(CUkernel kernel, CUlibrary library,
                                             const char *name,
                                             lupine_route route);

extern "C" void lupine_record_module_function(CUfunction function,
                                              CUmodule module, const char *name,
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

extern "C" CUresult lupine_cuCtxPushCurrent_virtual(CUcontext ctx);
extern "C" CUresult lupine_cuCtxPopCurrent_virtual(CUcontext *pctx);
extern "C" CUresult lupine_cuCtxSetCurrent_virtual(CUcontext ctx);
extern "C" CUresult lupine_cuCtxGetCurrent_virtual(CUcontext *pctx);
extern "C" CUresult lupine_cuCtxGetDevice_cached(CUdevice *device);
extern "C" void lupine_invalidate_current_context_cache();
extern "C" void lupine_invalidate_kernel_param_layout_cache();
extern "C" CUresult
lupine_cuDevicePrimaryCtxGetState_cached(CUdevice dev, unsigned int *flags,
                                         int *active);
extern "C" void lupine_note_primary_context_active(CUdevice dev);
extern "C" void lupine_note_primary_context_flags(CUdevice dev,
                                                  unsigned int flags);
extern "C" void lupine_invalidate_primary_context_state(CUdevice dev);
extern "C" CUresult
lupine_cuDeviceGetAttribute_cached(int *pi, CUdevice_attribute attrib,
                                   CUdevice dev);
extern "C" CUresult lupine_cuKernelGetFunction_cached(CUfunction *pFunc,
                                                      CUkernel kernel);
extern "C" CUresult lupine_cuOccupancyMaxActiveBlocksPerMultiprocessor_cached(
    int *numBlocks, CUfunction func, int blockSize, size_t dynamicSMemSize);
extern "C" CUresult
lupine_cuOccupancyMaxActiveBlocksPerMultiprocessorWithFlags_cached(
    int *numBlocks, CUfunction func, int blockSize, size_t dynamicSMemSize,
    unsigned int flags);
extern "C" CUresult lupine_flush_dirty_host_pages_to_server();

extern "C" int lupine_read_deferred_dtoh_copies(conn_t *conn);
extern "C" int lupine_forward_remote_stdout(conn_t *conn);
extern "C" CUresult lupine_sync_mapped_device_to_host();

CUresult cuDriverGetVersion(int *driverVersion) {
  lupine_route route = lupine_route_for_default();
  CUresult return_value;
  using real_fn_t = CUresult (*)(int *);
  if (lupine_call_local_cuda_if_routed<real_fn_t>(
          route, "cuDriverGetVersion", &return_value, driverVersion)) {
    if (driverVersion != nullptr) {
      const char *override_version = getenv("LUPINE_DRIVER_VERSION_OVERRIDE");
      if (override_version != nullptr)
        *driverVersion = atoi(override_version);
    }
    return return_value;
  }
  conn_t *conn = lupine_route_remote_conn(route);
  if (conn == nullptr ||
      rpc_write_start_request(conn, RPC_cuDriverGetVersion) < 0 ||
      rpc_wait_for_response(conn) < 0 ||
      rpc_read(conn, driverVersion, sizeof(int)) < 0 ||
      rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
      rpc_read_end(conn) < 0)
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  if (driverVersion != nullptr) {
    const char *override_version = getenv("LUPINE_DRIVER_VERSION_OVERRIDE");
    if (override_version != nullptr)
      *driverVersion = atoi(override_version);
  }
  return return_value;
}

CUresult cuDeviceGetName(char *name, int len, CUdevice dev) {
  lupine_route route = lupine_route_for_device(&dev);
  CUresult return_value;
  using real_fn_t = CUresult (*)(char *, int, CUdevice);
  if (lupine_call_local_cuda_if_routed<real_fn_t>(
          route, "cuDeviceGetName", &return_value, name, len, dev)) {
    return return_value;
  }
  conn_t *conn = lupine_route_remote_conn(route);
  if (conn == nullptr ||
      rpc_write_start_request(conn, RPC_cuDeviceGetName) < 0 ||
      rpc_write(conn, &len, sizeof(int)) < 0 ||
      rpc_write(conn, &dev, sizeof(CUdevice)) < 0 ||
      rpc_wait_for_response(conn) < 0 ||
      (lupine_prepare_host_range_write(name, len * sizeof(char)), false) ||
      (len * sizeof(char) != 0 &&
       rpc_read(conn, name, len * sizeof(char)) < 0) ||
      rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
      rpc_read_end(conn) < 0)
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  if (return_value == CUDA_SUCCESS)
    lupine_mark_host_range_clean(name, len * sizeof(char));
  return return_value;
}

CUresult cuDeviceGetUuid_v2(CUuuid *uuid, CUdevice dev) {
  lupine_route route = lupine_route_for_device(&dev);
  CUresult return_value;
  using real_fn_t = CUresult (*)(CUuuid *, CUdevice);
  if (lupine_call_local_cuda_if_routed<real_fn_t>(route, "cuDeviceGetUuid_v2",
                                                  &return_value, uuid, dev)) {
    return return_value;
  }
  conn_t *conn = lupine_route_remote_conn(route);
  if (conn == nullptr ||
      rpc_write_start_request(conn, RPC_cuDeviceGetUuid_v2) < 0 ||
      rpc_write(conn, &dev, sizeof(CUdevice)) < 0 ||
      rpc_wait_for_response(conn) < 0 ||
      (lupine_prepare_host_range_write(uuid, 16 * sizeof(CUuuid)), false) ||
      (16 != 0 && rpc_read(conn, uuid, 16) < 0) ||
      rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
      rpc_read_end(conn) < 0)
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  if (return_value == CUDA_SUCCESS)
    lupine_mark_host_range_clean(uuid, 16 * sizeof(CUuuid));
  return return_value;
}

CUresult cuDeviceGetLuid(char *luid, unsigned int *deviceNodeMask,
                         CUdevice dev) {
  lupine_route route = lupine_route_for_device(&dev);
  CUresult return_value;
  using real_fn_t = CUresult (*)(char *, unsigned int *, CUdevice);
  if (lupine_call_local_cuda_if_routed<real_fn_t>(
          route, "cuDeviceGetLuid", &return_value, luid, deviceNodeMask, dev)) {
    return return_value;
  }
  conn_t *conn = lupine_route_remote_conn(route);
  if (conn == nullptr ||
      rpc_write_start_request(conn, RPC_cuDeviceGetLuid) < 0 ||
      rpc_write(conn, &dev, sizeof(CUdevice)) < 0 ||
      rpc_wait_for_response(conn) < 0 ||
      (lupine_prepare_host_range_write(luid, 8 * sizeof(char)), false) ||
      (8 != 0 && rpc_read(conn, luid, 8) < 0) ||
      rpc_read(conn, deviceNodeMask, sizeof(unsigned int)) < 0 ||
      rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
      rpc_read_end(conn) < 0)
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  if (return_value == CUDA_SUCCESS)
    lupine_mark_host_range_clean(luid, 8 * sizeof(char));
  return return_value;
}

CUresult cuDeviceTotalMem_v2(size_t *bytes, CUdevice dev) {
  lupine_route route = lupine_route_for_device(&dev);
  CUresult return_value;
  using real_fn_t = CUresult (*)(size_t *, CUdevice);
  if (lupine_call_local_cuda_if_routed<real_fn_t>(route, "cuDeviceTotalMem_v2",
                                                  &return_value, bytes, dev)) {
    return return_value;
  }
  conn_t *conn = lupine_route_remote_conn(route);
  if (conn == nullptr ||
      rpc_write_start_request(conn, RPC_cuDeviceTotalMem_v2) < 0 ||
      rpc_write(conn, &dev, sizeof(CUdevice)) < 0 ||
      rpc_wait_for_response(conn) < 0 ||
      rpc_read(conn, bytes, sizeof(size_t)) < 0 ||
      rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
      rpc_read_end(conn) < 0)
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  return return_value;
}

CUresult cuDeviceGetTexture1DLinearMaxWidth(size_t *maxWidthInElements,
                                            CUarray_format format,
                                            unsigned numChannels,
                                            CUdevice dev) {
  lupine_route route = lupine_route_for_device(&dev);
  CUresult return_value;
  using real_fn_t = CUresult (*)(size_t *, CUarray_format, unsigned, CUdevice);
  if (lupine_call_local_cuda_if_routed<real_fn_t>(
          route, "cuDeviceGetTexture1DLinearMaxWidth", &return_value,
          maxWidthInElements, format, numChannels, dev)) {
    return return_value;
  }
  conn_t *conn = lupine_route_remote_conn(route);
  if (conn == nullptr ||
      rpc_write_start_request(conn, RPC_cuDeviceGetTexture1DLinearMaxWidth) <
          0 ||
      rpc_write(conn, &format, sizeof(CUarray_format)) < 0 ||
      rpc_write(conn, &numChannels, sizeof(unsigned)) < 0 ||
      rpc_write(conn, &dev, sizeof(CUdevice)) < 0 ||
      rpc_wait_for_response(conn) < 0 ||
      rpc_read(conn, maxWidthInElements, sizeof(size_t)) < 0 ||
      rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
      rpc_read_end(conn) < 0)
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  return return_value;
}

CUresult cuDeviceGetAttribute(int *pi, CUdevice_attribute attrib,
                              CUdevice dev) {
  return lupine_cuDeviceGetAttribute_cached(pi, attrib, dev);
}

CUresult cuDeviceSetMemPool(CUdevice dev, CUmemoryPool pool) {
  lupine_route route = lupine_route_for_device(&dev);
  CUresult return_value;
  using real_fn_t = CUresult (*)(CUdevice, CUmemoryPool);
  if (lupine_call_local_cuda_if_routed<real_fn_t>(route, "cuDeviceSetMemPool",
                                                  &return_value, dev, pool)) {
    return return_value;
  }
  conn_t *conn = lupine_route_remote_conn(route);
  if (conn == nullptr ||
      rpc_write_start_request(conn, RPC_cuDeviceSetMemPool) < 0 ||
      rpc_write(conn, &dev, sizeof(CUdevice)) < 0 ||
      rpc_write(conn, &pool, sizeof(CUmemoryPool)) < 0 ||
      rpc_wait_for_response(conn) < 0 ||
      rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
      rpc_read_end(conn) < 0)
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  return return_value;
}

CUresult cuDeviceGetMemPool(CUmemoryPool *pool, CUdevice dev) {
  lupine_route route = lupine_route_for_device(&dev);
  CUresult return_value;
  using real_fn_t = CUresult (*)(CUmemoryPool *, CUdevice);
  if (lupine_call_local_cuda_if_routed<real_fn_t>(route, "cuDeviceGetMemPool",
                                                  &return_value, pool, dev)) {
    if (return_value == CUDA_SUCCESS && pool != nullptr) {
      lupine_note_memory_pool_owner_route(*pool, route);
    }
    return return_value;
  }
  conn_t *conn = lupine_route_remote_conn(route);
  if (conn == nullptr ||
      rpc_write_start_request(conn, RPC_cuDeviceGetMemPool) < 0 ||
      rpc_write(conn, &dev, sizeof(CUdevice)) < 0 ||
      rpc_wait_for_response(conn) < 0 ||
      rpc_read(conn, pool, sizeof(CUmemoryPool)) < 0 ||
      rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
      rpc_read_end(conn) < 0)
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  if (return_value == CUDA_SUCCESS && pool != nullptr) {
    lupine_note_memory_pool_owner_route(*pool, route);
  }
  return return_value;
}

CUresult cuDeviceGetDefaultMemPool(CUmemoryPool *pool_out, CUdevice dev) {
  lupine_route route = lupine_route_for_device(&dev);
  CUresult return_value;
  using real_fn_t = CUresult (*)(CUmemoryPool *, CUdevice);
  if (lupine_call_local_cuda_if_routed<real_fn_t>(
          route, "cuDeviceGetDefaultMemPool", &return_value, pool_out, dev)) {
    if (return_value == CUDA_SUCCESS && pool_out != nullptr) {
      lupine_note_memory_pool_owner_route(*pool_out, route);
    }
    return return_value;
  }
  conn_t *conn = lupine_route_remote_conn(route);
  if (conn == nullptr ||
      rpc_write_start_request(conn, RPC_cuDeviceGetDefaultMemPool) < 0 ||
      rpc_write(conn, &dev, sizeof(CUdevice)) < 0 ||
      rpc_wait_for_response(conn) < 0 ||
      rpc_read(conn, pool_out, sizeof(CUmemoryPool)) < 0 ||
      rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
      rpc_read_end(conn) < 0)
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  if (return_value == CUDA_SUCCESS && pool_out != nullptr) {
    lupine_note_memory_pool_owner_route(*pool_out, route);
  }
  return return_value;
}

CUresult cuDeviceGetExecAffinitySupport(int *pi, CUexecAffinityType type,
                                        CUdevice dev) {
  lupine_route route = lupine_route_for_device(&dev);
  CUresult return_value;
  using real_fn_t = CUresult (*)(int *, CUexecAffinityType, CUdevice);
  if (lupine_call_local_cuda_if_routed<real_fn_t>(
          route, "cuDeviceGetExecAffinitySupport", &return_value, pi, type,
          dev)) {
    return return_value;
  }
  conn_t *conn = lupine_route_remote_conn(route);
  if (conn == nullptr ||
      rpc_write_start_request(conn, RPC_cuDeviceGetExecAffinitySupport) < 0 ||
      rpc_write(conn, &type, sizeof(CUexecAffinityType)) < 0 ||
      rpc_write(conn, &dev, sizeof(CUdevice)) < 0 ||
      rpc_wait_for_response(conn) < 0 || rpc_read(conn, pi, sizeof(int)) < 0 ||
      rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
      rpc_read_end(conn) < 0)
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  return return_value;
}

CUresult cuFlushGPUDirectRDMAWrites(CUflushGPUDirectRDMAWritesTarget target,
                                    CUflushGPUDirectRDMAWritesScope scope) {
  lupine_route route = lupine_route_for_default();
  CUresult return_value;
  using real_fn_t = CUresult (*)(CUflushGPUDirectRDMAWritesTarget,
                                 CUflushGPUDirectRDMAWritesScope);
  if (lupine_call_local_cuda_if_routed<real_fn_t>(
          route, "cuFlushGPUDirectRDMAWrites", &return_value, target, scope)) {
    return return_value;
  }
  conn_t *conn = lupine_route_remote_conn(route);
  if (conn == nullptr ||
      rpc_write_start_request(conn, RPC_cuFlushGPUDirectRDMAWrites) < 0 ||
      rpc_write(conn, &target, sizeof(CUflushGPUDirectRDMAWritesTarget)) < 0 ||
      rpc_write(conn, &scope, sizeof(CUflushGPUDirectRDMAWritesScope)) < 0 ||
      rpc_wait_for_response(conn) < 0 ||
      rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
      rpc_read_end(conn) < 0)
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  return return_value;
}

CUresult cuDeviceGetProperties(CUdevprop *prop, CUdevice dev) {
  lupine_route route = lupine_route_for_device(&dev);
  CUresult return_value;
  using real_fn_t = CUresult (*)(CUdevprop *, CUdevice);
  if (lupine_call_local_cuda_if_routed<real_fn_t>(
          route, "cuDeviceGetProperties", &return_value, prop, dev)) {
    return return_value;
  }
  conn_t *conn = lupine_route_remote_conn(route);
  if (conn == nullptr ||
      rpc_write_start_request(conn, RPC_cuDeviceGetProperties) < 0 ||
      rpc_write(conn, &dev, sizeof(CUdevice)) < 0 ||
      rpc_wait_for_response(conn) < 0 ||
      rpc_read(conn, prop, sizeof(CUdevprop)) < 0 ||
      rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
      rpc_read_end(conn) < 0)
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  return return_value;
}

CUresult cuDeviceComputeCapability(int *major, int *minor, CUdevice dev) {
  lupine_route route = lupine_route_for_device(&dev);
  CUresult return_value;
  using real_fn_t = CUresult (*)(int *, int *, CUdevice);
  if (lupine_call_local_cuda_if_routed<real_fn_t>(
          route, "cuDeviceComputeCapability", &return_value, major, minor,
          dev)) {
    return return_value;
  }
  conn_t *conn = lupine_route_remote_conn(route);
  if (conn == nullptr ||
      rpc_write_start_request(conn, RPC_cuDeviceComputeCapability) < 0 ||
      rpc_write(conn, &dev, sizeof(CUdevice)) < 0 ||
      rpc_wait_for_response(conn) < 0 ||
      rpc_read(conn, major, sizeof(int)) < 0 ||
      rpc_read(conn, minor, sizeof(int)) < 0 ||
      rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
      rpc_read_end(conn) < 0)
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  return return_value;
}

CUresult cuDevicePrimaryCtxRetain(CUcontext *pctx, CUdevice dev) {
  lupine_route route = lupine_route_for_device(&dev);
  CUresult return_value;
  using real_fn_t = CUresult (*)(CUcontext *, CUdevice);
  if (lupine_call_local_cuda_if_routed<real_fn_t>(
          route, "cuDevicePrimaryCtxRetain", &return_value, pctx, dev)) {
    if (return_value == CUDA_SUCCESS && pctx != nullptr) {
      lupine_note_context_owner_route(*pctx, route);
    }
    if (return_value == CUDA_SUCCESS)
      lupine_note_primary_context_active(dev);
    return return_value;
  }
  conn_t *conn = lupine_route_remote_conn(route);
  if (conn == nullptr ||
      rpc_write_start_request(conn, RPC_cuDevicePrimaryCtxRetain) < 0 ||
      rpc_write(conn, &dev, sizeof(CUdevice)) < 0 ||
      rpc_wait_for_response(conn) < 0 ||
      rpc_read(conn, pctx, sizeof(CUcontext)) < 0 ||
      rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
      rpc_read_end(conn) < 0)
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  if (return_value == CUDA_SUCCESS && pctx != nullptr) {
    lupine_note_context_owner_route(*pctx, route);
  }
  if (return_value == CUDA_SUCCESS)
    lupine_note_primary_context_active(dev);
  return return_value;
}

CUresult cuDevicePrimaryCtxRelease_v2(CUdevice dev) {
  lupine_route route = lupine_route_for_device(&dev);
  CUresult return_value;
  using real_fn_t = CUresult (*)(CUdevice);
  if (lupine_call_local_cuda_if_routed<real_fn_t>(
          route, "cuDevicePrimaryCtxRelease_v2", &return_value, dev)) {
    if (return_value == CUDA_SUCCESS)
      lupine_invalidate_primary_context_state(dev);
    if (return_value == CUDA_SUCCESS)
      lupine_invalidate_current_context_cache();
    if (return_value == CUDA_SUCCESS)
      lupine_invalidate_kernel_param_layout_cache();
    return return_value;
  }
  conn_t *conn = lupine_route_remote_conn(route);
  if (conn == nullptr ||
      rpc_write_start_request(conn, RPC_cuDevicePrimaryCtxRelease_v2) < 0 ||
      rpc_write(conn, &dev, sizeof(CUdevice)) < 0 ||
      rpc_wait_for_response(conn) < 0 ||
      rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
      rpc_read_end(conn) < 0)
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  if (return_value == CUDA_SUCCESS)
    lupine_invalidate_primary_context_state(dev);
  if (return_value == CUDA_SUCCESS)
    lupine_invalidate_current_context_cache();
  if (return_value == CUDA_SUCCESS)
    lupine_invalidate_kernel_param_layout_cache();
  return return_value;
}

CUresult cuDevicePrimaryCtxSetFlags_v2(CUdevice dev, unsigned int flags) {
  lupine_route route = lupine_route_for_device(&dev);
  CUresult return_value;
  using real_fn_t = CUresult (*)(CUdevice, unsigned int);
  if (lupine_call_local_cuda_if_routed<real_fn_t>(
          route, "cuDevicePrimaryCtxSetFlags_v2", &return_value, dev, flags)) {
    if (return_value == CUDA_SUCCESS)
      lupine_note_primary_context_flags(dev, flags);
    return return_value;
  }
  conn_t *conn = lupine_route_remote_conn(route);
  if (conn == nullptr ||
      rpc_write_start_request(conn, RPC_cuDevicePrimaryCtxSetFlags_v2) < 0 ||
      rpc_write(conn, &dev, sizeof(CUdevice)) < 0 ||
      rpc_write(conn, &flags, sizeof(unsigned int)) < 0 ||
      rpc_wait_for_response(conn) < 0 ||
      rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
      rpc_read_end(conn) < 0)
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  if (return_value == CUDA_SUCCESS)
    lupine_note_primary_context_flags(dev, flags);
  return return_value;
}

CUresult cuDevicePrimaryCtxGetState(CUdevice dev, unsigned int *flags,
                                    int *active) {
  return lupine_cuDevicePrimaryCtxGetState_cached(dev, flags, active);
}

CUresult cuDevicePrimaryCtxReset_v2(CUdevice dev) {
  lupine_route route = lupine_route_for_device(&dev);
  CUresult return_value;
  using real_fn_t = CUresult (*)(CUdevice);
  if (lupine_call_local_cuda_if_routed<real_fn_t>(
          route, "cuDevicePrimaryCtxReset_v2", &return_value, dev)) {
    if (return_value == CUDA_SUCCESS)
      lupine_invalidate_primary_context_state(dev);
    if (return_value == CUDA_SUCCESS)
      lupine_invalidate_current_context_cache();
    if (return_value == CUDA_SUCCESS)
      lupine_invalidate_kernel_param_layout_cache();
    return return_value;
  }
  conn_t *conn = lupine_route_remote_conn(route);
  if (conn == nullptr ||
      rpc_write_start_request(conn, RPC_cuDevicePrimaryCtxReset_v2) < 0 ||
      rpc_write(conn, &dev, sizeof(CUdevice)) < 0 ||
      rpc_wait_for_response(conn) < 0 ||
      rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
      rpc_read_end(conn) < 0)
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  if (return_value == CUDA_SUCCESS)
    lupine_invalidate_primary_context_state(dev);
  if (return_value == CUDA_SUCCESS)
    lupine_invalidate_current_context_cache();
  if (return_value == CUDA_SUCCESS)
    lupine_invalidate_kernel_param_layout_cache();
  return return_value;
}

CUresult cuCtxDestroy_v2(CUcontext ctx) {
  lupine_route route = lupine_route_for_context(ctx);
  CUresult return_value;
  using real_fn_t = CUresult (*)(CUcontext);
  if (lupine_call_local_cuda_if_routed<real_fn_t>(route, "cuCtxDestroy_v2",
                                                  &return_value, ctx)) {
    if (return_value == CUDA_SUCCESS)
      lupine_invalidate_current_context_cache();
    if (return_value == CUDA_SUCCESS)
      lupine_invalidate_kernel_param_layout_cache();
    return return_value;
  }
  conn_t *conn = lupine_route_remote_conn(route);
  if (conn == nullptr ||
      rpc_write_start_request(conn, RPC_cuCtxDestroy_v2) < 0 ||
      rpc_write(conn, &ctx, sizeof(CUcontext)) < 0 ||
      rpc_wait_for_response(conn) < 0 ||
      rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
      rpc_read_end(conn) < 0)
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  if (return_value == CUDA_SUCCESS)
    lupine_invalidate_current_context_cache();
  if (return_value == CUDA_SUCCESS)
    lupine_invalidate_kernel_param_layout_cache();
  return return_value;
}

CUresult cuCtxPushCurrent_v2(CUcontext ctx) {
  return lupine_cuCtxPushCurrent_virtual(ctx);
}

CUresult cuCtxPopCurrent_v2(CUcontext *pctx) {
  return lupine_cuCtxPopCurrent_virtual(pctx);
}

CUresult cuCtxSetCurrent(CUcontext ctx) {
  return lupine_cuCtxSetCurrent_virtual(ctx);
}

CUresult cuCtxGetCurrent(CUcontext *pctx) {
  return lupine_cuCtxGetCurrent_virtual(pctx);
}

CUresult cuCtxGetDevice(CUdevice *device) {
  return lupine_cuCtxGetDevice_cached(device);
}

CUresult cuCtxGetFlags(unsigned int *flags) {
  lupine_route route = lupine_route_for_default();
  CUresult return_value;
  using real_fn_t = CUresult (*)(unsigned int *);
  if (lupine_call_local_cuda_if_routed<real_fn_t>(route, "cuCtxGetFlags",
                                                  &return_value, flags)) {
    return return_value;
  }
  conn_t *conn = lupine_route_remote_conn(route);
  if (conn == nullptr || rpc_write_start_request(conn, RPC_cuCtxGetFlags) < 0 ||
      rpc_wait_for_response(conn) < 0 ||
      rpc_read(conn, flags, sizeof(unsigned int)) < 0 ||
      rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
      rpc_read_end(conn) < 0)
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  return return_value;
}

CUresult cuCtxGetId(CUcontext ctx, unsigned long long *ctxId) {
  lupine_route route = lupine_route_for_context(ctx);
  CUresult return_value;
  using real_fn_t = CUresult (*)(CUcontext, unsigned long long *);
  if (lupine_call_local_cuda_if_routed<real_fn_t>(route, "cuCtxGetId",
                                                  &return_value, ctx, ctxId)) {
    return return_value;
  }
  conn_t *conn = lupine_route_remote_conn(route);
  if (conn == nullptr || rpc_write_start_request(conn, RPC_cuCtxGetId) < 0 ||
      rpc_write(conn, &ctx, sizeof(CUcontext)) < 0 ||
      rpc_wait_for_response(conn) < 0 ||
      rpc_read(conn, ctxId, sizeof(unsigned long long)) < 0 ||
      rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
      rpc_read_end(conn) < 0)
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  return return_value;
}

CUresult cuCtxSynchronize() {
  CUresult lupine_sync_result = lupine_flush_dirty_host_pages_to_server();
  if (lupine_sync_result != CUDA_SUCCESS) {
    return lupine_sync_result;
  }
  lupine_route route = lupine_route_for_current_context();
  CUresult return_value;
  using real_fn_t = CUresult (*)();
  if (lupine_call_local_cuda_if_routed<real_fn_t>(route, "cuCtxSynchronize",
                                                  &return_value)) {
    if (return_value == CUDA_SUCCESS)
      return_value = lupine_sync_mapped_device_to_host();
    return return_value;
  }
  conn_t *conn = lupine_route_remote_conn(route);
  if (conn == nullptr ||
      rpc_write_start_request(conn, RPC_cuCtxSynchronize) < 0 ||
      rpc_wait_for_response(conn) < 0 ||
      lupine_read_deferred_dtoh_copies(conn) < 0 ||
      lupine_forward_remote_stdout(conn) < 0 ||
      rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
      rpc_read_end(conn) < 0)
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  if (return_value == CUDA_SUCCESS)
    return_value = lupine_sync_mapped_device_to_host();
  return return_value;
}

CUresult cuCtxSetLimit(CUlimit limit, size_t value) {
  lupine_route route = lupine_route_for_default();
  CUresult return_value;
  using real_fn_t = CUresult (*)(CUlimit, size_t);
  if (lupine_call_local_cuda_if_routed<real_fn_t>(
          route, "cuCtxSetLimit", &return_value, limit, value)) {
    return return_value;
  }
  conn_t *conn = lupine_route_remote_conn(route);
  if (conn == nullptr || rpc_write_start_request(conn, RPC_cuCtxSetLimit) < 0 ||
      rpc_write(conn, &limit, sizeof(CUlimit)) < 0 ||
      rpc_write(conn, &value, sizeof(size_t)) < 0 ||
      rpc_wait_for_response(conn) < 0 ||
      rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
      rpc_read_end(conn) < 0)
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  return return_value;
}

CUresult cuCtxGetLimit(size_t *pvalue, CUlimit limit) {
  lupine_route route = lupine_route_for_default();
  CUresult return_value;
  using real_fn_t = CUresult (*)(size_t *, CUlimit);
  if (lupine_call_local_cuda_if_routed<real_fn_t>(
          route, "cuCtxGetLimit", &return_value, pvalue, limit)) {
    return return_value;
  }
  conn_t *conn = lupine_route_remote_conn(route);
  if (conn == nullptr || rpc_write_start_request(conn, RPC_cuCtxGetLimit) < 0 ||
      rpc_write(conn, &limit, sizeof(CUlimit)) < 0 ||
      rpc_wait_for_response(conn) < 0 ||
      rpc_read(conn, pvalue, sizeof(size_t)) < 0 ||
      rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
      rpc_read_end(conn) < 0)
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  return return_value;
}

CUresult cuCtxGetCacheConfig(CUfunc_cache *pconfig) {
  lupine_route route = lupine_route_for_default();
  CUresult return_value;
  using real_fn_t = CUresult (*)(CUfunc_cache *);
  if (lupine_call_local_cuda_if_routed<real_fn_t>(route, "cuCtxGetCacheConfig",
                                                  &return_value, pconfig)) {
    return return_value;
  }
  conn_t *conn = lupine_route_remote_conn(route);
  if (conn == nullptr ||
      rpc_write_start_request(conn, RPC_cuCtxGetCacheConfig) < 0 ||
      rpc_wait_for_response(conn) < 0 ||
      rpc_read(conn, pconfig, sizeof(CUfunc_cache)) < 0 ||
      rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
      rpc_read_end(conn) < 0)
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  return return_value;
}

CUresult cuCtxSetCacheConfig(CUfunc_cache config) {
  lupine_route route = lupine_route_for_default();
  CUresult return_value;
  using real_fn_t = CUresult (*)(CUfunc_cache);
  if (lupine_call_local_cuda_if_routed<real_fn_t>(route, "cuCtxSetCacheConfig",
                                                  &return_value, config)) {
    return return_value;
  }
  conn_t *conn = lupine_route_remote_conn(route);
  if (conn == nullptr ||
      rpc_write_start_request(conn, RPC_cuCtxSetCacheConfig) < 0 ||
      rpc_write(conn, &config, sizeof(CUfunc_cache)) < 0 ||
      rpc_wait_for_response(conn) < 0 ||
      rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
      rpc_read_end(conn) < 0)
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  return return_value;
}

CUresult cuCtxGetApiVersion(CUcontext ctx, unsigned int *version) {
  lupine_route route = lupine_route_for_context(ctx);
  CUresult return_value;
  using real_fn_t = CUresult (*)(CUcontext, unsigned int *);
  if (lupine_call_local_cuda_if_routed<real_fn_t>(
          route, "cuCtxGetApiVersion", &return_value, ctx, version)) {
    return return_value;
  }
  conn_t *conn = lupine_route_remote_conn(route);
  if (conn == nullptr ||
      rpc_write_start_request(conn, RPC_cuCtxGetApiVersion) < 0 ||
      rpc_write(conn, &ctx, sizeof(CUcontext)) < 0 ||
      rpc_wait_for_response(conn) < 0 ||
      rpc_read(conn, version, sizeof(unsigned int)) < 0 ||
      rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
      rpc_read_end(conn) < 0)
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  return return_value;
}

CUresult cuCtxResetPersistingL2Cache() {
  lupine_route route = lupine_route_for_default();
  CUresult return_value;
  using real_fn_t = CUresult (*)();
  if (lupine_call_local_cuda_if_routed<real_fn_t>(
          route, "cuCtxResetPersistingL2Cache", &return_value)) {
    return return_value;
  }
  conn_t *conn = lupine_route_remote_conn(route);
  if (conn == nullptr ||
      rpc_write_start_request(conn, RPC_cuCtxResetPersistingL2Cache) < 0 ||
      rpc_wait_for_response(conn) < 0 ||
      rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
      rpc_read_end(conn) < 0)
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  return return_value;
}

CUresult cuCtxGetExecAffinity(CUexecAffinityParam *pExecAffinity,
                              CUexecAffinityType type) {
  lupine_route route = lupine_route_for_default();
  CUresult return_value;
  using real_fn_t = CUresult (*)(CUexecAffinityParam *, CUexecAffinityType);
  if (lupine_call_local_cuda_if_routed<real_fn_t>(
          route, "cuCtxGetExecAffinity", &return_value, pExecAffinity, type)) {
    return return_value;
  }
  conn_t *conn = lupine_route_remote_conn(route);
  if (conn == nullptr ||
      rpc_write_start_request(conn, RPC_cuCtxGetExecAffinity) < 0 ||
      rpc_write(conn, &type, sizeof(CUexecAffinityType)) < 0 ||
      rpc_wait_for_response(conn) < 0 ||
      rpc_read(conn, pExecAffinity, sizeof(CUexecAffinityParam)) < 0 ||
      rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
      rpc_read_end(conn) < 0)
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  return return_value;
}

CUresult cuCtxAttach(CUcontext *pctx, unsigned int flags) {
  lupine_route route = lupine_route_for_default();
  CUresult return_value;
  using real_fn_t = CUresult (*)(CUcontext *, unsigned int);
  if (lupine_call_local_cuda_if_routed<real_fn_t>(route, "cuCtxAttach",
                                                  &return_value, pctx, flags)) {
    return return_value;
  }
  conn_t *conn = lupine_route_remote_conn(route);
  if (conn == nullptr || rpc_write_start_request(conn, RPC_cuCtxAttach) < 0 ||
      rpc_write(conn, &flags, sizeof(unsigned int)) < 0 ||
      rpc_wait_for_response(conn) < 0 ||
      rpc_read(conn, pctx, sizeof(CUcontext)) < 0 ||
      rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
      rpc_read_end(conn) < 0)
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  return return_value;
}

CUresult cuCtxDetach(CUcontext ctx) {
  lupine_route route = lupine_route_for_context(ctx);
  CUresult return_value;
  using real_fn_t = CUresult (*)(CUcontext);
  if (lupine_call_local_cuda_if_routed<real_fn_t>(route, "cuCtxDetach",
                                                  &return_value, ctx)) {
    if (return_value == CUDA_SUCCESS)
      lupine_invalidate_current_context_cache();
    if (return_value == CUDA_SUCCESS)
      lupine_invalidate_kernel_param_layout_cache();
    return return_value;
  }
  conn_t *conn = lupine_route_remote_conn(route);
  if (conn == nullptr || rpc_write_start_request(conn, RPC_cuCtxDetach) < 0 ||
      rpc_write(conn, &ctx, sizeof(CUcontext)) < 0 ||
      rpc_wait_for_response(conn) < 0 ||
      rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
      rpc_read_end(conn) < 0)
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  if (return_value == CUDA_SUCCESS)
    lupine_invalidate_current_context_cache();
  if (return_value == CUDA_SUCCESS)
    lupine_invalidate_kernel_param_layout_cache();
  return return_value;
}

CUresult cuCtxGetSharedMemConfig(CUsharedconfig *pConfig) {
  lupine_route route = lupine_route_for_default();
  CUresult return_value;
  using real_fn_t = CUresult (*)(CUsharedconfig *);
  if (lupine_call_local_cuda_if_routed<real_fn_t>(
          route, "cuCtxGetSharedMemConfig", &return_value, pConfig)) {
    return return_value;
  }
  conn_t *conn = lupine_route_remote_conn(route);
  if (conn == nullptr ||
      rpc_write_start_request(conn, RPC_cuCtxGetSharedMemConfig) < 0 ||
      rpc_wait_for_response(conn) < 0 ||
      rpc_read(conn, pConfig, sizeof(CUsharedconfig)) < 0 ||
      rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
      rpc_read_end(conn) < 0)
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  return return_value;
}

CUresult cuCtxSetSharedMemConfig(CUsharedconfig config) {
  lupine_route route = lupine_route_for_default();
  CUresult return_value;
  using real_fn_t = CUresult (*)(CUsharedconfig);
  if (lupine_call_local_cuda_if_routed<real_fn_t>(
          route, "cuCtxSetSharedMemConfig", &return_value, config)) {
    return return_value;
  }
  conn_t *conn = lupine_route_remote_conn(route);
  if (conn == nullptr ||
      rpc_write_start_request(conn, RPC_cuCtxSetSharedMemConfig) < 0 ||
      rpc_write(conn, &config, sizeof(CUsharedconfig)) < 0 ||
      rpc_wait_for_response(conn) < 0 ||
      rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
      rpc_read_end(conn) < 0)
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  return return_value;
}

CUresult cuModuleUnload(CUmodule hmod) {
  lupine_route route = lupine_route_for_module(hmod);
  CUresult return_value;
  using real_fn_t = CUresult (*)(CUmodule);
  if (lupine_call_local_cuda_if_routed<real_fn_t>(route, "cuModuleUnload",
                                                  &return_value, hmod)) {
    if (return_value == CUDA_SUCCESS)
      lupine_invalidate_kernel_param_layout_cache();
    return return_value;
  }
  conn_t *conn = lupine_route_remote_conn(route);
  if (conn == nullptr ||
      rpc_write_start_request(conn, RPC_cuModuleUnload) < 0 ||
      rpc_write(conn, &hmod, sizeof(CUmodule)) < 0 ||
      rpc_wait_for_response(conn) < 0 ||
      rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
      rpc_read_end(conn) < 0)
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  if (return_value == CUDA_SUCCESS)
    lupine_invalidate_kernel_param_layout_cache();
  return return_value;
}

CUresult cuModuleGetLoadingMode(CUmoduleLoadingMode *mode) {
  lupine_route route = lupine_route_for_default();
  CUresult return_value;
  using real_fn_t = CUresult (*)(CUmoduleLoadingMode *);
  if (lupine_call_local_cuda_if_routed<real_fn_t>(
          route, "cuModuleGetLoadingMode", &return_value, mode)) {
    return return_value;
  }
  conn_t *conn = lupine_route_remote_conn(route);
  if (conn == nullptr ||
      rpc_write_start_request(conn, RPC_cuModuleGetLoadingMode) < 0 ||
      rpc_write(conn, mode, sizeof(CUmoduleLoadingMode)) < 0 ||
      rpc_wait_for_response(conn) < 0 ||
      rpc_read(conn, mode, sizeof(CUmoduleLoadingMode)) < 0 ||
      rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
      rpc_read_end(conn) < 0)
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  return return_value;
}

CUresult cuModuleGetFunction(CUfunction *hfunc, CUmodule hmod,
                             const char *name) {
  lupine_route route = lupine_route_for_module(hmod);
  CUresult return_value;
  using real_fn_t = CUresult (*)(CUfunction *, CUmodule, const char *);
  if (lupine_call_local_cuda_if_routed<real_fn_t>(
          route, "cuModuleGetFunction", &return_value, hfunc, hmod, name)) {
    if (return_value == CUDA_SUCCESS && hfunc != nullptr) {
      lupine_note_function_owner_route(*hfunc, route);
    }
    if (return_value == CUDA_SUCCESS && hfunc != nullptr)
      lupine_record_module_function(*hfunc, hmod, name, route);
    return return_value;
  }
  conn_t *conn = lupine_route_remote_conn(route);
  std::size_t name_len = std::strlen(name) + 1;
  if (conn == nullptr ||
      rpc_write_start_request(conn, RPC_cuModuleGetFunction) < 0 ||
      rpc_write(conn, &hmod, sizeof(CUmodule)) < 0 ||
      rpc_write(conn, &name_len, sizeof(std::size_t)) < 0 ||
      rpc_write(conn, name, name_len) < 0 || rpc_wait_for_response(conn) < 0 ||
      rpc_read(conn, hfunc, sizeof(CUfunction)) < 0 ||
      rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
      rpc_read_end(conn) < 0)
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  if (return_value == CUDA_SUCCESS && hfunc != nullptr) {
    lupine_note_function_owner_route(*hfunc, route);
  }
  if (return_value == CUDA_SUCCESS && hfunc != nullptr)
    lupine_record_module_function(*hfunc, hmod, name, route);
  return return_value;
}

CUresult cuModuleGetGlobal_v2(CUdeviceptr *dptr, size_t *bytes, CUmodule hmod,
                              const char *name) {
  conn_t *conn = lupine_rpc_conn_for_module(hmod);
  CUresult return_value;
  size_t remote_bytes = 0;
  std::size_t name_len = std::strlen(name) + 1;
  if (conn == nullptr ||
      rpc_write_start_request(conn, RPC_cuModuleGetGlobal_v2) < 0 ||
      rpc_write(conn, &hmod, sizeof(CUmodule)) < 0 ||
      rpc_write(conn, &name_len, sizeof(std::size_t)) < 0 ||
      rpc_write(conn, name, name_len) < 0 || rpc_wait_for_response(conn) < 0 ||
      (dptr != nullptr && rpc_read(conn, dptr, sizeof(CUdeviceptr)) < 0) ||
      rpc_read(conn, &remote_bytes, sizeof(size_t)) < 0 ||
      rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
      rpc_read_end(conn) < 0)
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  if (bytes != nullptr)
    *bytes = remote_bytes;
  if (return_value == CUDA_SUCCESS && dptr != nullptr)
    lupine_note_deviceptr_allocation(*dptr, remote_bytes, conn);
  return return_value;
}

CUresult cuModuleGetTexRef(CUtexref *pTexRef, CUmodule hmod, const char *name) {
  lupine_route route = lupine_route_for_module(hmod);
  CUresult return_value;
  using real_fn_t = CUresult (*)(CUtexref *, CUmodule, const char *);
  if (lupine_call_local_cuda_if_routed<real_fn_t>(
          route, "cuModuleGetTexRef", &return_value, pTexRef, hmod, name)) {
    return return_value;
  }
  conn_t *conn = lupine_route_remote_conn(route);
  std::size_t name_len = std::strlen(name) + 1;
  if (conn == nullptr ||
      rpc_write_start_request(conn, RPC_cuModuleGetTexRef) < 0 ||
      rpc_write(conn, &hmod, sizeof(CUmodule)) < 0 ||
      rpc_write(conn, &name_len, sizeof(std::size_t)) < 0 ||
      rpc_write(conn, name, name_len) < 0 || rpc_wait_for_response(conn) < 0 ||
      rpc_read(conn, pTexRef, sizeof(CUtexref)) < 0 ||
      rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
      rpc_read_end(conn) < 0)
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  return return_value;
}

CUresult cuModuleGetSurfRef(CUsurfref *pSurfRef, CUmodule hmod,
                            const char *name) {
  lupine_route route = lupine_route_for_module(hmod);
  CUresult return_value;
  using real_fn_t = CUresult (*)(CUsurfref *, CUmodule, const char *);
  if (lupine_call_local_cuda_if_routed<real_fn_t>(
          route, "cuModuleGetSurfRef", &return_value, pSurfRef, hmod, name)) {
    return return_value;
  }
  conn_t *conn = lupine_route_remote_conn(route);
  std::size_t name_len = std::strlen(name) + 1;
  if (conn == nullptr ||
      rpc_write_start_request(conn, RPC_cuModuleGetSurfRef) < 0 ||
      rpc_write(conn, &hmod, sizeof(CUmodule)) < 0 ||
      rpc_write(conn, &name_len, sizeof(std::size_t)) < 0 ||
      rpc_write(conn, name, name_len) < 0 || rpc_wait_for_response(conn) < 0 ||
      rpc_read(conn, pSurfRef, sizeof(CUsurfref)) < 0 ||
      rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
      rpc_read_end(conn) < 0)
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  return return_value;
}

CUresult cuLibraryLoadFromFile(CUlibrary *library, const char *fileName,
                               CUjit_option *jitOptions,
                               void **jitOptionsValues,
                               unsigned int numJitOptions,
                               CUlibraryOption *libraryOptions,
                               void **libraryOptionValues,
                               unsigned int numLibraryOptions) {
  lupine_route route = lupine_route_for_default();
  CUresult return_value;
  using real_fn_t =
      CUresult (*)(CUlibrary *, const char *, CUjit_option *, void **,
                   unsigned int, CUlibraryOption *, void **, unsigned int);
  if (lupine_call_local_cuda_if_routed<real_fn_t>(
          route, "cuLibraryLoadFromFile", &return_value, library, fileName,
          jitOptions, jitOptionsValues, numJitOptions, libraryOptions,
          libraryOptionValues, numLibraryOptions)) {
    if (return_value == CUDA_SUCCESS && library != nullptr) {
      lupine_note_library_owner_route(*library, route);
    }
    return return_value;
  }
  conn_t *conn = lupine_route_remote_conn(route);
  std::size_t fileName_len = std::strlen(fileName) + 1;
  if (numJitOptions * sizeof(CUjit_option) != 0 && jitOptions == nullptr)
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  if (numJitOptions * sizeof(void *) != 0 && jitOptionsValues == nullptr)
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  if (numLibraryOptions * sizeof(CUlibraryOption) != 0 &&
      libraryOptions == nullptr)
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  if (numLibraryOptions * sizeof(void *) != 0 && libraryOptionValues == nullptr)
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  if (conn == nullptr ||
      rpc_write_start_request(conn, RPC_cuLibraryLoadFromFile) < 0 ||
      rpc_write(conn, &fileName_len, sizeof(std::size_t)) < 0 ||
      rpc_write(conn, fileName, fileName_len) < 0 ||
      rpc_write(conn, &numJitOptions, sizeof(unsigned int)) < 0 ||
      (numJitOptions * sizeof(CUjit_option) != 0 &&
       rpc_write(conn, jitOptions, numJitOptions * sizeof(CUjit_option)) < 0) ||
      (numJitOptions * sizeof(void *) != 0 &&
       rpc_write(conn, jitOptionsValues, numJitOptions * sizeof(void *)) < 0) ||
      rpc_write(conn, &numLibraryOptions, sizeof(unsigned int)) < 0 ||
      (numLibraryOptions * sizeof(CUlibraryOption) != 0 &&
       rpc_write(conn, libraryOptions,
                 numLibraryOptions * sizeof(CUlibraryOption)) < 0) ||
      (numLibraryOptions * sizeof(void *) != 0 &&
       rpc_write(conn, libraryOptionValues,
                 numLibraryOptions * sizeof(void *)) < 0) ||
      rpc_wait_for_response(conn) < 0 ||
      rpc_read(conn, library, sizeof(CUlibrary)) < 0 ||
      rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
      rpc_read_end(conn) < 0)
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  if (return_value == CUDA_SUCCESS && library != nullptr) {
    lupine_note_library_owner_route(*library, route);
  }
  return return_value;
}

CUresult cuLibraryUnload(CUlibrary library) {
  lupine_route route = lupine_route_for_library(library);
  CUresult return_value;
  using real_fn_t = CUresult (*)(CUlibrary);
  if (lupine_call_local_cuda_if_routed<real_fn_t>(route, "cuLibraryUnload",
                                                  &return_value, library)) {
    if (return_value == CUDA_SUCCESS)
      lupine_invalidate_kernel_param_layout_cache();
    return return_value;
  }
  conn_t *conn = lupine_route_remote_conn(route);
  if (conn == nullptr ||
      rpc_write_start_request(conn, RPC_cuLibraryUnload) < 0 ||
      rpc_write(conn, &library, sizeof(CUlibrary)) < 0 ||
      rpc_wait_for_response(conn) < 0 ||
      rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
      rpc_read_end(conn) < 0)
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  if (return_value == CUDA_SUCCESS)
    lupine_invalidate_kernel_param_layout_cache();
  return return_value;
}

CUresult cuLibraryGetKernel(CUkernel *pKernel, CUlibrary library,
                            const char *name) {
  lupine_route route = lupine_route_for_library(library);
  CUresult return_value;
  using real_fn_t = CUresult (*)(CUkernel *, CUlibrary, const char *);
  if (lupine_call_local_cuda_if_routed<real_fn_t>(
          route, "cuLibraryGetKernel", &return_value, pKernel, library, name)) {
    if (return_value == CUDA_SUCCESS && pKernel != nullptr)
      lupine_record_library_kernel(*pKernel, library, name, route);
    return return_value;
  }
  conn_t *conn = lupine_route_remote_conn(route);
  std::size_t name_len = std::strlen(name) + 1;
  if (conn == nullptr ||
      rpc_write_start_request(conn, RPC_cuLibraryGetKernel) < 0 ||
      rpc_write(conn, &library, sizeof(CUlibrary)) < 0 ||
      rpc_write(conn, &name_len, sizeof(std::size_t)) < 0 ||
      rpc_write(conn, name, name_len) < 0 || rpc_wait_for_response(conn) < 0 ||
      rpc_read(conn, pKernel, sizeof(CUkernel)) < 0 ||
      rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
      rpc_read_end(conn) < 0)
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  if (return_value == CUDA_SUCCESS && pKernel != nullptr)
    lupine_record_library_kernel(*pKernel, library, name, route);
  return return_value;
}

CUresult cuLibraryGetModule(CUmodule *pMod, CUlibrary library) {
  lupine_route route = lupine_route_for_library(library);
  CUresult return_value;
  using real_fn_t = CUresult (*)(CUmodule *, CUlibrary);
  if (lupine_call_local_cuda_if_routed<real_fn_t>(
          route, "cuLibraryGetModule", &return_value, pMod, library)) {
    if (return_value == CUDA_SUCCESS && pMod != nullptr) {
      lupine_note_module_owner_route(*pMod, route);
    }
    return return_value;
  }
  conn_t *conn = lupine_route_remote_conn(route);
  if (conn == nullptr ||
      rpc_write_start_request(conn, RPC_cuLibraryGetModule) < 0 ||
      rpc_write(conn, &library, sizeof(CUlibrary)) < 0 ||
      rpc_wait_for_response(conn) < 0 ||
      rpc_read(conn, pMod, sizeof(CUmodule)) < 0 ||
      rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
      rpc_read_end(conn) < 0)
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  if (return_value == CUDA_SUCCESS && pMod != nullptr) {
    lupine_note_module_owner_route(*pMod, route);
  }
  return return_value;
}

CUresult cuKernelGetFunction(CUfunction *pFunc, CUkernel kernel) {
  return lupine_cuKernelGetFunction_cached(pFunc, kernel);
}

CUresult cuLibraryGetGlobal(CUdeviceptr *dptr, size_t *bytes, CUlibrary library,
                            const char *name) {
  lupine_route route = lupine_route_for_library(library);
  conn_t *conn = lupine_route_remote_conn(route);
  CUresult return_value;
  size_t remote_bytes = 0;
  std::size_t name_len = std::strlen(name) + 1;
  if (conn == nullptr ||
      rpc_write_start_request(conn, RPC_cuLibraryGetGlobal) < 0 ||
      rpc_write(conn, &library, sizeof(CUlibrary)) < 0 ||
      rpc_write(conn, &name_len, sizeof(std::size_t)) < 0 ||
      rpc_write(conn, name, name_len) < 0 || rpc_wait_for_response(conn) < 0 ||
      (dptr != nullptr && rpc_read(conn, dptr, sizeof(CUdeviceptr)) < 0) ||
      rpc_read(conn, &remote_bytes, sizeof(size_t)) < 0 ||
      rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
      rpc_read_end(conn) < 0)
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  if (bytes != nullptr)
    *bytes = remote_bytes;
  if (return_value == CUDA_SUCCESS && dptr != nullptr)
    lupine_note_deviceptr_allocation(*dptr, remote_bytes, conn);
  return return_value;
}

CUresult cuLibraryGetManaged(CUdeviceptr *dptr, size_t *bytes,
                             CUlibrary library, const char *name) {
  lupine_route route = lupine_route_for_library(library);
  conn_t *conn = lupine_route_remote_conn(route);
  CUresult return_value;
  size_t remote_bytes = 0;
  std::size_t name_len = std::strlen(name) + 1;
  if (conn == nullptr ||
      rpc_write_start_request(conn, RPC_cuLibraryGetManaged) < 0 ||
      rpc_write(conn, &library, sizeof(CUlibrary)) < 0 ||
      rpc_write(conn, &name_len, sizeof(std::size_t)) < 0 ||
      rpc_write(conn, name, name_len) < 0 || rpc_wait_for_response(conn) < 0 ||
      (dptr != nullptr && rpc_read(conn, dptr, sizeof(CUdeviceptr)) < 0) ||
      rpc_read(conn, &remote_bytes, sizeof(size_t)) < 0 ||
      rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
      rpc_read_end(conn) < 0)
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  if (bytes != nullptr)
    *bytes = remote_bytes;
  if (return_value == CUDA_SUCCESS && dptr != nullptr)
    lupine_note_deviceptr_allocation(*dptr, remote_bytes, conn);
  return return_value;
}

CUresult cuLibraryGetUnifiedFunction(void **fptr, CUlibrary library,
                                     const char *symbol) {
  lupine_route route = lupine_route_for_library(library);
  CUresult return_value;
  using real_fn_t = CUresult (*)(void **, CUlibrary, const char *);
  if (lupine_call_local_cuda_if_routed<real_fn_t>(
          route, "cuLibraryGetUnifiedFunction", &return_value, fptr, library,
          symbol)) {
    return return_value;
  }
  conn_t *conn = lupine_route_remote_conn(route);
  std::size_t symbol_len = std::strlen(symbol) + 1;
  if (conn == nullptr ||
      rpc_write_start_request(conn, RPC_cuLibraryGetUnifiedFunction) < 0 ||
      rpc_write(conn, &library, sizeof(CUlibrary)) < 0 ||
      rpc_write(conn, &symbol_len, sizeof(std::size_t)) < 0 ||
      rpc_write(conn, symbol, symbol_len) < 0 ||
      rpc_wait_for_response(conn) < 0 ||
      rpc_read(conn, fptr, sizeof(void *)) < 0 ||
      rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
      rpc_read_end(conn) < 0)
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  return return_value;
}

CUresult cuKernelGetAttribute(int *pi, CUfunction_attribute attrib,
                              CUkernel kernel, CUdevice dev) {
  lupine_route route = lupine_route_for_device(&dev);
  CUresult return_value;
  using real_fn_t =
      CUresult (*)(int *, CUfunction_attribute, CUkernel, CUdevice);
  if (lupine_call_local_cuda_if_routed<real_fn_t>(route, "cuKernelGetAttribute",
                                                  &return_value, pi, attrib,
                                                  kernel, dev)) {
    return return_value;
  }
  conn_t *conn = lupine_route_remote_conn(route);
  if (conn == nullptr ||
      rpc_write_start_request(conn, RPC_cuKernelGetAttribute) < 0 ||
      rpc_write(conn, pi, sizeof(int)) < 0 ||
      rpc_write(conn, &attrib, sizeof(CUfunction_attribute)) < 0 ||
      rpc_write(conn, &kernel, sizeof(CUkernel)) < 0 ||
      rpc_write(conn, &dev, sizeof(CUdevice)) < 0 ||
      rpc_wait_for_response(conn) < 0 || rpc_read(conn, pi, sizeof(int)) < 0 ||
      rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
      rpc_read_end(conn) < 0)
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  return return_value;
}

CUresult cuKernelSetAttribute(CUfunction_attribute attrib, int val,
                              CUkernel kernel, CUdevice dev) {
  lupine_route route = lupine_route_for_device(&dev);
  CUresult return_value;
  using real_fn_t = CUresult (*)(CUfunction_attribute, int, CUkernel, CUdevice);
  if (lupine_call_local_cuda_if_routed<real_fn_t>(route, "cuKernelSetAttribute",
                                                  &return_value, attrib, val,
                                                  kernel, dev)) {
    return return_value;
  }
  conn_t *conn = lupine_route_remote_conn(route);
  if (conn == nullptr ||
      rpc_write_start_request(conn, RPC_cuKernelSetAttribute) < 0 ||
      rpc_write(conn, &attrib, sizeof(CUfunction_attribute)) < 0 ||
      rpc_write(conn, &val, sizeof(int)) < 0 ||
      rpc_write(conn, &kernel, sizeof(CUkernel)) < 0 ||
      rpc_write(conn, &dev, sizeof(CUdevice)) < 0 ||
      rpc_wait_for_response(conn) < 0 ||
      rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
      rpc_read_end(conn) < 0)
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  return return_value;
}

CUresult cuKernelSetCacheConfig(CUkernel kernel, CUfunc_cache config,
                                CUdevice dev) {
  lupine_route route = lupine_route_for_device(&dev);
  CUresult return_value;
  using real_fn_t = CUresult (*)(CUkernel, CUfunc_cache, CUdevice);
  if (lupine_call_local_cuda_if_routed<real_fn_t>(
          route, "cuKernelSetCacheConfig", &return_value, kernel, config,
          dev)) {
    return return_value;
  }
  conn_t *conn = lupine_route_remote_conn(route);
  if (conn == nullptr ||
      rpc_write_start_request(conn, RPC_cuKernelSetCacheConfig) < 0 ||
      rpc_write(conn, &kernel, sizeof(CUkernel)) < 0 ||
      rpc_write(conn, &config, sizeof(CUfunc_cache)) < 0 ||
      rpc_write(conn, &dev, sizeof(CUdevice)) < 0 ||
      rpc_wait_for_response(conn) < 0 ||
      rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
      rpc_read_end(conn) < 0)
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  return return_value;
}

CUresult cuMemGetInfo_v2(size_t *free, size_t *total) {
  lupine_route route = lupine_route_for_current_context();
  CUresult return_value;
  using real_fn_t = CUresult (*)(size_t *, size_t *);
  if (lupine_call_local_cuda_if_routed<real_fn_t>(route, "cuMemGetInfo_v2",
                                                  &return_value, free, total)) {
    return return_value;
  }
  conn_t *conn = lupine_route_remote_conn(route);
  if (conn == nullptr ||
      rpc_write_start_request(conn, RPC_cuMemGetInfo_v2) < 0 ||
      rpc_write(conn, free, sizeof(size_t)) < 0 ||
      rpc_write(conn, total, sizeof(size_t)) < 0 ||
      rpc_wait_for_response(conn) < 0 ||
      rpc_read(conn, free, sizeof(size_t)) < 0 ||
      rpc_read(conn, total, sizeof(size_t)) < 0 ||
      rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
      rpc_read_end(conn) < 0)
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  return return_value;
}

CUresult cuMemAlloc_v2(CUdeviceptr *dptr, size_t bytesize) {
  lupine_route route = lupine_route_for_current_context();
  CUresult return_value;
  using real_fn_t = CUresult (*)(CUdeviceptr *, size_t);
  if (lupine_call_local_cuda_if_routed<real_fn_t>(
          route, "cuMemAlloc_v2", &return_value, dptr, bytesize)) {
    if (return_value == CUDA_SUCCESS && dptr != nullptr) {
      lupine_note_deviceptr_owner_route(*dptr, route);
    }
    if (return_value == CUDA_SUCCESS && dptr != nullptr)
      lupine_note_deviceptr_allocation_route(*dptr, bytesize, route);
    return return_value;
  }
  conn_t *conn = lupine_route_remote_conn(route);
  if (conn == nullptr || rpc_write_start_request(conn, RPC_cuMemAlloc_v2) < 0 ||
      rpc_write(conn, dptr, sizeof(CUdeviceptr)) < 0 ||
      rpc_write(conn, &bytesize, sizeof(size_t)) < 0 ||
      rpc_wait_for_response(conn) < 0 ||
      rpc_read(conn, dptr, sizeof(CUdeviceptr)) < 0 ||
      rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
      rpc_read_end(conn) < 0)
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  if (return_value == CUDA_SUCCESS && dptr != nullptr) {
    lupine_note_deviceptr_owner_route(*dptr, route);
  }
  if (return_value == CUDA_SUCCESS && dptr != nullptr)
    lupine_note_deviceptr_allocation_route(*dptr, bytesize, route);
  return return_value;
}

CUresult cuMemAllocPitch_v2(CUdeviceptr *dptr, size_t *pPitch,
                            size_t WidthInBytes, size_t Height,
                            unsigned int ElementSizeBytes) {
  lupine_route route = lupine_route_for_current_context();
  CUresult return_value;
  using real_fn_t =
      CUresult (*)(CUdeviceptr *, size_t *, size_t, size_t, unsigned int);
  if (lupine_call_local_cuda_if_routed<real_fn_t>(
          route, "cuMemAllocPitch_v2", &return_value, dptr, pPitch,
          WidthInBytes, Height, ElementSizeBytes)) {
    if (return_value == CUDA_SUCCESS && dptr != nullptr) {
      lupine_note_deviceptr_owner_route(*dptr, route);
    }
    if (return_value == CUDA_SUCCESS && dptr != nullptr) {
      size_t allocation_size = 0;
      if (pPitch != nullptr)
        allocation_size = (*pPitch) * Height;
      else
        allocation_size = WidthInBytes * Height;
      lupine_note_deviceptr_allocation_route(*dptr, allocation_size, route);
    }
    return return_value;
  }
  conn_t *conn = lupine_route_remote_conn(route);
  if (conn == nullptr ||
      rpc_write_start_request(conn, RPC_cuMemAllocPitch_v2) < 0 ||
      rpc_write(conn, dptr, sizeof(CUdeviceptr)) < 0 ||
      rpc_write(conn, pPitch, sizeof(size_t)) < 0 ||
      rpc_write(conn, &WidthInBytes, sizeof(size_t)) < 0 ||
      rpc_write(conn, &Height, sizeof(size_t)) < 0 ||
      rpc_write(conn, &ElementSizeBytes, sizeof(unsigned int)) < 0 ||
      rpc_wait_for_response(conn) < 0 ||
      rpc_read(conn, dptr, sizeof(CUdeviceptr)) < 0 ||
      rpc_read(conn, pPitch, sizeof(size_t)) < 0 ||
      rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
      rpc_read_end(conn) < 0)
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  if (return_value == CUDA_SUCCESS && dptr != nullptr) {
    lupine_note_deviceptr_owner_route(*dptr, route);
  }
  if (return_value == CUDA_SUCCESS && dptr != nullptr) {
    size_t allocation_size = 0;
    if (pPitch != nullptr)
      allocation_size = (*pPitch) * Height;
    else
      allocation_size = WidthInBytes * Height;
    lupine_note_deviceptr_allocation_route(*dptr, allocation_size, route);
  }
  return return_value;
}

CUresult cuMemGetAddressRange_v2(CUdeviceptr *pbase, size_t *psize,
                                 CUdeviceptr dptr) {
  lupine_route route = lupine_route_for_deviceptr(dptr);
  CUresult return_value;
  using real_fn_t = CUresult (*)(CUdeviceptr *, size_t *, CUdeviceptr);
  if (lupine_call_local_cuda_if_routed<real_fn_t>(
          route, "cuMemGetAddressRange_v2", &return_value, pbase, psize,
          dptr)) {
    return return_value;
  }
  conn_t *conn = lupine_route_remote_conn(route);
  if (conn == nullptr ||
      rpc_write_start_request(conn, RPC_cuMemGetAddressRange_v2) < 0 ||
      rpc_write(conn, pbase, sizeof(CUdeviceptr)) < 0 ||
      rpc_write(conn, psize, sizeof(size_t)) < 0 ||
      rpc_write(conn, &dptr, sizeof(CUdeviceptr)) < 0 ||
      rpc_wait_for_response(conn) < 0 ||
      rpc_read(conn, pbase, sizeof(CUdeviceptr)) < 0 ||
      rpc_read(conn, psize, sizeof(size_t)) < 0 ||
      rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
      rpc_read_end(conn) < 0)
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  return return_value;
}

CUresult cuDeviceGetByPCIBusId(CUdevice *dev, const char *pciBusId) {
  if (dev == nullptr || pciBusId == nullptr) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  return lupine_lookup_device_on_all_routes(
      dev, [&](lupine_route route, CUdevice *route_output) {
        CUdevice *dev = route_output;
        CUresult return_value;
        using real_fn_t = CUresult (*)(CUdevice *, const char *);
        if (lupine_call_local_cuda_if_routed<real_fn_t>(
                route, "cuDeviceGetByPCIBusId", &return_value, dev, pciBusId)) {
          return return_value;
        }
        conn_t *conn = lupine_route_remote_conn(route);
        CUdevice *dev_null_check;
        std::size_t pciBusId_len = std::strlen(pciBusId) + 1;
        if (conn == nullptr ||
            rpc_write_start_request(conn, RPC_cuDeviceGetByPCIBusId) < 0 ||
            rpc_write(conn, &dev, sizeof(CUdevice *)) < 0 ||
            rpc_write(conn, &pciBusId_len, sizeof(std::size_t)) < 0 ||
            rpc_write(conn, pciBusId, pciBusId_len) < 0 ||
            rpc_wait_for_response(conn) < 0 ||
            rpc_read(conn, &dev_null_check, sizeof(CUdevice *)) < 0 ||
            (dev_null_check && rpc_read(conn, dev, sizeof(CUdevice)) < 0) ||
            rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
            rpc_read_end(conn) < 0)
          return CUDA_ERROR_DEVICE_UNAVAILABLE;
        return return_value;
      });
}

CUresult cuDeviceGetPCIBusId(char *pciBusId, int len, CUdevice dev) {
  lupine_route route = lupine_route_for_device(&dev);
  CUresult return_value;
  using real_fn_t = CUresult (*)(char *, int, CUdevice);
  if (lupine_call_local_cuda_if_routed<real_fn_t>(
          route, "cuDeviceGetPCIBusId", &return_value, pciBusId, len, dev)) {
    return return_value;
  }
  conn_t *conn = lupine_route_remote_conn(route);
  if (conn == nullptr ||
      rpc_write_start_request(conn, RPC_cuDeviceGetPCIBusId) < 0 ||
      rpc_write(conn, &len, sizeof(int)) < 0 ||
      rpc_write(conn, &dev, sizeof(CUdevice)) < 0 ||
      rpc_wait_for_response(conn) < 0 ||
      (lupine_prepare_host_range_write(pciBusId, len * sizeof(char)), false) ||
      (len * sizeof(char) != 0 &&
       rpc_read(conn, pciBusId, len * sizeof(char)) < 0) ||
      rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
      rpc_read_end(conn) < 0)
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  if (return_value == CUDA_SUCCESS)
    lupine_mark_host_range_clean(pciBusId, len * sizeof(char));
  return return_value;
}

CUresult cuIpcGetEventHandle(CUipcEventHandle *pHandle, CUevent event) {
  lupine_route route = lupine_route_for_event(event);
  CUresult return_value;
  using real_fn_t = CUresult (*)(CUipcEventHandle *, CUevent);
  if (lupine_call_local_cuda_if_routed<real_fn_t>(
          route, "cuIpcGetEventHandle", &return_value, pHandle, event)) {
    return return_value;
  }
  conn_t *conn = lupine_route_remote_conn(route);
  if (conn == nullptr ||
      rpc_write_start_request(conn, RPC_cuIpcGetEventHandle) < 0 ||
      rpc_write(conn, pHandle, sizeof(CUipcEventHandle)) < 0 ||
      rpc_write(conn, &event, sizeof(CUevent)) < 0 ||
      rpc_wait_for_response(conn) < 0 ||
      rpc_read(conn, pHandle, sizeof(CUipcEventHandle)) < 0 ||
      rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
      rpc_read_end(conn) < 0)
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  return return_value;
}

CUresult cuIpcOpenEventHandle(CUevent *phEvent, CUipcEventHandle handle) {
  lupine_route route = lupine_route_for_default();
  CUresult return_value;
  using real_fn_t = CUresult (*)(CUevent *, CUipcEventHandle);
  if (lupine_call_local_cuda_if_routed<real_fn_t>(
          route, "cuIpcOpenEventHandle", &return_value, phEvent, handle)) {
    return return_value;
  }
  conn_t *conn = lupine_route_remote_conn(route);
  if (conn == nullptr ||
      rpc_write_start_request(conn, RPC_cuIpcOpenEventHandle) < 0 ||
      rpc_write(conn, phEvent, sizeof(CUevent)) < 0 ||
      rpc_write(conn, &handle, sizeof(CUipcEventHandle)) < 0 ||
      rpc_wait_for_response(conn) < 0 ||
      rpc_read(conn, phEvent, sizeof(CUevent)) < 0 ||
      rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
      rpc_read_end(conn) < 0)
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  return return_value;
}

CUresult cuIpcGetMemHandle(CUipcMemHandle *pHandle, CUdeviceptr dptr) {
  lupine_route route = lupine_route_for_deviceptr(dptr);
  CUresult return_value;
  using real_fn_t = CUresult (*)(CUipcMemHandle *, CUdeviceptr);
  if (lupine_call_local_cuda_if_routed<real_fn_t>(
          route, "cuIpcGetMemHandle", &return_value, pHandle, dptr)) {
    return return_value;
  }
  conn_t *conn = lupine_route_remote_conn(route);
  if (conn == nullptr ||
      rpc_write_start_request(conn, RPC_cuIpcGetMemHandle) < 0 ||
      rpc_write(conn, pHandle, sizeof(CUipcMemHandle)) < 0 ||
      rpc_write(conn, &dptr, sizeof(CUdeviceptr)) < 0 ||
      rpc_wait_for_response(conn) < 0 ||
      rpc_read(conn, pHandle, sizeof(CUipcMemHandle)) < 0 ||
      rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
      rpc_read_end(conn) < 0)
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  return return_value;
}

CUresult cuIpcOpenMemHandle_v2(CUdeviceptr *pdptr, CUipcMemHandle handle,
                               unsigned int Flags) {
  lupine_route route = lupine_route_for_default();
  CUresult return_value;
  using real_fn_t = CUresult (*)(CUdeviceptr *, CUipcMemHandle, unsigned int);
  if (lupine_call_local_cuda_if_routed<real_fn_t>(
          route, "cuIpcOpenMemHandle_v2", &return_value, pdptr, handle,
          Flags)) {
    return return_value;
  }
  conn_t *conn = lupine_route_remote_conn(route);
  if (conn == nullptr ||
      rpc_write_start_request(conn, RPC_cuIpcOpenMemHandle_v2) < 0 ||
      rpc_write(conn, pdptr, sizeof(CUdeviceptr)) < 0 ||
      rpc_write(conn, &handle, sizeof(CUipcMemHandle)) < 0 ||
      rpc_write(conn, &Flags, sizeof(unsigned int)) < 0 ||
      rpc_wait_for_response(conn) < 0 ||
      rpc_read(conn, pdptr, sizeof(CUdeviceptr)) < 0 ||
      rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
      rpc_read_end(conn) < 0)
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  return return_value;
}

CUresult cuIpcCloseMemHandle(CUdeviceptr dptr) {
  lupine_route route = lupine_route_for_deviceptr(dptr);
  CUresult return_value;
  using real_fn_t = CUresult (*)(CUdeviceptr);
  if (lupine_call_local_cuda_if_routed<real_fn_t>(route, "cuIpcCloseMemHandle",
                                                  &return_value, dptr)) {
    return return_value;
  }
  conn_t *conn = lupine_route_remote_conn(route);
  if (conn == nullptr ||
      rpc_write_start_request(conn, RPC_cuIpcCloseMemHandle) < 0 ||
      rpc_write(conn, &dptr, sizeof(CUdeviceptr)) < 0 ||
      rpc_wait_for_response(conn) < 0 ||
      rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
      rpc_read_end(conn) < 0)
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  return return_value;
}

CUresult cuMemcpy(CUdeviceptr dst, CUdeviceptr src, size_t ByteCount) {
  lupine_route route = lupine_route_for_deviceptr(dst);
  if (!lupine_deviceptrs_share_route(dst, src)) {
    return lupine_cuMemcpyDtoD_via_client(dst, src, ByteCount, nullptr, false);
  }
  CUresult return_value;
  using real_fn_t = CUresult (*)(CUdeviceptr, CUdeviceptr, size_t);
  if (lupine_call_local_cuda_if_routed<real_fn_t>(
          route, "cuMemcpy", &return_value, dst, src, ByteCount)) {
    return return_value;
  }
  conn_t *conn = lupine_route_remote_conn(route);
  if (conn == nullptr || rpc_write_start_request(conn, RPC_cuMemcpy) < 0 ||
      rpc_write(conn, &dst, sizeof(CUdeviceptr)) < 0 ||
      rpc_write(conn, &src, sizeof(CUdeviceptr)) < 0 ||
      rpc_write(conn, &ByteCount, sizeof(size_t)) < 0 ||
      rpc_wait_for_response(conn) < 0 ||
      rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
      rpc_read_end(conn) < 0)
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  return return_value;
}

CUresult cuMemcpyPeer(CUdeviceptr dstDevice, CUcontext dstContext,
                      CUdeviceptr srcDevice, CUcontext srcContext,
                      size_t ByteCount) {
  lupine_route route = lupine_route_for_deviceptr(dstDevice);
  if (!lupine_deviceptrs_share_route(dstDevice, srcDevice)) {
    return lupine_cuMemcpyDtoD_via_client(dstDevice, srcDevice, ByteCount,
                                          nullptr, false);
  }
  CUresult return_value;
  using real_fn_t =
      CUresult (*)(CUdeviceptr, CUcontext, CUdeviceptr, CUcontext, size_t);
  if (lupine_call_local_cuda_if_routed<real_fn_t>(
          route, "cuMemcpyPeer", &return_value, dstDevice, dstContext,
          srcDevice, srcContext, ByteCount)) {
    return return_value;
  }
  conn_t *conn = lupine_route_remote_conn(route);
  if (conn == nullptr || rpc_write_start_request(conn, RPC_cuMemcpyPeer) < 0 ||
      rpc_write(conn, &dstDevice, sizeof(CUdeviceptr)) < 0 ||
      rpc_write(conn, &dstContext, sizeof(CUcontext)) < 0 ||
      rpc_write(conn, &srcDevice, sizeof(CUdeviceptr)) < 0 ||
      rpc_write(conn, &srcContext, sizeof(CUcontext)) < 0 ||
      rpc_write(conn, &ByteCount, sizeof(size_t)) < 0 ||
      rpc_wait_for_response(conn) < 0 ||
      rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
      rpc_read_end(conn) < 0)
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  return return_value;
}

CUresult cuMemcpyDtoD_v2(CUdeviceptr dstDevice, CUdeviceptr srcDevice,
                         size_t ByteCount) {
  lupine_route route = lupine_route_for_deviceptr(dstDevice);
  if (!lupine_deviceptrs_share_route(dstDevice, srcDevice)) {
    return lupine_cuMemcpyDtoD_via_client(dstDevice, srcDevice, ByteCount,
                                          nullptr, false);
  }
  CUresult return_value;
  using real_fn_t = CUresult (*)(CUdeviceptr, CUdeviceptr, size_t);
  if (lupine_call_local_cuda_if_routed<real_fn_t>(route, "cuMemcpyDtoD_v2",
                                                  &return_value, dstDevice,
                                                  srcDevice, ByteCount)) {
    return return_value;
  }
  conn_t *conn = lupine_route_remote_conn(route);
  if (conn == nullptr ||
      rpc_write_start_request(conn, RPC_cuMemcpyDtoD_v2) < 0 ||
      rpc_write(conn, &dstDevice, sizeof(CUdeviceptr)) < 0 ||
      rpc_write(conn, &srcDevice, sizeof(CUdeviceptr)) < 0 ||
      rpc_write(conn, &ByteCount, sizeof(size_t)) < 0 ||
      rpc_wait_for_response(conn) < 0 ||
      rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
      rpc_read_end(conn) < 0)
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  return return_value;
}

CUresult cuMemcpyDtoA_v2(CUarray dstArray, size_t dstOffset,
                         CUdeviceptr srcDevice, size_t ByteCount) {
  lupine_route route = lupine_route_for_deviceptr(srcDevice);
  CUresult return_value;
  using real_fn_t = CUresult (*)(CUarray, size_t, CUdeviceptr, size_t);
  if (lupine_call_local_cuda_if_routed<real_fn_t>(
          route, "cuMemcpyDtoA_v2", &return_value, dstArray, dstOffset,
          srcDevice, ByteCount)) {
    return return_value;
  }
  conn_t *conn = lupine_route_remote_conn(route);
  if (conn == nullptr ||
      rpc_write_start_request(conn, RPC_cuMemcpyDtoA_v2) < 0 ||
      rpc_write(conn, &dstArray, sizeof(CUarray)) < 0 ||
      rpc_write(conn, &dstOffset, sizeof(size_t)) < 0 ||
      rpc_write(conn, &srcDevice, sizeof(CUdeviceptr)) < 0 ||
      rpc_write(conn, &ByteCount, sizeof(size_t)) < 0 ||
      rpc_wait_for_response(conn) < 0 ||
      rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
      rpc_read_end(conn) < 0)
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  return return_value;
}

CUresult cuMemcpyAtoD_v2(CUdeviceptr dstDevice, CUarray srcArray,
                         size_t srcOffset, size_t ByteCount) {
  lupine_route route = lupine_route_for_deviceptr(dstDevice);
  CUresult return_value;
  using real_fn_t = CUresult (*)(CUdeviceptr, CUarray, size_t, size_t);
  if (lupine_call_local_cuda_if_routed<real_fn_t>(
          route, "cuMemcpyAtoD_v2", &return_value, dstDevice, srcArray,
          srcOffset, ByteCount)) {
    return return_value;
  }
  conn_t *conn = lupine_route_remote_conn(route);
  if (conn == nullptr ||
      rpc_write_start_request(conn, RPC_cuMemcpyAtoD_v2) < 0 ||
      rpc_write(conn, &dstDevice, sizeof(CUdeviceptr)) < 0 ||
      rpc_write(conn, &srcArray, sizeof(CUarray)) < 0 ||
      rpc_write(conn, &srcOffset, sizeof(size_t)) < 0 ||
      rpc_write(conn, &ByteCount, sizeof(size_t)) < 0 ||
      rpc_wait_for_response(conn) < 0 ||
      rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
      rpc_read_end(conn) < 0)
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  return return_value;
}

CUresult cuMemcpyAtoA_v2(CUarray dstArray, size_t dstOffset, CUarray srcArray,
                         size_t srcOffset, size_t ByteCount) {
  lupine_route route = lupine_route_for_default();
  CUresult return_value;
  using real_fn_t = CUresult (*)(CUarray, size_t, CUarray, size_t, size_t);
  if (lupine_call_local_cuda_if_routed<real_fn_t>(
          route, "cuMemcpyAtoA_v2", &return_value, dstArray, dstOffset,
          srcArray, srcOffset, ByteCount)) {
    return return_value;
  }
  conn_t *conn = lupine_route_remote_conn(route);
  if (conn == nullptr ||
      rpc_write_start_request(conn, RPC_cuMemcpyAtoA_v2) < 0 ||
      rpc_write(conn, &dstArray, sizeof(CUarray)) < 0 ||
      rpc_write(conn, &dstOffset, sizeof(size_t)) < 0 ||
      rpc_write(conn, &srcArray, sizeof(CUarray)) < 0 ||
      rpc_write(conn, &srcOffset, sizeof(size_t)) < 0 ||
      rpc_write(conn, &ByteCount, sizeof(size_t)) < 0 ||
      rpc_wait_for_response(conn) < 0 ||
      rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
      rpc_read_end(conn) < 0)
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  return return_value;
}

CUresult cuMemcpyPeerAsync(CUdeviceptr dstDevice, CUcontext dstContext,
                           CUdeviceptr srcDevice, CUcontext srcContext,
                           size_t ByteCount, CUstream hStream) {
  lupine_route route = lupine_route_for_deviceptr(dstDevice);
  if (!lupine_deviceptrs_share_route(dstDevice, srcDevice)) {
    return lupine_cuMemcpyDtoD_via_client(dstDevice, srcDevice, ByteCount,
                                          hStream, true);
  }
  CUresult return_value;
  using real_fn_t = CUresult (*)(CUdeviceptr, CUcontext, CUdeviceptr, CUcontext,
                                 size_t, CUstream);
  if (lupine_call_local_cuda_if_routed<real_fn_t>(
          route, "cuMemcpyPeerAsync", &return_value, dstDevice, dstContext,
          srcDevice, srcContext, ByteCount, hStream)) {
    return return_value;
  }
  conn_t *conn = lupine_route_remote_conn(route);
  if (conn == nullptr ||
      rpc_write_start_request(conn, RPC_cuMemcpyPeerAsync) < 0 ||
      rpc_write(conn, &dstDevice, sizeof(CUdeviceptr)) < 0 ||
      rpc_write(conn, &dstContext, sizeof(CUcontext)) < 0 ||
      rpc_write(conn, &srcDevice, sizeof(CUdeviceptr)) < 0 ||
      rpc_write(conn, &srcContext, sizeof(CUcontext)) < 0 ||
      rpc_write(conn, &ByteCount, sizeof(size_t)) < 0 ||
      rpc_write(conn, &hStream, sizeof(CUstream)) < 0 ||
      rpc_wait_for_response(conn) < 0 ||
      rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
      rpc_read_end(conn) < 0)
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  return return_value;
}

CUresult cuMemcpyDtoDAsync_v2(CUdeviceptr dstDevice, CUdeviceptr srcDevice,
                              size_t ByteCount, CUstream hStream) {
  lupine_route route = lupine_route_for_deviceptr(dstDevice);
  if (!lupine_deviceptrs_share_route(dstDevice, srcDevice)) {
    return lupine_cuMemcpyDtoD_via_client(dstDevice, srcDevice, ByteCount,
                                          hStream, true);
  }
  CUresult return_value;
  using real_fn_t = CUresult (*)(CUdeviceptr, CUdeviceptr, size_t, CUstream);
  if (lupine_call_local_cuda_if_routed<real_fn_t>(
          route, "cuMemcpyDtoDAsync_v2", &return_value, dstDevice, srcDevice,
          ByteCount, hStream)) {
    return return_value;
  }
  conn_t *conn = lupine_route_remote_conn(route);
  if (conn == nullptr ||
      rpc_write_start_request(conn, RPC_cuMemcpyDtoDAsync_v2) < 0 ||
      rpc_write(conn, &dstDevice, sizeof(CUdeviceptr)) < 0 ||
      rpc_write(conn, &srcDevice, sizeof(CUdeviceptr)) < 0 ||
      rpc_write(conn, &ByteCount, sizeof(size_t)) < 0 ||
      rpc_write(conn, &hStream, sizeof(CUstream)) < 0 ||
      rpc_write_end(conn) < 0) {
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  }
  return CUDA_SUCCESS;
}

CUresult cuMemsetD8_v2(CUdeviceptr dstDevice, unsigned char uc, size_t N) {
  lupine_route route = lupine_route_for_deviceptr(dstDevice);
  CUresult return_value;
  using real_fn_t = CUresult (*)(CUdeviceptr, unsigned char, size_t);
  if (lupine_call_local_cuda_if_routed<real_fn_t>(
          route, "cuMemsetD8_v2", &return_value, dstDevice, uc, N)) {
    return return_value;
  }
  conn_t *conn = lupine_route_remote_conn(route);
  if (conn == nullptr || rpc_write_start_request(conn, RPC_cuMemsetD8_v2) < 0 ||
      rpc_write(conn, &dstDevice, sizeof(CUdeviceptr)) < 0 ||
      rpc_write(conn, &uc, sizeof(unsigned char)) < 0 ||
      rpc_write(conn, &N, sizeof(size_t)) < 0 ||
      rpc_wait_for_response(conn) < 0 ||
      rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
      rpc_read_end(conn) < 0)
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  return return_value;
}

CUresult cuMemsetD16_v2(CUdeviceptr dstDevice, unsigned short us, size_t N) {
  lupine_route route = lupine_route_for_deviceptr(dstDevice);
  CUresult return_value;
  using real_fn_t = CUresult (*)(CUdeviceptr, unsigned short, size_t);
  if (lupine_call_local_cuda_if_routed<real_fn_t>(
          route, "cuMemsetD16_v2", &return_value, dstDevice, us, N)) {
    return return_value;
  }
  conn_t *conn = lupine_route_remote_conn(route);
  if (conn == nullptr ||
      rpc_write_start_request(conn, RPC_cuMemsetD16_v2) < 0 ||
      rpc_write(conn, &dstDevice, sizeof(CUdeviceptr)) < 0 ||
      rpc_write(conn, &us, sizeof(unsigned short)) < 0 ||
      rpc_write(conn, &N, sizeof(size_t)) < 0 ||
      rpc_wait_for_response(conn) < 0 ||
      rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
      rpc_read_end(conn) < 0)
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  return return_value;
}

CUresult cuMemsetD32_v2(CUdeviceptr dstDevice, unsigned int ui, size_t N) {
  lupine_route route = lupine_route_for_deviceptr(dstDevice);
  CUresult return_value;
  using real_fn_t = CUresult (*)(CUdeviceptr, unsigned int, size_t);
  if (lupine_call_local_cuda_if_routed<real_fn_t>(
          route, "cuMemsetD32_v2", &return_value, dstDevice, ui, N)) {
    return return_value;
  }
  conn_t *conn = lupine_route_remote_conn(route);
  if (conn == nullptr ||
      rpc_write_start_request(conn, RPC_cuMemsetD32_v2) < 0 ||
      rpc_write(conn, &dstDevice, sizeof(CUdeviceptr)) < 0 ||
      rpc_write(conn, &ui, sizeof(unsigned int)) < 0 ||
      rpc_write(conn, &N, sizeof(size_t)) < 0 ||
      rpc_wait_for_response(conn) < 0 ||
      rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
      rpc_read_end(conn) < 0)
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  return return_value;
}

CUresult cuMemsetD2D8_v2(CUdeviceptr dstDevice, size_t dstPitch,
                         unsigned char uc, size_t Width, size_t Height) {
  lupine_route route = lupine_route_for_deviceptr(dstDevice);
  CUresult return_value;
  using real_fn_t =
      CUresult (*)(CUdeviceptr, size_t, unsigned char, size_t, size_t);
  if (lupine_call_local_cuda_if_routed<real_fn_t>(
          route, "cuMemsetD2D8_v2", &return_value, dstDevice, dstPitch, uc,
          Width, Height)) {
    return return_value;
  }
  conn_t *conn = lupine_route_remote_conn(route);
  if (conn == nullptr ||
      rpc_write_start_request(conn, RPC_cuMemsetD2D8_v2) < 0 ||
      rpc_write(conn, &dstDevice, sizeof(CUdeviceptr)) < 0 ||
      rpc_write(conn, &dstPitch, sizeof(size_t)) < 0 ||
      rpc_write(conn, &uc, sizeof(unsigned char)) < 0 ||
      rpc_write(conn, &Width, sizeof(size_t)) < 0 ||
      rpc_write(conn, &Height, sizeof(size_t)) < 0 ||
      rpc_wait_for_response(conn) < 0 ||
      rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
      rpc_read_end(conn) < 0)
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  return return_value;
}

CUresult cuMemsetD2D16_v2(CUdeviceptr dstDevice, size_t dstPitch,
                          unsigned short us, size_t Width, size_t Height) {
  lupine_route route = lupine_route_for_deviceptr(dstDevice);
  CUresult return_value;
  using real_fn_t =
      CUresult (*)(CUdeviceptr, size_t, unsigned short, size_t, size_t);
  if (lupine_call_local_cuda_if_routed<real_fn_t>(
          route, "cuMemsetD2D16_v2", &return_value, dstDevice, dstPitch, us,
          Width, Height)) {
    return return_value;
  }
  conn_t *conn = lupine_route_remote_conn(route);
  if (conn == nullptr ||
      rpc_write_start_request(conn, RPC_cuMemsetD2D16_v2) < 0 ||
      rpc_write(conn, &dstDevice, sizeof(CUdeviceptr)) < 0 ||
      rpc_write(conn, &dstPitch, sizeof(size_t)) < 0 ||
      rpc_write(conn, &us, sizeof(unsigned short)) < 0 ||
      rpc_write(conn, &Width, sizeof(size_t)) < 0 ||
      rpc_write(conn, &Height, sizeof(size_t)) < 0 ||
      rpc_wait_for_response(conn) < 0 ||
      rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
      rpc_read_end(conn) < 0)
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  return return_value;
}

CUresult cuMemsetD2D32_v2(CUdeviceptr dstDevice, size_t dstPitch,
                          unsigned int ui, size_t Width, size_t Height) {
  lupine_route route = lupine_route_for_deviceptr(dstDevice);
  CUresult return_value;
  using real_fn_t =
      CUresult (*)(CUdeviceptr, size_t, unsigned int, size_t, size_t);
  if (lupine_call_local_cuda_if_routed<real_fn_t>(
          route, "cuMemsetD2D32_v2", &return_value, dstDevice, dstPitch, ui,
          Width, Height)) {
    return return_value;
  }
  conn_t *conn = lupine_route_remote_conn(route);
  if (conn == nullptr ||
      rpc_write_start_request(conn, RPC_cuMemsetD2D32_v2) < 0 ||
      rpc_write(conn, &dstDevice, sizeof(CUdeviceptr)) < 0 ||
      rpc_write(conn, &dstPitch, sizeof(size_t)) < 0 ||
      rpc_write(conn, &ui, sizeof(unsigned int)) < 0 ||
      rpc_write(conn, &Width, sizeof(size_t)) < 0 ||
      rpc_write(conn, &Height, sizeof(size_t)) < 0 ||
      rpc_wait_for_response(conn) < 0 ||
      rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
      rpc_read_end(conn) < 0)
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  return return_value;
}

CUresult cuMemsetD8Async(CUdeviceptr dstDevice, unsigned char uc, size_t N,
                         CUstream hStream) {
  lupine_route route = lupine_route_for_deviceptr(dstDevice);
  CUresult return_value;
  using real_fn_t = CUresult (*)(CUdeviceptr, unsigned char, size_t, CUstream);
  if (lupine_call_local_cuda_if_routed<real_fn_t>(
          route, "cuMemsetD8Async", &return_value, dstDevice, uc, N, hStream)) {
    return return_value;
  }
  conn_t *conn = lupine_route_remote_conn(route);
  if (conn == nullptr ||
      rpc_write_start_request(conn, RPC_cuMemsetD8Async) < 0 ||
      rpc_write(conn, &dstDevice, sizeof(CUdeviceptr)) < 0 ||
      rpc_write(conn, &uc, sizeof(unsigned char)) < 0 ||
      rpc_write(conn, &N, sizeof(size_t)) < 0 ||
      rpc_write(conn, &hStream, sizeof(CUstream)) < 0 ||
      rpc_wait_for_response(conn) < 0 ||
      rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
      rpc_read_end(conn) < 0)
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  return return_value;
}

CUresult cuMemsetD16Async(CUdeviceptr dstDevice, unsigned short us, size_t N,
                          CUstream hStream) {
  lupine_route route = lupine_route_for_deviceptr(dstDevice);
  CUresult return_value;
  using real_fn_t = CUresult (*)(CUdeviceptr, unsigned short, size_t, CUstream);
  if (lupine_call_local_cuda_if_routed<real_fn_t>(route, "cuMemsetD16Async",
                                                  &return_value, dstDevice, us,
                                                  N, hStream)) {
    return return_value;
  }
  conn_t *conn = lupine_route_remote_conn(route);
  if (conn == nullptr ||
      rpc_write_start_request(conn, RPC_cuMemsetD16Async) < 0 ||
      rpc_write(conn, &dstDevice, sizeof(CUdeviceptr)) < 0 ||
      rpc_write(conn, &us, sizeof(unsigned short)) < 0 ||
      rpc_write(conn, &N, sizeof(size_t)) < 0 ||
      rpc_write(conn, &hStream, sizeof(CUstream)) < 0 ||
      rpc_wait_for_response(conn) < 0 ||
      rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
      rpc_read_end(conn) < 0)
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  return return_value;
}

CUresult cuMemsetD32Async(CUdeviceptr dstDevice, unsigned int ui, size_t N,
                          CUstream hStream) {
  lupine_route route = lupine_route_for_deviceptr(dstDevice);
  CUresult return_value;
  using real_fn_t = CUresult (*)(CUdeviceptr, unsigned int, size_t, CUstream);
  if (lupine_call_local_cuda_if_routed<real_fn_t>(route, "cuMemsetD32Async",
                                                  &return_value, dstDevice, ui,
                                                  N, hStream)) {
    return return_value;
  }
  conn_t *conn = lupine_route_remote_conn(route);
  if (conn == nullptr ||
      rpc_write_start_request(conn, RPC_cuMemsetD32Async) < 0 ||
      rpc_write(conn, &dstDevice, sizeof(CUdeviceptr)) < 0 ||
      rpc_write(conn, &ui, sizeof(unsigned int)) < 0 ||
      rpc_write(conn, &N, sizeof(size_t)) < 0 ||
      rpc_write(conn, &hStream, sizeof(CUstream)) < 0 ||
      rpc_wait_for_response(conn) < 0 ||
      rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
      rpc_read_end(conn) < 0)
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  return return_value;
}

CUresult cuMemsetD2D8Async(CUdeviceptr dstDevice, size_t dstPitch,
                           unsigned char uc, size_t Width, size_t Height,
                           CUstream hStream) {
  lupine_route route = lupine_route_for_deviceptr(dstDevice);
  CUresult return_value;
  using real_fn_t = CUresult (*)(CUdeviceptr, size_t, unsigned char, size_t,
                                 size_t, CUstream);
  if (lupine_call_local_cuda_if_routed<real_fn_t>(
          route, "cuMemsetD2D8Async", &return_value, dstDevice, dstPitch, uc,
          Width, Height, hStream)) {
    return return_value;
  }
  conn_t *conn = lupine_route_remote_conn(route);
  if (conn == nullptr ||
      rpc_write_start_request(conn, RPC_cuMemsetD2D8Async) < 0 ||
      rpc_write(conn, &dstDevice, sizeof(CUdeviceptr)) < 0 ||
      rpc_write(conn, &dstPitch, sizeof(size_t)) < 0 ||
      rpc_write(conn, &uc, sizeof(unsigned char)) < 0 ||
      rpc_write(conn, &Width, sizeof(size_t)) < 0 ||
      rpc_write(conn, &Height, sizeof(size_t)) < 0 ||
      rpc_write(conn, &hStream, sizeof(CUstream)) < 0 ||
      rpc_wait_for_response(conn) < 0 ||
      rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
      rpc_read_end(conn) < 0)
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  return return_value;
}

CUresult cuMemsetD2D16Async(CUdeviceptr dstDevice, size_t dstPitch,
                            unsigned short us, size_t Width, size_t Height,
                            CUstream hStream) {
  lupine_route route = lupine_route_for_deviceptr(dstDevice);
  CUresult return_value;
  using real_fn_t = CUresult (*)(CUdeviceptr, size_t, unsigned short, size_t,
                                 size_t, CUstream);
  if (lupine_call_local_cuda_if_routed<real_fn_t>(
          route, "cuMemsetD2D16Async", &return_value, dstDevice, dstPitch, us,
          Width, Height, hStream)) {
    return return_value;
  }
  conn_t *conn = lupine_route_remote_conn(route);
  if (conn == nullptr ||
      rpc_write_start_request(conn, RPC_cuMemsetD2D16Async) < 0 ||
      rpc_write(conn, &dstDevice, sizeof(CUdeviceptr)) < 0 ||
      rpc_write(conn, &dstPitch, sizeof(size_t)) < 0 ||
      rpc_write(conn, &us, sizeof(unsigned short)) < 0 ||
      rpc_write(conn, &Width, sizeof(size_t)) < 0 ||
      rpc_write(conn, &Height, sizeof(size_t)) < 0 ||
      rpc_write(conn, &hStream, sizeof(CUstream)) < 0 ||
      rpc_wait_for_response(conn) < 0 ||
      rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
      rpc_read_end(conn) < 0)
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  return return_value;
}

CUresult cuMemsetD2D32Async(CUdeviceptr dstDevice, size_t dstPitch,
                            unsigned int ui, size_t Width, size_t Height,
                            CUstream hStream) {
  lupine_route route = lupine_route_for_deviceptr(dstDevice);
  CUresult return_value;
  using real_fn_t =
      CUresult (*)(CUdeviceptr, size_t, unsigned int, size_t, size_t, CUstream);
  if (lupine_call_local_cuda_if_routed<real_fn_t>(
          route, "cuMemsetD2D32Async", &return_value, dstDevice, dstPitch, ui,
          Width, Height, hStream)) {
    return return_value;
  }
  conn_t *conn = lupine_route_remote_conn(route);
  if (conn == nullptr ||
      rpc_write_start_request(conn, RPC_cuMemsetD2D32Async) < 0 ||
      rpc_write(conn, &dstDevice, sizeof(CUdeviceptr)) < 0 ||
      rpc_write(conn, &dstPitch, sizeof(size_t)) < 0 ||
      rpc_write(conn, &ui, sizeof(unsigned int)) < 0 ||
      rpc_write(conn, &Width, sizeof(size_t)) < 0 ||
      rpc_write(conn, &Height, sizeof(size_t)) < 0 ||
      rpc_write(conn, &hStream, sizeof(CUstream)) < 0 ||
      rpc_wait_for_response(conn) < 0 ||
      rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
      rpc_read_end(conn) < 0)
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  return return_value;
}

CUresult cuArrayCreate_v2(CUarray *pHandle,
                          const CUDA_ARRAY_DESCRIPTOR *pAllocateArray) {
  lupine_route route = lupine_route_for_default();
  CUresult return_value;
  using real_fn_t = CUresult (*)(CUarray *, const CUDA_ARRAY_DESCRIPTOR *);
  if (lupine_call_local_cuda_if_routed<real_fn_t>(
          route, "cuArrayCreate_v2", &return_value, pHandle, pAllocateArray)) {
    return return_value;
  }
  conn_t *conn = lupine_route_remote_conn(route);
  if (conn == nullptr ||
      rpc_write_start_request(conn, RPC_cuArrayCreate_v2) < 0 ||
      rpc_write(conn, pHandle, sizeof(CUarray)) < 0 ||
      rpc_write(conn, pAllocateArray, sizeof(const CUDA_ARRAY_DESCRIPTOR)) <
          0 ||
      rpc_wait_for_response(conn) < 0 ||
      rpc_read(conn, pHandle, sizeof(CUarray)) < 0 ||
      rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
      rpc_read_end(conn) < 0)
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  return return_value;
}

CUresult cuArrayGetDescriptor_v2(CUDA_ARRAY_DESCRIPTOR *pArrayDescriptor,
                                 CUarray hArray) {
  lupine_route route = lupine_route_for_default();
  CUresult return_value;
  using real_fn_t = CUresult (*)(CUDA_ARRAY_DESCRIPTOR *, CUarray);
  if (lupine_call_local_cuda_if_routed<real_fn_t>(
          route, "cuArrayGetDescriptor_v2", &return_value, pArrayDescriptor,
          hArray)) {
    return return_value;
  }
  conn_t *conn = lupine_route_remote_conn(route);
  if (conn == nullptr ||
      rpc_write_start_request(conn, RPC_cuArrayGetDescriptor_v2) < 0 ||
      rpc_write(conn, pArrayDescriptor, sizeof(CUDA_ARRAY_DESCRIPTOR)) < 0 ||
      rpc_write(conn, &hArray, sizeof(CUarray)) < 0 ||
      rpc_wait_for_response(conn) < 0 ||
      rpc_read(conn, pArrayDescriptor, sizeof(CUDA_ARRAY_DESCRIPTOR)) < 0 ||
      rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
      rpc_read_end(conn) < 0)
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  return return_value;
}

CUresult
cuArrayGetSparseProperties(CUDA_ARRAY_SPARSE_PROPERTIES *sparseProperties,
                           CUarray array) {
  lupine_route route = lupine_route_for_default();
  CUresult return_value;
  using real_fn_t = CUresult (*)(CUDA_ARRAY_SPARSE_PROPERTIES *, CUarray);
  if (lupine_call_local_cuda_if_routed<real_fn_t>(
          route, "cuArrayGetSparseProperties", &return_value, sparseProperties,
          array)) {
    return return_value;
  }
  conn_t *conn = lupine_route_remote_conn(route);
  if (conn == nullptr ||
      rpc_write_start_request(conn, RPC_cuArrayGetSparseProperties) < 0 ||
      rpc_write(conn, sparseProperties, sizeof(CUDA_ARRAY_SPARSE_PROPERTIES)) <
          0 ||
      rpc_write(conn, &array, sizeof(CUarray)) < 0 ||
      rpc_wait_for_response(conn) < 0 ||
      rpc_read(conn, sparseProperties, sizeof(CUDA_ARRAY_SPARSE_PROPERTIES)) <
          0 ||
      rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
      rpc_read_end(conn) < 0)
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  return return_value;
}

CUresult cuMipmappedArrayGetSparseProperties(
    CUDA_ARRAY_SPARSE_PROPERTIES *sparseProperties, CUmipmappedArray mipmap) {
  lupine_route route = lupine_route_for_default();
  CUresult return_value;
  using real_fn_t =
      CUresult (*)(CUDA_ARRAY_SPARSE_PROPERTIES *, CUmipmappedArray);
  if (lupine_call_local_cuda_if_routed<real_fn_t>(
          route, "cuMipmappedArrayGetSparseProperties", &return_value,
          sparseProperties, mipmap)) {
    return return_value;
  }
  conn_t *conn = lupine_route_remote_conn(route);
  if (conn == nullptr ||
      rpc_write_start_request(conn, RPC_cuMipmappedArrayGetSparseProperties) <
          0 ||
      rpc_write(conn, sparseProperties, sizeof(CUDA_ARRAY_SPARSE_PROPERTIES)) <
          0 ||
      rpc_write(conn, &mipmap, sizeof(CUmipmappedArray)) < 0 ||
      rpc_wait_for_response(conn) < 0 ||
      rpc_read(conn, sparseProperties, sizeof(CUDA_ARRAY_SPARSE_PROPERTIES)) <
          0 ||
      rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
      rpc_read_end(conn) < 0)
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  return return_value;
}

CUresult
cuArrayGetMemoryRequirements(CUDA_ARRAY_MEMORY_REQUIREMENTS *memoryRequirements,
                             CUarray array, CUdevice device) {
  lupine_route route = lupine_route_for_device(&device);
  CUresult return_value;
  using real_fn_t =
      CUresult (*)(CUDA_ARRAY_MEMORY_REQUIREMENTS *, CUarray, CUdevice);
  if (lupine_call_local_cuda_if_routed<real_fn_t>(
          route, "cuArrayGetMemoryRequirements", &return_value,
          memoryRequirements, array, device)) {
    return return_value;
  }
  conn_t *conn = lupine_route_remote_conn(route);
  if (conn == nullptr ||
      rpc_write_start_request(conn, RPC_cuArrayGetMemoryRequirements) < 0 ||
      rpc_write(conn, memoryRequirements,
                sizeof(CUDA_ARRAY_MEMORY_REQUIREMENTS)) < 0 ||
      rpc_write(conn, &array, sizeof(CUarray)) < 0 ||
      rpc_write(conn, &device, sizeof(CUdevice)) < 0 ||
      rpc_wait_for_response(conn) < 0 ||
      rpc_read(conn, memoryRequirements,
               sizeof(CUDA_ARRAY_MEMORY_REQUIREMENTS)) < 0 ||
      rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
      rpc_read_end(conn) < 0)
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  return return_value;
}

CUresult cuMipmappedArrayGetMemoryRequirements(
    CUDA_ARRAY_MEMORY_REQUIREMENTS *memoryRequirements, CUmipmappedArray mipmap,
    CUdevice device) {
  lupine_route route = lupine_route_for_device(&device);
  CUresult return_value;
  using real_fn_t = CUresult (*)(CUDA_ARRAY_MEMORY_REQUIREMENTS *,
                                 CUmipmappedArray, CUdevice);
  if (lupine_call_local_cuda_if_routed<real_fn_t>(
          route, "cuMipmappedArrayGetMemoryRequirements", &return_value,
          memoryRequirements, mipmap, device)) {
    return return_value;
  }
  conn_t *conn = lupine_route_remote_conn(route);
  if (conn == nullptr ||
      rpc_write_start_request(conn, RPC_cuMipmappedArrayGetMemoryRequirements) <
          0 ||
      rpc_write(conn, memoryRequirements,
                sizeof(CUDA_ARRAY_MEMORY_REQUIREMENTS)) < 0 ||
      rpc_write(conn, &mipmap, sizeof(CUmipmappedArray)) < 0 ||
      rpc_write(conn, &device, sizeof(CUdevice)) < 0 ||
      rpc_wait_for_response(conn) < 0 ||
      rpc_read(conn, memoryRequirements,
               sizeof(CUDA_ARRAY_MEMORY_REQUIREMENTS)) < 0 ||
      rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
      rpc_read_end(conn) < 0)
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  return return_value;
}

CUresult cuArrayGetPlane(CUarray *pPlaneArray, CUarray hArray,
                         unsigned int planeIdx) {
  lupine_route route = lupine_route_for_default();
  CUresult return_value;
  using real_fn_t = CUresult (*)(CUarray *, CUarray, unsigned int);
  if (lupine_call_local_cuda_if_routed<real_fn_t>(route, "cuArrayGetPlane",
                                                  &return_value, pPlaneArray,
                                                  hArray, planeIdx)) {
    return return_value;
  }
  conn_t *conn = lupine_route_remote_conn(route);
  if (conn == nullptr ||
      rpc_write_start_request(conn, RPC_cuArrayGetPlane) < 0 ||
      rpc_write(conn, pPlaneArray, sizeof(CUarray)) < 0 ||
      rpc_write(conn, &hArray, sizeof(CUarray)) < 0 ||
      rpc_write(conn, &planeIdx, sizeof(unsigned int)) < 0 ||
      rpc_wait_for_response(conn) < 0 ||
      rpc_read(conn, pPlaneArray, sizeof(CUarray)) < 0 ||
      rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
      rpc_read_end(conn) < 0)
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  return return_value;
}

CUresult cuArrayDestroy(CUarray hArray) {
  lupine_route route = lupine_route_for_default();
  CUresult return_value;
  using real_fn_t = CUresult (*)(CUarray);
  if (lupine_call_local_cuda_if_routed<real_fn_t>(route, "cuArrayDestroy",
                                                  &return_value, hArray)) {
    return return_value;
  }
  conn_t *conn = lupine_route_remote_conn(route);
  if (conn == nullptr ||
      rpc_write_start_request(conn, RPC_cuArrayDestroy) < 0 ||
      rpc_write(conn, &hArray, sizeof(CUarray)) < 0 ||
      rpc_wait_for_response(conn) < 0 ||
      rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
      rpc_read_end(conn) < 0)
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  return return_value;
}

CUresult cuArray3DCreate_v2(CUarray *pHandle,
                            const CUDA_ARRAY3D_DESCRIPTOR *pAllocateArray) {
  lupine_route route = lupine_route_for_default();
  CUresult return_value;
  using real_fn_t = CUresult (*)(CUarray *, const CUDA_ARRAY3D_DESCRIPTOR *);
  if (lupine_call_local_cuda_if_routed<real_fn_t>(route, "cuArray3DCreate_v2",
                                                  &return_value, pHandle,
                                                  pAllocateArray)) {
    return return_value;
  }
  conn_t *conn = lupine_route_remote_conn(route);
  if (conn == nullptr ||
      rpc_write_start_request(conn, RPC_cuArray3DCreate_v2) < 0 ||
      rpc_write(conn, pHandle, sizeof(CUarray)) < 0 ||
      rpc_write(conn, pAllocateArray, sizeof(const CUDA_ARRAY3D_DESCRIPTOR)) <
          0 ||
      rpc_wait_for_response(conn) < 0 ||
      rpc_read(conn, pHandle, sizeof(CUarray)) < 0 ||
      rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
      rpc_read_end(conn) < 0)
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  return return_value;
}

CUresult cuArray3DGetDescriptor_v2(CUDA_ARRAY3D_DESCRIPTOR *pArrayDescriptor,
                                   CUarray hArray) {
  lupine_route route = lupine_route_for_default();
  CUresult return_value;
  using real_fn_t = CUresult (*)(CUDA_ARRAY3D_DESCRIPTOR *, CUarray);
  if (lupine_call_local_cuda_if_routed<real_fn_t>(
          route, "cuArray3DGetDescriptor_v2", &return_value, pArrayDescriptor,
          hArray)) {
    return return_value;
  }
  conn_t *conn = lupine_route_remote_conn(route);
  if (conn == nullptr ||
      rpc_write_start_request(conn, RPC_cuArray3DGetDescriptor_v2) < 0 ||
      rpc_write(conn, pArrayDescriptor, sizeof(CUDA_ARRAY3D_DESCRIPTOR)) < 0 ||
      rpc_write(conn, &hArray, sizeof(CUarray)) < 0 ||
      rpc_wait_for_response(conn) < 0 ||
      rpc_read(conn, pArrayDescriptor, sizeof(CUDA_ARRAY3D_DESCRIPTOR)) < 0 ||
      rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
      rpc_read_end(conn) < 0)
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  return return_value;
}

CUresult
cuMipmappedArrayCreate(CUmipmappedArray *pHandle,
                       const CUDA_ARRAY3D_DESCRIPTOR *pMipmappedArrayDesc,
                       unsigned int numMipmapLevels) {
  lupine_route route = lupine_route_for_default();
  CUresult return_value;
  using real_fn_t = CUresult (*)(CUmipmappedArray *,
                                 const CUDA_ARRAY3D_DESCRIPTOR *, unsigned int);
  if (lupine_call_local_cuda_if_routed<real_fn_t>(
          route, "cuMipmappedArrayCreate", &return_value, pHandle,
          pMipmappedArrayDesc, numMipmapLevels)) {
    return return_value;
  }
  conn_t *conn = lupine_route_remote_conn(route);
  if (conn == nullptr ||
      rpc_write_start_request(conn, RPC_cuMipmappedArrayCreate) < 0 ||
      rpc_write(conn, pHandle, sizeof(CUmipmappedArray)) < 0 ||
      rpc_write(conn, pMipmappedArrayDesc,
                sizeof(const CUDA_ARRAY3D_DESCRIPTOR)) < 0 ||
      rpc_write(conn, &numMipmapLevels, sizeof(unsigned int)) < 0 ||
      rpc_wait_for_response(conn) < 0 ||
      rpc_read(conn, pHandle, sizeof(CUmipmappedArray)) < 0 ||
      rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
      rpc_read_end(conn) < 0)
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  return return_value;
}

CUresult cuMipmappedArrayGetLevel(CUarray *pLevelArray,
                                  CUmipmappedArray hMipmappedArray,
                                  unsigned int level) {
  lupine_route route = lupine_route_for_default();
  CUresult return_value;
  using real_fn_t = CUresult (*)(CUarray *, CUmipmappedArray, unsigned int);
  if (lupine_call_local_cuda_if_routed<real_fn_t>(
          route, "cuMipmappedArrayGetLevel", &return_value, pLevelArray,
          hMipmappedArray, level)) {
    return return_value;
  }
  conn_t *conn = lupine_route_remote_conn(route);
  if (conn == nullptr ||
      rpc_write_start_request(conn, RPC_cuMipmappedArrayGetLevel) < 0 ||
      rpc_write(conn, pLevelArray, sizeof(CUarray)) < 0 ||
      rpc_write(conn, &hMipmappedArray, sizeof(CUmipmappedArray)) < 0 ||
      rpc_write(conn, &level, sizeof(unsigned int)) < 0 ||
      rpc_wait_for_response(conn) < 0 ||
      rpc_read(conn, pLevelArray, sizeof(CUarray)) < 0 ||
      rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
      rpc_read_end(conn) < 0)
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  return return_value;
}

CUresult cuMipmappedArrayDestroy(CUmipmappedArray hMipmappedArray) {
  lupine_route route = lupine_route_for_default();
  CUresult return_value;
  using real_fn_t = CUresult (*)(CUmipmappedArray);
  if (lupine_call_local_cuda_if_routed<real_fn_t>(
          route, "cuMipmappedArrayDestroy", &return_value, hMipmappedArray)) {
    return return_value;
  }
  conn_t *conn = lupine_route_remote_conn(route);
  if (conn == nullptr ||
      rpc_write_start_request(conn, RPC_cuMipmappedArrayDestroy) < 0 ||
      rpc_write(conn, &hMipmappedArray, sizeof(CUmipmappedArray)) < 0 ||
      rpc_wait_for_response(conn) < 0 ||
      rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
      rpc_read_end(conn) < 0)
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  return return_value;
}

CUresult cuMemAddressReserve(CUdeviceptr *ptr, size_t size, size_t alignment,
                             CUdeviceptr addr, unsigned long long flags) {
  lupine_route route = lupine_route_for_deviceptr(addr);
  CUresult return_value;
  using real_fn_t = CUresult (*)(CUdeviceptr *, size_t, size_t, CUdeviceptr,
                                 unsigned long long);
  if (lupine_call_local_cuda_if_routed<real_fn_t>(route, "cuMemAddressReserve",
                                                  &return_value, ptr, size,
                                                  alignment, addr, flags)) {
    return return_value;
  }
  conn_t *conn = lupine_route_remote_conn(route);
  if (conn == nullptr ||
      rpc_write_start_request(conn, RPC_cuMemAddressReserve) < 0 ||
      rpc_write(conn, ptr, sizeof(CUdeviceptr)) < 0 ||
      rpc_write(conn, &size, sizeof(size_t)) < 0 ||
      rpc_write(conn, &alignment, sizeof(size_t)) < 0 ||
      rpc_write(conn, &addr, sizeof(CUdeviceptr)) < 0 ||
      rpc_write(conn, &flags, sizeof(unsigned long long)) < 0 ||
      rpc_wait_for_response(conn) < 0 ||
      rpc_read(conn, ptr, sizeof(CUdeviceptr)) < 0 ||
      rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
      rpc_read_end(conn) < 0)
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  return return_value;
}

CUresult cuMemAddressFree(CUdeviceptr ptr, size_t size) {
  lupine_route route = lupine_route_for_deviceptr(ptr);
  CUresult return_value;
  using real_fn_t = CUresult (*)(CUdeviceptr, size_t);
  if (lupine_call_local_cuda_if_routed<real_fn_t>(route, "cuMemAddressFree",
                                                  &return_value, ptr, size)) {
    return return_value;
  }
  conn_t *conn = lupine_route_remote_conn(route);
  if (conn == nullptr ||
      rpc_write_start_request(conn, RPC_cuMemAddressFree) < 0 ||
      rpc_write(conn, &ptr, sizeof(CUdeviceptr)) < 0 ||
      rpc_write(conn, &size, sizeof(size_t)) < 0 ||
      rpc_wait_for_response(conn) < 0 ||
      rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
      rpc_read_end(conn) < 0)
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  return return_value;
}

CUresult cuMemCreate(CUmemGenericAllocationHandle *handle, size_t size,
                     const CUmemAllocationProp *prop,
                     unsigned long long flags) {
  lupine_route route = lupine_route_for_default();
  CUresult return_value;
  using real_fn_t =
      CUresult (*)(CUmemGenericAllocationHandle *, size_t,
                   const CUmemAllocationProp *, unsigned long long);
  if (lupine_call_local_cuda_if_routed<real_fn_t>(
          route, "cuMemCreate", &return_value, handle, size, prop, flags)) {
    return return_value;
  }
  conn_t *conn = lupine_route_remote_conn(route);
  if (conn == nullptr || rpc_write_start_request(conn, RPC_cuMemCreate) < 0 ||
      rpc_write(conn, &size, sizeof(size_t)) < 0 ||
      rpc_write(conn, prop, sizeof(const CUmemAllocationProp)) < 0 ||
      rpc_write(conn, &flags, sizeof(unsigned long long)) < 0 ||
      rpc_wait_for_response(conn) < 0 ||
      rpc_read(conn, handle, sizeof(CUmemGenericAllocationHandle)) < 0 ||
      rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
      rpc_read_end(conn) < 0)
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  return return_value;
}

CUresult cuMemRelease(CUmemGenericAllocationHandle handle) {
  lupine_route route = lupine_route_for_default();
  CUresult return_value;
  using real_fn_t = CUresult (*)(CUmemGenericAllocationHandle);
  if (lupine_call_local_cuda_if_routed<real_fn_t>(route, "cuMemRelease",
                                                  &return_value, handle)) {
    return return_value;
  }
  conn_t *conn = lupine_route_remote_conn(route);
  if (conn == nullptr || rpc_write_start_request(conn, RPC_cuMemRelease) < 0 ||
      rpc_write(conn, &handle, sizeof(CUmemGenericAllocationHandle)) < 0 ||
      rpc_wait_for_response(conn) < 0 ||
      rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
      rpc_read_end(conn) < 0)
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  return return_value;
}

CUresult cuMemMap(CUdeviceptr ptr, size_t size, size_t offset,
                  CUmemGenericAllocationHandle handle,
                  unsigned long long flags) {
  lupine_route route = lupine_route_for_deviceptr(ptr);
  CUresult return_value;
  using real_fn_t =
      CUresult (*)(CUdeviceptr, size_t, size_t, CUmemGenericAllocationHandle,
                   unsigned long long);
  if (lupine_call_local_cuda_if_routed<real_fn_t>(
          route, "cuMemMap", &return_value, ptr, size, offset, handle, flags)) {
    return return_value;
  }
  conn_t *conn = lupine_route_remote_conn(route);
  if (conn == nullptr || rpc_write_start_request(conn, RPC_cuMemMap) < 0 ||
      rpc_write(conn, &ptr, sizeof(CUdeviceptr)) < 0 ||
      rpc_write(conn, &size, sizeof(size_t)) < 0 ||
      rpc_write(conn, &offset, sizeof(size_t)) < 0 ||
      rpc_write(conn, &handle, sizeof(CUmemGenericAllocationHandle)) < 0 ||
      rpc_write(conn, &flags, sizeof(unsigned long long)) < 0 ||
      rpc_wait_for_response(conn) < 0 ||
      rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
      rpc_read_end(conn) < 0)
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  return return_value;
}

CUresult cuMemMapArrayAsync(CUarrayMapInfo *mapInfoList, unsigned int count,
                            CUstream hStream) {
  lupine_route route = (hStream != nullptr ? lupine_route_for_stream(hStream)
                                           : lupine_route_for_default());
  CUresult return_value;
  using real_fn_t = CUresult (*)(CUarrayMapInfo *, unsigned int, CUstream);
  if (lupine_call_local_cuda_if_routed<real_fn_t>(route, "cuMemMapArrayAsync",
                                                  &return_value, mapInfoList,
                                                  count, hStream)) {
    return return_value;
  }
  conn_t *conn = lupine_route_remote_conn(route);
  if (conn == nullptr ||
      rpc_write_start_request(conn, RPC_cuMemMapArrayAsync) < 0 ||
      rpc_write(conn, mapInfoList, sizeof(CUarrayMapInfo)) < 0 ||
      rpc_write(conn, &count, sizeof(unsigned int)) < 0 ||
      rpc_write(conn, &hStream, sizeof(CUstream)) < 0 ||
      rpc_wait_for_response(conn) < 0 ||
      rpc_read(conn, mapInfoList, sizeof(CUarrayMapInfo)) < 0 ||
      rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
      rpc_read_end(conn) < 0)
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  return return_value;
}

CUresult cuMemUnmap(CUdeviceptr ptr, size_t size) {
  lupine_route route = lupine_route_for_deviceptr(ptr);
  CUresult return_value;
  using real_fn_t = CUresult (*)(CUdeviceptr, size_t);
  if (lupine_call_local_cuda_if_routed<real_fn_t>(route, "cuMemUnmap",
                                                  &return_value, ptr, size)) {
    return return_value;
  }
  conn_t *conn = lupine_route_remote_conn(route);
  if (conn == nullptr || rpc_write_start_request(conn, RPC_cuMemUnmap) < 0 ||
      rpc_write(conn, &ptr, sizeof(CUdeviceptr)) < 0 ||
      rpc_write(conn, &size, sizeof(size_t)) < 0 ||
      rpc_wait_for_response(conn) < 0 ||
      rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
      rpc_read_end(conn) < 0)
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  return return_value;
}

CUresult cuMemSetAccess(CUdeviceptr ptr, size_t size,
                        const CUmemAccessDesc *desc, size_t count) {
  lupine_route route = lupine_route_for_deviceptr(ptr);
  CUresult return_value;
  using real_fn_t =
      CUresult (*)(CUdeviceptr, size_t, const CUmemAccessDesc *, size_t);
  if (lupine_call_local_cuda_if_routed<real_fn_t>(
          route, "cuMemSetAccess", &return_value, ptr, size, desc, count)) {
    return return_value;
  }
  conn_t *conn = lupine_route_remote_conn(route);
  if (count * sizeof(const CUmemAccessDesc) != 0 && desc == nullptr)
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  if (conn == nullptr ||
      rpc_write_start_request(conn, RPC_cuMemSetAccess) < 0 ||
      rpc_write(conn, &ptr, sizeof(CUdeviceptr)) < 0 ||
      rpc_write(conn, &size, sizeof(size_t)) < 0 ||
      rpc_write(conn, &count, sizeof(size_t)) < 0 ||
      (count * sizeof(const CUmemAccessDesc) != 0 &&
       rpc_write(conn, desc, count * sizeof(const CUmemAccessDesc)) < 0) ||
      rpc_wait_for_response(conn) < 0 ||
      rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
      rpc_read_end(conn) < 0)
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  return return_value;
}

CUresult cuMemGetAccess(unsigned long long *flags,
                        const CUmemLocation *location, CUdeviceptr ptr) {
  lupine_route route = lupine_route_for_deviceptr(ptr);
  CUresult return_value;
  using real_fn_t =
      CUresult (*)(unsigned long long *, const CUmemLocation *, CUdeviceptr);
  if (lupine_call_local_cuda_if_routed<real_fn_t>(
          route, "cuMemGetAccess", &return_value, flags, location, ptr)) {
    return return_value;
  }
  conn_t *conn = lupine_route_remote_conn(route);
  if (conn == nullptr ||
      rpc_write_start_request(conn, RPC_cuMemGetAccess) < 0 ||
      rpc_write(conn, location, sizeof(const CUmemLocation)) < 0 ||
      rpc_write(conn, &ptr, sizeof(CUdeviceptr)) < 0 ||
      rpc_wait_for_response(conn) < 0 ||
      rpc_read(conn, flags, sizeof(unsigned long long)) < 0 ||
      rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
      rpc_read_end(conn) < 0)
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  return return_value;
}

CUresult
cuMemGetAllocationGranularity(size_t *granularity,
                              const CUmemAllocationProp *prop,
                              CUmemAllocationGranularity_flags option) {
  lupine_route route = lupine_route_for_default();
  CUresult return_value;
  using real_fn_t = CUresult (*)(size_t *, const CUmemAllocationProp *,
                                 CUmemAllocationGranularity_flags);
  if (lupine_call_local_cuda_if_routed<real_fn_t>(
          route, "cuMemGetAllocationGranularity", &return_value, granularity,
          prop, option)) {
    return return_value;
  }
  conn_t *conn = lupine_route_remote_conn(route);
  if (conn == nullptr ||
      rpc_write_start_request(conn, RPC_cuMemGetAllocationGranularity) < 0 ||
      rpc_write(conn, prop, sizeof(const CUmemAllocationProp)) < 0 ||
      rpc_write(conn, &option, sizeof(CUmemAllocationGranularity_flags)) < 0 ||
      rpc_wait_for_response(conn) < 0 ||
      rpc_read(conn, granularity, sizeof(size_t)) < 0 ||
      rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
      rpc_read_end(conn) < 0)
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  return return_value;
}

CUresult
cuMemGetAllocationPropertiesFromHandle(CUmemAllocationProp *prop,
                                       CUmemGenericAllocationHandle handle) {
  lupine_route route = lupine_route_for_default();
  CUresult return_value;
  using real_fn_t =
      CUresult (*)(CUmemAllocationProp *, CUmemGenericAllocationHandle);
  if (lupine_call_local_cuda_if_routed<real_fn_t>(
          route, "cuMemGetAllocationPropertiesFromHandle", &return_value, prop,
          handle)) {
    return return_value;
  }
  conn_t *conn = lupine_route_remote_conn(route);
  if (conn == nullptr ||
      rpc_write_start_request(conn,
                              RPC_cuMemGetAllocationPropertiesFromHandle) < 0 ||
      rpc_write(conn, prop, sizeof(CUmemAllocationProp)) < 0 ||
      rpc_write(conn, &handle, sizeof(CUmemGenericAllocationHandle)) < 0 ||
      rpc_wait_for_response(conn) < 0 ||
      rpc_read(conn, prop, sizeof(CUmemAllocationProp)) < 0 ||
      rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
      rpc_read_end(conn) < 0)
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  return return_value;
}

CUresult cuMemFreeAsync(CUdeviceptr dptr, CUstream hStream) {
  lupine_route route = lupine_route_for_deviceptr(dptr);
  CUresult return_value;
  using real_fn_t = CUresult (*)(CUdeviceptr, CUstream);
  if (lupine_call_local_cuda_if_routed<real_fn_t>(
          route, "cuMemFreeAsync", &return_value, dptr, hStream)) {
    if (return_value == CUDA_SUCCESS)
      lupine_forget_deviceptr_owner(dptr);
    return return_value;
  }
  conn_t *conn = lupine_route_remote_conn(route);
  if (conn == nullptr ||
      rpc_write_start_request(conn, RPC_cuMemFreeAsync) < 0 ||
      rpc_write(conn, &dptr, sizeof(CUdeviceptr)) < 0 ||
      rpc_write(conn, &hStream, sizeof(CUstream)) < 0 ||
      rpc_wait_for_response(conn) < 0 ||
      rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
      rpc_read_end(conn) < 0)
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  if (return_value == CUDA_SUCCESS)
    lupine_forget_deviceptr_owner(dptr);
  return return_value;
}

CUresult cuMemAllocAsync(CUdeviceptr *dptr, size_t bytesize, CUstream hStream) {
  lupine_route route = (hStream != nullptr ? lupine_route_for_stream(hStream)
                                           : lupine_route_for_default());
  CUresult return_value;
  using real_fn_t = CUresult (*)(CUdeviceptr *, size_t, CUstream);
  if (lupine_call_local_cuda_if_routed<real_fn_t>(
          route, "cuMemAllocAsync", &return_value, dptr, bytesize, hStream)) {
    if (return_value == CUDA_SUCCESS && dptr != nullptr) {
      lupine_note_deviceptr_owner_route(*dptr, route);
    }
    if (return_value == CUDA_SUCCESS && dptr != nullptr)
      lupine_note_deviceptr_allocation_route(*dptr, bytesize, route);
    return return_value;
  }
  conn_t *conn = lupine_route_remote_conn(route);
  if (conn == nullptr ||
      rpc_write_start_request(conn, RPC_cuMemAllocAsync) < 0 ||
      rpc_write(conn, dptr, sizeof(CUdeviceptr)) < 0 ||
      rpc_write(conn, &bytesize, sizeof(size_t)) < 0 ||
      rpc_write(conn, &hStream, sizeof(CUstream)) < 0 ||
      rpc_wait_for_response(conn) < 0 ||
      rpc_read(conn, dptr, sizeof(CUdeviceptr)) < 0 ||
      rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
      rpc_read_end(conn) < 0)
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  if (return_value == CUDA_SUCCESS && dptr != nullptr) {
    lupine_note_deviceptr_owner_route(*dptr, route);
  }
  if (return_value == CUDA_SUCCESS && dptr != nullptr)
    lupine_note_deviceptr_allocation_route(*dptr, bytesize, route);
  return return_value;
}

CUresult cuMemPoolTrimTo(CUmemoryPool pool, size_t minBytesToKeep) {
  lupine_route route = lupine_route_for_memory_pool(pool);
  CUresult return_value;
  using real_fn_t = CUresult (*)(CUmemoryPool, size_t);
  if (lupine_call_local_cuda_if_routed<real_fn_t>(
          route, "cuMemPoolTrimTo", &return_value, pool, minBytesToKeep)) {
    return return_value;
  }
  conn_t *conn = lupine_route_remote_conn(route);
  if (conn == nullptr ||
      rpc_write_start_request(conn, RPC_cuMemPoolTrimTo) < 0 ||
      rpc_write(conn, &pool, sizeof(CUmemoryPool)) < 0 ||
      rpc_write(conn, &minBytesToKeep, sizeof(size_t)) < 0 ||
      rpc_wait_for_response(conn) < 0 ||
      rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
      rpc_read_end(conn) < 0)
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  return return_value;
}

CUresult cuMemPoolSetAccess(CUmemoryPool pool, const CUmemAccessDesc *map,
                            size_t count) {
  lupine_route route = lupine_route_for_memory_pool(pool);
  CUresult return_value;
  using real_fn_t = CUresult (*)(CUmemoryPool, const CUmemAccessDesc *, size_t);
  if (lupine_call_local_cuda_if_routed<real_fn_t>(
          route, "cuMemPoolSetAccess", &return_value, pool, map, count)) {
    return return_value;
  }
  conn_t *conn = lupine_route_remote_conn(route);
  if (conn == nullptr ||
      rpc_write_start_request(conn, RPC_cuMemPoolSetAccess) < 0 ||
      rpc_write(conn, &pool, sizeof(CUmemoryPool)) < 0 ||
      rpc_write(conn, &map, sizeof(const CUmemAccessDesc *)) < 0 ||
      rpc_write(conn, &count, sizeof(size_t)) < 0 ||
      rpc_wait_for_response(conn) < 0 ||
      rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
      rpc_read_end(conn) < 0)
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  return return_value;
}

CUresult cuMemPoolGetAccess(CUmemAccess_flags *flags, CUmemoryPool memPool,
                            CUmemLocation *location) {
  lupine_route route = lupine_route_for_memory_pool(memPool);
  CUresult return_value;
  using real_fn_t =
      CUresult (*)(CUmemAccess_flags *, CUmemoryPool, CUmemLocation *);
  if (lupine_call_local_cuda_if_routed<real_fn_t>(route, "cuMemPoolGetAccess",
                                                  &return_value, flags, memPool,
                                                  location)) {
    return return_value;
  }
  conn_t *conn = lupine_route_remote_conn(route);
  if (conn == nullptr ||
      rpc_write_start_request(conn, RPC_cuMemPoolGetAccess) < 0 ||
      rpc_write(conn, flags, sizeof(CUmemAccess_flags)) < 0 ||
      rpc_write(conn, &memPool, sizeof(CUmemoryPool)) < 0 ||
      rpc_write(conn, location, sizeof(CUmemLocation)) < 0 ||
      rpc_wait_for_response(conn) < 0 ||
      rpc_read(conn, flags, sizeof(CUmemAccess_flags)) < 0 ||
      rpc_read(conn, location, sizeof(CUmemLocation)) < 0 ||
      rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
      rpc_read_end(conn) < 0)
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  return return_value;
}

CUresult cuMemPoolCreate(CUmemoryPool *pool, const CUmemPoolProps *poolProps) {
  lupine_route route = lupine_route_for_default();
  CUresult return_value;
  using real_fn_t = CUresult (*)(CUmemoryPool *, const CUmemPoolProps *);
  if (lupine_call_local_cuda_if_routed<real_fn_t>(
          route, "cuMemPoolCreate", &return_value, pool, poolProps)) {
    if (return_value == CUDA_SUCCESS && pool != nullptr) {
      lupine_note_memory_pool_owner_route(*pool, route);
    }
    return return_value;
  }
  conn_t *conn = lupine_route_remote_conn(route);
  if (conn == nullptr ||
      rpc_write_start_request(conn, RPC_cuMemPoolCreate) < 0 ||
      rpc_write(conn, pool, sizeof(CUmemoryPool)) < 0 ||
      rpc_write(conn, &poolProps, sizeof(const CUmemPoolProps *)) < 0 ||
      rpc_wait_for_response(conn) < 0 ||
      rpc_read(conn, pool, sizeof(CUmemoryPool)) < 0 ||
      rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
      rpc_read_end(conn) < 0)
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  if (return_value == CUDA_SUCCESS && pool != nullptr) {
    lupine_note_memory_pool_owner_route(*pool, route);
  }
  return return_value;
}

CUresult cuMemPoolDestroy(CUmemoryPool pool) {
  lupine_route route = lupine_route_for_memory_pool(pool);
  CUresult return_value;
  using real_fn_t = CUresult (*)(CUmemoryPool);
  if (lupine_call_local_cuda_if_routed<real_fn_t>(route, "cuMemPoolDestroy",
                                                  &return_value, pool)) {
    return return_value;
  }
  conn_t *conn = lupine_route_remote_conn(route);
  if (conn == nullptr ||
      rpc_write_start_request(conn, RPC_cuMemPoolDestroy) < 0 ||
      rpc_write(conn, &pool, sizeof(CUmemoryPool)) < 0 ||
      rpc_wait_for_response(conn) < 0 ||
      rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
      rpc_read_end(conn) < 0)
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  return return_value;
}

CUresult cuMemAllocFromPoolAsync(CUdeviceptr *dptr, size_t bytesize,
                                 CUmemoryPool pool, CUstream hStream) {
  lupine_route route = (hStream != nullptr ? lupine_route_for_stream(hStream)
                                           : lupine_route_for_default());
  CUresult return_value;
  using real_fn_t = CUresult (*)(CUdeviceptr *, size_t, CUmemoryPool, CUstream);
  if (lupine_call_local_cuda_if_routed<real_fn_t>(
          route, "cuMemAllocFromPoolAsync", &return_value, dptr, bytesize, pool,
          hStream)) {
    if (return_value == CUDA_SUCCESS && dptr != nullptr) {
      lupine_note_deviceptr_owner_route(*dptr, route);
    }
    if (return_value == CUDA_SUCCESS && dptr != nullptr)
      lupine_note_deviceptr_allocation_route(*dptr, bytesize, route);
    return return_value;
  }
  conn_t *conn = lupine_route_remote_conn(route);
  if (conn == nullptr ||
      rpc_write_start_request(conn, RPC_cuMemAllocFromPoolAsync) < 0 ||
      rpc_write(conn, dptr, sizeof(CUdeviceptr)) < 0 ||
      rpc_write(conn, &bytesize, sizeof(size_t)) < 0 ||
      rpc_write(conn, &pool, sizeof(CUmemoryPool)) < 0 ||
      rpc_write(conn, &hStream, sizeof(CUstream)) < 0 ||
      rpc_wait_for_response(conn) < 0 ||
      rpc_read(conn, dptr, sizeof(CUdeviceptr)) < 0 ||
      rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
      rpc_read_end(conn) < 0)
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  if (return_value == CUDA_SUCCESS && dptr != nullptr) {
    lupine_note_deviceptr_owner_route(*dptr, route);
  }
  if (return_value == CUDA_SUCCESS && dptr != nullptr)
    lupine_note_deviceptr_allocation_route(*dptr, bytesize, route);
  return return_value;
}

CUresult cuMemPoolExportPointer(CUmemPoolPtrExportData *shareData_out,
                                CUdeviceptr ptr) {
  lupine_route route = lupine_route_for_deviceptr(ptr);
  CUresult return_value;
  using real_fn_t = CUresult (*)(CUmemPoolPtrExportData *, CUdeviceptr);
  if (lupine_call_local_cuda_if_routed<real_fn_t>(
          route, "cuMemPoolExportPointer", &return_value, shareData_out, ptr)) {
    return return_value;
  }
  conn_t *conn = lupine_route_remote_conn(route);
  if (conn == nullptr ||
      rpc_write_start_request(conn, RPC_cuMemPoolExportPointer) < 0 ||
      rpc_write(conn, shareData_out, sizeof(CUmemPoolPtrExportData)) < 0 ||
      rpc_write(conn, &ptr, sizeof(CUdeviceptr)) < 0 ||
      rpc_wait_for_response(conn) < 0 ||
      rpc_read(conn, shareData_out, sizeof(CUmemPoolPtrExportData)) < 0 ||
      rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
      rpc_read_end(conn) < 0)
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  return return_value;
}

CUresult cuMemPoolImportPointer(CUdeviceptr *ptr_out, CUmemoryPool pool,
                                CUmemPoolPtrExportData *shareData) {
  lupine_route route = lupine_route_for_memory_pool(pool);
  CUresult return_value;
  using real_fn_t =
      CUresult (*)(CUdeviceptr *, CUmemoryPool, CUmemPoolPtrExportData *);
  if (lupine_call_local_cuda_if_routed<real_fn_t>(
          route, "cuMemPoolImportPointer", &return_value, ptr_out, pool,
          shareData)) {
    return return_value;
  }
  conn_t *conn = lupine_route_remote_conn(route);
  if (conn == nullptr ||
      rpc_write_start_request(conn, RPC_cuMemPoolImportPointer) < 0 ||
      rpc_write(conn, ptr_out, sizeof(CUdeviceptr)) < 0 ||
      rpc_write(conn, &pool, sizeof(CUmemoryPool)) < 0 ||
      rpc_write(conn, shareData, sizeof(CUmemPoolPtrExportData)) < 0 ||
      rpc_wait_for_response(conn) < 0 ||
      rpc_read(conn, ptr_out, sizeof(CUdeviceptr)) < 0 ||
      rpc_read(conn, shareData, sizeof(CUmemPoolPtrExportData)) < 0 ||
      rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
      rpc_read_end(conn) < 0)
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  return return_value;
}

CUresult cuMemRangeGetAttributes(void **data, size_t *dataSizes,
                                 CUmem_range_attribute *attributes,
                                 size_t numAttributes, CUdeviceptr devPtr,
                                 size_t count) {
  lupine_route route = lupine_route_for_deviceptr(devPtr);
  CUresult return_value;
  using real_fn_t = CUresult (*)(void **, size_t *, CUmem_range_attribute *,
                                 size_t, CUdeviceptr, size_t);
  if (lupine_call_local_cuda_if_routed<real_fn_t>(
          route, "cuMemRangeGetAttributes", &return_value, data, dataSizes,
          attributes, numAttributes, devPtr, count)) {
    return return_value;
  }
  conn_t *conn = lupine_route_remote_conn(route);
  if (conn == nullptr ||
      rpc_write_start_request(conn, RPC_cuMemRangeGetAttributes) < 0 ||
      rpc_write(conn, data, sizeof(void *)) < 0 ||
      rpc_write(conn, dataSizes, sizeof(size_t)) < 0 ||
      rpc_write(conn, attributes, sizeof(CUmem_range_attribute)) < 0 ||
      rpc_write(conn, &numAttributes, sizeof(size_t)) < 0 ||
      rpc_write(conn, &devPtr, sizeof(CUdeviceptr)) < 0 ||
      rpc_write(conn, &count, sizeof(size_t)) < 0 ||
      rpc_wait_for_response(conn) < 0 ||
      rpc_read(conn, data, sizeof(void *)) < 0 ||
      rpc_read(conn, dataSizes, sizeof(size_t)) < 0 ||
      rpc_read(conn, attributes, sizeof(CUmem_range_attribute)) < 0 ||
      rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
      rpc_read_end(conn) < 0)
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  return return_value;
}

CUresult cuPointerSetAttribute(const void *value, CUpointer_attribute attribute,
                               CUdeviceptr ptr) {
  lupine_route route = lupine_route_for_deviceptr(ptr);
  CUresult return_value;
  using real_fn_t =
      CUresult (*)(const void *, CUpointer_attribute, CUdeviceptr);
  if (lupine_call_local_cuda_if_routed<real_fn_t>(
          route, "cuPointerSetAttribute", &return_value, value, attribute,
          ptr)) {
    return return_value;
  }
  conn_t *conn = lupine_route_remote_conn(route);
  if (conn == nullptr ||
      rpc_write_start_request(conn, RPC_cuPointerSetAttribute) < 0 ||
      rpc_write(conn, &value, sizeof(const void *)) < 0 ||
      rpc_write(conn, &attribute, sizeof(CUpointer_attribute)) < 0 ||
      rpc_write(conn, &ptr, sizeof(CUdeviceptr)) < 0 ||
      rpc_wait_for_response(conn) < 0 ||
      rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
      rpc_read_end(conn) < 0)
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  return return_value;
}

CUresult cuStreamCreate(CUstream *phStream, unsigned int Flags) {
  lupine_route route = lupine_route_for_current_context();
  CUresult return_value;
  using real_fn_t = CUresult (*)(CUstream *, unsigned int);
  if (lupine_call_local_cuda_if_routed<real_fn_t>(
          route, "cuStreamCreate", &return_value, phStream, Flags)) {
    if (return_value == CUDA_SUCCESS && phStream != nullptr) {
      lupine_note_stream_owner_route(*phStream, route);
    }
    return return_value;
  }
  conn_t *conn = lupine_route_remote_conn(route);
  if (conn == nullptr ||
      rpc_write_start_request(conn, RPC_cuStreamCreate) < 0 ||
      rpc_write(conn, phStream, sizeof(CUstream)) < 0 ||
      rpc_write(conn, &Flags, sizeof(unsigned int)) < 0 ||
      rpc_wait_for_response(conn) < 0 ||
      rpc_read(conn, phStream, sizeof(CUstream)) < 0 ||
      rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
      rpc_read_end(conn) < 0)
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  if (return_value == CUDA_SUCCESS && phStream != nullptr) {
    lupine_note_stream_owner_route(*phStream, route);
  }
  return return_value;
}

CUresult cuStreamCreateWithPriority(CUstream *phStream, unsigned int flags,
                                    int priority) {
  lupine_route route = lupine_route_for_current_context();
  CUresult return_value;
  using real_fn_t = CUresult (*)(CUstream *, unsigned int, int);
  if (lupine_call_local_cuda_if_routed<real_fn_t>(
          route, "cuStreamCreateWithPriority", &return_value, phStream, flags,
          priority)) {
    if (return_value == CUDA_SUCCESS && phStream != nullptr) {
      lupine_note_stream_owner_route(*phStream, route);
    }
    return return_value;
  }
  conn_t *conn = lupine_route_remote_conn(route);
  if (conn == nullptr ||
      rpc_write_start_request(conn, RPC_cuStreamCreateWithPriority) < 0 ||
      rpc_write(conn, phStream, sizeof(CUstream)) < 0 ||
      rpc_write(conn, &flags, sizeof(unsigned int)) < 0 ||
      rpc_write(conn, &priority, sizeof(int)) < 0 ||
      rpc_wait_for_response(conn) < 0 ||
      rpc_read(conn, phStream, sizeof(CUstream)) < 0 ||
      rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
      rpc_read_end(conn) < 0)
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  if (return_value == CUDA_SUCCESS && phStream != nullptr) {
    lupine_note_stream_owner_route(*phStream, route);
  }
  return return_value;
}

CUresult cuStreamGetPriority(CUstream hStream, int *priority) {
  lupine_route route = (hStream != nullptr ? lupine_route_for_stream(hStream)
                                           : lupine_route_for_default());
  CUresult return_value;
  using real_fn_t = CUresult (*)(CUstream, int *);
  if (lupine_call_local_cuda_if_routed<real_fn_t>(
          route, "cuStreamGetPriority", &return_value, hStream, priority)) {
    return return_value;
  }
  conn_t *conn = lupine_route_remote_conn(route);
  if (conn == nullptr ||
      rpc_write_start_request(conn, RPC_cuStreamGetPriority) < 0 ||
      rpc_write(conn, &hStream, sizeof(CUstream)) < 0 ||
      rpc_write(conn, priority, sizeof(int)) < 0 ||
      rpc_wait_for_response(conn) < 0 ||
      rpc_read(conn, priority, sizeof(int)) < 0 ||
      rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
      rpc_read_end(conn) < 0)
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  return return_value;
}

CUresult cuStreamGetFlags(CUstream hStream, unsigned int *flags) {
  lupine_route route = (hStream != nullptr ? lupine_route_for_stream(hStream)
                                           : lupine_route_for_default());
  CUresult return_value;
  using real_fn_t = CUresult (*)(CUstream, unsigned int *);
  if (lupine_call_local_cuda_if_routed<real_fn_t>(
          route, "cuStreamGetFlags", &return_value, hStream, flags)) {
    return return_value;
  }
  conn_t *conn = lupine_route_remote_conn(route);
  if (conn == nullptr ||
      rpc_write_start_request(conn, RPC_cuStreamGetFlags) < 0 ||
      rpc_write(conn, &hStream, sizeof(CUstream)) < 0 ||
      rpc_write(conn, flags, sizeof(unsigned int)) < 0 ||
      rpc_wait_for_response(conn) < 0 ||
      rpc_read(conn, flags, sizeof(unsigned int)) < 0 ||
      rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
      rpc_read_end(conn) < 0)
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  return return_value;
}

CUresult cuStreamGetId(CUstream hStream, unsigned long long *streamId) {
  lupine_route route = (hStream != nullptr ? lupine_route_for_stream(hStream)
                                           : lupine_route_for_default());
  CUresult return_value;
  using real_fn_t = CUresult (*)(CUstream, unsigned long long *);
  if (lupine_call_local_cuda_if_routed<real_fn_t>(
          route, "cuStreamGetId", &return_value, hStream, streamId)) {
    return return_value;
  }
  conn_t *conn = lupine_route_remote_conn(route);
  if (conn == nullptr || rpc_write_start_request(conn, RPC_cuStreamGetId) < 0 ||
      rpc_write(conn, &hStream, sizeof(CUstream)) < 0 ||
      rpc_write(conn, streamId, sizeof(unsigned long long)) < 0 ||
      rpc_wait_for_response(conn) < 0 ||
      rpc_read(conn, streamId, sizeof(unsigned long long)) < 0 ||
      rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
      rpc_read_end(conn) < 0)
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  return return_value;
}

CUresult cuStreamGetCtx(CUstream hStream, CUcontext *pctx) {
  lupine_route route = (hStream != nullptr ? lupine_route_for_stream(hStream)
                                           : lupine_route_for_default());
  CUresult return_value;
  using real_fn_t = CUresult (*)(CUstream, CUcontext *);
  if (lupine_call_local_cuda_if_routed<real_fn_t>(
          route, "cuStreamGetCtx", &return_value, hStream, pctx)) {
    return return_value;
  }
  conn_t *conn = lupine_route_remote_conn(route);
  if (conn == nullptr ||
      rpc_write_start_request(conn, RPC_cuStreamGetCtx) < 0 ||
      rpc_write(conn, &hStream, sizeof(CUstream)) < 0 ||
      rpc_write(conn, pctx, sizeof(CUcontext)) < 0 ||
      rpc_wait_for_response(conn) < 0 ||
      rpc_read(conn, pctx, sizeof(CUcontext)) < 0 ||
      rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
      rpc_read_end(conn) < 0)
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  return return_value;
}

CUresult cuStreamBeginCapture_v2(CUstream hStream, CUstreamCaptureMode mode) {
  lupine_route route = (hStream != nullptr ? lupine_route_for_stream(hStream)
                                           : lupine_route_for_default());
  CUresult return_value;
  using real_fn_t = CUresult (*)(CUstream, CUstreamCaptureMode);
  if (lupine_call_local_cuda_if_routed<real_fn_t>(
          route, "cuStreamBeginCapture_v2", &return_value, hStream, mode)) {
    return return_value;
  }
  conn_t *conn = lupine_route_remote_conn(route);
  if (conn == nullptr ||
      rpc_write_start_request(conn, RPC_cuStreamBeginCapture_v2) < 0 ||
      rpc_write(conn, &hStream, sizeof(CUstream)) < 0 ||
      rpc_write(conn, &mode, sizeof(CUstreamCaptureMode)) < 0 ||
      rpc_wait_for_response(conn) < 0 ||
      rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
      rpc_read_end(conn) < 0)
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  return return_value;
}

CUresult cuThreadExchangeStreamCaptureMode(CUstreamCaptureMode *mode) {
  lupine_route route = lupine_route_for_default();
  CUresult return_value;
  using real_fn_t = CUresult (*)(CUstreamCaptureMode *);
  if (lupine_call_local_cuda_if_routed<real_fn_t>(
          route, "cuThreadExchangeStreamCaptureMode", &return_value, mode)) {
    return return_value;
  }
  conn_t *conn = lupine_route_remote_conn(route);
  if (conn == nullptr ||
      rpc_write_start_request(conn, RPC_cuThreadExchangeStreamCaptureMode) <
          0 ||
      rpc_write(conn, mode, sizeof(CUstreamCaptureMode)) < 0 ||
      rpc_wait_for_response(conn) < 0 ||
      rpc_read(conn, mode, sizeof(CUstreamCaptureMode)) < 0 ||
      rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
      rpc_read_end(conn) < 0)
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  return return_value;
}

CUresult cuStreamEndCapture(CUstream hStream, CUgraph *phGraph) {
  lupine_route route = (hStream != nullptr ? lupine_route_for_stream(hStream)
                                           : lupine_route_for_default());
  CUresult return_value;
  using real_fn_t = CUresult (*)(CUstream, CUgraph *);
  if (lupine_call_local_cuda_if_routed<real_fn_t>(
          route, "cuStreamEndCapture", &return_value, hStream, phGraph)) {
    if (return_value == CUDA_SUCCESS && phGraph != nullptr) {
      lupine_note_graph_owner_route(*phGraph, route);
    }
    return return_value;
  }
  conn_t *conn = lupine_route_remote_conn(route);
  CUgraph *phGraph_null_check;
  if (conn == nullptr ||
      rpc_write_start_request(conn, RPC_cuStreamEndCapture) < 0 ||
      rpc_write(conn, &hStream, sizeof(CUstream)) < 0 ||
      rpc_write(conn, &phGraph, sizeof(CUgraph *)) < 0 ||
      (phGraph != nullptr && rpc_write(conn, phGraph, sizeof(CUgraph)) < 0) ||
      rpc_wait_for_response(conn) < 0 ||
      rpc_read(conn, &phGraph_null_check, sizeof(CUgraph *)) < 0 ||
      (phGraph_null_check && rpc_read(conn, phGraph, sizeof(CUgraph)) < 0) ||
      rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
      rpc_read_end(conn) < 0)
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  if (return_value == CUDA_SUCCESS && phGraph != nullptr) {
    lupine_note_graph_owner_route(*phGraph, route);
  }
  return return_value;
}

CUresult cuStreamIsCapturing(CUstream hStream,
                             CUstreamCaptureStatus *captureStatus) {
  lupine_route route = (hStream != nullptr ? lupine_route_for_stream(hStream)
                                           : lupine_route_for_default());
  CUresult return_value;
  using real_fn_t = CUresult (*)(CUstream, CUstreamCaptureStatus *);
  if (lupine_call_local_cuda_if_routed<real_fn_t>(route, "cuStreamIsCapturing",
                                                  &return_value, hStream,
                                                  captureStatus)) {
    return return_value;
  }
  conn_t *conn = lupine_route_remote_conn(route);
  if (conn == nullptr ||
      rpc_write_start_request(conn, RPC_cuStreamIsCapturing) < 0 ||
      rpc_write(conn, &hStream, sizeof(CUstream)) < 0 ||
      rpc_write(conn, captureStatus, sizeof(CUstreamCaptureStatus)) < 0 ||
      rpc_wait_for_response(conn) < 0 ||
      rpc_read(conn, captureStatus, sizeof(CUstreamCaptureStatus)) < 0 ||
      rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
      rpc_read_end(conn) < 0)
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  return return_value;
}

CUresult cuStreamAttachMemAsync(CUstream hStream, CUdeviceptr dptr,
                                size_t length, unsigned int flags) {
  CUdeviceptr dptr_rpc = dptr;
  bool dptr_is_managed_host =
      lupine_translate_managed_host_ptr(dptr, &dptr_rpc);
  if (dptr_is_managed_host) {
    CUresult managed_result = lupine_flush_dirty_host_pages_to_server();
    if (managed_result != CUDA_SUCCESS) {
      return managed_result;
    }
  }
  lupine_route route =
      (hStream != nullptr ? lupine_route_for_stream(hStream)
                          : lupine_route_for_deviceptr(dptr_rpc));
  CUresult return_value;
  using real_fn_t = CUresult (*)(CUstream, CUdeviceptr, size_t, unsigned int);
  if (lupine_call_local_cuda_if_routed<real_fn_t>(
          route, "cuStreamAttachMemAsync", &return_value, hStream, dptr_rpc,
          length, flags)) {
    return return_value;
  }
  conn_t *conn = lupine_route_remote_conn(route);
  if (conn == nullptr ||
      rpc_write_start_request(conn, RPC_cuStreamAttachMemAsync) < 0 ||
      rpc_write(conn, &hStream, sizeof(CUstream)) < 0 ||
      rpc_write(conn, &dptr_rpc, sizeof(CUdeviceptr)) < 0 ||
      rpc_write(conn, &length, sizeof(size_t)) < 0 ||
      rpc_write(conn, &flags, sizeof(unsigned int)) < 0 ||
      rpc_wait_for_response(conn) < 0 ||
      rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
      rpc_read_end(conn) < 0)
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  return return_value;
}

CUresult cuStreamQuery(CUstream hStream) {
  CUresult lupine_sync_result = lupine_flush_dirty_host_pages_to_server();
  if (lupine_sync_result != CUDA_SUCCESS) {
    return lupine_sync_result;
  }
  lupine_route route = (hStream != nullptr ? lupine_route_for_stream(hStream)
                                           : lupine_route_for_default());
  CUresult return_value;
  using real_fn_t = CUresult (*)(CUstream);
  if (lupine_call_local_cuda_if_routed<real_fn_t>(route, "cuStreamQuery",
                                                  &return_value, hStream)) {
    if (return_value == CUDA_SUCCESS)
      return_value = lupine_sync_mapped_device_to_host();
    return return_value;
  }
  conn_t *conn = lupine_route_remote_conn(route);
  if (conn == nullptr || rpc_write_start_request(conn, RPC_cuStreamQuery) < 0 ||
      rpc_write(conn, &hStream, sizeof(CUstream)) < 0 ||
      rpc_wait_for_response(conn) < 0 ||
      rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
      rpc_read_end(conn) < 0)
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  if (return_value == CUDA_SUCCESS)
    return_value = lupine_sync_mapped_device_to_host();
  return return_value;
}

CUresult cuStreamSynchronize(CUstream hStream) {
  CUresult lupine_sync_result = lupine_flush_dirty_host_pages_to_server();
  if (lupine_sync_result != CUDA_SUCCESS) {
    return lupine_sync_result;
  }
  lupine_route route = (hStream != nullptr ? lupine_route_for_stream(hStream)
                                           : lupine_route_for_default());
  CUresult return_value;
  using real_fn_t = CUresult (*)(CUstream);
  if (lupine_call_local_cuda_if_routed<real_fn_t>(route, "cuStreamSynchronize",
                                                  &return_value, hStream)) {
    if (return_value == CUDA_SUCCESS)
      return_value = lupine_sync_mapped_device_to_host();
    return return_value;
  }
  conn_t *conn = lupine_route_remote_conn(route);
  if (conn == nullptr ||
      rpc_write_start_request(conn, RPC_cuStreamSynchronize) < 0 ||
      rpc_write(conn, &hStream, sizeof(CUstream)) < 0 ||
      rpc_wait_for_response(conn) < 0 ||
      lupine_read_deferred_dtoh_copies(conn) < 0 ||
      lupine_forward_remote_stdout(conn) < 0 ||
      rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
      rpc_read_end(conn) < 0)
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  if (return_value == CUDA_SUCCESS)
    return_value = lupine_sync_mapped_device_to_host();
  return return_value;
}

CUresult cuStreamDestroy_v2(CUstream hStream) {
  lupine_route route = (hStream != nullptr ? lupine_route_for_stream(hStream)
                                           : lupine_route_for_default());
  CUresult return_value;
  using real_fn_t = CUresult (*)(CUstream);
  if (lupine_call_local_cuda_if_routed<real_fn_t>(route, "cuStreamDestroy_v2",
                                                  &return_value, hStream)) {
    return return_value;
  }
  conn_t *conn = lupine_route_remote_conn(route);
  if (conn == nullptr ||
      rpc_write_start_request(conn, RPC_cuStreamDestroy_v2) < 0 ||
      rpc_write(conn, &hStream, sizeof(CUstream)) < 0 ||
      rpc_wait_for_response(conn) < 0 ||
      rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
      rpc_read_end(conn) < 0)
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  return return_value;
}

CUresult cuStreamCopyAttributes(CUstream dst, CUstream src) {
  lupine_route route = (dst != nullptr ? lupine_route_for_stream(dst)
                                       : lupine_route_for_default());
  CUresult return_value;
  using real_fn_t = CUresult (*)(CUstream, CUstream);
  if (lupine_call_local_cuda_if_routed<real_fn_t>(
          route, "cuStreamCopyAttributes", &return_value, dst, src)) {
    return return_value;
  }
  conn_t *conn = lupine_route_remote_conn(route);
  if (conn == nullptr ||
      rpc_write_start_request(conn, RPC_cuStreamCopyAttributes) < 0 ||
      rpc_write(conn, &dst, sizeof(CUstream)) < 0 ||
      rpc_write(conn, &src, sizeof(CUstream)) < 0 ||
      rpc_wait_for_response(conn) < 0 ||
      rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
      rpc_read_end(conn) < 0)
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  return return_value;
}

CUresult cuStreamGetAttribute(CUstream hStream, CUstreamAttrID attr,
                              CUstreamAttrValue *value_out) {
  lupine_route route = (hStream != nullptr ? lupine_route_for_stream(hStream)
                                           : lupine_route_for_default());
  CUresult return_value;
  using real_fn_t = CUresult (*)(CUstream, CUstreamAttrID, CUstreamAttrValue *);
  if (lupine_call_local_cuda_if_routed<real_fn_t>(route, "cuStreamGetAttribute",
                                                  &return_value, hStream, attr,
                                                  value_out)) {
    return return_value;
  }
  conn_t *conn = lupine_route_remote_conn(route);
  if (conn == nullptr ||
      rpc_write_start_request(conn, RPC_cuStreamGetAttribute) < 0 ||
      rpc_write(conn, &hStream, sizeof(CUstream)) < 0 ||
      rpc_write(conn, &attr, sizeof(CUstreamAttrID)) < 0 ||
      rpc_write(conn, value_out, sizeof(CUstreamAttrValue)) < 0 ||
      rpc_wait_for_response(conn) < 0 ||
      rpc_read(conn, value_out, sizeof(CUstreamAttrValue)) < 0 ||
      rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
      rpc_read_end(conn) < 0)
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  return return_value;
}

CUresult cuStreamSetAttribute(CUstream hStream, CUstreamAttrID attr,
                              const CUstreamAttrValue *value) {
  lupine_route route = (hStream != nullptr ? lupine_route_for_stream(hStream)
                                           : lupine_route_for_default());
  CUresult return_value;
  using real_fn_t =
      CUresult (*)(CUstream, CUstreamAttrID, const CUstreamAttrValue *);
  if (lupine_call_local_cuda_if_routed<real_fn_t>(
          route, "cuStreamSetAttribute", &return_value, hStream, attr, value)) {
    return return_value;
  }
  conn_t *conn = lupine_route_remote_conn(route);
  if (conn == nullptr ||
      rpc_write_start_request(conn, RPC_cuStreamSetAttribute) < 0 ||
      rpc_write(conn, &hStream, sizeof(CUstream)) < 0 ||
      rpc_write(conn, &attr, sizeof(CUstreamAttrID)) < 0 ||
      rpc_write(conn, value, sizeof(const CUstreamAttrValue)) < 0 ||
      rpc_wait_for_response(conn) < 0 ||
      rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
      rpc_read_end(conn) < 0)
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  return return_value;
}

CUresult cuEventCreate(CUevent *phEvent, unsigned int Flags) {
  lupine_route route = lupine_route_for_current_context();
  CUresult return_value;
  using real_fn_t = CUresult (*)(CUevent *, unsigned int);
  if (lupine_call_local_cuda_if_routed<real_fn_t>(
          route, "cuEventCreate", &return_value, phEvent, Flags)) {
    if (return_value == CUDA_SUCCESS && phEvent != nullptr) {
      lupine_note_event_owner_route(*phEvent, route);
    }
    return return_value;
  }
  conn_t *conn = lupine_route_remote_conn(route);
  if (conn == nullptr || rpc_write_start_request(conn, RPC_cuEventCreate) < 0 ||
      rpc_write(conn, phEvent, sizeof(CUevent)) < 0 ||
      rpc_write(conn, &Flags, sizeof(unsigned int)) < 0 ||
      rpc_wait_for_response(conn) < 0 ||
      rpc_read(conn, phEvent, sizeof(CUevent)) < 0 ||
      rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
      rpc_read_end(conn) < 0)
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  if (return_value == CUDA_SUCCESS && phEvent != nullptr) {
    lupine_note_event_owner_route(*phEvent, route);
  }
  return return_value;
}

CUresult cuEventRecord(CUevent hEvent, CUstream hStream) {
  lupine_route route = (hStream != nullptr ? lupine_route_for_stream(hStream)
                                           : lupine_route_for_default());
  CUresult return_value;
  using real_fn_t = CUresult (*)(CUevent, CUstream);
  if (lupine_call_local_cuda_if_routed<real_fn_t>(
          route, "cuEventRecord", &return_value, hEvent, hStream)) {
    return return_value;
  }
  conn_t *conn = lupine_route_remote_conn(route);
  if (conn == nullptr || rpc_write_start_request(conn, RPC_cuEventRecord) < 0 ||
      rpc_write(conn, &hEvent, sizeof(CUevent)) < 0 ||
      rpc_write(conn, &hStream, sizeof(CUstream)) < 0 ||
      rpc_wait_for_response(conn) < 0 ||
      rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
      rpc_read_end(conn) < 0)
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  return return_value;
}

CUresult cuEventRecordWithFlags(CUevent hEvent, CUstream hStream,
                                unsigned int flags) {
  lupine_route route = (hStream != nullptr ? lupine_route_for_stream(hStream)
                                           : lupine_route_for_default());
  CUresult return_value;
  using real_fn_t = CUresult (*)(CUevent, CUstream, unsigned int);
  if (lupine_call_local_cuda_if_routed<real_fn_t>(
          route, "cuEventRecordWithFlags", &return_value, hEvent, hStream,
          flags)) {
    return return_value;
  }
  conn_t *conn = lupine_route_remote_conn(route);
  if (conn == nullptr ||
      rpc_write_start_request(conn, RPC_cuEventRecordWithFlags) < 0 ||
      rpc_write(conn, &hEvent, sizeof(CUevent)) < 0 ||
      rpc_write(conn, &hStream, sizeof(CUstream)) < 0 ||
      rpc_write(conn, &flags, sizeof(unsigned int)) < 0 ||
      rpc_wait_for_response(conn) < 0 ||
      rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
      rpc_read_end(conn) < 0)
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  return return_value;
}

CUresult cuEventQuery(CUevent hEvent) {
  CUresult lupine_sync_result = lupine_flush_dirty_host_pages_to_server();
  if (lupine_sync_result != CUDA_SUCCESS) {
    return lupine_sync_result;
  }
  lupine_route route = lupine_route_for_event(hEvent);
  CUresult return_value;
  using real_fn_t = CUresult (*)(CUevent);
  if (lupine_call_local_cuda_if_routed<real_fn_t>(route, "cuEventQuery",
                                                  &return_value, hEvent)) {
    if (return_value == CUDA_SUCCESS)
      return_value = lupine_sync_mapped_device_to_host();
    return return_value;
  }
  conn_t *conn = lupine_route_remote_conn(route);
  if (conn == nullptr || rpc_write_start_request(conn, RPC_cuEventQuery) < 0 ||
      rpc_write(conn, &hEvent, sizeof(CUevent)) < 0 ||
      rpc_wait_for_response(conn) < 0 ||
      lupine_read_deferred_dtoh_copies(conn) < 0 ||
      rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
      rpc_read_end(conn) < 0)
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  if (return_value == CUDA_SUCCESS)
    return_value = lupine_sync_mapped_device_to_host();
  return return_value;
}

CUresult cuEventSynchronize(CUevent hEvent) {
  CUresult lupine_sync_result = lupine_flush_dirty_host_pages_to_server();
  if (lupine_sync_result != CUDA_SUCCESS) {
    return lupine_sync_result;
  }
  lupine_route route = lupine_route_for_event(hEvent);
  CUresult return_value;
  using real_fn_t = CUresult (*)(CUevent);
  if (lupine_call_local_cuda_if_routed<real_fn_t>(route, "cuEventSynchronize",
                                                  &return_value, hEvent)) {
    if (return_value == CUDA_SUCCESS)
      return_value = lupine_sync_mapped_device_to_host();
    return return_value;
  }
  conn_t *conn = lupine_route_remote_conn(route);
  if (conn == nullptr ||
      rpc_write_start_request(conn, RPC_cuEventSynchronize) < 0 ||
      rpc_write(conn, &hEvent, sizeof(CUevent)) < 0 ||
      rpc_wait_for_response(conn) < 0 ||
      lupine_read_deferred_dtoh_copies(conn) < 0 ||
      lupine_forward_remote_stdout(conn) < 0 ||
      rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
      rpc_read_end(conn) < 0)
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  if (return_value == CUDA_SUCCESS)
    return_value = lupine_sync_mapped_device_to_host();
  return return_value;
}

CUresult cuEventDestroy_v2(CUevent hEvent) {
  lupine_route route = lupine_route_for_event(hEvent);
  CUresult return_value;
  using real_fn_t = CUresult (*)(CUevent);
  if (lupine_call_local_cuda_if_routed<real_fn_t>(route, "cuEventDestroy_v2",
                                                  &return_value, hEvent)) {
    return return_value;
  }
  conn_t *conn = lupine_route_remote_conn(route);
  if (conn == nullptr ||
      rpc_write_start_request(conn, RPC_cuEventDestroy_v2) < 0 ||
      rpc_write(conn, &hEvent, sizeof(CUevent)) < 0 ||
      rpc_wait_for_response(conn) < 0 ||
      rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
      rpc_read_end(conn) < 0)
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  return return_value;
}

CUresult cuEventElapsedTime_v2(float *pMilliseconds, CUevent hStart,
                               CUevent hEnd) {
  lupine_route route = lupine_route_for_event(hStart);
  CUresult return_value;
  using real_fn_t = CUresult (*)(float *, CUevent, CUevent);
  if (lupine_call_local_cuda_if_routed<real_fn_t>(
          route, "cuEventElapsedTime_v2", &return_value, pMilliseconds, hStart,
          hEnd)) {
    return return_value;
  }
  conn_t *conn = lupine_route_remote_conn(route);
  if (conn == nullptr ||
      rpc_write_start_request(conn, RPC_cuEventElapsedTime_v2) < 0 ||
      rpc_write(conn, pMilliseconds, sizeof(float)) < 0 ||
      rpc_write(conn, &hStart, sizeof(CUevent)) < 0 ||
      rpc_write(conn, &hEnd, sizeof(CUevent)) < 0 ||
      rpc_wait_for_response(conn) < 0 ||
      rpc_read(conn, pMilliseconds, sizeof(float)) < 0 ||
      rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
      rpc_read_end(conn) < 0)
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  return return_value;
}

CUresult
cuImportExternalMemory(CUexternalMemory *extMem_out,
                       const CUDA_EXTERNAL_MEMORY_HANDLE_DESC *memHandleDesc) {
  lupine_route route = lupine_route_for_default();
  CUresult return_value;
  using real_fn_t = CUresult (*)(CUexternalMemory *,
                                 const CUDA_EXTERNAL_MEMORY_HANDLE_DESC *);
  if (lupine_call_local_cuda_if_routed<real_fn_t>(
          route, "cuImportExternalMemory", &return_value, extMem_out,
          memHandleDesc)) {
    return return_value;
  }
  conn_t *conn = lupine_route_remote_conn(route);
  if (conn == nullptr ||
      rpc_write_start_request(conn, RPC_cuImportExternalMemory) < 0 ||
      rpc_write(conn, extMem_out, sizeof(CUexternalMemory)) < 0 ||
      rpc_write(conn, &memHandleDesc,
                sizeof(const CUDA_EXTERNAL_MEMORY_HANDLE_DESC *)) < 0 ||
      rpc_wait_for_response(conn) < 0 ||
      rpc_read(conn, extMem_out, sizeof(CUexternalMemory)) < 0 ||
      rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
      rpc_read_end(conn) < 0)
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  return return_value;
}

CUresult cuExternalMemoryGetMappedBuffer(
    CUdeviceptr *devPtr, CUexternalMemory extMem,
    const CUDA_EXTERNAL_MEMORY_BUFFER_DESC *bufferDesc) {
  lupine_route route = lupine_route_for_default();
  CUresult return_value;
  using real_fn_t = CUresult (*)(CUdeviceptr *, CUexternalMemory,
                                 const CUDA_EXTERNAL_MEMORY_BUFFER_DESC *);
  if (lupine_call_local_cuda_if_routed<real_fn_t>(
          route, "cuExternalMemoryGetMappedBuffer", &return_value, devPtr,
          extMem, bufferDesc)) {
    return return_value;
  }
  conn_t *conn = lupine_route_remote_conn(route);
  if (conn == nullptr ||
      rpc_write_start_request(conn, RPC_cuExternalMemoryGetMappedBuffer) < 0 ||
      rpc_write(conn, devPtr, sizeof(CUdeviceptr)) < 0 ||
      rpc_write(conn, &extMem, sizeof(CUexternalMemory)) < 0 ||
      rpc_write(conn, &bufferDesc,
                sizeof(const CUDA_EXTERNAL_MEMORY_BUFFER_DESC *)) < 0 ||
      rpc_wait_for_response(conn) < 0 ||
      rpc_read(conn, devPtr, sizeof(CUdeviceptr)) < 0 ||
      rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
      rpc_read_end(conn) < 0)
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  return return_value;
}

CUresult cuExternalMemoryGetMappedMipmappedArray(
    CUmipmappedArray *mipmap, CUexternalMemory extMem,
    const CUDA_EXTERNAL_MEMORY_MIPMAPPED_ARRAY_DESC *mipmapDesc) {
  lupine_route route = lupine_route_for_default();
  CUresult return_value;
  using real_fn_t =
      CUresult (*)(CUmipmappedArray *, CUexternalMemory,
                   const CUDA_EXTERNAL_MEMORY_MIPMAPPED_ARRAY_DESC *);
  if (lupine_call_local_cuda_if_routed<real_fn_t>(
          route, "cuExternalMemoryGetMappedMipmappedArray", &return_value,
          mipmap, extMem, mipmapDesc)) {
    return return_value;
  }
  conn_t *conn = lupine_route_remote_conn(route);
  if (conn == nullptr ||
      rpc_write_start_request(
          conn, RPC_cuExternalMemoryGetMappedMipmappedArray) < 0 ||
      rpc_write(conn, mipmap, sizeof(CUmipmappedArray)) < 0 ||
      rpc_write(conn, &extMem, sizeof(CUexternalMemory)) < 0 ||
      rpc_write(conn, &mipmapDesc,
                sizeof(const CUDA_EXTERNAL_MEMORY_MIPMAPPED_ARRAY_DESC *)) <
          0 ||
      rpc_wait_for_response(conn) < 0 ||
      rpc_read(conn, mipmap, sizeof(CUmipmappedArray)) < 0 ||
      rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
      rpc_read_end(conn) < 0)
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  return return_value;
}

CUresult cuDestroyExternalMemory(CUexternalMemory extMem) {
  lupine_route route = lupine_route_for_default();
  CUresult return_value;
  using real_fn_t = CUresult (*)(CUexternalMemory);
  if (lupine_call_local_cuda_if_routed<real_fn_t>(
          route, "cuDestroyExternalMemory", &return_value, extMem)) {
    return return_value;
  }
  conn_t *conn = lupine_route_remote_conn(route);
  if (conn == nullptr ||
      rpc_write_start_request(conn, RPC_cuDestroyExternalMemory) < 0 ||
      rpc_write(conn, &extMem, sizeof(CUexternalMemory)) < 0 ||
      rpc_wait_for_response(conn) < 0 ||
      rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
      rpc_read_end(conn) < 0)
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  return return_value;
}

CUresult cuImportExternalSemaphore(
    CUexternalSemaphore *extSem_out,
    const CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC *semHandleDesc) {
  lupine_route route = lupine_route_for_default();
  CUresult return_value;
  using real_fn_t = CUresult (*)(CUexternalSemaphore *,
                                 const CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC *);
  if (lupine_call_local_cuda_if_routed<real_fn_t>(
          route, "cuImportExternalSemaphore", &return_value, extSem_out,
          semHandleDesc)) {
    return return_value;
  }
  conn_t *conn = lupine_route_remote_conn(route);
  if (conn == nullptr ||
      rpc_write_start_request(conn, RPC_cuImportExternalSemaphore) < 0 ||
      rpc_write(conn, extSem_out, sizeof(CUexternalSemaphore)) < 0 ||
      rpc_write(conn, &semHandleDesc,
                sizeof(const CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC *)) < 0 ||
      rpc_wait_for_response(conn) < 0 ||
      rpc_read(conn, extSem_out, sizeof(CUexternalSemaphore)) < 0 ||
      rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
      rpc_read_end(conn) < 0)
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  return return_value;
}

CUresult cuSignalExternalSemaphoresAsync(
    const CUexternalSemaphore *extSemArray,
    const CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS *paramsArray,
    unsigned int numExtSems, CUstream stream) {
  lupine_route route = (stream != nullptr ? lupine_route_for_stream(stream)
                                          : lupine_route_for_default());
  CUresult return_value;
  using real_fn_t = CUresult (*)(const CUexternalSemaphore *,
                                 const CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS *,
                                 unsigned int, CUstream);
  if (lupine_call_local_cuda_if_routed<real_fn_t>(
          route, "cuSignalExternalSemaphoresAsync", &return_value, extSemArray,
          paramsArray, numExtSems, stream)) {
    return return_value;
  }
  conn_t *conn = lupine_route_remote_conn(route);
  if (conn == nullptr ||
      rpc_write_start_request(conn, RPC_cuSignalExternalSemaphoresAsync) < 0 ||
      rpc_write(conn, &extSemArray, sizeof(const CUexternalSemaphore *)) < 0 ||
      rpc_write(conn, &paramsArray,
                sizeof(const CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS *)) < 0 ||
      rpc_write(conn, &numExtSems, sizeof(unsigned int)) < 0 ||
      rpc_write(conn, &stream, sizeof(CUstream)) < 0 ||
      rpc_wait_for_response(conn) < 0 ||
      rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
      rpc_read_end(conn) < 0)
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  return return_value;
}

CUresult cuWaitExternalSemaphoresAsync(
    const CUexternalSemaphore *extSemArray,
    const CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS *paramsArray,
    unsigned int numExtSems, CUstream stream) {
  lupine_route route = (stream != nullptr ? lupine_route_for_stream(stream)
                                          : lupine_route_for_default());
  CUresult return_value;
  using real_fn_t = CUresult (*)(const CUexternalSemaphore *,
                                 const CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS *,
                                 unsigned int, CUstream);
  if (lupine_call_local_cuda_if_routed<real_fn_t>(
          route, "cuWaitExternalSemaphoresAsync", &return_value, extSemArray,
          paramsArray, numExtSems, stream)) {
    return return_value;
  }
  conn_t *conn = lupine_route_remote_conn(route);
  if (conn == nullptr ||
      rpc_write_start_request(conn, RPC_cuWaitExternalSemaphoresAsync) < 0 ||
      rpc_write(conn, &extSemArray, sizeof(const CUexternalSemaphore *)) < 0 ||
      rpc_write(conn, &paramsArray,
                sizeof(const CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS *)) < 0 ||
      rpc_write(conn, &numExtSems, sizeof(unsigned int)) < 0 ||
      rpc_write(conn, &stream, sizeof(CUstream)) < 0 ||
      rpc_wait_for_response(conn) < 0 ||
      rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
      rpc_read_end(conn) < 0)
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  return return_value;
}

CUresult cuDestroyExternalSemaphore(CUexternalSemaphore extSem) {
  lupine_route route = lupine_route_for_default();
  CUresult return_value;
  using real_fn_t = CUresult (*)(CUexternalSemaphore);
  if (lupine_call_local_cuda_if_routed<real_fn_t>(
          route, "cuDestroyExternalSemaphore", &return_value, extSem)) {
    return return_value;
  }
  conn_t *conn = lupine_route_remote_conn(route);
  if (conn == nullptr ||
      rpc_write_start_request(conn, RPC_cuDestroyExternalSemaphore) < 0 ||
      rpc_write(conn, &extSem, sizeof(CUexternalSemaphore)) < 0 ||
      rpc_wait_for_response(conn) < 0 ||
      rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
      rpc_read_end(conn) < 0)
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  return return_value;
}

CUresult cuStreamWaitValue32_v2(CUstream stream, CUdeviceptr addr,
                                cuuint32_t value, unsigned int flags) {
  lupine_route route = (stream != nullptr ? lupine_route_for_stream(stream)
                                          : lupine_route_for_default());
  CUresult return_value;
  using real_fn_t =
      CUresult (*)(CUstream, CUdeviceptr, cuuint32_t, unsigned int);
  if (lupine_call_local_cuda_if_routed<real_fn_t>(
          route, "cuStreamWaitValue32_v2", &return_value, stream, addr, value,
          flags)) {
    return return_value;
  }
  conn_t *conn = lupine_route_remote_conn(route);
  if (conn == nullptr ||
      rpc_write_start_request(conn, RPC_cuStreamWaitValue32_v2) < 0 ||
      rpc_write(conn, &stream, sizeof(CUstream)) < 0 ||
      rpc_write(conn, &addr, sizeof(CUdeviceptr)) < 0 ||
      rpc_write(conn, &value, sizeof(cuuint32_t)) < 0 ||
      rpc_write(conn, &flags, sizeof(unsigned int)) < 0 ||
      rpc_wait_for_response(conn) < 0 ||
      rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
      rpc_read_end(conn) < 0)
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  return return_value;
}

CUresult cuStreamWaitValue64_v2(CUstream stream, CUdeviceptr addr,
                                cuuint64_t value, unsigned int flags) {
  lupine_route route = (stream != nullptr ? lupine_route_for_stream(stream)
                                          : lupine_route_for_default());
  CUresult return_value;
  using real_fn_t =
      CUresult (*)(CUstream, CUdeviceptr, cuuint64_t, unsigned int);
  if (lupine_call_local_cuda_if_routed<real_fn_t>(
          route, "cuStreamWaitValue64_v2", &return_value, stream, addr, value,
          flags)) {
    return return_value;
  }
  conn_t *conn = lupine_route_remote_conn(route);
  if (conn == nullptr ||
      rpc_write_start_request(conn, RPC_cuStreamWaitValue64_v2) < 0 ||
      rpc_write(conn, &stream, sizeof(CUstream)) < 0 ||
      rpc_write(conn, &addr, sizeof(CUdeviceptr)) < 0 ||
      rpc_write(conn, &value, sizeof(cuuint64_t)) < 0 ||
      rpc_write(conn, &flags, sizeof(unsigned int)) < 0 ||
      rpc_wait_for_response(conn) < 0 ||
      rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
      rpc_read_end(conn) < 0)
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  return return_value;
}

CUresult cuStreamWriteValue32_v2(CUstream stream, CUdeviceptr addr,
                                 cuuint32_t value, unsigned int flags) {
  lupine_route route = (stream != nullptr ? lupine_route_for_stream(stream)
                                          : lupine_route_for_default());
  CUresult return_value;
  using real_fn_t =
      CUresult (*)(CUstream, CUdeviceptr, cuuint32_t, unsigned int);
  if (lupine_call_local_cuda_if_routed<real_fn_t>(
          route, "cuStreamWriteValue32_v2", &return_value, stream, addr, value,
          flags)) {
    return return_value;
  }
  conn_t *conn = lupine_route_remote_conn(route);
  if (conn == nullptr ||
      rpc_write_start_request(conn, RPC_cuStreamWriteValue32_v2) < 0 ||
      rpc_write(conn, &stream, sizeof(CUstream)) < 0 ||
      rpc_write(conn, &addr, sizeof(CUdeviceptr)) < 0 ||
      rpc_write(conn, &value, sizeof(cuuint32_t)) < 0 ||
      rpc_write(conn, &flags, sizeof(unsigned int)) < 0 ||
      rpc_wait_for_response(conn) < 0 ||
      rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
      rpc_read_end(conn) < 0)
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  return return_value;
}

CUresult cuStreamWriteValue64_v2(CUstream stream, CUdeviceptr addr,
                                 cuuint64_t value, unsigned int flags) {
  lupine_route route = (stream != nullptr ? lupine_route_for_stream(stream)
                                          : lupine_route_for_default());
  CUresult return_value;
  using real_fn_t =
      CUresult (*)(CUstream, CUdeviceptr, cuuint64_t, unsigned int);
  if (lupine_call_local_cuda_if_routed<real_fn_t>(
          route, "cuStreamWriteValue64_v2", &return_value, stream, addr, value,
          flags)) {
    return return_value;
  }
  conn_t *conn = lupine_route_remote_conn(route);
  if (conn == nullptr ||
      rpc_write_start_request(conn, RPC_cuStreamWriteValue64_v2) < 0 ||
      rpc_write(conn, &stream, sizeof(CUstream)) < 0 ||
      rpc_write(conn, &addr, sizeof(CUdeviceptr)) < 0 ||
      rpc_write(conn, &value, sizeof(cuuint64_t)) < 0 ||
      rpc_write(conn, &flags, sizeof(unsigned int)) < 0 ||
      rpc_wait_for_response(conn) < 0 ||
      rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
      rpc_read_end(conn) < 0)
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  return return_value;
}

CUresult cuStreamBatchMemOp_v2(CUstream stream, unsigned int count,
                               CUstreamBatchMemOpParams *paramArray,
                               unsigned int flags) {
  lupine_route route = (stream != nullptr ? lupine_route_for_stream(stream)
                                          : lupine_route_for_default());
  CUresult return_value;
  using real_fn_t = CUresult (*)(CUstream, unsigned int,
                                 CUstreamBatchMemOpParams *, unsigned int);
  if (lupine_call_local_cuda_if_routed<real_fn_t>(
          route, "cuStreamBatchMemOp_v2", &return_value, stream, count,
          paramArray, flags)) {
    return return_value;
  }
  conn_t *conn = lupine_route_remote_conn(route);
  if (conn == nullptr ||
      rpc_write_start_request(conn, RPC_cuStreamBatchMemOp_v2) < 0 ||
      rpc_write(conn, &stream, sizeof(CUstream)) < 0 ||
      rpc_write(conn, &count, sizeof(unsigned int)) < 0 ||
      rpc_write(conn, paramArray, sizeof(CUstreamBatchMemOpParams)) < 0 ||
      rpc_write(conn, &flags, sizeof(unsigned int)) < 0 ||
      rpc_wait_for_response(conn) < 0 ||
      rpc_read(conn, paramArray, sizeof(CUstreamBatchMemOpParams)) < 0 ||
      rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
      rpc_read_end(conn) < 0)
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  return return_value;
}

CUresult cuFuncGetAttribute(int *pi, CUfunction_attribute attrib,
                            CUfunction hfunc) {
  lupine_route route = lupine_route_for_function(hfunc);
  CUresult return_value;
  using real_fn_t = CUresult (*)(int *, CUfunction_attribute, CUfunction);
  if (lupine_call_local_cuda_if_routed<real_fn_t>(
          route, "cuFuncGetAttribute", &return_value, pi, attrib, hfunc)) {
    return return_value;
  }
  conn_t *conn = lupine_route_remote_conn(route);
  CUfunction hfunc_rpc = lupine_translate_private_function_for_rpc(hfunc);
  if (conn == nullptr ||
      rpc_write_start_request(conn, RPC_cuFuncGetAttribute) < 0 ||
      rpc_write(conn, pi, sizeof(int)) < 0 ||
      rpc_write(conn, &attrib, sizeof(CUfunction_attribute)) < 0 ||
      rpc_write(conn, &hfunc_rpc, sizeof(CUfunction)) < 0 ||
      rpc_wait_for_response(conn) < 0 || rpc_read(conn, pi, sizeof(int)) < 0 ||
      rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
      rpc_read_end(conn) < 0)
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  return return_value;
}

CUresult cuFuncSetAttribute(CUfunction hfunc, CUfunction_attribute attrib,
                            int value) {
  lupine_route route = lupine_route_for_function(hfunc);
  CUresult return_value;
  using real_fn_t = CUresult (*)(CUfunction, CUfunction_attribute, int);
  if (lupine_call_local_cuda_if_routed<real_fn_t>(
          route, "cuFuncSetAttribute", &return_value, hfunc, attrib, value)) {
    return return_value;
  }
  conn_t *conn = lupine_route_remote_conn(route);
  CUfunction hfunc_rpc = lupine_translate_private_function_for_rpc(hfunc);
  if (conn == nullptr ||
      rpc_write_start_request(conn, RPC_cuFuncSetAttribute) < 0 ||
      rpc_write(conn, &hfunc_rpc, sizeof(CUfunction)) < 0 ||
      rpc_write(conn, &attrib, sizeof(CUfunction_attribute)) < 0 ||
      rpc_write(conn, &value, sizeof(int)) < 0 ||
      rpc_wait_for_response(conn) < 0 ||
      rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
      rpc_read_end(conn) < 0)
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  return return_value;
}

CUresult cuFuncSetCacheConfig(CUfunction hfunc, CUfunc_cache config) {
  lupine_route route = lupine_route_for_function(hfunc);
  CUresult return_value;
  using real_fn_t = CUresult (*)(CUfunction, CUfunc_cache);
  if (lupine_call_local_cuda_if_routed<real_fn_t>(
          route, "cuFuncSetCacheConfig", &return_value, hfunc, config)) {
    return return_value;
  }
  conn_t *conn = lupine_route_remote_conn(route);
  CUfunction hfunc_rpc = lupine_translate_private_function_for_rpc(hfunc);
  if (conn == nullptr ||
      rpc_write_start_request(conn, RPC_cuFuncSetCacheConfig) < 0 ||
      rpc_write(conn, &hfunc_rpc, sizeof(CUfunction)) < 0 ||
      rpc_write(conn, &config, sizeof(CUfunc_cache)) < 0 ||
      rpc_wait_for_response(conn) < 0 ||
      rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
      rpc_read_end(conn) < 0)
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  return return_value;
}

CUresult cuFuncGetModule(CUmodule *hmod, CUfunction hfunc) {
  lupine_route route = lupine_route_for_function(hfunc);
  CUresult return_value;
  using real_fn_t = CUresult (*)(CUmodule *, CUfunction);
  if (lupine_call_local_cuda_if_routed<real_fn_t>(route, "cuFuncGetModule",
                                                  &return_value, hmod, hfunc)) {
    if (return_value == CUDA_SUCCESS && hmod != nullptr) {
      lupine_note_module_owner_route(*hmod, route);
    }
    return return_value;
  }
  conn_t *conn = lupine_route_remote_conn(route);
  CUfunction hfunc_rpc = lupine_translate_private_function_for_rpc(hfunc);
  if (conn == nullptr ||
      rpc_write_start_request(conn, RPC_cuFuncGetModule) < 0 ||
      rpc_write(conn, hmod, sizeof(CUmodule)) < 0 ||
      rpc_write(conn, &hfunc_rpc, sizeof(CUfunction)) < 0 ||
      rpc_wait_for_response(conn) < 0 ||
      rpc_read(conn, hmod, sizeof(CUmodule)) < 0 ||
      rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
      rpc_read_end(conn) < 0)
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  if (return_value == CUDA_SUCCESS && hmod != nullptr) {
    lupine_note_module_owner_route(*hmod, route);
  }
  return return_value;
}

CUresult
cuLaunchCooperativeKernelMultiDevice(CUDA_LAUNCH_PARAMS *launchParamsList,
                                     unsigned int numDevices,
                                     unsigned int flags) {
  lupine_route route = lupine_route_for_default();
  CUresult return_value;
  using real_fn_t =
      CUresult (*)(CUDA_LAUNCH_PARAMS *, unsigned int, unsigned int);
  if (lupine_call_local_cuda_if_routed<real_fn_t>(
          route, "cuLaunchCooperativeKernelMultiDevice", &return_value,
          launchParamsList, numDevices, flags)) {
    return return_value;
  }
  conn_t *conn = lupine_route_remote_conn(route);
  if (conn == nullptr ||
      rpc_write_start_request(conn, RPC_cuLaunchCooperativeKernelMultiDevice) <
          0 ||
      rpc_write(conn, launchParamsList, sizeof(CUDA_LAUNCH_PARAMS)) < 0 ||
      rpc_write(conn, &numDevices, sizeof(unsigned int)) < 0 ||
      rpc_write(conn, &flags, sizeof(unsigned int)) < 0 ||
      rpc_wait_for_response(conn) < 0 ||
      rpc_read(conn, launchParamsList, sizeof(CUDA_LAUNCH_PARAMS)) < 0 ||
      rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
      rpc_read_end(conn) < 0)
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  return return_value;
}

CUresult cuFuncSetBlockShape(CUfunction hfunc, int x, int y, int z) {
  lupine_route route = lupine_route_for_function(hfunc);
  CUresult return_value;
  using real_fn_t = CUresult (*)(CUfunction, int, int, int);
  if (lupine_call_local_cuda_if_routed<real_fn_t>(
          route, "cuFuncSetBlockShape", &return_value, hfunc, x, y, z)) {
    return return_value;
  }
  conn_t *conn = lupine_route_remote_conn(route);
  CUfunction hfunc_rpc = lupine_translate_private_function_for_rpc(hfunc);
  if (conn == nullptr ||
      rpc_write_start_request(conn, RPC_cuFuncSetBlockShape) < 0 ||
      rpc_write(conn, &hfunc_rpc, sizeof(CUfunction)) < 0 ||
      rpc_write(conn, &x, sizeof(int)) < 0 ||
      rpc_write(conn, &y, sizeof(int)) < 0 ||
      rpc_write(conn, &z, sizeof(int)) < 0 || rpc_wait_for_response(conn) < 0 ||
      rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
      rpc_read_end(conn) < 0)
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  return return_value;
}

CUresult cuFuncSetSharedSize(CUfunction hfunc, unsigned int bytes) {
  lupine_route route = lupine_route_for_function(hfunc);
  CUresult return_value;
  using real_fn_t = CUresult (*)(CUfunction, unsigned int);
  if (lupine_call_local_cuda_if_routed<real_fn_t>(
          route, "cuFuncSetSharedSize", &return_value, hfunc, bytes)) {
    return return_value;
  }
  conn_t *conn = lupine_route_remote_conn(route);
  CUfunction hfunc_rpc = lupine_translate_private_function_for_rpc(hfunc);
  if (conn == nullptr ||
      rpc_write_start_request(conn, RPC_cuFuncSetSharedSize) < 0 ||
      rpc_write(conn, &hfunc_rpc, sizeof(CUfunction)) < 0 ||
      rpc_write(conn, &bytes, sizeof(unsigned int)) < 0 ||
      rpc_wait_for_response(conn) < 0 ||
      rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
      rpc_read_end(conn) < 0)
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  return return_value;
}

CUresult cuParamSetSize(CUfunction hfunc, unsigned int numbytes) {
  lupine_route route = lupine_route_for_function(hfunc);
  CUresult return_value;
  using real_fn_t = CUresult (*)(CUfunction, unsigned int);
  if (lupine_call_local_cuda_if_routed<real_fn_t>(
          route, "cuParamSetSize", &return_value, hfunc, numbytes)) {
    return return_value;
  }
  conn_t *conn = lupine_route_remote_conn(route);
  CUfunction hfunc_rpc = lupine_translate_private_function_for_rpc(hfunc);
  if (conn == nullptr ||
      rpc_write_start_request(conn, RPC_cuParamSetSize) < 0 ||
      rpc_write(conn, &hfunc_rpc, sizeof(CUfunction)) < 0 ||
      rpc_write(conn, &numbytes, sizeof(unsigned int)) < 0 ||
      rpc_wait_for_response(conn) < 0 ||
      rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
      rpc_read_end(conn) < 0)
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  return return_value;
}

CUresult cuParamSeti(CUfunction hfunc, int offset, unsigned int value) {
  lupine_route route = lupine_route_for_function(hfunc);
  CUresult return_value;
  using real_fn_t = CUresult (*)(CUfunction, int, unsigned int);
  if (lupine_call_local_cuda_if_routed<real_fn_t>(
          route, "cuParamSeti", &return_value, hfunc, offset, value)) {
    return return_value;
  }
  conn_t *conn = lupine_route_remote_conn(route);
  CUfunction hfunc_rpc = lupine_translate_private_function_for_rpc(hfunc);
  if (conn == nullptr || rpc_write_start_request(conn, RPC_cuParamSeti) < 0 ||
      rpc_write(conn, &hfunc_rpc, sizeof(CUfunction)) < 0 ||
      rpc_write(conn, &offset, sizeof(int)) < 0 ||
      rpc_write(conn, &value, sizeof(unsigned int)) < 0 ||
      rpc_wait_for_response(conn) < 0 ||
      rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
      rpc_read_end(conn) < 0)
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  return return_value;
}

CUresult cuParamSetf(CUfunction hfunc, int offset, float value) {
  lupine_route route = lupine_route_for_function(hfunc);
  CUresult return_value;
  using real_fn_t = CUresult (*)(CUfunction, int, float);
  if (lupine_call_local_cuda_if_routed<real_fn_t>(
          route, "cuParamSetf", &return_value, hfunc, offset, value)) {
    return return_value;
  }
  conn_t *conn = lupine_route_remote_conn(route);
  CUfunction hfunc_rpc = lupine_translate_private_function_for_rpc(hfunc);
  if (conn == nullptr || rpc_write_start_request(conn, RPC_cuParamSetf) < 0 ||
      rpc_write(conn, &hfunc_rpc, sizeof(CUfunction)) < 0 ||
      rpc_write(conn, &offset, sizeof(int)) < 0 ||
      rpc_write(conn, &value, sizeof(float)) < 0 ||
      rpc_wait_for_response(conn) < 0 ||
      rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
      rpc_read_end(conn) < 0)
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  return return_value;
}

CUresult cuLaunch(CUfunction f) {
  lupine_route route = lupine_route_for_function(f);
  CUresult return_value;
  using real_fn_t = CUresult (*)(CUfunction);
  if (lupine_call_local_cuda_if_routed<real_fn_t>(route, "cuLaunch",
                                                  &return_value, f)) {
    return return_value;
  }
  conn_t *conn = lupine_route_remote_conn(route);
  CUfunction f_rpc = lupine_translate_private_function_for_rpc(f);
  if (conn == nullptr || rpc_write_start_request(conn, RPC_cuLaunch) < 0 ||
      rpc_write(conn, &f_rpc, sizeof(CUfunction)) < 0 ||
      rpc_wait_for_response(conn) < 0 ||
      rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
      rpc_read_end(conn) < 0)
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  return return_value;
}

CUresult cuLaunchGrid(CUfunction f, int grid_width, int grid_height) {
  lupine_route route = lupine_route_for_function(f);
  CUresult return_value;
  using real_fn_t = CUresult (*)(CUfunction, int, int);
  if (lupine_call_local_cuda_if_routed<real_fn_t>(
          route, "cuLaunchGrid", &return_value, f, grid_width, grid_height)) {
    return return_value;
  }
  conn_t *conn = lupine_route_remote_conn(route);
  CUfunction f_rpc = lupine_translate_private_function_for_rpc(f);
  if (conn == nullptr || rpc_write_start_request(conn, RPC_cuLaunchGrid) < 0 ||
      rpc_write(conn, &f_rpc, sizeof(CUfunction)) < 0 ||
      rpc_write(conn, &grid_width, sizeof(int)) < 0 ||
      rpc_write(conn, &grid_height, sizeof(int)) < 0 ||
      rpc_wait_for_response(conn) < 0 ||
      rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
      rpc_read_end(conn) < 0)
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  return return_value;
}

CUresult cuLaunchGridAsync(CUfunction f, int grid_width, int grid_height,
                           CUstream hStream) {
  lupine_route route = lupine_route_for_function(f);
  CUresult return_value;
  using real_fn_t = CUresult (*)(CUfunction, int, int, CUstream);
  if (lupine_call_local_cuda_if_routed<real_fn_t>(route, "cuLaunchGridAsync",
                                                  &return_value, f, grid_width,
                                                  grid_height, hStream)) {
    return return_value;
  }
  conn_t *conn = lupine_route_remote_conn(route);
  CUfunction f_rpc = lupine_translate_private_function_for_rpc(f);
  if (conn == nullptr ||
      rpc_write_start_request(conn, RPC_cuLaunchGridAsync) < 0 ||
      rpc_write(conn, &f_rpc, sizeof(CUfunction)) < 0 ||
      rpc_write(conn, &grid_width, sizeof(int)) < 0 ||
      rpc_write(conn, &grid_height, sizeof(int)) < 0 ||
      rpc_write(conn, &hStream, sizeof(CUstream)) < 0 ||
      rpc_wait_for_response(conn) < 0 ||
      rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
      rpc_read_end(conn) < 0)
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  return return_value;
}

CUresult cuParamSetTexRef(CUfunction hfunc, int texunit, CUtexref hTexRef) {
  lupine_route route = lupine_route_for_function(hfunc);
  CUresult return_value;
  using real_fn_t = CUresult (*)(CUfunction, int, CUtexref);
  if (lupine_call_local_cuda_if_routed<real_fn_t>(
          route, "cuParamSetTexRef", &return_value, hfunc, texunit, hTexRef)) {
    return return_value;
  }
  conn_t *conn = lupine_route_remote_conn(route);
  CUfunction hfunc_rpc = lupine_translate_private_function_for_rpc(hfunc);
  if (conn == nullptr ||
      rpc_write_start_request(conn, RPC_cuParamSetTexRef) < 0 ||
      rpc_write(conn, &hfunc_rpc, sizeof(CUfunction)) < 0 ||
      rpc_write(conn, &texunit, sizeof(int)) < 0 ||
      rpc_write(conn, &hTexRef, sizeof(CUtexref)) < 0 ||
      rpc_wait_for_response(conn) < 0 ||
      rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
      rpc_read_end(conn) < 0)
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  return return_value;
}

CUresult cuFuncSetSharedMemConfig(CUfunction hfunc, CUsharedconfig config) {
  lupine_route route = lupine_route_for_function(hfunc);
  CUresult return_value;
  using real_fn_t = CUresult (*)(CUfunction, CUsharedconfig);
  if (lupine_call_local_cuda_if_routed<real_fn_t>(
          route, "cuFuncSetSharedMemConfig", &return_value, hfunc, config)) {
    return return_value;
  }
  conn_t *conn = lupine_route_remote_conn(route);
  CUfunction hfunc_rpc = lupine_translate_private_function_for_rpc(hfunc);
  if (conn == nullptr ||
      rpc_write_start_request(conn, RPC_cuFuncSetSharedMemConfig) < 0 ||
      rpc_write(conn, &hfunc_rpc, sizeof(CUfunction)) < 0 ||
      rpc_write(conn, &config, sizeof(CUsharedconfig)) < 0 ||
      rpc_wait_for_response(conn) < 0 ||
      rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
      rpc_read_end(conn) < 0)
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  return return_value;
}

CUresult cuGraphCreate(CUgraph *phGraph, unsigned int flags) {
  lupine_route route = lupine_route_for_default();
  CUresult return_value;
  using real_fn_t = CUresult (*)(CUgraph *, unsigned int);
  if (lupine_call_local_cuda_if_routed<real_fn_t>(
          route, "cuGraphCreate", &return_value, phGraph, flags)) {
    if (return_value == CUDA_SUCCESS && phGraph != nullptr) {
      lupine_note_graph_owner_route(*phGraph, route);
    }
    return return_value;
  }
  conn_t *conn = lupine_route_remote_conn(route);
  if (conn == nullptr || rpc_write_start_request(conn, RPC_cuGraphCreate) < 0 ||
      rpc_write(conn, phGraph, sizeof(CUgraph)) < 0 ||
      rpc_write(conn, &flags, sizeof(unsigned int)) < 0 ||
      rpc_wait_for_response(conn) < 0 ||
      rpc_read(conn, phGraph, sizeof(CUgraph)) < 0 ||
      rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
      rpc_read_end(conn) < 0)
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  if (return_value == CUDA_SUCCESS && phGraph != nullptr) {
    lupine_note_graph_owner_route(*phGraph, route);
  }
  return return_value;
}

CUresult cuGraphMemcpyNodeGetParams(CUgraphNode hNode,
                                    CUDA_MEMCPY3D *nodeParams) {
  lupine_route route = lupine_route_for_graph_node(hNode);
  CUresult return_value;
  using real_fn_t = CUresult (*)(CUgraphNode, CUDA_MEMCPY3D *);
  if (lupine_call_local_cuda_if_routed<real_fn_t>(
          route, "cuGraphMemcpyNodeGetParams", &return_value, hNode,
          nodeParams)) {
    return return_value;
  }
  conn_t *conn = lupine_route_remote_conn(route);
  if (conn == nullptr ||
      rpc_write_start_request(conn, RPC_cuGraphMemcpyNodeGetParams) < 0 ||
      rpc_write(conn, &hNode, sizeof(CUgraphNode)) < 0 ||
      rpc_write(conn, nodeParams, sizeof(CUDA_MEMCPY3D)) < 0 ||
      rpc_wait_for_response(conn) < 0 ||
      rpc_read(conn, nodeParams, sizeof(CUDA_MEMCPY3D)) < 0 ||
      rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
      rpc_read_end(conn) < 0)
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  return return_value;
}

CUresult cuGraphMemcpyNodeSetParams(CUgraphNode hNode,
                                    const CUDA_MEMCPY3D *nodeParams) {
  lupine_route route = lupine_route_for_graph_node(hNode);
  CUresult return_value;
  using real_fn_t = CUresult (*)(CUgraphNode, const CUDA_MEMCPY3D *);
  if (lupine_call_local_cuda_if_routed<real_fn_t>(
          route, "cuGraphMemcpyNodeSetParams", &return_value, hNode,
          nodeParams)) {
    return return_value;
  }
  conn_t *conn = lupine_route_remote_conn(route);
  if (conn == nullptr ||
      rpc_write_start_request(conn, RPC_cuGraphMemcpyNodeSetParams) < 0 ||
      rpc_write(conn, &hNode, sizeof(CUgraphNode)) < 0 ||
      rpc_write(conn, nodeParams, sizeof(const CUDA_MEMCPY3D)) < 0 ||
      rpc_wait_for_response(conn) < 0 ||
      rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
      rpc_read_end(conn) < 0)
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  return return_value;
}

CUresult cuGraphMemsetNodeGetParams(CUgraphNode hNode,
                                    CUDA_MEMSET_NODE_PARAMS *nodeParams) {
  lupine_route route = lupine_route_for_graph_node(hNode);
  CUresult return_value;
  using real_fn_t = CUresult (*)(CUgraphNode, CUDA_MEMSET_NODE_PARAMS *);
  if (lupine_call_local_cuda_if_routed<real_fn_t>(
          route, "cuGraphMemsetNodeGetParams", &return_value, hNode,
          nodeParams)) {
    return return_value;
  }
  conn_t *conn = lupine_route_remote_conn(route);
  if (conn == nullptr ||
      rpc_write_start_request(conn, RPC_cuGraphMemsetNodeGetParams) < 0 ||
      rpc_write(conn, &hNode, sizeof(CUgraphNode)) < 0 ||
      rpc_write(conn, nodeParams, sizeof(CUDA_MEMSET_NODE_PARAMS)) < 0 ||
      rpc_wait_for_response(conn) < 0 ||
      rpc_read(conn, nodeParams, sizeof(CUDA_MEMSET_NODE_PARAMS)) < 0 ||
      rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
      rpc_read_end(conn) < 0)
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  return return_value;
}

CUresult cuGraphMemsetNodeSetParams(CUgraphNode hNode,
                                    const CUDA_MEMSET_NODE_PARAMS *nodeParams) {
  lupine_route route = lupine_route_for_graph_node(hNode);
  CUresult return_value;
  using real_fn_t = CUresult (*)(CUgraphNode, const CUDA_MEMSET_NODE_PARAMS *);
  if (lupine_call_local_cuda_if_routed<real_fn_t>(
          route, "cuGraphMemsetNodeSetParams", &return_value, hNode,
          nodeParams)) {
    return return_value;
  }
  conn_t *conn = lupine_route_remote_conn(route);
  if (conn == nullptr ||
      rpc_write_start_request(conn, RPC_cuGraphMemsetNodeSetParams) < 0 ||
      rpc_write(conn, &hNode, sizeof(CUgraphNode)) < 0 ||
      rpc_write(conn, nodeParams, sizeof(const CUDA_MEMSET_NODE_PARAMS)) < 0 ||
      rpc_wait_for_response(conn) < 0 ||
      rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
      rpc_read_end(conn) < 0)
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  return return_value;
}

CUresult cuGraphAddChildGraphNode(CUgraphNode *phGraphNode, CUgraph hGraph,
                                  const CUgraphNode *dependencies,
                                  size_t numDependencies, CUgraph childGraph) {
  lupine_route route = lupine_route_for_graph(hGraph);
  CUresult return_value;
  using real_fn_t = CUresult (*)(CUgraphNode *, CUgraph, const CUgraphNode *,
                                 size_t, CUgraph);
  if (lupine_call_local_cuda_if_routed<real_fn_t>(
          route, "cuGraphAddChildGraphNode", &return_value, phGraphNode, hGraph,
          dependencies, numDependencies, childGraph)) {
    if (return_value == CUDA_SUCCESS && phGraphNode != nullptr) {
      lupine_note_graph_node_owner_route(*phGraphNode, route);
    }
    return return_value;
  }
  conn_t *conn = lupine_route_remote_conn(route);
  if (numDependencies * sizeof(const CUgraphNode) != 0 &&
      dependencies == nullptr)
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  if (conn == nullptr ||
      rpc_write_start_request(conn, RPC_cuGraphAddChildGraphNode) < 0 ||
      rpc_write(conn, phGraphNode, sizeof(CUgraphNode)) < 0 ||
      rpc_write(conn, &hGraph, sizeof(CUgraph)) < 0 ||
      rpc_write(conn, &numDependencies, sizeof(size_t)) < 0 ||
      (numDependencies * sizeof(const CUgraphNode) != 0 &&
       rpc_write(conn, dependencies,
                 numDependencies * sizeof(const CUgraphNode)) < 0) ||
      rpc_write(conn, &childGraph, sizeof(CUgraph)) < 0 ||
      rpc_wait_for_response(conn) < 0 ||
      rpc_read(conn, phGraphNode, sizeof(CUgraphNode)) < 0 ||
      rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
      rpc_read_end(conn) < 0)
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  if (return_value == CUDA_SUCCESS && phGraphNode != nullptr) {
    lupine_note_graph_node_owner_route(*phGraphNode, route);
  }
  return return_value;
}

CUresult cuGraphChildGraphNodeGetGraph(CUgraphNode hNode, CUgraph *phGraph) {
  lupine_route route = lupine_route_for_graph_node(hNode);
  CUresult return_value;
  using real_fn_t = CUresult (*)(CUgraphNode, CUgraph *);
  if (lupine_call_local_cuda_if_routed<real_fn_t>(
          route, "cuGraphChildGraphNodeGetGraph", &return_value, hNode,
          phGraph)) {
    if (return_value == CUDA_SUCCESS && phGraph != nullptr) {
      lupine_note_graph_owner_route(*phGraph, route);
    }
    return return_value;
  }
  conn_t *conn = lupine_route_remote_conn(route);
  if (conn == nullptr ||
      rpc_write_start_request(conn, RPC_cuGraphChildGraphNodeGetGraph) < 0 ||
      rpc_write(conn, &hNode, sizeof(CUgraphNode)) < 0 ||
      rpc_write(conn, phGraph, sizeof(CUgraph)) < 0 ||
      rpc_wait_for_response(conn) < 0 ||
      rpc_read(conn, phGraph, sizeof(CUgraph)) < 0 ||
      rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
      rpc_read_end(conn) < 0)
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  if (return_value == CUDA_SUCCESS && phGraph != nullptr) {
    lupine_note_graph_owner_route(*phGraph, route);
  }
  return return_value;
}

CUresult cuGraphAddEmptyNode(CUgraphNode *phGraphNode, CUgraph hGraph,
                             const CUgraphNode *dependencies,
                             size_t numDependencies) {
  lupine_route route = lupine_route_for_graph(hGraph);
  CUresult return_value;
  using real_fn_t =
      CUresult (*)(CUgraphNode *, CUgraph, const CUgraphNode *, size_t);
  if (lupine_call_local_cuda_if_routed<real_fn_t>(
          route, "cuGraphAddEmptyNode", &return_value, phGraphNode, hGraph,
          dependencies, numDependencies)) {
    if (return_value == CUDA_SUCCESS && phGraphNode != nullptr) {
      lupine_note_graph_node_owner_route(*phGraphNode, route);
    }
    return return_value;
  }
  conn_t *conn = lupine_route_remote_conn(route);
  if (numDependencies * sizeof(const CUgraphNode) != 0 &&
      dependencies == nullptr)
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  if (conn == nullptr ||
      rpc_write_start_request(conn, RPC_cuGraphAddEmptyNode) < 0 ||
      rpc_write(conn, phGraphNode, sizeof(CUgraphNode)) < 0 ||
      rpc_write(conn, &hGraph, sizeof(CUgraph)) < 0 ||
      rpc_write(conn, &numDependencies, sizeof(size_t)) < 0 ||
      (numDependencies * sizeof(const CUgraphNode) != 0 &&
       rpc_write(conn, dependencies,
                 numDependencies * sizeof(const CUgraphNode)) < 0) ||
      rpc_wait_for_response(conn) < 0 ||
      rpc_read(conn, phGraphNode, sizeof(CUgraphNode)) < 0 ||
      rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
      rpc_read_end(conn) < 0)
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  if (return_value == CUDA_SUCCESS && phGraphNode != nullptr) {
    lupine_note_graph_node_owner_route(*phGraphNode, route);
  }
  return return_value;
}

CUresult cuGraphAddEventRecordNode(CUgraphNode *phGraphNode, CUgraph hGraph,
                                   const CUgraphNode *dependencies,
                                   size_t numDependencies, CUevent event) {
  lupine_route route = lupine_route_for_graph(hGraph);
  CUresult return_value;
  using real_fn_t = CUresult (*)(CUgraphNode *, CUgraph, const CUgraphNode *,
                                 size_t, CUevent);
  if (lupine_call_local_cuda_if_routed<real_fn_t>(
          route, "cuGraphAddEventRecordNode", &return_value, phGraphNode,
          hGraph, dependencies, numDependencies, event)) {
    if (return_value == CUDA_SUCCESS && phGraphNode != nullptr) {
      lupine_note_graph_node_owner_route(*phGraphNode, route);
    }
    return return_value;
  }
  conn_t *conn = lupine_route_remote_conn(route);
  if (numDependencies * sizeof(const CUgraphNode) != 0 &&
      dependencies == nullptr)
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  if (conn == nullptr ||
      rpc_write_start_request(conn, RPC_cuGraphAddEventRecordNode) < 0 ||
      rpc_write(conn, phGraphNode, sizeof(CUgraphNode)) < 0 ||
      rpc_write(conn, &hGraph, sizeof(CUgraph)) < 0 ||
      rpc_write(conn, &numDependencies, sizeof(size_t)) < 0 ||
      (numDependencies * sizeof(const CUgraphNode) != 0 &&
       rpc_write(conn, dependencies,
                 numDependencies * sizeof(const CUgraphNode)) < 0) ||
      rpc_write(conn, &event, sizeof(CUevent)) < 0 ||
      rpc_wait_for_response(conn) < 0 ||
      rpc_read(conn, phGraphNode, sizeof(CUgraphNode)) < 0 ||
      rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
      rpc_read_end(conn) < 0)
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  if (return_value == CUDA_SUCCESS && phGraphNode != nullptr) {
    lupine_note_graph_node_owner_route(*phGraphNode, route);
  }
  return return_value;
}

CUresult cuGraphEventRecordNodeGetEvent(CUgraphNode hNode, CUevent *event_out) {
  lupine_route route = lupine_route_for_graph_node(hNode);
  CUresult return_value;
  using real_fn_t = CUresult (*)(CUgraphNode, CUevent *);
  if (lupine_call_local_cuda_if_routed<real_fn_t>(
          route, "cuGraphEventRecordNodeGetEvent", &return_value, hNode,
          event_out)) {
    return return_value;
  }
  conn_t *conn = lupine_route_remote_conn(route);
  if (conn == nullptr ||
      rpc_write_start_request(conn, RPC_cuGraphEventRecordNodeGetEvent) < 0 ||
      rpc_write(conn, &hNode, sizeof(CUgraphNode)) < 0 ||
      rpc_write(conn, event_out, sizeof(CUevent)) < 0 ||
      rpc_wait_for_response(conn) < 0 ||
      rpc_read(conn, event_out, sizeof(CUevent)) < 0 ||
      rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
      rpc_read_end(conn) < 0)
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  return return_value;
}

CUresult cuGraphEventRecordNodeSetEvent(CUgraphNode hNode, CUevent event) {
  lupine_route route = lupine_route_for_graph_node(hNode);
  CUresult return_value;
  using real_fn_t = CUresult (*)(CUgraphNode, CUevent);
  if (lupine_call_local_cuda_if_routed<real_fn_t>(
          route, "cuGraphEventRecordNodeSetEvent", &return_value, hNode,
          event)) {
    return return_value;
  }
  conn_t *conn = lupine_route_remote_conn(route);
  if (conn == nullptr ||
      rpc_write_start_request(conn, RPC_cuGraphEventRecordNodeSetEvent) < 0 ||
      rpc_write(conn, &hNode, sizeof(CUgraphNode)) < 0 ||
      rpc_write(conn, &event, sizeof(CUevent)) < 0 ||
      rpc_wait_for_response(conn) < 0 ||
      rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
      rpc_read_end(conn) < 0)
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  return return_value;
}

CUresult cuGraphAddEventWaitNode(CUgraphNode *phGraphNode, CUgraph hGraph,
                                 const CUgraphNode *dependencies,
                                 size_t numDependencies, CUevent event) {
  lupine_route route = lupine_route_for_graph(hGraph);
  CUresult return_value;
  using real_fn_t = CUresult (*)(CUgraphNode *, CUgraph, const CUgraphNode *,
                                 size_t, CUevent);
  if (lupine_call_local_cuda_if_routed<real_fn_t>(
          route, "cuGraphAddEventWaitNode", &return_value, phGraphNode, hGraph,
          dependencies, numDependencies, event)) {
    if (return_value == CUDA_SUCCESS && phGraphNode != nullptr) {
      lupine_note_graph_node_owner_route(*phGraphNode, route);
    }
    return return_value;
  }
  conn_t *conn = lupine_route_remote_conn(route);
  if (numDependencies * sizeof(const CUgraphNode) != 0 &&
      dependencies == nullptr)
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  if (conn == nullptr ||
      rpc_write_start_request(conn, RPC_cuGraphAddEventWaitNode) < 0 ||
      rpc_write(conn, phGraphNode, sizeof(CUgraphNode)) < 0 ||
      rpc_write(conn, &hGraph, sizeof(CUgraph)) < 0 ||
      rpc_write(conn, &numDependencies, sizeof(size_t)) < 0 ||
      (numDependencies * sizeof(const CUgraphNode) != 0 &&
       rpc_write(conn, dependencies,
                 numDependencies * sizeof(const CUgraphNode)) < 0) ||
      rpc_write(conn, &event, sizeof(CUevent)) < 0 ||
      rpc_wait_for_response(conn) < 0 ||
      rpc_read(conn, phGraphNode, sizeof(CUgraphNode)) < 0 ||
      rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
      rpc_read_end(conn) < 0)
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  if (return_value == CUDA_SUCCESS && phGraphNode != nullptr) {
    lupine_note_graph_node_owner_route(*phGraphNode, route);
  }
  return return_value;
}

CUresult cuGraphEventWaitNodeGetEvent(CUgraphNode hNode, CUevent *event_out) {
  lupine_route route = lupine_route_for_graph_node(hNode);
  CUresult return_value;
  using real_fn_t = CUresult (*)(CUgraphNode, CUevent *);
  if (lupine_call_local_cuda_if_routed<real_fn_t>(
          route, "cuGraphEventWaitNodeGetEvent", &return_value, hNode,
          event_out)) {
    return return_value;
  }
  conn_t *conn = lupine_route_remote_conn(route);
  if (conn == nullptr ||
      rpc_write_start_request(conn, RPC_cuGraphEventWaitNodeGetEvent) < 0 ||
      rpc_write(conn, &hNode, sizeof(CUgraphNode)) < 0 ||
      rpc_write(conn, event_out, sizeof(CUevent)) < 0 ||
      rpc_wait_for_response(conn) < 0 ||
      rpc_read(conn, event_out, sizeof(CUevent)) < 0 ||
      rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
      rpc_read_end(conn) < 0)
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  return return_value;
}

CUresult cuGraphEventWaitNodeSetEvent(CUgraphNode hNode, CUevent event) {
  lupine_route route = lupine_route_for_graph_node(hNode);
  CUresult return_value;
  using real_fn_t = CUresult (*)(CUgraphNode, CUevent);
  if (lupine_call_local_cuda_if_routed<real_fn_t>(
          route, "cuGraphEventWaitNodeSetEvent", &return_value, hNode, event)) {
    return return_value;
  }
  conn_t *conn = lupine_route_remote_conn(route);
  if (conn == nullptr ||
      rpc_write_start_request(conn, RPC_cuGraphEventWaitNodeSetEvent) < 0 ||
      rpc_write(conn, &hNode, sizeof(CUgraphNode)) < 0 ||
      rpc_write(conn, &event, sizeof(CUevent)) < 0 ||
      rpc_wait_for_response(conn) < 0 ||
      rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
      rpc_read_end(conn) < 0)
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  return return_value;
}

CUresult cuGraphAddExternalSemaphoresSignalNode(
    CUgraphNode *phGraphNode, CUgraph hGraph, const CUgraphNode *dependencies,
    size_t numDependencies, const CUDA_EXT_SEM_SIGNAL_NODE_PARAMS *nodeParams) {
  lupine_route route = lupine_route_for_graph(hGraph);
  CUresult return_value;
  using real_fn_t =
      CUresult (*)(CUgraphNode *, CUgraph, const CUgraphNode *, size_t,
                   const CUDA_EXT_SEM_SIGNAL_NODE_PARAMS *);
  if (lupine_call_local_cuda_if_routed<real_fn_t>(
          route, "cuGraphAddExternalSemaphoresSignalNode", &return_value,
          phGraphNode, hGraph, dependencies, numDependencies, nodeParams)) {
    if (return_value == CUDA_SUCCESS && phGraphNode != nullptr) {
      lupine_note_graph_node_owner_route(*phGraphNode, route);
    }
    return return_value;
  }
  conn_t *conn = lupine_route_remote_conn(route);
  if (nodeParams == nullptr)
    return CUDA_ERROR_INVALID_VALUE;
  if (numDependencies * sizeof(const CUgraphNode) != 0 &&
      dependencies == nullptr)
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  if (conn == nullptr ||
      rpc_write_start_request(conn,
                              RPC_cuGraphAddExternalSemaphoresSignalNode) < 0 ||
      rpc_write(conn, phGraphNode, sizeof(CUgraphNode)) < 0 ||
      rpc_write(conn, &hGraph, sizeof(CUgraph)) < 0 ||
      rpc_write(conn, &numDependencies, sizeof(size_t)) < 0 ||
      (numDependencies * sizeof(const CUgraphNode) != 0 &&
       rpc_write(conn, dependencies,
                 numDependencies * sizeof(const CUgraphNode)) < 0) ||
      rpc_write(conn, nodeParams, sizeof(*nodeParams)) < 0 ||
      (nodeParams->numExtSems != 0 &&
       rpc_write(conn, nodeParams->extSemArray,
                 nodeParams->numExtSems * sizeof(*nodeParams->extSemArray)) <
           0) ||
      (nodeParams->numExtSems != 0 &&
       rpc_write(conn, nodeParams->paramsArray,
                 nodeParams->numExtSems * sizeof(*nodeParams->paramsArray)) <
           0) ||
      rpc_wait_for_response(conn) < 0 ||
      rpc_read(conn, phGraphNode, sizeof(CUgraphNode)) < 0 ||
      rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
      rpc_read_end(conn) < 0)
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  if (return_value == CUDA_SUCCESS && phGraphNode != nullptr) {
    lupine_note_graph_node_owner_route(*phGraphNode, route);
  }
  return return_value;
}

CUresult cuGraphExternalSemaphoresSignalNodeGetParams(
    CUgraphNode hNode, CUDA_EXT_SEM_SIGNAL_NODE_PARAMS *params_out) {
  lupine_route route = lupine_route_for_graph_node(hNode);
  CUresult return_value;
  using real_fn_t =
      CUresult (*)(CUgraphNode, CUDA_EXT_SEM_SIGNAL_NODE_PARAMS *);
  if (lupine_call_local_cuda_if_routed<real_fn_t>(
          route, "cuGraphExternalSemaphoresSignalNodeGetParams", &return_value,
          hNode, params_out)) {
    return return_value;
  }
  conn_t *conn = lupine_route_remote_conn(route);
  if (params_out == nullptr)
    return CUDA_ERROR_INVALID_VALUE;
  if (conn == nullptr ||
      rpc_write_start_request(
          conn, RPC_cuGraphExternalSemaphoresSignalNodeGetParams) < 0 ||
      rpc_write(conn, &hNode, sizeof(CUgraphNode)) < 0 ||
      rpc_wait_for_response(conn) < 0 ||
      (lupine_deep_cache_reset((const void *)params_out), false) ||
      rpc_read(conn, params_out, sizeof(*params_out)) < 0 ||
      ((params_out->extSemArray =
            (params_out->numExtSems != 0
                 ? (decltype(params_out->extSemArray))lupine_deep_cache_add(
                       (const void *)params_out,
                       params_out->numExtSems *
                           sizeof(*params_out->extSemArray))
                 : nullptr)),
       false) ||
      (params_out->numExtSems != 0 && params_out->extSemArray == nullptr) ||
      (params_out->numExtSems != 0 &&
       rpc_read(conn, (void *)params_out->extSemArray,
                params_out->numExtSems * sizeof(*params_out->extSemArray)) <
           0) ||
      ((params_out->paramsArray =
            (params_out->numExtSems != 0
                 ? (decltype(params_out->paramsArray))lupine_deep_cache_add(
                       (const void *)params_out,
                       params_out->numExtSems *
                           sizeof(*params_out->paramsArray))
                 : nullptr)),
       false) ||
      (params_out->numExtSems != 0 && params_out->paramsArray == nullptr) ||
      (params_out->numExtSems != 0 &&
       rpc_read(conn, (void *)params_out->paramsArray,
                params_out->numExtSems * sizeof(*params_out->paramsArray)) <
           0) ||
      rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
      rpc_read_end(conn) < 0)
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  return return_value;
}

CUresult cuGraphExternalSemaphoresSignalNodeSetParams(
    CUgraphNode hNode, const CUDA_EXT_SEM_SIGNAL_NODE_PARAMS *nodeParams) {
  lupine_route route = lupine_route_for_graph_node(hNode);
  CUresult return_value;
  using real_fn_t =
      CUresult (*)(CUgraphNode, const CUDA_EXT_SEM_SIGNAL_NODE_PARAMS *);
  if (lupine_call_local_cuda_if_routed<real_fn_t>(
          route, "cuGraphExternalSemaphoresSignalNodeSetParams", &return_value,
          hNode, nodeParams)) {
    return return_value;
  }
  conn_t *conn = lupine_route_remote_conn(route);
  if (nodeParams == nullptr)
    return CUDA_ERROR_INVALID_VALUE;
  if (conn == nullptr ||
      rpc_write_start_request(
          conn, RPC_cuGraphExternalSemaphoresSignalNodeSetParams) < 0 ||
      rpc_write(conn, &hNode, sizeof(CUgraphNode)) < 0 ||
      rpc_write(conn, nodeParams, sizeof(*nodeParams)) < 0 ||
      (nodeParams->numExtSems != 0 &&
       rpc_write(conn, nodeParams->extSemArray,
                 nodeParams->numExtSems * sizeof(*nodeParams->extSemArray)) <
           0) ||
      (nodeParams->numExtSems != 0 &&
       rpc_write(conn, nodeParams->paramsArray,
                 nodeParams->numExtSems * sizeof(*nodeParams->paramsArray)) <
           0) ||
      rpc_wait_for_response(conn) < 0 ||
      rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
      rpc_read_end(conn) < 0)
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  return return_value;
}

CUresult cuGraphAddExternalSemaphoresWaitNode(
    CUgraphNode *phGraphNode, CUgraph hGraph, const CUgraphNode *dependencies,
    size_t numDependencies, const CUDA_EXT_SEM_WAIT_NODE_PARAMS *nodeParams) {
  lupine_route route = lupine_route_for_graph(hGraph);
  CUresult return_value;
  using real_fn_t = CUresult (*)(CUgraphNode *, CUgraph, const CUgraphNode *,
                                 size_t, const CUDA_EXT_SEM_WAIT_NODE_PARAMS *);
  if (lupine_call_local_cuda_if_routed<real_fn_t>(
          route, "cuGraphAddExternalSemaphoresWaitNode", &return_value,
          phGraphNode, hGraph, dependencies, numDependencies, nodeParams)) {
    if (return_value == CUDA_SUCCESS && phGraphNode != nullptr) {
      lupine_note_graph_node_owner_route(*phGraphNode, route);
    }
    return return_value;
  }
  conn_t *conn = lupine_route_remote_conn(route);
  if (nodeParams == nullptr)
    return CUDA_ERROR_INVALID_VALUE;
  if (numDependencies * sizeof(const CUgraphNode) != 0 &&
      dependencies == nullptr)
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  if (conn == nullptr ||
      rpc_write_start_request(conn, RPC_cuGraphAddExternalSemaphoresWaitNode) <
          0 ||
      rpc_write(conn, phGraphNode, sizeof(CUgraphNode)) < 0 ||
      rpc_write(conn, &hGraph, sizeof(CUgraph)) < 0 ||
      rpc_write(conn, &numDependencies, sizeof(size_t)) < 0 ||
      (numDependencies * sizeof(const CUgraphNode) != 0 &&
       rpc_write(conn, dependencies,
                 numDependencies * sizeof(const CUgraphNode)) < 0) ||
      rpc_write(conn, nodeParams, sizeof(*nodeParams)) < 0 ||
      (nodeParams->numExtSems != 0 &&
       rpc_write(conn, nodeParams->extSemArray,
                 nodeParams->numExtSems * sizeof(*nodeParams->extSemArray)) <
           0) ||
      (nodeParams->numExtSems != 0 &&
       rpc_write(conn, nodeParams->paramsArray,
                 nodeParams->numExtSems * sizeof(*nodeParams->paramsArray)) <
           0) ||
      rpc_wait_for_response(conn) < 0 ||
      rpc_read(conn, phGraphNode, sizeof(CUgraphNode)) < 0 ||
      rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
      rpc_read_end(conn) < 0)
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  if (return_value == CUDA_SUCCESS && phGraphNode != nullptr) {
    lupine_note_graph_node_owner_route(*phGraphNode, route);
  }
  return return_value;
}

CUresult cuGraphExternalSemaphoresWaitNodeGetParams(
    CUgraphNode hNode, CUDA_EXT_SEM_WAIT_NODE_PARAMS *params_out) {
  lupine_route route = lupine_route_for_graph_node(hNode);
  CUresult return_value;
  using real_fn_t = CUresult (*)(CUgraphNode, CUDA_EXT_SEM_WAIT_NODE_PARAMS *);
  if (lupine_call_local_cuda_if_routed<real_fn_t>(
          route, "cuGraphExternalSemaphoresWaitNodeGetParams", &return_value,
          hNode, params_out)) {
    return return_value;
  }
  conn_t *conn = lupine_route_remote_conn(route);
  if (params_out == nullptr)
    return CUDA_ERROR_INVALID_VALUE;
  if (conn == nullptr ||
      rpc_write_start_request(
          conn, RPC_cuGraphExternalSemaphoresWaitNodeGetParams) < 0 ||
      rpc_write(conn, &hNode, sizeof(CUgraphNode)) < 0 ||
      rpc_wait_for_response(conn) < 0 ||
      (lupine_deep_cache_reset((const void *)params_out), false) ||
      rpc_read(conn, params_out, sizeof(*params_out)) < 0 ||
      ((params_out->extSemArray =
            (params_out->numExtSems != 0
                 ? (decltype(params_out->extSemArray))lupine_deep_cache_add(
                       (const void *)params_out,
                       params_out->numExtSems *
                           sizeof(*params_out->extSemArray))
                 : nullptr)),
       false) ||
      (params_out->numExtSems != 0 && params_out->extSemArray == nullptr) ||
      (params_out->numExtSems != 0 &&
       rpc_read(conn, (void *)params_out->extSemArray,
                params_out->numExtSems * sizeof(*params_out->extSemArray)) <
           0) ||
      ((params_out->paramsArray =
            (params_out->numExtSems != 0
                 ? (decltype(params_out->paramsArray))lupine_deep_cache_add(
                       (const void *)params_out,
                       params_out->numExtSems *
                           sizeof(*params_out->paramsArray))
                 : nullptr)),
       false) ||
      (params_out->numExtSems != 0 && params_out->paramsArray == nullptr) ||
      (params_out->numExtSems != 0 &&
       rpc_read(conn, (void *)params_out->paramsArray,
                params_out->numExtSems * sizeof(*params_out->paramsArray)) <
           0) ||
      rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
      rpc_read_end(conn) < 0)
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  return return_value;
}

CUresult cuGraphExternalSemaphoresWaitNodeSetParams(
    CUgraphNode hNode, const CUDA_EXT_SEM_WAIT_NODE_PARAMS *nodeParams) {
  lupine_route route = lupine_route_for_graph_node(hNode);
  CUresult return_value;
  using real_fn_t =
      CUresult (*)(CUgraphNode, const CUDA_EXT_SEM_WAIT_NODE_PARAMS *);
  if (lupine_call_local_cuda_if_routed<real_fn_t>(
          route, "cuGraphExternalSemaphoresWaitNodeSetParams", &return_value,
          hNode, nodeParams)) {
    return return_value;
  }
  conn_t *conn = lupine_route_remote_conn(route);
  if (nodeParams == nullptr)
    return CUDA_ERROR_INVALID_VALUE;
  if (conn == nullptr ||
      rpc_write_start_request(
          conn, RPC_cuGraphExternalSemaphoresWaitNodeSetParams) < 0 ||
      rpc_write(conn, &hNode, sizeof(CUgraphNode)) < 0 ||
      rpc_write(conn, nodeParams, sizeof(*nodeParams)) < 0 ||
      (nodeParams->numExtSems != 0 &&
       rpc_write(conn, nodeParams->extSemArray,
                 nodeParams->numExtSems * sizeof(*nodeParams->extSemArray)) <
           0) ||
      (nodeParams->numExtSems != 0 &&
       rpc_write(conn, nodeParams->paramsArray,
                 nodeParams->numExtSems * sizeof(*nodeParams->paramsArray)) <
           0) ||
      rpc_wait_for_response(conn) < 0 ||
      rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
      rpc_read_end(conn) < 0)
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  return return_value;
}

CUresult cuGraphAddBatchMemOpNode(
    CUgraphNode *phGraphNode, CUgraph hGraph, const CUgraphNode *dependencies,
    size_t numDependencies, const CUDA_BATCH_MEM_OP_NODE_PARAMS *nodeParams) {
  lupine_route route = lupine_route_for_graph(hGraph);
  CUresult return_value;
  using real_fn_t = CUresult (*)(CUgraphNode *, CUgraph, const CUgraphNode *,
                                 size_t, const CUDA_BATCH_MEM_OP_NODE_PARAMS *);
  if (lupine_call_local_cuda_if_routed<real_fn_t>(
          route, "cuGraphAddBatchMemOpNode", &return_value, phGraphNode, hGraph,
          dependencies, numDependencies, nodeParams)) {
    if (return_value == CUDA_SUCCESS && phGraphNode != nullptr) {
      lupine_note_graph_node_owner_route(*phGraphNode, route);
    }
    return return_value;
  }
  conn_t *conn = lupine_route_remote_conn(route);
  if (nodeParams == nullptr)
    return CUDA_ERROR_INVALID_VALUE;
  if (numDependencies * sizeof(const CUgraphNode) != 0 &&
      dependencies == nullptr)
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  if (conn == nullptr ||
      rpc_write_start_request(conn, RPC_cuGraphAddBatchMemOpNode) < 0 ||
      rpc_write(conn, phGraphNode, sizeof(CUgraphNode)) < 0 ||
      rpc_write(conn, &hGraph, sizeof(CUgraph)) < 0 ||
      rpc_write(conn, &numDependencies, sizeof(size_t)) < 0 ||
      (numDependencies * sizeof(const CUgraphNode) != 0 &&
       rpc_write(conn, dependencies,
                 numDependencies * sizeof(const CUgraphNode)) < 0) ||
      rpc_write(conn, nodeParams, sizeof(*nodeParams)) < 0 ||
      (nodeParams->count != 0 &&
       rpc_write(conn, nodeParams->paramArray,
                 nodeParams->count * sizeof(*nodeParams->paramArray)) < 0) ||
      rpc_wait_for_response(conn) < 0 ||
      rpc_read(conn, phGraphNode, sizeof(CUgraphNode)) < 0 ||
      rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
      rpc_read_end(conn) < 0)
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  if (return_value == CUDA_SUCCESS && phGraphNode != nullptr) {
    lupine_note_graph_node_owner_route(*phGraphNode, route);
  }
  return return_value;
}

CUresult
cuGraphBatchMemOpNodeGetParams(CUgraphNode hNode,
                               CUDA_BATCH_MEM_OP_NODE_PARAMS *nodeParams_out) {
  lupine_route route = lupine_route_for_graph_node(hNode);
  CUresult return_value;
  using real_fn_t = CUresult (*)(CUgraphNode, CUDA_BATCH_MEM_OP_NODE_PARAMS *);
  if (lupine_call_local_cuda_if_routed<real_fn_t>(
          route, "cuGraphBatchMemOpNodeGetParams", &return_value, hNode,
          nodeParams_out)) {
    return return_value;
  }
  conn_t *conn = lupine_route_remote_conn(route);
  if (nodeParams_out == nullptr)
    return CUDA_ERROR_INVALID_VALUE;
  if (conn == nullptr ||
      rpc_write_start_request(conn, RPC_cuGraphBatchMemOpNodeGetParams) < 0 ||
      rpc_write(conn, &hNode, sizeof(CUgraphNode)) < 0 ||
      rpc_wait_for_response(conn) < 0 ||
      (lupine_deep_cache_reset((const void *)nodeParams_out), false) ||
      rpc_read(conn, nodeParams_out, sizeof(*nodeParams_out)) < 0 ||
      ((nodeParams_out->paramArray =
            (nodeParams_out->count != 0
                 ? (decltype(nodeParams_out->paramArray))lupine_deep_cache_add(
                       (const void *)nodeParams_out,
                       nodeParams_out->count *
                           sizeof(*nodeParams_out->paramArray))
                 : nullptr)),
       false) ||
      (nodeParams_out->count != 0 && nodeParams_out->paramArray == nullptr) ||
      (nodeParams_out->count != 0 &&
       rpc_read(conn, (void *)nodeParams_out->paramArray,
                nodeParams_out->count * sizeof(*nodeParams_out->paramArray)) <
           0) ||
      rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
      rpc_read_end(conn) < 0)
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  return return_value;
}

CUresult cuGraphBatchMemOpNodeSetParams(
    CUgraphNode hNode, const CUDA_BATCH_MEM_OP_NODE_PARAMS *nodeParams) {
  lupine_route route = lupine_route_for_graph_node(hNode);
  CUresult return_value;
  using real_fn_t =
      CUresult (*)(CUgraphNode, const CUDA_BATCH_MEM_OP_NODE_PARAMS *);
  if (lupine_call_local_cuda_if_routed<real_fn_t>(
          route, "cuGraphBatchMemOpNodeSetParams", &return_value, hNode,
          nodeParams)) {
    return return_value;
  }
  conn_t *conn = lupine_route_remote_conn(route);
  if (nodeParams == nullptr)
    return CUDA_ERROR_INVALID_VALUE;
  if (conn == nullptr ||
      rpc_write_start_request(conn, RPC_cuGraphBatchMemOpNodeSetParams) < 0 ||
      rpc_write(conn, &hNode, sizeof(CUgraphNode)) < 0 ||
      rpc_write(conn, nodeParams, sizeof(*nodeParams)) < 0 ||
      (nodeParams->count != 0 &&
       rpc_write(conn, nodeParams->paramArray,
                 nodeParams->count * sizeof(*nodeParams->paramArray)) < 0) ||
      rpc_wait_for_response(conn) < 0 ||
      rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
      rpc_read_end(conn) < 0)
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  return return_value;
}

CUresult cuGraphExecBatchMemOpNodeSetParams(
    CUgraphExec hGraphExec, CUgraphNode hNode,
    const CUDA_BATCH_MEM_OP_NODE_PARAMS *nodeParams) {
  lupine_route route = lupine_route_for_graph_exec(hGraphExec);
  CUresult return_value;
  using real_fn_t = CUresult (*)(CUgraphExec, CUgraphNode,
                                 const CUDA_BATCH_MEM_OP_NODE_PARAMS *);
  if (lupine_call_local_cuda_if_routed<real_fn_t>(
          route, "cuGraphExecBatchMemOpNodeSetParams", &return_value,
          hGraphExec, hNode, nodeParams)) {
    return return_value;
  }
  conn_t *conn = lupine_route_remote_conn(route);
  if (nodeParams == nullptr)
    return CUDA_ERROR_INVALID_VALUE;
  if (conn == nullptr ||
      rpc_write_start_request(conn, RPC_cuGraphExecBatchMemOpNodeSetParams) <
          0 ||
      rpc_write(conn, &hGraphExec, sizeof(CUgraphExec)) < 0 ||
      rpc_write(conn, &hNode, sizeof(CUgraphNode)) < 0 ||
      rpc_write(conn, nodeParams, sizeof(*nodeParams)) < 0 ||
      (nodeParams->count != 0 &&
       rpc_write(conn, nodeParams->paramArray,
                 nodeParams->count * sizeof(*nodeParams->paramArray)) < 0) ||
      rpc_wait_for_response(conn) < 0 ||
      rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
      rpc_read_end(conn) < 0)
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  return return_value;
}

CUresult cuGraphAddMemAllocNode(CUgraphNode *phGraphNode, CUgraph hGraph,
                                const CUgraphNode *dependencies,
                                size_t numDependencies,
                                CUDA_MEM_ALLOC_NODE_PARAMS *nodeParams) {
  lupine_route route = lupine_route_for_graph(hGraph);
  CUresult return_value;
  using real_fn_t = CUresult (*)(CUgraphNode *, CUgraph, const CUgraphNode *,
                                 size_t, CUDA_MEM_ALLOC_NODE_PARAMS *);
  if (lupine_call_local_cuda_if_routed<real_fn_t>(
          route, "cuGraphAddMemAllocNode", &return_value, phGraphNode, hGraph,
          dependencies, numDependencies, nodeParams)) {
    if (return_value == CUDA_SUCCESS && phGraphNode != nullptr) {
      lupine_note_graph_node_owner_route(*phGraphNode, route);
    }
    return return_value;
  }
  conn_t *conn = lupine_route_remote_conn(route);
  if (numDependencies * sizeof(const CUgraphNode) != 0 &&
      dependencies == nullptr)
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  if (conn == nullptr ||
      rpc_write_start_request(conn, RPC_cuGraphAddMemAllocNode) < 0 ||
      rpc_write(conn, phGraphNode, sizeof(CUgraphNode)) < 0 ||
      rpc_write(conn, &hGraph, sizeof(CUgraph)) < 0 ||
      rpc_write(conn, &numDependencies, sizeof(size_t)) < 0 ||
      (numDependencies * sizeof(const CUgraphNode) != 0 &&
       rpc_write(conn, dependencies,
                 numDependencies * sizeof(const CUgraphNode)) < 0) ||
      rpc_write(conn, nodeParams, sizeof(CUDA_MEM_ALLOC_NODE_PARAMS)) < 0 ||
      rpc_wait_for_response(conn) < 0 ||
      rpc_read(conn, phGraphNode, sizeof(CUgraphNode)) < 0 ||
      rpc_read(conn, nodeParams, sizeof(CUDA_MEM_ALLOC_NODE_PARAMS)) < 0 ||
      rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
      rpc_read_end(conn) < 0)
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  if (return_value == CUDA_SUCCESS && phGraphNode != nullptr) {
    lupine_note_graph_node_owner_route(*phGraphNode, route);
  }
  return return_value;
}

CUresult cuGraphMemAllocNodeGetParams(CUgraphNode hNode,
                                      CUDA_MEM_ALLOC_NODE_PARAMS *params_out) {
  lupine_route route = lupine_route_for_graph_node(hNode);
  CUresult return_value;
  using real_fn_t = CUresult (*)(CUgraphNode, CUDA_MEM_ALLOC_NODE_PARAMS *);
  if (lupine_call_local_cuda_if_routed<real_fn_t>(
          route, "cuGraphMemAllocNodeGetParams", &return_value, hNode,
          params_out)) {
    return return_value;
  }
  conn_t *conn = lupine_route_remote_conn(route);
  if (conn == nullptr ||
      rpc_write_start_request(conn, RPC_cuGraphMemAllocNodeGetParams) < 0 ||
      rpc_write(conn, &hNode, sizeof(CUgraphNode)) < 0 ||
      rpc_write(conn, params_out, sizeof(CUDA_MEM_ALLOC_NODE_PARAMS)) < 0 ||
      rpc_wait_for_response(conn) < 0 ||
      rpc_read(conn, params_out, sizeof(CUDA_MEM_ALLOC_NODE_PARAMS)) < 0 ||
      rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
      rpc_read_end(conn) < 0)
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  return return_value;
}

CUresult cuGraphAddMemFreeNode(CUgraphNode *phGraphNode, CUgraph hGraph,
                               const CUgraphNode *dependencies,
                               size_t numDependencies, CUdeviceptr dptr) {
  lupine_route route = lupine_route_for_graph(hGraph);
  CUresult return_value;
  using real_fn_t = CUresult (*)(CUgraphNode *, CUgraph, const CUgraphNode *,
                                 size_t, CUdeviceptr);
  if (lupine_call_local_cuda_if_routed<real_fn_t>(
          route, "cuGraphAddMemFreeNode", &return_value, phGraphNode, hGraph,
          dependencies, numDependencies, dptr)) {
    if (return_value == CUDA_SUCCESS && phGraphNode != nullptr) {
      lupine_note_graph_node_owner_route(*phGraphNode, route);
    }
    return return_value;
  }
  conn_t *conn = lupine_route_remote_conn(route);
  if (numDependencies * sizeof(const CUgraphNode) != 0 &&
      dependencies == nullptr)
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  if (conn == nullptr ||
      rpc_write_start_request(conn, RPC_cuGraphAddMemFreeNode) < 0 ||
      rpc_write(conn, phGraphNode, sizeof(CUgraphNode)) < 0 ||
      rpc_write(conn, &hGraph, sizeof(CUgraph)) < 0 ||
      rpc_write(conn, &numDependencies, sizeof(size_t)) < 0 ||
      (numDependencies * sizeof(const CUgraphNode) != 0 &&
       rpc_write(conn, dependencies,
                 numDependencies * sizeof(const CUgraphNode)) < 0) ||
      rpc_write(conn, &dptr, sizeof(CUdeviceptr)) < 0 ||
      rpc_wait_for_response(conn) < 0 ||
      rpc_read(conn, phGraphNode, sizeof(CUgraphNode)) < 0 ||
      rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
      rpc_read_end(conn) < 0)
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  if (return_value == CUDA_SUCCESS && phGraphNode != nullptr) {
    lupine_note_graph_node_owner_route(*phGraphNode, route);
  }
  return return_value;
}

CUresult cuGraphMemFreeNodeGetParams(CUgraphNode hNode, CUdeviceptr *dptr_out) {
  lupine_route route = lupine_route_for_graph_node(hNode);
  CUresult return_value;
  using real_fn_t = CUresult (*)(CUgraphNode, CUdeviceptr *);
  if (lupine_call_local_cuda_if_routed<real_fn_t>(
          route, "cuGraphMemFreeNodeGetParams", &return_value, hNode,
          dptr_out)) {
    return return_value;
  }
  conn_t *conn = lupine_route_remote_conn(route);
  if (conn == nullptr ||
      rpc_write_start_request(conn, RPC_cuGraphMemFreeNodeGetParams) < 0 ||
      rpc_write(conn, &hNode, sizeof(CUgraphNode)) < 0 ||
      rpc_write(conn, dptr_out, sizeof(CUdeviceptr)) < 0 ||
      rpc_wait_for_response(conn) < 0 ||
      rpc_read(conn, dptr_out, sizeof(CUdeviceptr)) < 0 ||
      rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
      rpc_read_end(conn) < 0)
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  return return_value;
}

CUresult cuDeviceGraphMemTrim(CUdevice device) {
  lupine_route route = lupine_route_for_device(&device);
  CUresult return_value;
  using real_fn_t = CUresult (*)(CUdevice);
  if (lupine_call_local_cuda_if_routed<real_fn_t>(route, "cuDeviceGraphMemTrim",
                                                  &return_value, device)) {
    return return_value;
  }
  conn_t *conn = lupine_route_remote_conn(route);
  if (conn == nullptr ||
      rpc_write_start_request(conn, RPC_cuDeviceGraphMemTrim) < 0 ||
      rpc_write(conn, &device, sizeof(CUdevice)) < 0 ||
      rpc_wait_for_response(conn) < 0 ||
      rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
      rpc_read_end(conn) < 0)
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  return return_value;
}

CUresult cuGraphClone(CUgraph *phGraphClone, CUgraph originalGraph) {
  lupine_route route = lupine_route_for_graph(originalGraph);
  CUresult return_value;
  using real_fn_t = CUresult (*)(CUgraph *, CUgraph);
  if (lupine_call_local_cuda_if_routed<real_fn_t>(
          route, "cuGraphClone", &return_value, phGraphClone, originalGraph)) {
    if (return_value == CUDA_SUCCESS && phGraphClone != nullptr) {
      lupine_note_graph_owner_route(*phGraphClone, route);
    }
    return return_value;
  }
  conn_t *conn = lupine_route_remote_conn(route);
  if (conn == nullptr || rpc_write_start_request(conn, RPC_cuGraphClone) < 0 ||
      rpc_write(conn, phGraphClone, sizeof(CUgraph)) < 0 ||
      rpc_write(conn, &originalGraph, sizeof(CUgraph)) < 0 ||
      rpc_wait_for_response(conn) < 0 ||
      rpc_read(conn, phGraphClone, sizeof(CUgraph)) < 0 ||
      rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
      rpc_read_end(conn) < 0)
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  if (return_value == CUDA_SUCCESS && phGraphClone != nullptr) {
    lupine_note_graph_owner_route(*phGraphClone, route);
  }
  return return_value;
}

CUresult cuGraphNodeFindInClone(CUgraphNode *phNode, CUgraphNode hOriginalNode,
                                CUgraph hClonedGraph) {
  lupine_route route = lupine_route_for_graph_node(hOriginalNode);
  CUresult return_value;
  using real_fn_t = CUresult (*)(CUgraphNode *, CUgraphNode, CUgraph);
  if (lupine_call_local_cuda_if_routed<real_fn_t>(
          route, "cuGraphNodeFindInClone", &return_value, phNode, hOriginalNode,
          hClonedGraph)) {
    if (return_value == CUDA_SUCCESS && phNode != nullptr) {
      lupine_note_graph_node_owner_route(*phNode, route);
    }
    return return_value;
  }
  conn_t *conn = lupine_route_remote_conn(route);
  if (conn == nullptr ||
      rpc_write_start_request(conn, RPC_cuGraphNodeFindInClone) < 0 ||
      rpc_write(conn, phNode, sizeof(CUgraphNode)) < 0 ||
      rpc_write(conn, &hOriginalNode, sizeof(CUgraphNode)) < 0 ||
      rpc_write(conn, &hClonedGraph, sizeof(CUgraph)) < 0 ||
      rpc_wait_for_response(conn) < 0 ||
      rpc_read(conn, phNode, sizeof(CUgraphNode)) < 0 ||
      rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
      rpc_read_end(conn) < 0)
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  if (return_value == CUDA_SUCCESS && phNode != nullptr) {
    lupine_note_graph_node_owner_route(*phNode, route);
  }
  return return_value;
}

CUresult cuGraphNodeGetType(CUgraphNode hNode, CUgraphNodeType *type) {
  lupine_route route = lupine_route_for_graph_node(hNode);
  CUresult return_value;
  using real_fn_t = CUresult (*)(CUgraphNode, CUgraphNodeType *);
  if (lupine_call_local_cuda_if_routed<real_fn_t>(route, "cuGraphNodeGetType",
                                                  &return_value, hNode, type)) {
    return return_value;
  }
  conn_t *conn = lupine_route_remote_conn(route);
  if (conn == nullptr ||
      rpc_write_start_request(conn, RPC_cuGraphNodeGetType) < 0 ||
      rpc_write(conn, &hNode, sizeof(CUgraphNode)) < 0 ||
      rpc_write(conn, type, sizeof(CUgraphNodeType)) < 0 ||
      rpc_wait_for_response(conn) < 0 ||
      rpc_read(conn, type, sizeof(CUgraphNodeType)) < 0 ||
      rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
      rpc_read_end(conn) < 0)
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  return return_value;
}

CUresult cuGraphGetNodes(CUgraph hGraph, CUgraphNode *nodes, size_t *numNodes) {
  lupine_route route = lupine_route_for_graph(hGraph);
  CUresult return_value;
  using real_fn_t = CUresult (*)(CUgraph, CUgraphNode *, size_t *);
  if (lupine_call_local_cuda_if_routed<real_fn_t>(
          route, "cuGraphGetNodes", &return_value, hGraph, nodes, numNodes)) {
    return return_value;
  }
  conn_t *conn = lupine_route_remote_conn(route);
  size_t numNodes_requested = (nodes != nullptr) ? *numNodes : 0;
  uint8_t nodes_present = nodes != nullptr ? 1 : 0;
  if (conn == nullptr ||
      rpc_write_start_request(conn, RPC_cuGraphGetNodes) < 0 ||
      rpc_write(conn, &hGraph, sizeof(CUgraph)) < 0 ||
      rpc_write(conn, &numNodes_requested, sizeof(size_t)) < 0 ||
      rpc_write(conn, &nodes_present, sizeof(uint8_t)) < 0 ||
      rpc_wait_for_response(conn) < 0 ||
      rpc_read(conn, numNodes, sizeof(size_t)) < 0 ||
      (nodes != nullptr && *numNodes != 0 &&
       rpc_read(conn, nodes, *numNodes * sizeof(CUgraphNode)) < 0) ||
      rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
      rpc_read_end(conn) < 0)
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  return return_value;
}

CUresult cuGraphGetRootNodes(CUgraph hGraph, CUgraphNode *rootNodes,
                             size_t *numRootNodes) {
  lupine_route route = lupine_route_for_graph(hGraph);
  CUresult return_value;
  using real_fn_t = CUresult (*)(CUgraph, CUgraphNode *, size_t *);
  if (lupine_call_local_cuda_if_routed<real_fn_t>(route, "cuGraphGetRootNodes",
                                                  &return_value, hGraph,
                                                  rootNodes, numRootNodes)) {
    return return_value;
  }
  conn_t *conn = lupine_route_remote_conn(route);
  size_t numRootNodes_requested = (rootNodes != nullptr) ? *numRootNodes : 0;
  uint8_t rootNodes_present = rootNodes != nullptr ? 1 : 0;
  if (conn == nullptr ||
      rpc_write_start_request(conn, RPC_cuGraphGetRootNodes) < 0 ||
      rpc_write(conn, &hGraph, sizeof(CUgraph)) < 0 ||
      rpc_write(conn, &numRootNodes_requested, sizeof(size_t)) < 0 ||
      rpc_write(conn, &rootNodes_present, sizeof(uint8_t)) < 0 ||
      rpc_wait_for_response(conn) < 0 ||
      rpc_read(conn, numRootNodes, sizeof(size_t)) < 0 ||
      (rootNodes != nullptr && *numRootNodes != 0 &&
       rpc_read(conn, rootNodes, *numRootNodes * sizeof(CUgraphNode)) < 0) ||
      rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
      rpc_read_end(conn) < 0)
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  return return_value;
}

CUresult cuGraphDestroyNode(CUgraphNode hNode) {
  lupine_route route = lupine_route_for_graph_node(hNode);
  CUresult return_value;
  using real_fn_t = CUresult (*)(CUgraphNode);
  if (lupine_call_local_cuda_if_routed<real_fn_t>(route, "cuGraphDestroyNode",
                                                  &return_value, hNode)) {
    return return_value;
  }
  conn_t *conn = lupine_route_remote_conn(route);
  if (conn == nullptr ||
      rpc_write_start_request(conn, RPC_cuGraphDestroyNode) < 0 ||
      rpc_write(conn, &hNode, sizeof(CUgraphNode)) < 0 ||
      rpc_wait_for_response(conn) < 0 ||
      rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
      rpc_read_end(conn) < 0)
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  return return_value;
}

CUresult cuGraphInstantiateWithFlags(CUgraphExec *phGraphExec, CUgraph hGraph,
                                     unsigned long long flags) {
  lupine_route route = lupine_route_for_graph(hGraph);
  CUresult return_value;
  using real_fn_t = CUresult (*)(CUgraphExec *, CUgraph, unsigned long long);
  if (lupine_call_local_cuda_if_routed<real_fn_t>(
          route, "cuGraphInstantiateWithFlags", &return_value, phGraphExec,
          hGraph, flags)) {
    if (return_value == CUDA_SUCCESS && phGraphExec != nullptr) {
      lupine_note_graph_exec_owner_route(*phGraphExec, route);
    }
    return return_value;
  }
  conn_t *conn = lupine_route_remote_conn(route);
  if (conn == nullptr ||
      rpc_write_start_request(conn, RPC_cuGraphInstantiateWithFlags) < 0 ||
      rpc_write(conn, phGraphExec, sizeof(CUgraphExec)) < 0 ||
      rpc_write(conn, &hGraph, sizeof(CUgraph)) < 0 ||
      rpc_write(conn, &flags, sizeof(unsigned long long)) < 0 ||
      rpc_wait_for_response(conn) < 0 ||
      rpc_read(conn, phGraphExec, sizeof(CUgraphExec)) < 0 ||
      rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
      rpc_read_end(conn) < 0)
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  if (return_value == CUDA_SUCCESS && phGraphExec != nullptr) {
    lupine_note_graph_exec_owner_route(*phGraphExec, route);
  }
  return return_value;
}

CUresult
cuGraphInstantiateWithParams(CUgraphExec *phGraphExec, CUgraph hGraph,
                             CUDA_GRAPH_INSTANTIATE_PARAMS *instantiateParams) {
  lupine_route route = lupine_route_for_graph(hGraph);
  CUresult return_value;
  using real_fn_t =
      CUresult (*)(CUgraphExec *, CUgraph, CUDA_GRAPH_INSTANTIATE_PARAMS *);
  if (lupine_call_local_cuda_if_routed<real_fn_t>(
          route, "cuGraphInstantiateWithParams", &return_value, phGraphExec,
          hGraph, instantiateParams)) {
    if (return_value == CUDA_SUCCESS && phGraphExec != nullptr) {
      lupine_note_graph_exec_owner_route(*phGraphExec, route);
    }
    return return_value;
  }
  conn_t *conn = lupine_route_remote_conn(route);
  if (conn == nullptr ||
      rpc_write_start_request(conn, RPC_cuGraphInstantiateWithParams) < 0 ||
      rpc_write(conn, phGraphExec, sizeof(CUgraphExec)) < 0 ||
      rpc_write(conn, &hGraph, sizeof(CUgraph)) < 0 ||
      rpc_write(conn, instantiateParams,
                sizeof(CUDA_GRAPH_INSTANTIATE_PARAMS)) < 0 ||
      rpc_wait_for_response(conn) < 0 ||
      rpc_read(conn, phGraphExec, sizeof(CUgraphExec)) < 0 ||
      rpc_read(conn, instantiateParams, sizeof(CUDA_GRAPH_INSTANTIATE_PARAMS)) <
          0 ||
      rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
      rpc_read_end(conn) < 0)
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  if (return_value == CUDA_SUCCESS && phGraphExec != nullptr) {
    lupine_note_graph_exec_owner_route(*phGraphExec, route);
  }
  return return_value;
}

CUresult cuGraphExecGetFlags(CUgraphExec hGraphExec, cuuint64_t *flags) {
  lupine_route route = lupine_route_for_graph_exec(hGraphExec);
  CUresult return_value;
  using real_fn_t = CUresult (*)(CUgraphExec, cuuint64_t *);
  if (lupine_call_local_cuda_if_routed<real_fn_t>(
          route, "cuGraphExecGetFlags", &return_value, hGraphExec, flags)) {
    return return_value;
  }
  conn_t *conn = lupine_route_remote_conn(route);
  if (conn == nullptr ||
      rpc_write_start_request(conn, RPC_cuGraphExecGetFlags) < 0 ||
      rpc_write(conn, &hGraphExec, sizeof(CUgraphExec)) < 0 ||
      rpc_write(conn, flags, sizeof(cuuint64_t)) < 0 ||
      rpc_wait_for_response(conn) < 0 ||
      rpc_read(conn, flags, sizeof(cuuint64_t)) < 0 ||
      rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
      rpc_read_end(conn) < 0)
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  return return_value;
}

CUresult cuGraphExecMemcpyNodeSetParams(CUgraphExec hGraphExec,
                                        CUgraphNode hNode,
                                        const CUDA_MEMCPY3D *copyParams,
                                        CUcontext ctx) {
  lupine_route route = lupine_route_for_graph_exec(hGraphExec);
  CUresult return_value;
  using real_fn_t =
      CUresult (*)(CUgraphExec, CUgraphNode, const CUDA_MEMCPY3D *, CUcontext);
  if (lupine_call_local_cuda_if_routed<real_fn_t>(
          route, "cuGraphExecMemcpyNodeSetParams", &return_value, hGraphExec,
          hNode, copyParams, ctx)) {
    return return_value;
  }
  conn_t *conn = lupine_route_remote_conn(route);
  if (conn == nullptr ||
      rpc_write_start_request(conn, RPC_cuGraphExecMemcpyNodeSetParams) < 0 ||
      rpc_write(conn, &hGraphExec, sizeof(CUgraphExec)) < 0 ||
      rpc_write(conn, &hNode, sizeof(CUgraphNode)) < 0 ||
      rpc_write(conn, copyParams, sizeof(const CUDA_MEMCPY3D)) < 0 ||
      rpc_write(conn, &ctx, sizeof(CUcontext)) < 0 ||
      rpc_wait_for_response(conn) < 0 ||
      rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
      rpc_read_end(conn) < 0)
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  return return_value;
}

CUresult
cuGraphExecMemsetNodeSetParams(CUgraphExec hGraphExec, CUgraphNode hNode,
                               const CUDA_MEMSET_NODE_PARAMS *memsetParams,
                               CUcontext ctx) {
  lupine_route route = lupine_route_for_graph_exec(hGraphExec);
  CUresult return_value;
  using real_fn_t = CUresult (*)(CUgraphExec, CUgraphNode,
                                 const CUDA_MEMSET_NODE_PARAMS *, CUcontext);
  if (lupine_call_local_cuda_if_routed<real_fn_t>(
          route, "cuGraphExecMemsetNodeSetParams", &return_value, hGraphExec,
          hNode, memsetParams, ctx)) {
    return return_value;
  }
  conn_t *conn = lupine_route_remote_conn(route);
  if (conn == nullptr ||
      rpc_write_start_request(conn, RPC_cuGraphExecMemsetNodeSetParams) < 0 ||
      rpc_write(conn, &hGraphExec, sizeof(CUgraphExec)) < 0 ||
      rpc_write(conn, &hNode, sizeof(CUgraphNode)) < 0 ||
      rpc_write(conn, memsetParams, sizeof(const CUDA_MEMSET_NODE_PARAMS)) <
          0 ||
      rpc_write(conn, &ctx, sizeof(CUcontext)) < 0 ||
      rpc_wait_for_response(conn) < 0 ||
      rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
      rpc_read_end(conn) < 0)
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  return return_value;
}

CUresult cuGraphExecChildGraphNodeSetParams(CUgraphExec hGraphExec,
                                            CUgraphNode hNode,
                                            CUgraph childGraph) {
  lupine_route route = lupine_route_for_graph_exec(hGraphExec);
  CUresult return_value;
  using real_fn_t = CUresult (*)(CUgraphExec, CUgraphNode, CUgraph);
  if (lupine_call_local_cuda_if_routed<real_fn_t>(
          route, "cuGraphExecChildGraphNodeSetParams", &return_value,
          hGraphExec, hNode, childGraph)) {
    return return_value;
  }
  conn_t *conn = lupine_route_remote_conn(route);
  if (conn == nullptr ||
      rpc_write_start_request(conn, RPC_cuGraphExecChildGraphNodeSetParams) <
          0 ||
      rpc_write(conn, &hGraphExec, sizeof(CUgraphExec)) < 0 ||
      rpc_write(conn, &hNode, sizeof(CUgraphNode)) < 0 ||
      rpc_write(conn, &childGraph, sizeof(CUgraph)) < 0 ||
      rpc_wait_for_response(conn) < 0 ||
      rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
      rpc_read_end(conn) < 0)
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  return return_value;
}

CUresult cuGraphExecEventRecordNodeSetEvent(CUgraphExec hGraphExec,
                                            CUgraphNode hNode, CUevent event) {
  lupine_route route = lupine_route_for_graph_exec(hGraphExec);
  CUresult return_value;
  using real_fn_t = CUresult (*)(CUgraphExec, CUgraphNode, CUevent);
  if (lupine_call_local_cuda_if_routed<real_fn_t>(
          route, "cuGraphExecEventRecordNodeSetEvent", &return_value,
          hGraphExec, hNode, event)) {
    return return_value;
  }
  conn_t *conn = lupine_route_remote_conn(route);
  if (conn == nullptr ||
      rpc_write_start_request(conn, RPC_cuGraphExecEventRecordNodeSetEvent) <
          0 ||
      rpc_write(conn, &hGraphExec, sizeof(CUgraphExec)) < 0 ||
      rpc_write(conn, &hNode, sizeof(CUgraphNode)) < 0 ||
      rpc_write(conn, &event, sizeof(CUevent)) < 0 ||
      rpc_wait_for_response(conn) < 0 ||
      rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
      rpc_read_end(conn) < 0)
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  return return_value;
}

CUresult cuGraphExecEventWaitNodeSetEvent(CUgraphExec hGraphExec,
                                          CUgraphNode hNode, CUevent event) {
  lupine_route route = lupine_route_for_graph_exec(hGraphExec);
  CUresult return_value;
  using real_fn_t = CUresult (*)(CUgraphExec, CUgraphNode, CUevent);
  if (lupine_call_local_cuda_if_routed<real_fn_t>(
          route, "cuGraphExecEventWaitNodeSetEvent", &return_value, hGraphExec,
          hNode, event)) {
    return return_value;
  }
  conn_t *conn = lupine_route_remote_conn(route);
  if (conn == nullptr ||
      rpc_write_start_request(conn, RPC_cuGraphExecEventWaitNodeSetEvent) < 0 ||
      rpc_write(conn, &hGraphExec, sizeof(CUgraphExec)) < 0 ||
      rpc_write(conn, &hNode, sizeof(CUgraphNode)) < 0 ||
      rpc_write(conn, &event, sizeof(CUevent)) < 0 ||
      rpc_wait_for_response(conn) < 0 ||
      rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
      rpc_read_end(conn) < 0)
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  return return_value;
}

CUresult cuGraphExecExternalSemaphoresSignalNodeSetParams(
    CUgraphExec hGraphExec, CUgraphNode hNode,
    const CUDA_EXT_SEM_SIGNAL_NODE_PARAMS *nodeParams) {
  lupine_route route = lupine_route_for_graph_exec(hGraphExec);
  CUresult return_value;
  using real_fn_t = CUresult (*)(CUgraphExec, CUgraphNode,
                                 const CUDA_EXT_SEM_SIGNAL_NODE_PARAMS *);
  if (lupine_call_local_cuda_if_routed<real_fn_t>(
          route, "cuGraphExecExternalSemaphoresSignalNodeSetParams",
          &return_value, hGraphExec, hNode, nodeParams)) {
    return return_value;
  }
  conn_t *conn = lupine_route_remote_conn(route);
  if (nodeParams == nullptr)
    return CUDA_ERROR_INVALID_VALUE;
  if (conn == nullptr ||
      rpc_write_start_request(
          conn, RPC_cuGraphExecExternalSemaphoresSignalNodeSetParams) < 0 ||
      rpc_write(conn, &hGraphExec, sizeof(CUgraphExec)) < 0 ||
      rpc_write(conn, &hNode, sizeof(CUgraphNode)) < 0 ||
      rpc_write(conn, nodeParams, sizeof(*nodeParams)) < 0 ||
      (nodeParams->numExtSems != 0 &&
       rpc_write(conn, nodeParams->extSemArray,
                 nodeParams->numExtSems * sizeof(*nodeParams->extSemArray)) <
           0) ||
      (nodeParams->numExtSems != 0 &&
       rpc_write(conn, nodeParams->paramsArray,
                 nodeParams->numExtSems * sizeof(*nodeParams->paramsArray)) <
           0) ||
      rpc_wait_for_response(conn) < 0 ||
      rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
      rpc_read_end(conn) < 0)
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  return return_value;
}

CUresult cuGraphExecExternalSemaphoresWaitNodeSetParams(
    CUgraphExec hGraphExec, CUgraphNode hNode,
    const CUDA_EXT_SEM_WAIT_NODE_PARAMS *nodeParams) {
  lupine_route route = lupine_route_for_graph_exec(hGraphExec);
  CUresult return_value;
  using real_fn_t = CUresult (*)(CUgraphExec, CUgraphNode,
                                 const CUDA_EXT_SEM_WAIT_NODE_PARAMS *);
  if (lupine_call_local_cuda_if_routed<real_fn_t>(
          route, "cuGraphExecExternalSemaphoresWaitNodeSetParams",
          &return_value, hGraphExec, hNode, nodeParams)) {
    return return_value;
  }
  conn_t *conn = lupine_route_remote_conn(route);
  if (nodeParams == nullptr)
    return CUDA_ERROR_INVALID_VALUE;
  if (conn == nullptr ||
      rpc_write_start_request(
          conn, RPC_cuGraphExecExternalSemaphoresWaitNodeSetParams) < 0 ||
      rpc_write(conn, &hGraphExec, sizeof(CUgraphExec)) < 0 ||
      rpc_write(conn, &hNode, sizeof(CUgraphNode)) < 0 ||
      rpc_write(conn, nodeParams, sizeof(*nodeParams)) < 0 ||
      (nodeParams->numExtSems != 0 &&
       rpc_write(conn, nodeParams->extSemArray,
                 nodeParams->numExtSems * sizeof(*nodeParams->extSemArray)) <
           0) ||
      (nodeParams->numExtSems != 0 &&
       rpc_write(conn, nodeParams->paramsArray,
                 nodeParams->numExtSems * sizeof(*nodeParams->paramsArray)) <
           0) ||
      rpc_wait_for_response(conn) < 0 ||
      rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
      rpc_read_end(conn) < 0)
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  return return_value;
}

CUresult cuGraphNodeSetEnabled(CUgraphExec hGraphExec, CUgraphNode hNode,
                               unsigned int isEnabled) {
  lupine_route route = lupine_route_for_graph_exec(hGraphExec);
  CUresult return_value;
  using real_fn_t = CUresult (*)(CUgraphExec, CUgraphNode, unsigned int);
  if (lupine_call_local_cuda_if_routed<real_fn_t>(
          route, "cuGraphNodeSetEnabled", &return_value, hGraphExec, hNode,
          isEnabled)) {
    return return_value;
  }
  conn_t *conn = lupine_route_remote_conn(route);
  if (conn == nullptr ||
      rpc_write_start_request(conn, RPC_cuGraphNodeSetEnabled) < 0 ||
      rpc_write(conn, &hGraphExec, sizeof(CUgraphExec)) < 0 ||
      rpc_write(conn, &hNode, sizeof(CUgraphNode)) < 0 ||
      rpc_write(conn, &isEnabled, sizeof(unsigned int)) < 0 ||
      rpc_wait_for_response(conn) < 0 ||
      rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
      rpc_read_end(conn) < 0)
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  return return_value;
}

CUresult cuGraphNodeGetEnabled(CUgraphExec hGraphExec, CUgraphNode hNode,
                               unsigned int *isEnabled) {
  lupine_route route = lupine_route_for_graph_exec(hGraphExec);
  CUresult return_value;
  using real_fn_t = CUresult (*)(CUgraphExec, CUgraphNode, unsigned int *);
  if (lupine_call_local_cuda_if_routed<real_fn_t>(
          route, "cuGraphNodeGetEnabled", &return_value, hGraphExec, hNode,
          isEnabled)) {
    return return_value;
  }
  conn_t *conn = lupine_route_remote_conn(route);
  if (conn == nullptr ||
      rpc_write_start_request(conn, RPC_cuGraphNodeGetEnabled) < 0 ||
      rpc_write(conn, &hGraphExec, sizeof(CUgraphExec)) < 0 ||
      rpc_write(conn, &hNode, sizeof(CUgraphNode)) < 0 ||
      rpc_write(conn, isEnabled, sizeof(unsigned int)) < 0 ||
      rpc_wait_for_response(conn) < 0 ||
      rpc_read(conn, isEnabled, sizeof(unsigned int)) < 0 ||
      rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
      rpc_read_end(conn) < 0)
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  return return_value;
}

CUresult cuGraphUpload(CUgraphExec hGraphExec, CUstream hStream) {
  lupine_route route = lupine_route_for_graph_exec(hGraphExec);
  CUresult return_value;
  using real_fn_t = CUresult (*)(CUgraphExec, CUstream);
  if (lupine_call_local_cuda_if_routed<real_fn_t>(
          route, "cuGraphUpload", &return_value, hGraphExec, hStream)) {
    return return_value;
  }
  conn_t *conn = lupine_route_remote_conn(route);
  if (conn == nullptr || rpc_write_start_request(conn, RPC_cuGraphUpload) < 0 ||
      rpc_write(conn, &hGraphExec, sizeof(CUgraphExec)) < 0 ||
      rpc_write(conn, &hStream, sizeof(CUstream)) < 0 ||
      rpc_wait_for_response(conn) < 0 ||
      rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
      rpc_read_end(conn) < 0)
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  return return_value;
}

CUresult cuGraphLaunch(CUgraphExec hGraphExec, CUstream hStream) {
  lupine_route route = lupine_route_for_graph_exec(hGraphExec);
  CUresult return_value;
  using real_fn_t = CUresult (*)(CUgraphExec, CUstream);
  if (lupine_call_local_cuda_if_routed<real_fn_t>(
          route, "cuGraphLaunch", &return_value, hGraphExec, hStream)) {
    return return_value;
  }
  conn_t *conn = lupine_route_remote_conn(route);
  if (conn == nullptr || rpc_write_start_request(conn, RPC_cuGraphLaunch) < 0 ||
      rpc_write(conn, &hGraphExec, sizeof(CUgraphExec)) < 0 ||
      rpc_write(conn, &hStream, sizeof(CUstream)) < 0 ||
      rpc_wait_for_response(conn) < 0 ||
      rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
      rpc_read_end(conn) < 0)
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  return return_value;
}

CUresult cuGraphExecDestroy(CUgraphExec hGraphExec) {
  lupine_route route = lupine_route_for_graph_exec(hGraphExec);
  CUresult return_value;
  using real_fn_t = CUresult (*)(CUgraphExec);
  if (lupine_call_local_cuda_if_routed<real_fn_t>(route, "cuGraphExecDestroy",
                                                  &return_value, hGraphExec)) {
    return return_value;
  }
  conn_t *conn = lupine_route_remote_conn(route);
  if (conn == nullptr ||
      rpc_write_start_request(conn, RPC_cuGraphExecDestroy) < 0 ||
      rpc_write(conn, &hGraphExec, sizeof(CUgraphExec)) < 0 ||
      rpc_wait_for_response(conn) < 0 ||
      rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
      rpc_read_end(conn) < 0)
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  return return_value;
}

CUresult cuGraphDestroy(CUgraph hGraph) {
  lupine_route route = lupine_route_for_graph(hGraph);
  CUresult return_value;
  using real_fn_t = CUresult (*)(CUgraph);
  if (lupine_call_local_cuda_if_routed<real_fn_t>(route, "cuGraphDestroy",
                                                  &return_value, hGraph)) {
    return return_value;
  }
  conn_t *conn = lupine_route_remote_conn(route);
  if (conn == nullptr ||
      rpc_write_start_request(conn, RPC_cuGraphDestroy) < 0 ||
      rpc_write(conn, &hGraph, sizeof(CUgraph)) < 0 ||
      rpc_wait_for_response(conn) < 0 ||
      rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
      rpc_read_end(conn) < 0)
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  return return_value;
}

CUresult cuGraphExecUpdate_v2(CUgraphExec hGraphExec, CUgraph hGraph,
                              CUgraphExecUpdateResultInfo *resultInfo) {
  lupine_route route = lupine_route_for_graph_exec(hGraphExec);
  CUresult return_value;
  using real_fn_t =
      CUresult (*)(CUgraphExec, CUgraph, CUgraphExecUpdateResultInfo *);
  if (lupine_call_local_cuda_if_routed<real_fn_t>(route, "cuGraphExecUpdate_v2",
                                                  &return_value, hGraphExec,
                                                  hGraph, resultInfo)) {
    return return_value;
  }
  conn_t *conn = lupine_route_remote_conn(route);
  if (conn == nullptr ||
      rpc_write_start_request(conn, RPC_cuGraphExecUpdate_v2) < 0 ||
      rpc_write(conn, &hGraphExec, sizeof(CUgraphExec)) < 0 ||
      rpc_write(conn, &hGraph, sizeof(CUgraph)) < 0 ||
      rpc_write(conn, resultInfo, sizeof(CUgraphExecUpdateResultInfo)) < 0 ||
      rpc_wait_for_response(conn) < 0 ||
      rpc_read(conn, resultInfo, sizeof(CUgraphExecUpdateResultInfo)) < 0 ||
      rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
      rpc_read_end(conn) < 0)
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  return return_value;
}

CUresult cuGraphKernelNodeCopyAttributes(CUgraphNode dst, CUgraphNode src) {
  lupine_route route = lupine_route_for_graph_node(dst);
  CUresult return_value;
  using real_fn_t = CUresult (*)(CUgraphNode, CUgraphNode);
  if (lupine_call_local_cuda_if_routed<real_fn_t>(
          route, "cuGraphKernelNodeCopyAttributes", &return_value, dst, src)) {
    return return_value;
  }
  conn_t *conn = lupine_route_remote_conn(route);
  if (conn == nullptr ||
      rpc_write_start_request(conn, RPC_cuGraphKernelNodeCopyAttributes) < 0 ||
      rpc_write(conn, &dst, sizeof(CUgraphNode)) < 0 ||
      rpc_write(conn, &src, sizeof(CUgraphNode)) < 0 ||
      rpc_wait_for_response(conn) < 0 ||
      rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
      rpc_read_end(conn) < 0)
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  return return_value;
}

CUresult cuGraphKernelNodeGetAttribute(CUgraphNode hNode,
                                       CUkernelNodeAttrID attr,
                                       CUkernelNodeAttrValue *value_out) {
  lupine_route route = lupine_route_for_graph_node(hNode);
  CUresult return_value;
  using real_fn_t =
      CUresult (*)(CUgraphNode, CUkernelNodeAttrID, CUkernelNodeAttrValue *);
  if (lupine_call_local_cuda_if_routed<real_fn_t>(
          route, "cuGraphKernelNodeGetAttribute", &return_value, hNode, attr,
          value_out)) {
    return return_value;
  }
  conn_t *conn = lupine_route_remote_conn(route);
  if (conn == nullptr ||
      rpc_write_start_request(conn, RPC_cuGraphKernelNodeGetAttribute) < 0 ||
      rpc_write(conn, &hNode, sizeof(CUgraphNode)) < 0 ||
      rpc_write(conn, &attr, sizeof(CUkernelNodeAttrID)) < 0 ||
      rpc_write(conn, value_out, sizeof(CUkernelNodeAttrValue)) < 0 ||
      rpc_wait_for_response(conn) < 0 ||
      rpc_read(conn, value_out, sizeof(CUkernelNodeAttrValue)) < 0 ||
      rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
      rpc_read_end(conn) < 0)
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  return return_value;
}

CUresult cuGraphKernelNodeSetAttribute(CUgraphNode hNode,
                                       CUkernelNodeAttrID attr,
                                       const CUkernelNodeAttrValue *value) {
  lupine_route route = lupine_route_for_graph_node(hNode);
  CUresult return_value;
  using real_fn_t = CUresult (*)(CUgraphNode, CUkernelNodeAttrID,
                                 const CUkernelNodeAttrValue *);
  if (lupine_call_local_cuda_if_routed<real_fn_t>(
          route, "cuGraphKernelNodeSetAttribute", &return_value, hNode, attr,
          value)) {
    return return_value;
  }
  conn_t *conn = lupine_route_remote_conn(route);
  if (conn == nullptr ||
      rpc_write_start_request(conn, RPC_cuGraphKernelNodeSetAttribute) < 0 ||
      rpc_write(conn, &hNode, sizeof(CUgraphNode)) < 0 ||
      rpc_write(conn, &attr, sizeof(CUkernelNodeAttrID)) < 0 ||
      rpc_write(conn, &value, sizeof(const CUkernelNodeAttrValue *)) < 0 ||
      rpc_wait_for_response(conn) < 0 ||
      rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
      rpc_read_end(conn) < 0)
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  return return_value;
}

CUresult cuGraphDebugDotPrint(CUgraph hGraph, const char *path,
                              unsigned int flags) {
  lupine_route route = lupine_route_for_graph(hGraph);
  CUresult return_value;
  using real_fn_t = CUresult (*)(CUgraph, const char *, unsigned int);
  if (lupine_call_local_cuda_if_routed<real_fn_t>(
          route, "cuGraphDebugDotPrint", &return_value, hGraph, path, flags)) {
    return return_value;
  }
  conn_t *conn = lupine_route_remote_conn(route);
  std::size_t path_len = std::strlen(path) + 1;
  if (conn == nullptr ||
      rpc_write_start_request(conn, RPC_cuGraphDebugDotPrint) < 0 ||
      rpc_write(conn, &hGraph, sizeof(CUgraph)) < 0 ||
      rpc_write(conn, &path_len, sizeof(std::size_t)) < 0 ||
      rpc_write(conn, path, path_len) < 0 ||
      rpc_write(conn, &flags, sizeof(unsigned int)) < 0 ||
      rpc_wait_for_response(conn) < 0 ||
      rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
      rpc_read_end(conn) < 0)
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  return return_value;
}

CUresult cuUserObjectRetain(CUuserObject object, unsigned int count) {
  lupine_route route = lupine_route_for_default();
  CUresult return_value;
  using real_fn_t = CUresult (*)(CUuserObject, unsigned int);
  if (lupine_call_local_cuda_if_routed<real_fn_t>(
          route, "cuUserObjectRetain", &return_value, object, count)) {
    return return_value;
  }
  conn_t *conn = lupine_route_remote_conn(route);
  if (conn == nullptr ||
      rpc_write_start_request(conn, RPC_cuUserObjectRetain) < 0 ||
      rpc_write(conn, &object, sizeof(CUuserObject)) < 0 ||
      rpc_write(conn, &count, sizeof(unsigned int)) < 0 ||
      rpc_wait_for_response(conn) < 0 ||
      rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
      rpc_read_end(conn) < 0)
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  return return_value;
}

CUresult cuUserObjectRelease(CUuserObject object, unsigned int count) {
  lupine_route route = lupine_route_for_default();
  CUresult return_value;
  using real_fn_t = CUresult (*)(CUuserObject, unsigned int);
  if (lupine_call_local_cuda_if_routed<real_fn_t>(
          route, "cuUserObjectRelease", &return_value, object, count)) {
    return return_value;
  }
  conn_t *conn = lupine_route_remote_conn(route);
  if (conn == nullptr ||
      rpc_write_start_request(conn, RPC_cuUserObjectRelease) < 0 ||
      rpc_write(conn, &object, sizeof(CUuserObject)) < 0 ||
      rpc_write(conn, &count, sizeof(unsigned int)) < 0 ||
      rpc_wait_for_response(conn) < 0 ||
      rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
      rpc_read_end(conn) < 0)
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  return return_value;
}

CUresult cuGraphRetainUserObject(CUgraph graph, CUuserObject object,
                                 unsigned int count, unsigned int flags) {
  lupine_route route = lupine_route_for_graph(graph);
  CUresult return_value;
  using real_fn_t =
      CUresult (*)(CUgraph, CUuserObject, unsigned int, unsigned int);
  if (lupine_call_local_cuda_if_routed<real_fn_t>(
          route, "cuGraphRetainUserObject", &return_value, graph, object, count,
          flags)) {
    return return_value;
  }
  conn_t *conn = lupine_route_remote_conn(route);
  if (conn == nullptr ||
      rpc_write_start_request(conn, RPC_cuGraphRetainUserObject) < 0 ||
      rpc_write(conn, &graph, sizeof(CUgraph)) < 0 ||
      rpc_write(conn, &object, sizeof(CUuserObject)) < 0 ||
      rpc_write(conn, &count, sizeof(unsigned int)) < 0 ||
      rpc_write(conn, &flags, sizeof(unsigned int)) < 0 ||
      rpc_wait_for_response(conn) < 0 ||
      rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
      rpc_read_end(conn) < 0)
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  return return_value;
}

CUresult cuGraphReleaseUserObject(CUgraph graph, CUuserObject object,
                                  unsigned int count) {
  lupine_route route = lupine_route_for_graph(graph);
  CUresult return_value;
  using real_fn_t = CUresult (*)(CUgraph, CUuserObject, unsigned int);
  if (lupine_call_local_cuda_if_routed<real_fn_t>(
          route, "cuGraphReleaseUserObject", &return_value, graph, object,
          count)) {
    return return_value;
  }
  conn_t *conn = lupine_route_remote_conn(route);
  if (conn == nullptr ||
      rpc_write_start_request(conn, RPC_cuGraphReleaseUserObject) < 0 ||
      rpc_write(conn, &graph, sizeof(CUgraph)) < 0 ||
      rpc_write(conn, &object, sizeof(CUuserObject)) < 0 ||
      rpc_write(conn, &count, sizeof(unsigned int)) < 0 ||
      rpc_wait_for_response(conn) < 0 ||
      rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
      rpc_read_end(conn) < 0)
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  return return_value;
}

CUresult cuOccupancyMaxActiveBlocksPerMultiprocessor(int *numBlocks,
                                                     CUfunction func,
                                                     int blockSize,
                                                     size_t dynamicSMemSize) {
  return lupine_cuOccupancyMaxActiveBlocksPerMultiprocessor_cached(
      numBlocks, func, blockSize, dynamicSMemSize);
}

CUresult cuOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(
    int *numBlocks, CUfunction func, int blockSize, size_t dynamicSMemSize,
    unsigned int flags) {
  return lupine_cuOccupancyMaxActiveBlocksPerMultiprocessorWithFlags_cached(
      numBlocks, func, blockSize, dynamicSMemSize, flags);
}

CUresult cuOccupancyAvailableDynamicSMemPerBlock(size_t *dynamicSmemSize,
                                                 CUfunction func, int numBlocks,
                                                 int blockSize) {
  lupine_route route = lupine_route_for_function(func);
  CUresult return_value;
  using real_fn_t = CUresult (*)(size_t *, CUfunction, int, int);
  if (lupine_call_local_cuda_if_routed<real_fn_t>(
          route, "cuOccupancyAvailableDynamicSMemPerBlock", &return_value,
          dynamicSmemSize, func, numBlocks, blockSize)) {
    return return_value;
  }
  conn_t *conn = lupine_route_remote_conn(route);
  CUfunction func_rpc = lupine_translate_private_function_for_rpc(func);
  if (conn == nullptr ||
      rpc_write_start_request(
          conn, RPC_cuOccupancyAvailableDynamicSMemPerBlock) < 0 ||
      rpc_write(conn, dynamicSmemSize, sizeof(size_t)) < 0 ||
      rpc_write(conn, &func_rpc, sizeof(CUfunction)) < 0 ||
      rpc_write(conn, &numBlocks, sizeof(int)) < 0 ||
      rpc_write(conn, &blockSize, sizeof(int)) < 0 ||
      rpc_wait_for_response(conn) < 0 ||
      rpc_read(conn, dynamicSmemSize, sizeof(size_t)) < 0 ||
      rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
      rpc_read_end(conn) < 0)
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  return return_value;
}

CUresult cuOccupancyMaxPotentialClusterSize(int *clusterSize, CUfunction func,
                                            const CUlaunchConfig *config) {
  lupine_route route = lupine_route_for_function(func);
  CUresult return_value;
  using real_fn_t = CUresult (*)(int *, CUfunction, const CUlaunchConfig *);
  if (lupine_call_local_cuda_if_routed<real_fn_t>(
          route, "cuOccupancyMaxPotentialClusterSize", &return_value,
          clusterSize, func, config)) {
    return return_value;
  }
  conn_t *conn = lupine_route_remote_conn(route);
  CUfunction func_rpc = lupine_translate_private_function_for_rpc(func);
  if (conn == nullptr ||
      rpc_write_start_request(conn, RPC_cuOccupancyMaxPotentialClusterSize) <
          0 ||
      rpc_write(conn, clusterSize, sizeof(int)) < 0 ||
      rpc_write(conn, &func_rpc, sizeof(CUfunction)) < 0 ||
      rpc_write(conn, &config, sizeof(const CUlaunchConfig *)) < 0 ||
      rpc_wait_for_response(conn) < 0 ||
      rpc_read(conn, clusterSize, sizeof(int)) < 0 ||
      rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
      rpc_read_end(conn) < 0)
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  return return_value;
}

CUresult cuOccupancyMaxActiveClusters(int *numClusters, CUfunction func,
                                      const CUlaunchConfig *config) {
  lupine_route route = lupine_route_for_function(func);
  CUresult return_value;
  using real_fn_t = CUresult (*)(int *, CUfunction, const CUlaunchConfig *);
  if (lupine_call_local_cuda_if_routed<real_fn_t>(
          route, "cuOccupancyMaxActiveClusters", &return_value, numClusters,
          func, config)) {
    return return_value;
  }
  conn_t *conn = lupine_route_remote_conn(route);
  CUfunction func_rpc = lupine_translate_private_function_for_rpc(func);
  if (conn == nullptr ||
      rpc_write_start_request(conn, RPC_cuOccupancyMaxActiveClusters) < 0 ||
      rpc_write(conn, numClusters, sizeof(int)) < 0 ||
      rpc_write(conn, &func_rpc, sizeof(CUfunction)) < 0 ||
      rpc_write(conn, &config, sizeof(const CUlaunchConfig *)) < 0 ||
      rpc_wait_for_response(conn) < 0 ||
      rpc_read(conn, numClusters, sizeof(int)) < 0 ||
      rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
      rpc_read_end(conn) < 0)
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  return return_value;
}

CUresult cuTexRefSetArray(CUtexref hTexRef, CUarray hArray,
                          unsigned int Flags) {
  lupine_route route = lupine_route_for_default();
  CUresult return_value;
  using real_fn_t = CUresult (*)(CUtexref, CUarray, unsigned int);
  if (lupine_call_local_cuda_if_routed<real_fn_t>(
          route, "cuTexRefSetArray", &return_value, hTexRef, hArray, Flags)) {
    return return_value;
  }
  conn_t *conn = lupine_route_remote_conn(route);
  if (conn == nullptr ||
      rpc_write_start_request(conn, RPC_cuTexRefSetArray) < 0 ||
      rpc_write(conn, &hTexRef, sizeof(CUtexref)) < 0 ||
      rpc_write(conn, &hArray, sizeof(CUarray)) < 0 ||
      rpc_write(conn, &Flags, sizeof(unsigned int)) < 0 ||
      rpc_wait_for_response(conn) < 0 ||
      rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
      rpc_read_end(conn) < 0)
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  return return_value;
}

CUresult cuTexRefSetMipmappedArray(CUtexref hTexRef,
                                   CUmipmappedArray hMipmappedArray,
                                   unsigned int Flags) {
  lupine_route route = lupine_route_for_default();
  CUresult return_value;
  using real_fn_t = CUresult (*)(CUtexref, CUmipmappedArray, unsigned int);
  if (lupine_call_local_cuda_if_routed<real_fn_t>(
          route, "cuTexRefSetMipmappedArray", &return_value, hTexRef,
          hMipmappedArray, Flags)) {
    return return_value;
  }
  conn_t *conn = lupine_route_remote_conn(route);
  if (conn == nullptr ||
      rpc_write_start_request(conn, RPC_cuTexRefSetMipmappedArray) < 0 ||
      rpc_write(conn, &hTexRef, sizeof(CUtexref)) < 0 ||
      rpc_write(conn, &hMipmappedArray, sizeof(CUmipmappedArray)) < 0 ||
      rpc_write(conn, &Flags, sizeof(unsigned int)) < 0 ||
      rpc_wait_for_response(conn) < 0 ||
      rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
      rpc_read_end(conn) < 0)
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  return return_value;
}

CUresult cuTexRefSetAddress_v2(size_t *ByteOffset, CUtexref hTexRef,
                               CUdeviceptr dptr, size_t bytes) {
  lupine_route route = lupine_route_for_deviceptr(dptr);
  CUresult return_value;
  using real_fn_t = CUresult (*)(size_t *, CUtexref, CUdeviceptr, size_t);
  if (lupine_call_local_cuda_if_routed<real_fn_t>(
          route, "cuTexRefSetAddress_v2", &return_value, ByteOffset, hTexRef,
          dptr, bytes)) {
    return return_value;
  }
  conn_t *conn = lupine_route_remote_conn(route);
  if (conn == nullptr ||
      rpc_write_start_request(conn, RPC_cuTexRefSetAddress_v2) < 0 ||
      rpc_write(conn, ByteOffset, sizeof(size_t)) < 0 ||
      rpc_write(conn, &hTexRef, sizeof(CUtexref)) < 0 ||
      rpc_write(conn, &dptr, sizeof(CUdeviceptr)) < 0 ||
      rpc_write(conn, &bytes, sizeof(size_t)) < 0 ||
      rpc_wait_for_response(conn) < 0 ||
      rpc_read(conn, ByteOffset, sizeof(size_t)) < 0 ||
      rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
      rpc_read_end(conn) < 0)
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  return return_value;
}

CUresult cuTexRefSetAddress2D_v3(CUtexref hTexRef,
                                 const CUDA_ARRAY_DESCRIPTOR *desc,
                                 CUdeviceptr dptr, size_t Pitch) {
  lupine_route route = lupine_route_for_deviceptr(dptr);
  CUresult return_value;
  using real_fn_t = CUresult (*)(CUtexref, const CUDA_ARRAY_DESCRIPTOR *,
                                 CUdeviceptr, size_t);
  if (lupine_call_local_cuda_if_routed<real_fn_t>(
          route, "cuTexRefSetAddress2D_v3", &return_value, hTexRef, desc, dptr,
          Pitch)) {
    return return_value;
  }
  conn_t *conn = lupine_route_remote_conn(route);
  if (conn == nullptr ||
      rpc_write_start_request(conn, RPC_cuTexRefSetAddress2D_v3) < 0 ||
      rpc_write(conn, &hTexRef, sizeof(CUtexref)) < 0 ||
      rpc_write(conn, desc, sizeof(const CUDA_ARRAY_DESCRIPTOR)) < 0 ||
      rpc_write(conn, &dptr, sizeof(CUdeviceptr)) < 0 ||
      rpc_write(conn, &Pitch, sizeof(size_t)) < 0 ||
      rpc_wait_for_response(conn) < 0 ||
      rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
      rpc_read_end(conn) < 0)
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  return return_value;
}

CUresult cuTexRefSetFormat(CUtexref hTexRef, CUarray_format fmt,
                           int NumPackedComponents) {
  lupine_route route = lupine_route_for_default();
  CUresult return_value;
  using real_fn_t = CUresult (*)(CUtexref, CUarray_format, int);
  if (lupine_call_local_cuda_if_routed<real_fn_t>(route, "cuTexRefSetFormat",
                                                  &return_value, hTexRef, fmt,
                                                  NumPackedComponents)) {
    return return_value;
  }
  conn_t *conn = lupine_route_remote_conn(route);
  if (conn == nullptr ||
      rpc_write_start_request(conn, RPC_cuTexRefSetFormat) < 0 ||
      rpc_write(conn, &hTexRef, sizeof(CUtexref)) < 0 ||
      rpc_write(conn, &fmt, sizeof(CUarray_format)) < 0 ||
      rpc_write(conn, &NumPackedComponents, sizeof(int)) < 0 ||
      rpc_wait_for_response(conn) < 0 ||
      rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
      rpc_read_end(conn) < 0)
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  return return_value;
}

CUresult cuTexRefSetAddressMode(CUtexref hTexRef, int dim, CUaddress_mode am) {
  lupine_route route = lupine_route_for_default();
  CUresult return_value;
  using real_fn_t = CUresult (*)(CUtexref, int, CUaddress_mode);
  if (lupine_call_local_cuda_if_routed<real_fn_t>(
          route, "cuTexRefSetAddressMode", &return_value, hTexRef, dim, am)) {
    return return_value;
  }
  conn_t *conn = lupine_route_remote_conn(route);
  if (conn == nullptr ||
      rpc_write_start_request(conn, RPC_cuTexRefSetAddressMode) < 0 ||
      rpc_write(conn, &hTexRef, sizeof(CUtexref)) < 0 ||
      rpc_write(conn, &dim, sizeof(int)) < 0 ||
      rpc_write(conn, &am, sizeof(CUaddress_mode)) < 0 ||
      rpc_wait_for_response(conn) < 0 ||
      rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
      rpc_read_end(conn) < 0)
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  return return_value;
}

CUresult cuTexRefSetFilterMode(CUtexref hTexRef, CUfilter_mode fm) {
  lupine_route route = lupine_route_for_default();
  CUresult return_value;
  using real_fn_t = CUresult (*)(CUtexref, CUfilter_mode);
  if (lupine_call_local_cuda_if_routed<real_fn_t>(
          route, "cuTexRefSetFilterMode", &return_value, hTexRef, fm)) {
    return return_value;
  }
  conn_t *conn = lupine_route_remote_conn(route);
  if (conn == nullptr ||
      rpc_write_start_request(conn, RPC_cuTexRefSetFilterMode) < 0 ||
      rpc_write(conn, &hTexRef, sizeof(CUtexref)) < 0 ||
      rpc_write(conn, &fm, sizeof(CUfilter_mode)) < 0 ||
      rpc_wait_for_response(conn) < 0 ||
      rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
      rpc_read_end(conn) < 0)
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  return return_value;
}

CUresult cuTexRefSetMipmapFilterMode(CUtexref hTexRef, CUfilter_mode fm) {
  lupine_route route = lupine_route_for_default();
  CUresult return_value;
  using real_fn_t = CUresult (*)(CUtexref, CUfilter_mode);
  if (lupine_call_local_cuda_if_routed<real_fn_t>(
          route, "cuTexRefSetMipmapFilterMode", &return_value, hTexRef, fm)) {
    return return_value;
  }
  conn_t *conn = lupine_route_remote_conn(route);
  if (conn == nullptr ||
      rpc_write_start_request(conn, RPC_cuTexRefSetMipmapFilterMode) < 0 ||
      rpc_write(conn, &hTexRef, sizeof(CUtexref)) < 0 ||
      rpc_write(conn, &fm, sizeof(CUfilter_mode)) < 0 ||
      rpc_wait_for_response(conn) < 0 ||
      rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
      rpc_read_end(conn) < 0)
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  return return_value;
}

CUresult cuTexRefSetMipmapLevelBias(CUtexref hTexRef, float bias) {
  lupine_route route = lupine_route_for_default();
  CUresult return_value;
  using real_fn_t = CUresult (*)(CUtexref, float);
  if (lupine_call_local_cuda_if_routed<real_fn_t>(
          route, "cuTexRefSetMipmapLevelBias", &return_value, hTexRef, bias)) {
    return return_value;
  }
  conn_t *conn = lupine_route_remote_conn(route);
  if (conn == nullptr ||
      rpc_write_start_request(conn, RPC_cuTexRefSetMipmapLevelBias) < 0 ||
      rpc_write(conn, &hTexRef, sizeof(CUtexref)) < 0 ||
      rpc_write(conn, &bias, sizeof(float)) < 0 ||
      rpc_wait_for_response(conn) < 0 ||
      rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
      rpc_read_end(conn) < 0)
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  return return_value;
}

CUresult cuTexRefSetMipmapLevelClamp(CUtexref hTexRef,
                                     float minMipmapLevelClamp,
                                     float maxMipmapLevelClamp) {
  lupine_route route = lupine_route_for_default();
  CUresult return_value;
  using real_fn_t = CUresult (*)(CUtexref, float, float);
  if (lupine_call_local_cuda_if_routed<real_fn_t>(
          route, "cuTexRefSetMipmapLevelClamp", &return_value, hTexRef,
          minMipmapLevelClamp, maxMipmapLevelClamp)) {
    return return_value;
  }
  conn_t *conn = lupine_route_remote_conn(route);
  if (conn == nullptr ||
      rpc_write_start_request(conn, RPC_cuTexRefSetMipmapLevelClamp) < 0 ||
      rpc_write(conn, &hTexRef, sizeof(CUtexref)) < 0 ||
      rpc_write(conn, &minMipmapLevelClamp, sizeof(float)) < 0 ||
      rpc_write(conn, &maxMipmapLevelClamp, sizeof(float)) < 0 ||
      rpc_wait_for_response(conn) < 0 ||
      rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
      rpc_read_end(conn) < 0)
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  return return_value;
}

CUresult cuTexRefSetMaxAnisotropy(CUtexref hTexRef, unsigned int maxAniso) {
  lupine_route route = lupine_route_for_default();
  CUresult return_value;
  using real_fn_t = CUresult (*)(CUtexref, unsigned int);
  if (lupine_call_local_cuda_if_routed<real_fn_t>(
          route, "cuTexRefSetMaxAnisotropy", &return_value, hTexRef,
          maxAniso)) {
    return return_value;
  }
  conn_t *conn = lupine_route_remote_conn(route);
  if (conn == nullptr ||
      rpc_write_start_request(conn, RPC_cuTexRefSetMaxAnisotropy) < 0 ||
      rpc_write(conn, &hTexRef, sizeof(CUtexref)) < 0 ||
      rpc_write(conn, &maxAniso, sizeof(unsigned int)) < 0 ||
      rpc_wait_for_response(conn) < 0 ||
      rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
      rpc_read_end(conn) < 0)
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  return return_value;
}

CUresult cuTexRefSetBorderColor(CUtexref hTexRef, float *pBorderColor) {
  lupine_route route = lupine_route_for_default();
  CUresult return_value;
  using real_fn_t = CUresult (*)(CUtexref, float *);
  if (lupine_call_local_cuda_if_routed<real_fn_t>(
          route, "cuTexRefSetBorderColor", &return_value, hTexRef,
          pBorderColor)) {
    return return_value;
  }
  conn_t *conn = lupine_route_remote_conn(route);
  if (conn == nullptr ||
      rpc_write_start_request(conn, RPC_cuTexRefSetBorderColor) < 0 ||
      rpc_write(conn, &hTexRef, sizeof(CUtexref)) < 0 ||
      rpc_write(conn, pBorderColor, sizeof(float)) < 0 ||
      rpc_wait_for_response(conn) < 0 ||
      rpc_read(conn, pBorderColor, sizeof(float)) < 0 ||
      rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
      rpc_read_end(conn) < 0)
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  return return_value;
}

CUresult cuTexRefSetFlags(CUtexref hTexRef, unsigned int Flags) {
  lupine_route route = lupine_route_for_default();
  CUresult return_value;
  using real_fn_t = CUresult (*)(CUtexref, unsigned int);
  if (lupine_call_local_cuda_if_routed<real_fn_t>(
          route, "cuTexRefSetFlags", &return_value, hTexRef, Flags)) {
    return return_value;
  }
  conn_t *conn = lupine_route_remote_conn(route);
  if (conn == nullptr ||
      rpc_write_start_request(conn, RPC_cuTexRefSetFlags) < 0 ||
      rpc_write(conn, &hTexRef, sizeof(CUtexref)) < 0 ||
      rpc_write(conn, &Flags, sizeof(unsigned int)) < 0 ||
      rpc_wait_for_response(conn) < 0 ||
      rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
      rpc_read_end(conn) < 0)
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  return return_value;
}

CUresult cuTexRefGetAddress_v2(CUdeviceptr *pdptr, CUtexref hTexRef) {
  lupine_route route = lupine_route_for_default();
  CUresult return_value;
  using real_fn_t = CUresult (*)(CUdeviceptr *, CUtexref);
  if (lupine_call_local_cuda_if_routed<real_fn_t>(
          route, "cuTexRefGetAddress_v2", &return_value, pdptr, hTexRef)) {
    return return_value;
  }
  conn_t *conn = lupine_route_remote_conn(route);
  if (conn == nullptr ||
      rpc_write_start_request(conn, RPC_cuTexRefGetAddress_v2) < 0 ||
      rpc_write(conn, pdptr, sizeof(CUdeviceptr)) < 0 ||
      rpc_write(conn, &hTexRef, sizeof(CUtexref)) < 0 ||
      rpc_wait_for_response(conn) < 0 ||
      rpc_read(conn, pdptr, sizeof(CUdeviceptr)) < 0 ||
      rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
      rpc_read_end(conn) < 0)
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  return return_value;
}

CUresult cuTexRefGetArray(CUarray *phArray, CUtexref hTexRef) {
  lupine_route route = lupine_route_for_default();
  CUresult return_value;
  using real_fn_t = CUresult (*)(CUarray *, CUtexref);
  if (lupine_call_local_cuda_if_routed<real_fn_t>(
          route, "cuTexRefGetArray", &return_value, phArray, hTexRef)) {
    return return_value;
  }
  conn_t *conn = lupine_route_remote_conn(route);
  if (conn == nullptr ||
      rpc_write_start_request(conn, RPC_cuTexRefGetArray) < 0 ||
      rpc_write(conn, phArray, sizeof(CUarray)) < 0 ||
      rpc_write(conn, &hTexRef, sizeof(CUtexref)) < 0 ||
      rpc_wait_for_response(conn) < 0 ||
      rpc_read(conn, phArray, sizeof(CUarray)) < 0 ||
      rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
      rpc_read_end(conn) < 0)
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  return return_value;
}

CUresult cuTexRefGetMipmappedArray(CUmipmappedArray *phMipmappedArray,
                                   CUtexref hTexRef) {
  lupine_route route = lupine_route_for_default();
  CUresult return_value;
  using real_fn_t = CUresult (*)(CUmipmappedArray *, CUtexref);
  if (lupine_call_local_cuda_if_routed<real_fn_t>(
          route, "cuTexRefGetMipmappedArray", &return_value, phMipmappedArray,
          hTexRef)) {
    return return_value;
  }
  conn_t *conn = lupine_route_remote_conn(route);
  if (conn == nullptr ||
      rpc_write_start_request(conn, RPC_cuTexRefGetMipmappedArray) < 0 ||
      rpc_write(conn, phMipmappedArray, sizeof(CUmipmappedArray)) < 0 ||
      rpc_write(conn, &hTexRef, sizeof(CUtexref)) < 0 ||
      rpc_wait_for_response(conn) < 0 ||
      rpc_read(conn, phMipmappedArray, sizeof(CUmipmappedArray)) < 0 ||
      rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
      rpc_read_end(conn) < 0)
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  return return_value;
}

CUresult cuTexRefGetAddressMode(CUaddress_mode *pam, CUtexref hTexRef,
                                int dim) {
  lupine_route route = lupine_route_for_default();
  CUresult return_value;
  using real_fn_t = CUresult (*)(CUaddress_mode *, CUtexref, int);
  if (lupine_call_local_cuda_if_routed<real_fn_t>(
          route, "cuTexRefGetAddressMode", &return_value, pam, hTexRef, dim)) {
    return return_value;
  }
  conn_t *conn = lupine_route_remote_conn(route);
  if (conn == nullptr ||
      rpc_write_start_request(conn, RPC_cuTexRefGetAddressMode) < 0 ||
      rpc_write(conn, pam, sizeof(CUaddress_mode)) < 0 ||
      rpc_write(conn, &hTexRef, sizeof(CUtexref)) < 0 ||
      rpc_write(conn, &dim, sizeof(int)) < 0 ||
      rpc_wait_for_response(conn) < 0 ||
      rpc_read(conn, pam, sizeof(CUaddress_mode)) < 0 ||
      rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
      rpc_read_end(conn) < 0)
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  return return_value;
}

CUresult cuTexRefGetFilterMode(CUfilter_mode *pfm, CUtexref hTexRef) {
  lupine_route route = lupine_route_for_default();
  CUresult return_value;
  using real_fn_t = CUresult (*)(CUfilter_mode *, CUtexref);
  if (lupine_call_local_cuda_if_routed<real_fn_t>(
          route, "cuTexRefGetFilterMode", &return_value, pfm, hTexRef)) {
    return return_value;
  }
  conn_t *conn = lupine_route_remote_conn(route);
  if (conn == nullptr ||
      rpc_write_start_request(conn, RPC_cuTexRefGetFilterMode) < 0 ||
      rpc_write(conn, pfm, sizeof(CUfilter_mode)) < 0 ||
      rpc_write(conn, &hTexRef, sizeof(CUtexref)) < 0 ||
      rpc_wait_for_response(conn) < 0 ||
      rpc_read(conn, pfm, sizeof(CUfilter_mode)) < 0 ||
      rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
      rpc_read_end(conn) < 0)
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  return return_value;
}

CUresult cuTexRefGetFormat(CUarray_format *pFormat, int *pNumChannels,
                           CUtexref hTexRef) {
  lupine_route route = lupine_route_for_default();
  CUresult return_value;
  using real_fn_t = CUresult (*)(CUarray_format *, int *, CUtexref);
  if (lupine_call_local_cuda_if_routed<real_fn_t>(route, "cuTexRefGetFormat",
                                                  &return_value, pFormat,
                                                  pNumChannels, hTexRef)) {
    return return_value;
  }
  conn_t *conn = lupine_route_remote_conn(route);
  if (conn == nullptr ||
      rpc_write_start_request(conn, RPC_cuTexRefGetFormat) < 0 ||
      rpc_write(conn, pFormat, sizeof(CUarray_format)) < 0 ||
      rpc_write(conn, pNumChannels, sizeof(int)) < 0 ||
      rpc_write(conn, &hTexRef, sizeof(CUtexref)) < 0 ||
      rpc_wait_for_response(conn) < 0 ||
      rpc_read(conn, pFormat, sizeof(CUarray_format)) < 0 ||
      rpc_read(conn, pNumChannels, sizeof(int)) < 0 ||
      rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
      rpc_read_end(conn) < 0)
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  return return_value;
}

CUresult cuTexRefGetMipmapFilterMode(CUfilter_mode *pfm, CUtexref hTexRef) {
  lupine_route route = lupine_route_for_default();
  CUresult return_value;
  using real_fn_t = CUresult (*)(CUfilter_mode *, CUtexref);
  if (lupine_call_local_cuda_if_routed<real_fn_t>(
          route, "cuTexRefGetMipmapFilterMode", &return_value, pfm, hTexRef)) {
    return return_value;
  }
  conn_t *conn = lupine_route_remote_conn(route);
  if (conn == nullptr ||
      rpc_write_start_request(conn, RPC_cuTexRefGetMipmapFilterMode) < 0 ||
      rpc_write(conn, pfm, sizeof(CUfilter_mode)) < 0 ||
      rpc_write(conn, &hTexRef, sizeof(CUtexref)) < 0 ||
      rpc_wait_for_response(conn) < 0 ||
      rpc_read(conn, pfm, sizeof(CUfilter_mode)) < 0 ||
      rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
      rpc_read_end(conn) < 0)
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  return return_value;
}

CUresult cuTexRefGetMipmapLevelBias(float *pbias, CUtexref hTexRef) {
  lupine_route route = lupine_route_for_default();
  CUresult return_value;
  using real_fn_t = CUresult (*)(float *, CUtexref);
  if (lupine_call_local_cuda_if_routed<real_fn_t>(
          route, "cuTexRefGetMipmapLevelBias", &return_value, pbias, hTexRef)) {
    return return_value;
  }
  conn_t *conn = lupine_route_remote_conn(route);
  if (conn == nullptr ||
      rpc_write_start_request(conn, RPC_cuTexRefGetMipmapLevelBias) < 0 ||
      rpc_write(conn, pbias, sizeof(float)) < 0 ||
      rpc_write(conn, &hTexRef, sizeof(CUtexref)) < 0 ||
      rpc_wait_for_response(conn) < 0 ||
      rpc_read(conn, pbias, sizeof(float)) < 0 ||
      rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
      rpc_read_end(conn) < 0)
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  return return_value;
}

CUresult cuTexRefGetMipmapLevelClamp(float *pminMipmapLevelClamp,
                                     float *pmaxMipmapLevelClamp,
                                     CUtexref hTexRef) {
  lupine_route route = lupine_route_for_default();
  CUresult return_value;
  using real_fn_t = CUresult (*)(float *, float *, CUtexref);
  if (lupine_call_local_cuda_if_routed<real_fn_t>(
          route, "cuTexRefGetMipmapLevelClamp", &return_value,
          pminMipmapLevelClamp, pmaxMipmapLevelClamp, hTexRef)) {
    return return_value;
  }
  conn_t *conn = lupine_route_remote_conn(route);
  if (conn == nullptr ||
      rpc_write_start_request(conn, RPC_cuTexRefGetMipmapLevelClamp) < 0 ||
      rpc_write(conn, pminMipmapLevelClamp, sizeof(float)) < 0 ||
      rpc_write(conn, pmaxMipmapLevelClamp, sizeof(float)) < 0 ||
      rpc_write(conn, &hTexRef, sizeof(CUtexref)) < 0 ||
      rpc_wait_for_response(conn) < 0 ||
      rpc_read(conn, pminMipmapLevelClamp, sizeof(float)) < 0 ||
      rpc_read(conn, pmaxMipmapLevelClamp, sizeof(float)) < 0 ||
      rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
      rpc_read_end(conn) < 0)
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  return return_value;
}

CUresult cuTexRefGetMaxAnisotropy(int *pmaxAniso, CUtexref hTexRef) {
  lupine_route route = lupine_route_for_default();
  CUresult return_value;
  using real_fn_t = CUresult (*)(int *, CUtexref);
  if (lupine_call_local_cuda_if_routed<real_fn_t>(
          route, "cuTexRefGetMaxAnisotropy", &return_value, pmaxAniso,
          hTexRef)) {
    return return_value;
  }
  conn_t *conn = lupine_route_remote_conn(route);
  if (conn == nullptr ||
      rpc_write_start_request(conn, RPC_cuTexRefGetMaxAnisotropy) < 0 ||
      rpc_write(conn, pmaxAniso, sizeof(int)) < 0 ||
      rpc_write(conn, &hTexRef, sizeof(CUtexref)) < 0 ||
      rpc_wait_for_response(conn) < 0 ||
      rpc_read(conn, pmaxAniso, sizeof(int)) < 0 ||
      rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
      rpc_read_end(conn) < 0)
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  return return_value;
}

CUresult cuTexRefGetBorderColor(float *pBorderColor, CUtexref hTexRef) {
  lupine_route route = lupine_route_for_default();
  CUresult return_value;
  using real_fn_t = CUresult (*)(float *, CUtexref);
  if (lupine_call_local_cuda_if_routed<real_fn_t>(
          route, "cuTexRefGetBorderColor", &return_value, pBorderColor,
          hTexRef)) {
    return return_value;
  }
  conn_t *conn = lupine_route_remote_conn(route);
  if (conn == nullptr ||
      rpc_write_start_request(conn, RPC_cuTexRefGetBorderColor) < 0 ||
      rpc_write(conn, pBorderColor, sizeof(float)) < 0 ||
      rpc_write(conn, &hTexRef, sizeof(CUtexref)) < 0 ||
      rpc_wait_for_response(conn) < 0 ||
      rpc_read(conn, pBorderColor, sizeof(float)) < 0 ||
      rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
      rpc_read_end(conn) < 0)
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  return return_value;
}

CUresult cuTexRefGetFlags(unsigned int *pFlags, CUtexref hTexRef) {
  lupine_route route = lupine_route_for_default();
  CUresult return_value;
  using real_fn_t = CUresult (*)(unsigned int *, CUtexref);
  if (lupine_call_local_cuda_if_routed<real_fn_t>(
          route, "cuTexRefGetFlags", &return_value, pFlags, hTexRef)) {
    return return_value;
  }
  conn_t *conn = lupine_route_remote_conn(route);
  if (conn == nullptr ||
      rpc_write_start_request(conn, RPC_cuTexRefGetFlags) < 0 ||
      rpc_write(conn, pFlags, sizeof(unsigned int)) < 0 ||
      rpc_write(conn, &hTexRef, sizeof(CUtexref)) < 0 ||
      rpc_wait_for_response(conn) < 0 ||
      rpc_read(conn, pFlags, sizeof(unsigned int)) < 0 ||
      rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
      rpc_read_end(conn) < 0)
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  return return_value;
}

CUresult cuTexRefCreate(CUtexref *pTexRef) {
  lupine_route route = lupine_route_for_default();
  CUresult return_value;
  using real_fn_t = CUresult (*)(CUtexref *);
  if (lupine_call_local_cuda_if_routed<real_fn_t>(route, "cuTexRefCreate",
                                                  &return_value, pTexRef)) {
    return return_value;
  }
  conn_t *conn = lupine_route_remote_conn(route);
  if (conn == nullptr ||
      rpc_write_start_request(conn, RPC_cuTexRefCreate) < 0 ||
      rpc_write(conn, pTexRef, sizeof(CUtexref)) < 0 ||
      rpc_wait_for_response(conn) < 0 ||
      rpc_read(conn, pTexRef, sizeof(CUtexref)) < 0 ||
      rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
      rpc_read_end(conn) < 0)
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  return return_value;
}

CUresult cuTexRefDestroy(CUtexref hTexRef) {
  lupine_route route = lupine_route_for_default();
  CUresult return_value;
  using real_fn_t = CUresult (*)(CUtexref);
  if (lupine_call_local_cuda_if_routed<real_fn_t>(route, "cuTexRefDestroy",
                                                  &return_value, hTexRef)) {
    return return_value;
  }
  conn_t *conn = lupine_route_remote_conn(route);
  if (conn == nullptr ||
      rpc_write_start_request(conn, RPC_cuTexRefDestroy) < 0 ||
      rpc_write(conn, &hTexRef, sizeof(CUtexref)) < 0 ||
      rpc_wait_for_response(conn) < 0 ||
      rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
      rpc_read_end(conn) < 0)
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  return return_value;
}

CUresult cuSurfRefSetArray(CUsurfref hSurfRef, CUarray hArray,
                           unsigned int Flags) {
  lupine_route route = lupine_route_for_default();
  CUresult return_value;
  using real_fn_t = CUresult (*)(CUsurfref, CUarray, unsigned int);
  if (lupine_call_local_cuda_if_routed<real_fn_t>(
          route, "cuSurfRefSetArray", &return_value, hSurfRef, hArray, Flags)) {
    return return_value;
  }
  conn_t *conn = lupine_route_remote_conn(route);
  if (conn == nullptr ||
      rpc_write_start_request(conn, RPC_cuSurfRefSetArray) < 0 ||
      rpc_write(conn, &hSurfRef, sizeof(CUsurfref)) < 0 ||
      rpc_write(conn, &hArray, sizeof(CUarray)) < 0 ||
      rpc_write(conn, &Flags, sizeof(unsigned int)) < 0 ||
      rpc_wait_for_response(conn) < 0 ||
      rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
      rpc_read_end(conn) < 0)
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  return return_value;
}

CUresult cuSurfRefGetArray(CUarray *phArray, CUsurfref hSurfRef) {
  lupine_route route = lupine_route_for_default();
  CUresult return_value;
  using real_fn_t = CUresult (*)(CUarray *, CUsurfref);
  if (lupine_call_local_cuda_if_routed<real_fn_t>(
          route, "cuSurfRefGetArray", &return_value, phArray, hSurfRef)) {
    return return_value;
  }
  conn_t *conn = lupine_route_remote_conn(route);
  if (conn == nullptr ||
      rpc_write_start_request(conn, RPC_cuSurfRefGetArray) < 0 ||
      rpc_write(conn, phArray, sizeof(CUarray)) < 0 ||
      rpc_write(conn, &hSurfRef, sizeof(CUsurfref)) < 0 ||
      rpc_wait_for_response(conn) < 0 ||
      rpc_read(conn, phArray, sizeof(CUarray)) < 0 ||
      rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
      rpc_read_end(conn) < 0)
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  return return_value;
}

CUresult cuTexObjectCreate(CUtexObject *pTexObject,
                           const CUDA_RESOURCE_DESC *pResDesc,
                           const CUDA_TEXTURE_DESC *pTexDesc,
                           const CUDA_RESOURCE_VIEW_DESC *pResViewDesc) {
  lupine_route route = lupine_route_for_default();
  CUresult return_value;
  using real_fn_t =
      CUresult (*)(CUtexObject *, const CUDA_RESOURCE_DESC *,
                   const CUDA_TEXTURE_DESC *, const CUDA_RESOURCE_VIEW_DESC *);
  if (lupine_call_local_cuda_if_routed<real_fn_t>(
          route, "cuTexObjectCreate", &return_value, pTexObject, pResDesc,
          pTexDesc, pResViewDesc)) {
    return return_value;
  }
  conn_t *conn = lupine_route_remote_conn(route);
  if (conn == nullptr ||
      rpc_write_start_request(conn, RPC_cuTexObjectCreate) < 0 ||
      rpc_write(conn, pTexObject, sizeof(CUtexObject)) < 0 ||
      rpc_write(conn, pResDesc, sizeof(const CUDA_RESOURCE_DESC)) < 0 ||
      rpc_write(conn, &pTexDesc, sizeof(const CUDA_TEXTURE_DESC *)) < 0 ||
      (pTexDesc != nullptr &&
       rpc_write(conn, pTexDesc, sizeof(const CUDA_TEXTURE_DESC)) < 0) ||
      rpc_write(conn, &pResViewDesc, sizeof(const CUDA_RESOURCE_VIEW_DESC *)) <
          0 ||
      (pResViewDesc != nullptr &&
       rpc_write(conn, pResViewDesc, sizeof(const CUDA_RESOURCE_VIEW_DESC)) <
           0) ||
      rpc_wait_for_response(conn) < 0 ||
      rpc_read(conn, pTexObject, sizeof(CUtexObject)) < 0 ||
      rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
      rpc_read_end(conn) < 0)
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  return return_value;
}

CUresult cuTexObjectDestroy(CUtexObject texObject) {
  lupine_route route = lupine_route_for_default();
  CUresult return_value;
  using real_fn_t = CUresult (*)(CUtexObject);
  if (lupine_call_local_cuda_if_routed<real_fn_t>(route, "cuTexObjectDestroy",
                                                  &return_value, texObject)) {
    return return_value;
  }
  conn_t *conn = lupine_route_remote_conn(route);
  if (conn == nullptr ||
      rpc_write_start_request(conn, RPC_cuTexObjectDestroy) < 0 ||
      rpc_write(conn, &texObject, sizeof(CUtexObject)) < 0 ||
      rpc_wait_for_response(conn) < 0 ||
      rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
      rpc_read_end(conn) < 0)
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  return return_value;
}

CUresult cuTexObjectGetResourceDesc(CUDA_RESOURCE_DESC *pResDesc,
                                    CUtexObject texObject) {
  lupine_route route = lupine_route_for_default();
  CUresult return_value;
  using real_fn_t = CUresult (*)(CUDA_RESOURCE_DESC *, CUtexObject);
  if (lupine_call_local_cuda_if_routed<real_fn_t>(
          route, "cuTexObjectGetResourceDesc", &return_value, pResDesc,
          texObject)) {
    return return_value;
  }
  conn_t *conn = lupine_route_remote_conn(route);
  if (conn == nullptr ||
      rpc_write_start_request(conn, RPC_cuTexObjectGetResourceDesc) < 0 ||
      rpc_write(conn, pResDesc, sizeof(CUDA_RESOURCE_DESC)) < 0 ||
      rpc_write(conn, &texObject, sizeof(CUtexObject)) < 0 ||
      rpc_wait_for_response(conn) < 0 ||
      rpc_read(conn, pResDesc, sizeof(CUDA_RESOURCE_DESC)) < 0 ||
      rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
      rpc_read_end(conn) < 0)
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  return return_value;
}

CUresult cuTexObjectGetTextureDesc(CUDA_TEXTURE_DESC *pTexDesc,
                                   CUtexObject texObject) {
  lupine_route route = lupine_route_for_default();
  CUresult return_value;
  using real_fn_t = CUresult (*)(CUDA_TEXTURE_DESC *, CUtexObject);
  if (lupine_call_local_cuda_if_routed<real_fn_t>(
          route, "cuTexObjectGetTextureDesc", &return_value, pTexDesc,
          texObject)) {
    return return_value;
  }
  conn_t *conn = lupine_route_remote_conn(route);
  if (conn == nullptr ||
      rpc_write_start_request(conn, RPC_cuTexObjectGetTextureDesc) < 0 ||
      rpc_write(conn, pTexDesc, sizeof(CUDA_TEXTURE_DESC)) < 0 ||
      rpc_write(conn, &texObject, sizeof(CUtexObject)) < 0 ||
      rpc_wait_for_response(conn) < 0 ||
      rpc_read(conn, pTexDesc, sizeof(CUDA_TEXTURE_DESC)) < 0 ||
      rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
      rpc_read_end(conn) < 0)
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  return return_value;
}

CUresult cuTexObjectGetResourceViewDesc(CUDA_RESOURCE_VIEW_DESC *pResViewDesc,
                                        CUtexObject texObject) {
  lupine_route route = lupine_route_for_default();
  CUresult return_value;
  using real_fn_t = CUresult (*)(CUDA_RESOURCE_VIEW_DESC *, CUtexObject);
  if (lupine_call_local_cuda_if_routed<real_fn_t>(
          route, "cuTexObjectGetResourceViewDesc", &return_value, pResViewDesc,
          texObject)) {
    return return_value;
  }
  conn_t *conn = lupine_route_remote_conn(route);
  if (conn == nullptr ||
      rpc_write_start_request(conn, RPC_cuTexObjectGetResourceViewDesc) < 0 ||
      rpc_write(conn, pResViewDesc, sizeof(CUDA_RESOURCE_VIEW_DESC)) < 0 ||
      rpc_write(conn, &texObject, sizeof(CUtexObject)) < 0 ||
      rpc_wait_for_response(conn) < 0 ||
      rpc_read(conn, pResViewDesc, sizeof(CUDA_RESOURCE_VIEW_DESC)) < 0 ||
      rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
      rpc_read_end(conn) < 0)
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  return return_value;
}

CUresult cuSurfObjectCreate(CUsurfObject *pSurfObject,
                            const CUDA_RESOURCE_DESC *pResDesc) {
  lupine_route route = lupine_route_for_default();
  CUresult return_value;
  using real_fn_t = CUresult (*)(CUsurfObject *, const CUDA_RESOURCE_DESC *);
  if (lupine_call_local_cuda_if_routed<real_fn_t>(
          route, "cuSurfObjectCreate", &return_value, pSurfObject, pResDesc)) {
    return return_value;
  }
  conn_t *conn = lupine_route_remote_conn(route);
  if (conn == nullptr ||
      rpc_write_start_request(conn, RPC_cuSurfObjectCreate) < 0 ||
      rpc_write(conn, pSurfObject, sizeof(CUsurfObject)) < 0 ||
      rpc_write(conn, pResDesc, sizeof(const CUDA_RESOURCE_DESC)) < 0 ||
      rpc_wait_for_response(conn) < 0 ||
      rpc_read(conn, pSurfObject, sizeof(CUsurfObject)) < 0 ||
      rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
      rpc_read_end(conn) < 0)
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  return return_value;
}

CUresult cuSurfObjectDestroy(CUsurfObject surfObject) {
  lupine_route route = lupine_route_for_default();
  CUresult return_value;
  using real_fn_t = CUresult (*)(CUsurfObject);
  if (lupine_call_local_cuda_if_routed<real_fn_t>(route, "cuSurfObjectDestroy",
                                                  &return_value, surfObject)) {
    return return_value;
  }
  conn_t *conn = lupine_route_remote_conn(route);
  if (conn == nullptr ||
      rpc_write_start_request(conn, RPC_cuSurfObjectDestroy) < 0 ||
      rpc_write(conn, &surfObject, sizeof(CUsurfObject)) < 0 ||
      rpc_wait_for_response(conn) < 0 ||
      rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
      rpc_read_end(conn) < 0)
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  return return_value;
}

CUresult cuSurfObjectGetResourceDesc(CUDA_RESOURCE_DESC *pResDesc,
                                     CUsurfObject surfObject) {
  lupine_route route = lupine_route_for_default();
  CUresult return_value;
  using real_fn_t = CUresult (*)(CUDA_RESOURCE_DESC *, CUsurfObject);
  if (lupine_call_local_cuda_if_routed<real_fn_t>(
          route, "cuSurfObjectGetResourceDesc", &return_value, pResDesc,
          surfObject)) {
    return return_value;
  }
  conn_t *conn = lupine_route_remote_conn(route);
  if (conn == nullptr ||
      rpc_write_start_request(conn, RPC_cuSurfObjectGetResourceDesc) < 0 ||
      rpc_write(conn, pResDesc, sizeof(CUDA_RESOURCE_DESC)) < 0 ||
      rpc_write(conn, &surfObject, sizeof(CUsurfObject)) < 0 ||
      rpc_wait_for_response(conn) < 0 ||
      rpc_read(conn, pResDesc, sizeof(CUDA_RESOURCE_DESC)) < 0 ||
      rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
      rpc_read_end(conn) < 0)
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  return return_value;
}

CUresult cuGraphicsUnregisterResource(CUgraphicsResource resource) {
  lupine_route route = lupine_route_for_default();
  CUresult return_value;
  using real_fn_t = CUresult (*)(CUgraphicsResource);
  if (lupine_call_local_cuda_if_routed<real_fn_t>(
          route, "cuGraphicsUnregisterResource", &return_value, resource)) {
    return return_value;
  }
  conn_t *conn = lupine_route_remote_conn(route);
  if (conn == nullptr ||
      rpc_write_start_request(conn, RPC_cuGraphicsUnregisterResource) < 0 ||
      rpc_write(conn, &resource, sizeof(CUgraphicsResource)) < 0 ||
      rpc_wait_for_response(conn) < 0 ||
      rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
      rpc_read_end(conn) < 0)
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  return return_value;
}

CUresult cuGraphicsSubResourceGetMappedArray(CUarray *pArray,
                                             CUgraphicsResource resource,
                                             unsigned int arrayIndex,
                                             unsigned int mipLevel) {
  lupine_route route = lupine_route_for_default();
  CUresult return_value;
  using real_fn_t =
      CUresult (*)(CUarray *, CUgraphicsResource, unsigned int, unsigned int);
  if (lupine_call_local_cuda_if_routed<real_fn_t>(
          route, "cuGraphicsSubResourceGetMappedArray", &return_value, pArray,
          resource, arrayIndex, mipLevel)) {
    return return_value;
  }
  conn_t *conn = lupine_route_remote_conn(route);
  if (conn == nullptr ||
      rpc_write_start_request(conn, RPC_cuGraphicsSubResourceGetMappedArray) <
          0 ||
      rpc_write(conn, pArray, sizeof(CUarray)) < 0 ||
      rpc_write(conn, &resource, sizeof(CUgraphicsResource)) < 0 ||
      rpc_write(conn, &arrayIndex, sizeof(unsigned int)) < 0 ||
      rpc_write(conn, &mipLevel, sizeof(unsigned int)) < 0 ||
      rpc_wait_for_response(conn) < 0 ||
      rpc_read(conn, pArray, sizeof(CUarray)) < 0 ||
      rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
      rpc_read_end(conn) < 0)
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  return return_value;
}

CUresult
cuGraphicsResourceGetMappedMipmappedArray(CUmipmappedArray *pMipmappedArray,
                                          CUgraphicsResource resource) {
  lupine_route route = lupine_route_for_default();
  CUresult return_value;
  using real_fn_t = CUresult (*)(CUmipmappedArray *, CUgraphicsResource);
  if (lupine_call_local_cuda_if_routed<real_fn_t>(
          route, "cuGraphicsResourceGetMappedMipmappedArray", &return_value,
          pMipmappedArray, resource)) {
    return return_value;
  }
  conn_t *conn = lupine_route_remote_conn(route);
  if (conn == nullptr ||
      rpc_write_start_request(
          conn, RPC_cuGraphicsResourceGetMappedMipmappedArray) < 0 ||
      rpc_write(conn, pMipmappedArray, sizeof(CUmipmappedArray)) < 0 ||
      rpc_write(conn, &resource, sizeof(CUgraphicsResource)) < 0 ||
      rpc_wait_for_response(conn) < 0 ||
      rpc_read(conn, pMipmappedArray, sizeof(CUmipmappedArray)) < 0 ||
      rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
      rpc_read_end(conn) < 0)
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  return return_value;
}

CUresult cuGraphicsResourceGetMappedPointer_v2(CUdeviceptr *pDevPtr,
                                               size_t *pSize,
                                               CUgraphicsResource resource) {
  lupine_route route = lupine_route_for_default();
  CUresult return_value;
  using real_fn_t = CUresult (*)(CUdeviceptr *, size_t *, CUgraphicsResource);
  if (lupine_call_local_cuda_if_routed<real_fn_t>(
          route, "cuGraphicsResourceGetMappedPointer_v2", &return_value,
          pDevPtr, pSize, resource)) {
    return return_value;
  }
  conn_t *conn = lupine_route_remote_conn(route);
  if (conn == nullptr ||
      rpc_write_start_request(conn, RPC_cuGraphicsResourceGetMappedPointer_v2) <
          0 ||
      rpc_write(conn, pDevPtr, sizeof(CUdeviceptr)) < 0 ||
      rpc_write(conn, pSize, sizeof(size_t)) < 0 ||
      rpc_write(conn, &resource, sizeof(CUgraphicsResource)) < 0 ||
      rpc_wait_for_response(conn) < 0 ||
      rpc_read(conn, pDevPtr, sizeof(CUdeviceptr)) < 0 ||
      rpc_read(conn, pSize, sizeof(size_t)) < 0 ||
      rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
      rpc_read_end(conn) < 0)
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  return return_value;
}

CUresult cuGraphicsResourceSetMapFlags_v2(CUgraphicsResource resource,
                                          unsigned int flags) {
  lupine_route route = lupine_route_for_default();
  CUresult return_value;
  using real_fn_t = CUresult (*)(CUgraphicsResource, unsigned int);
  if (lupine_call_local_cuda_if_routed<real_fn_t>(
          route, "cuGraphicsResourceSetMapFlags_v2", &return_value, resource,
          flags)) {
    return return_value;
  }
  conn_t *conn = lupine_route_remote_conn(route);
  if (conn == nullptr ||
      rpc_write_start_request(conn, RPC_cuGraphicsResourceSetMapFlags_v2) < 0 ||
      rpc_write(conn, &resource, sizeof(CUgraphicsResource)) < 0 ||
      rpc_write(conn, &flags, sizeof(unsigned int)) < 0 ||
      rpc_wait_for_response(conn) < 0 ||
      rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
      rpc_read_end(conn) < 0)
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  return return_value;
}

CUresult cuGraphicsMapResources(unsigned int count,
                                CUgraphicsResource *resources,
                                CUstream hStream) {
  lupine_route route = (hStream != nullptr ? lupine_route_for_stream(hStream)
                                           : lupine_route_for_default());
  CUresult return_value;
  using real_fn_t = CUresult (*)(unsigned int, CUgraphicsResource *, CUstream);
  if (lupine_call_local_cuda_if_routed<real_fn_t>(
          route, "cuGraphicsMapResources", &return_value, count, resources,
          hStream)) {
    return return_value;
  }
  conn_t *conn = lupine_route_remote_conn(route);
  if (count * sizeof(CUgraphicsResource) != 0 && resources == nullptr)
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  if (conn == nullptr ||
      rpc_write_start_request(conn, RPC_cuGraphicsMapResources) < 0 ||
      rpc_write(conn, &count, sizeof(unsigned int)) < 0 ||
      (count * sizeof(CUgraphicsResource) != 0 &&
       rpc_write(conn, resources, count * sizeof(CUgraphicsResource)) < 0) ||
      rpc_write(conn, &hStream, sizeof(CUstream)) < 0 ||
      rpc_wait_for_response(conn) < 0 ||
      rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
      rpc_read_end(conn) < 0)
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  return return_value;
}

CUresult cuGraphicsUnmapResources(unsigned int count,
                                  CUgraphicsResource *resources,
                                  CUstream hStream) {
  lupine_route route = (hStream != nullptr ? lupine_route_for_stream(hStream)
                                           : lupine_route_for_default());
  CUresult return_value;
  using real_fn_t = CUresult (*)(unsigned int, CUgraphicsResource *, CUstream);
  if (lupine_call_local_cuda_if_routed<real_fn_t>(
          route, "cuGraphicsUnmapResources", &return_value, count, resources,
          hStream)) {
    return return_value;
  }
  conn_t *conn = lupine_route_remote_conn(route);
  if (count * sizeof(CUgraphicsResource) != 0 && resources == nullptr)
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  if (conn == nullptr ||
      rpc_write_start_request(conn, RPC_cuGraphicsUnmapResources) < 0 ||
      rpc_write(conn, &count, sizeof(unsigned int)) < 0 ||
      (count * sizeof(CUgraphicsResource) != 0 &&
       rpc_write(conn, resources, count * sizeof(CUgraphicsResource)) < 0) ||
      rpc_write(conn, &hStream, sizeof(CUstream)) < 0 ||
      rpc_wait_for_response(conn) < 0 ||
      rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
      rpc_read_end(conn) < 0)
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  return return_value;
}

#ifdef cuDeviceTotalMem
#undef cuDeviceTotalMem
#endif
extern "C" CUresult cuDeviceTotalMem(size_t *bytes, CUdevice dev) {
  return cuDeviceTotalMem_v2(bytes, dev);
}

#ifdef cuDeviceGetUuid
#undef cuDeviceGetUuid
#endif
extern "C" CUresult cuDeviceGetUuid(CUuuid *uuid, CUdevice dev) {
  return cuDeviceGetUuid_v2(uuid, dev);
}

#ifdef cuDevicePrimaryCtxRelease
#undef cuDevicePrimaryCtxRelease
#endif
extern "C" CUresult cuDevicePrimaryCtxRelease(CUdevice dev) {
  return cuDevicePrimaryCtxRelease_v2(dev);
}

#ifdef cuDevicePrimaryCtxSetFlags
#undef cuDevicePrimaryCtxSetFlags
#endif
extern "C" CUresult cuDevicePrimaryCtxSetFlags(CUdevice dev,
                                               unsigned int flags) {
  return cuDevicePrimaryCtxSetFlags_v2(dev, flags);
}

#ifdef cuDevicePrimaryCtxReset
#undef cuDevicePrimaryCtxReset
#endif
extern "C" CUresult cuDevicePrimaryCtxReset(CUdevice dev) {
  return cuDevicePrimaryCtxReset_v2(dev);
}

#ifdef cuCtxDestroy
#undef cuCtxDestroy
#endif
extern "C" CUresult cuCtxDestroy(CUcontext ctx) { return cuCtxDestroy_v2(ctx); }

#ifdef cuCtxPopCurrent
#undef cuCtxPopCurrent
#endif
extern "C" CUresult cuCtxPopCurrent(CUcontext *pctx) {
  return cuCtxPopCurrent_v2(pctx);
}

#ifdef cuCtxPushCurrent
#undef cuCtxPushCurrent
#endif
extern "C" CUresult cuCtxPushCurrent(CUcontext ctx) {
  return cuCtxPushCurrent_v2(ctx);
}

#ifdef cuModuleGetGlobal
#undef cuModuleGetGlobal
#endif
extern "C" CUresult cuModuleGetGlobal(CUdeviceptr *dptr, size_t *bytes,
                                      CUmodule hmod, const char *name) {
  return cuModuleGetGlobal_v2(dptr, bytes, hmod, name);
}

#ifdef cuMemAlloc
#undef cuMemAlloc
#endif
extern "C" CUresult cuMemAlloc(CUdeviceptr *dptr, size_t bytesize) {
  return cuMemAlloc_v2(dptr, bytesize);
}

#ifdef cuMemAllocPitch
#undef cuMemAllocPitch
#endif
extern "C" CUresult cuMemAllocPitch(CUdeviceptr *dptr, size_t *pPitch,
                                    size_t WidthInBytes, size_t Height,
                                    unsigned int ElementSizeBytes) {
  return cuMemAllocPitch_v2(dptr, pPitch, WidthInBytes, Height,
                            ElementSizeBytes);
}

#ifdef cuMemcpyDtoD
#undef cuMemcpyDtoD
#endif
extern "C" CUresult cuMemcpyDtoD(CUdeviceptr dstDevice, CUdeviceptr srcDevice,
                                 size_t ByteCount) {
  return cuMemcpyDtoD_v2(dstDevice, srcDevice, ByteCount);
}

#ifdef cuMemcpyDtoDAsync
#undef cuMemcpyDtoDAsync
#endif
extern "C" CUresult cuMemcpyDtoDAsync(CUdeviceptr dstDevice,
                                      CUdeviceptr srcDevice, size_t ByteCount,
                                      CUstream hStream) {
  return cuMemcpyDtoDAsync_v2(dstDevice, srcDevice, ByteCount, hStream);
}

#ifdef cuMemsetD8
#undef cuMemsetD8
#endif
extern "C" CUresult cuMemsetD8(CUdeviceptr dstDevice, unsigned char uc,
                               size_t N) {
  return cuMemsetD8_v2(dstDevice, uc, N);
}

#ifdef cuMemsetD2D8
#undef cuMemsetD2D8
#endif
extern "C" CUresult cuMemsetD2D8(CUdeviceptr dstDevice, size_t dstPitch,
                                 unsigned char uc, size_t Width,
                                 size_t Height) {
  return cuMemsetD2D8_v2(dstDevice, dstPitch, uc, Width, Height);
}

#ifdef cuMemsetD2D16
#undef cuMemsetD2D16
#endif
extern "C" CUresult cuMemsetD2D16(CUdeviceptr dstDevice, size_t dstPitch,
                                  unsigned short us, size_t Width,
                                  size_t Height) {
  return cuMemsetD2D16_v2(dstDevice, dstPitch, us, Width, Height);
}

#ifdef cuMemsetD2D32
#undef cuMemsetD2D32
#endif
extern "C" CUresult cuMemsetD2D32(CUdeviceptr dstDevice, size_t dstPitch,
                                  unsigned int ui, size_t Width,
                                  size_t Height) {
  return cuMemsetD2D32_v2(dstDevice, dstPitch, ui, Width, Height);
}

#ifdef cuIpcOpenMemHandle
#undef cuIpcOpenMemHandle
#endif
extern "C" CUresult cuIpcOpenMemHandle(CUdeviceptr *pdptr,
                                       CUipcMemHandle handle,
                                       unsigned int Flags) {
  return cuIpcOpenMemHandle_v2(pdptr, handle, Flags);
}

#ifdef cuStreamBeginCapture
#undef cuStreamBeginCapture
#endif
extern "C" CUresult cuStreamBeginCapture(CUstream hStream,
                                         CUstreamCaptureMode mode) {
  return cuStreamBeginCapture_v2(hStream, mode);
}

#if CUDA_VERSION >= 12000
#ifdef cuGraphExecUpdate
#undef cuGraphExecUpdate
#endif
extern "C" CUresult cuGraphExecUpdate(CUgraphExec hGraphExec, CUgraph hGraph,
                                      CUgraphExecUpdateResultInfo *resultInfo) {
  return cuGraphExecUpdate_v2(hGraphExec, hGraph, resultInfo);
}

#endif

#ifdef cuMemcpy_ptds
#undef cuMemcpy_ptds
#endif
extern "C" CUresult cuMemcpy_ptds(CUdeviceptr dst, CUdeviceptr src,
                                  size_t ByteCount) {
  return cuMemcpy(dst, src, ByteCount);
}

#ifdef cuMemcpyPeer_ptds
#undef cuMemcpyPeer_ptds
#endif
extern "C" CUresult cuMemcpyPeer_ptds(CUdeviceptr dstDevice,
                                      CUcontext dstContext,
                                      CUdeviceptr srcDevice,
                                      CUcontext srcContext, size_t ByteCount) {
  return cuMemcpyPeer(dstDevice, dstContext, srcDevice, srcContext, ByteCount);
}

#ifdef cuMemcpyPeerAsync_ptsz
#undef cuMemcpyPeerAsync_ptsz
#endif
extern "C" CUresult cuMemcpyPeerAsync_ptsz(CUdeviceptr dstDevice,
                                           CUcontext dstContext,
                                           CUdeviceptr srcDevice,
                                           CUcontext srcContext,
                                           size_t ByteCount, CUstream hStream) {
  return cuMemcpyPeerAsync(dstDevice, dstContext, srcDevice, srcContext,
                           ByteCount, hStream);
}

#ifdef cuMemsetD8Async_ptsz
#undef cuMemsetD8Async_ptsz
#endif
extern "C" CUresult cuMemsetD8Async_ptsz(CUdeviceptr dstDevice,
                                         unsigned char uc, size_t N,
                                         CUstream hStream) {
  return cuMemsetD8Async(dstDevice, uc, N, hStream);
}

#ifdef cuMemsetD16Async_ptsz
#undef cuMemsetD16Async_ptsz
#endif
extern "C" CUresult cuMemsetD16Async_ptsz(CUdeviceptr dstDevice,
                                          unsigned short us, size_t N,
                                          CUstream hStream) {
  return cuMemsetD16Async(dstDevice, us, N, hStream);
}

#ifdef cuMemsetD32Async_ptsz
#undef cuMemsetD32Async_ptsz
#endif
extern "C" CUresult cuMemsetD32Async_ptsz(CUdeviceptr dstDevice,
                                          unsigned int ui, size_t N,
                                          CUstream hStream) {
  return cuMemsetD32Async(dstDevice, ui, N, hStream);
}

#ifdef cuMemsetD2D8Async_ptsz
#undef cuMemsetD2D8Async_ptsz
#endif
extern "C" CUresult cuMemsetD2D8Async_ptsz(CUdeviceptr dstDevice,
                                           size_t dstPitch, unsigned char uc,
                                           size_t Width, size_t Height,
                                           CUstream hStream) {
  return cuMemsetD2D8Async(dstDevice, dstPitch, uc, Width, Height, hStream);
}

#ifdef cuMemsetD2D16Async_ptsz
#undef cuMemsetD2D16Async_ptsz
#endif
extern "C" CUresult cuMemsetD2D16Async_ptsz(CUdeviceptr dstDevice,
                                            size_t dstPitch, unsigned short us,
                                            size_t Width, size_t Height,
                                            CUstream hStream) {
  return cuMemsetD2D16Async(dstDevice, dstPitch, us, Width, Height, hStream);
}

#ifdef cuMemsetD2D32Async_ptsz
#undef cuMemsetD2D32Async_ptsz
#endif
extern "C" CUresult cuMemsetD2D32Async_ptsz(CUdeviceptr dstDevice,
                                            size_t dstPitch, unsigned int ui,
                                            size_t Width, size_t Height,
                                            CUstream hStream) {
  return cuMemsetD2D32Async(dstDevice, dstPitch, ui, Width, Height, hStream);
}

#ifdef cuStreamGetPriority_ptsz
#undef cuStreamGetPriority_ptsz
#endif
extern "C" CUresult cuStreamGetPriority_ptsz(CUstream hStream, int *priority) {
  return cuStreamGetPriority(hStream, priority);
}

#ifdef cuStreamGetId_ptsz
#undef cuStreamGetId_ptsz
#endif
extern "C" CUresult cuStreamGetId_ptsz(CUstream hStream,
                                       unsigned long long *streamId) {
  return cuStreamGetId(hStream, streamId);
}

#ifdef cuStreamGetFlags_ptsz
#undef cuStreamGetFlags_ptsz
#endif
extern "C" CUresult cuStreamGetFlags_ptsz(CUstream hStream,
                                          unsigned int *flags) {
  return cuStreamGetFlags(hStream, flags);
}

#ifdef cuStreamGetCtx_ptsz
#undef cuStreamGetCtx_ptsz
#endif
extern "C" CUresult cuStreamGetCtx_ptsz(CUstream hStream, CUcontext *pctx) {
  return cuStreamGetCtx(hStream, pctx);
}

#ifdef cuStreamEndCapture_ptsz
#undef cuStreamEndCapture_ptsz
#endif
extern "C" CUresult cuStreamEndCapture_ptsz(CUstream hStream,
                                            CUgraph *phGraph) {
  return cuStreamEndCapture(hStream, phGraph);
}

#ifdef cuStreamIsCapturing_ptsz
#undef cuStreamIsCapturing_ptsz
#endif
extern "C" CUresult
cuStreamIsCapturing_ptsz(CUstream hStream,
                         CUstreamCaptureStatus *captureStatus) {
  return cuStreamIsCapturing(hStream, captureStatus);
}

#ifdef cuStreamAttachMemAsync_ptsz
#undef cuStreamAttachMemAsync_ptsz
#endif
extern "C" CUresult cuStreamAttachMemAsync_ptsz(CUstream hStream,
                                                CUdeviceptr dptr, size_t length,
                                                unsigned int flags) {
  return cuStreamAttachMemAsync(hStream, dptr, length, flags);
}

#ifdef cuStreamQuery_ptsz
#undef cuStreamQuery_ptsz
#endif
extern "C" CUresult cuStreamQuery_ptsz(CUstream hStream) {
  return cuStreamQuery(hStream);
}

#ifdef cuStreamSynchronize_ptsz
#undef cuStreamSynchronize_ptsz
#endif
extern "C" CUresult cuStreamSynchronize_ptsz(CUstream hStream) {
  return cuStreamSynchronize(hStream);
}

#ifdef cuEventRecord_ptsz
#undef cuEventRecord_ptsz
#endif
extern "C" CUresult cuEventRecord_ptsz(CUevent hEvent, CUstream hStream) {
  return cuEventRecord(hEvent, hStream);
}

#ifdef cuEventRecordWithFlags_ptsz
#undef cuEventRecordWithFlags_ptsz
#endif
extern "C" CUresult cuEventRecordWithFlags_ptsz(CUevent hEvent,
                                                CUstream hStream,
                                                unsigned int flags) {
  return cuEventRecordWithFlags(hEvent, hStream, flags);
}

#ifdef cuGraphicsMapResources_ptsz
#undef cuGraphicsMapResources_ptsz
#endif
extern "C" CUresult cuGraphicsMapResources_ptsz(unsigned int count,
                                                CUgraphicsResource *resources,
                                                CUstream hStream) {
  return cuGraphicsMapResources(count, resources, hStream);
}

#ifdef cuGraphicsUnmapResources_ptsz
#undef cuGraphicsUnmapResources_ptsz
#endif
extern "C" CUresult cuGraphicsUnmapResources_ptsz(unsigned int count,
                                                  CUgraphicsResource *resources,
                                                  CUstream hStream) {
  return cuGraphicsUnmapResources(count, resources, hStream);
}

#ifdef cuSignalExternalSemaphoresAsync_ptsz
#undef cuSignalExternalSemaphoresAsync_ptsz
#endif
extern "C" CUresult cuSignalExternalSemaphoresAsync_ptsz(
    const CUexternalSemaphore *extSemArray,
    const CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS *paramsArray,
    unsigned int numExtSems, CUstream stream) {
  return cuSignalExternalSemaphoresAsync(extSemArray, paramsArray, numExtSems,
                                         stream);
}

#ifdef cuWaitExternalSemaphoresAsync_ptsz
#undef cuWaitExternalSemaphoresAsync_ptsz
#endif
extern "C" CUresult cuWaitExternalSemaphoresAsync_ptsz(
    const CUexternalSemaphore *extSemArray,
    const CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS *paramsArray,
    unsigned int numExtSems, CUstream stream) {
  return cuWaitExternalSemaphoresAsync(extSemArray, paramsArray, numExtSems,
                                       stream);
}

#ifdef cuGraphInstantiateWithParams_ptsz
#undef cuGraphInstantiateWithParams_ptsz
#endif
extern "C" CUresult cuGraphInstantiateWithParams_ptsz(
    CUgraphExec *phGraphExec, CUgraph hGraph,
    CUDA_GRAPH_INSTANTIATE_PARAMS *instantiateParams) {
  return cuGraphInstantiateWithParams(phGraphExec, hGraph, instantiateParams);
}

#ifdef cuGraphUpload_ptsz
#undef cuGraphUpload_ptsz
#endif
extern "C" CUresult cuGraphUpload_ptsz(CUgraphExec hGraphExec,
                                       CUstream hStream) {
  return cuGraphUpload(hGraphExec, hStream);
}

#ifdef cuGraphLaunch_ptsz
#undef cuGraphLaunch_ptsz
#endif
extern "C" CUresult cuGraphLaunch_ptsz(CUgraphExec hGraphExec,
                                       CUstream hStream) {
  return cuGraphLaunch(hGraphExec, hStream);
}

#ifdef cuStreamCopyAttributes_ptsz
#undef cuStreamCopyAttributes_ptsz
#endif
extern "C" CUresult cuStreamCopyAttributes_ptsz(CUstream dst, CUstream src) {
  return cuStreamCopyAttributes(dst, src);
}

#ifdef cuStreamGetAttribute_ptsz
#undef cuStreamGetAttribute_ptsz
#endif
extern "C" CUresult cuStreamGetAttribute_ptsz(CUstream hStream,
                                              CUstreamAttrID attr,
                                              CUstreamAttrValue *value_out) {
  return cuStreamGetAttribute(hStream, attr, value_out);
}

#ifdef cuStreamSetAttribute_ptsz
#undef cuStreamSetAttribute_ptsz
#endif
extern "C" CUresult cuStreamSetAttribute_ptsz(CUstream hStream,
                                              CUstreamAttrID attr,
                                              const CUstreamAttrValue *value) {
  return cuStreamSetAttribute(hStream, attr, value);
}

#ifdef cuMemMapArrayAsync_ptsz
#undef cuMemMapArrayAsync_ptsz
#endif
extern "C" CUresult cuMemMapArrayAsync_ptsz(CUarrayMapInfo *mapInfoList,
                                            unsigned int count,
                                            CUstream hStream) {
  return cuMemMapArrayAsync(mapInfoList, count, hStream);
}

#ifdef cuMemFreeAsync_ptsz
#undef cuMemFreeAsync_ptsz
#endif
extern "C" CUresult cuMemFreeAsync_ptsz(CUdeviceptr dptr, CUstream hStream) {
  return cuMemFreeAsync(dptr, hStream);
}

#ifdef cuMemAllocAsync_ptsz
#undef cuMemAllocAsync_ptsz
#endif
extern "C" CUresult cuMemAllocAsync_ptsz(CUdeviceptr *dptr, size_t bytesize,
                                         CUstream hStream) {
  return cuMemAllocAsync(dptr, bytesize, hStream);
}

#ifdef cuMemAllocFromPoolAsync_ptsz
#undef cuMemAllocFromPoolAsync_ptsz
#endif
extern "C" CUresult cuMemAllocFromPoolAsync_ptsz(CUdeviceptr *dptr,
                                                 size_t bytesize,
                                                 CUmemoryPool pool,
                                                 CUstream hStream) {
  return cuMemAllocFromPoolAsync(dptr, bytesize, pool, hStream);
}

std::unordered_map<std::string, void *> functionMap = {
    {"cuInit", (void *)cuInit},
    {"cuDriverGetVersion", (void *)cuDriverGetVersion},
    {"cuDeviceGet", (void *)cuDeviceGet},
    {"cuDeviceGetCount", (void *)cuDeviceGetCount},
    {"cuDeviceGetName", (void *)cuDeviceGetName},
    {"cuDeviceGetUuid_v2", (void *)cuDeviceGetUuid_v2},
    {"cuDeviceGetLuid", (void *)cuDeviceGetLuid},
    {"cuDeviceTotalMem_v2", (void *)cuDeviceTotalMem_v2},
    {"cuDeviceGetTexture1DLinearMaxWidth",
     (void *)cuDeviceGetTexture1DLinearMaxWidth},
    {"cuDeviceGetAttribute", (void *)cuDeviceGetAttribute},
    {"cuDeviceSetMemPool", (void *)cuDeviceSetMemPool},
    {"cuDeviceGetMemPool", (void *)cuDeviceGetMemPool},
    {"cuDeviceGetDefaultMemPool", (void *)cuDeviceGetDefaultMemPool},
    {"cuDeviceGetExecAffinitySupport", (void *)cuDeviceGetExecAffinitySupport},
    {"cuFlushGPUDirectRDMAWrites", (void *)cuFlushGPUDirectRDMAWrites},
    {"cuDeviceGetProperties", (void *)cuDeviceGetProperties},
    {"cuDeviceComputeCapability", (void *)cuDeviceComputeCapability},
    {"cuDevicePrimaryCtxRetain", (void *)cuDevicePrimaryCtxRetain},
    {"cuDevicePrimaryCtxRelease_v2", (void *)cuDevicePrimaryCtxRelease_v2},
    {"cuDevicePrimaryCtxSetFlags_v2", (void *)cuDevicePrimaryCtxSetFlags_v2},
    {"cuDevicePrimaryCtxGetState", (void *)cuDevicePrimaryCtxGetState},
    {"cuDevicePrimaryCtxReset_v2", (void *)cuDevicePrimaryCtxReset_v2},
    {"cuCtxDestroy_v2", (void *)cuCtxDestroy_v2},
    {"cuCtxPushCurrent_v2", (void *)cuCtxPushCurrent_v2},
    {"cuCtxPopCurrent_v2", (void *)cuCtxPopCurrent_v2},
    {"cuCtxSetCurrent", (void *)cuCtxSetCurrent},
    {"cuCtxGetCurrent", (void *)cuCtxGetCurrent},
    {"cuCtxGetDevice", (void *)cuCtxGetDevice},
    {"cuCtxGetFlags", (void *)cuCtxGetFlags},
    {"cuCtxGetId", (void *)cuCtxGetId},
    {"cuCtxSynchronize", (void *)cuCtxSynchronize},
    {"cuCtxSetLimit", (void *)cuCtxSetLimit},
    {"cuCtxGetLimit", (void *)cuCtxGetLimit},
    {"cuCtxGetCacheConfig", (void *)cuCtxGetCacheConfig},
    {"cuCtxSetCacheConfig", (void *)cuCtxSetCacheConfig},
    {"cuCtxGetApiVersion", (void *)cuCtxGetApiVersion},
    {"cuCtxGetStreamPriorityRange", (void *)cuCtxGetStreamPriorityRange},
    {"cuCtxResetPersistingL2Cache", (void *)cuCtxResetPersistingL2Cache},
    {"cuCtxGetExecAffinity", (void *)cuCtxGetExecAffinity},
    {"cuCtxAttach", (void *)cuCtxAttach},
    {"cuCtxDetach", (void *)cuCtxDetach},
    {"cuCtxGetSharedMemConfig", (void *)cuCtxGetSharedMemConfig},
    {"cuCtxSetSharedMemConfig", (void *)cuCtxSetSharedMemConfig},
    {"cuModuleUnload", (void *)cuModuleUnload},
    {"cuModuleGetLoadingMode", (void *)cuModuleGetLoadingMode},
    {"cuModuleGetFunction", (void *)cuModuleGetFunction},
    {"cuModuleGetGlobal_v2", (void *)cuModuleGetGlobal_v2},
    {"cuLinkCreate_v2", (void *)cuLinkCreate_v2},
    {"cuLinkAddData_v2", (void *)cuLinkAddData_v2},
    {"cuLinkAddFile_v2", (void *)cuLinkAddFile_v2},
    {"cuLinkComplete", (void *)cuLinkComplete},
    {"cuLinkDestroy", (void *)cuLinkDestroy},
    {"cuModuleGetTexRef", (void *)cuModuleGetTexRef},
    {"cuModuleGetSurfRef", (void *)cuModuleGetSurfRef},
    {"cuLibraryLoadFromFile", (void *)cuLibraryLoadFromFile},
    {"cuLibraryUnload", (void *)cuLibraryUnload},
    {"cuLibraryGetKernel", (void *)cuLibraryGetKernel},
    {"cuLibraryGetModule", (void *)cuLibraryGetModule},
    {"cuKernelGetFunction", (void *)cuKernelGetFunction},
    {"cuLibraryGetGlobal", (void *)cuLibraryGetGlobal},
    {"cuLibraryGetManaged", (void *)cuLibraryGetManaged},
    {"cuLibraryGetUnifiedFunction", (void *)cuLibraryGetUnifiedFunction},
    {"cuKernelGetAttribute", (void *)cuKernelGetAttribute},
    {"cuKernelSetAttribute", (void *)cuKernelSetAttribute},
    {"cuKernelSetCacheConfig", (void *)cuKernelSetCacheConfig},
    {"cuMemGetInfo_v2", (void *)cuMemGetInfo_v2},
    {"cuMemAlloc_v2", (void *)cuMemAlloc_v2},
    {"cuMemAllocPitch_v2", (void *)cuMemAllocPitch_v2},
    {"cuMemFree_v2", (void *)cuMemFree_v2},
    {"cuMemGetAddressRange_v2", (void *)cuMemGetAddressRange_v2},
    {"cuMemAllocHost_v2", (void *)cuMemAllocHost_v2},
    {"cuMemFreeHost", (void *)cuMemFreeHost},
    {"cuMemHostAlloc", (void *)cuMemHostAlloc},
    {"cuMemHostGetDevicePointer_v2", (void *)cuMemHostGetDevicePointer_v2},
    {"cuMemAllocManaged", (void *)cuMemAllocManaged},
    {"cuDeviceGetByPCIBusId", (void *)cuDeviceGetByPCIBusId},
    {"cuDeviceGetPCIBusId", (void *)cuDeviceGetPCIBusId},
    {"cuIpcGetEventHandle", (void *)cuIpcGetEventHandle},
    {"cuIpcOpenEventHandle", (void *)cuIpcOpenEventHandle},
    {"cuIpcGetMemHandle", (void *)cuIpcGetMemHandle},
    {"cuIpcOpenMemHandle_v2", (void *)cuIpcOpenMemHandle_v2},
    {"cuIpcCloseMemHandle", (void *)cuIpcCloseMemHandle},
    {"cuMemcpy", (void *)cuMemcpy},
    {"cuMemcpyPeer", (void *)cuMemcpyPeer},
    {"cuMemcpyDtoD_v2", (void *)cuMemcpyDtoD_v2},
    {"cuMemcpyDtoA_v2", (void *)cuMemcpyDtoA_v2},
    {"cuMemcpyAtoD_v2", (void *)cuMemcpyAtoD_v2},
    {"cuMemcpyAtoA_v2", (void *)cuMemcpyAtoA_v2},
    {"cuMemcpyPeerAsync", (void *)cuMemcpyPeerAsync},
    {"cuMemcpyDtoDAsync_v2", (void *)cuMemcpyDtoDAsync_v2},
    {"cuMemsetD8_v2", (void *)cuMemsetD8_v2},
    {"cuMemsetD16_v2", (void *)cuMemsetD16_v2},
    {"cuMemsetD32_v2", (void *)cuMemsetD32_v2},
    {"cuMemsetD2D8_v2", (void *)cuMemsetD2D8_v2},
    {"cuMemsetD2D16_v2", (void *)cuMemsetD2D16_v2},
    {"cuMemsetD2D32_v2", (void *)cuMemsetD2D32_v2},
    {"cuMemsetD8Async", (void *)cuMemsetD8Async},
    {"cuMemsetD16Async", (void *)cuMemsetD16Async},
    {"cuMemsetD32Async", (void *)cuMemsetD32Async},
    {"cuMemsetD2D8Async", (void *)cuMemsetD2D8Async},
    {"cuMemsetD2D16Async", (void *)cuMemsetD2D16Async},
    {"cuMemsetD2D32Async", (void *)cuMemsetD2D32Async},
    {"cuArrayCreate_v2", (void *)cuArrayCreate_v2},
    {"cuArrayGetDescriptor_v2", (void *)cuArrayGetDescriptor_v2},
    {"cuArrayGetSparseProperties", (void *)cuArrayGetSparseProperties},
    {"cuMipmappedArrayGetSparseProperties",
     (void *)cuMipmappedArrayGetSparseProperties},
    {"cuArrayGetMemoryRequirements", (void *)cuArrayGetMemoryRequirements},
    {"cuMipmappedArrayGetMemoryRequirements",
     (void *)cuMipmappedArrayGetMemoryRequirements},
    {"cuArrayGetPlane", (void *)cuArrayGetPlane},
    {"cuArrayDestroy", (void *)cuArrayDestroy},
    {"cuArray3DCreate_v2", (void *)cuArray3DCreate_v2},
    {"cuArray3DGetDescriptor_v2", (void *)cuArray3DGetDescriptor_v2},
    {"cuMipmappedArrayCreate", (void *)cuMipmappedArrayCreate},
    {"cuMipmappedArrayGetLevel", (void *)cuMipmappedArrayGetLevel},
    {"cuMipmappedArrayDestroy", (void *)cuMipmappedArrayDestroy},
    {"cuMemAddressReserve", (void *)cuMemAddressReserve},
    {"cuMemAddressFree", (void *)cuMemAddressFree},
    {"cuMemCreate", (void *)cuMemCreate},
    {"cuMemRelease", (void *)cuMemRelease},
    {"cuMemMap", (void *)cuMemMap},
    {"cuMemMapArrayAsync", (void *)cuMemMapArrayAsync},
    {"cuMemUnmap", (void *)cuMemUnmap},
    {"cuMemSetAccess", (void *)cuMemSetAccess},
    {"cuMemGetAccess", (void *)cuMemGetAccess},
    {"cuMemGetAllocationGranularity", (void *)cuMemGetAllocationGranularity},
    {"cuMemGetAllocationPropertiesFromHandle",
     (void *)cuMemGetAllocationPropertiesFromHandle},
    {"cuMemFreeAsync", (void *)cuMemFreeAsync},
    {"cuMemAllocAsync", (void *)cuMemAllocAsync},
    {"cuMemPoolTrimTo", (void *)cuMemPoolTrimTo},
    {"cuMemPoolSetAccess", (void *)cuMemPoolSetAccess},
    {"cuMemPoolGetAccess", (void *)cuMemPoolGetAccess},
    {"cuMemPoolCreate", (void *)cuMemPoolCreate},
    {"cuMemPoolDestroy", (void *)cuMemPoolDestroy},
    {"cuMemAllocFromPoolAsync", (void *)cuMemAllocFromPoolAsync},
    {"cuMemPoolExportPointer", (void *)cuMemPoolExportPointer},
    {"cuMemPoolImportPointer", (void *)cuMemPoolImportPointer},
    {"cuMemRangeGetAttributes", (void *)cuMemRangeGetAttributes},
    {"cuPointerSetAttribute", (void *)cuPointerSetAttribute},
    {"cuPointerGetAttributes", (void *)cuPointerGetAttributes},
    {"cuStreamCreate", (void *)cuStreamCreate},
    {"cuStreamCreateWithPriority", (void *)cuStreamCreateWithPriority},
    {"cuStreamGetPriority", (void *)cuStreamGetPriority},
    {"cuStreamGetFlags", (void *)cuStreamGetFlags},
    {"cuStreamGetId", (void *)cuStreamGetId},
    {"cuStreamGetCtx", (void *)cuStreamGetCtx},
    {"cuStreamBeginCapture_v2", (void *)cuStreamBeginCapture_v2},
    {"cuThreadExchangeStreamCaptureMode",
     (void *)cuThreadExchangeStreamCaptureMode},
    {"cuStreamEndCapture", (void *)cuStreamEndCapture},
    {"cuStreamIsCapturing", (void *)cuStreamIsCapturing},
    {"cuStreamAttachMemAsync", (void *)cuStreamAttachMemAsync},
    {"cuStreamQuery", (void *)cuStreamQuery},
    {"cuStreamSynchronize", (void *)cuStreamSynchronize},
    {"cuStreamDestroy_v2", (void *)cuStreamDestroy_v2},
    {"cuStreamCopyAttributes", (void *)cuStreamCopyAttributes},
    {"cuStreamGetAttribute", (void *)cuStreamGetAttribute},
    {"cuStreamSetAttribute", (void *)cuStreamSetAttribute},
    {"cuEventCreate", (void *)cuEventCreate},
    {"cuEventRecord", (void *)cuEventRecord},
    {"cuEventRecordWithFlags", (void *)cuEventRecordWithFlags},
    {"cuEventQuery", (void *)cuEventQuery},
    {"cuEventSynchronize", (void *)cuEventSynchronize},
    {"cuEventDestroy_v2", (void *)cuEventDestroy_v2},
    {"cuEventElapsedTime_v2", (void *)cuEventElapsedTime_v2},
    {"cuImportExternalMemory", (void *)cuImportExternalMemory},
    {"cuExternalMemoryGetMappedBuffer",
     (void *)cuExternalMemoryGetMappedBuffer},
    {"cuExternalMemoryGetMappedMipmappedArray",
     (void *)cuExternalMemoryGetMappedMipmappedArray},
    {"cuDestroyExternalMemory", (void *)cuDestroyExternalMemory},
    {"cuImportExternalSemaphore", (void *)cuImportExternalSemaphore},
    {"cuSignalExternalSemaphoresAsync",
     (void *)cuSignalExternalSemaphoresAsync},
    {"cuWaitExternalSemaphoresAsync", (void *)cuWaitExternalSemaphoresAsync},
    {"cuDestroyExternalSemaphore", (void *)cuDestroyExternalSemaphore},
    {"cuStreamWaitValue32_v2", (void *)cuStreamWaitValue32_v2},
    {"cuStreamWaitValue64_v2", (void *)cuStreamWaitValue64_v2},
    {"cuStreamWriteValue32_v2", (void *)cuStreamWriteValue32_v2},
    {"cuStreamWriteValue64_v2", (void *)cuStreamWriteValue64_v2},
    {"cuStreamBatchMemOp_v2", (void *)cuStreamBatchMemOp_v2},
    {"cuFuncGetAttribute", (void *)cuFuncGetAttribute},
    {"cuFuncSetAttribute", (void *)cuFuncSetAttribute},
    {"cuFuncSetCacheConfig", (void *)cuFuncSetCacheConfig},
    {"cuFuncGetModule", (void *)cuFuncGetModule},
    {"cuLaunchCooperativeKernel", (void *)cuLaunchCooperativeKernel},
    {"cuLaunchCooperativeKernelMultiDevice",
     (void *)cuLaunchCooperativeKernelMultiDevice},
    {"cuFuncSetBlockShape", (void *)cuFuncSetBlockShape},
    {"cuFuncSetSharedSize", (void *)cuFuncSetSharedSize},
    {"cuParamSetSize", (void *)cuParamSetSize},
    {"cuParamSeti", (void *)cuParamSeti},
    {"cuParamSetf", (void *)cuParamSetf},
    {"cuLaunch", (void *)cuLaunch},
    {"cuLaunchGrid", (void *)cuLaunchGrid},
    {"cuLaunchGridAsync", (void *)cuLaunchGridAsync},
    {"cuParamSetTexRef", (void *)cuParamSetTexRef},
    {"cuFuncSetSharedMemConfig", (void *)cuFuncSetSharedMemConfig},
    {"cuGraphCreate", (void *)cuGraphCreate},
    {"cuGraphMemcpyNodeGetParams", (void *)cuGraphMemcpyNodeGetParams},
    {"cuGraphMemcpyNodeSetParams", (void *)cuGraphMemcpyNodeSetParams},
    {"cuGraphMemsetNodeGetParams", (void *)cuGraphMemsetNodeGetParams},
    {"cuGraphMemsetNodeSetParams", (void *)cuGraphMemsetNodeSetParams},
    {"cuGraphAddChildGraphNode", (void *)cuGraphAddChildGraphNode},
    {"cuGraphChildGraphNodeGetGraph", (void *)cuGraphChildGraphNodeGetGraph},
    {"cuGraphAddEmptyNode", (void *)cuGraphAddEmptyNode},
    {"cuGraphAddEventRecordNode", (void *)cuGraphAddEventRecordNode},
    {"cuGraphEventRecordNodeGetEvent", (void *)cuGraphEventRecordNodeGetEvent},
    {"cuGraphEventRecordNodeSetEvent", (void *)cuGraphEventRecordNodeSetEvent},
    {"cuGraphAddEventWaitNode", (void *)cuGraphAddEventWaitNode},
    {"cuGraphEventWaitNodeGetEvent", (void *)cuGraphEventWaitNodeGetEvent},
    {"cuGraphEventWaitNodeSetEvent", (void *)cuGraphEventWaitNodeSetEvent},
    {"cuGraphAddExternalSemaphoresSignalNode",
     (void *)cuGraphAddExternalSemaphoresSignalNode},
    {"cuGraphExternalSemaphoresSignalNodeGetParams",
     (void *)cuGraphExternalSemaphoresSignalNodeGetParams},
    {"cuGraphExternalSemaphoresSignalNodeSetParams",
     (void *)cuGraphExternalSemaphoresSignalNodeSetParams},
    {"cuGraphAddExternalSemaphoresWaitNode",
     (void *)cuGraphAddExternalSemaphoresWaitNode},
    {"cuGraphExternalSemaphoresWaitNodeGetParams",
     (void *)cuGraphExternalSemaphoresWaitNodeGetParams},
    {"cuGraphExternalSemaphoresWaitNodeSetParams",
     (void *)cuGraphExternalSemaphoresWaitNodeSetParams},
    {"cuGraphAddBatchMemOpNode", (void *)cuGraphAddBatchMemOpNode},
    {"cuGraphBatchMemOpNodeGetParams", (void *)cuGraphBatchMemOpNodeGetParams},
    {"cuGraphBatchMemOpNodeSetParams", (void *)cuGraphBatchMemOpNodeSetParams},
    {"cuGraphExecBatchMemOpNodeSetParams",
     (void *)cuGraphExecBatchMemOpNodeSetParams},
    {"cuGraphAddMemAllocNode", (void *)cuGraphAddMemAllocNode},
    {"cuGraphMemAllocNodeGetParams", (void *)cuGraphMemAllocNodeGetParams},
    {"cuGraphAddMemFreeNode", (void *)cuGraphAddMemFreeNode},
    {"cuGraphMemFreeNodeGetParams", (void *)cuGraphMemFreeNodeGetParams},
    {"cuDeviceGraphMemTrim", (void *)cuDeviceGraphMemTrim},
    {"cuGraphClone", (void *)cuGraphClone},
    {"cuGraphNodeFindInClone", (void *)cuGraphNodeFindInClone},
    {"cuGraphNodeGetType", (void *)cuGraphNodeGetType},
    {"cuGraphGetNodes", (void *)cuGraphGetNodes},
    {"cuGraphGetRootNodes", (void *)cuGraphGetRootNodes},
    {"cuGraphDestroyNode", (void *)cuGraphDestroyNode},
    {"cuGraphInstantiateWithFlags", (void *)cuGraphInstantiateWithFlags},
    {"cuGraphInstantiateWithParams", (void *)cuGraphInstantiateWithParams},
    {"cuGraphExecGetFlags", (void *)cuGraphExecGetFlags},
    {"cuGraphExecMemcpyNodeSetParams", (void *)cuGraphExecMemcpyNodeSetParams},
    {"cuGraphExecMemsetNodeSetParams", (void *)cuGraphExecMemsetNodeSetParams},
    {"cuGraphExecChildGraphNodeSetParams",
     (void *)cuGraphExecChildGraphNodeSetParams},
    {"cuGraphExecEventRecordNodeSetEvent",
     (void *)cuGraphExecEventRecordNodeSetEvent},
    {"cuGraphExecEventWaitNodeSetEvent",
     (void *)cuGraphExecEventWaitNodeSetEvent},
    {"cuGraphExecExternalSemaphoresSignalNodeSetParams",
     (void *)cuGraphExecExternalSemaphoresSignalNodeSetParams},
    {"cuGraphExecExternalSemaphoresWaitNodeSetParams",
     (void *)cuGraphExecExternalSemaphoresWaitNodeSetParams},
    {"cuGraphNodeSetEnabled", (void *)cuGraphNodeSetEnabled},
    {"cuGraphNodeGetEnabled", (void *)cuGraphNodeGetEnabled},
    {"cuGraphUpload", (void *)cuGraphUpload},
    {"cuGraphLaunch", (void *)cuGraphLaunch},
    {"cuGraphExecDestroy", (void *)cuGraphExecDestroy},
    {"cuGraphDestroy", (void *)cuGraphDestroy},
    {"cuGraphExecUpdate_v2", (void *)cuGraphExecUpdate_v2},
    {"cuGraphKernelNodeCopyAttributes",
     (void *)cuGraphKernelNodeCopyAttributes},
    {"cuGraphKernelNodeGetAttribute", (void *)cuGraphKernelNodeGetAttribute},
    {"cuGraphKernelNodeSetAttribute", (void *)cuGraphKernelNodeSetAttribute},
    {"cuGraphDebugDotPrint", (void *)cuGraphDebugDotPrint},
    {"cuUserObjectRetain", (void *)cuUserObjectRetain},
    {"cuUserObjectRelease", (void *)cuUserObjectRelease},
    {"cuGraphRetainUserObject", (void *)cuGraphRetainUserObject},
    {"cuGraphReleaseUserObject", (void *)cuGraphReleaseUserObject},
    {"cuOccupancyMaxActiveBlocksPerMultiprocessor",
     (void *)cuOccupancyMaxActiveBlocksPerMultiprocessor},
    {"cuOccupancyMaxActiveBlocksPerMultiprocessorWithFlags",
     (void *)cuOccupancyMaxActiveBlocksPerMultiprocessorWithFlags},
    {"cuOccupancyAvailableDynamicSMemPerBlock",
     (void *)cuOccupancyAvailableDynamicSMemPerBlock},
    {"cuOccupancyMaxPotentialClusterSize",
     (void *)cuOccupancyMaxPotentialClusterSize},
    {"cuOccupancyMaxActiveClusters", (void *)cuOccupancyMaxActiveClusters},
    {"cuTexRefSetArray", (void *)cuTexRefSetArray},
    {"cuTexRefSetMipmappedArray", (void *)cuTexRefSetMipmappedArray},
    {"cuTexRefSetAddress_v2", (void *)cuTexRefSetAddress_v2},
    {"cuTexRefSetAddress2D_v3", (void *)cuTexRefSetAddress2D_v3},
    {"cuTexRefSetFormat", (void *)cuTexRefSetFormat},
    {"cuTexRefSetAddressMode", (void *)cuTexRefSetAddressMode},
    {"cuTexRefSetFilterMode", (void *)cuTexRefSetFilterMode},
    {"cuTexRefSetMipmapFilterMode", (void *)cuTexRefSetMipmapFilterMode},
    {"cuTexRefSetMipmapLevelBias", (void *)cuTexRefSetMipmapLevelBias},
    {"cuTexRefSetMipmapLevelClamp", (void *)cuTexRefSetMipmapLevelClamp},
    {"cuTexRefSetMaxAnisotropy", (void *)cuTexRefSetMaxAnisotropy},
    {"cuTexRefSetBorderColor", (void *)cuTexRefSetBorderColor},
    {"cuTexRefSetFlags", (void *)cuTexRefSetFlags},
    {"cuTexRefGetAddress_v2", (void *)cuTexRefGetAddress_v2},
    {"cuTexRefGetArray", (void *)cuTexRefGetArray},
    {"cuTexRefGetMipmappedArray", (void *)cuTexRefGetMipmappedArray},
    {"cuTexRefGetAddressMode", (void *)cuTexRefGetAddressMode},
    {"cuTexRefGetFilterMode", (void *)cuTexRefGetFilterMode},
    {"cuTexRefGetFormat", (void *)cuTexRefGetFormat},
    {"cuTexRefGetMipmapFilterMode", (void *)cuTexRefGetMipmapFilterMode},
    {"cuTexRefGetMipmapLevelBias", (void *)cuTexRefGetMipmapLevelBias},
    {"cuTexRefGetMipmapLevelClamp", (void *)cuTexRefGetMipmapLevelClamp},
    {"cuTexRefGetMaxAnisotropy", (void *)cuTexRefGetMaxAnisotropy},
    {"cuTexRefGetBorderColor", (void *)cuTexRefGetBorderColor},
    {"cuTexRefGetFlags", (void *)cuTexRefGetFlags},
    {"cuTexRefCreate", (void *)cuTexRefCreate},
    {"cuTexRefDestroy", (void *)cuTexRefDestroy},
    {"cuSurfRefSetArray", (void *)cuSurfRefSetArray},
    {"cuSurfRefGetArray", (void *)cuSurfRefGetArray},
    {"cuTexObjectCreate", (void *)cuTexObjectCreate},
    {"cuTexObjectDestroy", (void *)cuTexObjectDestroy},
    {"cuTexObjectGetResourceDesc", (void *)cuTexObjectGetResourceDesc},
    {"cuTexObjectGetTextureDesc", (void *)cuTexObjectGetTextureDesc},
    {"cuTexObjectGetResourceViewDesc", (void *)cuTexObjectGetResourceViewDesc},
    {"cuSurfObjectCreate", (void *)cuSurfObjectCreate},
    {"cuSurfObjectDestroy", (void *)cuSurfObjectDestroy},
    {"cuSurfObjectGetResourceDesc", (void *)cuSurfObjectGetResourceDesc},
    {"cuDeviceCanAccessPeer", (void *)cuDeviceCanAccessPeer},
    {"cuCtxEnablePeerAccess", (void *)cuCtxEnablePeerAccess},
    {"cuCtxDisablePeerAccess", (void *)cuCtxDisablePeerAccess},
    {"cuDeviceGetP2PAttribute", (void *)cuDeviceGetP2PAttribute},
    {"cuGraphicsUnregisterResource", (void *)cuGraphicsUnregisterResource},
    {"cuGraphicsSubResourceGetMappedArray",
     (void *)cuGraphicsSubResourceGetMappedArray},
    {"cuGraphicsResourceGetMappedMipmappedArray",
     (void *)cuGraphicsResourceGetMappedMipmappedArray},
    {"cuGraphicsResourceGetMappedPointer_v2",
     (void *)cuGraphicsResourceGetMappedPointer_v2},
    {"cuGraphicsResourceSetMapFlags_v2",
     (void *)cuGraphicsResourceSetMapFlags_v2},
    {"cuGraphicsMapResources", (void *)cuGraphicsMapResources},
    {"cuGraphicsUnmapResources", (void *)cuGraphicsUnmapResources},
    {"cuDeviceTotalMem", (void *)cuDeviceTotalMem_v2},
    {"cuDeviceGetUuid", (void *)cuDeviceGetUuid_v2},
    {"cuDevicePrimaryCtxRelease", (void *)cuDevicePrimaryCtxRelease_v2},
    {"cuDevicePrimaryCtxSetFlags", (void *)cuDevicePrimaryCtxSetFlags_v2},
    {"cuDevicePrimaryCtxReset", (void *)cuDevicePrimaryCtxReset_v2},
    {"cuCtxDestroy", (void *)cuCtxDestroy_v2},
    {"cuCtxPopCurrent", (void *)cuCtxPopCurrent_v2},
    {"cuCtxPushCurrent", (void *)cuCtxPushCurrent_v2},
    {"cuModuleGetGlobal", (void *)cuModuleGetGlobal_v2},
    {"cuMemAlloc", (void *)cuMemAlloc_v2},
    {"cuMemAllocPitch", (void *)cuMemAllocPitch_v2},
    {"cuMemcpyDtoD", (void *)cuMemcpyDtoD_v2},
    {"cuMemcpyDtoDAsync", (void *)cuMemcpyDtoDAsync_v2},
    {"cuMemsetD8", (void *)cuMemsetD8_v2},
    {"cuMemsetD2D8", (void *)cuMemsetD2D8_v2},
    {"cuMemsetD2D16", (void *)cuMemsetD2D16_v2},
    {"cuMemsetD2D32", (void *)cuMemsetD2D32_v2},
    {"cuIpcOpenMemHandle", (void *)cuIpcOpenMemHandle_v2},
    {"cuStreamBeginCapture", (void *)cuStreamBeginCapture_v2},
    {"cuGraphExecUpdate", (void *)cuGraphExecUpdate_v2},
    {"cuMemcpy_ptds", (void *)cuMemcpy},
    {"cuMemcpyPeer_ptds", (void *)cuMemcpyPeer},
    {"cuMemcpyPeerAsync_ptsz", (void *)cuMemcpyPeerAsync},
    {"cuMemsetD8Async_ptsz", (void *)cuMemsetD8Async},
    {"cuMemsetD16Async_ptsz", (void *)cuMemsetD16Async},
    {"cuMemsetD32Async_ptsz", (void *)cuMemsetD32Async},
    {"cuMemsetD2D8Async_ptsz", (void *)cuMemsetD2D8Async},
    {"cuMemsetD2D16Async_ptsz", (void *)cuMemsetD2D16Async},
    {"cuMemsetD2D32Async_ptsz", (void *)cuMemsetD2D32Async},
    {"cuStreamGetPriority_ptsz", (void *)cuStreamGetPriority},
    {"cuStreamGetId_ptsz", (void *)cuStreamGetId},
    {"cuStreamGetFlags_ptsz", (void *)cuStreamGetFlags},
    {"cuStreamGetCtx_ptsz", (void *)cuStreamGetCtx},
    {"cuStreamEndCapture_ptsz", (void *)cuStreamEndCapture},
    {"cuStreamIsCapturing_ptsz", (void *)cuStreamIsCapturing},
    {"cuStreamAttachMemAsync_ptsz", (void *)cuStreamAttachMemAsync},
    {"cuStreamQuery_ptsz", (void *)cuStreamQuery},
    {"cuStreamSynchronize_ptsz", (void *)cuStreamSynchronize},
    {"cuEventRecord_ptsz", (void *)cuEventRecord},
    {"cuEventRecordWithFlags_ptsz", (void *)cuEventRecordWithFlags},
    {"cuGraphicsMapResources_ptsz", (void *)cuGraphicsMapResources},
    {"cuGraphicsUnmapResources_ptsz", (void *)cuGraphicsUnmapResources},
    {"cuSignalExternalSemaphoresAsync_ptsz",
     (void *)cuSignalExternalSemaphoresAsync},
    {"cuWaitExternalSemaphoresAsync_ptsz",
     (void *)cuWaitExternalSemaphoresAsync},
    {"cuGraphInstantiateWithParams_ptsz", (void *)cuGraphInstantiateWithParams},
    {"cuGraphUpload_ptsz", (void *)cuGraphUpload},
    {"cuGraphLaunch_ptsz", (void *)cuGraphLaunch},
    {"cuStreamCopyAttributes_ptsz", (void *)cuStreamCopyAttributes},
    {"cuStreamGetAttribute_ptsz", (void *)cuStreamGetAttribute},
    {"cuStreamSetAttribute_ptsz", (void *)cuStreamSetAttribute},
    {"cuMemMapArrayAsync_ptsz", (void *)cuMemMapArrayAsync},
    {"cuMemFreeAsync_ptsz", (void *)cuMemFreeAsync},
    {"cuMemAllocAsync_ptsz", (void *)cuMemAllocAsync},
    {"cuMemAllocFromPoolAsync_ptsz", (void *)cuMemAllocFromPoolAsync},
};

void *get_function_pointer(const char *name) {
  auto it = functionMap.find(name);
  if (it == functionMap.end())
    return nullptr;
  return it->second;
}
