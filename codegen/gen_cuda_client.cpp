#include <cuda.h>

#define LUPINE_CUDA_COMPAT_TYPES_ONLY
#include "cuda_compat.h"
#undef LUPINE_CUDA_COMPAT_TYPES_ONLY

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "gen_rpc_ids.h"

#include "client_routing.h"
#include "cuda_client.h"
#include "cuda_client_memcpy.h"
#include "events.h"
#include "rpc.h"

extern "C" void *lupine_deep_node_cache_get(CUgraphNode node, size_t slot,
                                            size_t bytes);
extern "C" void lupine_deep_node_cache_reset(CUgraphNode node);
std::vector<CUgraphNode> lupine_deep_cache_graph_nodes(CUgraph graph);

extern "C" CUfunction
lupine_translate_private_function_for_rpc(CUfunction function);
extern "C" const char *lupine_retain_returned_string(const void *handle,
                                                     const char *data,
                                                     size_t size);

extern "C" void lupine_release_module_retained_strings(CUmodule module);
extern "C" void lupine_release_library_retained_strings(CUlibrary library);

extern "C" void lupine_record_library_module(CUmodule module,
                                             CUlibrary library);

extern "C" CUresult lupine_record_library_kernel(CUkernel kernel,
                                                 CUlibrary library,
                                                 const char *name,
                                                 lupine_route route);

extern "C" CUresult
lupine_cuMemcpyDtoD_via_client(CUdeviceptr dstDevice, CUdeviceptr srcDevice,
                               size_t ByteCount, CUstream hStream, bool async);

extern "C" void lupine_invalidate_current_context_cache();
extern "C" void lupine_forget_destroyed_context(CUcontext ctx);
extern "C" void lupine_disable_local_occupancy();
extern "C" CUresult lupine_resolve_kernel_for_route(CUkernel *kernel,
                                                    lupine_route route);
extern "C" CUresult lupine_resolve_library_for_route(CUlibrary *library,
                                                     lupine_route route);
extern "C" int lupine_read_dtoh_watermark(conn_t *conn);
extern "C" int lupine_forward_remote_stdout(conn_t *conn);
extern "C" CUresult lupine_sync_mapped_device_to_host();

#ifdef cuMemPrefetchAsync
#undef cuMemPrefetchAsync
#endif
extern "C" CUresult CUDAAPI cuMemPrefetchAsync(CUdeviceptr devPtr, size_t count,
                                               CUdevice dstDevice,
                                               CUstream hStream);

#ifdef cuMemAdvise
#undef cuMemAdvise
#endif
extern "C" CUresult CUDAAPI cuMemAdvise(CUdeviceptr devPtr, size_t count,
                                        CUmem_advise advice, CUdevice device);

#ifdef cuGraphInstantiate_v2
#undef cuGraphInstantiate_v2
#endif
extern "C" CUresult CUDAAPI cuGraphInstantiate_v2(CUgraphExec *phGraphExec,
                                                  CUgraph hGraph,
                                                  CUgraphNode *phErrorNode,
                                                  char *logBuffer,
                                                  size_t bufferSize);

#ifdef cuGraphExecUpdate
#undef cuGraphExecUpdate
#endif
extern "C" CUresult CUDAAPI cuGraphExecUpdate(
    CUgraphExec hGraphExec, CUgraph hGraph, CUgraphNode *hErrorNode_out,
    CUgraphExecUpdateResult *updateResult_out);

CUresult cuDriverGetVersion(int *driverVersion) {
  lupine_route route = lupine_route_for_default();
  if (driverVersion == nullptr)
    return CUDA_ERROR_INVALID_VALUE;
  if (lupine_driver_version_cache_lookup(lupine_route_remote_conn(route),
                                         driverVersion))
    return CUDA_SUCCESS;
  CUresult return_value;
  if (lupine_route_is_local(route)) {
    return_value = lupine_call_real_cuda_fn(
        LUPINE_REAL_CUDA_SYMBOL("cuDriverGetVersion"), driverVersion);
    if (return_value == CUDA_SUCCESS)
      lupine_driver_version_cache_insert(lupine_route_remote_conn(route),
                                         *driverVersion);
    return return_value;
  }
  conn_t *conn = lupine_route_remote_conn(route);
  uint8_t driverVersion_present = driverVersion != nullptr;
  if (lupine_prepare_rpc(conn) < 0 ||
      rpc_write_start_request_on(conn, RPC_cuDriverGetVersion, RPC_SCOPE_NONE) <
          0 ||
      rpc_write(conn, &driverVersion_present, sizeof(uint8_t)) < 0 ||
      (driverVersion_present &&
       rpc_write(conn, driverVersion, sizeof(int)) < 0) ||
      rpc_wait_for_response(conn) < 0 ||
      (driverVersion_present &&
       rpc_read(conn, driverVersion, sizeof(int)) < 0) ||
      rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
      rpc_read_end(conn) < 0)
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  if (return_value == CUDA_SUCCESS)
    lupine_driver_version_cache_insert(lupine_route_remote_conn(route),
                                       *driverVersion);
  return return_value;
}

CUresult cuDeviceGetName(char *name, int len, CUdevice dev) {
  CUdevice virtual_dev = dev;
  lupine_route route = lupine_route_for_device(&dev);
  if (route.kind == LUPINE_ROUTE_UNKNOWN_DEVICE)
    return CUDA_ERROR_INVALID_DEVICE;
  if (name == nullptr || len < 0)
    return CUDA_ERROR_INVALID_VALUE;
  if (lupine_device_snapshot_lookup_name(lupine_route_remote_conn(route),
                                         virtual_dev, name, len))
    return CUDA_SUCCESS;
  CUresult return_value;
  if (lupine_route_is_local(route))
    return lupine_call_real_cuda_fn(LUPINE_REAL_CUDA_SYMBOL("cuDeviceGetName"),
                                    name, len, dev);
  conn_t *conn = lupine_route_remote_conn(route);
  uint8_t name_null = name == nullptr ? 1 : 0;
  if (lupine_prepare_rpc(conn) < 0 ||
      rpc_write_start_request_on(conn, RPC_cuDeviceGetName, RPC_SCOPE_NONE) <
          0 ||
      rpc_write(conn, &len, sizeof(int)) < 0 ||
      rpc_write(conn, &name_null, sizeof(uint8_t)) < 0 ||
      rpc_write(conn, &dev, sizeof(CUdevice)) < 0 ||
      rpc_wait_for_response(conn) < 0 ||
      (!name_null && (len <= 0 ? 0 : len * sizeof(char)) != 0 &&
       rpc_read(conn, name, (len <= 0 ? 0 : len * sizeof(char))) < 0) ||
      rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
      rpc_read_end(conn) < 0)
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  return return_value;
}

CUresult cuDeviceGetUuid_v2(CUuuid *uuid, CUdevice dev) {
  CUdevice virtual_dev = dev;
  lupine_route route = lupine_route_for_device(&dev);
  if (route.kind == LUPINE_ROUTE_UNKNOWN_DEVICE)
    return CUDA_ERROR_INVALID_DEVICE;
  if (uuid == nullptr)
    return CUDA_ERROR_INVALID_VALUE;
  if (lupine_device_snapshot_lookup_uuid(lupine_route_remote_conn(route),
                                         virtual_dev, uuid))
    return CUDA_SUCCESS;
  CUresult return_value;
  if (lupine_route_is_local(route))
    return lupine_call_real_cuda_fn(
        LUPINE_REAL_CUDA_SYMBOL("cuDeviceGetUuid_v2"), uuid, dev);
  conn_t *conn = lupine_route_remote_conn(route);
  uint8_t uuid_null = uuid == nullptr ? 1 : 0;
  if (lupine_prepare_rpc(conn) < 0 ||
      rpc_write_start_request_on(conn, RPC_cuDeviceGetUuid_v2, RPC_SCOPE_NONE) <
          0 ||
      rpc_write(conn, &uuid_null, sizeof(uint8_t)) < 0 ||
      rpc_write(conn, &dev, sizeof(CUdevice)) < 0 ||
      rpc_wait_for_response(conn) < 0 ||
      (!uuid_null && 16 != 0 && rpc_read(conn, uuid, 16) < 0) ||
      rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
      rpc_read_end(conn) < 0)
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  return return_value;
}

CUresult cuDeviceGetLuid(char *luid, unsigned int *deviceNodeMask,
                         CUdevice dev) {
  lupine_route route = lupine_route_for_device(&dev);
  if (route.kind == LUPINE_ROUTE_UNKNOWN_DEVICE)
    return CUDA_ERROR_INVALID_DEVICE;
  CUresult return_value;
  if (lupine_route_is_local(route))
    return lupine_call_real_cuda_fn(LUPINE_REAL_CUDA_SYMBOL("cuDeviceGetLuid"),
                                    luid, deviceNodeMask, dev);
  conn_t *conn = lupine_route_remote_conn(route);
  uint8_t luid_null = luid == nullptr ? 1 : 0;
  uint8_t deviceNodeMask_present = deviceNodeMask != nullptr;
  if (lupine_prepare_rpc(conn) < 0 ||
      rpc_write_start_request_on(conn, RPC_cuDeviceGetLuid, RPC_SCOPE_NONE) <
          0 ||
      rpc_write(conn, &luid_null, sizeof(uint8_t)) < 0 ||
      rpc_write(conn, &deviceNodeMask_present, sizeof(uint8_t)) < 0 ||
      (deviceNodeMask_present &&
       rpc_write(conn, deviceNodeMask, sizeof(unsigned int)) < 0) ||
      rpc_write(conn, &dev, sizeof(CUdevice)) < 0 ||
      rpc_wait_for_response(conn) < 0 ||
      (!luid_null && 8 != 0 && rpc_read(conn, luid, 8) < 0) ||
      (deviceNodeMask_present &&
       rpc_read(conn, deviceNodeMask, sizeof(unsigned int)) < 0) ||
      rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
      rpc_read_end(conn) < 0)
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  return return_value;
}

CUresult cuDeviceTotalMem_v2(size_t *bytes, CUdevice dev) {
  CUdevice virtual_dev = dev;
  lupine_route route = lupine_route_for_device(&dev);
  if (route.kind == LUPINE_ROUTE_UNKNOWN_DEVICE)
    return CUDA_ERROR_INVALID_DEVICE;
  if (bytes == nullptr)
    return CUDA_ERROR_INVALID_VALUE;
  if (lupine_device_snapshot_lookup_total_mem(lupine_route_remote_conn(route),
                                              virtual_dev, bytes))
    return CUDA_SUCCESS;
  CUresult return_value;
  if (lupine_route_is_local(route))
    return lupine_call_real_cuda_fn(
        LUPINE_REAL_CUDA_SYMBOL("cuDeviceTotalMem_v2"), bytes, dev);
  conn_t *conn = lupine_route_remote_conn(route);
  uint8_t bytes_present = bytes != nullptr;
  if (lupine_prepare_rpc(conn) < 0 ||
      rpc_write_start_request_on(conn, RPC_cuDeviceTotalMem_v2,
                                 RPC_SCOPE_NONE) < 0 ||
      rpc_write(conn, &bytes_present, sizeof(uint8_t)) < 0 ||
      (bytes_present && rpc_write(conn, bytes, sizeof(size_t)) < 0) ||
      rpc_write(conn, &dev, sizeof(CUdevice)) < 0 ||
      rpc_wait_for_response(conn) < 0 ||
      (bytes_present && rpc_read(conn, bytes, sizeof(size_t)) < 0) ||
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
  if (route.kind == LUPINE_ROUTE_UNKNOWN_DEVICE)
    return CUDA_ERROR_INVALID_DEVICE;
  CUresult return_value;
  if (lupine_route_is_local(route))
    return lupine_call_real_cuda_fn(
        LUPINE_REAL_CUDA_SYMBOL("cuDeviceGetTexture1DLinearMaxWidth"),
        maxWidthInElements, format, numChannels, dev);
  conn_t *conn = lupine_route_remote_conn(route);
  uint8_t maxWidthInElements_present = maxWidthInElements != nullptr;
  if (lupine_prepare_rpc(conn) < 0 ||
      rpc_write_start_request_on(conn, RPC_cuDeviceGetTexture1DLinearMaxWidth,
                                 RPC_SCOPE_NONE) < 0 ||
      rpc_write(conn, &maxWidthInElements_present, sizeof(uint8_t)) < 0 ||
      (maxWidthInElements_present &&
       rpc_write(conn, maxWidthInElements, sizeof(size_t)) < 0) ||
      rpc_write(conn, &format, sizeof(CUarray_format)) < 0 ||
      rpc_write(conn, &numChannels, sizeof(unsigned)) < 0 ||
      rpc_write(conn, &dev, sizeof(CUdevice)) < 0 ||
      rpc_wait_for_response(conn) < 0 ||
      (maxWidthInElements_present &&
       rpc_read(conn, maxWidthInElements, sizeof(size_t)) < 0) ||
      rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
      rpc_read_end(conn) < 0)
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  return return_value;
}

CUresult cuDeviceGetAttribute(int *pi, CUdevice_attribute attrib,
                              CUdevice dev) {
  CUdevice virtual_dev = dev;
  lupine_route route = lupine_route_for_device(&dev);
  if (route.kind == LUPINE_ROUTE_UNKNOWN_DEVICE)
    return CUDA_ERROR_INVALID_DEVICE;
  if (pi == nullptr)
    return CUDA_ERROR_INVALID_VALUE;
  if (lupine_device_attribute_cache_lookup(lupine_route_remote_conn(route),
                                           virtual_dev, attrib, pi))
    return CUDA_SUCCESS;
  CUresult return_value;
  if (lupine_route_is_local(route)) {
    return_value = lupine_call_real_cuda_fn(
        LUPINE_REAL_CUDA_SYMBOL("cuDeviceGetAttribute"), pi, attrib, dev);
    if (return_value == CUDA_SUCCESS)
      lupine_device_attribute_cache_insert(lupine_route_remote_conn(route),
                                           virtual_dev, attrib, pi);
    return return_value;
  }
  conn_t *conn = lupine_route_remote_conn(route);
  uint8_t pi_present = pi != nullptr;
  if (lupine_prepare_rpc(conn) < 0 ||
      rpc_write_start_request_on(conn, RPC_cuDeviceGetAttribute,
                                 RPC_SCOPE_NONE) < 0 ||
      rpc_write(conn, &pi_present, sizeof(uint8_t)) < 0 ||
      (pi_present && rpc_write(conn, pi, sizeof(int)) < 0) ||
      rpc_write(conn, &attrib, sizeof(CUdevice_attribute)) < 0 ||
      rpc_write(conn, &dev, sizeof(CUdevice)) < 0 ||
      rpc_wait_for_response(conn) < 0 ||
      (pi_present && rpc_read(conn, pi, sizeof(int)) < 0) ||
      rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
      rpc_read_end(conn) < 0)
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  if (return_value == CUDA_SUCCESS)
    lupine_device_attribute_cache_insert(lupine_route_remote_conn(route),
                                         virtual_dev, attrib, pi);
  return return_value;
}

CUresult cuDeviceSetMemPool(CUdevice dev, CUmemoryPool pool) {
  lupine_route route = lupine_route_for_device(&dev);
  if (route.kind == LUPINE_ROUTE_UNKNOWN_DEVICE)
    return CUDA_ERROR_INVALID_DEVICE;
  CUresult return_value;
  if (lupine_route_is_local(route))
    return lupine_call_real_cuda_fn(
        LUPINE_REAL_CUDA_SYMBOL("cuDeviceSetMemPool"), dev, pool);
  conn_t *conn = lupine_route_remote_conn(route);
  if (lupine_prepare_rpc(conn) < 0 ||
      rpc_write_start_request_on(conn, RPC_cuDeviceSetMemPool, RPC_SCOPE_NONE) <
          0 ||
      rpc_write(conn, &dev, sizeof(CUdevice)) < 0 ||
      rpc_write(conn, &pool, sizeof(CUmemoryPool)) < 0 ||
      rpc_write_end(conn) < 0)
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  return_value = CUDA_SUCCESS;
  return return_value;
}

CUresult cuDeviceGetMemPool(CUmemoryPool *pool, CUdevice dev) {
  lupine_route route = lupine_route_for_device(&dev);
  if (route.kind == LUPINE_ROUTE_UNKNOWN_DEVICE)
    return CUDA_ERROR_INVALID_DEVICE;
  CUresult return_value;
  if (lupine_route_is_local(route)) {
    return_value = lupine_call_real_cuda_fn(
        LUPINE_REAL_CUDA_SYMBOL("cuDeviceGetMemPool"), pool, dev);
    if (return_value == CUDA_SUCCESS && pool != nullptr) {
      lupine_note_memory_pool_owner_route(*pool, route);
    }
    return return_value;
  }
  conn_t *conn = lupine_route_remote_conn(route);
  uint8_t pool_present = pool != nullptr;
  if (lupine_prepare_rpc(conn) < 0 ||
      rpc_write_start_request_on(conn, RPC_cuDeviceGetMemPool, RPC_SCOPE_NONE) <
          0 ||
      rpc_write(conn, &pool_present, sizeof(uint8_t)) < 0 ||
      (pool_present && rpc_write(conn, pool, sizeof(CUmemoryPool)) < 0) ||
      rpc_write(conn, &dev, sizeof(CUdevice)) < 0 ||
      rpc_wait_for_response(conn) < 0 ||
      (pool_present && rpc_read(conn, pool, sizeof(CUmemoryPool)) < 0) ||
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
  if (route.kind == LUPINE_ROUTE_UNKNOWN_DEVICE)
    return CUDA_ERROR_INVALID_DEVICE;
  CUresult return_value;
  if (lupine_route_is_local(route)) {
    return_value = lupine_call_real_cuda_fn(
        LUPINE_REAL_CUDA_SYMBOL("cuDeviceGetDefaultMemPool"), pool_out, dev);
    if (return_value == CUDA_SUCCESS && pool_out != nullptr) {
      lupine_note_memory_pool_owner_route(*pool_out, route);
    }
    return return_value;
  }
  conn_t *conn = lupine_route_remote_conn(route);
  uint8_t pool_out_present = pool_out != nullptr;
  if (lupine_prepare_rpc(conn) < 0 ||
      rpc_write_start_request_on(conn, RPC_cuDeviceGetDefaultMemPool,
                                 RPC_SCOPE_NONE) < 0 ||
      rpc_write(conn, &pool_out_present, sizeof(uint8_t)) < 0 ||
      (pool_out_present &&
       rpc_write(conn, pool_out, sizeof(CUmemoryPool)) < 0) ||
      rpc_write(conn, &dev, sizeof(CUdevice)) < 0 ||
      rpc_wait_for_response(conn) < 0 ||
      (pool_out_present &&
       rpc_read(conn, pool_out, sizeof(CUmemoryPool)) < 0) ||
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
  if (route.kind == LUPINE_ROUTE_UNKNOWN_DEVICE)
    return CUDA_ERROR_INVALID_DEVICE;
  CUresult return_value;
  if (lupine_route_is_local(route))
    return lupine_call_real_cuda_fn(
        LUPINE_REAL_CUDA_SYMBOL("cuDeviceGetExecAffinitySupport"), pi, type,
        dev);
  conn_t *conn = lupine_route_remote_conn(route);
  uint8_t pi_present = pi != nullptr;
  if (lupine_prepare_rpc(conn) < 0 ||
      rpc_write_start_request_on(conn, RPC_cuDeviceGetExecAffinitySupport,
                                 RPC_SCOPE_NONE) < 0 ||
      rpc_write(conn, &pi_present, sizeof(uint8_t)) < 0 ||
      (pi_present && rpc_write(conn, pi, sizeof(int)) < 0) ||
      rpc_write(conn, &type, sizeof(CUexecAffinityType)) < 0 ||
      rpc_write(conn, &dev, sizeof(CUdevice)) < 0 ||
      rpc_wait_for_response(conn) < 0 ||
      (pi_present && rpc_read(conn, pi, sizeof(int)) < 0) ||
      rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
      rpc_read_end(conn) < 0)
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  return return_value;
}

CUresult cuFlushGPUDirectRDMAWrites(CUflushGPUDirectRDMAWritesTarget target,
                                    CUflushGPUDirectRDMAWritesScope scope) {
  lupine_route route = lupine_route_for_default();
  CUresult return_value;
  if (lupine_route_is_local(route))
    return lupine_call_real_cuda_fn(
        LUPINE_REAL_CUDA_SYMBOL("cuFlushGPUDirectRDMAWrites"), target, scope);
  conn_t *conn = lupine_route_remote_conn(route);
  if (lupine_prepare_rpc(conn) < 0 ||
      rpc_write_start_request_on(conn, RPC_cuFlushGPUDirectRDMAWrites,
                                 RPC_SCOPE_NONE) < 0 ||
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
  if (route.kind == LUPINE_ROUTE_UNKNOWN_DEVICE)
    return CUDA_ERROR_INVALID_DEVICE;
  CUresult return_value;
  if (lupine_route_is_local(route))
    return lupine_call_real_cuda_fn(
        LUPINE_REAL_CUDA_SYMBOL("cuDeviceGetProperties"), prop, dev);
  conn_t *conn = lupine_route_remote_conn(route);
  uint8_t prop_present = prop != nullptr;
  if (lupine_prepare_rpc(conn) < 0 ||
      rpc_write_start_request_on(conn, RPC_cuDeviceGetProperties,
                                 RPC_SCOPE_NONE) < 0 ||
      rpc_write(conn, &prop_present, sizeof(uint8_t)) < 0 ||
      (prop_present && rpc_write(conn, prop, sizeof(CUdevprop)) < 0) ||
      rpc_write(conn, &dev, sizeof(CUdevice)) < 0 ||
      rpc_wait_for_response(conn) < 0 ||
      (prop_present && rpc_read(conn, prop, sizeof(CUdevprop)) < 0) ||
      rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
      rpc_read_end(conn) < 0)
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  return return_value;
}

CUresult cuDeviceComputeCapability(int *major, int *minor, CUdevice dev) {
  lupine_route route = lupine_route_for_device(&dev);
  if (route.kind == LUPINE_ROUTE_UNKNOWN_DEVICE)
    return CUDA_ERROR_INVALID_DEVICE;
  CUresult return_value;
  if (lupine_route_is_local(route))
    return lupine_call_real_cuda_fn(
        LUPINE_REAL_CUDA_SYMBOL("cuDeviceComputeCapability"), major, minor,
        dev);
  conn_t *conn = lupine_route_remote_conn(route);
  uint8_t major_present = major != nullptr;
  uint8_t minor_present = minor != nullptr;
  if (lupine_prepare_rpc(conn) < 0 ||
      rpc_write_start_request_on(conn, RPC_cuDeviceComputeCapability,
                                 RPC_SCOPE_NONE) < 0 ||
      rpc_write(conn, &major_present, sizeof(uint8_t)) < 0 ||
      (major_present && rpc_write(conn, major, sizeof(int)) < 0) ||
      rpc_write(conn, &minor_present, sizeof(uint8_t)) < 0 ||
      (minor_present && rpc_write(conn, minor, sizeof(int)) < 0) ||
      rpc_write(conn, &dev, sizeof(CUdevice)) < 0 ||
      rpc_wait_for_response(conn) < 0 ||
      (major_present && rpc_read(conn, major, sizeof(int)) < 0) ||
      (minor_present && rpc_read(conn, minor, sizeof(int)) < 0) ||
      rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
      rpc_read_end(conn) < 0)
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  return return_value;
}

CUresult cuDevicePrimaryCtxSetFlags_v2(CUdevice dev, unsigned int flags) {
  CUdevice virtual_dev = dev;
  lupine_route route = lupine_route_for_device(&dev);
  if (route.kind == LUPINE_ROUTE_UNKNOWN_DEVICE)
    return CUDA_ERROR_INVALID_DEVICE;
  lupine_invalidate_primary_ctx_state(virtual_dev);
  CUresult return_value;
  if (lupine_route_is_local(route))
    return lupine_call_real_cuda_fn(
        LUPINE_REAL_CUDA_SYMBOL("cuDevicePrimaryCtxSetFlags_v2"), dev, flags);
  conn_t *conn = lupine_route_remote_conn(route);
  if (lupine_prepare_rpc(conn) < 0 ||
      rpc_write_start_request_on(conn, RPC_cuDevicePrimaryCtxSetFlags_v2,
                                 RPC_SCOPE_NONE) < 0 ||
      rpc_write(conn, &dev, sizeof(CUdevice)) < 0 ||
      rpc_write(conn, &flags, sizeof(unsigned int)) < 0 ||
      rpc_write_end(conn) < 0)
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  return_value = CUDA_SUCCESS;
  return return_value;
}

CUresult cuDevicePrimaryCtxGetState(CUdevice dev, unsigned int *flags,
                                    int *active) {
  CUdevice virtual_dev = dev;
  lupine_route route = lupine_route_for_device(&dev);
  if (route.kind == LUPINE_ROUTE_UNKNOWN_DEVICE)
    return CUDA_ERROR_INVALID_DEVICE;
  if (flags == nullptr || active == nullptr)
    return CUDA_ERROR_INVALID_VALUE;
  if (lupine_primary_ctx_state_cache_lookup(lupine_route_remote_conn(route),
                                            virtual_dev, flags, active))
    return CUDA_SUCCESS;
  CUresult return_value;
  if (lupine_route_is_local(route)) {
    return_value = lupine_call_real_cuda_fn(
        LUPINE_REAL_CUDA_SYMBOL("cuDevicePrimaryCtxGetState"), dev, flags,
        active);
    if (return_value == CUDA_SUCCESS)
      lupine_primary_ctx_state_cache_insert(lupine_route_remote_conn(route),
                                            virtual_dev, *flags, *active);
    return return_value;
  }
  conn_t *conn = lupine_route_remote_conn(route);
  uint8_t flags_present = flags != nullptr;
  uint8_t active_present = active != nullptr;
  if (lupine_prepare_rpc(conn) < 0 ||
      rpc_write_start_request_on(conn, RPC_cuDevicePrimaryCtxGetState,
                                 RPC_SCOPE_NONE) < 0 ||
      rpc_write(conn, &dev, sizeof(CUdevice)) < 0 ||
      rpc_write(conn, &flags_present, sizeof(uint8_t)) < 0 ||
      (flags_present && rpc_write(conn, flags, sizeof(unsigned int)) < 0) ||
      rpc_write(conn, &active_present, sizeof(uint8_t)) < 0 ||
      (active_present && rpc_write(conn, active, sizeof(int)) < 0) ||
      rpc_wait_for_response(conn) < 0 ||
      (flags_present && rpc_read(conn, flags, sizeof(unsigned int)) < 0) ||
      (active_present && rpc_read(conn, active, sizeof(int)) < 0) ||
      rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
      rpc_read_end(conn) < 0)
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  if (return_value == CUDA_SUCCESS)
    lupine_primary_ctx_state_cache_insert(lupine_route_remote_conn(route),
                                          virtual_dev, *flags, *active);
  return return_value;
}

CUresult cuCtxDestroy_v2(CUcontext ctx) {
  lupine_route route = lupine_route_for_context(ctx);
  CUcontext lupine_current_before_destroy = nullptr;
  if (cuCtxGetCurrent(&lupine_current_before_destroy) == CUDA_SUCCESS &&
      lupine_current_before_destroy == ctx) {
    cuCtxSetCurrent(nullptr);
  }
  CUresult return_value;
  if (lupine_route_is_local(route)) {
    return_value = lupine_call_real_cuda_fn(
        LUPINE_REAL_CUDA_SYMBOL("cuCtxDestroy_v2"), ctx);
    if (return_value == CUDA_SUCCESS)
      lupine_forget_destroyed_context(ctx);
    if (return_value == CUDA_SUCCESS)
      lupine_invalidate_current_context_cache();
    return return_value;
  }
  conn_t *conn = lupine_route_remote_conn(route);
  if (lupine_prepare_rpc(conn) < 0 ||
      rpc_write_start_request(conn, RPC_cuCtxDestroy_v2) < 0 ||
      rpc_write(conn, &ctx, sizeof(CUcontext)) < 0 ||
      rpc_wait_for_response(conn) < 0 ||
      rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
      rpc_read_end(conn) < 0)
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  if (return_value == CUDA_SUCCESS)
    lupine_forget_destroyed_context(ctx);
  if (return_value == CUDA_SUCCESS)
    lupine_invalidate_current_context_cache();
  return return_value;
}

#if CUDA_VERSION >= 13000
CUresult cuCtxGetDevice_v2(CUdevice *device, CUcontext ctx) {
  lupine_route route = lupine_route_for_context(ctx);
  if (device == nullptr)
    return CUDA_ERROR_INVALID_VALUE;
  if (!lupine_cuda_is_initialized())
    return CUDA_ERROR_NOT_INITIALIZED;
  if (ctx == nullptr)
    return cuCtxGetDevice(device);
  if (!lupine_route_is_local(route) &&
      lupine_current_context_device_cache_lookup(ctx, device))
    return CUDA_SUCCESS;
  CUresult return_value;
  if (lupine_route_is_local(route)) {
    return_value = lupine_call_real_cuda_fn(
        LUPINE_REAL_CUDA_SYMBOL("cuCtxGetDevice_v2"), device, ctx);
    if (!lupine_route_is_local(route) && return_value == CUDA_SUCCESS) {
      *device = lupine_local_device_for_remote(lupine_route_remote_conn(route),
                                               *device);
      lupine_current_context_device_cache_insert(ctx, *device);
    }
    return return_value;
  }
  conn_t *conn = lupine_route_remote_conn(route);
  uint8_t device_present = device != nullptr;
  if (lupine_prepare_rpc(conn) < 0 ||
      rpc_write_start_request_on(conn, RPC_cuCtxGetDevice_v2, RPC_SCOPE_NONE) <
          0 ||
      rpc_write(conn, &device_present, sizeof(uint8_t)) < 0 ||
      (device_present && rpc_write(conn, device, sizeof(CUdevice)) < 0) ||
      rpc_write(conn, &ctx, sizeof(CUcontext)) < 0 ||
      rpc_wait_for_response(conn) < 0 ||
      (device_present && rpc_read(conn, device, sizeof(CUdevice)) < 0) ||
      rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
      rpc_read_end(conn) < 0)
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  if (!lupine_route_is_local(route) && return_value == CUDA_SUCCESS) {
    *device = lupine_local_device_for_remote(lupine_route_remote_conn(route),
                                             *device);
    lupine_current_context_device_cache_insert(ctx, *device);
  }
  return return_value;
}

#endif

CUresult cuCtxGetFlags(unsigned int *flags) {
  lupine_route route = lupine_route_for_default();
  CUresult return_value;
  if (lupine_route_is_local(route))
    return lupine_call_real_cuda_fn(LUPINE_REAL_CUDA_SYMBOL("cuCtxGetFlags"),
                                    flags);
  conn_t *conn = lupine_route_remote_conn(route);
  uint8_t flags_present = flags != nullptr;
  if (lupine_prepare_rpc(conn) < 0 ||
      rpc_write_start_request_on(conn, RPC_cuCtxGetFlags, RPC_SCOPE_NONE) < 0 ||
      rpc_write(conn, &flags_present, sizeof(uint8_t)) < 0 ||
      (flags_present && rpc_write(conn, flags, sizeof(unsigned int)) < 0) ||
      rpc_wait_for_response(conn) < 0 ||
      (flags_present && rpc_read(conn, flags, sizeof(unsigned int)) < 0) ||
      rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
      rpc_read_end(conn) < 0)
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  return return_value;
}

CUresult cuCtxGetId(CUcontext ctx, unsigned long long *ctxId) {
  lupine_route route = lupine_route_for_context(ctx);
  CUresult return_value;
  if (lupine_route_is_local(route))
    return lupine_call_real_cuda_fn(LUPINE_REAL_CUDA_SYMBOL("cuCtxGetId"), ctx,
                                    ctxId);
  conn_t *conn = lupine_route_remote_conn(route);
  uint8_t ctxId_present = ctxId != nullptr;
  if (lupine_prepare_rpc(conn) < 0 ||
      rpc_write_start_request_on(conn, RPC_cuCtxGetId, RPC_SCOPE_NONE) < 0 ||
      rpc_write(conn, &ctx, sizeof(CUcontext)) < 0 ||
      rpc_write(conn, &ctxId_present, sizeof(uint8_t)) < 0 ||
      (ctxId_present &&
       rpc_write(conn, ctxId, sizeof(unsigned long long)) < 0) ||
      rpc_wait_for_response(conn) < 0 ||
      (ctxId_present &&
       rpc_read(conn, ctxId, sizeof(unsigned long long)) < 0) ||
      rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
      rpc_read_end(conn) < 0)
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  return return_value;
}

CUresult cuCtxSynchronize() {
  lupine_route route = lupine_route_for_current_context();
  CUresult return_value;
  if (lupine_route_is_local(route)) {
    return_value =
        lupine_call_real_cuda_fn(LUPINE_REAL_CUDA_SYMBOL("cuCtxSynchronize"));
    if (return_value == CUDA_SUCCESS)
      return_value = lupine_sync_mapped_device_to_host();
    return return_value;
  }
  conn_t *conn = lupine_route_remote_conn(route);
  if (lupine_prepare_rpc(conn) < 0 ||
      rpc_write_start_request(conn, RPC_cuCtxSynchronize) < 0 ||
      rpc_wait_for_response(conn) < 0 || lupine_read_dtoh_watermark(conn) < 0 ||
      lupine_forward_remote_stdout(conn) < 0 ||
      rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
      rpc_read_end(conn) < 0)
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  if (return_value == CUDA_SUCCESS)
    return_value = lupine_sync_mapped_device_to_host();
  return return_value;
}

#if CUDA_VERSION >= 13000
CUresult cuCtxSynchronize_v2(CUcontext ctx) {
  lupine_route route = lupine_route_for_context(ctx);
  CUresult return_value;
  if (lupine_route_is_local(route)) {
    return_value = lupine_call_real_cuda_fn(
        LUPINE_REAL_CUDA_SYMBOL("cuCtxSynchronize_v2"), ctx);
    if (return_value == CUDA_SUCCESS)
      return_value = lupine_sync_mapped_device_to_host();
    return return_value;
  }
  conn_t *conn = lupine_route_remote_conn(route);
  if (lupine_prepare_rpc(conn) < 0 ||
      rpc_write_start_request(conn, RPC_cuCtxSynchronize_v2) < 0 ||
      rpc_write(conn, &ctx, sizeof(CUcontext)) < 0 ||
      rpc_wait_for_response(conn) < 0 || lupine_read_dtoh_watermark(conn) < 0 ||
      lupine_forward_remote_stdout(conn) < 0 ||
      rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
      rpc_read_end(conn) < 0)
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  if (return_value == CUDA_SUCCESS)
    return_value = lupine_sync_mapped_device_to_host();
  return return_value;
}

#endif

CUresult cuCtxSetLimit(CUlimit limit, size_t value) {
  lupine_route route = lupine_route_for_default();
  CUresult return_value;
  if (lupine_route_is_local(route))
    return lupine_call_real_cuda_fn(LUPINE_REAL_CUDA_SYMBOL("cuCtxSetLimit"),
                                    limit, value);
  conn_t *conn = lupine_route_remote_conn(route);
  if (lupine_prepare_rpc(conn) < 0 ||
      rpc_write_start_request(conn, RPC_cuCtxSetLimit) < 0 ||
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
  if (lupine_route_is_local(route))
    return lupine_call_real_cuda_fn(LUPINE_REAL_CUDA_SYMBOL("cuCtxGetLimit"),
                                    pvalue, limit);
  conn_t *conn = lupine_route_remote_conn(route);
  uint8_t pvalue_present = pvalue != nullptr;
  if (lupine_prepare_rpc(conn) < 0 ||
      rpc_write_start_request_on(conn, RPC_cuCtxGetLimit, RPC_SCOPE_NONE) < 0 ||
      rpc_write(conn, &pvalue_present, sizeof(uint8_t)) < 0 ||
      (pvalue_present && rpc_write(conn, pvalue, sizeof(size_t)) < 0) ||
      rpc_write(conn, &limit, sizeof(CUlimit)) < 0 ||
      rpc_wait_for_response(conn) < 0 ||
      (pvalue_present && rpc_read(conn, pvalue, sizeof(size_t)) < 0) ||
      rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
      rpc_read_end(conn) < 0)
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  return return_value;
}

CUresult cuCtxGetCacheConfig(CUfunc_cache *pconfig) {
  lupine_route route = lupine_route_for_default();
  CUresult return_value;
  if (lupine_route_is_local(route))
    return lupine_call_real_cuda_fn(
        LUPINE_REAL_CUDA_SYMBOL("cuCtxGetCacheConfig"), pconfig);
  conn_t *conn = lupine_route_remote_conn(route);
  uint8_t pconfig_present = pconfig != nullptr;
  if (lupine_prepare_rpc(conn) < 0 ||
      rpc_write_start_request_on(conn, RPC_cuCtxGetCacheConfig,
                                 RPC_SCOPE_NONE) < 0 ||
      rpc_write(conn, &pconfig_present, sizeof(uint8_t)) < 0 ||
      (pconfig_present && rpc_write(conn, pconfig, sizeof(CUfunc_cache)) < 0) ||
      rpc_wait_for_response(conn) < 0 ||
      (pconfig_present && rpc_read(conn, pconfig, sizeof(CUfunc_cache)) < 0) ||
      rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
      rpc_read_end(conn) < 0)
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  return return_value;
}

CUresult cuCtxSetCacheConfig(CUfunc_cache config) {
  lupine_route route = lupine_route_for_default();
  CUresult return_value;
  if (lupine_route_is_local(route)) {
    return_value = lupine_call_real_cuda_fn(
        LUPINE_REAL_CUDA_SYMBOL("cuCtxSetCacheConfig"), config);
    if (return_value == CUDA_SUCCESS)
      lupine_disable_local_occupancy();
    return return_value;
  }
  conn_t *conn = lupine_route_remote_conn(route);
  if (lupine_prepare_rpc(conn) < 0 ||
      rpc_write_start_request_on(conn, RPC_cuCtxSetCacheConfig,
                                 RPC_SCOPE_NONE) < 0 ||
      rpc_write(conn, &config, sizeof(CUfunc_cache)) < 0 ||
      rpc_write_end(conn) < 0)
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  return_value = CUDA_SUCCESS;
  if (return_value == CUDA_SUCCESS)
    lupine_disable_local_occupancy();
  return return_value;
}

CUresult cuCtxGetApiVersion(CUcontext ctx, unsigned int *version) {
  lupine_route route = lupine_route_for_context(ctx);
  CUresult return_value;
  if (lupine_route_is_local(route))
    return lupine_call_real_cuda_fn(
        LUPINE_REAL_CUDA_SYMBOL("cuCtxGetApiVersion"), ctx, version);
  conn_t *conn = lupine_route_remote_conn(route);
  uint8_t version_present = version != nullptr;
  if (lupine_prepare_rpc(conn) < 0 ||
      rpc_write_start_request_on(conn, RPC_cuCtxGetApiVersion, RPC_SCOPE_NONE) <
          0 ||
      rpc_write(conn, &ctx, sizeof(CUcontext)) < 0 ||
      rpc_write(conn, &version_present, sizeof(uint8_t)) < 0 ||
      (version_present && rpc_write(conn, version, sizeof(unsigned int)) < 0) ||
      rpc_wait_for_response(conn) < 0 ||
      (version_present && rpc_read(conn, version, sizeof(unsigned int)) < 0) ||
      rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
      rpc_read_end(conn) < 0)
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  return return_value;
}

CUresult cuCtxGetStreamPriorityRange(int *leastPriority,
                                     int *greatestPriority) {
  lupine_route route = lupine_route_for_default();
  CUresult return_value;
  if (lupine_route_is_local(route))
    return lupine_call_real_cuda_fn(
        LUPINE_REAL_CUDA_SYMBOL("cuCtxGetStreamPriorityRange"), leastPriority,
        greatestPriority);
  conn_t *conn = lupine_route_remote_conn(route);
  uint8_t leastPriority_present = leastPriority != nullptr;
  uint8_t greatestPriority_present = greatestPriority != nullptr;
  if (lupine_prepare_rpc(conn) < 0 ||
      rpc_write_start_request_on(conn, RPC_cuCtxGetStreamPriorityRange,
                                 RPC_SCOPE_NONE) < 0 ||
      rpc_write(conn, &leastPriority_present, sizeof(uint8_t)) < 0 ||
      (leastPriority_present &&
       rpc_write(conn, leastPriority, sizeof(int)) < 0) ||
      rpc_write(conn, &greatestPriority_present, sizeof(uint8_t)) < 0 ||
      (greatestPriority_present &&
       rpc_write(conn, greatestPriority, sizeof(int)) < 0) ||
      rpc_wait_for_response(conn) < 0 ||
      (leastPriority_present &&
       rpc_read(conn, leastPriority, sizeof(int)) < 0) ||
      (greatestPriority_present &&
       rpc_read(conn, greatestPriority, sizeof(int)) < 0) ||
      rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
      rpc_read_end(conn) < 0)
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  return return_value;
}

CUresult cuCtxResetPersistingL2Cache() {
  lupine_route route = lupine_route_for_default();
  CUresult return_value;
  if (lupine_route_is_local(route))
    return lupine_call_real_cuda_fn(
        LUPINE_REAL_CUDA_SYMBOL("cuCtxResetPersistingL2Cache"));
  conn_t *conn = lupine_route_remote_conn(route);
  if (lupine_prepare_rpc(conn) < 0 ||
      rpc_write_start_request(conn, RPC_cuCtxResetPersistingL2Cache) < 0 ||
      rpc_write_end(conn) < 0)
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  return_value = CUDA_SUCCESS;
  return return_value;
}

CUresult cuCtxGetExecAffinity(CUexecAffinityParam *pExecAffinity,
                              CUexecAffinityType type) {
  lupine_route route = lupine_route_for_default();
  CUresult return_value;
  if (lupine_route_is_local(route))
    return lupine_call_real_cuda_fn(
        LUPINE_REAL_CUDA_SYMBOL("cuCtxGetExecAffinity"), pExecAffinity, type);
  conn_t *conn = lupine_route_remote_conn(route);
  uint8_t pExecAffinity_present = pExecAffinity != nullptr;
  if (lupine_prepare_rpc(conn) < 0 ||
      rpc_write_start_request_on(conn, RPC_cuCtxGetExecAffinity,
                                 RPC_SCOPE_NONE) < 0 ||
      rpc_write(conn, &pExecAffinity_present, sizeof(uint8_t)) < 0 ||
      (pExecAffinity_present &&
       rpc_write(conn, pExecAffinity, sizeof(CUexecAffinityParam)) < 0) ||
      rpc_write(conn, &type, sizeof(CUexecAffinityType)) < 0 ||
      rpc_wait_for_response(conn) < 0 ||
      (pExecAffinity_present &&
       rpc_read(conn, pExecAffinity, sizeof(CUexecAffinityParam)) < 0) ||
      rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
      rpc_read_end(conn) < 0)
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  return return_value;
}

#if CUDA_VERSION >= 12050
CUresult cuCtxRecordEvent(CUcontext hCtx, CUevent hEvent) {
  lupine_route route = lupine_route_for_context(hCtx);
  CUresult return_value;
  if (lupine_route_is_local(route)) {
    return_value = lupine_call_real_cuda_fn(
        LUPINE_REAL_CUDA_SYMBOL("cuCtxRecordEvent"), hCtx, hEvent);
    // Re-recording re-arms the event, so its cached completion is stale.
    // Invalidate after enqueue so an overlapping query cannot retain old state.
    lupine_event_invalidate_completion(hEvent);
    return return_value;
  }
  conn_t *conn = lupine_route_remote_conn(route);
  if (lupine_prepare_rpc(conn) < 0 ||
      rpc_write_start_request(conn, RPC_cuCtxRecordEvent) < 0 ||
      rpc_request_on(conn, reinterpret_cast<uintptr_t>(hEvent)) < 0 ||
      rpc_write(conn, &hCtx, sizeof(CUcontext)) < 0 ||
      rpc_write(conn, &hEvent, sizeof(CUevent)) < 0 || rpc_write_end(conn) < 0)
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  return_value = CUDA_SUCCESS;
  // Re-recording re-arms the event, so its cached completion is stale.
  // Invalidate after enqueue so an overlapping query cannot retain old state.
  lupine_event_invalidate_completion(hEvent);
  return return_value;
}

#endif

#if CUDA_VERSION >= 12050
CUresult cuCtxWaitEvent(CUcontext hCtx, CUevent hEvent) {
  lupine_route route = lupine_route_for_context(hCtx);
  CUresult return_value;
  if (lupine_route_is_local(route))
    return lupine_call_real_cuda_fn(LUPINE_REAL_CUDA_SYMBOL("cuCtxWaitEvent"),
                                    hCtx, hEvent);
  conn_t *conn = lupine_route_remote_conn(route);
  if (lupine_prepare_rpc(conn) < 0 ||
      rpc_write_start_request(conn, RPC_cuCtxWaitEvent) < 0 ||
      rpc_request_on(conn, reinterpret_cast<uintptr_t>(hEvent)) < 0 ||
      rpc_write(conn, &hCtx, sizeof(CUcontext)) < 0 ||
      rpc_write(conn, &hEvent, sizeof(CUevent)) < 0 || rpc_write_end(conn) < 0)
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  return_value = CUDA_SUCCESS;
  return return_value;
}

#endif

CUresult cuCtxAttach(CUcontext *pctx, unsigned int flags) {
  lupine_route route = lupine_route_for_default();
  CUresult return_value;
  if (lupine_route_is_local(route))
    return lupine_call_real_cuda_fn(LUPINE_REAL_CUDA_SYMBOL("cuCtxAttach"),
                                    pctx, flags);
  conn_t *conn = lupine_route_remote_conn(route);
  if (pctx == nullptr)
    return CUDA_ERROR_INVALID_VALUE;
  if (lupine_prepare_rpc(conn) < 0 ||
      rpc_write_start_request_on(conn, RPC_cuCtxAttach, RPC_SCOPE_NONE) < 0 ||
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
  if (lupine_route_is_local(route)) {
    return_value =
        lupine_call_real_cuda_fn(LUPINE_REAL_CUDA_SYMBOL("cuCtxDetach"), ctx);
    if (return_value == CUDA_SUCCESS)
      lupine_invalidate_current_context_cache();
    return return_value;
  }
  conn_t *conn = lupine_route_remote_conn(route);
  if (lupine_prepare_rpc(conn) < 0 ||
      rpc_write_start_request(conn, RPC_cuCtxDetach) < 0 ||
      rpc_write(conn, &ctx, sizeof(CUcontext)) < 0 ||
      rpc_wait_for_response(conn) < 0 ||
      rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
      rpc_read_end(conn) < 0)
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  if (return_value == CUDA_SUCCESS)
    lupine_invalidate_current_context_cache();
  return return_value;
}

CUresult cuCtxGetSharedMemConfig(CUsharedconfig *pConfig) {
  lupine_route route = lupine_route_for_default();
  CUresult return_value;
  if (lupine_route_is_local(route))
    return lupine_call_real_cuda_fn(
        LUPINE_REAL_CUDA_SYMBOL("cuCtxGetSharedMemConfig"), pConfig);
  conn_t *conn = lupine_route_remote_conn(route);
  uint8_t pConfig_present = pConfig != nullptr;
  if (lupine_prepare_rpc(conn) < 0 ||
      rpc_write_start_request_on(conn, RPC_cuCtxGetSharedMemConfig,
                                 RPC_SCOPE_NONE) < 0 ||
      rpc_write(conn, &pConfig_present, sizeof(uint8_t)) < 0 ||
      (pConfig_present &&
       rpc_write(conn, pConfig, sizeof(CUsharedconfig)) < 0) ||
      rpc_wait_for_response(conn) < 0 ||
      (pConfig_present &&
       rpc_read(conn, pConfig, sizeof(CUsharedconfig)) < 0) ||
      rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
      rpc_read_end(conn) < 0)
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  return return_value;
}

CUresult cuCtxSetSharedMemConfig(CUsharedconfig config) {
  lupine_route route = lupine_route_for_default();
  CUresult return_value;
  if (lupine_route_is_local(route))
    return lupine_call_real_cuda_fn(
        LUPINE_REAL_CUDA_SYMBOL("cuCtxSetSharedMemConfig"), config);
  conn_t *conn = lupine_route_remote_conn(route);
  if (lupine_prepare_rpc(conn) < 0 ||
      rpc_write_start_request_on(conn, RPC_cuCtxSetSharedMemConfig,
                                 RPC_SCOPE_NONE) < 0 ||
      rpc_write(conn, &config, sizeof(CUsharedconfig)) < 0 ||
      rpc_write_end(conn) < 0)
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  return_value = CUDA_SUCCESS;
  return return_value;
}

CUresult cuModuleUnload(CUmodule hmod) {
  lupine_route route = lupine_route_for_module(hmod);
  CUresult return_value;
  if (lupine_route_is_local(route)) {
    return_value = lupine_call_real_cuda_fn(
        LUPINE_REAL_CUDA_SYMBOL("cuModuleUnload"), hmod);
    if (return_value == CUDA_SUCCESS)
      lupine_release_module_retained_strings(hmod);
    return return_value;
  }
  conn_t *conn = lupine_route_remote_conn(route);
  if (lupine_prepare_rpc(conn) < 0 ||
      rpc_write_start_request(conn, RPC_cuModuleUnload) < 0 ||
      rpc_write(conn, &hmod, sizeof(CUmodule)) < 0 ||
      rpc_wait_for_response(conn) < 0 ||
      rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
      rpc_read_end(conn) < 0)
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  if (return_value == CUDA_SUCCESS)
    lupine_release_module_retained_strings(hmod);
  return return_value;
}

CUresult cuModuleGetLoadingMode(CUmoduleLoadingMode *mode) {
  lupine_route route = lupine_route_for_default();
  if (mode == nullptr)
    return CUDA_ERROR_INVALID_VALUE;
  if (lupine_loading_mode_cache_lookup(lupine_route_remote_conn(route), mode))
    return CUDA_SUCCESS;
  CUresult return_value;
  if (lupine_route_is_local(route)) {
    return_value = lupine_call_real_cuda_fn(
        LUPINE_REAL_CUDA_SYMBOL("cuModuleGetLoadingMode"), mode);
    if (return_value == CUDA_SUCCESS)
      lupine_loading_mode_cache_insert(lupine_route_remote_conn(route), *mode);
    return return_value;
  }
  conn_t *conn = lupine_route_remote_conn(route);
  uint8_t mode_present = mode != nullptr;
  if (lupine_prepare_rpc(conn) < 0 ||
      rpc_write_start_request_on(conn, RPC_cuModuleGetLoadingMode,
                                 RPC_SCOPE_NONE) < 0 ||
      rpc_write(conn, &mode_present, sizeof(uint8_t)) < 0 ||
      (mode_present &&
       rpc_write(conn, mode, sizeof(CUmoduleLoadingMode)) < 0) ||
      rpc_wait_for_response(conn) < 0 ||
      (mode_present && rpc_read(conn, mode, sizeof(CUmoduleLoadingMode)) < 0) ||
      rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
      rpc_read_end(conn) < 0)
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  if (return_value == CUDA_SUCCESS)
    lupine_loading_mode_cache_insert(lupine_route_remote_conn(route), *mode);
  return return_value;
}

CUresult cuModuleGetFunction(CUfunction *hfunc, CUmodule hmod,
                             const char *name) {
  lupine_route route = lupine_route_for_module(hmod);
  if (hfunc == nullptr || name == nullptr)
    return CUDA_ERROR_INVALID_VALUE;
  if (!lupine_route_is_local(route) &&
      lupine_try_cached_module_function(hmod, name, route, hfunc))
    return CUDA_SUCCESS;
  CUresult return_value;
  if (lupine_route_is_local(route)) {
    return_value = lupine_call_real_cuda_fn(
        LUPINE_REAL_CUDA_SYMBOL("cuModuleGetFunction"), hfunc, hmod, name);
    if (return_value == CUDA_SUCCESS)
      lupine_finish_module_get_function(*hfunc, hmod, name, route);
    return return_value;
  }
  conn_t *conn = lupine_route_remote_conn(route);
  uint8_t hfunc_present = hfunc != nullptr;
  std::size_t name_len =
      name == nullptr ? 0 : static_cast<std::size_t>(std::strlen(name) + 1);
  if (lupine_prepare_rpc(conn) < 0 ||
      rpc_write_start_request_on(conn, RPC_cuModuleGetFunction,
                                 RPC_SCOPE_NONE) < 0 ||
      rpc_write(conn, &hfunc_present, sizeof(uint8_t)) < 0 ||
      (hfunc_present && rpc_write(conn, hfunc, sizeof(CUfunction)) < 0) ||
      rpc_write(conn, &hmod, sizeof(CUmodule)) < 0 ||
      rpc_write(conn, &name_len, sizeof(std::size_t)) < 0 ||
      (name_len != 0 && rpc_write(conn, name, name_len) < 0) ||
      rpc_wait_for_response(conn) < 0 ||
      (hfunc_present && rpc_read(conn, hfunc, sizeof(CUfunction)) < 0) ||
      rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
      rpc_read_end(conn) < 0)
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  if (return_value == CUDA_SUCCESS)
    lupine_finish_module_get_function(*hfunc, hmod, name, route);
  return return_value;
}

CUresult cuModuleGetGlobal_v2(CUdeviceptr *dptr, size_t *bytes, CUmodule hmod,
                              const char *name) {
  lupine_route route = lupine_route_for_module(hmod);
  size_t global_bytes = 0;
  if (bytes == nullptr && dptr != nullptr)
    bytes = &global_bytes;
  CUresult return_value;
  if (lupine_route_is_local(route)) {
    return_value = lupine_call_real_cuda_fn(
        LUPINE_REAL_CUDA_SYMBOL("cuModuleGetGlobal_v2"), dptr, bytes, hmod,
        name);
    if (return_value == CUDA_SUCCESS && dptr != nullptr) {
      lupine_note_deviceptr_owner_route(*dptr, route);
    }
    if (return_value == CUDA_SUCCESS && dptr != nullptr) {
      lupine_note_deviceptr_allocation_route(*dptr, *bytes, route);
      lupine_adopt_managed_global(route, *dptr, *bytes);
    }
    return return_value;
  }
  conn_t *conn = lupine_route_remote_conn(route);
  uint8_t dptr_present = dptr != nullptr;
  uint8_t bytes_present = bytes != nullptr;
  std::size_t name_len =
      name == nullptr ? 0 : static_cast<std::size_t>(std::strlen(name) + 1);
  if (lupine_prepare_rpc(conn) < 0 ||
      rpc_write_start_request_on(conn, RPC_cuModuleGetGlobal_v2,
                                 RPC_SCOPE_NONE) < 0 ||
      rpc_write(conn, &dptr_present, sizeof(uint8_t)) < 0 ||
      (dptr_present && rpc_write(conn, dptr, sizeof(CUdeviceptr)) < 0) ||
      rpc_write(conn, &bytes_present, sizeof(uint8_t)) < 0 ||
      (bytes_present && rpc_write(conn, bytes, sizeof(size_t)) < 0) ||
      rpc_write(conn, &hmod, sizeof(CUmodule)) < 0 ||
      rpc_write(conn, &name_len, sizeof(std::size_t)) < 0 ||
      (name_len != 0 && rpc_write(conn, name, name_len) < 0) ||
      rpc_wait_for_response(conn) < 0 ||
      (dptr_present && rpc_read(conn, dptr, sizeof(CUdeviceptr)) < 0) ||
      (bytes_present && rpc_read(conn, bytes, sizeof(size_t)) < 0) ||
      rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
      rpc_read_end(conn) < 0)
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  if (return_value == CUDA_SUCCESS && dptr != nullptr) {
    lupine_note_deviceptr_owner_route(*dptr, route);
  }
  if (return_value == CUDA_SUCCESS && dptr != nullptr) {
    lupine_note_deviceptr_allocation_route(*dptr, *bytes, route);
    lupine_adopt_managed_global(route, *dptr, *bytes);
  }
  return return_value;
}

CUresult cuModuleGetTexRef(CUtexref *pTexRef, CUmodule hmod, const char *name) {
  lupine_route route = lupine_route_for_module(hmod);
  CUresult return_value;
  if (lupine_route_is_local(route))
    return lupine_call_real_cuda_fn(
        LUPINE_REAL_CUDA_SYMBOL("cuModuleGetTexRef"), pTexRef, hmod, name);
  conn_t *conn = lupine_route_remote_conn(route);
  uint8_t pTexRef_present = pTexRef != nullptr;
  std::size_t name_len =
      name == nullptr ? 0 : static_cast<std::size_t>(std::strlen(name) + 1);
  if (lupine_prepare_rpc(conn) < 0 ||
      rpc_write_start_request_on(conn, RPC_cuModuleGetTexRef, RPC_SCOPE_NONE) <
          0 ||
      rpc_write(conn, &pTexRef_present, sizeof(uint8_t)) < 0 ||
      (pTexRef_present && rpc_write(conn, pTexRef, sizeof(CUtexref)) < 0) ||
      rpc_write(conn, &hmod, sizeof(CUmodule)) < 0 ||
      rpc_write(conn, &name_len, sizeof(std::size_t)) < 0 ||
      (name_len != 0 && rpc_write(conn, name, name_len) < 0) ||
      rpc_wait_for_response(conn) < 0 ||
      (pTexRef_present && rpc_read(conn, pTexRef, sizeof(CUtexref)) < 0) ||
      rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
      rpc_read_end(conn) < 0)
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  return return_value;
}

CUresult cuModuleGetSurfRef(CUsurfref *pSurfRef, CUmodule hmod,
                            const char *name) {
  lupine_route route = lupine_route_for_module(hmod);
  CUresult return_value;
  if (lupine_route_is_local(route))
    return lupine_call_real_cuda_fn(
        LUPINE_REAL_CUDA_SYMBOL("cuModuleGetSurfRef"), pSurfRef, hmod, name);
  conn_t *conn = lupine_route_remote_conn(route);
  uint8_t pSurfRef_present = pSurfRef != nullptr;
  std::size_t name_len =
      name == nullptr ? 0 : static_cast<std::size_t>(std::strlen(name) + 1);
  if (lupine_prepare_rpc(conn) < 0 ||
      rpc_write_start_request_on(conn, RPC_cuModuleGetSurfRef, RPC_SCOPE_NONE) <
          0 ||
      rpc_write(conn, &pSurfRef_present, sizeof(uint8_t)) < 0 ||
      (pSurfRef_present && rpc_write(conn, pSurfRef, sizeof(CUsurfref)) < 0) ||
      rpc_write(conn, &hmod, sizeof(CUmodule)) < 0 ||
      rpc_write(conn, &name_len, sizeof(std::size_t)) < 0 ||
      (name_len != 0 && rpc_write(conn, name, name_len) < 0) ||
      rpc_wait_for_response(conn) < 0 ||
      (pSurfRef_present && rpc_read(conn, pSurfRef, sizeof(CUsurfref)) < 0) ||
      rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
      rpc_read_end(conn) < 0)
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  return return_value;
}

CUresult cuLibraryUnload(CUlibrary library) {
  lupine_route route = lupine_route_for_library(library);
  CUresult return_value;
  if (lupine_route_is_local(route)) {
    return_value = lupine_call_real_cuda_fn(
        LUPINE_REAL_CUDA_SYMBOL("cuLibraryUnload"), library);
    if (return_value == CUDA_SUCCESS)
      lupine_release_library_retained_strings(library);
    return return_value;
  }
  conn_t *conn = lupine_route_remote_conn(route);
  if (lupine_prepare_rpc(conn) < 0 ||
      rpc_write_start_request_on(conn, RPC_cuLibraryUnload, RPC_SCOPE_NONE) <
          0 ||
      rpc_write(conn, &library, sizeof(CUlibrary)) < 0 ||
      rpc_write_end(conn) < 0)
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  return_value = CUDA_SUCCESS;
  if (return_value == CUDA_SUCCESS)
    lupine_release_library_retained_strings(library);
  return return_value;
}

CUresult cuLibraryGetKernel(CUkernel *pKernel, CUlibrary library,
                            const char *name) {
  lupine_route route = lupine_route_for_library(library);
  if (pKernel == nullptr || name == nullptr)
    return CUDA_ERROR_INVALID_VALUE;
  if (!lupine_route_is_local(route) &&
      lupine_try_cached_library_kernel(library, name, pKernel))
    return CUDA_SUCCESS;
  CUresult return_value;
  if (lupine_route_is_local(route)) {
    return_value = lupine_call_real_cuda_fn(
        LUPINE_REAL_CUDA_SYMBOL("cuLibraryGetKernel"), pKernel, library, name);
    if (return_value == CUDA_SUCCESS)
      return_value =
          lupine_record_library_kernel(*pKernel, library, name, route);
    return return_value;
  }
  conn_t *conn = lupine_route_remote_conn(route);
  uint8_t pKernel_present = pKernel != nullptr;
  std::size_t name_len =
      name == nullptr ? 0 : static_cast<std::size_t>(std::strlen(name) + 1);
  if (lupine_prepare_rpc(conn) < 0 ||
      rpc_write_start_request_on(conn, RPC_cuLibraryGetKernel, RPC_SCOPE_NONE) <
          0 ||
      rpc_write(conn, &pKernel_present, sizeof(uint8_t)) < 0 ||
      (pKernel_present && rpc_write(conn, pKernel, sizeof(CUkernel)) < 0) ||
      rpc_write(conn, &library, sizeof(CUlibrary)) < 0 ||
      rpc_write(conn, &name_len, sizeof(std::size_t)) < 0 ||
      (name_len != 0 && rpc_write(conn, name, name_len) < 0) ||
      rpc_wait_for_response(conn) < 0 ||
      (pKernel_present && rpc_read(conn, pKernel, sizeof(CUkernel)) < 0) ||
      rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
      rpc_read_end(conn) < 0)
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  if (return_value == CUDA_SUCCESS)
    return_value = lupine_record_library_kernel(*pKernel, library, name, route);
  return return_value;
}

CUresult cuLibraryGetModule(CUmodule *pMod, CUlibrary library) {
  lupine_route route = lupine_route_for_current_context();
  if (pMod == nullptr)
    return CUDA_ERROR_INVALID_VALUE;
  CUresult resolved = lupine_resolve_library_for_route(&library, route);
  if (resolved != CUDA_SUCCESS)
    return resolved;
  CUresult return_value;
  if (lupine_route_is_local(route)) {
    return_value = lupine_call_real_cuda_fn(
        LUPINE_REAL_CUDA_SYMBOL("cuLibraryGetModule"), pMod, library);
    if (return_value == CUDA_SUCCESS && pMod != nullptr) {
      lupine_note_module_owner_route(*pMod, route);
    }
    if (return_value == CUDA_SUCCESS && pMod != nullptr)
      lupine_record_library_module(*pMod, library);
    return return_value;
  }
  conn_t *conn = lupine_route_remote_conn(route);
  uint8_t pMod_present = pMod != nullptr;
  if (lupine_prepare_rpc(conn) < 0 ||
      rpc_write_start_request_on(conn, RPC_cuLibraryGetModule, RPC_SCOPE_NONE) <
          0 ||
      rpc_write(conn, &pMod_present, sizeof(uint8_t)) < 0 ||
      (pMod_present && rpc_write(conn, pMod, sizeof(CUmodule)) < 0) ||
      rpc_write(conn, &library, sizeof(CUlibrary)) < 0 ||
      rpc_wait_for_response(conn) < 0 ||
      (pMod_present && rpc_read(conn, pMod, sizeof(CUmodule)) < 0) ||
      rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
      rpc_read_end(conn) < 0)
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  if (return_value == CUDA_SUCCESS && pMod != nullptr) {
    lupine_note_module_owner_route(*pMod, route);
  }
  if (return_value == CUDA_SUCCESS && pMod != nullptr)
    lupine_record_library_module(*pMod, library);
  return return_value;
}

CUresult cuKernelGetFunction(CUfunction *pFunc, CUkernel kernel) {
  lupine_route route = lupine_route_for_default();
  CUcontext current_context = nullptr;
  CUfunction staged_func = nullptr;
  CUfunction *caller_func = pFunc;
  CUresult status = lupine_prepare_kernel_get_function(
      caller_func, &kernel, &route, &current_context);
  if (status != CUDA_ERROR_NOT_READY)
    return status;
  pFunc = &staged_func;
  CUresult return_value;
  if (lupine_route_is_local(route)) {
    return_value = lupine_call_real_cuda_fn(
        LUPINE_REAL_CUDA_SYMBOL("cuKernelGetFunction"), pFunc, kernel);
    return_value = lupine_finish_kernel_get_function(
        return_value, route, current_context, kernel, staged_func, caller_func);
    return return_value;
  }
  conn_t *conn = lupine_route_remote_conn(route);
  uint8_t pFunc_present = pFunc != nullptr;
  if (lupine_prepare_rpc(conn) < 0 ||
      rpc_write_start_request_on(conn, RPC_cuKernelGetFunction,
                                 RPC_SCOPE_NONE) < 0 ||
      rpc_write(conn, &pFunc_present, sizeof(uint8_t)) < 0 ||
      (pFunc_present && rpc_write(conn, pFunc, sizeof(CUfunction)) < 0) ||
      rpc_write(conn, &kernel, sizeof(CUkernel)) < 0 ||
      rpc_wait_for_response(conn) < 0 ||
      (pFunc_present && rpc_read(conn, pFunc, sizeof(CUfunction)) < 0) ||
      rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
      rpc_read_end(conn) < 0)
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  return_value = lupine_finish_kernel_get_function(
      return_value, route, current_context, kernel, staged_func, caller_func);
  return return_value;
}

CUresult cuKernelGetLibrary(CUlibrary *pLib, CUkernel kernel) {
  lupine_route route =
      lupine_route_for_function(reinterpret_cast<CUfunction>(kernel));
  if (pLib == nullptr)
    return CUDA_ERROR_INVALID_VALUE;
  if (lupine_try_cached_kernel_library(kernel, pLib))
    return CUDA_SUCCESS;
  CUresult return_value;
  if (lupine_route_is_local(route)) {
    return_value = lupine_call_real_cuda_fn(
        LUPINE_REAL_CUDA_SYMBOL("cuKernelGetLibrary"), pLib, kernel);
    if (return_value == CUDA_SUCCESS && pLib != nullptr) {
      lupine_note_library_owner_route(*pLib, route);
    }
    return return_value;
  }
  conn_t *conn = lupine_route_remote_conn(route);
  uint8_t pLib_present = pLib != nullptr;
  if (lupine_prepare_rpc(conn) < 0 ||
      rpc_write_start_request_on(conn, RPC_cuKernelGetLibrary, RPC_SCOPE_NONE) <
          0 ||
      rpc_write(conn, &pLib_present, sizeof(uint8_t)) < 0 ||
      (pLib_present && rpc_write(conn, pLib, sizeof(CUlibrary)) < 0) ||
      rpc_write(conn, &kernel, sizeof(CUkernel)) < 0 ||
      rpc_wait_for_response(conn) < 0 ||
      (pLib_present && rpc_read(conn, pLib, sizeof(CUlibrary)) < 0) ||
      rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
      rpc_read_end(conn) < 0)
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  if (return_value == CUDA_SUCCESS && pLib != nullptr) {
    lupine_note_library_owner_route(*pLib, route);
  }
  return return_value;
}

CUresult cuLibraryGetGlobal(CUdeviceptr *dptr, size_t *bytes, CUlibrary library,
                            const char *name) {
  lupine_route route = lupine_route_for_current_context();
  CUresult resolved = lupine_resolve_library_for_route(&library, route);
  if (resolved != CUDA_SUCCESS)
    return resolved;
  CUresult return_value;
  if (lupine_route_is_local(route)) {
    return_value =
        lupine_call_real_cuda_fn(LUPINE_REAL_CUDA_SYMBOL("cuLibraryGetGlobal"),
                                 dptr, bytes, library, name);
    if (return_value == CUDA_SUCCESS && dptr != nullptr) {
      lupine_note_deviceptr_owner_route(*dptr, route);
    }
    if (return_value == CUDA_SUCCESS && dptr != nullptr && bytes != nullptr)
      lupine_note_deviceptr_allocation_route(*dptr, *bytes, route);
    return return_value;
  }
  conn_t *conn = lupine_route_remote_conn(route);
  uint8_t dptr_present = dptr != nullptr;
  uint8_t bytes_present = bytes != nullptr;
  std::size_t name_len =
      name == nullptr ? 0 : static_cast<std::size_t>(std::strlen(name) + 1);
  if (lupine_prepare_rpc(conn) < 0 ||
      rpc_write_start_request_on(conn, RPC_cuLibraryGetGlobal, RPC_SCOPE_NONE) <
          0 ||
      rpc_write(conn, &dptr_present, sizeof(uint8_t)) < 0 ||
      (dptr_present && rpc_write(conn, dptr, sizeof(CUdeviceptr)) < 0) ||
      rpc_write(conn, &bytes_present, sizeof(uint8_t)) < 0 ||
      (bytes_present && rpc_write(conn, bytes, sizeof(size_t)) < 0) ||
      rpc_write(conn, &library, sizeof(CUlibrary)) < 0 ||
      rpc_write(conn, &name_len, sizeof(std::size_t)) < 0 ||
      (name_len != 0 && rpc_write(conn, name, name_len) < 0) ||
      rpc_wait_for_response(conn) < 0 ||
      (dptr_present && rpc_read(conn, dptr, sizeof(CUdeviceptr)) < 0) ||
      (bytes_present && rpc_read(conn, bytes, sizeof(size_t)) < 0) ||
      rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
      rpc_read_end(conn) < 0)
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  if (return_value == CUDA_SUCCESS && dptr != nullptr) {
    lupine_note_deviceptr_owner_route(*dptr, route);
  }
  if (return_value == CUDA_SUCCESS && dptr != nullptr && bytes != nullptr)
    lupine_note_deviceptr_allocation_route(*dptr, *bytes, route);
  return return_value;
}

CUresult cuLibraryGetManaged(CUdeviceptr *dptr, size_t *bytes,
                             CUlibrary library, const char *name) {
  lupine_route route = lupine_route_for_library(library);
  size_t global_bytes = 0;
  if (bytes == nullptr && dptr != nullptr)
    bytes = &global_bytes;
  CUresult return_value;
  if (lupine_route_is_local(route)) {
    return_value =
        lupine_call_real_cuda_fn(LUPINE_REAL_CUDA_SYMBOL("cuLibraryGetManaged"),
                                 dptr, bytes, library, name);
    if (return_value == CUDA_SUCCESS && dptr != nullptr) {
      lupine_note_deviceptr_owner_route(*dptr, route);
    }
    if (return_value == CUDA_SUCCESS && dptr != nullptr) {
      lupine_note_deviceptr_allocation_route(*dptr, *bytes, route);
      lupine_adopt_managed_global(route, *dptr, *bytes);
    }
    return return_value;
  }
  conn_t *conn = lupine_route_remote_conn(route);
  uint8_t dptr_present = dptr != nullptr;
  uint8_t bytes_present = bytes != nullptr;
  std::size_t name_len =
      name == nullptr ? 0 : static_cast<std::size_t>(std::strlen(name) + 1);
  if (lupine_prepare_rpc(conn) < 0 ||
      rpc_write_start_request_on(conn, RPC_cuLibraryGetManaged,
                                 RPC_SCOPE_NONE) < 0 ||
      rpc_write(conn, &dptr_present, sizeof(uint8_t)) < 0 ||
      (dptr_present && rpc_write(conn, dptr, sizeof(CUdeviceptr)) < 0) ||
      rpc_write(conn, &bytes_present, sizeof(uint8_t)) < 0 ||
      (bytes_present && rpc_write(conn, bytes, sizeof(size_t)) < 0) ||
      rpc_write(conn, &library, sizeof(CUlibrary)) < 0 ||
      rpc_write(conn, &name_len, sizeof(std::size_t)) < 0 ||
      (name_len != 0 && rpc_write(conn, name, name_len) < 0) ||
      rpc_wait_for_response(conn) < 0 ||
      (dptr_present && rpc_read(conn, dptr, sizeof(CUdeviceptr)) < 0) ||
      (bytes_present && rpc_read(conn, bytes, sizeof(size_t)) < 0) ||
      rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
      rpc_read_end(conn) < 0)
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  if (return_value == CUDA_SUCCESS && dptr != nullptr) {
    lupine_note_deviceptr_owner_route(*dptr, route);
  }
  if (return_value == CUDA_SUCCESS && dptr != nullptr) {
    lupine_note_deviceptr_allocation_route(*dptr, *bytes, route);
    lupine_adopt_managed_global(route, *dptr, *bytes);
  }
  return return_value;
}

CUresult cuLibraryGetUnifiedFunction(void **fptr, CUlibrary library,
                                     const char *symbol) {
  lupine_route route = lupine_route_for_library(library);
  CUresult return_value;
  if (lupine_route_is_local(route))
    return lupine_call_real_cuda_fn(
        LUPINE_REAL_CUDA_SYMBOL("cuLibraryGetUnifiedFunction"), fptr, library,
        symbol);
  conn_t *conn = lupine_route_remote_conn(route);
  uint8_t fptr_present = fptr != nullptr;
  std::size_t symbol_len =
      symbol == nullptr ? 0 : static_cast<std::size_t>(std::strlen(symbol) + 1);
  if (lupine_prepare_rpc(conn) < 0 ||
      rpc_write_start_request_on(conn, RPC_cuLibraryGetUnifiedFunction,
                                 RPC_SCOPE_NONE) < 0 ||
      rpc_write(conn, &fptr_present, sizeof(uint8_t)) < 0 ||
      (fptr_present && rpc_write(conn, fptr, sizeof(void *)) < 0) ||
      rpc_write(conn, &library, sizeof(CUlibrary)) < 0 ||
      rpc_write(conn, &symbol_len, sizeof(std::size_t)) < 0 ||
      (symbol_len != 0 && rpc_write(conn, symbol, symbol_len) < 0) ||
      rpc_wait_for_response(conn) < 0 ||
      (fptr_present && rpc_read(conn, fptr, sizeof(void *)) < 0) ||
      rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
      rpc_read_end(conn) < 0)
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  return return_value;
}

CUresult cuKernelGetAttribute(int *pi, CUfunction_attribute attrib,
                              CUkernel kernel, CUdevice dev) {
  lupine_route route = lupine_route_for_device(&dev);
  if (route.kind == LUPINE_ROUTE_UNKNOWN_DEVICE)
    return CUDA_ERROR_INVALID_DEVICE;
  if (pi == nullptr)
    return CUDA_ERROR_INVALID_VALUE;
  CUresult resolved = lupine_resolve_kernel_for_route(&kernel, route);
  if (resolved != CUDA_SUCCESS)
    return resolved;
  lupine_kernel_attribute_key key{lupine_route_identity(route), kernel,
                                  (int)attrib, (int)dev};
  if (!lupine_route_is_local(route) && lupine_attribute_get(key, *pi))
    return CUDA_SUCCESS;
  CUresult return_value;
  if (lupine_route_is_local(route)) {
    return_value = lupine_call_real_cuda_fn(
        LUPINE_REAL_CUDA_SYMBOL("cuKernelGetAttribute"), pi, attrib, kernel,
        dev);
    if (return_value == CUDA_SUCCESS && !lupine_route_is_local(route))
      lupine_attribute_set(key, *pi);
    return return_value;
  }
  conn_t *conn = lupine_route_remote_conn(route);
  uint8_t pi_present = pi != nullptr;
  if (lupine_prepare_rpc(conn) < 0 ||
      rpc_write_start_request_on(conn, RPC_cuKernelGetAttribute,
                                 RPC_SCOPE_NONE) < 0 ||
      rpc_write(conn, &pi_present, sizeof(uint8_t)) < 0 ||
      (pi_present && rpc_write(conn, pi, sizeof(int)) < 0) ||
      rpc_write(conn, &attrib, sizeof(CUfunction_attribute)) < 0 ||
      rpc_write(conn, &kernel, sizeof(CUkernel)) < 0 ||
      rpc_write(conn, &dev, sizeof(CUdevice)) < 0 ||
      rpc_wait_for_response(conn) < 0 ||
      (pi_present && rpc_read(conn, pi, sizeof(int)) < 0) ||
      rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
      rpc_read_end(conn) < 0)
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  if (return_value == CUDA_SUCCESS && !lupine_route_is_local(route))
    lupine_attribute_set(key, *pi);
  return return_value;
}

CUresult cuKernelSetAttribute(CUfunction_attribute attrib, int val,
                              CUkernel kernel, CUdevice dev) {
  lupine_route route = lupine_route_for_device(&dev);
  if (route.kind == LUPINE_ROUTE_UNKNOWN_DEVICE)
    return CUDA_ERROR_INVALID_DEVICE;
  CUresult resolved = lupine_resolve_kernel_for_route(&kernel, route);
  if (resolved != CUDA_SUCCESS)
    return resolved;
  CUresult return_value;
  if (lupine_route_is_local(route)) {
    return_value = lupine_call_real_cuda_fn(
        LUPINE_REAL_CUDA_SYMBOL("cuKernelSetAttribute"), attrib, val, kernel,
        dev);
    if (return_value == CUDA_SUCCESS)
      lupine_attribute_set(
          lupine_kernel_attribute_key{lupine_route_identity(route), kernel,
                                      (int)attrib, (int)dev},
          val, lupine_attribute_source::set);
    return return_value;
  }
  conn_t *conn = lupine_route_remote_conn(route);
  if (lupine_prepare_rpc(conn) < 0 ||
      rpc_write_start_request_on(conn, RPC_cuKernelSetAttribute,
                                 RPC_SCOPE_NONE) < 0 ||
      rpc_write(conn, &attrib, sizeof(CUfunction_attribute)) < 0 ||
      rpc_write(conn, &val, sizeof(int)) < 0 ||
      rpc_write(conn, &kernel, sizeof(CUkernel)) < 0 ||
      rpc_write(conn, &dev, sizeof(CUdevice)) < 0 ||
      rpc_wait_for_response(conn) < 0 ||
      rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
      rpc_read_end(conn) < 0)
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  if (return_value == CUDA_SUCCESS)
    lupine_attribute_set(
        lupine_kernel_attribute_key{lupine_route_identity(route), kernel,
                                    (int)attrib, (int)dev},
        val, lupine_attribute_source::set);
  return return_value;
}

CUresult cuKernelSetCacheConfig(CUkernel kernel, CUfunc_cache config,
                                CUdevice dev) {
  lupine_route route = lupine_route_for_device(&dev);
  if (route.kind == LUPINE_ROUTE_UNKNOWN_DEVICE)
    return CUDA_ERROR_INVALID_DEVICE;
  CUresult resolved = lupine_resolve_kernel_for_route(&kernel, route);
  if (resolved != CUDA_SUCCESS)
    return resolved;
  CUresult return_value;
  if (lupine_route_is_local(route)) {
    return_value = lupine_call_real_cuda_fn(
        LUPINE_REAL_CUDA_SYMBOL("cuKernelSetCacheConfig"), kernel, config, dev);
    if (return_value == CUDA_SUCCESS)
      lupine_disable_local_occupancy();
    return return_value;
  }
  conn_t *conn = lupine_route_remote_conn(route);
  if (lupine_prepare_rpc(conn) < 0 ||
      rpc_write_start_request_on(conn, RPC_cuKernelSetCacheConfig,
                                 RPC_SCOPE_NONE) < 0 ||
      rpc_write(conn, &kernel, sizeof(CUkernel)) < 0 ||
      rpc_write(conn, &config, sizeof(CUfunc_cache)) < 0 ||
      rpc_write(conn, &dev, sizeof(CUdevice)) < 0 || rpc_write_end(conn) < 0)
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  return_value = CUDA_SUCCESS;
  if (return_value == CUDA_SUCCESS)
    lupine_disable_local_occupancy();
  return return_value;
}

CUresult cuKernelGetName(const char **name, CUkernel hfunc) {
  lupine_route route =
      lupine_route_for_function(reinterpret_cast<CUfunction>(hfunc));
  CUresult return_value;
  if (lupine_route_is_local(route))
    return lupine_call_real_cuda_fn(LUPINE_REAL_CUDA_SYMBOL("cuKernelGetName"),
                                    name, hfunc);
  conn_t *conn = lupine_route_remote_conn(route);
  uint8_t name_present = name != nullptr;
  std::size_t name_len = 0;
  lupine_scratch_buffer<256> lupine_name_buf;
  char *name_data = nullptr;
  if (lupine_prepare_rpc(conn) < 0 ||
      rpc_write_start_request_on(conn, RPC_cuKernelGetName, RPC_SCOPE_NONE) <
          0 ||
      rpc_write(conn, &name_present, sizeof(uint8_t)) < 0 ||
      rpc_write(conn, &hfunc, sizeof(CUkernel)) < 0 ||
      rpc_wait_for_response(conn) < 0 ||
      (name_present &&
       (rpc_read(conn, &name_len, sizeof(std::size_t)) < 0 ||
        name_len > (1U << 20) ||
        (name_len != 0 &&
         ((name_data = static_cast<char *>(
               lupine_name_buf.alloc_uninit(name_len))) == nullptr ||
          rpc_read(conn, name_data, name_len) < 0)))) ||
      rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
      rpc_read_end(conn) < 0)
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  if (return_value == CUDA_SUCCESS && name_present) {
    const char *name_stored = lupine_retain_returned_string(
        reinterpret_cast<const void *>(hfunc),
        name_data != nullptr ? name_data : "", name_len);
    if (name_stored == nullptr)
      return CUDA_ERROR_OUT_OF_MEMORY;
    *name = name_stored;
  }
  return return_value;
}

CUresult cuMemGetInfo_v2(size_t *free, size_t *total) {
  lupine_route route = lupine_route_for_current_context();
  CUresult return_value;
  if (lupine_route_is_local(route))
    return lupine_call_real_cuda_fn(LUPINE_REAL_CUDA_SYMBOL("cuMemGetInfo_v2"),
                                    free, total);
  conn_t *conn = lupine_route_remote_conn(route);
  uint8_t free_present = free != nullptr;
  uint8_t total_present = total != nullptr;
  if (lupine_prepare_rpc(conn) < 0 ||
      rpc_write_start_request_on(conn, RPC_cuMemGetInfo_v2, RPC_SCOPE_NONE) <
          0 ||
      rpc_write(conn, &free_present, sizeof(uint8_t)) < 0 ||
      (free_present && rpc_write(conn, free, sizeof(size_t)) < 0) ||
      rpc_write(conn, &total_present, sizeof(uint8_t)) < 0 ||
      (total_present && rpc_write(conn, total, sizeof(size_t)) < 0) ||
      rpc_wait_for_response(conn) < 0 ||
      (free_present && rpc_read(conn, free, sizeof(size_t)) < 0) ||
      (total_present && rpc_read(conn, total, sizeof(size_t)) < 0) ||
      rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
      rpc_read_end(conn) < 0)
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  return return_value;
}

CUresult cuMemAlloc_v2(CUdeviceptr *dptr, size_t bytesize) {
  lupine_route route = lupine_route_for_current_context();
  CUresult return_value;
  if (lupine_route_is_local(route)) {
    return_value = lupine_call_real_cuda_fn(
        LUPINE_REAL_CUDA_SYMBOL("cuMemAlloc_v2"), dptr, bytesize);
    if (return_value == CUDA_SUCCESS && dptr != nullptr)
      return_value = lupine_record_disjoint_allocation(
          dptr, bytesize, route, [&] { return cuMemAlloc_v2(dptr, bytesize); });
    return return_value;
  }
  conn_t *conn = lupine_route_remote_conn(route);
  uint8_t dptr_present = dptr != nullptr;
  if (lupine_prepare_rpc(conn) < 0 ||
      rpc_write_start_request_on(conn, RPC_cuMemAlloc_v2, RPC_SCOPE_NONE) < 0 ||
      rpc_write(conn, &dptr_present, sizeof(uint8_t)) < 0 ||
      (dptr_present && rpc_write(conn, dptr, sizeof(CUdeviceptr)) < 0) ||
      rpc_write(conn, &bytesize, sizeof(size_t)) < 0 ||
      rpc_wait_for_response(conn) < 0 ||
      (dptr_present && rpc_read(conn, dptr, sizeof(CUdeviceptr)) < 0) ||
      rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
      rpc_read_end(conn) < 0)
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  if (return_value == CUDA_SUCCESS && dptr != nullptr)
    return_value = lupine_record_disjoint_allocation(
        dptr, bytesize, route, [&] { return cuMemAlloc_v2(dptr, bytesize); });
  return return_value;
}

CUresult cuMemAllocPitch_v2(CUdeviceptr *dptr, size_t *pPitch,
                            size_t WidthInBytes, size_t Height,
                            unsigned int ElementSizeBytes) {
  lupine_route route = lupine_route_for_current_context();
  CUresult return_value;
  if (lupine_route_is_local(route)) {
    return_value = lupine_call_real_cuda_fn(
        LUPINE_REAL_CUDA_SYMBOL("cuMemAllocPitch_v2"), dptr, pPitch,
        WidthInBytes, Height, ElementSizeBytes);
    if (return_value == CUDA_SUCCESS && dptr != nullptr) {
      size_t allocation_size = 0;
      if (pPitch != nullptr)
        allocation_size = (*pPitch) * Height;
      else
        allocation_size = WidthInBytes * Height;
      return_value =
          lupine_record_disjoint_allocation(dptr, allocation_size, route, [&] {
            return cuMemAllocPitch_v2(dptr, pPitch, WidthInBytes, Height,
                                      ElementSizeBytes);
          });
    }
    return return_value;
  }
  conn_t *conn = lupine_route_remote_conn(route);
  uint8_t dptr_present = dptr != nullptr;
  uint8_t pPitch_present = pPitch != nullptr;
  if (lupine_prepare_rpc(conn) < 0 ||
      rpc_write_start_request_on(conn, RPC_cuMemAllocPitch_v2, RPC_SCOPE_NONE) <
          0 ||
      rpc_write(conn, &dptr_present, sizeof(uint8_t)) < 0 ||
      (dptr_present && rpc_write(conn, dptr, sizeof(CUdeviceptr)) < 0) ||
      rpc_write(conn, &pPitch_present, sizeof(uint8_t)) < 0 ||
      (pPitch_present && rpc_write(conn, pPitch, sizeof(size_t)) < 0) ||
      rpc_write(conn, &WidthInBytes, sizeof(size_t)) < 0 ||
      rpc_write(conn, &Height, sizeof(size_t)) < 0 ||
      rpc_write(conn, &ElementSizeBytes, sizeof(unsigned int)) < 0 ||
      rpc_wait_for_response(conn) < 0 ||
      (dptr_present && rpc_read(conn, dptr, sizeof(CUdeviceptr)) < 0) ||
      (pPitch_present && rpc_read(conn, pPitch, sizeof(size_t)) < 0) ||
      rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
      rpc_read_end(conn) < 0)
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  if (return_value == CUDA_SUCCESS && dptr != nullptr) {
    size_t allocation_size = 0;
    if (pPitch != nullptr)
      allocation_size = (*pPitch) * Height;
    else
      allocation_size = WidthInBytes * Height;
    return_value =
        lupine_record_disjoint_allocation(dptr, allocation_size, route, [&] {
          return cuMemAllocPitch_v2(dptr, pPitch, WidthInBytes, Height,
                                    ElementSizeBytes);
        });
  }
  return return_value;
}

CUresult cuMemGetAddressRange_v2(CUdeviceptr *pbase, size_t *psize,
                                 CUdeviceptr dptr) {
  lupine_route route = lupine_route_for_deviceptr(dptr);
  CUresult return_value;
  if (lupine_route_is_local(route))
    return lupine_call_real_cuda_fn(
        LUPINE_REAL_CUDA_SYMBOL("cuMemGetAddressRange_v2"), pbase, psize, dptr);
  conn_t *conn = lupine_route_remote_conn(route);
  uint8_t pbase_present = pbase != nullptr;
  uint8_t psize_present = psize != nullptr;
  if (lupine_prepare_rpc(conn) < 0 ||
      rpc_write_start_request_on(conn, RPC_cuMemGetAddressRange_v2,
                                 RPC_SCOPE_NONE) < 0 ||
      rpc_write(conn, &pbase_present, sizeof(uint8_t)) < 0 ||
      (pbase_present && rpc_write(conn, pbase, sizeof(CUdeviceptr)) < 0) ||
      rpc_write(conn, &psize_present, sizeof(uint8_t)) < 0 ||
      (psize_present && rpc_write(conn, psize, sizeof(size_t)) < 0) ||
      rpc_write(conn, &dptr, sizeof(CUdeviceptr)) < 0 ||
      rpc_wait_for_response(conn) < 0 ||
      (pbase_present && rpc_read(conn, pbase, sizeof(CUdeviceptr)) < 0) ||
      (psize_present && rpc_read(conn, psize, sizeof(size_t)) < 0) ||
      rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
      rpc_read_end(conn) < 0)
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  return return_value;
}

CUresult cuDeviceGetByPCIBusId(CUdevice *dev, const char *pciBusId) {
  return lupine_lookup_device_on_all_routes(dev, [&](lupine_route route,
                                                     CUdevice *route_output) {
    CUdevice *dev = route_output;
    CUresult return_value;
    if (lupine_route_is_local(route))
      return lupine_call_real_cuda_fn(
          LUPINE_REAL_CUDA_SYMBOL("cuDeviceGetByPCIBusId"), dev, pciBusId);
    conn_t *conn = lupine_route_remote_conn(route);
    uint8_t dev_present = dev != nullptr;
    std::size_t pciBusId_len =
        pciBusId == nullptr
            ? 0
            : static_cast<std::size_t>(std::strlen(pciBusId) + 1);
    if (lupine_prepare_rpc(conn) < 0 ||
        rpc_write_start_request_on(conn, RPC_cuDeviceGetByPCIBusId,
                                   RPC_SCOPE_NONE) < 0 ||
        rpc_write(conn, &dev_present, sizeof(uint8_t)) < 0 ||
        (dev_present && rpc_write(conn, dev, sizeof(CUdevice)) < 0) ||
        rpc_write(conn, &pciBusId_len, sizeof(std::size_t)) < 0 ||
        (pciBusId_len != 0 && rpc_write(conn, pciBusId, pciBusId_len) < 0) ||
        rpc_wait_for_response(conn) < 0 ||
        (dev_present && rpc_read(conn, dev, sizeof(CUdevice)) < 0) ||
        rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
        rpc_read_end(conn) < 0)
      return CUDA_ERROR_DEVICE_UNAVAILABLE;
    return return_value;
  });
}

CUresult cuDeviceGetPCIBusId(char *pciBusId, int len, CUdevice dev) {
  lupine_route route = lupine_route_for_device(&dev);
  if (route.kind == LUPINE_ROUTE_UNKNOWN_DEVICE)
    return CUDA_ERROR_INVALID_DEVICE;
  CUresult return_value;
  if (lupine_route_is_local(route))
    return lupine_call_real_cuda_fn(
        LUPINE_REAL_CUDA_SYMBOL("cuDeviceGetPCIBusId"), pciBusId, len, dev);
  conn_t *conn = lupine_route_remote_conn(route);
  uint8_t pciBusId_null = pciBusId == nullptr ? 1 : 0;
  if (lupine_prepare_rpc(conn) < 0 ||
      rpc_write_start_request_on(conn, RPC_cuDeviceGetPCIBusId,
                                 RPC_SCOPE_NONE) < 0 ||
      rpc_write(conn, &len, sizeof(int)) < 0 ||
      rpc_write(conn, &pciBusId_null, sizeof(uint8_t)) < 0 ||
      rpc_write(conn, &dev, sizeof(CUdevice)) < 0 ||
      rpc_wait_for_response(conn) < 0 ||
      (!pciBusId_null && (len <= 0 ? 0 : len * sizeof(char)) != 0 &&
       rpc_read(conn, pciBusId, (len <= 0 ? 0 : len * sizeof(char))) < 0) ||
      rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
      rpc_read_end(conn) < 0)
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  return return_value;
}

CUresult cuIpcGetEventHandle(CUipcEventHandle *pHandle, CUevent event) {
  lupine_route route = lupine_route_for_event(event);
  CUresult return_value;
  if (lupine_route_is_local(route))
    return lupine_call_real_cuda_fn(
        LUPINE_REAL_CUDA_SYMBOL("cuIpcGetEventHandle"), pHandle, event);
  conn_t *conn = lupine_route_remote_conn(route);
  uint8_t pHandle_present = pHandle != nullptr;
  if (lupine_prepare_rpc(conn) < 0 ||
      rpc_write_start_request_on(conn, RPC_cuIpcGetEventHandle,
                                 reinterpret_cast<uintptr_t>(event)) < 0 ||
      rpc_write(conn, &pHandle_present, sizeof(uint8_t)) < 0 ||
      (pHandle_present &&
       rpc_write(conn, pHandle, sizeof(CUipcEventHandle)) < 0) ||
      rpc_write(conn, &event, sizeof(CUevent)) < 0 ||
      rpc_wait_for_response(conn) < 0 ||
      (pHandle_present &&
       rpc_read(conn, pHandle, sizeof(CUipcEventHandle)) < 0) ||
      rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
      rpc_read_end(conn) < 0)
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  return return_value;
}

CUresult cuIpcOpenEventHandle(CUevent *phEvent, CUipcEventHandle handle) {
  lupine_route route = lupine_route_for_default();
  CUresult return_value;
  if (lupine_route_is_local(route))
    return lupine_call_real_cuda_fn(
        LUPINE_REAL_CUDA_SYMBOL("cuIpcOpenEventHandle"), phEvent, handle);
  conn_t *conn = lupine_route_remote_conn(route);
  uint8_t phEvent_present = phEvent != nullptr;
  if (lupine_prepare_rpc(conn) < 0 ||
      rpc_write_start_request_on(conn, RPC_cuIpcOpenEventHandle,
                                 RPC_SCOPE_NONE) < 0 ||
      rpc_write(conn, &phEvent_present, sizeof(uint8_t)) < 0 ||
      (phEvent_present && rpc_write(conn, phEvent, sizeof(CUevent)) < 0) ||
      rpc_write(conn, &handle, sizeof(CUipcEventHandle)) < 0 ||
      rpc_wait_for_response(conn) < 0 ||
      (phEvent_present && rpc_read(conn, phEvent, sizeof(CUevent)) < 0) ||
      rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
      rpc_read_end(conn) < 0)
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  return return_value;
}

CUresult cuIpcGetMemHandle(CUipcMemHandle *pHandle, CUdeviceptr dptr) {
  lupine_route route = lupine_route_for_deviceptr(dptr);
  CUresult return_value;
  if (lupine_route_is_local(route))
    return lupine_call_real_cuda_fn(
        LUPINE_REAL_CUDA_SYMBOL("cuIpcGetMemHandle"), pHandle, dptr);
  conn_t *conn = lupine_route_remote_conn(route);
  uint8_t pHandle_present = pHandle != nullptr;
  if (lupine_prepare_rpc(conn) < 0 ||
      rpc_write_start_request_on(conn, RPC_cuIpcGetMemHandle, RPC_SCOPE_NONE) <
          0 ||
      rpc_write(conn, &pHandle_present, sizeof(uint8_t)) < 0 ||
      (pHandle_present &&
       rpc_write(conn, pHandle, sizeof(CUipcMemHandle)) < 0) ||
      rpc_write(conn, &dptr, sizeof(CUdeviceptr)) < 0 ||
      rpc_wait_for_response(conn) < 0 ||
      (pHandle_present &&
       rpc_read(conn, pHandle, sizeof(CUipcMemHandle)) < 0) ||
      rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
      rpc_read_end(conn) < 0)
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  return return_value;
}

CUresult cuIpcOpenMemHandle_v2(CUdeviceptr *pdptr, CUipcMemHandle handle,
                               unsigned int Flags) {
  lupine_route route = lupine_route_for_default();
  CUresult return_value;
  if (lupine_route_is_local(route))
    return lupine_call_real_cuda_fn(
        LUPINE_REAL_CUDA_SYMBOL("cuIpcOpenMemHandle_v2"), pdptr, handle, Flags);
  conn_t *conn = lupine_route_remote_conn(route);
  uint8_t pdptr_present = pdptr != nullptr;
  if (lupine_prepare_rpc(conn) < 0 ||
      rpc_write_start_request_on(conn, RPC_cuIpcOpenMemHandle_v2,
                                 RPC_SCOPE_NONE) < 0 ||
      rpc_write(conn, &pdptr_present, sizeof(uint8_t)) < 0 ||
      (pdptr_present && rpc_write(conn, pdptr, sizeof(CUdeviceptr)) < 0) ||
      rpc_write(conn, &handle, sizeof(CUipcMemHandle)) < 0 ||
      rpc_write(conn, &Flags, sizeof(unsigned int)) < 0 ||
      rpc_wait_for_response(conn) < 0 ||
      (pdptr_present && rpc_read(conn, pdptr, sizeof(CUdeviceptr)) < 0) ||
      rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
      rpc_read_end(conn) < 0)
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  return return_value;
}

CUresult cuIpcCloseMemHandle(CUdeviceptr dptr) {
  lupine_route route = lupine_route_for_deviceptr(dptr);
  CUresult return_value;
  if (lupine_route_is_local(route))
    return lupine_call_real_cuda_fn(
        LUPINE_REAL_CUDA_SYMBOL("cuIpcCloseMemHandle"), dptr);
  conn_t *conn = lupine_route_remote_conn(route);
  if (lupine_prepare_rpc(conn) < 0 ||
      rpc_write_start_request(conn, RPC_cuIpcCloseMemHandle) < 0 ||
      rpc_write(conn, &dptr, sizeof(CUdeviceptr)) < 0 ||
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
  if (lupine_route_is_local(route))
    return lupine_call_real_cuda_fn(LUPINE_REAL_CUDA_SYMBOL("cuMemcpyPeer"),
                                    dstDevice, dstContext, srcDevice,
                                    srcContext, ByteCount);
  conn_t *conn = lupine_route_remote_conn(route);
  if (lupine_prepare_rpc(conn) < 0 ||
      lupine_rpc_start_on_stream(conn, RPC_cuMemcpyPeer, nullptr) < 0 ||
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
  lupine_note_device_write(dstDevice);
  if (!lupine_deviceptrs_share_route(dstDevice, srcDevice)) {
    return lupine_cuMemcpyDtoD_via_client(dstDevice, srcDevice, ByteCount,
                                          nullptr, false);
  }
  CUresult return_value;
  if (lupine_route_is_local(route)) {
    return_value =
        lupine_call_real_cuda_fn(LUPINE_REAL_CUDA_SYMBOL("cuMemcpyDtoD_v2"),
                                 dstDevice, srcDevice, ByteCount);
    if (return_value == CUDA_SUCCESS)
      return_value = lupine_sync_mapped_device_to_host();
    return return_value;
  }
  conn_t *conn = lupine_route_remote_conn(route);
  if (lupine_prepare_rpc(conn) < 0 ||
      lupine_rpc_start_on_stream(conn, RPC_cuMemcpyDtoD_v2, nullptr) < 0 ||
      rpc_write(conn, &dstDevice, sizeof(CUdeviceptr)) < 0 ||
      rpc_write(conn, &srcDevice, sizeof(CUdeviceptr)) < 0 ||
      rpc_write(conn, &ByteCount, sizeof(size_t)) < 0 ||
      rpc_wait_for_response(conn) < 0 || lupine_read_dtoh_watermark(conn) < 0 ||
      rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
      rpc_read_end(conn) < 0)
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  if (return_value == CUDA_SUCCESS)
    return_value = lupine_sync_mapped_device_to_host();
  return return_value;
}

CUresult cuMemcpyDtoA_v2(CUarray dstArray, size_t dstOffset,
                         CUdeviceptr srcDevice, size_t ByteCount) {
  lupine_route route = lupine_route_for_deviceptr(srcDevice);
  CUresult return_value;
  if (lupine_route_is_local(route))
    return lupine_call_real_cuda_fn(LUPINE_REAL_CUDA_SYMBOL("cuMemcpyDtoA_v2"),
                                    dstArray, dstOffset, srcDevice, ByteCount);
  conn_t *conn = lupine_route_remote_conn(route);
  if (lupine_prepare_rpc(conn) < 0 ||
      lupine_rpc_start_on_stream(conn, RPC_cuMemcpyDtoA_v2, nullptr) < 0 ||
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
  lupine_note_device_write(dstDevice);
  CUresult return_value;
  if (lupine_route_is_local(route))
    return lupine_call_real_cuda_fn(LUPINE_REAL_CUDA_SYMBOL("cuMemcpyAtoD_v2"),
                                    dstDevice, srcArray, srcOffset, ByteCount);
  conn_t *conn = lupine_route_remote_conn(route);
  if (lupine_prepare_rpc(conn) < 0 ||
      lupine_rpc_start_on_stream(conn, RPC_cuMemcpyAtoD_v2, nullptr) < 0 ||
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
  if (lupine_route_is_local(route))
    return lupine_call_real_cuda_fn(LUPINE_REAL_CUDA_SYMBOL("cuMemcpyAtoA_v2"),
                                    dstArray, dstOffset, srcArray, srcOffset,
                                    ByteCount);
  conn_t *conn = lupine_route_remote_conn(route);
  if (lupine_prepare_rpc(conn) < 0 ||
      lupine_rpc_start_on_stream(conn, RPC_cuMemcpyAtoA_v2, nullptr) < 0 ||
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
  if (lupine_route_is_local(route))
    return lupine_call_real_cuda_fn(
        LUPINE_REAL_CUDA_SYMBOL("cuMemcpyPeerAsync"), dstDevice, dstContext,
        srcDevice, srcContext, ByteCount, hStream);
  conn_t *conn = lupine_route_remote_conn(route);
  if (lupine_prepare_rpc(conn) < 0 ||
      lupine_rpc_start_on_stream(conn, RPC_cuMemcpyPeerAsync, hStream) < 0 ||
      rpc_write(conn, &dstDevice, sizeof(CUdeviceptr)) < 0 ||
      rpc_write(conn, &dstContext, sizeof(CUcontext)) < 0 ||
      rpc_write(conn, &srcDevice, sizeof(CUdeviceptr)) < 0 ||
      rpc_write(conn, &srcContext, sizeof(CUcontext)) < 0 ||
      rpc_write(conn, &ByteCount, sizeof(size_t)) < 0 ||
      rpc_write(conn, &hStream, sizeof(CUstream)) < 0 ||
      rpc_write_end(conn) < 0)
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  return_value = CUDA_SUCCESS;
  return return_value;
}

CUresult cuMemcpyDtoDAsync_v2(CUdeviceptr dstDevice, CUdeviceptr srcDevice,
                              size_t ByteCount, CUstream hStream) {
  lupine_route route = lupine_route_for_deviceptr(dstDevice);
  lupine_note_device_write(dstDevice);
  if (!lupine_deviceptrs_share_route(dstDevice, srcDevice)) {
    return lupine_cuMemcpyDtoD_via_client(dstDevice, srcDevice, ByteCount,
                                          hStream, true);
  }
  CUresult return_value;
  if (lupine_route_is_local(route))
    return lupine_call_real_cuda_fn(
        LUPINE_REAL_CUDA_SYMBOL("cuMemcpyDtoDAsync_v2"), dstDevice, srcDevice,
        ByteCount, hStream);
  conn_t *conn = lupine_route_remote_conn(route);
  if (lupine_prepare_rpc(conn) < 0 ||
      lupine_rpc_start_on_stream(conn, RPC_cuMemcpyDtoDAsync_v2, hStream) < 0 ||
      rpc_write(conn, &dstDevice, sizeof(CUdeviceptr)) < 0 ||
      rpc_write(conn, &srcDevice, sizeof(CUdeviceptr)) < 0 ||
      rpc_write(conn, &ByteCount, sizeof(size_t)) < 0 ||
      rpc_write(conn, &hStream, sizeof(CUstream)) < 0 ||
      rpc_write_end(conn) < 0)
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  return_value = CUDA_SUCCESS;
  return return_value;
}

CUresult cuMemsetD8_v2(CUdeviceptr dstDevice, unsigned char uc, size_t N) {
  lupine_route route = lupine_route_for_deviceptr(dstDevice);
  lupine_note_device_write(dstDevice);
  CUresult return_value;
  if (lupine_route_is_local(route)) {
    return_value = lupine_call_real_cuda_fn(
        LUPINE_REAL_CUDA_SYMBOL("cuMemsetD8_v2"), dstDevice, uc, N);
    if (return_value == CUDA_SUCCESS)
      return_value = lupine_sync_mapped_device_to_host();
    return return_value;
  }
  conn_t *conn = lupine_route_remote_conn(route);
  if (lupine_prepare_rpc(conn) < 0 ||
      lupine_rpc_start_on_stream(conn, RPC_cuMemsetD8_v2, nullptr) < 0 ||
      rpc_write(conn, &dstDevice, sizeof(CUdeviceptr)) < 0 ||
      rpc_write(conn, &uc, sizeof(unsigned char)) < 0 ||
      rpc_write(conn, &N, sizeof(size_t)) < 0 ||
      rpc_wait_for_response(conn) < 0 || lupine_read_dtoh_watermark(conn) < 0 ||
      rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
      rpc_read_end(conn) < 0)
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  if (return_value == CUDA_SUCCESS)
    return_value = lupine_sync_mapped_device_to_host();
  return return_value;
}

CUresult cuMemsetD16_v2(CUdeviceptr dstDevice, unsigned short us, size_t N) {
  lupine_route route = lupine_route_for_deviceptr(dstDevice);
  lupine_note_device_write(dstDevice);
  CUresult return_value;
  if (lupine_route_is_local(route)) {
    return_value = lupine_call_real_cuda_fn(
        LUPINE_REAL_CUDA_SYMBOL("cuMemsetD16_v2"), dstDevice, us, N);
    if (return_value == CUDA_SUCCESS)
      return_value = lupine_sync_mapped_device_to_host();
    return return_value;
  }
  conn_t *conn = lupine_route_remote_conn(route);
  if (lupine_prepare_rpc(conn) < 0 ||
      lupine_rpc_start_on_stream(conn, RPC_cuMemsetD16_v2, nullptr) < 0 ||
      rpc_write(conn, &dstDevice, sizeof(CUdeviceptr)) < 0 ||
      rpc_write(conn, &us, sizeof(unsigned short)) < 0 ||
      rpc_write(conn, &N, sizeof(size_t)) < 0 ||
      rpc_wait_for_response(conn) < 0 || lupine_read_dtoh_watermark(conn) < 0 ||
      rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
      rpc_read_end(conn) < 0)
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  if (return_value == CUDA_SUCCESS)
    return_value = lupine_sync_mapped_device_to_host();
  return return_value;
}

CUresult cuMemsetD32_v2(CUdeviceptr dstDevice, unsigned int ui, size_t N) {
  lupine_route route = lupine_route_for_deviceptr(dstDevice);
  lupine_note_device_write(dstDevice);
  CUresult return_value;
  if (lupine_route_is_local(route)) {
    return_value = lupine_call_real_cuda_fn(
        LUPINE_REAL_CUDA_SYMBOL("cuMemsetD32_v2"), dstDevice, ui, N);
    if (return_value == CUDA_SUCCESS)
      return_value = lupine_sync_mapped_device_to_host();
    return return_value;
  }
  conn_t *conn = lupine_route_remote_conn(route);
  if (lupine_prepare_rpc(conn) < 0 ||
      lupine_rpc_start_on_stream(conn, RPC_cuMemsetD32_v2, nullptr) < 0 ||
      rpc_write(conn, &dstDevice, sizeof(CUdeviceptr)) < 0 ||
      rpc_write(conn, &ui, sizeof(unsigned int)) < 0 ||
      rpc_write(conn, &N, sizeof(size_t)) < 0 ||
      rpc_wait_for_response(conn) < 0 || lupine_read_dtoh_watermark(conn) < 0 ||
      rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
      rpc_read_end(conn) < 0)
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  if (return_value == CUDA_SUCCESS)
    return_value = lupine_sync_mapped_device_to_host();
  return return_value;
}

CUresult cuMemsetD2D8_v2(CUdeviceptr dstDevice, size_t dstPitch,
                         unsigned char uc, size_t Width, size_t Height) {
  lupine_route route = lupine_route_for_deviceptr(dstDevice);
  lupine_note_device_write(dstDevice);
  CUresult return_value;
  if (lupine_route_is_local(route)) {
    return_value =
        lupine_call_real_cuda_fn(LUPINE_REAL_CUDA_SYMBOL("cuMemsetD2D8_v2"),
                                 dstDevice, dstPitch, uc, Width, Height);
    if (return_value == CUDA_SUCCESS)
      return_value = lupine_sync_mapped_device_to_host();
    return return_value;
  }
  conn_t *conn = lupine_route_remote_conn(route);
  if (lupine_prepare_rpc(conn) < 0 ||
      lupine_rpc_start_on_stream(conn, RPC_cuMemsetD2D8_v2, nullptr) < 0 ||
      rpc_write(conn, &dstDevice, sizeof(CUdeviceptr)) < 0 ||
      rpc_write(conn, &dstPitch, sizeof(size_t)) < 0 ||
      rpc_write(conn, &uc, sizeof(unsigned char)) < 0 ||
      rpc_write(conn, &Width, sizeof(size_t)) < 0 ||
      rpc_write(conn, &Height, sizeof(size_t)) < 0 ||
      rpc_wait_for_response(conn) < 0 || lupine_read_dtoh_watermark(conn) < 0 ||
      rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
      rpc_read_end(conn) < 0)
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  if (return_value == CUDA_SUCCESS)
    return_value = lupine_sync_mapped_device_to_host();
  return return_value;
}

CUresult cuMemsetD2D16_v2(CUdeviceptr dstDevice, size_t dstPitch,
                          unsigned short us, size_t Width, size_t Height) {
  lupine_route route = lupine_route_for_deviceptr(dstDevice);
  lupine_note_device_write(dstDevice);
  CUresult return_value;
  if (lupine_route_is_local(route)) {
    return_value =
        lupine_call_real_cuda_fn(LUPINE_REAL_CUDA_SYMBOL("cuMemsetD2D16_v2"),
                                 dstDevice, dstPitch, us, Width, Height);
    if (return_value == CUDA_SUCCESS)
      return_value = lupine_sync_mapped_device_to_host();
    return return_value;
  }
  conn_t *conn = lupine_route_remote_conn(route);
  if (lupine_prepare_rpc(conn) < 0 ||
      lupine_rpc_start_on_stream(conn, RPC_cuMemsetD2D16_v2, nullptr) < 0 ||
      rpc_write(conn, &dstDevice, sizeof(CUdeviceptr)) < 0 ||
      rpc_write(conn, &dstPitch, sizeof(size_t)) < 0 ||
      rpc_write(conn, &us, sizeof(unsigned short)) < 0 ||
      rpc_write(conn, &Width, sizeof(size_t)) < 0 ||
      rpc_write(conn, &Height, sizeof(size_t)) < 0 ||
      rpc_wait_for_response(conn) < 0 || lupine_read_dtoh_watermark(conn) < 0 ||
      rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
      rpc_read_end(conn) < 0)
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  if (return_value == CUDA_SUCCESS)
    return_value = lupine_sync_mapped_device_to_host();
  return return_value;
}

CUresult cuMemsetD2D32_v2(CUdeviceptr dstDevice, size_t dstPitch,
                          unsigned int ui, size_t Width, size_t Height) {
  lupine_route route = lupine_route_for_deviceptr(dstDevice);
  lupine_note_device_write(dstDevice);
  CUresult return_value;
  if (lupine_route_is_local(route)) {
    return_value =
        lupine_call_real_cuda_fn(LUPINE_REAL_CUDA_SYMBOL("cuMemsetD2D32_v2"),
                                 dstDevice, dstPitch, ui, Width, Height);
    if (return_value == CUDA_SUCCESS)
      return_value = lupine_sync_mapped_device_to_host();
    return return_value;
  }
  conn_t *conn = lupine_route_remote_conn(route);
  if (lupine_prepare_rpc(conn) < 0 ||
      lupine_rpc_start_on_stream(conn, RPC_cuMemsetD2D32_v2, nullptr) < 0 ||
      rpc_write(conn, &dstDevice, sizeof(CUdeviceptr)) < 0 ||
      rpc_write(conn, &dstPitch, sizeof(size_t)) < 0 ||
      rpc_write(conn, &ui, sizeof(unsigned int)) < 0 ||
      rpc_write(conn, &Width, sizeof(size_t)) < 0 ||
      rpc_write(conn, &Height, sizeof(size_t)) < 0 ||
      rpc_wait_for_response(conn) < 0 || lupine_read_dtoh_watermark(conn) < 0 ||
      rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
      rpc_read_end(conn) < 0)
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  if (return_value == CUDA_SUCCESS)
    return_value = lupine_sync_mapped_device_to_host();
  return return_value;
}

CUresult cuMemsetD8Async(CUdeviceptr dstDevice, unsigned char uc, size_t N,
                         CUstream hStream) {
  lupine_route route = lupine_route_for_deviceptr(dstDevice);
  lupine_note_device_write(dstDevice);
  CUresult return_value;
  if (lupine_route_is_local(route))
    return lupine_call_real_cuda_fn(LUPINE_REAL_CUDA_SYMBOL("cuMemsetD8Async"),
                                    dstDevice, uc, N, hStream);
  conn_t *conn = lupine_route_remote_conn(route);
  if (lupine_prepare_rpc(conn) < 0 ||
      lupine_rpc_start_on_stream(conn, RPC_cuMemsetD8Async, hStream) < 0 ||
      rpc_write(conn, &dstDevice, sizeof(CUdeviceptr)) < 0 ||
      rpc_write(conn, &uc, sizeof(unsigned char)) < 0 ||
      rpc_write(conn, &N, sizeof(size_t)) < 0 ||
      rpc_write(conn, &hStream, sizeof(CUstream)) < 0 ||
      rpc_write_end(conn) < 0)
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  return_value = CUDA_SUCCESS;
  return return_value;
}

CUresult cuMemsetD16Async(CUdeviceptr dstDevice, unsigned short us, size_t N,
                          CUstream hStream) {
  lupine_route route = lupine_route_for_deviceptr(dstDevice);
  lupine_note_device_write(dstDevice);
  CUresult return_value;
  if (lupine_route_is_local(route))
    return lupine_call_real_cuda_fn(LUPINE_REAL_CUDA_SYMBOL("cuMemsetD16Async"),
                                    dstDevice, us, N, hStream);
  conn_t *conn = lupine_route_remote_conn(route);
  if (lupine_prepare_rpc(conn) < 0 ||
      lupine_rpc_start_on_stream(conn, RPC_cuMemsetD16Async, hStream) < 0 ||
      rpc_write(conn, &dstDevice, sizeof(CUdeviceptr)) < 0 ||
      rpc_write(conn, &us, sizeof(unsigned short)) < 0 ||
      rpc_write(conn, &N, sizeof(size_t)) < 0 ||
      rpc_write(conn, &hStream, sizeof(CUstream)) < 0 ||
      rpc_write_end(conn) < 0)
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  return_value = CUDA_SUCCESS;
  return return_value;
}

CUresult cuMemsetD32Async(CUdeviceptr dstDevice, unsigned int ui, size_t N,
                          CUstream hStream) {
  lupine_route route = lupine_route_for_deviceptr(dstDevice);
  lupine_note_device_write(dstDevice);
  CUresult return_value;
  if (lupine_route_is_local(route))
    return lupine_call_real_cuda_fn(LUPINE_REAL_CUDA_SYMBOL("cuMemsetD32Async"),
                                    dstDevice, ui, N, hStream);
  conn_t *conn = lupine_route_remote_conn(route);
  if (lupine_prepare_rpc(conn) < 0 ||
      lupine_rpc_start_on_stream(conn, RPC_cuMemsetD32Async, hStream) < 0 ||
      rpc_write(conn, &dstDevice, sizeof(CUdeviceptr)) < 0 ||
      rpc_write(conn, &ui, sizeof(unsigned int)) < 0 ||
      rpc_write(conn, &N, sizeof(size_t)) < 0 ||
      rpc_write(conn, &hStream, sizeof(CUstream)) < 0 ||
      rpc_write_end(conn) < 0)
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  return_value = CUDA_SUCCESS;
  return return_value;
}

CUresult cuMemsetD2D8Async(CUdeviceptr dstDevice, size_t dstPitch,
                           unsigned char uc, size_t Width, size_t Height,
                           CUstream hStream) {
  lupine_route route = lupine_route_for_deviceptr(dstDevice);
  lupine_note_device_write(dstDevice);
  CUresult return_value;
  if (lupine_route_is_local(route))
    return lupine_call_real_cuda_fn(
        LUPINE_REAL_CUDA_SYMBOL("cuMemsetD2D8Async"), dstDevice, dstPitch, uc,
        Width, Height, hStream);
  conn_t *conn = lupine_route_remote_conn(route);
  if (lupine_prepare_rpc(conn) < 0 ||
      lupine_rpc_start_on_stream(conn, RPC_cuMemsetD2D8Async, hStream) < 0 ||
      rpc_write(conn, &dstDevice, sizeof(CUdeviceptr)) < 0 ||
      rpc_write(conn, &dstPitch, sizeof(size_t)) < 0 ||
      rpc_write(conn, &uc, sizeof(unsigned char)) < 0 ||
      rpc_write(conn, &Width, sizeof(size_t)) < 0 ||
      rpc_write(conn, &Height, sizeof(size_t)) < 0 ||
      rpc_write(conn, &hStream, sizeof(CUstream)) < 0 ||
      rpc_write_end(conn) < 0)
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  return_value = CUDA_SUCCESS;
  return return_value;
}

CUresult cuMemsetD2D16Async(CUdeviceptr dstDevice, size_t dstPitch,
                            unsigned short us, size_t Width, size_t Height,
                            CUstream hStream) {
  lupine_route route = lupine_route_for_deviceptr(dstDevice);
  lupine_note_device_write(dstDevice);
  CUresult return_value;
  if (lupine_route_is_local(route))
    return lupine_call_real_cuda_fn(
        LUPINE_REAL_CUDA_SYMBOL("cuMemsetD2D16Async"), dstDevice, dstPitch, us,
        Width, Height, hStream);
  conn_t *conn = lupine_route_remote_conn(route);
  if (lupine_prepare_rpc(conn) < 0 ||
      lupine_rpc_start_on_stream(conn, RPC_cuMemsetD2D16Async, hStream) < 0 ||
      rpc_write(conn, &dstDevice, sizeof(CUdeviceptr)) < 0 ||
      rpc_write(conn, &dstPitch, sizeof(size_t)) < 0 ||
      rpc_write(conn, &us, sizeof(unsigned short)) < 0 ||
      rpc_write(conn, &Width, sizeof(size_t)) < 0 ||
      rpc_write(conn, &Height, sizeof(size_t)) < 0 ||
      rpc_write(conn, &hStream, sizeof(CUstream)) < 0 ||
      rpc_write_end(conn) < 0)
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  return_value = CUDA_SUCCESS;
  return return_value;
}

CUresult cuMemsetD2D32Async(CUdeviceptr dstDevice, size_t dstPitch,
                            unsigned int ui, size_t Width, size_t Height,
                            CUstream hStream) {
  lupine_route route = lupine_route_for_deviceptr(dstDevice);
  lupine_note_device_write(dstDevice);
  CUresult return_value;
  if (lupine_route_is_local(route))
    return lupine_call_real_cuda_fn(
        LUPINE_REAL_CUDA_SYMBOL("cuMemsetD2D32Async"), dstDevice, dstPitch, ui,
        Width, Height, hStream);
  conn_t *conn = lupine_route_remote_conn(route);
  if (lupine_prepare_rpc(conn) < 0 ||
      lupine_rpc_start_on_stream(conn, RPC_cuMemsetD2D32Async, hStream) < 0 ||
      rpc_write(conn, &dstDevice, sizeof(CUdeviceptr)) < 0 ||
      rpc_write(conn, &dstPitch, sizeof(size_t)) < 0 ||
      rpc_write(conn, &ui, sizeof(unsigned int)) < 0 ||
      rpc_write(conn, &Width, sizeof(size_t)) < 0 ||
      rpc_write(conn, &Height, sizeof(size_t)) < 0 ||
      rpc_write(conn, &hStream, sizeof(CUstream)) < 0 ||
      rpc_write_end(conn) < 0)
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  return_value = CUDA_SUCCESS;
  return return_value;
}

CUresult cuArrayCreate_v2(CUarray *pHandle,
                          const CUDA_ARRAY_DESCRIPTOR *pAllocateArray) {
  lupine_route route = lupine_route_for_default();
  CUresult return_value;
  if (lupine_route_is_local(route))
    return lupine_call_real_cuda_fn(LUPINE_REAL_CUDA_SYMBOL("cuArrayCreate_v2"),
                                    pHandle, pAllocateArray);
  conn_t *conn = lupine_route_remote_conn(route);
  uint8_t pHandle_present = pHandle != nullptr;
  uint8_t pAllocateArray_present = pAllocateArray != nullptr;
  if (lupine_prepare_rpc(conn) < 0 ||
      rpc_write_start_request_on(conn, RPC_cuArrayCreate_v2, RPC_SCOPE_NONE) <
          0 ||
      rpc_write(conn, &pHandle_present, sizeof(uint8_t)) < 0 ||
      (pHandle_present && rpc_write(conn, pHandle, sizeof(CUarray)) < 0) ||
      rpc_write(conn, &pAllocateArray_present, sizeof(uint8_t)) < 0 ||
      (pAllocateArray_present &&
       rpc_write(conn, pAllocateArray, sizeof(const CUDA_ARRAY_DESCRIPTOR)) <
           0) ||
      rpc_wait_for_response(conn) < 0 ||
      (pHandle_present && rpc_read(conn, pHandle, sizeof(CUarray)) < 0) ||
      rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
      rpc_read_end(conn) < 0)
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  return return_value;
}

CUresult cuArrayGetDescriptor_v2(CUDA_ARRAY_DESCRIPTOR *pArrayDescriptor,
                                 CUarray hArray) {
  lupine_route route = lupine_route_for_default();
  CUresult return_value;
  if (lupine_route_is_local(route))
    return lupine_call_real_cuda_fn(
        LUPINE_REAL_CUDA_SYMBOL("cuArrayGetDescriptor_v2"), pArrayDescriptor,
        hArray);
  conn_t *conn = lupine_route_remote_conn(route);
  uint8_t pArrayDescriptor_present = pArrayDescriptor != nullptr;
  if (lupine_prepare_rpc(conn) < 0 ||
      rpc_write_start_request_on(conn, RPC_cuArrayGetDescriptor_v2,
                                 RPC_SCOPE_NONE) < 0 ||
      rpc_write(conn, &pArrayDescriptor_present, sizeof(uint8_t)) < 0 ||
      (pArrayDescriptor_present &&
       rpc_write(conn, pArrayDescriptor, sizeof(CUDA_ARRAY_DESCRIPTOR)) < 0) ||
      rpc_write(conn, &hArray, sizeof(CUarray)) < 0 ||
      rpc_wait_for_response(conn) < 0 ||
      (pArrayDescriptor_present &&
       rpc_read(conn, pArrayDescriptor, sizeof(CUDA_ARRAY_DESCRIPTOR)) < 0) ||
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
  if (lupine_route_is_local(route))
    return lupine_call_real_cuda_fn(
        LUPINE_REAL_CUDA_SYMBOL("cuArrayGetSparseProperties"), sparseProperties,
        array);
  conn_t *conn = lupine_route_remote_conn(route);
  uint8_t sparseProperties_present = sparseProperties != nullptr;
  if (lupine_prepare_rpc(conn) < 0 ||
      rpc_write_start_request_on(conn, RPC_cuArrayGetSparseProperties,
                                 RPC_SCOPE_NONE) < 0 ||
      rpc_write(conn, &sparseProperties_present, sizeof(uint8_t)) < 0 ||
      (sparseProperties_present &&
       rpc_write(conn, sparseProperties, sizeof(CUDA_ARRAY_SPARSE_PROPERTIES)) <
           0) ||
      rpc_write(conn, &array, sizeof(CUarray)) < 0 ||
      rpc_wait_for_response(conn) < 0 ||
      (sparseProperties_present &&
       rpc_read(conn, sparseProperties, sizeof(CUDA_ARRAY_SPARSE_PROPERTIES)) <
           0) ||
      rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
      rpc_read_end(conn) < 0)
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  return return_value;
}

CUresult cuMipmappedArrayGetSparseProperties(
    CUDA_ARRAY_SPARSE_PROPERTIES *sparseProperties, CUmipmappedArray mipmap) {
  lupine_route route = lupine_route_for_default();
  CUresult return_value;
  if (lupine_route_is_local(route))
    return lupine_call_real_cuda_fn(
        LUPINE_REAL_CUDA_SYMBOL("cuMipmappedArrayGetSparseProperties"),
        sparseProperties, mipmap);
  conn_t *conn = lupine_route_remote_conn(route);
  uint8_t sparseProperties_present = sparseProperties != nullptr;
  if (lupine_prepare_rpc(conn) < 0 ||
      rpc_write_start_request_on(conn, RPC_cuMipmappedArrayGetSparseProperties,
                                 RPC_SCOPE_NONE) < 0 ||
      rpc_write(conn, &sparseProperties_present, sizeof(uint8_t)) < 0 ||
      (sparseProperties_present &&
       rpc_write(conn, sparseProperties, sizeof(CUDA_ARRAY_SPARSE_PROPERTIES)) <
           0) ||
      rpc_write(conn, &mipmap, sizeof(CUmipmappedArray)) < 0 ||
      rpc_wait_for_response(conn) < 0 ||
      (sparseProperties_present &&
       rpc_read(conn, sparseProperties, sizeof(CUDA_ARRAY_SPARSE_PROPERTIES)) <
           0) ||
      rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
      rpc_read_end(conn) < 0)
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  return return_value;
}

CUresult
cuArrayGetMemoryRequirements(CUDA_ARRAY_MEMORY_REQUIREMENTS *memoryRequirements,
                             CUarray array, CUdevice device) {
  lupine_route route = lupine_route_for_device(&device);
  if (route.kind == LUPINE_ROUTE_UNKNOWN_DEVICE)
    return CUDA_ERROR_INVALID_DEVICE;
  CUresult return_value;
  if (lupine_route_is_local(route))
    return lupine_call_real_cuda_fn(
        LUPINE_REAL_CUDA_SYMBOL("cuArrayGetMemoryRequirements"),
        memoryRequirements, array, device);
  conn_t *conn = lupine_route_remote_conn(route);
  uint8_t memoryRequirements_present = memoryRequirements != nullptr;
  if (lupine_prepare_rpc(conn) < 0 ||
      rpc_write_start_request_on(conn, RPC_cuArrayGetMemoryRequirements,
                                 RPC_SCOPE_NONE) < 0 ||
      rpc_write(conn, &memoryRequirements_present, sizeof(uint8_t)) < 0 ||
      (memoryRequirements_present &&
       rpc_write(conn, memoryRequirements,
                 sizeof(CUDA_ARRAY_MEMORY_REQUIREMENTS)) < 0) ||
      rpc_write(conn, &array, sizeof(CUarray)) < 0 ||
      rpc_write(conn, &device, sizeof(CUdevice)) < 0 ||
      rpc_wait_for_response(conn) < 0 ||
      (memoryRequirements_present &&
       rpc_read(conn, memoryRequirements,
                sizeof(CUDA_ARRAY_MEMORY_REQUIREMENTS)) < 0) ||
      rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
      rpc_read_end(conn) < 0)
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  return return_value;
}

CUresult cuMipmappedArrayGetMemoryRequirements(
    CUDA_ARRAY_MEMORY_REQUIREMENTS *memoryRequirements, CUmipmappedArray mipmap,
    CUdevice device) {
  lupine_route route = lupine_route_for_device(&device);
  if (route.kind == LUPINE_ROUTE_UNKNOWN_DEVICE)
    return CUDA_ERROR_INVALID_DEVICE;
  CUresult return_value;
  if (lupine_route_is_local(route))
    return lupine_call_real_cuda_fn(
        LUPINE_REAL_CUDA_SYMBOL("cuMipmappedArrayGetMemoryRequirements"),
        memoryRequirements, mipmap, device);
  conn_t *conn = lupine_route_remote_conn(route);
  uint8_t memoryRequirements_present = memoryRequirements != nullptr;
  if (lupine_prepare_rpc(conn) < 0 ||
      rpc_write_start_request_on(conn,
                                 RPC_cuMipmappedArrayGetMemoryRequirements,
                                 RPC_SCOPE_NONE) < 0 ||
      rpc_write(conn, &memoryRequirements_present, sizeof(uint8_t)) < 0 ||
      (memoryRequirements_present &&
       rpc_write(conn, memoryRequirements,
                 sizeof(CUDA_ARRAY_MEMORY_REQUIREMENTS)) < 0) ||
      rpc_write(conn, &mipmap, sizeof(CUmipmappedArray)) < 0 ||
      rpc_write(conn, &device, sizeof(CUdevice)) < 0 ||
      rpc_wait_for_response(conn) < 0 ||
      (memoryRequirements_present &&
       rpc_read(conn, memoryRequirements,
                sizeof(CUDA_ARRAY_MEMORY_REQUIREMENTS)) < 0) ||
      rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
      rpc_read_end(conn) < 0)
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  return return_value;
}

CUresult cuArrayGetPlane(CUarray *pPlaneArray, CUarray hArray,
                         unsigned int planeIdx) {
  lupine_route route = lupine_route_for_default();
  CUresult return_value;
  if (lupine_route_is_local(route))
    return lupine_call_real_cuda_fn(LUPINE_REAL_CUDA_SYMBOL("cuArrayGetPlane"),
                                    pPlaneArray, hArray, planeIdx);
  conn_t *conn = lupine_route_remote_conn(route);
  uint8_t pPlaneArray_present = pPlaneArray != nullptr;
  if (lupine_prepare_rpc(conn) < 0 ||
      rpc_write_start_request_on(conn, RPC_cuArrayGetPlane, RPC_SCOPE_NONE) <
          0 ||
      rpc_write(conn, &pPlaneArray_present, sizeof(uint8_t)) < 0 ||
      (pPlaneArray_present &&
       rpc_write(conn, pPlaneArray, sizeof(CUarray)) < 0) ||
      rpc_write(conn, &hArray, sizeof(CUarray)) < 0 ||
      rpc_write(conn, &planeIdx, sizeof(unsigned int)) < 0 ||
      rpc_wait_for_response(conn) < 0 ||
      (pPlaneArray_present &&
       rpc_read(conn, pPlaneArray, sizeof(CUarray)) < 0) ||
      rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
      rpc_read_end(conn) < 0)
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  return return_value;
}

CUresult cuArrayDestroy(CUarray hArray) {
  lupine_route route = lupine_route_for_default();
  CUresult return_value;
  if (lupine_route_is_local(route))
    return lupine_call_real_cuda_fn(LUPINE_REAL_CUDA_SYMBOL("cuArrayDestroy"),
                                    hArray);
  conn_t *conn = lupine_route_remote_conn(route);
  if (lupine_prepare_rpc(conn) < 0 ||
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
  if (lupine_route_is_local(route))
    return lupine_call_real_cuda_fn(
        LUPINE_REAL_CUDA_SYMBOL("cuArray3DCreate_v2"), pHandle, pAllocateArray);
  conn_t *conn = lupine_route_remote_conn(route);
  uint8_t pHandle_present = pHandle != nullptr;
  uint8_t pAllocateArray_present = pAllocateArray != nullptr;
  if (lupine_prepare_rpc(conn) < 0 ||
      rpc_write_start_request_on(conn, RPC_cuArray3DCreate_v2, RPC_SCOPE_NONE) <
          0 ||
      rpc_write(conn, &pHandle_present, sizeof(uint8_t)) < 0 ||
      (pHandle_present && rpc_write(conn, pHandle, sizeof(CUarray)) < 0) ||
      rpc_write(conn, &pAllocateArray_present, sizeof(uint8_t)) < 0 ||
      (pAllocateArray_present &&
       rpc_write(conn, pAllocateArray, sizeof(const CUDA_ARRAY3D_DESCRIPTOR)) <
           0) ||
      rpc_wait_for_response(conn) < 0 ||
      (pHandle_present && rpc_read(conn, pHandle, sizeof(CUarray)) < 0) ||
      rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
      rpc_read_end(conn) < 0)
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  return return_value;
}

CUresult cuArray3DGetDescriptor_v2(CUDA_ARRAY3D_DESCRIPTOR *pArrayDescriptor,
                                   CUarray hArray) {
  lupine_route route = lupine_route_for_default();
  CUresult return_value;
  if (lupine_route_is_local(route))
    return lupine_call_real_cuda_fn(
        LUPINE_REAL_CUDA_SYMBOL("cuArray3DGetDescriptor_v2"), pArrayDescriptor,
        hArray);
  conn_t *conn = lupine_route_remote_conn(route);
  uint8_t pArrayDescriptor_present = pArrayDescriptor != nullptr;
  if (lupine_prepare_rpc(conn) < 0 ||
      rpc_write_start_request_on(conn, RPC_cuArray3DGetDescriptor_v2,
                                 RPC_SCOPE_NONE) < 0 ||
      rpc_write(conn, &pArrayDescriptor_present, sizeof(uint8_t)) < 0 ||
      (pArrayDescriptor_present &&
       rpc_write(conn, pArrayDescriptor, sizeof(CUDA_ARRAY3D_DESCRIPTOR)) <
           0) ||
      rpc_write(conn, &hArray, sizeof(CUarray)) < 0 ||
      rpc_wait_for_response(conn) < 0 ||
      (pArrayDescriptor_present &&
       rpc_read(conn, pArrayDescriptor, sizeof(CUDA_ARRAY3D_DESCRIPTOR)) < 0) ||
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
  if (lupine_route_is_local(route))
    return lupine_call_real_cuda_fn(
        LUPINE_REAL_CUDA_SYMBOL("cuMipmappedArrayCreate"), pHandle,
        pMipmappedArrayDesc, numMipmapLevels);
  conn_t *conn = lupine_route_remote_conn(route);
  uint8_t pHandle_present = pHandle != nullptr;
  uint8_t pMipmappedArrayDesc_present = pMipmappedArrayDesc != nullptr;
  if (lupine_prepare_rpc(conn) < 0 ||
      rpc_write_start_request_on(conn, RPC_cuMipmappedArrayCreate,
                                 RPC_SCOPE_NONE) < 0 ||
      rpc_write(conn, &pHandle_present, sizeof(uint8_t)) < 0 ||
      (pHandle_present &&
       rpc_write(conn, pHandle, sizeof(CUmipmappedArray)) < 0) ||
      rpc_write(conn, &pMipmappedArrayDesc_present, sizeof(uint8_t)) < 0 ||
      (pMipmappedArrayDesc_present &&
       rpc_write(conn, pMipmappedArrayDesc,
                 sizeof(const CUDA_ARRAY3D_DESCRIPTOR)) < 0) ||
      rpc_write(conn, &numMipmapLevels, sizeof(unsigned int)) < 0 ||
      rpc_wait_for_response(conn) < 0 ||
      (pHandle_present &&
       rpc_read(conn, pHandle, sizeof(CUmipmappedArray)) < 0) ||
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
  if (lupine_route_is_local(route))
    return lupine_call_real_cuda_fn(
        LUPINE_REAL_CUDA_SYMBOL("cuMipmappedArrayGetLevel"), pLevelArray,
        hMipmappedArray, level);
  conn_t *conn = lupine_route_remote_conn(route);
  uint8_t pLevelArray_present = pLevelArray != nullptr;
  if (lupine_prepare_rpc(conn) < 0 ||
      rpc_write_start_request_on(conn, RPC_cuMipmappedArrayGetLevel,
                                 RPC_SCOPE_NONE) < 0 ||
      rpc_write(conn, &pLevelArray_present, sizeof(uint8_t)) < 0 ||
      (pLevelArray_present &&
       rpc_write(conn, pLevelArray, sizeof(CUarray)) < 0) ||
      rpc_write(conn, &hMipmappedArray, sizeof(CUmipmappedArray)) < 0 ||
      rpc_write(conn, &level, sizeof(unsigned int)) < 0 ||
      rpc_wait_for_response(conn) < 0 ||
      (pLevelArray_present &&
       rpc_read(conn, pLevelArray, sizeof(CUarray)) < 0) ||
      rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
      rpc_read_end(conn) < 0)
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  return return_value;
}

CUresult cuMipmappedArrayDestroy(CUmipmappedArray hMipmappedArray) {
  lupine_route route = lupine_route_for_default();
  CUresult return_value;
  if (lupine_route_is_local(route))
    return lupine_call_real_cuda_fn(
        LUPINE_REAL_CUDA_SYMBOL("cuMipmappedArrayDestroy"), hMipmappedArray);
  conn_t *conn = lupine_route_remote_conn(route);
  if (lupine_prepare_rpc(conn) < 0 ||
      rpc_write_start_request(conn, RPC_cuMipmappedArrayDestroy) < 0 ||
      rpc_write(conn, &hMipmappedArray, sizeof(CUmipmappedArray)) < 0 ||
      rpc_wait_for_response(conn) < 0 ||
      rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
      rpc_read_end(conn) < 0)
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  return return_value;
}

CUresult cuMemGetHandleForAddressRange(void *handle, CUdeviceptr dptr,
                                       size_t size,
                                       CUmemRangeHandleType handleType,
                                       unsigned long long flags) {
  lupine_route route = lupine_route_for_deviceptr(dptr);
  CUresult return_value;
  if (lupine_route_is_local(route))
    return lupine_call_real_cuda_fn(
        LUPINE_REAL_CUDA_SYMBOL("cuMemGetHandleForAddressRange"), handle, dptr,
        size, handleType, flags);
  conn_t *conn = lupine_route_remote_conn(route);
  uint8_t handle_null = handle == nullptr ? 1 : 0;
  if (lupine_prepare_rpc(conn) < 0 ||
      rpc_write_start_request_on(conn, RPC_cuMemGetHandleForAddressRange,
                                 RPC_SCOPE_NONE) < 0 ||
      rpc_write(conn, &dptr, sizeof(CUdeviceptr)) < 0 ||
      rpc_write(conn, &size, sizeof(size_t)) < 0 ||
      rpc_write(conn, &handleType, sizeof(CUmemRangeHandleType)) < 0 ||
      rpc_write(conn, &flags, sizeof(unsigned long long)) < 0 ||
      rpc_write(conn, &handle_null, sizeof(uint8_t)) < 0 ||
      rpc_wait_for_response(conn) < 0 ||
      (!handle_null && 4 != 0 && rpc_read(conn, handle, 4) < 0) ||
      rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
      rpc_read_end(conn) < 0)
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  return return_value;
}

CUresult cuMemAddressReserve(CUdeviceptr *ptr, size_t size, size_t alignment,
                             CUdeviceptr addr, unsigned long long flags) {
  lupine_route route = lupine_route_for_deviceptr(addr);
  CUresult return_value;
  if (lupine_route_is_local(route))
    return lupine_call_real_cuda_fn(
        LUPINE_REAL_CUDA_SYMBOL("cuMemAddressReserve"), ptr, size, alignment,
        addr, flags);
  conn_t *conn = lupine_route_remote_conn(route);
  uint8_t ptr_present = ptr != nullptr;
  if (lupine_prepare_rpc(conn) < 0 ||
      rpc_write_start_request_on(conn, RPC_cuMemAddressReserve,
                                 RPC_SCOPE_NONE) < 0 ||
      rpc_write(conn, &ptr_present, sizeof(uint8_t)) < 0 ||
      (ptr_present && rpc_write(conn, ptr, sizeof(CUdeviceptr)) < 0) ||
      rpc_write(conn, &size, sizeof(size_t)) < 0 ||
      rpc_write(conn, &alignment, sizeof(size_t)) < 0 ||
      rpc_write(conn, &addr, sizeof(CUdeviceptr)) < 0 ||
      rpc_write(conn, &flags, sizeof(unsigned long long)) < 0 ||
      rpc_wait_for_response(conn) < 0 ||
      (ptr_present && rpc_read(conn, ptr, sizeof(CUdeviceptr)) < 0) ||
      rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
      rpc_read_end(conn) < 0)
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  return return_value;
}

CUresult cuMemAddressFree(CUdeviceptr ptr, size_t size) {
  lupine_route route = lupine_route_for_deviceptr(ptr);
  CUresult return_value;
  if (lupine_route_is_local(route))
    return lupine_call_real_cuda_fn(LUPINE_REAL_CUDA_SYMBOL("cuMemAddressFree"),
                                    ptr, size);
  conn_t *conn = lupine_route_remote_conn(route);
  if (lupine_prepare_rpc(conn) < 0 ||
      rpc_write_start_request_on(conn, RPC_cuMemAddressFree, RPC_SCOPE_NONE) <
          0 ||
      rpc_write(conn, &ptr, sizeof(CUdeviceptr)) < 0 ||
      rpc_write(conn, &size, sizeof(size_t)) < 0 || rpc_write_end(conn) < 0)
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  return_value = CUDA_SUCCESS;
  return return_value;
}

CUresult cuMemCreate(CUmemGenericAllocationHandle *handle, size_t size,
                     const CUmemAllocationProp *prop,
                     unsigned long long flags) {
  lupine_route route = lupine_route_for_default();
  CUresult return_value;
  if (lupine_route_is_local(route))
    return lupine_call_real_cuda_fn(LUPINE_REAL_CUDA_SYMBOL("cuMemCreate"),
                                    handle, size, prop, flags);
  conn_t *conn = lupine_route_remote_conn(route);
  uint8_t handle_present = handle != nullptr;
  uint8_t prop_present = prop != nullptr;
  if (lupine_prepare_rpc(conn) < 0 ||
      rpc_write_start_request_on(conn, RPC_cuMemCreate, RPC_SCOPE_NONE) < 0 ||
      rpc_write(conn, &handle_present, sizeof(uint8_t)) < 0 ||
      (handle_present &&
       rpc_write(conn, handle, sizeof(CUmemGenericAllocationHandle)) < 0) ||
      rpc_write(conn, &size, sizeof(size_t)) < 0 ||
      rpc_write(conn, &prop_present, sizeof(uint8_t)) < 0 ||
      (prop_present &&
       rpc_write(conn, prop, sizeof(const CUmemAllocationProp)) < 0) ||
      rpc_write(conn, &flags, sizeof(unsigned long long)) < 0 ||
      rpc_wait_for_response(conn) < 0 ||
      (handle_present &&
       rpc_read(conn, handle, sizeof(CUmemGenericAllocationHandle)) < 0) ||
      rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
      rpc_read_end(conn) < 0)
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  return return_value;
}

CUresult cuMemRelease(CUmemGenericAllocationHandle handle) {
  lupine_route route = lupine_route_for_default();
  CUresult return_value;
  if (lupine_route_is_local(route))
    return lupine_call_real_cuda_fn(LUPINE_REAL_CUDA_SYMBOL("cuMemRelease"),
                                    handle);
  conn_t *conn = lupine_route_remote_conn(route);
  if (lupine_prepare_rpc(conn) < 0 ||
      rpc_write_start_request_on(conn, RPC_cuMemRelease, RPC_SCOPE_NONE) < 0 ||
      rpc_write(conn, &handle, sizeof(CUmemGenericAllocationHandle)) < 0 ||
      rpc_write_end(conn) < 0)
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  return_value = CUDA_SUCCESS;
  return return_value;
}

CUresult cuMemMap(CUdeviceptr ptr, size_t size, size_t offset,
                  CUmemGenericAllocationHandle handle,
                  unsigned long long flags) {
  lupine_route route = lupine_route_for_deviceptr(ptr);
  CUresult return_value;
  if (lupine_route_is_local(route))
    return lupine_call_real_cuda_fn(LUPINE_REAL_CUDA_SYMBOL("cuMemMap"), ptr,
                                    size, offset, handle, flags);
  conn_t *conn = lupine_route_remote_conn(route);
  if (lupine_prepare_rpc(conn) < 0 ||
      rpc_write_start_request_on(conn, RPC_cuMemMap, RPC_SCOPE_NONE) < 0 ||
      rpc_write(conn, &ptr, sizeof(CUdeviceptr)) < 0 ||
      rpc_write(conn, &size, sizeof(size_t)) < 0 ||
      rpc_write(conn, &offset, sizeof(size_t)) < 0 ||
      rpc_write(conn, &handle, sizeof(CUmemGenericAllocationHandle)) < 0 ||
      rpc_write(conn, &flags, sizeof(unsigned long long)) < 0 ||
      rpc_write_end(conn) < 0)
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  return_value = CUDA_SUCCESS;
  return return_value;
}

CUresult cuMemMapArrayAsync(CUarrayMapInfo *mapInfoList, unsigned int count,
                            CUstream hStream) {
  lupine_route route = (hStream != nullptr ? lupine_route_for_stream(hStream)
                                           : lupine_route_for_default());
  CUresult return_value;
  if (lupine_route_is_local(route))
    return lupine_call_real_cuda_fn(
        LUPINE_REAL_CUDA_SYMBOL("cuMemMapArrayAsync"), mapInfoList, count,
        hStream);
  conn_t *conn = lupine_route_remote_conn(route);
  uint8_t mapInfoList_null = mapInfoList == nullptr ? 1 : 0;
  if (lupine_prepare_rpc(conn) < 0 ||
      lupine_rpc_start_on_stream(conn, RPC_cuMemMapArrayAsync, hStream) < 0 ||
      rpc_write(conn, &count, sizeof(unsigned int)) < 0 ||
      rpc_write(conn, &mapInfoList_null, sizeof(uint8_t)) < 0 ||
      (!mapInfoList_null && count * sizeof(CUarrayMapInfo) != 0 &&
       rpc_write(conn, mapInfoList, count * sizeof(CUarrayMapInfo)) < 0) ||
      rpc_write(conn, &hStream, sizeof(CUstream)) < 0 ||
      rpc_write_end(conn) < 0)
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  return_value = CUDA_SUCCESS;
  return return_value;
}

CUresult cuMemUnmap(CUdeviceptr ptr, size_t size) {
  lupine_route route = lupine_route_for_deviceptr(ptr);
  CUresult return_value;
  if (lupine_route_is_local(route))
    return lupine_call_real_cuda_fn(LUPINE_REAL_CUDA_SYMBOL("cuMemUnmap"), ptr,
                                    size);
  conn_t *conn = lupine_route_remote_conn(route);
  if (lupine_prepare_rpc(conn) < 0 ||
      rpc_write_start_request_on(conn, RPC_cuMemUnmap, RPC_SCOPE_NONE) < 0 ||
      rpc_write(conn, &ptr, sizeof(CUdeviceptr)) < 0 ||
      rpc_write(conn, &size, sizeof(size_t)) < 0 || rpc_write_end(conn) < 0)
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  return_value = CUDA_SUCCESS;
  return return_value;
}

CUresult cuMemSetAccess(CUdeviceptr ptr, size_t size,
                        const CUmemAccessDesc *desc, size_t count) {
  lupine_route route = lupine_route_for_deviceptr(ptr);
  CUresult return_value;
  if (lupine_route_is_local(route))
    return lupine_call_real_cuda_fn(LUPINE_REAL_CUDA_SYMBOL("cuMemSetAccess"),
                                    ptr, size, desc, count);
  conn_t *conn = lupine_route_remote_conn(route);
  uint8_t desc_null = desc == nullptr ? 1 : 0;
  if (lupine_prepare_rpc(conn) < 0 ||
      rpc_write_start_request_on(conn, RPC_cuMemSetAccess, RPC_SCOPE_NONE) <
          0 ||
      rpc_write(conn, &ptr, sizeof(CUdeviceptr)) < 0 ||
      rpc_write(conn, &size, sizeof(size_t)) < 0 ||
      rpc_write(conn, &count, sizeof(size_t)) < 0 ||
      rpc_write(conn, &desc_null, sizeof(uint8_t)) < 0 ||
      (!desc_null && count * sizeof(const CUmemAccessDesc) != 0 &&
       rpc_write(conn, desc, count * sizeof(const CUmemAccessDesc)) < 0) ||
      rpc_write_end(conn) < 0)
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  return_value = CUDA_SUCCESS;
  return return_value;
}

CUresult cuMemGetAccess(unsigned long long *flags,
                        const CUmemLocation *location, CUdeviceptr ptr) {
  lupine_route route = lupine_route_for_deviceptr(ptr);
  CUresult return_value;
  if (lupine_route_is_local(route))
    return lupine_call_real_cuda_fn(LUPINE_REAL_CUDA_SYMBOL("cuMemGetAccess"),
                                    flags, location, ptr);
  conn_t *conn = lupine_route_remote_conn(route);
  uint8_t flags_present = flags != nullptr;
  uint8_t location_present = location != nullptr;
  if (lupine_prepare_rpc(conn) < 0 ||
      rpc_write_start_request_on(conn, RPC_cuMemGetAccess, RPC_SCOPE_NONE) <
          0 ||
      rpc_write(conn, &flags_present, sizeof(uint8_t)) < 0 ||
      (flags_present &&
       rpc_write(conn, flags, sizeof(unsigned long long)) < 0) ||
      rpc_write(conn, &location_present, sizeof(uint8_t)) < 0 ||
      (location_present &&
       rpc_write(conn, location, sizeof(const CUmemLocation)) < 0) ||
      rpc_write(conn, &ptr, sizeof(CUdeviceptr)) < 0 ||
      rpc_wait_for_response(conn) < 0 ||
      (flags_present &&
       rpc_read(conn, flags, sizeof(unsigned long long)) < 0) ||
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
  if (lupine_route_is_local(route))
    return lupine_call_real_cuda_fn(
        LUPINE_REAL_CUDA_SYMBOL("cuMemGetAllocationGranularity"), granularity,
        prop, option);
  conn_t *conn = lupine_route_remote_conn(route);
  uint8_t granularity_present = granularity != nullptr;
  uint8_t prop_present = prop != nullptr;
  if (lupine_prepare_rpc(conn) < 0 ||
      rpc_write_start_request_on(conn, RPC_cuMemGetAllocationGranularity,
                                 RPC_SCOPE_NONE) < 0 ||
      rpc_write(conn, &granularity_present, sizeof(uint8_t)) < 0 ||
      (granularity_present &&
       rpc_write(conn, granularity, sizeof(size_t)) < 0) ||
      rpc_write(conn, &prop_present, sizeof(uint8_t)) < 0 ||
      (prop_present &&
       rpc_write(conn, prop, sizeof(const CUmemAllocationProp)) < 0) ||
      rpc_write(conn, &option, sizeof(CUmemAllocationGranularity_flags)) < 0 ||
      rpc_wait_for_response(conn) < 0 ||
      (granularity_present &&
       rpc_read(conn, granularity, sizeof(size_t)) < 0) ||
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
  if (lupine_route_is_local(route))
    return lupine_call_real_cuda_fn(
        LUPINE_REAL_CUDA_SYMBOL("cuMemGetAllocationPropertiesFromHandle"), prop,
        handle);
  conn_t *conn = lupine_route_remote_conn(route);
  uint8_t prop_present = prop != nullptr;
  if (lupine_prepare_rpc(conn) < 0 ||
      rpc_write_start_request_on(conn,
                                 RPC_cuMemGetAllocationPropertiesFromHandle,
                                 RPC_SCOPE_NONE) < 0 ||
      rpc_write(conn, &prop_present, sizeof(uint8_t)) < 0 ||
      (prop_present &&
       rpc_write(conn, prop, sizeof(CUmemAllocationProp)) < 0) ||
      rpc_write(conn, &handle, sizeof(CUmemGenericAllocationHandle)) < 0 ||
      rpc_wait_for_response(conn) < 0 ||
      (prop_present && rpc_read(conn, prop, sizeof(CUmemAllocationProp)) < 0) ||
      rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
      rpc_read_end(conn) < 0)
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  return return_value;
}

CUresult cuMemRetainAllocationHandle(CUmemGenericAllocationHandle *handle,
                                     void *addr) {
  lupine_route route = lupine_route_for_default();
  CUresult return_value;
  if (lupine_route_is_local(route))
    return lupine_call_real_cuda_fn(
        LUPINE_REAL_CUDA_SYMBOL("cuMemRetainAllocationHandle"), handle, addr);
  conn_t *conn = lupine_route_remote_conn(route);
  uint8_t handle_present = handle != nullptr;
  if (lupine_prepare_rpc(conn) < 0 ||
      rpc_write_start_request_on(conn, RPC_cuMemRetainAllocationHandle,
                                 RPC_SCOPE_NONE) < 0 ||
      rpc_write(conn, &handle_present, sizeof(uint8_t)) < 0 ||
      (handle_present &&
       rpc_write(conn, handle, sizeof(CUmemGenericAllocationHandle)) < 0) ||
      rpc_write(conn, &addr, sizeof(void *)) < 0 ||
      rpc_wait_for_response(conn) < 0 ||
      (handle_present &&
       rpc_read(conn, handle, sizeof(CUmemGenericAllocationHandle)) < 0) ||
      rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
      rpc_read_end(conn) < 0)
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  return return_value;
}

CUresult cuMemFreeAsync(CUdeviceptr dptr, CUstream hStream) {
  lupine_route route = lupine_route_for_deviceptr(dptr);
  CUresult return_value;
  if (lupine_route_is_local(route)) {
    return_value = lupine_call_real_cuda_fn(
        LUPINE_REAL_CUDA_SYMBOL("cuMemFreeAsync"), dptr, hStream);
    if (return_value == CUDA_SUCCESS)
      lupine_forget_deviceptr_owner(dptr);
    return return_value;
  }
  conn_t *conn = lupine_route_remote_conn(route);
  if (lupine_prepare_rpc(conn) < 0 ||
      lupine_rpc_start_on_stream(conn, RPC_cuMemFreeAsync, hStream) < 0 ||
      rpc_write(conn, &dptr, sizeof(CUdeviceptr)) < 0 ||
      rpc_write(conn, &hStream, sizeof(CUstream)) < 0 ||
      rpc_write_end(conn) < 0)
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  return_value = CUDA_SUCCESS;
  if (return_value == CUDA_SUCCESS)
    lupine_forget_deviceptr_owner(dptr);
  return return_value;
}

CUresult cuMemAllocAsync(CUdeviceptr *dptr, size_t bytesize, CUstream hStream) {
  lupine_route route = (hStream != nullptr ? lupine_route_for_stream(hStream)
                                           : lupine_route_for_default());
  CUresult return_value;
  if (lupine_route_is_local(route)) {
    return_value = lupine_call_real_cuda_fn(
        LUPINE_REAL_CUDA_SYMBOL("cuMemAllocAsync"), dptr, bytesize, hStream);
    if (return_value == CUDA_SUCCESS && dptr != nullptr) {
      lupine_note_deviceptr_owner_route(*dptr, route);
    }
    if (return_value == CUDA_SUCCESS && dptr != nullptr)
      lupine_note_deviceptr_allocation_route(*dptr, bytesize, route);
    return return_value;
  }
  conn_t *conn = lupine_route_remote_conn(route);
  uint8_t dptr_present = dptr != nullptr;
  if (lupine_prepare_rpc(conn) < 0 ||
      lupine_rpc_start_on_stream(conn, RPC_cuMemAllocAsync, hStream) < 0 ||
      rpc_write(conn, &dptr_present, sizeof(uint8_t)) < 0 ||
      (dptr_present && rpc_write(conn, dptr, sizeof(CUdeviceptr)) < 0) ||
      rpc_write(conn, &bytesize, sizeof(size_t)) < 0 ||
      rpc_write(conn, &hStream, sizeof(CUstream)) < 0 ||
      rpc_wait_for_response(conn) < 0 ||
      (dptr_present && rpc_read(conn, dptr, sizeof(CUdeviceptr)) < 0) ||
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
  if (lupine_route_is_local(route))
    return lupine_call_real_cuda_fn(LUPINE_REAL_CUDA_SYMBOL("cuMemPoolTrimTo"),
                                    pool, minBytesToKeep);
  conn_t *conn = lupine_route_remote_conn(route);
  if (lupine_prepare_rpc(conn) < 0 ||
      rpc_write_start_request_on(conn, RPC_cuMemPoolTrimTo, RPC_SCOPE_NONE) <
          0 ||
      rpc_write(conn, &pool, sizeof(CUmemoryPool)) < 0 ||
      rpc_write(conn, &minBytesToKeep, sizeof(size_t)) < 0 ||
      rpc_write_end(conn) < 0)
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  return_value = CUDA_SUCCESS;
  return return_value;
}

CUresult cuMemPoolSetAttribute(CUmemoryPool pool, CUmemPool_attribute attr,
                               void *value) {
  lupine_route route = lupine_route_for_memory_pool(pool);
  size_t value_size = 0;
  if (value == nullptr || !lupine_mem_pool_attribute_size(attr, &value_size))
    return CUDA_ERROR_INVALID_VALUE;
  cuuint64_t staged = 0;
  memcpy(&staged, value, value_size);
  value = &staged;
  CUresult return_value;
  if (lupine_route_is_local(route))
    return lupine_call_real_cuda_fn(
        LUPINE_REAL_CUDA_SYMBOL("cuMemPoolSetAttribute"), pool, attr, value);
  conn_t *conn = lupine_route_remote_conn(route);
  uint8_t value_null = value == nullptr ? 1 : 0;
  if (lupine_prepare_rpc(conn) < 0 ||
      rpc_write_start_request_on(conn, RPC_cuMemPoolSetAttribute,
                                 RPC_SCOPE_NONE) < 0 ||
      rpc_write(conn, &pool, sizeof(CUmemoryPool)) < 0 ||
      rpc_write(conn, &attr, sizeof(CUmemPool_attribute)) < 0 ||
      rpc_write(conn, &value_null, sizeof(uint8_t)) < 0 ||
      (!value_null && 8 != 0 && rpc_write(conn, value, 8) < 0) ||
      rpc_write_end(conn) < 0)
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  return_value = CUDA_SUCCESS;
  return return_value;
}

CUresult cuMemPoolGetAttribute(CUmemoryPool pool, CUmemPool_attribute attr,
                               void *value) {
  lupine_route route = lupine_route_for_memory_pool(pool);
  size_t value_size = 0;
  if (value == nullptr || !lupine_mem_pool_attribute_size(attr, &value_size))
    return CUDA_ERROR_INVALID_VALUE;
  void *caller_value = value;
  cuuint64_t staged = 0;
  value = &staged;
  CUresult return_value;
  if (lupine_route_is_local(route)) {
    return_value = lupine_call_real_cuda_fn(
        LUPINE_REAL_CUDA_SYMBOL("cuMemPoolGetAttribute"), pool, attr, value);
    if (return_value == CUDA_SUCCESS)
      memcpy(caller_value, &staged, value_size);
    return return_value;
  }
  conn_t *conn = lupine_route_remote_conn(route);
  uint8_t value_null = value == nullptr ? 1 : 0;
  if (lupine_prepare_rpc(conn) < 0 ||
      rpc_write_start_request_on(conn, RPC_cuMemPoolGetAttribute,
                                 RPC_SCOPE_NONE) < 0 ||
      rpc_write(conn, &pool, sizeof(CUmemoryPool)) < 0 ||
      rpc_write(conn, &attr, sizeof(CUmemPool_attribute)) < 0 ||
      rpc_write(conn, &value_null, sizeof(uint8_t)) < 0 ||
      rpc_wait_for_response(conn) < 0 ||
      (!value_null && 8 != 0 && rpc_read(conn, value, 8) < 0) ||
      rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
      rpc_read_end(conn) < 0)
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  if (return_value == CUDA_SUCCESS)
    memcpy(caller_value, &staged, value_size);
  return return_value;
}

CUresult cuMemPoolSetAccess(CUmemoryPool pool, const CUmemAccessDesc *map,
                            size_t count) {
  lupine_route route = lupine_route_for_memory_pool(pool);
  if (count > SIZE_MAX / sizeof(*map) || (count != 0 && map == nullptr))
    return CUDA_ERROR_INVALID_VALUE;
  std::vector<CUmemAccessDesc> route_map;
  if (count != 0)
    route_map.assign(map, map + count);
  for (auto &desc : route_map) {
    CUresult status = lupine_translate_mem_location(route, desc.location);
    if (status != CUDA_SUCCESS)
      return status;
  }
  map = route_map.data();
  CUresult return_value;
  if (lupine_route_is_local(route))
    return lupine_call_real_cuda_fn(
        LUPINE_REAL_CUDA_SYMBOL("cuMemPoolSetAccess"), pool, map, count);
  conn_t *conn = lupine_route_remote_conn(route);
  uint8_t map_null = map == nullptr ? 1 : 0;
  if (lupine_prepare_rpc(conn) < 0 ||
      rpc_write_start_request_on(conn, RPC_cuMemPoolSetAccess, RPC_SCOPE_NONE) <
          0 ||
      rpc_write(conn, &pool, sizeof(CUmemoryPool)) < 0 ||
      rpc_write(conn, &count, sizeof(size_t)) < 0 ||
      rpc_write(conn, &map_null, sizeof(uint8_t)) < 0 ||
      (!map_null && count * sizeof(const CUmemAccessDesc) != 0 &&
       rpc_write(conn, map, count * sizeof(const CUmemAccessDesc)) < 0) ||
      rpc_write_end(conn) < 0)
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  return_value = CUDA_SUCCESS;
  return return_value;
}

CUresult cuMemPoolGetAccess(CUmemAccess_flags *flags, CUmemoryPool memPool,
                            CUmemLocation *location) {
  lupine_route route = lupine_route_for_memory_pool(memPool);
  if (flags == nullptr || location == nullptr)
    return CUDA_ERROR_INVALID_VALUE;
  CUmemLocation route_location = *location;
  CUresult status = lupine_translate_mem_location(route, route_location);
  if (status != CUDA_SUCCESS)
    return status;
  location = &route_location;
  CUresult return_value;
  if (lupine_route_is_local(route))
    return lupine_call_real_cuda_fn(
        LUPINE_REAL_CUDA_SYMBOL("cuMemPoolGetAccess"), flags, memPool,
        location);
  conn_t *conn = lupine_route_remote_conn(route);
  uint8_t flags_present = flags != nullptr;
  uint8_t location_present = location != nullptr;
  if (lupine_prepare_rpc(conn) < 0 ||
      rpc_write_start_request_on(conn, RPC_cuMemPoolGetAccess, RPC_SCOPE_NONE) <
          0 ||
      rpc_write(conn, &flags_present, sizeof(uint8_t)) < 0 ||
      (flags_present &&
       rpc_write(conn, flags, sizeof(CUmemAccess_flags)) < 0) ||
      rpc_write(conn, &memPool, sizeof(CUmemoryPool)) < 0 ||
      rpc_write(conn, &location_present, sizeof(uint8_t)) < 0 ||
      (location_present &&
       rpc_write(conn, location, sizeof(CUmemLocation)) < 0) ||
      rpc_wait_for_response(conn) < 0 ||
      (flags_present && rpc_read(conn, flags, sizeof(CUmemAccess_flags)) < 0) ||
      rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
      rpc_read_end(conn) < 0)
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  return return_value;
}

CUresult cuMemPoolCreate(CUmemoryPool *pool, const CUmemPoolProps *poolProps) {
  lupine_route route = lupine_route_for_default();
  if (pool == nullptr || poolProps == nullptr)
    return CUDA_ERROR_INVALID_VALUE;
  CUmemPoolProps route_props = *poolProps;
  if (route_props.location.type == CU_MEM_LOCATION_TYPE_DEVICE) {
    CUdevice device = route_props.location.id;
    route = lupine_route_for_device(&device);
    if (route.kind == LUPINE_ROUTE_UNKNOWN_DEVICE)
      return CUDA_ERROR_INVALID_DEVICE;
    route_props.location.id = device;
  }
  poolProps = &route_props;
  CUresult return_value;
  if (lupine_route_is_local(route)) {
    return_value = lupine_call_real_cuda_fn(
        LUPINE_REAL_CUDA_SYMBOL("cuMemPoolCreate"), pool, poolProps);
    if (return_value == CUDA_SUCCESS && pool != nullptr) {
      lupine_note_memory_pool_owner_route(*pool, route);
    }
    return return_value;
  }
  conn_t *conn = lupine_route_remote_conn(route);
  uint8_t pool_present = pool != nullptr;
  uint8_t poolProps_present = poolProps != nullptr;
  if (lupine_prepare_rpc(conn) < 0 ||
      rpc_write_start_request_on(conn, RPC_cuMemPoolCreate, RPC_SCOPE_NONE) <
          0 ||
      rpc_write(conn, &pool_present, sizeof(uint8_t)) < 0 ||
      (pool_present && rpc_write(conn, pool, sizeof(CUmemoryPool)) < 0) ||
      rpc_write(conn, &poolProps_present, sizeof(uint8_t)) < 0 ||
      (poolProps_present &&
       rpc_write(conn, poolProps, sizeof(const CUmemPoolProps)) < 0) ||
      rpc_wait_for_response(conn) < 0 ||
      (pool_present && rpc_read(conn, pool, sizeof(CUmemoryPool)) < 0) ||
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
  if (lupine_route_is_local(route))
    return lupine_call_real_cuda_fn(LUPINE_REAL_CUDA_SYMBOL("cuMemPoolDestroy"),
                                    pool);
  conn_t *conn = lupine_route_remote_conn(route);
  if (lupine_prepare_rpc(conn) < 0 ||
      rpc_write_start_request_on(conn, RPC_cuMemPoolDestroy, RPC_SCOPE_NONE) <
          0 ||
      rpc_write(conn, &pool, sizeof(CUmemoryPool)) < 0 ||
      rpc_write_end(conn) < 0)
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  return_value = CUDA_SUCCESS;
  return return_value;
}

CUresult cuMemAllocFromPoolAsync(CUdeviceptr *dptr, size_t bytesize,
                                 CUmemoryPool pool, CUstream hStream) {
  lupine_route route = (hStream != nullptr ? lupine_route_for_stream(hStream)
                                           : lupine_route_for_default());
  CUresult return_value;
  if (lupine_route_is_local(route)) {
    return_value = lupine_call_real_cuda_fn(
        LUPINE_REAL_CUDA_SYMBOL("cuMemAllocFromPoolAsync"), dptr, bytesize,
        pool, hStream);
    if (return_value == CUDA_SUCCESS && dptr != nullptr) {
      lupine_note_deviceptr_owner_route(*dptr, route);
    }
    if (return_value == CUDA_SUCCESS && dptr != nullptr)
      lupine_note_deviceptr_allocation_route(*dptr, bytesize, route);
    return return_value;
  }
  conn_t *conn = lupine_route_remote_conn(route);
  uint8_t dptr_present = dptr != nullptr;
  if (lupine_prepare_rpc(conn) < 0 ||
      lupine_rpc_start_on_stream(conn, RPC_cuMemAllocFromPoolAsync, hStream) <
          0 ||
      rpc_write(conn, &dptr_present, sizeof(uint8_t)) < 0 ||
      (dptr_present && rpc_write(conn, dptr, sizeof(CUdeviceptr)) < 0) ||
      rpc_write(conn, &bytesize, sizeof(size_t)) < 0 ||
      rpc_write(conn, &pool, sizeof(CUmemoryPool)) < 0 ||
      rpc_write(conn, &hStream, sizeof(CUstream)) < 0 ||
      rpc_wait_for_response(conn) < 0 ||
      (dptr_present && rpc_read(conn, dptr, sizeof(CUdeviceptr)) < 0) ||
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
  if (lupine_route_is_local(route))
    return lupine_call_real_cuda_fn(
        LUPINE_REAL_CUDA_SYMBOL("cuMemPoolExportPointer"), shareData_out, ptr);
  conn_t *conn = lupine_route_remote_conn(route);
  uint8_t shareData_out_present = shareData_out != nullptr;
  if (lupine_prepare_rpc(conn) < 0 ||
      rpc_write_start_request_on(conn, RPC_cuMemPoolExportPointer,
                                 RPC_SCOPE_NONE) < 0 ||
      rpc_write(conn, &shareData_out_present, sizeof(uint8_t)) < 0 ||
      (shareData_out_present &&
       rpc_write(conn, shareData_out, sizeof(CUmemPoolPtrExportData)) < 0) ||
      rpc_write(conn, &ptr, sizeof(CUdeviceptr)) < 0 ||
      rpc_wait_for_response(conn) < 0 ||
      (shareData_out_present &&
       rpc_read(conn, shareData_out, sizeof(CUmemPoolPtrExportData)) < 0) ||
      rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
      rpc_read_end(conn) < 0)
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  return return_value;
}

CUresult cuMemPoolImportPointer(CUdeviceptr *ptr_out, CUmemoryPool pool,
                                CUmemPoolPtrExportData *shareData) {
  lupine_route route = lupine_route_for_memory_pool(pool);
  CUresult return_value;
  if (lupine_route_is_local(route))
    return lupine_call_real_cuda_fn(
        LUPINE_REAL_CUDA_SYMBOL("cuMemPoolImportPointer"), ptr_out, pool,
        shareData);
  conn_t *conn = lupine_route_remote_conn(route);
  uint8_t ptr_out_present = ptr_out != nullptr;
  uint8_t shareData_present = shareData != nullptr;
  if (lupine_prepare_rpc(conn) < 0 ||
      rpc_write_start_request_on(conn, RPC_cuMemPoolImportPointer,
                                 RPC_SCOPE_NONE) < 0 ||
      rpc_write(conn, &ptr_out_present, sizeof(uint8_t)) < 0 ||
      (ptr_out_present && rpc_write(conn, ptr_out, sizeof(CUdeviceptr)) < 0) ||
      rpc_write(conn, &pool, sizeof(CUmemoryPool)) < 0 ||
      rpc_write(conn, &shareData_present, sizeof(uint8_t)) < 0 ||
      (shareData_present &&
       rpc_write(conn, shareData, sizeof(CUmemPoolPtrExportData)) < 0) ||
      rpc_wait_for_response(conn) < 0 ||
      (ptr_out_present && rpc_read(conn, ptr_out, sizeof(CUdeviceptr)) < 0) ||
      (shareData_present &&
       rpc_read(conn, shareData, sizeof(CUmemPoolPtrExportData)) < 0) ||
      rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
      rpc_read_end(conn) < 0)
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  return return_value;
}

CUresult cuPointerGetAttribute(void *data, CUpointer_attribute attribute,
                               CUdeviceptr ptr) {
  lupine_route route = lupine_route_for_deviceptr(ptr);
  size_t value_size = 0;
  if (data == nullptr || !lupine_pointer_attribute_size(attribute, &value_size))
    return CUDA_ERROR_INVALID_VALUE;
  lupine_pointer_query_context query = lupine_resolve_pointer_query(ptr);
  ptr = query.query_ptr;
  route = lupine_route_for_deviceptr(ptr);
  if (!lupine_route_is_local(route) && query.managed_alias &&
      attribute == CU_POINTER_ATTRIBUTE_HOST_POINTER) {
    void *host = reinterpret_cast<void *>(query.orig_ptr);
    memcpy(data, &host, sizeof(host));
    return CUDA_SUCCESS;
  }
  void *caller_data = data;
  unsigned char staged[16] = {};
  memcpy(staged, caller_data, value_size);
  data = staged;
  CUresult return_value;
  if (lupine_route_is_local(route)) {
    return_value = lupine_call_real_cuda_fn(
        LUPINE_REAL_CUDA_SYMBOL("cuPointerGetAttribute"), data, attribute, ptr);
    if (return_value == CUDA_SUCCESS && !lupine_route_is_local(route))
      lupine_fixup_pointer_attribute_value(query, attribute, staged);
    memcpy(caller_data, staged, value_size);
    return return_value;
  }
  conn_t *conn = lupine_route_remote_conn(route);
  uint8_t data_null = data == nullptr ? 1 : 0;
  if (lupine_prepare_rpc(conn) < 0 ||
      rpc_write_start_request_on(conn, RPC_cuPointerGetAttribute,
                                 RPC_SCOPE_NONE) < 0 ||
      rpc_write(conn, &data_null, sizeof(uint8_t)) < 0 ||
      (!data_null && 16 != 0 && rpc_write(conn, data, 16) < 0) ||
      rpc_write(conn, &attribute, sizeof(CUpointer_attribute)) < 0 ||
      rpc_write(conn, &ptr, sizeof(CUdeviceptr)) < 0 ||
      rpc_wait_for_response(conn) < 0 ||
      (!data_null && 16 != 0 && rpc_read(conn, data, 16) < 0) ||
      rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
      rpc_read_end(conn) < 0)
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  if (return_value == CUDA_SUCCESS && !lupine_route_is_local(route))
    lupine_fixup_pointer_attribute_value(query, attribute, staged);
  memcpy(caller_data, staged, value_size);
  return return_value;
}

#if CUDA_VERSION >= 12020
CUresult cuMemPrefetchAsync_v2(CUdeviceptr devPtr, size_t count,
                               CUmemLocation location, unsigned int flags,
                               CUstream hStream) {
  lupine_route route = lupine_route_for_deviceptr(devPtr);
  if (location.type == CU_MEM_LOCATION_TYPE_DEVICE) {
    CUdevice route_device = location.id;
    route = lupine_route_for_device(&route_device);
    if (route.kind == LUPINE_ROUTE_UNKNOWN_DEVICE)
      return CUDA_ERROR_INVALID_DEVICE;
    location.id = route_device;
  } else if (hStream != nullptr) {
    route = lupine_route_for_stream(hStream);
  }
  if (hStream != nullptr &&
      !lupine_routes_share_server(route, lupine_route_for_stream(hStream)))
    return CUDA_ERROR_INVALID_VALUE;
  if (!lupine_route_is_local(route) && lupine_is_managed_host_alias(devPtr))
    return CUDA_SUCCESS;
  CUresult return_value;
  if (lupine_route_is_local(route))
    return lupine_call_real_cuda_fn(
        LUPINE_REAL_CUDA_SYMBOL("cuMemPrefetchAsync_v2"), devPtr, count,
        location, flags, hStream);
  conn_t *conn = lupine_route_remote_conn(route);
  if (lupine_prepare_rpc(conn) < 0 ||
      lupine_rpc_start_on_stream(conn, RPC_cuMemPrefetchAsync_v2, hStream) <
          0 ||
      rpc_write(conn, &devPtr, sizeof(CUdeviceptr)) < 0 ||
      rpc_write(conn, &count, sizeof(size_t)) < 0 ||
      rpc_write(conn, &location, sizeof(CUmemLocation)) < 0 ||
      rpc_write(conn, &flags, sizeof(unsigned int)) < 0 ||
      rpc_write(conn, &hStream, sizeof(CUstream)) < 0 ||
      rpc_write_end(conn) < 0)
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  return_value = CUDA_SUCCESS;
  return return_value;
}

#endif

#if CUDA_VERSION >= 12020
CUresult cuMemAdvise_v2(CUdeviceptr devPtr, size_t count, CUmem_advise advice,
                        CUmemLocation location) {
  lupine_route route = lupine_route_for_deviceptr(devPtr);
  CUdevice virtual_device = location.id;
  if (location.type == CU_MEM_LOCATION_TYPE_DEVICE) {
    CUdevice route_device = location.id;
    route = lupine_route_for_device(&route_device);
    if (route.kind == LUPINE_ROUTE_UNKNOWN_DEVICE)
      return CUDA_ERROR_INVALID_DEVICE;
    location.id = route_device;
  }
  if (!lupine_route_is_local(route) && lupine_is_managed_host_alias(devPtr)) {
    if (location.type == CU_MEM_LOCATION_TYPE_DEVICE)
      return lupine_managed_mem_advise(devPtr, count, advice, virtual_device);
    if (location.type == CU_MEM_LOCATION_TYPE_HOST)
      return lupine_managed_mem_advise(devPtr, count, advice, CU_DEVICE_CPU);
    return CUDA_SUCCESS;
  }
  CUresult return_value;
  if (lupine_route_is_local(route))
    return lupine_call_real_cuda_fn(LUPINE_REAL_CUDA_SYMBOL("cuMemAdvise_v2"),
                                    devPtr, count, advice, location);
  conn_t *conn = lupine_route_remote_conn(route);
  if (lupine_prepare_rpc(conn) < 0 ||
      rpc_write_start_request_on(conn, RPC_cuMemAdvise_v2, RPC_SCOPE_NONE) <
          0 ||
      rpc_write(conn, &devPtr, sizeof(CUdeviceptr)) < 0 ||
      rpc_write(conn, &count, sizeof(size_t)) < 0 ||
      rpc_write(conn, &advice, sizeof(CUmem_advise)) < 0 ||
      rpc_write(conn, &location, sizeof(CUmemLocation)) < 0 ||
      rpc_write_end(conn) < 0)
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  return_value = CUDA_SUCCESS;
  return return_value;
}

#endif

CUresult cuMemRangeGetAttribute(void *data, size_t dataSize,
                                CUmem_range_attribute attribute,
                                CUdeviceptr devPtr, size_t count) {
  lupine_route route = lupine_route_for_deviceptr(devPtr);
  if (data == nullptr || count == 0 ||
      !lupine_mem_range_attribute_size_valid(attribute, dataSize))
    return CUDA_ERROR_INVALID_VALUE;
  CUresult advice_result;
  if (lupine_managed_mem_range_attribute(data, dataSize, attribute, devPtr,
                                         count, &advice_result))
    return advice_result;
  CUresult return_value;
  if (lupine_route_is_local(route))
    return lupine_call_real_cuda_fn(
        LUPINE_REAL_CUDA_SYMBOL("cuMemRangeGetAttribute"), data, dataSize,
        attribute, devPtr, count);
  conn_t *conn = lupine_route_remote_conn(route);
  uint8_t data_null = data == nullptr ? 1 : 0;
  if (lupine_prepare_rpc(conn) < 0 ||
      rpc_write_start_request_on(conn, RPC_cuMemRangeGetAttribute,
                                 RPC_SCOPE_NONE) < 0 ||
      rpc_write(conn, &dataSize, sizeof(size_t)) < 0 ||
      rpc_write(conn, &data_null, sizeof(uint8_t)) < 0 ||
      rpc_write(conn, &attribute, sizeof(CUmem_range_attribute)) < 0 ||
      rpc_write(conn, &devPtr, sizeof(CUdeviceptr)) < 0 ||
      rpc_write(conn, &count, sizeof(size_t)) < 0 ||
      rpc_wait_for_response(conn) < 0 ||
      (!data_null && dataSize != 0 && rpc_read(conn, data, dataSize) < 0) ||
      rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
      rpc_read_end(conn) < 0)
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  return return_value;
}

CUresult cuPointerSetAttribute(const void *value, CUpointer_attribute attribute,
                               CUdeviceptr ptr) {
  lupine_route route = lupine_route_for_deviceptr(ptr);
  size_t value_size = 0;
  if (value == nullptr ||
      !lupine_settable_pointer_attribute_size(attribute, &value_size))
    return CUDA_ERROR_INVALID_VALUE;
  cuuint64_t staged = 0;
  memcpy(&staged, value, value_size);
  value = &staged;
  ptr = lupine_resolve_pointer_attribute_target(ptr);
  route = lupine_route_for_deviceptr(ptr);
  CUresult return_value;
  if (lupine_route_is_local(route))
    return lupine_call_real_cuda_fn(
        LUPINE_REAL_CUDA_SYMBOL("cuPointerSetAttribute"), value, attribute,
        ptr);
  conn_t *conn = lupine_route_remote_conn(route);
  uint8_t value_null = value == nullptr ? 1 : 0;
  if (lupine_prepare_rpc(conn) < 0 ||
      rpc_write_start_request_on(conn, RPC_cuPointerSetAttribute,
                                 RPC_SCOPE_NONE) < 0 ||
      rpc_write(conn, &value_null, sizeof(uint8_t)) < 0 ||
      (!value_null && 8 != 0 && rpc_write(conn, value, 8) < 0) ||
      rpc_write(conn, &attribute, sizeof(CUpointer_attribute)) < 0 ||
      rpc_write(conn, &ptr, sizeof(CUdeviceptr)) < 0 || rpc_write_end(conn) < 0)
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  return_value = CUDA_SUCCESS;
  return return_value;
}

CUresult cuStreamGetPriority(CUstream hStream, int *priority) {
  lupine_route route = (hStream != nullptr ? lupine_route_for_stream(hStream)
                                           : lupine_route_for_default());
  CUresult return_value;
  if (lupine_route_is_local(route))
    return lupine_call_real_cuda_fn(
        LUPINE_REAL_CUDA_SYMBOL("cuStreamGetPriority"), hStream, priority);
  conn_t *conn = lupine_route_remote_conn(route);
  uint8_t priority_present = priority != nullptr;
  if (lupine_prepare_rpc(conn) < 0 ||
      lupine_rpc_start_on_stream(conn, RPC_cuStreamGetPriority, hStream) < 0 ||
      rpc_write(conn, &hStream, sizeof(CUstream)) < 0 ||
      rpc_write(conn, &priority_present, sizeof(uint8_t)) < 0 ||
      (priority_present && rpc_write(conn, priority, sizeof(int)) < 0) ||
      rpc_wait_for_response(conn) < 0 ||
      (priority_present && rpc_read(conn, priority, sizeof(int)) < 0) ||
      rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
      rpc_read_end(conn) < 0)
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  return return_value;
}

CUresult cuStreamGetDevice(CUstream hStream, CUdevice *device) {
  lupine_route route = (hStream != nullptr ? lupine_route_for_stream(hStream)
                                           : lupine_route_for_default());
  if (device == nullptr)
    return CUDA_ERROR_INVALID_VALUE;
  if (!lupine_cuda_is_initialized())
    return CUDA_ERROR_NOT_INITIALIZED;
  CUresult return_value;
  if (lupine_route_is_local(route)) {
    return_value = lupine_call_real_cuda_fn(
        LUPINE_REAL_CUDA_SYMBOL("cuStreamGetDevice"), hStream, device);
    if (!lupine_route_is_local(route) && return_value == CUDA_SUCCESS)
      *device = lupine_local_device_for_remote(lupine_route_remote_conn(route),
                                               *device);
    return return_value;
  }
  conn_t *conn = lupine_route_remote_conn(route);
  uint8_t device_present = device != nullptr;
  if (lupine_prepare_rpc(conn) < 0 ||
      lupine_rpc_start_on_stream(conn, RPC_cuStreamGetDevice, hStream) < 0 ||
      rpc_write(conn, &hStream, sizeof(CUstream)) < 0 ||
      rpc_write(conn, &device_present, sizeof(uint8_t)) < 0 ||
      (device_present && rpc_write(conn, device, sizeof(CUdevice)) < 0) ||
      rpc_wait_for_response(conn) < 0 ||
      (device_present && rpc_read(conn, device, sizeof(CUdevice)) < 0) ||
      rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
      rpc_read_end(conn) < 0)
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  if (!lupine_route_is_local(route) && return_value == CUDA_SUCCESS)
    *device = lupine_local_device_for_remote(lupine_route_remote_conn(route),
                                             *device);
  return return_value;
}

CUresult cuStreamGetFlags(CUstream hStream, unsigned int *flags) {
  lupine_route route = (hStream != nullptr ? lupine_route_for_stream(hStream)
                                           : lupine_route_for_default());
  CUresult return_value;
  if (lupine_route_is_local(route))
    return lupine_call_real_cuda_fn(LUPINE_REAL_CUDA_SYMBOL("cuStreamGetFlags"),
                                    hStream, flags);
  conn_t *conn = lupine_route_remote_conn(route);
  uint8_t flags_present = flags != nullptr;
  if (lupine_prepare_rpc(conn) < 0 ||
      lupine_rpc_start_on_stream(conn, RPC_cuStreamGetFlags, hStream) < 0 ||
      rpc_write(conn, &hStream, sizeof(CUstream)) < 0 ||
      rpc_write(conn, &flags_present, sizeof(uint8_t)) < 0 ||
      (flags_present && rpc_write(conn, flags, sizeof(unsigned int)) < 0) ||
      rpc_wait_for_response(conn) < 0 ||
      (flags_present && rpc_read(conn, flags, sizeof(unsigned int)) < 0) ||
      rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
      rpc_read_end(conn) < 0)
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  return return_value;
}

CUresult cuStreamGetId(CUstream hStream, unsigned long long *streamId) {
  lupine_route route = (hStream != nullptr ? lupine_route_for_stream(hStream)
                                           : lupine_route_for_default());
  CUresult return_value;
  if (lupine_route_is_local(route))
    return lupine_call_real_cuda_fn(LUPINE_REAL_CUDA_SYMBOL("cuStreamGetId"),
                                    hStream, streamId);
  conn_t *conn = lupine_route_remote_conn(route);
  uint8_t streamId_present = streamId != nullptr;
  if (lupine_prepare_rpc(conn) < 0 ||
      lupine_rpc_start_on_stream(conn, RPC_cuStreamGetId, hStream) < 0 ||
      rpc_write(conn, &hStream, sizeof(CUstream)) < 0 ||
      rpc_write(conn, &streamId_present, sizeof(uint8_t)) < 0 ||
      (streamId_present &&
       rpc_write(conn, streamId, sizeof(unsigned long long)) < 0) ||
      rpc_wait_for_response(conn) < 0 ||
      (streamId_present &&
       rpc_read(conn, streamId, sizeof(unsigned long long)) < 0) ||
      rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
      rpc_read_end(conn) < 0)
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  return return_value;
}

CUresult cuStreamGetCtx(CUstream hStream, CUcontext *pctx) {
  lupine_route route = (hStream != nullptr ? lupine_route_for_stream(hStream)
                                           : lupine_route_for_default());
  CUresult return_value;
  if (lupine_route_is_local(route))
    return lupine_call_real_cuda_fn(LUPINE_REAL_CUDA_SYMBOL("cuStreamGetCtx"),
                                    hStream, pctx);
  conn_t *conn = lupine_route_remote_conn(route);
  uint8_t pctx_present = pctx != nullptr;
  if (lupine_prepare_rpc(conn) < 0 ||
      lupine_rpc_start_on_stream(conn, RPC_cuStreamGetCtx, hStream) < 0 ||
      rpc_write(conn, &hStream, sizeof(CUstream)) < 0 ||
      rpc_write(conn, &pctx_present, sizeof(uint8_t)) < 0 ||
      (pctx_present && rpc_write(conn, pctx, sizeof(CUcontext)) < 0) ||
      rpc_wait_for_response(conn) < 0 ||
      (pctx_present && rpc_read(conn, pctx, sizeof(CUcontext)) < 0) ||
      rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
      rpc_read_end(conn) < 0)
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  return return_value;
}

CUresult cuStreamWaitEvent(CUstream hStream, CUevent hEvent,
                           unsigned int Flags) {
  lupine_route route = (hStream != nullptr ? lupine_route_for_stream(hStream)
                                           : lupine_route_for_default());
  if (hEvent != nullptr &&
      !lupine_routes_share_server(route, lupine_route_for_event(hEvent)))
    return cuEventSynchronize(hEvent);
  CUresult return_value;
  if (lupine_route_is_local(route))
    return lupine_call_real_cuda_fn(
        LUPINE_REAL_CUDA_SYMBOL("cuStreamWaitEvent"), hStream, hEvent, Flags);
  conn_t *conn = lupine_route_remote_conn(route);
  if (lupine_prepare_rpc(conn) < 0 ||
      lupine_rpc_start_on_stream(conn, RPC_cuStreamWaitEvent, hStream) < 0 ||
      rpc_request_on(conn, reinterpret_cast<uintptr_t>(hEvent)) < 0 ||
      rpc_write(conn, &hStream, sizeof(CUstream)) < 0 ||
      rpc_write(conn, &hEvent, sizeof(CUevent)) < 0 ||
      rpc_write(conn, &Flags, sizeof(unsigned int)) < 0 ||
      rpc_write_end(conn) < 0)
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  return_value = CUDA_SUCCESS;
  return return_value;
}

CUresult cuStreamAddCallback(CUstream hStream, CUstreamCallback callback,
                             void *userData, unsigned int flags) {
  lupine_route route = (hStream != nullptr ? lupine_route_for_stream(hStream)
                                           : lupine_route_for_default());
  if (callback == nullptr)
    return CUDA_ERROR_INVALID_VALUE;
  CUresult return_value;
  if (lupine_route_is_local(route))
    return lupine_call_real_cuda_fn(
        LUPINE_REAL_CUDA_SYMBOL("cuStreamAddCallback"), hStream, callback,
        userData, flags);
  conn_t *conn = lupine_route_remote_conn(route);
  if (lupine_prepare_rpc(conn) < 0 ||
      lupine_rpc_start_on_stream(conn, RPC_cuStreamAddCallback, hStream) < 0 ||
      rpc_write(conn, &hStream, sizeof(CUstream)) < 0 ||
      rpc_write(conn, &callback, sizeof(CUstreamCallback)) < 0 ||
      rpc_write(conn, &userData, sizeof(void *)) < 0 ||
      rpc_write(conn, &flags, sizeof(unsigned int)) < 0 ||
      rpc_write_end(conn) < 0)
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  return_value = CUDA_SUCCESS;
  return return_value;
}

CUresult cuStreamBeginCapture_v2(CUstream hStream, CUstreamCaptureMode mode) {
  lupine_route route = (hStream != nullptr ? lupine_route_for_stream(hStream)
                                           : lupine_route_for_default());
  lupine_materialize_host_allocations();
  lupine_capture_begin_guard capture_guard;
  CUresult return_value;
  if (lupine_route_is_local(route)) {
    return_value = lupine_call_real_cuda_fn(
        LUPINE_REAL_CUDA_SYMBOL("cuStreamBeginCapture_v2"), hStream, mode);
    (void)capture_guard.complete(return_value);
    return return_value;
  }
  conn_t *conn = lupine_route_remote_conn(route);
  if (lupine_prepare_rpc(conn) < 0 ||
      rpc_write_start_request(conn, RPC_cuStreamBeginCapture_v2) < 0 ||
      rpc_write(conn, &hStream, sizeof(CUstream)) < 0 ||
      rpc_write(conn, &mode, sizeof(CUstreamCaptureMode)) < 0 ||
      rpc_wait_for_response(conn) < 0 ||
      rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
      rpc_read_end(conn) < 0)
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  (void)capture_guard.complete(return_value);
  return return_value;
}

CUresult cuStreamEndCapture(CUstream hStream, CUgraph *phGraph) {
  lupine_route route = (hStream != nullptr ? lupine_route_for_stream(hStream)
                                           : lupine_route_for_default());
  CUresult return_value;
  if (lupine_route_is_local(route)) {
    return_value = lupine_call_real_cuda_fn(
        LUPINE_REAL_CUDA_SYMBOL("cuStreamEndCapture"), hStream, phGraph);
    if (return_value == CUDA_SUCCESS && phGraph != nullptr) {
      lupine_note_graph_owner_route(*phGraph, route);
    }
    (void)lupine_complete_stream_end_capture(return_value);
    return return_value;
  }
  conn_t *conn = lupine_route_remote_conn(route);
  uint8_t phGraph_present = phGraph != nullptr;
  if (lupine_prepare_rpc(conn) < 0 ||
      lupine_rpc_start_on_stream(conn, RPC_cuStreamEndCapture, hStream) < 0 ||
      rpc_write(conn, &hStream, sizeof(CUstream)) < 0 ||
      rpc_write(conn, &phGraph_present, sizeof(uint8_t)) < 0 ||
      (phGraph_present && rpc_write(conn, phGraph, sizeof(CUgraph)) < 0) ||
      rpc_wait_for_response(conn) < 0 ||
      (phGraph_present && rpc_read(conn, phGraph, sizeof(CUgraph)) < 0) ||
      rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
      rpc_read_end(conn) < 0)
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  if (return_value == CUDA_SUCCESS && phGraph != nullptr) {
    lupine_note_graph_owner_route(*phGraph, route);
  }
  (void)lupine_complete_stream_end_capture(return_value);
  return return_value;
}

CUresult cuStreamIsCapturing(CUstream hStream,
                             CUstreamCaptureStatus *captureStatus) {
  lupine_route route = (hStream != nullptr ? lupine_route_for_stream(hStream)
                                           : lupine_route_for_default());
  if (captureStatus == nullptr)
    return CUDA_ERROR_INVALID_VALUE;
  if (lupine_try_cached_stream_is_capturing(hStream, captureStatus))
    return CUDA_SUCCESS;
  CUresult return_value;
  if (lupine_route_is_local(route))
    return lupine_call_real_cuda_fn(
        LUPINE_REAL_CUDA_SYMBOL("cuStreamIsCapturing"), hStream, captureStatus);
  conn_t *conn = lupine_route_remote_conn(route);
  uint8_t captureStatus_present = captureStatus != nullptr;
  if (lupine_prepare_rpc(conn) < 0 ||
      lupine_rpc_start_on_stream(conn, RPC_cuStreamIsCapturing, hStream) < 0 ||
      rpc_write(conn, &hStream, sizeof(CUstream)) < 0 ||
      rpc_write(conn, &captureStatus_present, sizeof(uint8_t)) < 0 ||
      (captureStatus_present &&
       rpc_write(conn, captureStatus, sizeof(CUstreamCaptureStatus)) < 0) ||
      rpc_wait_for_response(conn) < 0 ||
      (captureStatus_present &&
       rpc_read(conn, captureStatus, sizeof(CUstreamCaptureStatus)) < 0) ||
      rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
      rpc_read_end(conn) < 0)
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  return return_value;
}

CUresult cuStreamUpdateCaptureDependencies_v2(
    CUstream hStream, CUgraphNode *dependencies,
    const CUgraphEdgeData *dependencyData, size_t numDependencies,
    unsigned int flags) {
  lupine_route route = (hStream != nullptr ? lupine_route_for_stream(hStream)
                                           : lupine_route_for_default());
  CUresult return_value;
  if (lupine_route_is_local(route))
    return lupine_call_real_cuda_fn(
        LUPINE_REAL_CUDA_SYMBOL("cuStreamUpdateCaptureDependencies_v2"),
        hStream, dependencies, dependencyData, numDependencies, flags);
  conn_t *conn = lupine_route_remote_conn(route);
  uint8_t dependencies_null = dependencies == nullptr ? 1 : 0;
  uint8_t dependencyData_null = dependencyData == nullptr ? 1 : 0;
  if (lupine_prepare_rpc(conn) < 0 ||
      lupine_rpc_start_on_stream(conn, RPC_cuStreamUpdateCaptureDependencies_v2,
                                 hStream) < 0 ||
      rpc_write(conn, &hStream, sizeof(CUstream)) < 0 ||
      rpc_write(conn, &numDependencies, sizeof(size_t)) < 0 ||
      rpc_write(conn, &dependencies_null, sizeof(uint8_t)) < 0 ||
      (!dependencies_null && numDependencies * sizeof(CUgraphNode) != 0 &&
       rpc_write(conn, dependencies, numDependencies * sizeof(CUgraphNode)) <
           0) ||
      rpc_write(conn, &dependencyData_null, sizeof(uint8_t)) < 0 ||
      (!dependencyData_null &&
       numDependencies * sizeof(const CUgraphEdgeData) != 0 &&
       rpc_write(conn, dependencyData,
                 numDependencies * sizeof(const CUgraphEdgeData)) < 0) ||
      rpc_write(conn, &flags, sizeof(unsigned int)) < 0 ||
      rpc_wait_for_response(conn) < 0 ||
      rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
      rpc_read_end(conn) < 0)
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  return return_value;
}

CUresult cuStreamAttachMemAsync(CUstream hStream, CUdeviceptr dptr,
                                size_t length, unsigned int flags) {
  lupine_route route = (hStream != nullptr ? lupine_route_for_stream(hStream)
                                           : lupine_route_for_deviceptr(dptr));
  CUresult return_value;
  if (lupine_route_is_local(route))
    return lupine_call_real_cuda_fn(
        LUPINE_REAL_CUDA_SYMBOL("cuStreamAttachMemAsync"), hStream, dptr,
        length, flags);
  conn_t *conn = lupine_route_remote_conn(route);
  if (lupine_prepare_rpc(conn) < 0 ||
      lupine_rpc_start_on_stream(conn, RPC_cuStreamAttachMemAsync, hStream) <
          0 ||
      rpc_write(conn, &hStream, sizeof(CUstream)) < 0 ||
      rpc_write(conn, &dptr, sizeof(CUdeviceptr)) < 0 ||
      rpc_write(conn, &length, sizeof(size_t)) < 0 ||
      rpc_write(conn, &flags, sizeof(unsigned int)) < 0 ||
      rpc_write_end(conn) < 0)
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  return_value = CUDA_SUCCESS;
  return return_value;
}

CUresult cuStreamQuery(CUstream hStream) {
  lupine_route route = (hStream != nullptr ? lupine_route_for_stream(hStream)
                                           : lupine_route_for_default());
  CUresult return_value;
  if (lupine_route_is_local(route)) {
    return_value = lupine_call_real_cuda_fn(
        LUPINE_REAL_CUDA_SYMBOL("cuStreamQuery"), hStream);
    if (return_value == CUDA_SUCCESS)
      return_value = lupine_sync_mapped_device_to_host();
    return return_value;
  }
  conn_t *conn = lupine_route_remote_conn(route);
  if (lupine_prepare_rpc(conn) < 0 ||
      lupine_rpc_start_on_stream(conn, RPC_cuStreamQuery, hStream,
                                 RPC_REQUEST_SOFT) < 0 ||
      rpc_write(conn, &hStream, sizeof(CUstream)) < 0 ||
      rpc_wait_for_response(conn) < 0 || lupine_read_dtoh_watermark(conn) < 0 ||
      rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
      rpc_read_end(conn) < 0)
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  if (return_value == CUDA_SUCCESS)
    return_value = lupine_sync_mapped_device_to_host();
  return return_value;
}

CUresult cuStreamSynchronize(CUstream hStream) {
  lupine_route route = (hStream != nullptr ? lupine_route_for_stream(hStream)
                                           : lupine_route_for_default());
  CUresult return_value;
  if (lupine_route_is_local(route)) {
    return_value = lupine_call_real_cuda_fn(
        LUPINE_REAL_CUDA_SYMBOL("cuStreamSynchronize"), hStream);
    if (return_value == CUDA_SUCCESS)
      return_value = lupine_sync_mapped_device_to_host();
    return return_value;
  }
  conn_t *conn = lupine_route_remote_conn(route);
  if (lupine_prepare_rpc(conn) < 0 ||
      lupine_rpc_start_on_stream(conn, RPC_cuStreamSynchronize, hStream) < 0 ||
      rpc_write(conn, &hStream, sizeof(CUstream)) < 0 ||
      rpc_wait_for_response(conn) < 0 || lupine_read_dtoh_watermark(conn) < 0 ||
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
  if (lupine_route_is_local(route)) {
    return_value = lupine_call_real_cuda_fn(
        LUPINE_REAL_CUDA_SYMBOL("cuStreamDestroy_v2"), hStream);
    if (return_value == CUDA_SUCCESS)
      lupine_forget_stream_owner(hStream);
    return return_value;
  }
  conn_t *conn = lupine_route_remote_conn(route);
  if (lupine_prepare_rpc(conn) < 0 ||
      lupine_rpc_start_on_stream(conn, RPC_cuStreamDestroy_v2, hStream) < 0 ||
      rpc_write(conn, &hStream, sizeof(CUstream)) < 0 ||
      rpc_write_end(conn) < 0)
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  return_value = CUDA_SUCCESS;
  if (return_value == CUDA_SUCCESS)
    lupine_forget_stream_owner(hStream);
  return return_value;
}

CUresult cuStreamCopyAttributes(CUstream dst, CUstream src) {
  lupine_route route = (dst != nullptr ? lupine_route_for_stream(dst)
                                       : lupine_route_for_default());
  CUresult return_value;
  if (lupine_route_is_local(route))
    return lupine_call_real_cuda_fn(
        LUPINE_REAL_CUDA_SYMBOL("cuStreamCopyAttributes"), dst, src);
  conn_t *conn = lupine_route_remote_conn(route);
  if (lupine_prepare_rpc(conn) < 0 ||
      lupine_rpc_start_on_stream(conn, RPC_cuStreamCopyAttributes, dst) < 0 ||
      rpc_request_on(conn, reinterpret_cast<uintptr_t>(src)) < 0 ||
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
  if (lupine_route_is_local(route))
    return lupine_call_real_cuda_fn(
        LUPINE_REAL_CUDA_SYMBOL("cuStreamGetAttribute"), hStream, attr,
        value_out);
  conn_t *conn = lupine_route_remote_conn(route);
  uint8_t value_out_present = value_out != nullptr;
  if (lupine_prepare_rpc(conn) < 0 ||
      lupine_rpc_start_on_stream(conn, RPC_cuStreamGetAttribute, hStream) < 0 ||
      rpc_write(conn, &hStream, sizeof(CUstream)) < 0 ||
      rpc_write(conn, &attr, sizeof(CUstreamAttrID)) < 0 ||
      rpc_write(conn, &value_out_present, sizeof(uint8_t)) < 0 ||
      (value_out_present &&
       rpc_write(conn, value_out, sizeof(CUstreamAttrValue)) < 0) ||
      rpc_wait_for_response(conn) < 0 ||
      (value_out_present &&
       rpc_read(conn, value_out, sizeof(CUstreamAttrValue)) < 0) ||
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
  if (lupine_route_is_local(route))
    return lupine_call_real_cuda_fn(
        LUPINE_REAL_CUDA_SYMBOL("cuStreamSetAttribute"), hStream, attr, value);
  conn_t *conn = lupine_route_remote_conn(route);
  uint8_t value_present = value != nullptr;
  if (lupine_prepare_rpc(conn) < 0 ||
      lupine_rpc_start_on_stream(conn, RPC_cuStreamSetAttribute, hStream) < 0 ||
      rpc_write(conn, &hStream, sizeof(CUstream)) < 0 ||
      rpc_write(conn, &attr, sizeof(CUstreamAttrID)) < 0 ||
      rpc_write(conn, &value_present, sizeof(uint8_t)) < 0 ||
      (value_present &&
       rpc_write(conn, value, sizeof(const CUstreamAttrValue)) < 0) ||
      rpc_wait_for_response(conn) < 0 ||
      rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
      rpc_read_end(conn) < 0)
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  return return_value;
}

CUresult cuEventRecord(CUevent hEvent, CUstream hStream) {
  lupine_route route = (hStream != nullptr ? lupine_route_for_stream(hStream)
                                           : lupine_route_for_default());
  CUresult return_value;
  if (lupine_route_is_local(route)) {
    return_value = lupine_call_real_cuda_fn(
        LUPINE_REAL_CUDA_SYMBOL("cuEventRecord"), hEvent, hStream);
    if (!lupine_route_is_local(route))
      lupine_event_invalidate_completion(hEvent);
    return return_value;
  }
  conn_t *conn = lupine_route_remote_conn(route);
  if (lupine_prepare_rpc(conn) < 0 ||
      lupine_rpc_start_on_stream(conn, RPC_cuEventRecord, hStream) < 0 ||
      rpc_request_on(conn, reinterpret_cast<uintptr_t>(hEvent)) < 0 ||
      rpc_write(conn, &hEvent, sizeof(CUevent)) < 0 ||
      rpc_write(conn, &hStream, sizeof(CUstream)) < 0 ||
      rpc_write_end(conn) < 0)
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  return_value = CUDA_SUCCESS;
  if (!lupine_route_is_local(route))
    lupine_event_invalidate_completion(hEvent);
  return return_value;
}

CUresult cuEventRecordWithFlags(CUevent hEvent, CUstream hStream,
                                unsigned int flags) {
  lupine_route route = (hStream != nullptr ? lupine_route_for_stream(hStream)
                                           : lupine_route_for_default());
  CUresult return_value;
  if (lupine_route_is_local(route)) {
    return_value = lupine_call_real_cuda_fn(
        LUPINE_REAL_CUDA_SYMBOL("cuEventRecordWithFlags"), hEvent, hStream,
        flags);
    if (!lupine_route_is_local(route))
      lupine_event_invalidate_completion(hEvent);
    return return_value;
  }
  conn_t *conn = lupine_route_remote_conn(route);
  if (lupine_prepare_rpc(conn) < 0 ||
      lupine_rpc_start_on_stream(conn, RPC_cuEventRecordWithFlags, hStream) <
          0 ||
      rpc_request_on(conn, reinterpret_cast<uintptr_t>(hEvent)) < 0 ||
      rpc_write(conn, &hEvent, sizeof(CUevent)) < 0 ||
      rpc_write(conn, &hStream, sizeof(CUstream)) < 0 ||
      rpc_write(conn, &flags, sizeof(unsigned int)) < 0 ||
      rpc_write_end(conn) < 0)
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  return_value = CUDA_SUCCESS;
  if (!lupine_route_is_local(route))
    lupine_event_invalidate_completion(hEvent);
  return return_value;
}

CUresult cuEventSynchronize(CUevent hEvent) {
  lupine_route route = lupine_route_for_event(hEvent);
  CUresult return_value;
  if (lupine_route_is_local(route)) {
    return_value = lupine_call_real_cuda_fn(
        LUPINE_REAL_CUDA_SYMBOL("cuEventSynchronize"), hEvent);
    if (return_value == CUDA_SUCCESS)
      return_value = lupine_sync_mapped_device_to_host();
    return return_value;
  }
  conn_t *conn = lupine_route_remote_conn(route);
  if (lupine_prepare_rpc(conn) < 0 ||
      rpc_write_start_request_on(conn, RPC_cuEventSynchronize,
                                 reinterpret_cast<uintptr_t>(hEvent)) < 0 ||
      rpc_write(conn, &hEvent, sizeof(CUevent)) < 0 ||
      rpc_wait_for_response(conn) < 0 || lupine_read_dtoh_watermark(conn) < 0 ||
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
  std::unique_lock<std::shared_mutex> event_lifecycle_lock(
      lupine_event_lifecycle_mutex());
  CUresult return_value;
  if (lupine_route_is_local(route)) {
    return_value = lupine_call_real_cuda_fn(
        LUPINE_REAL_CUDA_SYMBOL("cuEventDestroy_v2"), hEvent);
    if (return_value == CUDA_SUCCESS)
      lupine_forget_event_owner(hEvent);
    return return_value;
  }
  conn_t *conn = lupine_route_remote_conn(route);
  if (lupine_prepare_rpc(conn) < 0 ||
      rpc_write_start_request_on(conn, RPC_cuEventDestroy_v2,
                                 reinterpret_cast<uintptr_t>(hEvent)) < 0 ||
      rpc_write(conn, &hEvent, sizeof(CUevent)) < 0 || rpc_write_end(conn) < 0)
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  return_value = CUDA_SUCCESS;
  if (return_value == CUDA_SUCCESS)
    lupine_forget_event_owner(hEvent);
  return return_value;
}

CUresult cuEventElapsedTime_v2(float *pMilliseconds, CUevent hStart,
                               CUevent hEnd) {
  lupine_route route = lupine_route_for_event(hStart);
  CUresult return_value;
  if (lupine_route_is_local(route))
    return lupine_call_real_cuda_fn(
        LUPINE_REAL_CUDA_SYMBOL("cuEventElapsedTime_v2"), pMilliseconds, hStart,
        hEnd);
  conn_t *conn = lupine_route_remote_conn(route);
  uint8_t pMilliseconds_present = pMilliseconds != nullptr;
  if (lupine_prepare_rpc(conn) < 0 ||
      rpc_write_start_request_on(conn, RPC_cuEventElapsedTime_v2,
                                 reinterpret_cast<uintptr_t>(hStart)) < 0 ||
      rpc_request_on(conn, reinterpret_cast<uintptr_t>(hEnd)) < 0 ||
      rpc_write(conn, &pMilliseconds_present, sizeof(uint8_t)) < 0 ||
      (pMilliseconds_present &&
       rpc_write(conn, pMilliseconds, sizeof(float)) < 0) ||
      rpc_write(conn, &hStart, sizeof(CUevent)) < 0 ||
      rpc_write(conn, &hEnd, sizeof(CUevent)) < 0 ||
      rpc_wait_for_response(conn) < 0 ||
      (pMilliseconds_present &&
       rpc_read(conn, pMilliseconds, sizeof(float)) < 0) ||
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
  if (lupine_route_is_local(route))
    return lupine_call_real_cuda_fn(
        LUPINE_REAL_CUDA_SYMBOL("cuImportExternalMemory"), extMem_out,
        memHandleDesc);
  conn_t *conn = lupine_route_remote_conn(route);
  uint8_t extMem_out_present = extMem_out != nullptr;
  uint8_t memHandleDesc_present = memHandleDesc != nullptr;
  if (lupine_prepare_rpc(conn) < 0 ||
      rpc_write_start_request_on(conn, RPC_cuImportExternalMemory,
                                 RPC_SCOPE_NONE) < 0 ||
      rpc_write(conn, &extMem_out_present, sizeof(uint8_t)) < 0 ||
      (extMem_out_present &&
       rpc_write(conn, extMem_out, sizeof(CUexternalMemory)) < 0) ||
      rpc_write(conn, &memHandleDesc_present, sizeof(uint8_t)) < 0 ||
      (memHandleDesc_present &&
       rpc_write(conn, memHandleDesc,
                 sizeof(const CUDA_EXTERNAL_MEMORY_HANDLE_DESC)) < 0) ||
      rpc_wait_for_response(conn) < 0 ||
      (extMem_out_present &&
       rpc_read(conn, extMem_out, sizeof(CUexternalMemory)) < 0) ||
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
  if (lupine_route_is_local(route))
    return lupine_call_real_cuda_fn(
        LUPINE_REAL_CUDA_SYMBOL("cuExternalMemoryGetMappedBuffer"), devPtr,
        extMem, bufferDesc);
  conn_t *conn = lupine_route_remote_conn(route);
  uint8_t devPtr_present = devPtr != nullptr;
  uint8_t bufferDesc_present = bufferDesc != nullptr;
  if (lupine_prepare_rpc(conn) < 0 ||
      rpc_write_start_request_on(conn, RPC_cuExternalMemoryGetMappedBuffer,
                                 RPC_SCOPE_NONE) < 0 ||
      rpc_write(conn, &devPtr_present, sizeof(uint8_t)) < 0 ||
      (devPtr_present && rpc_write(conn, devPtr, sizeof(CUdeviceptr)) < 0) ||
      rpc_write(conn, &extMem, sizeof(CUexternalMemory)) < 0 ||
      rpc_write(conn, &bufferDesc_present, sizeof(uint8_t)) < 0 ||
      (bufferDesc_present &&
       rpc_write(conn, bufferDesc,
                 sizeof(const CUDA_EXTERNAL_MEMORY_BUFFER_DESC)) < 0) ||
      rpc_wait_for_response(conn) < 0 ||
      (devPtr_present && rpc_read(conn, devPtr, sizeof(CUdeviceptr)) < 0) ||
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
  if (lupine_route_is_local(route))
    return lupine_call_real_cuda_fn(
        LUPINE_REAL_CUDA_SYMBOL("cuExternalMemoryGetMappedMipmappedArray"),
        mipmap, extMem, mipmapDesc);
  conn_t *conn = lupine_route_remote_conn(route);
  uint8_t mipmap_present = mipmap != nullptr;
  uint8_t mipmapDesc_present = mipmapDesc != nullptr;
  if (lupine_prepare_rpc(conn) < 0 ||
      rpc_write_start_request_on(conn,
                                 RPC_cuExternalMemoryGetMappedMipmappedArray,
                                 RPC_SCOPE_NONE) < 0 ||
      rpc_write(conn, &mipmap_present, sizeof(uint8_t)) < 0 ||
      (mipmap_present &&
       rpc_write(conn, mipmap, sizeof(CUmipmappedArray)) < 0) ||
      rpc_write(conn, &extMem, sizeof(CUexternalMemory)) < 0 ||
      rpc_write(conn, &mipmapDesc_present, sizeof(uint8_t)) < 0 ||
      (mipmapDesc_present &&
       rpc_write(conn, mipmapDesc,
                 sizeof(const CUDA_EXTERNAL_MEMORY_MIPMAPPED_ARRAY_DESC)) <
           0) ||
      rpc_wait_for_response(conn) < 0 ||
      (mipmap_present &&
       rpc_read(conn, mipmap, sizeof(CUmipmappedArray)) < 0) ||
      rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
      rpc_read_end(conn) < 0)
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  return return_value;
}

CUresult cuDestroyExternalMemory(CUexternalMemory extMem) {
  lupine_route route = lupine_route_for_default();
  CUresult return_value;
  if (lupine_route_is_local(route))
    return lupine_call_real_cuda_fn(
        LUPINE_REAL_CUDA_SYMBOL("cuDestroyExternalMemory"), extMem);
  conn_t *conn = lupine_route_remote_conn(route);
  if (lupine_prepare_rpc(conn) < 0 ||
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
  if (lupine_route_is_local(route))
    return lupine_call_real_cuda_fn(
        LUPINE_REAL_CUDA_SYMBOL("cuImportExternalSemaphore"), extSem_out,
        semHandleDesc);
  conn_t *conn = lupine_route_remote_conn(route);
  uint8_t extSem_out_present = extSem_out != nullptr;
  uint8_t semHandleDesc_present = semHandleDesc != nullptr;
  if (lupine_prepare_rpc(conn) < 0 ||
      rpc_write_start_request_on(conn, RPC_cuImportExternalSemaphore,
                                 RPC_SCOPE_NONE) < 0 ||
      rpc_write(conn, &extSem_out_present, sizeof(uint8_t)) < 0 ||
      (extSem_out_present &&
       rpc_write(conn, extSem_out, sizeof(CUexternalSemaphore)) < 0) ||
      rpc_write(conn, &semHandleDesc_present, sizeof(uint8_t)) < 0 ||
      (semHandleDesc_present &&
       rpc_write(conn, semHandleDesc,
                 sizeof(const CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC)) < 0) ||
      rpc_wait_for_response(conn) < 0 ||
      (extSem_out_present &&
       rpc_read(conn, extSem_out, sizeof(CUexternalSemaphore)) < 0) ||
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
  if (lupine_route_is_local(route))
    return lupine_call_real_cuda_fn(
        LUPINE_REAL_CUDA_SYMBOL("cuSignalExternalSemaphoresAsync"), extSemArray,
        paramsArray, numExtSems, stream);
  conn_t *conn = lupine_route_remote_conn(route);
  uint8_t extSemArray_null = extSemArray == nullptr ? 1 : 0;
  uint8_t paramsArray_null = paramsArray == nullptr ? 1 : 0;
  if (lupine_prepare_rpc(conn) < 0 ||
      lupine_rpc_start_on_stream(conn, RPC_cuSignalExternalSemaphoresAsync,
                                 stream) < 0 ||
      lupine_rpc_request_on_each(
          conn, reinterpret_cast<const void *const *>(extSemArray),
          numExtSems) < 0 ||
      rpc_write(conn, &numExtSems, sizeof(unsigned int)) < 0 ||
      rpc_write(conn, &extSemArray_null, sizeof(uint8_t)) < 0 ||
      (!extSemArray_null &&
       numExtSems * sizeof(const CUexternalSemaphore) != 0 &&
       rpc_write(conn, extSemArray,
                 numExtSems * sizeof(const CUexternalSemaphore)) < 0) ||
      rpc_write(conn, &paramsArray_null, sizeof(uint8_t)) < 0 ||
      (!paramsArray_null &&
       numExtSems * sizeof(const CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS) != 0 &&
       rpc_write(conn, paramsArray,
                 numExtSems *
                     sizeof(const CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS)) <
           0) ||
      rpc_write(conn, &stream, sizeof(CUstream)) < 0 || rpc_write_end(conn) < 0)
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  return_value = CUDA_SUCCESS;
  return return_value;
}

CUresult cuWaitExternalSemaphoresAsync(
    const CUexternalSemaphore *extSemArray,
    const CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS *paramsArray,
    unsigned int numExtSems, CUstream stream) {
  lupine_route route = (stream != nullptr ? lupine_route_for_stream(stream)
                                          : lupine_route_for_default());
  CUresult return_value;
  if (lupine_route_is_local(route))
    return lupine_call_real_cuda_fn(
        LUPINE_REAL_CUDA_SYMBOL("cuWaitExternalSemaphoresAsync"), extSemArray,
        paramsArray, numExtSems, stream);
  conn_t *conn = lupine_route_remote_conn(route);
  uint8_t extSemArray_null = extSemArray == nullptr ? 1 : 0;
  uint8_t paramsArray_null = paramsArray == nullptr ? 1 : 0;
  if (lupine_prepare_rpc(conn) < 0 ||
      lupine_rpc_start_on_stream(conn, RPC_cuWaitExternalSemaphoresAsync,
                                 stream) < 0 ||
      lupine_rpc_request_on_each(
          conn, reinterpret_cast<const void *const *>(extSemArray),
          numExtSems) < 0 ||
      rpc_write(conn, &numExtSems, sizeof(unsigned int)) < 0 ||
      rpc_write(conn, &extSemArray_null, sizeof(uint8_t)) < 0 ||
      (!extSemArray_null &&
       numExtSems * sizeof(const CUexternalSemaphore) != 0 &&
       rpc_write(conn, extSemArray,
                 numExtSems * sizeof(const CUexternalSemaphore)) < 0) ||
      rpc_write(conn, &paramsArray_null, sizeof(uint8_t)) < 0 ||
      (!paramsArray_null &&
       numExtSems * sizeof(const CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS) != 0 &&
       rpc_write(conn, paramsArray,
                 numExtSems *
                     sizeof(const CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS)) < 0) ||
      rpc_write(conn, &stream, sizeof(CUstream)) < 0 || rpc_write_end(conn) < 0)
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  return_value = CUDA_SUCCESS;
  return return_value;
}

CUresult cuDestroyExternalSemaphore(CUexternalSemaphore extSem) {
  lupine_route route = lupine_route_for_default();
  CUresult return_value;
  if (lupine_route_is_local(route))
    return lupine_call_real_cuda_fn(
        LUPINE_REAL_CUDA_SYMBOL("cuDestroyExternalSemaphore"), extSem);
  conn_t *conn = lupine_route_remote_conn(route);
  if (lupine_prepare_rpc(conn) < 0 ||
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
  if (lupine_route_is_local(route))
    return lupine_call_real_cuda_fn(
        LUPINE_REAL_CUDA_SYMBOL("cuStreamWaitValue32_v2"), stream, addr, value,
        flags);
  conn_t *conn = lupine_route_remote_conn(route);
  if (lupine_prepare_rpc(conn) < 0 ||
      lupine_rpc_start_on_stream(conn, RPC_cuStreamWaitValue32_v2, stream) <
          0 ||
      rpc_write(conn, &stream, sizeof(CUstream)) < 0 ||
      rpc_write(conn, &addr, sizeof(CUdeviceptr)) < 0 ||
      rpc_write(conn, &value, sizeof(cuuint32_t)) < 0 ||
      rpc_write(conn, &flags, sizeof(unsigned int)) < 0 ||
      rpc_write_end(conn) < 0)
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  return_value = CUDA_SUCCESS;
  return return_value;
}

CUresult cuStreamWaitValue64_v2(CUstream stream, CUdeviceptr addr,
                                cuuint64_t value, unsigned int flags) {
  lupine_route route = (stream != nullptr ? lupine_route_for_stream(stream)
                                          : lupine_route_for_default());
  CUresult return_value;
  if (lupine_route_is_local(route))
    return lupine_call_real_cuda_fn(
        LUPINE_REAL_CUDA_SYMBOL("cuStreamWaitValue64_v2"), stream, addr, value,
        flags);
  conn_t *conn = lupine_route_remote_conn(route);
  if (lupine_prepare_rpc(conn) < 0 ||
      lupine_rpc_start_on_stream(conn, RPC_cuStreamWaitValue64_v2, stream) <
          0 ||
      rpc_write(conn, &stream, sizeof(CUstream)) < 0 ||
      rpc_write(conn, &addr, sizeof(CUdeviceptr)) < 0 ||
      rpc_write(conn, &value, sizeof(cuuint64_t)) < 0 ||
      rpc_write(conn, &flags, sizeof(unsigned int)) < 0 ||
      rpc_write_end(conn) < 0)
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  return_value = CUDA_SUCCESS;
  return return_value;
}

CUresult cuStreamWriteValue32_v2(CUstream stream, CUdeviceptr addr,
                                 cuuint32_t value, unsigned int flags) {
  lupine_route route = (stream != nullptr ? lupine_route_for_stream(stream)
                                          : lupine_route_for_default());
  CUresult return_value;
  if (lupine_route_is_local(route))
    return lupine_call_real_cuda_fn(
        LUPINE_REAL_CUDA_SYMBOL("cuStreamWriteValue32_v2"), stream, addr, value,
        flags);
  conn_t *conn = lupine_route_remote_conn(route);
  if (lupine_prepare_rpc(conn) < 0 ||
      lupine_rpc_start_on_stream(conn, RPC_cuStreamWriteValue32_v2, stream) <
          0 ||
      rpc_write(conn, &stream, sizeof(CUstream)) < 0 ||
      rpc_write(conn, &addr, sizeof(CUdeviceptr)) < 0 ||
      rpc_write(conn, &value, sizeof(cuuint32_t)) < 0 ||
      rpc_write(conn, &flags, sizeof(unsigned int)) < 0 ||
      rpc_write_end(conn) < 0)
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  return_value = CUDA_SUCCESS;
  return return_value;
}

CUresult cuStreamWriteValue64_v2(CUstream stream, CUdeviceptr addr,
                                 cuuint64_t value, unsigned int flags) {
  lupine_route route = (stream != nullptr ? lupine_route_for_stream(stream)
                                          : lupine_route_for_default());
  CUresult return_value;
  if (lupine_route_is_local(route))
    return lupine_call_real_cuda_fn(
        LUPINE_REAL_CUDA_SYMBOL("cuStreamWriteValue64_v2"), stream, addr, value,
        flags);
  conn_t *conn = lupine_route_remote_conn(route);
  if (lupine_prepare_rpc(conn) < 0 ||
      lupine_rpc_start_on_stream(conn, RPC_cuStreamWriteValue64_v2, stream) <
          0 ||
      rpc_write(conn, &stream, sizeof(CUstream)) < 0 ||
      rpc_write(conn, &addr, sizeof(CUdeviceptr)) < 0 ||
      rpc_write(conn, &value, sizeof(cuuint64_t)) < 0 ||
      rpc_write(conn, &flags, sizeof(unsigned int)) < 0 ||
      rpc_write_end(conn) < 0)
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  return_value = CUDA_SUCCESS;
  return return_value;
}

CUresult cuStreamBatchMemOp_v2(CUstream stream, unsigned int count,
                               CUstreamBatchMemOpParams *paramArray,
                               unsigned int flags) {
  lupine_route route = (stream != nullptr ? lupine_route_for_stream(stream)
                                          : lupine_route_for_default());
  CUresult return_value;
  if (lupine_route_is_local(route))
    return lupine_call_real_cuda_fn(
        LUPINE_REAL_CUDA_SYMBOL("cuStreamBatchMemOp_v2"), stream, count,
        paramArray, flags);
  conn_t *conn = lupine_route_remote_conn(route);
  uint8_t paramArray_null = paramArray == nullptr ? 1 : 0;
  if (lupine_prepare_rpc(conn) < 0 ||
      lupine_rpc_start_on_stream(conn, RPC_cuStreamBatchMemOp_v2, stream) < 0 ||
      rpc_write(conn, &stream, sizeof(CUstream)) < 0 ||
      rpc_write(conn, &count, sizeof(unsigned int)) < 0 ||
      rpc_write(conn, &paramArray_null, sizeof(uint8_t)) < 0 ||
      (!paramArray_null && count * sizeof(CUstreamBatchMemOpParams) != 0 &&
       rpc_write(conn, paramArray, count * sizeof(CUstreamBatchMemOpParams)) <
           0) ||
      rpc_write(conn, &flags, sizeof(unsigned int)) < 0 ||
      rpc_write_end(conn) < 0)
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  return_value = CUDA_SUCCESS;
  return return_value;
}

CUresult cuFuncGetAttribute(int *pi, CUfunction_attribute attrib,
                            CUfunction hfunc) {
  lupine_route route = lupine_route_for_function(hfunc);
  if (pi == nullptr)
    return CUDA_ERROR_INVALID_VALUE;
  hfunc = lupine_translate_private_function_for_rpc(hfunc);
  route = lupine_route_for_function(hfunc);
  lupine_function_attribute_key key{lupine_route_identity(route), hfunc,
                                    (int)attrib};
  if (!lupine_route_is_local(route) && lupine_attribute_get(key, *pi))
    return CUDA_SUCCESS;
  CUresult return_value;
  if (lupine_route_is_local(route)) {
    return_value = lupine_call_real_cuda_fn(
        LUPINE_REAL_CUDA_SYMBOL("cuFuncGetAttribute"), pi, attrib, hfunc);
    if (return_value == CUDA_SUCCESS && !lupine_route_is_local(route))
      lupine_attribute_set(key, *pi);
    return return_value;
  }
  conn_t *conn = lupine_route_remote_conn(route);
  uint8_t pi_present = pi != nullptr;
  CUfunction hfunc_rpc = lupine_translate_private_function_for_rpc(hfunc);
  if (lupine_prepare_rpc(conn) < 0 ||
      rpc_write_start_request_on(conn, RPC_cuFuncGetAttribute, RPC_SCOPE_NONE) <
          0 ||
      rpc_write(conn, &pi_present, sizeof(uint8_t)) < 0 ||
      (pi_present && rpc_write(conn, pi, sizeof(int)) < 0) ||
      rpc_write(conn, &attrib, sizeof(CUfunction_attribute)) < 0 ||
      rpc_write(conn, &hfunc_rpc, sizeof(CUfunction)) < 0 ||
      rpc_wait_for_response(conn) < 0 ||
      (pi_present && rpc_read(conn, pi, sizeof(int)) < 0) ||
      rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
      rpc_read_end(conn) < 0)
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  if (return_value == CUDA_SUCCESS && !lupine_route_is_local(route))
    lupine_attribute_set(key, *pi);
  return return_value;
}

CUresult cuFuncSetAttribute(CUfunction hfunc, CUfunction_attribute attrib,
                            int value) {
  lupine_route route = lupine_route_for_function(hfunc);
  lupine_function_attribute_key key{
      lupine_route_identity(route),
      lupine_translate_private_function_for_rpc(hfunc), (int)attrib};
  int cached = 0;
  if (lupine_attribute_get(key, cached, lupine_attribute_source::set) &&
      cached == value)
    return CUDA_SUCCESS;
  CUresult return_value;
  if (lupine_route_is_local(route)) {
    return_value = lupine_call_real_cuda_fn(
        LUPINE_REAL_CUDA_SYMBOL("cuFuncSetAttribute"), hfunc, attrib, value);
    if (return_value == CUDA_SUCCESS)
      lupine_attribute_set(key, value, lupine_attribute_source::set);
    return return_value;
  }
  conn_t *conn = lupine_route_remote_conn(route);
  CUfunction hfunc_rpc = lupine_translate_private_function_for_rpc(hfunc);
  if (lupine_prepare_rpc(conn) < 0 ||
      rpc_write_start_request_on(conn, RPC_cuFuncSetAttribute, RPC_SCOPE_NONE) <
          0 ||
      rpc_write(conn, &hfunc_rpc, sizeof(CUfunction)) < 0 ||
      rpc_write(conn, &attrib, sizeof(CUfunction_attribute)) < 0 ||
      rpc_write(conn, &value, sizeof(int)) < 0 ||
      rpc_wait_for_response(conn) < 0 ||
      rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
      rpc_read_end(conn) < 0)
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  if (return_value == CUDA_SUCCESS)
    lupine_attribute_set(key, value, lupine_attribute_source::set);
  return return_value;
}

CUresult cuFuncSetCacheConfig(CUfunction hfunc, CUfunc_cache config) {
  lupine_route route = lupine_route_for_function(hfunc);
  CUresult return_value;
  if (lupine_route_is_local(route)) {
    return_value = lupine_call_real_cuda_fn(
        LUPINE_REAL_CUDA_SYMBOL("cuFuncSetCacheConfig"), hfunc, config);
    if (return_value == CUDA_SUCCESS)
      lupine_disable_local_occupancy();
    return return_value;
  }
  conn_t *conn = lupine_route_remote_conn(route);
  CUfunction hfunc_rpc = lupine_translate_private_function_for_rpc(hfunc);
  if (lupine_prepare_rpc(conn) < 0 ||
      rpc_write_start_request_on(conn, RPC_cuFuncSetCacheConfig,
                                 RPC_SCOPE_NONE) < 0 ||
      rpc_write(conn, &hfunc_rpc, sizeof(CUfunction)) < 0 ||
      rpc_write(conn, &config, sizeof(CUfunc_cache)) < 0 ||
      rpc_write_end(conn) < 0)
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  return_value = CUDA_SUCCESS;
  if (return_value == CUDA_SUCCESS)
    lupine_disable_local_occupancy();
  return return_value;
}

CUresult cuFuncGetModule(CUmodule *hmod, CUfunction hfunc) {
  lupine_route route = lupine_route_for_function(hfunc);
  CUresult return_value;
  if (lupine_route_is_local(route)) {
    return_value = lupine_call_real_cuda_fn(
        LUPINE_REAL_CUDA_SYMBOL("cuFuncGetModule"), hmod, hfunc);
    if (return_value == CUDA_SUCCESS && hmod != nullptr) {
      lupine_note_module_owner_route(*hmod, route);
    }
    return return_value;
  }
  conn_t *conn = lupine_route_remote_conn(route);
  uint8_t hmod_present = hmod != nullptr;
  CUfunction hfunc_rpc = lupine_translate_private_function_for_rpc(hfunc);
  if (lupine_prepare_rpc(conn) < 0 ||
      rpc_write_start_request_on(conn, RPC_cuFuncGetModule, RPC_SCOPE_NONE) <
          0 ||
      rpc_write(conn, &hmod_present, sizeof(uint8_t)) < 0 ||
      (hmod_present && rpc_write(conn, hmod, sizeof(CUmodule)) < 0) ||
      rpc_write(conn, &hfunc_rpc, sizeof(CUfunction)) < 0 ||
      rpc_wait_for_response(conn) < 0 ||
      (hmod_present && rpc_read(conn, hmod, sizeof(CUmodule)) < 0) ||
      rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
      rpc_read_end(conn) < 0)
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  if (return_value == CUDA_SUCCESS && hmod != nullptr) {
    lupine_note_module_owner_route(*hmod, route);
  }
  return return_value;
}

CUresult cuFuncGetName(const char **name, CUfunction hfunc) {
  lupine_route route = lupine_route_for_function(hfunc);
  CUresult return_value;
  if (lupine_route_is_local(route))
    return lupine_call_real_cuda_fn(LUPINE_REAL_CUDA_SYMBOL("cuFuncGetName"),
                                    name, hfunc);
  conn_t *conn = lupine_route_remote_conn(route);
  uint8_t name_present = name != nullptr;
  std::size_t name_len = 0;
  lupine_scratch_buffer<256> lupine_name_buf;
  char *name_data = nullptr;
  CUfunction hfunc_rpc = lupine_translate_private_function_for_rpc(hfunc);
  if (lupine_prepare_rpc(conn) < 0 ||
      rpc_write_start_request_on(conn, RPC_cuFuncGetName, RPC_SCOPE_NONE) < 0 ||
      rpc_write(conn, &name_present, sizeof(uint8_t)) < 0 ||
      rpc_write(conn, &hfunc_rpc, sizeof(CUfunction)) < 0 ||
      rpc_wait_for_response(conn) < 0 ||
      (name_present &&
       (rpc_read(conn, &name_len, sizeof(std::size_t)) < 0 ||
        name_len > (1U << 20) ||
        (name_len != 0 &&
         ((name_data = static_cast<char *>(
               lupine_name_buf.alloc_uninit(name_len))) == nullptr ||
          rpc_read(conn, name_data, name_len) < 0)))) ||
      rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
      rpc_read_end(conn) < 0)
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  if (return_value == CUDA_SUCCESS && name_present) {
    const char *name_stored = lupine_retain_returned_string(
        reinterpret_cast<const void *>(hfunc),
        name_data != nullptr ? name_data : "", name_len);
    if (name_stored == nullptr)
      return CUDA_ERROR_OUT_OF_MEMORY;
    *name = name_stored;
  }
  return return_value;
}

CUresult cuLaunchHostFunc(CUstream hStream, CUhostFn fn, void *userData) {
  lupine_route route = (hStream != nullptr ? lupine_route_for_stream(hStream)
                                           : lupine_route_for_default());
  if (fn == nullptr)
    return CUDA_ERROR_INVALID_VALUE;
  CUresult return_value;
  if (lupine_route_is_local(route))
    return lupine_call_real_cuda_fn(LUPINE_REAL_CUDA_SYMBOL("cuLaunchHostFunc"),
                                    hStream, fn, userData);
  conn_t *conn = lupine_route_remote_conn(route);
  if (lupine_prepare_rpc(conn) < 0 ||
      lupine_rpc_start_on_stream(conn, RPC_cuLaunchHostFunc, hStream) < 0 ||
      rpc_write(conn, &hStream, sizeof(CUstream)) < 0 ||
      rpc_write(conn, &fn, sizeof(CUhostFn)) < 0 ||
      rpc_write(conn, &userData, sizeof(void *)) < 0 || rpc_write_end(conn) < 0)
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  return_value = CUDA_SUCCESS;
  return return_value;
}

CUresult cuFuncSetBlockShape(CUfunction hfunc, int x, int y, int z) {
  lupine_route route = lupine_route_for_function(hfunc);
  CUresult return_value;
  if (lupine_route_is_local(route))
    return lupine_call_real_cuda_fn(
        LUPINE_REAL_CUDA_SYMBOL("cuFuncSetBlockShape"), hfunc, x, y, z);
  conn_t *conn = lupine_route_remote_conn(route);
  CUfunction hfunc_rpc = lupine_translate_private_function_for_rpc(hfunc);
  if (lupine_prepare_rpc(conn) < 0 ||
      rpc_write_start_request_on(conn, RPC_cuFuncSetBlockShape,
                                 RPC_SCOPE_NONE) < 0 ||
      rpc_write(conn, &hfunc_rpc, sizeof(CUfunction)) < 0 ||
      rpc_write(conn, &x, sizeof(int)) < 0 ||
      rpc_write(conn, &y, sizeof(int)) < 0 ||
      rpc_write(conn, &z, sizeof(int)) < 0 || rpc_write_end(conn) < 0)
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  return_value = CUDA_SUCCESS;
  return return_value;
}

CUresult cuFuncSetSharedSize(CUfunction hfunc, unsigned int bytes) {
  lupine_route route = lupine_route_for_function(hfunc);
  CUresult return_value;
  if (lupine_route_is_local(route))
    return lupine_call_real_cuda_fn(
        LUPINE_REAL_CUDA_SYMBOL("cuFuncSetSharedSize"), hfunc, bytes);
  conn_t *conn = lupine_route_remote_conn(route);
  CUfunction hfunc_rpc = lupine_translate_private_function_for_rpc(hfunc);
  if (lupine_prepare_rpc(conn) < 0 ||
      rpc_write_start_request_on(conn, RPC_cuFuncSetSharedSize,
                                 RPC_SCOPE_NONE) < 0 ||
      rpc_write(conn, &hfunc_rpc, sizeof(CUfunction)) < 0 ||
      rpc_write(conn, &bytes, sizeof(unsigned int)) < 0 ||
      rpc_write_end(conn) < 0)
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  return_value = CUDA_SUCCESS;
  return return_value;
}

CUresult cuParamSetSize(CUfunction hfunc, unsigned int numbytes) {
  lupine_route route = lupine_route_for_function(hfunc);
  CUresult return_value;
  if (lupine_route_is_local(route))
    return lupine_call_real_cuda_fn(LUPINE_REAL_CUDA_SYMBOL("cuParamSetSize"),
                                    hfunc, numbytes);
  conn_t *conn = lupine_route_remote_conn(route);
  CUfunction hfunc_rpc = lupine_translate_private_function_for_rpc(hfunc);
  if (lupine_prepare_rpc(conn) < 0 ||
      rpc_write_start_request_on(conn, RPC_cuParamSetSize, RPC_SCOPE_NONE) <
          0 ||
      rpc_write(conn, &hfunc_rpc, sizeof(CUfunction)) < 0 ||
      rpc_write(conn, &numbytes, sizeof(unsigned int)) < 0 ||
      rpc_write_end(conn) < 0)
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  return_value = CUDA_SUCCESS;
  return return_value;
}

CUresult cuParamSeti(CUfunction hfunc, int offset, unsigned int value) {
  lupine_route route = lupine_route_for_function(hfunc);
  CUresult return_value;
  if (lupine_route_is_local(route))
    return lupine_call_real_cuda_fn(LUPINE_REAL_CUDA_SYMBOL("cuParamSeti"),
                                    hfunc, offset, value);
  conn_t *conn = lupine_route_remote_conn(route);
  CUfunction hfunc_rpc = lupine_translate_private_function_for_rpc(hfunc);
  if (lupine_prepare_rpc(conn) < 0 ||
      rpc_write_start_request_on(conn, RPC_cuParamSeti, RPC_SCOPE_NONE) < 0 ||
      rpc_write(conn, &hfunc_rpc, sizeof(CUfunction)) < 0 ||
      rpc_write(conn, &offset, sizeof(int)) < 0 ||
      rpc_write(conn, &value, sizeof(unsigned int)) < 0 ||
      rpc_write_end(conn) < 0)
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  return_value = CUDA_SUCCESS;
  return return_value;
}

CUresult cuParamSetf(CUfunction hfunc, int offset, float value) {
  lupine_route route = lupine_route_for_function(hfunc);
  CUresult return_value;
  if (lupine_route_is_local(route))
    return lupine_call_real_cuda_fn(LUPINE_REAL_CUDA_SYMBOL("cuParamSetf"),
                                    hfunc, offset, value);
  conn_t *conn = lupine_route_remote_conn(route);
  CUfunction hfunc_rpc = lupine_translate_private_function_for_rpc(hfunc);
  if (lupine_prepare_rpc(conn) < 0 ||
      rpc_write_start_request_on(conn, RPC_cuParamSetf, RPC_SCOPE_NONE) < 0 ||
      rpc_write(conn, &hfunc_rpc, sizeof(CUfunction)) < 0 ||
      rpc_write(conn, &offset, sizeof(int)) < 0 ||
      rpc_write(conn, &value, sizeof(float)) < 0 || rpc_write_end(conn) < 0)
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  return_value = CUDA_SUCCESS;
  return return_value;
}

CUresult cuParamSetv(CUfunction hfunc, int offset, void *ptr,
                     unsigned int numbytes) {
  lupine_route route = lupine_route_for_function(hfunc);
  CUresult return_value;
  if (lupine_route_is_local(route))
    return lupine_call_real_cuda_fn(LUPINE_REAL_CUDA_SYMBOL("cuParamSetv"),
                                    hfunc, offset, ptr, numbytes);
  conn_t *conn = lupine_route_remote_conn(route);
  CUfunction hfunc_rpc = lupine_translate_private_function_for_rpc(hfunc);
  uint8_t ptr_null = ptr == nullptr ? 1 : 0;
  if (lupine_prepare_rpc(conn) < 0 ||
      rpc_write_start_request_on(conn, RPC_cuParamSetv, RPC_SCOPE_NONE) < 0 ||
      rpc_write(conn, &hfunc_rpc, sizeof(CUfunction)) < 0 ||
      rpc_write(conn, &offset, sizeof(int)) < 0 ||
      rpc_write(conn, &numbytes, sizeof(unsigned int)) < 0 ||
      rpc_write(conn, &ptr_null, sizeof(uint8_t)) < 0 ||
      (!ptr_null && numbytes != 0 && rpc_write(conn, ptr, numbytes) < 0) ||
      rpc_write_end(conn) < 0)
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  return_value = CUDA_SUCCESS;
  return return_value;
}

CUresult cuLaunch(CUfunction f) {
  lupine_route route = lupine_route_for_function(f);
  CUresult return_value;
  if (lupine_route_is_local(route))
    return lupine_call_real_cuda_fn(LUPINE_REAL_CUDA_SYMBOL("cuLaunch"), f);
  conn_t *conn = lupine_route_remote_conn(route);
  CUfunction f_rpc = lupine_translate_private_function_for_rpc(f);
  if (lupine_prepare_rpc(conn) < 0 ||
      lupine_rpc_start_on_stream(conn, RPC_cuLaunch, nullptr) < 0 ||
      rpc_write(conn, &f_rpc, sizeof(CUfunction)) < 0 ||
      rpc_write_end(conn) < 0)
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  return_value = CUDA_SUCCESS;
  return return_value;
}

CUresult cuLaunchGrid(CUfunction f, int grid_width, int grid_height) {
  lupine_route route = lupine_route_for_function(f);
  CUresult return_value;
  if (lupine_route_is_local(route))
    return lupine_call_real_cuda_fn(LUPINE_REAL_CUDA_SYMBOL("cuLaunchGrid"), f,
                                    grid_width, grid_height);
  conn_t *conn = lupine_route_remote_conn(route);
  CUfunction f_rpc = lupine_translate_private_function_for_rpc(f);
  if (lupine_prepare_rpc(conn) < 0 ||
      lupine_rpc_start_on_stream(conn, RPC_cuLaunchGrid, nullptr) < 0 ||
      rpc_write(conn, &f_rpc, sizeof(CUfunction)) < 0 ||
      rpc_write(conn, &grid_width, sizeof(int)) < 0 ||
      rpc_write(conn, &grid_height, sizeof(int)) < 0 || rpc_write_end(conn) < 0)
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  return_value = CUDA_SUCCESS;
  return return_value;
}

CUresult cuLaunchGridAsync(CUfunction f, int grid_width, int grid_height,
                           CUstream hStream) {
  lupine_route route = lupine_route_for_function(f);
  CUresult return_value;
  if (lupine_route_is_local(route))
    return lupine_call_real_cuda_fn(
        LUPINE_REAL_CUDA_SYMBOL("cuLaunchGridAsync"), f, grid_width,
        grid_height, hStream);
  conn_t *conn = lupine_route_remote_conn(route);
  CUfunction f_rpc = lupine_translate_private_function_for_rpc(f);
  if (lupine_prepare_rpc(conn) < 0 ||
      lupine_rpc_start_on_stream(conn, RPC_cuLaunchGridAsync, hStream) < 0 ||
      rpc_write(conn, &f_rpc, sizeof(CUfunction)) < 0 ||
      rpc_write(conn, &grid_width, sizeof(int)) < 0 ||
      rpc_write(conn, &grid_height, sizeof(int)) < 0 ||
      rpc_write(conn, &hStream, sizeof(CUstream)) < 0 ||
      rpc_write_end(conn) < 0)
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  return_value = CUDA_SUCCESS;
  return return_value;
}

extern "C" CUresult
cuLaunchCooperativeKernelMultiDevice(CUDA_LAUNCH_PARAMS *launchParamsList,
                                     unsigned int numDevices,
                                     unsigned int flags) {
  (void)launchParamsList;
  (void)numDevices;
  (void)flags;
  return CUDA_ERROR_NOT_SUPPORTED;
}

CUresult cuParamSetTexRef(CUfunction hfunc, int texunit, CUtexref hTexRef) {
  lupine_route route = lupine_route_for_function(hfunc);
  CUresult return_value;
  if (lupine_route_is_local(route))
    return lupine_call_real_cuda_fn(LUPINE_REAL_CUDA_SYMBOL("cuParamSetTexRef"),
                                    hfunc, texunit, hTexRef);
  conn_t *conn = lupine_route_remote_conn(route);
  CUfunction hfunc_rpc = lupine_translate_private_function_for_rpc(hfunc);
  if (lupine_prepare_rpc(conn) < 0 ||
      rpc_write_start_request_on(conn, RPC_cuParamSetTexRef, RPC_SCOPE_NONE) <
          0 ||
      rpc_write(conn, &hfunc_rpc, sizeof(CUfunction)) < 0 ||
      rpc_write(conn, &texunit, sizeof(int)) < 0 ||
      rpc_write(conn, &hTexRef, sizeof(CUtexref)) < 0 ||
      rpc_write_end(conn) < 0)
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  return_value = CUDA_SUCCESS;
  return return_value;
}

CUresult cuFuncSetSharedMemConfig(CUfunction hfunc, CUsharedconfig config) {
  lupine_route route = lupine_route_for_function(hfunc);
  CUresult return_value;
  if (lupine_route_is_local(route))
    return lupine_call_real_cuda_fn(
        LUPINE_REAL_CUDA_SYMBOL("cuFuncSetSharedMemConfig"), hfunc, config);
  conn_t *conn = lupine_route_remote_conn(route);
  CUfunction hfunc_rpc = lupine_translate_private_function_for_rpc(hfunc);
  if (lupine_prepare_rpc(conn) < 0 ||
      rpc_write_start_request_on(conn, RPC_cuFuncSetSharedMemConfig,
                                 RPC_SCOPE_NONE) < 0 ||
      rpc_write(conn, &hfunc_rpc, sizeof(CUfunction)) < 0 ||
      rpc_write(conn, &config, sizeof(CUsharedconfig)) < 0 ||
      rpc_write_end(conn) < 0)
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  return_value = CUDA_SUCCESS;
  return return_value;
}

CUresult cuGraphCreate(CUgraph *phGraph, unsigned int flags) {
  lupine_route route = lupine_route_for_default();
  CUresult return_value;
  if (lupine_route_is_local(route)) {
    return_value = lupine_call_real_cuda_fn(
        LUPINE_REAL_CUDA_SYMBOL("cuGraphCreate"), phGraph, flags);
    if (return_value == CUDA_SUCCESS && phGraph != nullptr) {
      lupine_note_graph_owner_route(*phGraph, route);
    }
    return return_value;
  }
  conn_t *conn = lupine_route_remote_conn(route);
  uint8_t phGraph_present = phGraph != nullptr;
  if (lupine_prepare_rpc(conn) < 0 ||
      rpc_write_start_request_on(conn, RPC_cuGraphCreate, RPC_SCOPE_NONE) < 0 ||
      rpc_write(conn, &phGraph_present, sizeof(uint8_t)) < 0 ||
      (phGraph_present && rpc_write(conn, phGraph, sizeof(CUgraph)) < 0) ||
      rpc_write(conn, &flags, sizeof(unsigned int)) < 0 ||
      rpc_wait_for_response(conn) < 0 ||
      (phGraph_present && rpc_read(conn, phGraph, sizeof(CUgraph)) < 0) ||
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
  if (lupine_route_is_local(route))
    return lupine_call_real_cuda_fn(
        LUPINE_REAL_CUDA_SYMBOL("cuGraphMemcpyNodeGetParams"), hNode,
        nodeParams);
  conn_t *conn = lupine_route_remote_conn(route);
  uint8_t nodeParams_present = nodeParams != nullptr;
  if (lupine_prepare_rpc(conn) < 0 ||
      rpc_write_start_request_on(conn, RPC_cuGraphMemcpyNodeGetParams,
                                 RPC_SCOPE_NONE) < 0 ||
      rpc_write(conn, &hNode, sizeof(CUgraphNode)) < 0 ||
      rpc_write(conn, &nodeParams_present, sizeof(uint8_t)) < 0 ||
      (nodeParams_present &&
       rpc_write(conn, nodeParams, sizeof(CUDA_MEMCPY3D)) < 0) ||
      rpc_wait_for_response(conn) < 0 ||
      (nodeParams_present &&
       rpc_read(conn, nodeParams, sizeof(CUDA_MEMCPY3D)) < 0) ||
      rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
      rpc_read_end(conn) < 0)
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  return return_value;
}

CUresult cuGraphMemcpyNodeSetParams(CUgraphNode hNode,
                                    const CUDA_MEMCPY3D *nodeParams) {
  lupine_route route = lupine_route_for_graph_node(hNode);
  CUresult return_value;
  if (lupine_route_is_local(route))
    return lupine_call_real_cuda_fn(
        LUPINE_REAL_CUDA_SYMBOL("cuGraphMemcpyNodeSetParams"), hNode,
        nodeParams);
  conn_t *conn = lupine_route_remote_conn(route);
  if (nodeParams == nullptr)
    return CUDA_ERROR_INVALID_VALUE;
  if (lupine_prepare_rpc(conn) < 0 ||
      rpc_write_start_request_on(conn, RPC_cuGraphMemcpyNodeSetParams,
                                 RPC_SCOPE_NONE) < 0 ||
      rpc_write(conn, &hNode, sizeof(CUgraphNode)) < 0 ||
      rpc_write(conn, nodeParams, sizeof(const CUDA_MEMCPY3D)) < 0 ||
      rpc_wait_for_response(conn) < 0 ||
      rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
      rpc_read_end(conn) < 0)
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  return return_value;
}

CUresult cuGraphAddMemsetNode(CUgraphNode *phGraphNode, CUgraph hGraph,
                              const CUgraphNode *dependencies,
                              size_t numDependencies,
                              const CUDA_MEMSET_NODE_PARAMS *memsetParams,
                              CUcontext ctx) {
  lupine_route route = lupine_route_for_graph(hGraph);
  CUresult return_value;
  if (lupine_route_is_local(route)) {
    return_value = lupine_call_real_cuda_fn(
        LUPINE_REAL_CUDA_SYMBOL("cuGraphAddMemsetNode"), phGraphNode, hGraph,
        dependencies, numDependencies, memsetParams, ctx);
    if (return_value == CUDA_SUCCESS && phGraphNode != nullptr) {
      lupine_note_graph_node_owner_route(*phGraphNode, route);
    }
    return return_value;
  }
  conn_t *conn = lupine_route_remote_conn(route);
  uint8_t phGraphNode_present = phGraphNode != nullptr;
  uint8_t dependencies_null = dependencies == nullptr ? 1 : 0;
  uint8_t memsetParams_present = memsetParams != nullptr;
  if (lupine_prepare_rpc(conn) < 0 ||
      rpc_write_start_request_on(conn, RPC_cuGraphAddMemsetNode,
                                 RPC_SCOPE_NONE) < 0 ||
      rpc_write(conn, &phGraphNode_present, sizeof(uint8_t)) < 0 ||
      (phGraphNode_present &&
       rpc_write(conn, phGraphNode, sizeof(CUgraphNode)) < 0) ||
      rpc_write(conn, &hGraph, sizeof(CUgraph)) < 0 ||
      rpc_write(conn, &numDependencies, sizeof(size_t)) < 0 ||
      rpc_write(conn, &dependencies_null, sizeof(uint8_t)) < 0 ||
      (!dependencies_null && numDependencies * sizeof(const CUgraphNode) != 0 &&
       rpc_write(conn, dependencies,
                 numDependencies * sizeof(const CUgraphNode)) < 0) ||
      rpc_write(conn, &memsetParams_present, sizeof(uint8_t)) < 0 ||
      (memsetParams_present &&
       rpc_write(conn, memsetParams, sizeof(const CUDA_MEMSET_NODE_PARAMS)) <
           0) ||
      rpc_write(conn, &ctx, sizeof(CUcontext)) < 0 ||
      rpc_wait_for_response(conn) < 0 ||
      (phGraphNode_present &&
       rpc_read(conn, phGraphNode, sizeof(CUgraphNode)) < 0) ||
      rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
      rpc_read_end(conn) < 0)
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  if (return_value == CUDA_SUCCESS && phGraphNode != nullptr) {
    lupine_note_graph_node_owner_route(*phGraphNode, route);
  }
  return return_value;
}

CUresult cuGraphMemsetNodeGetParams(CUgraphNode hNode,
                                    CUDA_MEMSET_NODE_PARAMS *nodeParams) {
  lupine_route route = lupine_route_for_graph_node(hNode);
  CUresult return_value;
  if (lupine_route_is_local(route))
    return lupine_call_real_cuda_fn(
        LUPINE_REAL_CUDA_SYMBOL("cuGraphMemsetNodeGetParams"), hNode,
        nodeParams);
  conn_t *conn = lupine_route_remote_conn(route);
  uint8_t nodeParams_present = nodeParams != nullptr;
  if (lupine_prepare_rpc(conn) < 0 ||
      rpc_write_start_request_on(conn, RPC_cuGraphMemsetNodeGetParams,
                                 RPC_SCOPE_NONE) < 0 ||
      rpc_write(conn, &hNode, sizeof(CUgraphNode)) < 0 ||
      rpc_write(conn, &nodeParams_present, sizeof(uint8_t)) < 0 ||
      (nodeParams_present &&
       rpc_write(conn, nodeParams, sizeof(CUDA_MEMSET_NODE_PARAMS)) < 0) ||
      rpc_wait_for_response(conn) < 0 ||
      (nodeParams_present &&
       rpc_read(conn, nodeParams, sizeof(CUDA_MEMSET_NODE_PARAMS)) < 0) ||
      rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
      rpc_read_end(conn) < 0)
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  return return_value;
}

CUresult cuGraphMemsetNodeSetParams(CUgraphNode hNode,
                                    const CUDA_MEMSET_NODE_PARAMS *nodeParams) {
  lupine_route route = lupine_route_for_graph_node(hNode);
  CUresult return_value;
  if (lupine_route_is_local(route))
    return lupine_call_real_cuda_fn(
        LUPINE_REAL_CUDA_SYMBOL("cuGraphMemsetNodeSetParams"), hNode,
        nodeParams);
  conn_t *conn = lupine_route_remote_conn(route);
  uint8_t nodeParams_present = nodeParams != nullptr;
  if (lupine_prepare_rpc(conn) < 0 ||
      rpc_write_start_request_on(conn, RPC_cuGraphMemsetNodeSetParams,
                                 RPC_SCOPE_NONE) < 0 ||
      rpc_write(conn, &hNode, sizeof(CUgraphNode)) < 0 ||
      rpc_write(conn, &nodeParams_present, sizeof(uint8_t)) < 0 ||
      (nodeParams_present &&
       rpc_write(conn, nodeParams, sizeof(const CUDA_MEMSET_NODE_PARAMS)) <
           0) ||
      rpc_wait_for_response(conn) < 0 ||
      rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
      rpc_read_end(conn) < 0)
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  return return_value;
}

CUresult cuGraphAddHostNode(CUgraphNode *phGraphNode, CUgraph hGraph,
                            const CUgraphNode *dependencies,
                            size_t numDependencies,
                            const CUDA_HOST_NODE_PARAMS *nodeParams) {
  lupine_route route = lupine_route_for_graph(hGraph);
  CUresult return_value;
  if (lupine_route_is_local(route)) {
    return_value = lupine_call_real_cuda_fn(
        LUPINE_REAL_CUDA_SYMBOL("cuGraphAddHostNode"), phGraphNode, hGraph,
        dependencies, numDependencies, nodeParams);
    if (return_value == CUDA_SUCCESS && phGraphNode != nullptr) {
      lupine_note_graph_node_owner_route(*phGraphNode, route);
    }
    return return_value;
  }
  conn_t *conn = lupine_route_remote_conn(route);
  if (phGraphNode == nullptr)
    return CUDA_ERROR_INVALID_VALUE;
  if (numDependencies * sizeof(const CUgraphNode) != 0 &&
      dependencies == nullptr)
    return CUDA_ERROR_INVALID_VALUE;
  if (nodeParams == nullptr)
    return CUDA_ERROR_INVALID_VALUE;
  if (lupine_prepare_rpc(conn) < 0 ||
      rpc_write_start_request_on(conn, RPC_cuGraphAddHostNode, RPC_SCOPE_NONE) <
          0 ||
      rpc_write(conn, phGraphNode, sizeof(CUgraphNode)) < 0 ||
      rpc_write(conn, &hGraph, sizeof(CUgraph)) < 0 ||
      rpc_write(conn, &numDependencies, sizeof(size_t)) < 0 ||
      (numDependencies * sizeof(const CUgraphNode) != 0 &&
       rpc_write(conn, dependencies,
                 numDependencies * sizeof(const CUgraphNode)) < 0) ||
      rpc_write(conn, nodeParams, sizeof(const CUDA_HOST_NODE_PARAMS)) < 0 ||
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

CUresult cuGraphHostNodeGetParams(CUgraphNode hNode,
                                  CUDA_HOST_NODE_PARAMS *nodeParams) {
  lupine_route route = lupine_route_for_graph_node(hNode);
  CUresult return_value;
  if (lupine_route_is_local(route))
    return lupine_call_real_cuda_fn(
        LUPINE_REAL_CUDA_SYMBOL("cuGraphHostNodeGetParams"), hNode, nodeParams);
  conn_t *conn = lupine_route_remote_conn(route);
  if (nodeParams == nullptr)
    return CUDA_ERROR_INVALID_VALUE;
  if (lupine_prepare_rpc(conn) < 0 ||
      rpc_write_start_request_on(conn, RPC_cuGraphHostNodeGetParams,
                                 RPC_SCOPE_NONE) < 0 ||
      rpc_write(conn, &hNode, sizeof(CUgraphNode)) < 0 ||
      rpc_write(conn, nodeParams, sizeof(CUDA_HOST_NODE_PARAMS)) < 0 ||
      rpc_wait_for_response(conn) < 0 ||
      rpc_read(conn, nodeParams, sizeof(CUDA_HOST_NODE_PARAMS)) < 0 ||
      rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
      rpc_read_end(conn) < 0)
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  return return_value;
}

CUresult cuGraphHostNodeSetParams(CUgraphNode hNode,
                                  const CUDA_HOST_NODE_PARAMS *nodeParams) {
  lupine_route route = lupine_route_for_graph_node(hNode);
  CUresult return_value;
  if (lupine_route_is_local(route))
    return lupine_call_real_cuda_fn(
        LUPINE_REAL_CUDA_SYMBOL("cuGraphHostNodeSetParams"), hNode, nodeParams);
  conn_t *conn = lupine_route_remote_conn(route);
  if (nodeParams == nullptr)
    return CUDA_ERROR_INVALID_VALUE;
  if (lupine_prepare_rpc(conn) < 0 ||
      rpc_write_start_request_on(conn, RPC_cuGraphHostNodeSetParams,
                                 RPC_SCOPE_NONE) < 0 ||
      rpc_write(conn, &hNode, sizeof(CUgraphNode)) < 0 ||
      rpc_write(conn, nodeParams, sizeof(const CUDA_HOST_NODE_PARAMS)) < 0 ||
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
  if (lupine_route_is_local(route)) {
    return_value = lupine_call_real_cuda_fn(
        LUPINE_REAL_CUDA_SYMBOL("cuGraphAddChildGraphNode"), phGraphNode,
        hGraph, dependencies, numDependencies, childGraph);
    if (return_value == CUDA_SUCCESS && phGraphNode != nullptr) {
      lupine_note_graph_node_owner_route(*phGraphNode, route);
    }
    return return_value;
  }
  conn_t *conn = lupine_route_remote_conn(route);
  uint8_t phGraphNode_present = phGraphNode != nullptr;
  uint8_t dependencies_null = dependencies == nullptr ? 1 : 0;
  if (lupine_prepare_rpc(conn) < 0 ||
      rpc_write_start_request_on(conn, RPC_cuGraphAddChildGraphNode,
                                 RPC_SCOPE_NONE) < 0 ||
      rpc_write(conn, &phGraphNode_present, sizeof(uint8_t)) < 0 ||
      (phGraphNode_present &&
       rpc_write(conn, phGraphNode, sizeof(CUgraphNode)) < 0) ||
      rpc_write(conn, &hGraph, sizeof(CUgraph)) < 0 ||
      rpc_write(conn, &numDependencies, sizeof(size_t)) < 0 ||
      rpc_write(conn, &dependencies_null, sizeof(uint8_t)) < 0 ||
      (!dependencies_null && numDependencies * sizeof(const CUgraphNode) != 0 &&
       rpc_write(conn, dependencies,
                 numDependencies * sizeof(const CUgraphNode)) < 0) ||
      rpc_write(conn, &childGraph, sizeof(CUgraph)) < 0 ||
      rpc_wait_for_response(conn) < 0 ||
      (phGraphNode_present &&
       rpc_read(conn, phGraphNode, sizeof(CUgraphNode)) < 0) ||
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
  if (lupine_route_is_local(route)) {
    return_value = lupine_call_real_cuda_fn(
        LUPINE_REAL_CUDA_SYMBOL("cuGraphChildGraphNodeGetGraph"), hNode,
        phGraph);
    if (return_value == CUDA_SUCCESS && phGraph != nullptr) {
      lupine_note_graph_owner_route(*phGraph, route);
    }
    return return_value;
  }
  conn_t *conn = lupine_route_remote_conn(route);
  uint8_t phGraph_present = phGraph != nullptr;
  if (lupine_prepare_rpc(conn) < 0 ||
      rpc_write_start_request_on(conn, RPC_cuGraphChildGraphNodeGetGraph,
                                 RPC_SCOPE_NONE) < 0 ||
      rpc_write(conn, &hNode, sizeof(CUgraphNode)) < 0 ||
      rpc_write(conn, &phGraph_present, sizeof(uint8_t)) < 0 ||
      (phGraph_present && rpc_write(conn, phGraph, sizeof(CUgraph)) < 0) ||
      rpc_wait_for_response(conn) < 0 ||
      (phGraph_present && rpc_read(conn, phGraph, sizeof(CUgraph)) < 0) ||
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
  if (lupine_route_is_local(route)) {
    return_value = lupine_call_real_cuda_fn(
        LUPINE_REAL_CUDA_SYMBOL("cuGraphAddEmptyNode"), phGraphNode, hGraph,
        dependencies, numDependencies);
    if (return_value == CUDA_SUCCESS && phGraphNode != nullptr) {
      lupine_note_graph_node_owner_route(*phGraphNode, route);
    }
    return return_value;
  }
  conn_t *conn = lupine_route_remote_conn(route);
  uint8_t phGraphNode_present = phGraphNode != nullptr;
  uint8_t dependencies_null = dependencies == nullptr ? 1 : 0;
  if (lupine_prepare_rpc(conn) < 0 ||
      rpc_write_start_request_on(conn, RPC_cuGraphAddEmptyNode,
                                 RPC_SCOPE_NONE) < 0 ||
      rpc_write(conn, &phGraphNode_present, sizeof(uint8_t)) < 0 ||
      (phGraphNode_present &&
       rpc_write(conn, phGraphNode, sizeof(CUgraphNode)) < 0) ||
      rpc_write(conn, &hGraph, sizeof(CUgraph)) < 0 ||
      rpc_write(conn, &numDependencies, sizeof(size_t)) < 0 ||
      rpc_write(conn, &dependencies_null, sizeof(uint8_t)) < 0 ||
      (!dependencies_null && numDependencies * sizeof(const CUgraphNode) != 0 &&
       rpc_write(conn, dependencies,
                 numDependencies * sizeof(const CUgraphNode)) < 0) ||
      rpc_wait_for_response(conn) < 0 ||
      (phGraphNode_present &&
       rpc_read(conn, phGraphNode, sizeof(CUgraphNode)) < 0) ||
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
  if (lupine_route_is_local(route)) {
    return_value = lupine_call_real_cuda_fn(
        LUPINE_REAL_CUDA_SYMBOL("cuGraphAddEventRecordNode"), phGraphNode,
        hGraph, dependencies, numDependencies, event);
    if (return_value == CUDA_SUCCESS && phGraphNode != nullptr) {
      lupine_note_graph_node_owner_route(*phGraphNode, route);
    }
    return return_value;
  }
  conn_t *conn = lupine_route_remote_conn(route);
  uint8_t phGraphNode_present = phGraphNode != nullptr;
  uint8_t dependencies_null = dependencies == nullptr ? 1 : 0;
  if (lupine_prepare_rpc(conn) < 0 ||
      rpc_write_start_request_on(conn, RPC_cuGraphAddEventRecordNode,
                                 reinterpret_cast<uintptr_t>(event)) < 0 ||
      rpc_write(conn, &phGraphNode_present, sizeof(uint8_t)) < 0 ||
      (phGraphNode_present &&
       rpc_write(conn, phGraphNode, sizeof(CUgraphNode)) < 0) ||
      rpc_write(conn, &hGraph, sizeof(CUgraph)) < 0 ||
      rpc_write(conn, &numDependencies, sizeof(size_t)) < 0 ||
      rpc_write(conn, &dependencies_null, sizeof(uint8_t)) < 0 ||
      (!dependencies_null && numDependencies * sizeof(const CUgraphNode) != 0 &&
       rpc_write(conn, dependencies,
                 numDependencies * sizeof(const CUgraphNode)) < 0) ||
      rpc_write(conn, &event, sizeof(CUevent)) < 0 ||
      rpc_wait_for_response(conn) < 0 ||
      (phGraphNode_present &&
       rpc_read(conn, phGraphNode, sizeof(CUgraphNode)) < 0) ||
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
  if (lupine_route_is_local(route))
    return lupine_call_real_cuda_fn(
        LUPINE_REAL_CUDA_SYMBOL("cuGraphEventRecordNodeGetEvent"), hNode,
        event_out);
  conn_t *conn = lupine_route_remote_conn(route);
  uint8_t event_out_present = event_out != nullptr;
  if (lupine_prepare_rpc(conn) < 0 ||
      rpc_write_start_request_on(conn, RPC_cuGraphEventRecordNodeGetEvent,
                                 RPC_SCOPE_NONE) < 0 ||
      rpc_write(conn, &hNode, sizeof(CUgraphNode)) < 0 ||
      rpc_write(conn, &event_out_present, sizeof(uint8_t)) < 0 ||
      (event_out_present && rpc_write(conn, event_out, sizeof(CUevent)) < 0) ||
      rpc_wait_for_response(conn) < 0 ||
      (event_out_present && rpc_read(conn, event_out, sizeof(CUevent)) < 0) ||
      rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
      rpc_read_end(conn) < 0)
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  return return_value;
}

CUresult cuGraphEventRecordNodeSetEvent(CUgraphNode hNode, CUevent event) {
  lupine_route route = lupine_route_for_graph_node(hNode);
  CUresult return_value;
  if (lupine_route_is_local(route))
    return lupine_call_real_cuda_fn(
        LUPINE_REAL_CUDA_SYMBOL("cuGraphEventRecordNodeSetEvent"), hNode,
        event);
  conn_t *conn = lupine_route_remote_conn(route);
  if (lupine_prepare_rpc(conn) < 0 ||
      rpc_write_start_request_on(conn, RPC_cuGraphEventRecordNodeSetEvent,
                                 reinterpret_cast<uintptr_t>(event)) < 0 ||
      rpc_write(conn, &hNode, sizeof(CUgraphNode)) < 0 ||
      rpc_write(conn, &event, sizeof(CUevent)) < 0 || rpc_write_end(conn) < 0)
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  return_value = CUDA_SUCCESS;
  return return_value;
}

CUresult cuGraphAddEventWaitNode(CUgraphNode *phGraphNode, CUgraph hGraph,
                                 const CUgraphNode *dependencies,
                                 size_t numDependencies, CUevent event) {
  lupine_route route = lupine_route_for_graph(hGraph);
  CUresult return_value;
  if (lupine_route_is_local(route)) {
    return_value = lupine_call_real_cuda_fn(
        LUPINE_REAL_CUDA_SYMBOL("cuGraphAddEventWaitNode"), phGraphNode, hGraph,
        dependencies, numDependencies, event);
    if (return_value == CUDA_SUCCESS && phGraphNode != nullptr) {
      lupine_note_graph_node_owner_route(*phGraphNode, route);
    }
    return return_value;
  }
  conn_t *conn = lupine_route_remote_conn(route);
  uint8_t phGraphNode_present = phGraphNode != nullptr;
  uint8_t dependencies_null = dependencies == nullptr ? 1 : 0;
  if (lupine_prepare_rpc(conn) < 0 ||
      rpc_write_start_request_on(conn, RPC_cuGraphAddEventWaitNode,
                                 reinterpret_cast<uintptr_t>(event)) < 0 ||
      rpc_write(conn, &phGraphNode_present, sizeof(uint8_t)) < 0 ||
      (phGraphNode_present &&
       rpc_write(conn, phGraphNode, sizeof(CUgraphNode)) < 0) ||
      rpc_write(conn, &hGraph, sizeof(CUgraph)) < 0 ||
      rpc_write(conn, &numDependencies, sizeof(size_t)) < 0 ||
      rpc_write(conn, &dependencies_null, sizeof(uint8_t)) < 0 ||
      (!dependencies_null && numDependencies * sizeof(const CUgraphNode) != 0 &&
       rpc_write(conn, dependencies,
                 numDependencies * sizeof(const CUgraphNode)) < 0) ||
      rpc_write(conn, &event, sizeof(CUevent)) < 0 ||
      rpc_wait_for_response(conn) < 0 ||
      (phGraphNode_present &&
       rpc_read(conn, phGraphNode, sizeof(CUgraphNode)) < 0) ||
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
  if (lupine_route_is_local(route))
    return lupine_call_real_cuda_fn(
        LUPINE_REAL_CUDA_SYMBOL("cuGraphEventWaitNodeGetEvent"), hNode,
        event_out);
  conn_t *conn = lupine_route_remote_conn(route);
  uint8_t event_out_present = event_out != nullptr;
  if (lupine_prepare_rpc(conn) < 0 ||
      rpc_write_start_request_on(conn, RPC_cuGraphEventWaitNodeGetEvent,
                                 RPC_SCOPE_NONE) < 0 ||
      rpc_write(conn, &hNode, sizeof(CUgraphNode)) < 0 ||
      rpc_write(conn, &event_out_present, sizeof(uint8_t)) < 0 ||
      (event_out_present && rpc_write(conn, event_out, sizeof(CUevent)) < 0) ||
      rpc_wait_for_response(conn) < 0 ||
      (event_out_present && rpc_read(conn, event_out, sizeof(CUevent)) < 0) ||
      rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
      rpc_read_end(conn) < 0)
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  return return_value;
}

CUresult cuGraphEventWaitNodeSetEvent(CUgraphNode hNode, CUevent event) {
  lupine_route route = lupine_route_for_graph_node(hNode);
  CUresult return_value;
  if (lupine_route_is_local(route))
    return lupine_call_real_cuda_fn(
        LUPINE_REAL_CUDA_SYMBOL("cuGraphEventWaitNodeSetEvent"), hNode, event);
  conn_t *conn = lupine_route_remote_conn(route);
  if (lupine_prepare_rpc(conn) < 0 ||
      rpc_write_start_request_on(conn, RPC_cuGraphEventWaitNodeSetEvent,
                                 reinterpret_cast<uintptr_t>(event)) < 0 ||
      rpc_write(conn, &hNode, sizeof(CUgraphNode)) < 0 ||
      rpc_write(conn, &event, sizeof(CUevent)) < 0 || rpc_write_end(conn) < 0)
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  return_value = CUDA_SUCCESS;
  return return_value;
}

CUresult cuGraphAddExternalSemaphoresSignalNode(
    CUgraphNode *phGraphNode, CUgraph hGraph, const CUgraphNode *dependencies,
    size_t numDependencies, const CUDA_EXT_SEM_SIGNAL_NODE_PARAMS *nodeParams) {
  lupine_route route = lupine_route_for_graph(hGraph);
  CUresult return_value;
  if (lupine_route_is_local(route)) {
    return_value = lupine_call_real_cuda_fn(
        LUPINE_REAL_CUDA_SYMBOL("cuGraphAddExternalSemaphoresSignalNode"),
        phGraphNode, hGraph, dependencies, numDependencies, nodeParams);
    if (return_value == CUDA_SUCCESS && phGraphNode != nullptr) {
      lupine_note_graph_node_owner_route(*phGraphNode, route);
    }
    return return_value;
  }
  conn_t *conn = lupine_route_remote_conn(route);
  uint8_t phGraphNode_present = phGraphNode != nullptr;
  uint8_t dependencies_null = dependencies == nullptr ? 1 : 0;
  if (nodeParams == nullptr)
    return CUDA_ERROR_INVALID_VALUE;
  if (nodeParams->numExtSems > SIZE_MAX / sizeof(*nodeParams->extSemArray) ||
      (nodeParams->numExtSems != 0 && nodeParams->extSemArray == nullptr))
    return CUDA_ERROR_INVALID_VALUE;
  if (nodeParams->numExtSems > SIZE_MAX / sizeof(*nodeParams->paramsArray) ||
      (nodeParams->numExtSems != 0 && nodeParams->paramsArray == nullptr))
    return CUDA_ERROR_INVALID_VALUE;
  if (lupine_prepare_rpc(conn) < 0 ||
      rpc_write_start_request_on(conn,
                                 RPC_cuGraphAddExternalSemaphoresSignalNode,
                                 RPC_SCOPE_NONE) < 0 ||
      rpc_write(conn, &phGraphNode_present, sizeof(uint8_t)) < 0 ||
      (phGraphNode_present &&
       rpc_write(conn, phGraphNode, sizeof(CUgraphNode)) < 0) ||
      rpc_write(conn, &hGraph, sizeof(CUgraph)) < 0 ||
      rpc_write(conn, &numDependencies, sizeof(size_t)) < 0 ||
      rpc_write(conn, &dependencies_null, sizeof(uint8_t)) < 0 ||
      (!dependencies_null && numDependencies * sizeof(const CUgraphNode) != 0 &&
       rpc_write(conn, dependencies,
                 numDependencies * sizeof(const CUgraphNode)) < 0) ||
      rpc_write(conn, nodeParams, sizeof(*nodeParams)) < 0 ||
      rpc_write(conn, nodeParams->extSemArray,
                nodeParams->numExtSems * sizeof(*nodeParams->extSemArray)) <
          0 ||
      rpc_write(conn, nodeParams->paramsArray,
                nodeParams->numExtSems * sizeof(*nodeParams->paramsArray)) <
          0 ||
      rpc_wait_for_response(conn) < 0 ||
      (phGraphNode_present &&
       rpc_read(conn, phGraphNode, sizeof(CUgraphNode)) < 0) ||
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
  if (lupine_route_is_local(route))
    return lupine_call_real_cuda_fn(
        LUPINE_REAL_CUDA_SYMBOL("cuGraphExternalSemaphoresSignalNodeGetParams"),
        hNode, params_out);
  conn_t *conn = lupine_route_remote_conn(route);
  if (params_out == nullptr)
    return CUDA_ERROR_INVALID_VALUE;
  if (lupine_prepare_rpc(conn) < 0 ||
      rpc_write_start_request_on(
          conn, RPC_cuGraphExternalSemaphoresSignalNodeGetParams,
          RPC_SCOPE_NONE) < 0 ||
      rpc_write(conn, &hNode, sizeof(CUgraphNode)) < 0 ||
      rpc_wait_for_response(conn) < 0 ||
      rpc_read(conn, params_out, sizeof(*params_out)) < 0 ||
      ((params_out->extSemArray =
            (params_out->numExtSems != 0
                 ? (decltype(params_out->extSemArray))
                       lupine_deep_node_cache_get(
                           hNode, 0,
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
                 ? (decltype(params_out->paramsArray))
                       lupine_deep_node_cache_get(
                           hNode, 1,
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
  if (lupine_route_is_local(route))
    return lupine_call_real_cuda_fn(
        LUPINE_REAL_CUDA_SYMBOL("cuGraphExternalSemaphoresSignalNodeSetParams"),
        hNode, nodeParams);
  conn_t *conn = lupine_route_remote_conn(route);
  if (nodeParams == nullptr)
    return CUDA_ERROR_INVALID_VALUE;
  if (nodeParams->numExtSems > SIZE_MAX / sizeof(*nodeParams->extSemArray) ||
      (nodeParams->numExtSems != 0 && nodeParams->extSemArray == nullptr))
    return CUDA_ERROR_INVALID_VALUE;
  if (nodeParams->numExtSems > SIZE_MAX / sizeof(*nodeParams->paramsArray) ||
      (nodeParams->numExtSems != 0 && nodeParams->paramsArray == nullptr))
    return CUDA_ERROR_INVALID_VALUE;
  if (lupine_prepare_rpc(conn) < 0 ||
      rpc_write_start_request_on(
          conn, RPC_cuGraphExternalSemaphoresSignalNodeSetParams,
          RPC_SCOPE_NONE) < 0 ||
      rpc_write(conn, &hNode, sizeof(CUgraphNode)) < 0 ||
      rpc_write(conn, nodeParams, sizeof(*nodeParams)) < 0 ||
      rpc_write(conn, nodeParams->extSemArray,
                nodeParams->numExtSems * sizeof(*nodeParams->extSemArray)) <
          0 ||
      rpc_write(conn, nodeParams->paramsArray,
                nodeParams->numExtSems * sizeof(*nodeParams->paramsArray)) <
          0 ||
      rpc_write_end(conn) < 0)
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  return_value = CUDA_SUCCESS;
  return return_value;
}

CUresult cuGraphAddExternalSemaphoresWaitNode(
    CUgraphNode *phGraphNode, CUgraph hGraph, const CUgraphNode *dependencies,
    size_t numDependencies, const CUDA_EXT_SEM_WAIT_NODE_PARAMS *nodeParams) {
  lupine_route route = lupine_route_for_graph(hGraph);
  CUresult return_value;
  if (lupine_route_is_local(route)) {
    return_value = lupine_call_real_cuda_fn(
        LUPINE_REAL_CUDA_SYMBOL("cuGraphAddExternalSemaphoresWaitNode"),
        phGraphNode, hGraph, dependencies, numDependencies, nodeParams);
    if (return_value == CUDA_SUCCESS && phGraphNode != nullptr) {
      lupine_note_graph_node_owner_route(*phGraphNode, route);
    }
    return return_value;
  }
  conn_t *conn = lupine_route_remote_conn(route);
  uint8_t phGraphNode_present = phGraphNode != nullptr;
  uint8_t dependencies_null = dependencies == nullptr ? 1 : 0;
  if (nodeParams == nullptr)
    return CUDA_ERROR_INVALID_VALUE;
  if (nodeParams->numExtSems > SIZE_MAX / sizeof(*nodeParams->extSemArray) ||
      (nodeParams->numExtSems != 0 && nodeParams->extSemArray == nullptr))
    return CUDA_ERROR_INVALID_VALUE;
  if (nodeParams->numExtSems > SIZE_MAX / sizeof(*nodeParams->paramsArray) ||
      (nodeParams->numExtSems != 0 && nodeParams->paramsArray == nullptr))
    return CUDA_ERROR_INVALID_VALUE;
  if (lupine_prepare_rpc(conn) < 0 ||
      rpc_write_start_request_on(conn, RPC_cuGraphAddExternalSemaphoresWaitNode,
                                 RPC_SCOPE_NONE) < 0 ||
      rpc_write(conn, &phGraphNode_present, sizeof(uint8_t)) < 0 ||
      (phGraphNode_present &&
       rpc_write(conn, phGraphNode, sizeof(CUgraphNode)) < 0) ||
      rpc_write(conn, &hGraph, sizeof(CUgraph)) < 0 ||
      rpc_write(conn, &numDependencies, sizeof(size_t)) < 0 ||
      rpc_write(conn, &dependencies_null, sizeof(uint8_t)) < 0 ||
      (!dependencies_null && numDependencies * sizeof(const CUgraphNode) != 0 &&
       rpc_write(conn, dependencies,
                 numDependencies * sizeof(const CUgraphNode)) < 0) ||
      rpc_write(conn, nodeParams, sizeof(*nodeParams)) < 0 ||
      rpc_write(conn, nodeParams->extSemArray,
                nodeParams->numExtSems * sizeof(*nodeParams->extSemArray)) <
          0 ||
      rpc_write(conn, nodeParams->paramsArray,
                nodeParams->numExtSems * sizeof(*nodeParams->paramsArray)) <
          0 ||
      rpc_wait_for_response(conn) < 0 ||
      (phGraphNode_present &&
       rpc_read(conn, phGraphNode, sizeof(CUgraphNode)) < 0) ||
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
  if (lupine_route_is_local(route))
    return lupine_call_real_cuda_fn(
        LUPINE_REAL_CUDA_SYMBOL("cuGraphExternalSemaphoresWaitNodeGetParams"),
        hNode, params_out);
  conn_t *conn = lupine_route_remote_conn(route);
  if (params_out == nullptr)
    return CUDA_ERROR_INVALID_VALUE;
  if (lupine_prepare_rpc(conn) < 0 ||
      rpc_write_start_request_on(conn,
                                 RPC_cuGraphExternalSemaphoresWaitNodeGetParams,
                                 RPC_SCOPE_NONE) < 0 ||
      rpc_write(conn, &hNode, sizeof(CUgraphNode)) < 0 ||
      rpc_wait_for_response(conn) < 0 ||
      rpc_read(conn, params_out, sizeof(*params_out)) < 0 ||
      ((params_out->extSemArray =
            (params_out->numExtSems != 0
                 ? (decltype(params_out->extSemArray))
                       lupine_deep_node_cache_get(
                           hNode, 0,
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
                 ? (decltype(params_out->paramsArray))
                       lupine_deep_node_cache_get(
                           hNode, 1,
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
  if (lupine_route_is_local(route))
    return lupine_call_real_cuda_fn(
        LUPINE_REAL_CUDA_SYMBOL("cuGraphExternalSemaphoresWaitNodeSetParams"),
        hNode, nodeParams);
  conn_t *conn = lupine_route_remote_conn(route);
  if (nodeParams == nullptr)
    return CUDA_ERROR_INVALID_VALUE;
  if (nodeParams->numExtSems > SIZE_MAX / sizeof(*nodeParams->extSemArray) ||
      (nodeParams->numExtSems != 0 && nodeParams->extSemArray == nullptr))
    return CUDA_ERROR_INVALID_VALUE;
  if (nodeParams->numExtSems > SIZE_MAX / sizeof(*nodeParams->paramsArray) ||
      (nodeParams->numExtSems != 0 && nodeParams->paramsArray == nullptr))
    return CUDA_ERROR_INVALID_VALUE;
  if (lupine_prepare_rpc(conn) < 0 ||
      rpc_write_start_request_on(conn,
                                 RPC_cuGraphExternalSemaphoresWaitNodeSetParams,
                                 RPC_SCOPE_NONE) < 0 ||
      rpc_write(conn, &hNode, sizeof(CUgraphNode)) < 0 ||
      rpc_write(conn, nodeParams, sizeof(*nodeParams)) < 0 ||
      rpc_write(conn, nodeParams->extSemArray,
                nodeParams->numExtSems * sizeof(*nodeParams->extSemArray)) <
          0 ||
      rpc_write(conn, nodeParams->paramsArray,
                nodeParams->numExtSems * sizeof(*nodeParams->paramsArray)) <
          0 ||
      rpc_write_end(conn) < 0)
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  return_value = CUDA_SUCCESS;
  return return_value;
}

CUresult cuGraphAddBatchMemOpNode(
    CUgraphNode *phGraphNode, CUgraph hGraph, const CUgraphNode *dependencies,
    size_t numDependencies, const CUDA_BATCH_MEM_OP_NODE_PARAMS *nodeParams) {
  lupine_route route = lupine_route_for_graph(hGraph);
  CUresult return_value;
  if (lupine_route_is_local(route)) {
    return_value = lupine_call_real_cuda_fn(
        LUPINE_REAL_CUDA_SYMBOL("cuGraphAddBatchMemOpNode"), phGraphNode,
        hGraph, dependencies, numDependencies, nodeParams);
    if (return_value == CUDA_SUCCESS && phGraphNode != nullptr) {
      lupine_note_graph_node_owner_route(*phGraphNode, route);
    }
    return return_value;
  }
  conn_t *conn = lupine_route_remote_conn(route);
  uint8_t phGraphNode_present = phGraphNode != nullptr;
  uint8_t dependencies_null = dependencies == nullptr ? 1 : 0;
  if (nodeParams == nullptr)
    return CUDA_ERROR_INVALID_VALUE;
  if (nodeParams->count > SIZE_MAX / sizeof(*nodeParams->paramArray) ||
      (nodeParams->count != 0 && nodeParams->paramArray == nullptr))
    return CUDA_ERROR_INVALID_VALUE;
  if (lupine_prepare_rpc(conn) < 0 ||
      rpc_write_start_request_on(conn, RPC_cuGraphAddBatchMemOpNode,
                                 RPC_SCOPE_NONE) < 0 ||
      rpc_write(conn, &phGraphNode_present, sizeof(uint8_t)) < 0 ||
      (phGraphNode_present &&
       rpc_write(conn, phGraphNode, sizeof(CUgraphNode)) < 0) ||
      rpc_write(conn, &hGraph, sizeof(CUgraph)) < 0 ||
      rpc_write(conn, &numDependencies, sizeof(size_t)) < 0 ||
      rpc_write(conn, &dependencies_null, sizeof(uint8_t)) < 0 ||
      (!dependencies_null && numDependencies * sizeof(const CUgraphNode) != 0 &&
       rpc_write(conn, dependencies,
                 numDependencies * sizeof(const CUgraphNode)) < 0) ||
      rpc_write(conn, nodeParams, sizeof(*nodeParams)) < 0 ||
      rpc_write(conn, nodeParams->paramArray,
                nodeParams->count * sizeof(*nodeParams->paramArray)) < 0 ||
      rpc_wait_for_response(conn) < 0 ||
      (phGraphNode_present &&
       rpc_read(conn, phGraphNode, sizeof(CUgraphNode)) < 0) ||
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
  if (lupine_route_is_local(route))
    return lupine_call_real_cuda_fn(
        LUPINE_REAL_CUDA_SYMBOL("cuGraphBatchMemOpNodeGetParams"), hNode,
        nodeParams_out);
  conn_t *conn = lupine_route_remote_conn(route);
  if (nodeParams_out == nullptr)
    return CUDA_ERROR_INVALID_VALUE;
  if (lupine_prepare_rpc(conn) < 0 ||
      rpc_write_start_request_on(conn, RPC_cuGraphBatchMemOpNodeGetParams,
                                 RPC_SCOPE_NONE) < 0 ||
      rpc_write(conn, &hNode, sizeof(CUgraphNode)) < 0 ||
      rpc_wait_for_response(conn) < 0 ||
      rpc_read(conn, nodeParams_out, sizeof(*nodeParams_out)) < 0 ||
      ((nodeParams_out->paramArray =
            (nodeParams_out->count != 0
                 ? (decltype(nodeParams_out->paramArray))
                       lupine_deep_node_cache_get(
                           hNode, 0,
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
  if (lupine_route_is_local(route))
    return lupine_call_real_cuda_fn(
        LUPINE_REAL_CUDA_SYMBOL("cuGraphBatchMemOpNodeSetParams"), hNode,
        nodeParams);
  conn_t *conn = lupine_route_remote_conn(route);
  if (nodeParams == nullptr)
    return CUDA_ERROR_INVALID_VALUE;
  if (nodeParams->count > SIZE_MAX / sizeof(*nodeParams->paramArray) ||
      (nodeParams->count != 0 && nodeParams->paramArray == nullptr))
    return CUDA_ERROR_INVALID_VALUE;
  if (lupine_prepare_rpc(conn) < 0 ||
      rpc_write_start_request_on(conn, RPC_cuGraphBatchMemOpNodeSetParams,
                                 RPC_SCOPE_NONE) < 0 ||
      rpc_write(conn, &hNode, sizeof(CUgraphNode)) < 0 ||
      rpc_write(conn, nodeParams, sizeof(*nodeParams)) < 0 ||
      rpc_write(conn, nodeParams->paramArray,
                nodeParams->count * sizeof(*nodeParams->paramArray)) < 0 ||
      rpc_write_end(conn) < 0)
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  return_value = CUDA_SUCCESS;
  return return_value;
}

CUresult cuGraphExecBatchMemOpNodeSetParams(
    CUgraphExec hGraphExec, CUgraphNode hNode,
    const CUDA_BATCH_MEM_OP_NODE_PARAMS *nodeParams) {
  lupine_route route = lupine_route_for_graph_exec(hGraphExec);
  CUresult return_value;
  if (lupine_route_is_local(route))
    return lupine_call_real_cuda_fn(
        LUPINE_REAL_CUDA_SYMBOL("cuGraphExecBatchMemOpNodeSetParams"),
        hGraphExec, hNode, nodeParams);
  conn_t *conn = lupine_route_remote_conn(route);
  if (nodeParams == nullptr)
    return CUDA_ERROR_INVALID_VALUE;
  if (nodeParams->count > SIZE_MAX / sizeof(*nodeParams->paramArray) ||
      (nodeParams->count != 0 && nodeParams->paramArray == nullptr))
    return CUDA_ERROR_INVALID_VALUE;
  if (lupine_prepare_rpc(conn) < 0 ||
      rpc_write_start_request_on(conn, RPC_cuGraphExecBatchMemOpNodeSetParams,
                                 reinterpret_cast<uintptr_t>(hGraphExec)) < 0 ||
      rpc_write(conn, &hGraphExec, sizeof(CUgraphExec)) < 0 ||
      rpc_write(conn, &hNode, sizeof(CUgraphNode)) < 0 ||
      rpc_write(conn, nodeParams, sizeof(*nodeParams)) < 0 ||
      rpc_write(conn, nodeParams->paramArray,
                nodeParams->count * sizeof(*nodeParams->paramArray)) < 0 ||
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
  if (nodeParams == nullptr)
    return CUDA_ERROR_INVALID_VALUE;
  CUDA_MEM_ALLOC_NODE_PARAMS *original_params = nodeParams;
  CUDA_MEM_ALLOC_NODE_PARAMS route_params = *nodeParams;
  CUresult status =
      lupine_translate_mem_location(route, route_params.poolProps.location);
  if (status != CUDA_SUCCESS)
    return status;
  if (route_params.accessDescCount >
          SIZE_MAX / sizeof(*route_params.accessDescs) ||
      (route_params.accessDescCount != 0 &&
       route_params.accessDescs == nullptr))
    return CUDA_ERROR_INVALID_VALUE;
  std::vector<CUmemAccessDesc> route_descriptors;
  if (route_params.accessDescCount != 0)
    route_descriptors.assign(route_params.accessDescs,
                             route_params.accessDescs +
                                 route_params.accessDescCount);
  for (auto &desc : route_descriptors) {
    status = lupine_translate_mem_location(route, desc.location);
    if (status != CUDA_SUCCESS)
      return status;
  }
  route_params.accessDescs = route_descriptors.data();
  nodeParams = &route_params;
  CUresult return_value;
  if (lupine_route_is_local(route)) {
    return_value = lupine_call_real_cuda_fn(
        LUPINE_REAL_CUDA_SYMBOL("cuGraphAddMemAllocNode"), phGraphNode, hGraph,
        dependencies, numDependencies, nodeParams);
    if (return_value == CUDA_SUCCESS && phGraphNode != nullptr) {
      lupine_note_graph_node_owner_route(*phGraphNode, route);
    }
    if (return_value == CUDA_SUCCESS)
      original_params->dptr = route_params.dptr;
    return return_value;
  }
  conn_t *conn = lupine_route_remote_conn(route);
  uint8_t phGraphNode_present = phGraphNode != nullptr;
  uint8_t dependencies_null = dependencies == nullptr ? 1 : 0;
  if (nodeParams == nullptr)
    return CUDA_ERROR_INVALID_VALUE;
  if (nodeParams->accessDescCount >
          SIZE_MAX / sizeof(*nodeParams->accessDescs) ||
      (nodeParams->accessDescCount != 0 && nodeParams->accessDescs == nullptr))
    return CUDA_ERROR_INVALID_VALUE;
  auto nodeParams_accessDescs_input = nodeParams->accessDescs;
  if (lupine_prepare_rpc(conn) < 0 ||
      rpc_write_start_request_on(conn, RPC_cuGraphAddMemAllocNode,
                                 RPC_SCOPE_NONE) < 0 ||
      rpc_write(conn, &phGraphNode_present, sizeof(uint8_t)) < 0 ||
      (phGraphNode_present &&
       rpc_write(conn, phGraphNode, sizeof(CUgraphNode)) < 0) ||
      rpc_write(conn, &hGraph, sizeof(CUgraph)) < 0 ||
      rpc_write(conn, &numDependencies, sizeof(size_t)) < 0 ||
      rpc_write(conn, &dependencies_null, sizeof(uint8_t)) < 0 ||
      (!dependencies_null && numDependencies * sizeof(const CUgraphNode) != 0 &&
       rpc_write(conn, dependencies,
                 numDependencies * sizeof(const CUgraphNode)) < 0) ||
      rpc_write(conn, nodeParams, sizeof(*nodeParams)) < 0 ||
      rpc_write(conn, nodeParams->accessDescs,
                nodeParams->accessDescCount *
                    sizeof(*nodeParams->accessDescs)) < 0 ||
      rpc_wait_for_response(conn) < 0 ||
      (phGraphNode_present &&
       rpc_read(conn, phGraphNode, sizeof(CUgraphNode)) < 0) ||
      rpc_read(conn, nodeParams, sizeof(*nodeParams)) < 0 ||
      ((nodeParams->accessDescs = nodeParams_accessDescs_input), false) ||
      rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
      rpc_read_end(conn) < 0)
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  if (return_value == CUDA_SUCCESS && phGraphNode != nullptr) {
    lupine_note_graph_node_owner_route(*phGraphNode, route);
  }
  if (return_value == CUDA_SUCCESS)
    original_params->dptr = route_params.dptr;
  return return_value;
}

CUresult cuGraphMemAllocNodeGetParams(CUgraphNode hNode,
                                      CUDA_MEM_ALLOC_NODE_PARAMS *params_out) {
  lupine_route route = lupine_route_for_graph_node(hNode);
  CUresult return_value;
  if (lupine_route_is_local(route)) {
    return_value = lupine_call_real_cuda_fn(
        LUPINE_REAL_CUDA_SYMBOL("cuGraphMemAllocNodeGetParams"), hNode,
        params_out);
    if (return_value == CUDA_SUCCESS) {
      lupine_restore_mem_location(route, params_out->poolProps.location);
      // Remote descriptors already live in the node cache. Copy native output
      // before translating it so the driver's immutable array is not changed.
      if (lupine_route_is_local(route) && params_out->accessDescCount != 0) {
        size_t bytes =
            params_out->accessDescCount * sizeof(*params_out->accessDescs);
        void *descriptors = lupine_deep_node_cache_get(hNode, 0, bytes);
        if (descriptors == nullptr)
          return CUDA_ERROR_OUT_OF_MEMORY;
        std::memcpy(descriptors, params_out->accessDescs, bytes);
        params_out->accessDescs = static_cast<CUmemAccessDesc *>(descriptors);
      }
      for (size_t i = 0; i < params_out->accessDescCount; ++i)
        lupine_restore_mem_location(
            route,
            const_cast<CUmemAccessDesc &>(params_out->accessDescs[i]).location);
    }
    return return_value;
  }
  conn_t *conn = lupine_route_remote_conn(route);
  if (params_out == nullptr)
    return CUDA_ERROR_INVALID_VALUE;
  if (lupine_prepare_rpc(conn) < 0 ||
      rpc_write_start_request_on(conn, RPC_cuGraphMemAllocNodeGetParams,
                                 RPC_SCOPE_NONE) < 0 ||
      rpc_write(conn, &hNode, sizeof(CUgraphNode)) < 0 ||
      rpc_wait_for_response(conn) < 0 ||
      rpc_read(conn, params_out, sizeof(*params_out)) < 0 ||
      ((params_out->accessDescs =
            (params_out->accessDescCount != 0
                 ? (decltype(params_out->accessDescs))
                       lupine_deep_node_cache_get(
                           hNode, 0,
                           params_out->accessDescCount *
                               sizeof(*params_out->accessDescs))
                 : nullptr)),
       false) ||
      (params_out->accessDescCount != 0 &&
       params_out->accessDescs == nullptr) ||
      (params_out->accessDescCount != 0 &&
       rpc_read(conn, (void *)params_out->accessDescs,
                params_out->accessDescCount *
                    sizeof(*params_out->accessDescs)) < 0) ||
      rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
      rpc_read_end(conn) < 0)
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  if (return_value == CUDA_SUCCESS) {
    lupine_restore_mem_location(route, params_out->poolProps.location);
    // Remote descriptors already live in the node cache. Copy native output
    // before translating it so the driver's immutable array is not changed.
    if (lupine_route_is_local(route) && params_out->accessDescCount != 0) {
      size_t bytes =
          params_out->accessDescCount * sizeof(*params_out->accessDescs);
      void *descriptors = lupine_deep_node_cache_get(hNode, 0, bytes);
      if (descriptors == nullptr)
        return CUDA_ERROR_OUT_OF_MEMORY;
      std::memcpy(descriptors, params_out->accessDescs, bytes);
      params_out->accessDescs = static_cast<CUmemAccessDesc *>(descriptors);
    }
    for (size_t i = 0; i < params_out->accessDescCount; ++i)
      lupine_restore_mem_location(
          route,
          const_cast<CUmemAccessDesc &>(params_out->accessDescs[i]).location);
  }
  return return_value;
}

CUresult cuGraphAddMemFreeNode(CUgraphNode *phGraphNode, CUgraph hGraph,
                               const CUgraphNode *dependencies,
                               size_t numDependencies, CUdeviceptr dptr) {
  lupine_route route = lupine_route_for_graph(hGraph);
  CUresult return_value;
  if (lupine_route_is_local(route)) {
    return_value = lupine_call_real_cuda_fn(
        LUPINE_REAL_CUDA_SYMBOL("cuGraphAddMemFreeNode"), phGraphNode, hGraph,
        dependencies, numDependencies, dptr);
    if (return_value == CUDA_SUCCESS && phGraphNode != nullptr) {
      lupine_note_graph_node_owner_route(*phGraphNode, route);
    }
    return return_value;
  }
  conn_t *conn = lupine_route_remote_conn(route);
  uint8_t phGraphNode_present = phGraphNode != nullptr;
  uint8_t dependencies_null = dependencies == nullptr ? 1 : 0;
  if (lupine_prepare_rpc(conn) < 0 ||
      rpc_write_start_request_on(conn, RPC_cuGraphAddMemFreeNode,
                                 RPC_SCOPE_NONE) < 0 ||
      rpc_write(conn, &phGraphNode_present, sizeof(uint8_t)) < 0 ||
      (phGraphNode_present &&
       rpc_write(conn, phGraphNode, sizeof(CUgraphNode)) < 0) ||
      rpc_write(conn, &hGraph, sizeof(CUgraph)) < 0 ||
      rpc_write(conn, &numDependencies, sizeof(size_t)) < 0 ||
      rpc_write(conn, &dependencies_null, sizeof(uint8_t)) < 0 ||
      (!dependencies_null && numDependencies * sizeof(const CUgraphNode) != 0 &&
       rpc_write(conn, dependencies,
                 numDependencies * sizeof(const CUgraphNode)) < 0) ||
      rpc_write(conn, &dptr, sizeof(CUdeviceptr)) < 0 ||
      rpc_wait_for_response(conn) < 0 ||
      (phGraphNode_present &&
       rpc_read(conn, phGraphNode, sizeof(CUgraphNode)) < 0) ||
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
  if (lupine_route_is_local(route))
    return lupine_call_real_cuda_fn(
        LUPINE_REAL_CUDA_SYMBOL("cuGraphMemFreeNodeGetParams"), hNode,
        dptr_out);
  conn_t *conn = lupine_route_remote_conn(route);
  uint8_t dptr_out_present = dptr_out != nullptr;
  if (lupine_prepare_rpc(conn) < 0 ||
      rpc_write_start_request_on(conn, RPC_cuGraphMemFreeNodeGetParams,
                                 RPC_SCOPE_NONE) < 0 ||
      rpc_write(conn, &hNode, sizeof(CUgraphNode)) < 0 ||
      rpc_write(conn, &dptr_out_present, sizeof(uint8_t)) < 0 ||
      (dptr_out_present &&
       rpc_write(conn, dptr_out, sizeof(CUdeviceptr)) < 0) ||
      rpc_wait_for_response(conn) < 0 ||
      (dptr_out_present && rpc_read(conn, dptr_out, sizeof(CUdeviceptr)) < 0) ||
      rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
      rpc_read_end(conn) < 0)
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  return return_value;
}

CUresult cuDeviceGraphMemTrim(CUdevice device) {
  lupine_route route = lupine_route_for_device(&device);
  if (route.kind == LUPINE_ROUTE_UNKNOWN_DEVICE)
    return CUDA_ERROR_INVALID_DEVICE;
  CUresult return_value;
  if (lupine_route_is_local(route))
    return lupine_call_real_cuda_fn(
        LUPINE_REAL_CUDA_SYMBOL("cuDeviceGraphMemTrim"), device);
  conn_t *conn = lupine_route_remote_conn(route);
  if (lupine_prepare_rpc(conn) < 0 ||
      rpc_write_start_request_on(conn, RPC_cuDeviceGraphMemTrim,
                                 RPC_SCOPE_NONE) < 0 ||
      rpc_write(conn, &device, sizeof(CUdevice)) < 0 || rpc_write_end(conn) < 0)
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  return_value = CUDA_SUCCESS;
  return return_value;
}

CUresult cuDeviceGetGraphMemAttribute(CUdevice device,
                                      CUgraphMem_attribute attr, void *value) {
  lupine_route route = lupine_route_for_device(&device);
  if (route.kind == LUPINE_ROUTE_UNKNOWN_DEVICE)
    return CUDA_ERROR_INVALID_DEVICE;
  CUresult return_value;
  if (lupine_route_is_local(route))
    return lupine_call_real_cuda_fn(
        LUPINE_REAL_CUDA_SYMBOL("cuDeviceGetGraphMemAttribute"), device, attr,
        value);
  conn_t *conn = lupine_route_remote_conn(route);
  uint8_t value_null = value == nullptr ? 1 : 0;
  if (lupine_prepare_rpc(conn) < 0 ||
      rpc_write_start_request_on(conn, RPC_cuDeviceGetGraphMemAttribute,
                                 RPC_SCOPE_NONE) < 0 ||
      rpc_write(conn, &device, sizeof(CUdevice)) < 0 ||
      rpc_write(conn, &attr, sizeof(CUgraphMem_attribute)) < 0 ||
      rpc_write(conn, &value_null, sizeof(uint8_t)) < 0 ||
      rpc_wait_for_response(conn) < 0 ||
      (!value_null && 8 != 0 && rpc_read(conn, value, 8) < 0) ||
      rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
      rpc_read_end(conn) < 0)
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  return return_value;
}

CUresult cuDeviceSetGraphMemAttribute(CUdevice device,
                                      CUgraphMem_attribute attr, void *value) {
  lupine_route route = lupine_route_for_device(&device);
  if (route.kind == LUPINE_ROUTE_UNKNOWN_DEVICE)
    return CUDA_ERROR_INVALID_DEVICE;
  CUresult return_value;
  if (lupine_route_is_local(route))
    return lupine_call_real_cuda_fn(
        LUPINE_REAL_CUDA_SYMBOL("cuDeviceSetGraphMemAttribute"), device, attr,
        value);
  conn_t *conn = lupine_route_remote_conn(route);
  uint8_t value_null = value == nullptr ? 1 : 0;
  if (lupine_prepare_rpc(conn) < 0 ||
      rpc_write_start_request_on(conn, RPC_cuDeviceSetGraphMemAttribute,
                                 RPC_SCOPE_NONE) < 0 ||
      rpc_write(conn, &device, sizeof(CUdevice)) < 0 ||
      rpc_write(conn, &attr, sizeof(CUgraphMem_attribute)) < 0 ||
      rpc_write(conn, &value_null, sizeof(uint8_t)) < 0 ||
      (!value_null && 8 != 0 && rpc_write(conn, value, 8) < 0) ||
      rpc_wait_for_response(conn) < 0 ||
      rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
      rpc_read_end(conn) < 0)
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  return return_value;
}

CUresult cuGraphClone(CUgraph *phGraphClone, CUgraph originalGraph) {
  lupine_route route = lupine_route_for_graph(originalGraph);
  CUresult return_value;
  if (lupine_route_is_local(route)) {
    return_value = lupine_call_real_cuda_fn(
        LUPINE_REAL_CUDA_SYMBOL("cuGraphClone"), phGraphClone, originalGraph);
    if (return_value == CUDA_SUCCESS && phGraphClone != nullptr) {
      lupine_note_graph_owner_route(*phGraphClone, route);
    }
    return return_value;
  }
  conn_t *conn = lupine_route_remote_conn(route);
  if (phGraphClone == nullptr)
    return CUDA_ERROR_INVALID_VALUE;
  if (lupine_prepare_rpc(conn) < 0 ||
      rpc_write_start_request_on(conn, RPC_cuGraphClone, RPC_SCOPE_NONE) < 0 ||
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
  if (lupine_route_is_local(route)) {
    return_value = lupine_call_real_cuda_fn(
        LUPINE_REAL_CUDA_SYMBOL("cuGraphNodeFindInClone"), phNode,
        hOriginalNode, hClonedGraph);
    if (return_value == CUDA_SUCCESS && phNode != nullptr) {
      lupine_note_graph_node_owner_route(*phNode, route);
    }
    return return_value;
  }
  conn_t *conn = lupine_route_remote_conn(route);
  uint8_t phNode_present = phNode != nullptr;
  if (lupine_prepare_rpc(conn) < 0 ||
      rpc_write_start_request_on(conn, RPC_cuGraphNodeFindInClone,
                                 RPC_SCOPE_NONE) < 0 ||
      rpc_write(conn, &phNode_present, sizeof(uint8_t)) < 0 ||
      (phNode_present && rpc_write(conn, phNode, sizeof(CUgraphNode)) < 0) ||
      rpc_write(conn, &hOriginalNode, sizeof(CUgraphNode)) < 0 ||
      rpc_write(conn, &hClonedGraph, sizeof(CUgraph)) < 0 ||
      rpc_wait_for_response(conn) < 0 ||
      (phNode_present && rpc_read(conn, phNode, sizeof(CUgraphNode)) < 0) ||
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
  if (lupine_route_is_local(route))
    return lupine_call_real_cuda_fn(
        LUPINE_REAL_CUDA_SYMBOL("cuGraphNodeGetType"), hNode, type);
  conn_t *conn = lupine_route_remote_conn(route);
  uint8_t type_present = type != nullptr;
  if (lupine_prepare_rpc(conn) < 0 ||
      rpc_write_start_request_on(conn, RPC_cuGraphNodeGetType, RPC_SCOPE_NONE) <
          0 ||
      rpc_write(conn, &hNode, sizeof(CUgraphNode)) < 0 ||
      rpc_write(conn, &type_present, sizeof(uint8_t)) < 0 ||
      (type_present && rpc_write(conn, type, sizeof(CUgraphNodeType)) < 0) ||
      rpc_wait_for_response(conn) < 0 ||
      (type_present && rpc_read(conn, type, sizeof(CUgraphNodeType)) < 0) ||
      rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
      rpc_read_end(conn) < 0)
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  return return_value;
}

#if CUDA_VERSION >= 13010
CUresult cuGraphNodeGetContainingGraph(CUgraphNode hNode, CUgraph *phGraph) {
  lupine_route route = lupine_route_for_graph_node(hNode);
  CUresult return_value;
  if (lupine_route_is_local(route)) {
    return_value = lupine_call_real_cuda_fn(
        LUPINE_REAL_CUDA_SYMBOL("cuGraphNodeGetContainingGraph"), hNode,
        phGraph);
    if (return_value == CUDA_SUCCESS && phGraph != nullptr) {
      lupine_note_graph_owner_route(*phGraph, route);
    }
    return return_value;
  }
  conn_t *conn = lupine_route_remote_conn(route);
  uint8_t phGraph_present = phGraph != nullptr;
  if (lupine_prepare_rpc(conn) < 0 ||
      rpc_write_start_request_on(conn, RPC_cuGraphNodeGetContainingGraph,
                                 RPC_SCOPE_NONE) < 0 ||
      rpc_write(conn, &hNode, sizeof(CUgraphNode)) < 0 ||
      rpc_write(conn, &phGraph_present, sizeof(uint8_t)) < 0 ||
      (phGraph_present && rpc_write(conn, phGraph, sizeof(CUgraph)) < 0) ||
      rpc_wait_for_response(conn) < 0 ||
      (phGraph_present && rpc_read(conn, phGraph, sizeof(CUgraph)) < 0) ||
      rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
      rpc_read_end(conn) < 0)
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  if (return_value == CUDA_SUCCESS && phGraph != nullptr) {
    lupine_note_graph_owner_route(*phGraph, route);
  }
  return return_value;
}

#endif

#if CUDA_VERSION >= 13010
CUresult cuGraphNodeGetLocalId(CUgraphNode hNode, unsigned int *nodeId) {
  lupine_route route = lupine_route_for_graph_node(hNode);
  CUresult return_value;
  if (lupine_route_is_local(route))
    return lupine_call_real_cuda_fn(
        LUPINE_REAL_CUDA_SYMBOL("cuGraphNodeGetLocalId"), hNode, nodeId);
  conn_t *conn = lupine_route_remote_conn(route);
  uint8_t nodeId_present = nodeId != nullptr;
  if (lupine_prepare_rpc(conn) < 0 ||
      rpc_write_start_request_on(conn, RPC_cuGraphNodeGetLocalId,
                                 RPC_SCOPE_NONE) < 0 ||
      rpc_write(conn, &hNode, sizeof(CUgraphNode)) < 0 ||
      rpc_write(conn, &nodeId_present, sizeof(uint8_t)) < 0 ||
      (nodeId_present && rpc_write(conn, nodeId, sizeof(unsigned int)) < 0) ||
      rpc_wait_for_response(conn) < 0 ||
      (nodeId_present && rpc_read(conn, nodeId, sizeof(unsigned int)) < 0) ||
      rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
      rpc_read_end(conn) < 0)
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  return return_value;
}

#endif

#if CUDA_VERSION >= 13010
CUresult cuGraphNodeGetToolsId(CUgraphNode hNode,
                               unsigned long long *toolsNodeId) {
  lupine_route route = lupine_route_for_graph_node(hNode);
  CUresult return_value;
  if (lupine_route_is_local(route))
    return lupine_call_real_cuda_fn(
        LUPINE_REAL_CUDA_SYMBOL("cuGraphNodeGetToolsId"), hNode, toolsNodeId);
  conn_t *conn = lupine_route_remote_conn(route);
  uint8_t toolsNodeId_present = toolsNodeId != nullptr;
  if (lupine_prepare_rpc(conn) < 0 ||
      rpc_write_start_request_on(conn, RPC_cuGraphNodeGetToolsId,
                                 RPC_SCOPE_NONE) < 0 ||
      rpc_write(conn, &hNode, sizeof(CUgraphNode)) < 0 ||
      rpc_write(conn, &toolsNodeId_present, sizeof(uint8_t)) < 0 ||
      (toolsNodeId_present &&
       rpc_write(conn, toolsNodeId, sizeof(unsigned long long)) < 0) ||
      rpc_wait_for_response(conn) < 0 ||
      (toolsNodeId_present &&
       rpc_read(conn, toolsNodeId, sizeof(unsigned long long)) < 0) ||
      rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
      rpc_read_end(conn) < 0)
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  return return_value;
}

#endif

#if CUDA_VERSION >= 13010
CUresult cuGraphGetId(CUgraph hGraph, unsigned int *graphId) {
  lupine_route route = lupine_route_for_graph(hGraph);
  CUresult return_value;
  if (lupine_route_is_local(route))
    return lupine_call_real_cuda_fn(LUPINE_REAL_CUDA_SYMBOL("cuGraphGetId"),
                                    hGraph, graphId);
  conn_t *conn = lupine_route_remote_conn(route);
  uint8_t graphId_present = graphId != nullptr;
  if (lupine_prepare_rpc(conn) < 0 ||
      rpc_write_start_request_on(conn, RPC_cuGraphGetId, RPC_SCOPE_NONE) < 0 ||
      rpc_write(conn, &hGraph, sizeof(CUgraph)) < 0 ||
      rpc_write(conn, &graphId_present, sizeof(uint8_t)) < 0 ||
      (graphId_present && rpc_write(conn, graphId, sizeof(unsigned int)) < 0) ||
      rpc_wait_for_response(conn) < 0 ||
      (graphId_present && rpc_read(conn, graphId, sizeof(unsigned int)) < 0) ||
      rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
      rpc_read_end(conn) < 0)
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  return return_value;
}

#endif

#if CUDA_VERSION >= 13010
CUresult cuGraphExecGetId(CUgraphExec hGraphExec, unsigned int *graphId) {
  lupine_route route = lupine_route_for_graph_exec(hGraphExec);
  CUresult return_value;
  if (lupine_route_is_local(route))
    return lupine_call_real_cuda_fn(LUPINE_REAL_CUDA_SYMBOL("cuGraphExecGetId"),
                                    hGraphExec, graphId);
  conn_t *conn = lupine_route_remote_conn(route);
  uint8_t graphId_present = graphId != nullptr;
  if (lupine_prepare_rpc(conn) < 0 ||
      rpc_write_start_request_on(conn, RPC_cuGraphExecGetId,
                                 reinterpret_cast<uintptr_t>(hGraphExec)) < 0 ||
      rpc_write(conn, &hGraphExec, sizeof(CUgraphExec)) < 0 ||
      rpc_write(conn, &graphId_present, sizeof(uint8_t)) < 0 ||
      (graphId_present && rpc_write(conn, graphId, sizeof(unsigned int)) < 0) ||
      rpc_wait_for_response(conn) < 0 ||
      (graphId_present && rpc_read(conn, graphId, sizeof(unsigned int)) < 0) ||
      rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
      rpc_read_end(conn) < 0)
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  return return_value;
}

#endif

CUresult cuGraphGetNodes(CUgraph hGraph, CUgraphNode *nodes, size_t *numNodes) {
  lupine_route route = lupine_route_for_graph(hGraph);
  CUresult return_value;
  if (lupine_route_is_local(route))
    return lupine_call_real_cuda_fn(LUPINE_REAL_CUDA_SYMBOL("cuGraphGetNodes"),
                                    hGraph, nodes, numNodes);
  conn_t *conn = lupine_route_remote_conn(route);
  uint8_t numNodes_present = numNodes != nullptr;
  size_t numNodes_requested =
      (numNodes_present && nodes != nullptr) ? *numNodes : 0;
  uint8_t nodes_null = nodes == nullptr ? 1 : 0;
  if (lupine_prepare_rpc(conn) < 0 ||
      rpc_write_start_request_on(conn, RPC_cuGraphGetNodes, RPC_SCOPE_NONE) <
          0 ||
      rpc_write(conn, &hGraph, sizeof(CUgraph)) < 0 ||
      rpc_write(conn, &numNodes_present, sizeof(uint8_t)) < 0 ||
      rpc_write(conn, &numNodes_requested, sizeof(size_t)) < 0 ||
      rpc_write(conn, &nodes_null, sizeof(uint8_t)) < 0 ||
      rpc_wait_for_response(conn) < 0 ||
      (numNodes_present && rpc_read(conn, numNodes, sizeof(size_t)) < 0) ||
      (nodes != nullptr && numNodes_requested != 0 && numNodes_present &&
       *numNodes != 0 &&
       rpc_read(conn, nodes,
                ((numNodes_present ? *numNodes : 0) < numNodes_requested
                     ? (numNodes_present ? *numNodes : 0)
                     : numNodes_requested) *
                    sizeof(CUgraphNode)) < 0) ||
      rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
      rpc_read_end(conn) < 0)
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  return return_value;
}

CUresult cuGraphGetRootNodes(CUgraph hGraph, CUgraphNode *rootNodes,
                             size_t *numRootNodes) {
  lupine_route route = lupine_route_for_graph(hGraph);
  CUresult return_value;
  if (lupine_route_is_local(route))
    return lupine_call_real_cuda_fn(
        LUPINE_REAL_CUDA_SYMBOL("cuGraphGetRootNodes"), hGraph, rootNodes,
        numRootNodes);
  conn_t *conn = lupine_route_remote_conn(route);
  uint8_t numRootNodes_present = numRootNodes != nullptr;
  size_t numRootNodes_requested =
      (numRootNodes_present && rootNodes != nullptr) ? *numRootNodes : 0;
  uint8_t rootNodes_null = rootNodes == nullptr ? 1 : 0;
  if (lupine_prepare_rpc(conn) < 0 ||
      rpc_write_start_request_on(conn, RPC_cuGraphGetRootNodes,
                                 RPC_SCOPE_NONE) < 0 ||
      rpc_write(conn, &hGraph, sizeof(CUgraph)) < 0 ||
      rpc_write(conn, &numRootNodes_present, sizeof(uint8_t)) < 0 ||
      rpc_write(conn, &numRootNodes_requested, sizeof(size_t)) < 0 ||
      rpc_write(conn, &rootNodes_null, sizeof(uint8_t)) < 0 ||
      rpc_wait_for_response(conn) < 0 ||
      (numRootNodes_present &&
       rpc_read(conn, numRootNodes, sizeof(size_t)) < 0) ||
      (rootNodes != nullptr && numRootNodes_requested != 0 &&
       numRootNodes_present && *numRootNodes != 0 &&
       rpc_read(
           conn, rootNodes,
           ((numRootNodes_present ? *numRootNodes : 0) < numRootNodes_requested
                ? (numRootNodes_present ? *numRootNodes : 0)
                : numRootNodes_requested) *
               sizeof(CUgraphNode)) < 0) ||
      rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
      rpc_read_end(conn) < 0)
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  return return_value;
}

CUresult cuGraphGetEdges_v2(CUgraph hGraph, CUgraphNode *from, CUgraphNode *to,
                            CUgraphEdgeData *edgeData, size_t *numEdges) {
  lupine_route route = lupine_route_for_graph(hGraph);
  CUresult return_value;
  if (lupine_route_is_local(route))
    return lupine_call_real_cuda_fn(
        LUPINE_REAL_CUDA_SYMBOL("cuGraphGetEdges_v2"), hGraph, from, to,
        edgeData, numEdges);
  conn_t *conn = lupine_route_remote_conn(route);
  uint8_t numEdges_present = numEdges != nullptr;
  size_t numEdges_requested =
      (numEdges_present && from != nullptr) ? *numEdges : 0;
  uint8_t from_null = from == nullptr ? 1 : 0;
  uint8_t to_null = to == nullptr ? 1 : 0;
  uint8_t edgeData_null = edgeData == nullptr ? 1 : 0;
  if (lupine_prepare_rpc(conn) < 0 ||
      rpc_write_start_request_on(conn, RPC_cuGraphGetEdges_v2, RPC_SCOPE_NONE) <
          0 ||
      rpc_write(conn, &hGraph, sizeof(CUgraph)) < 0 ||
      rpc_write(conn, &numEdges_present, sizeof(uint8_t)) < 0 ||
      rpc_write(conn, &numEdges_requested, sizeof(size_t)) < 0 ||
      rpc_write(conn, &from_null, sizeof(uint8_t)) < 0 ||
      rpc_write(conn, &to_null, sizeof(uint8_t)) < 0 ||
      rpc_write(conn, &edgeData_null, sizeof(uint8_t)) < 0 ||
      rpc_wait_for_response(conn) < 0 ||
      (numEdges_present && rpc_read(conn, numEdges, sizeof(size_t)) < 0) ||
      (from != nullptr && numEdges_requested != 0 && numEdges_present &&
       *numEdges != 0 &&
       rpc_read(conn, from,
                ((numEdges_present ? *numEdges : 0) < numEdges_requested
                     ? (numEdges_present ? *numEdges : 0)
                     : numEdges_requested) *
                    sizeof(CUgraphNode)) < 0) ||
      (to != nullptr && numEdges_requested != 0 && numEdges_present &&
       *numEdges != 0 &&
       rpc_read(conn, to,
                ((numEdges_present ? *numEdges : 0) < numEdges_requested
                     ? (numEdges_present ? *numEdges : 0)
                     : numEdges_requested) *
                    sizeof(CUgraphNode)) < 0) ||
      (edgeData != nullptr && numEdges_requested != 0 && numEdges_present &&
       *numEdges != 0 &&
       rpc_read(conn, edgeData,
                ((numEdges_present ? *numEdges : 0) < numEdges_requested
                     ? (numEdges_present ? *numEdges : 0)
                     : numEdges_requested) *
                    sizeof(CUgraphEdgeData)) < 0) ||
      rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
      rpc_read_end(conn) < 0)
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  return return_value;
}

CUresult cuGraphNodeGetDependencies_v2(CUgraphNode hNode,
                                       CUgraphNode *dependencies,
                                       CUgraphEdgeData *edgeData,
                                       size_t *numDependencies) {
  lupine_route route = lupine_route_for_graph_node(hNode);
  CUresult return_value;
  if (lupine_route_is_local(route))
    return lupine_call_real_cuda_fn(
        LUPINE_REAL_CUDA_SYMBOL("cuGraphNodeGetDependencies_v2"), hNode,
        dependencies, edgeData, numDependencies);
  conn_t *conn = lupine_route_remote_conn(route);
  uint8_t numDependencies_present = numDependencies != nullptr;
  size_t numDependencies_requested =
      (numDependencies_present && dependencies != nullptr) ? *numDependencies
                                                           : 0;
  uint8_t dependencies_null = dependencies == nullptr ? 1 : 0;
  uint8_t edgeData_null = edgeData == nullptr ? 1 : 0;
  if (lupine_prepare_rpc(conn) < 0 ||
      rpc_write_start_request_on(conn, RPC_cuGraphNodeGetDependencies_v2,
                                 RPC_SCOPE_NONE) < 0 ||
      rpc_write(conn, &hNode, sizeof(CUgraphNode)) < 0 ||
      rpc_write(conn, &numDependencies_present, sizeof(uint8_t)) < 0 ||
      rpc_write(conn, &numDependencies_requested, sizeof(size_t)) < 0 ||
      rpc_write(conn, &dependencies_null, sizeof(uint8_t)) < 0 ||
      rpc_write(conn, &edgeData_null, sizeof(uint8_t)) < 0 ||
      rpc_wait_for_response(conn) < 0 ||
      (numDependencies_present &&
       rpc_read(conn, numDependencies, sizeof(size_t)) < 0) ||
      (dependencies != nullptr && numDependencies_requested != 0 &&
       numDependencies_present && *numDependencies != 0 &&
       rpc_read(conn, dependencies,
                ((numDependencies_present ? *numDependencies : 0) <
                         numDependencies_requested
                     ? (numDependencies_present ? *numDependencies : 0)
                     : numDependencies_requested) *
                    sizeof(CUgraphNode)) < 0) ||
      (edgeData != nullptr && numDependencies_requested != 0 &&
       numDependencies_present && *numDependencies != 0 &&
       rpc_read(conn, edgeData,
                ((numDependencies_present ? *numDependencies : 0) <
                         numDependencies_requested
                     ? (numDependencies_present ? *numDependencies : 0)
                     : numDependencies_requested) *
                    sizeof(CUgraphEdgeData)) < 0) ||
      rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
      rpc_read_end(conn) < 0)
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  return return_value;
}

CUresult cuGraphNodeGetDependentNodes_v2(CUgraphNode hNode,
                                         CUgraphNode *dependentNodes,
                                         CUgraphEdgeData *edgeData,
                                         size_t *numDependentNodes) {
  lupine_route route = lupine_route_for_graph_node(hNode);
  CUresult return_value;
  if (lupine_route_is_local(route))
    return lupine_call_real_cuda_fn(
        LUPINE_REAL_CUDA_SYMBOL("cuGraphNodeGetDependentNodes_v2"), hNode,
        dependentNodes, edgeData, numDependentNodes);
  conn_t *conn = lupine_route_remote_conn(route);
  uint8_t numDependentNodes_present = numDependentNodes != nullptr;
  size_t numDependentNodes_requested =
      (numDependentNodes_present && dependentNodes != nullptr)
          ? *numDependentNodes
          : 0;
  uint8_t dependentNodes_null = dependentNodes == nullptr ? 1 : 0;
  uint8_t edgeData_null = edgeData == nullptr ? 1 : 0;
  if (lupine_prepare_rpc(conn) < 0 ||
      rpc_write_start_request_on(conn, RPC_cuGraphNodeGetDependentNodes_v2,
                                 RPC_SCOPE_NONE) < 0 ||
      rpc_write(conn, &hNode, sizeof(CUgraphNode)) < 0 ||
      rpc_write(conn, &numDependentNodes_present, sizeof(uint8_t)) < 0 ||
      rpc_write(conn, &numDependentNodes_requested, sizeof(size_t)) < 0 ||
      rpc_write(conn, &dependentNodes_null, sizeof(uint8_t)) < 0 ||
      rpc_write(conn, &edgeData_null, sizeof(uint8_t)) < 0 ||
      rpc_wait_for_response(conn) < 0 ||
      (numDependentNodes_present &&
       rpc_read(conn, numDependentNodes, sizeof(size_t)) < 0) ||
      (dependentNodes != nullptr && numDependentNodes_requested != 0 &&
       numDependentNodes_present && *numDependentNodes != 0 &&
       rpc_read(conn, dependentNodes,
                ((numDependentNodes_present ? *numDependentNodes : 0) <
                         numDependentNodes_requested
                     ? (numDependentNodes_present ? *numDependentNodes : 0)
                     : numDependentNodes_requested) *
                    sizeof(CUgraphNode)) < 0) ||
      (edgeData != nullptr && numDependentNodes_requested != 0 &&
       numDependentNodes_present && *numDependentNodes != 0 &&
       rpc_read(conn, edgeData,
                ((numDependentNodes_present ? *numDependentNodes : 0) <
                         numDependentNodes_requested
                     ? (numDependentNodes_present ? *numDependentNodes : 0)
                     : numDependentNodes_requested) *
                    sizeof(CUgraphEdgeData)) < 0) ||
      rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
      rpc_read_end(conn) < 0)
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  return return_value;
}

CUresult cuGraphAddDependencies_v2(CUgraph hGraph, const CUgraphNode *from,
                                   const CUgraphNode *to,
                                   const CUgraphEdgeData *edgeData,
                                   size_t numDependencies) {
  lupine_route route = lupine_route_for_graph(hGraph);
  CUresult return_value;
  if (lupine_route_is_local(route))
    return lupine_call_real_cuda_fn(
        LUPINE_REAL_CUDA_SYMBOL("cuGraphAddDependencies_v2"), hGraph, from, to,
        edgeData, numDependencies);
  conn_t *conn = lupine_route_remote_conn(route);
  uint8_t from_null = from == nullptr ? 1 : 0;
  uint8_t to_null = to == nullptr ? 1 : 0;
  uint8_t edgeData_null = edgeData == nullptr ? 1 : 0;
  if (lupine_prepare_rpc(conn) < 0 ||
      rpc_write_start_request_on(conn, RPC_cuGraphAddDependencies_v2,
                                 RPC_SCOPE_NONE) < 0 ||
      rpc_write(conn, &hGraph, sizeof(CUgraph)) < 0 ||
      rpc_write(conn, &numDependencies, sizeof(size_t)) < 0 ||
      rpc_write(conn, &from_null, sizeof(uint8_t)) < 0 ||
      (!from_null && numDependencies * sizeof(const CUgraphNode) != 0 &&
       rpc_write(conn, from, numDependencies * sizeof(const CUgraphNode)) <
           0) ||
      rpc_write(conn, &to_null, sizeof(uint8_t)) < 0 ||
      (!to_null && numDependencies * sizeof(const CUgraphNode) != 0 &&
       rpc_write(conn, to, numDependencies * sizeof(const CUgraphNode)) < 0) ||
      rpc_write(conn, &edgeData_null, sizeof(uint8_t)) < 0 ||
      (!edgeData_null && numDependencies * sizeof(const CUgraphEdgeData) != 0 &&
       rpc_write(conn, edgeData,
                 numDependencies * sizeof(const CUgraphEdgeData)) < 0) ||
      rpc_wait_for_response(conn) < 0 ||
      rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
      rpc_read_end(conn) < 0)
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  return return_value;
}

CUresult cuGraphRemoveDependencies_v2(CUgraph hGraph, const CUgraphNode *from,
                                      const CUgraphNode *to,
                                      const CUgraphEdgeData *edgeData,
                                      size_t numDependencies) {
  lupine_route route = lupine_route_for_graph(hGraph);
  CUresult return_value;
  if (lupine_route_is_local(route))
    return lupine_call_real_cuda_fn(
        LUPINE_REAL_CUDA_SYMBOL("cuGraphRemoveDependencies_v2"), hGraph, from,
        to, edgeData, numDependencies);
  conn_t *conn = lupine_route_remote_conn(route);
  uint8_t from_null = from == nullptr ? 1 : 0;
  uint8_t to_null = to == nullptr ? 1 : 0;
  uint8_t edgeData_null = edgeData == nullptr ? 1 : 0;
  if (lupine_prepare_rpc(conn) < 0 ||
      rpc_write_start_request_on(conn, RPC_cuGraphRemoveDependencies_v2,
                                 RPC_SCOPE_NONE) < 0 ||
      rpc_write(conn, &hGraph, sizeof(CUgraph)) < 0 ||
      rpc_write(conn, &numDependencies, sizeof(size_t)) < 0 ||
      rpc_write(conn, &from_null, sizeof(uint8_t)) < 0 ||
      (!from_null && numDependencies * sizeof(const CUgraphNode) != 0 &&
       rpc_write(conn, from, numDependencies * sizeof(const CUgraphNode)) <
           0) ||
      rpc_write(conn, &to_null, sizeof(uint8_t)) < 0 ||
      (!to_null && numDependencies * sizeof(const CUgraphNode) != 0 &&
       rpc_write(conn, to, numDependencies * sizeof(const CUgraphNode)) < 0) ||
      rpc_write(conn, &edgeData_null, sizeof(uint8_t)) < 0 ||
      (!edgeData_null && numDependencies * sizeof(const CUgraphEdgeData) != 0 &&
       rpc_write(conn, edgeData,
                 numDependencies * sizeof(const CUgraphEdgeData)) < 0) ||
      rpc_wait_for_response(conn) < 0 ||
      rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
      rpc_read_end(conn) < 0)
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  return return_value;
}

CUresult cuGraphDestroyNode(CUgraphNode hNode) {
  lupine_route route = lupine_route_for_graph_node(hNode);
  CUresult return_value;
  if (lupine_route_is_local(route)) {
    return_value = lupine_call_real_cuda_fn(
        LUPINE_REAL_CUDA_SYMBOL("cuGraphDestroyNode"), hNode);
    if (return_value == CUDA_SUCCESS)
      lupine_deep_node_cache_reset(hNode);
    return return_value;
  }
  conn_t *conn = lupine_route_remote_conn(route);
  if (lupine_prepare_rpc(conn) < 0 ||
      rpc_write_start_request_on(conn, RPC_cuGraphDestroyNode, RPC_SCOPE_NONE) <
          0 ||
      rpc_write(conn, &hNode, sizeof(CUgraphNode)) < 0 ||
      rpc_write_end(conn) < 0)
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  return_value = CUDA_SUCCESS;
  if (return_value == CUDA_SUCCESS)
    lupine_deep_node_cache_reset(hNode);
  return return_value;
}

CUresult cuGraphInstantiateWithFlags(CUgraphExec *phGraphExec, CUgraph hGraph,
                                     unsigned long long flags) {
  lupine_route route = lupine_route_for_graph(hGraph);
  CUresult return_value;
  if (lupine_route_is_local(route)) {
    return_value = lupine_call_real_cuda_fn(
        LUPINE_REAL_CUDA_SYMBOL("cuGraphInstantiateWithFlags"), phGraphExec,
        hGraph, flags);
    if (return_value == CUDA_SUCCESS && phGraphExec != nullptr) {
      lupine_note_graph_exec_owner_route(*phGraphExec, route);
    }
    return return_value;
  }
  conn_t *conn = lupine_route_remote_conn(route);
  if (phGraphExec == nullptr)
    return CUDA_ERROR_INVALID_VALUE;
  if (lupine_prepare_rpc(conn) < 0 ||
      rpc_write_start_request_on(conn, RPC_cuGraphInstantiateWithFlags,
                                 RPC_SCOPE_NONE) < 0 ||
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
  if (lupine_route_is_local(route)) {
    return_value = lupine_call_real_cuda_fn(
        LUPINE_REAL_CUDA_SYMBOL("cuGraphInstantiateWithParams"), phGraphExec,
        hGraph, instantiateParams);
    if (return_value == CUDA_SUCCESS && phGraphExec != nullptr) {
      lupine_note_graph_exec_owner_route(*phGraphExec, route);
    }
    return return_value;
  }
  conn_t *conn = lupine_route_remote_conn(route);
  if (phGraphExec == nullptr)
    return CUDA_ERROR_INVALID_VALUE;
  if (instantiateParams == nullptr)
    return CUDA_ERROR_INVALID_VALUE;
  if (lupine_prepare_rpc(conn) < 0 ||
      rpc_write_start_request_on(conn, RPC_cuGraphInstantiateWithParams,
                                 RPC_SCOPE_NONE) < 0 ||
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
  if (lupine_route_is_local(route))
    return lupine_call_real_cuda_fn(
        LUPINE_REAL_CUDA_SYMBOL("cuGraphExecGetFlags"), hGraphExec, flags);
  conn_t *conn = lupine_route_remote_conn(route);
  uint8_t flags_present = flags != nullptr;
  if (lupine_prepare_rpc(conn) < 0 ||
      rpc_write_start_request_on(conn, RPC_cuGraphExecGetFlags,
                                 reinterpret_cast<uintptr_t>(hGraphExec)) < 0 ||
      rpc_write(conn, &hGraphExec, sizeof(CUgraphExec)) < 0 ||
      rpc_write(conn, &flags_present, sizeof(uint8_t)) < 0 ||
      (flags_present && rpc_write(conn, flags, sizeof(cuuint64_t)) < 0) ||
      rpc_wait_for_response(conn) < 0 ||
      (flags_present && rpc_read(conn, flags, sizeof(cuuint64_t)) < 0) ||
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
  if (lupine_route_is_local(route))
    return lupine_call_real_cuda_fn(
        LUPINE_REAL_CUDA_SYMBOL("cuGraphExecMemcpyNodeSetParams"), hGraphExec,
        hNode, copyParams, ctx);
  conn_t *conn = lupine_route_remote_conn(route);
  if (copyParams == nullptr)
    return CUDA_ERROR_INVALID_VALUE;
  if (lupine_prepare_rpc(conn) < 0 ||
      rpc_write_start_request_on(conn, RPC_cuGraphExecMemcpyNodeSetParams,
                                 reinterpret_cast<uintptr_t>(hGraphExec)) < 0 ||
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
  if (lupine_route_is_local(route))
    return lupine_call_real_cuda_fn(
        LUPINE_REAL_CUDA_SYMBOL("cuGraphExecMemsetNodeSetParams"), hGraphExec,
        hNode, memsetParams, ctx);
  conn_t *conn = lupine_route_remote_conn(route);
  uint8_t memsetParams_present = memsetParams != nullptr;
  if (lupine_prepare_rpc(conn) < 0 ||
      rpc_write_start_request_on(conn, RPC_cuGraphExecMemsetNodeSetParams,
                                 reinterpret_cast<uintptr_t>(hGraphExec)) < 0 ||
      rpc_write(conn, &hGraphExec, sizeof(CUgraphExec)) < 0 ||
      rpc_write(conn, &hNode, sizeof(CUgraphNode)) < 0 ||
      rpc_write(conn, &memsetParams_present, sizeof(uint8_t)) < 0 ||
      (memsetParams_present &&
       rpc_write(conn, memsetParams, sizeof(const CUDA_MEMSET_NODE_PARAMS)) <
           0) ||
      rpc_write(conn, &ctx, sizeof(CUcontext)) < 0 ||
      rpc_wait_for_response(conn) < 0 ||
      rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
      rpc_read_end(conn) < 0)
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  return return_value;
}

CUresult cuGraphExecHostNodeSetParams(CUgraphExec hGraphExec, CUgraphNode hNode,
                                      const CUDA_HOST_NODE_PARAMS *nodeParams) {
  lupine_route route = lupine_route_for_graph_exec(hGraphExec);
  CUresult return_value;
  if (lupine_route_is_local(route))
    return lupine_call_real_cuda_fn(
        LUPINE_REAL_CUDA_SYMBOL("cuGraphExecHostNodeSetParams"), hGraphExec,
        hNode, nodeParams);
  conn_t *conn = lupine_route_remote_conn(route);
  if (nodeParams == nullptr)
    return CUDA_ERROR_INVALID_VALUE;
  if (lupine_prepare_rpc(conn) < 0 ||
      rpc_write_start_request_on(conn, RPC_cuGraphExecHostNodeSetParams,
                                 reinterpret_cast<uintptr_t>(hGraphExec)) < 0 ||
      rpc_write(conn, &hGraphExec, sizeof(CUgraphExec)) < 0 ||
      rpc_write(conn, &hNode, sizeof(CUgraphNode)) < 0 ||
      rpc_write(conn, nodeParams, sizeof(const CUDA_HOST_NODE_PARAMS)) < 0 ||
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
  if (lupine_route_is_local(route))
    return lupine_call_real_cuda_fn(
        LUPINE_REAL_CUDA_SYMBOL("cuGraphExecChildGraphNodeSetParams"),
        hGraphExec, hNode, childGraph);
  conn_t *conn = lupine_route_remote_conn(route);
  if (lupine_prepare_rpc(conn) < 0 ||
      rpc_write_start_request_on(conn, RPC_cuGraphExecChildGraphNodeSetParams,
                                 reinterpret_cast<uintptr_t>(hGraphExec)) < 0 ||
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
  if (lupine_route_is_local(route))
    return lupine_call_real_cuda_fn(
        LUPINE_REAL_CUDA_SYMBOL("cuGraphExecEventRecordNodeSetEvent"),
        hGraphExec, hNode, event);
  conn_t *conn = lupine_route_remote_conn(route);
  if (lupine_prepare_rpc(conn) < 0 ||
      rpc_write_start_request_on(conn, RPC_cuGraphExecEventRecordNodeSetEvent,
                                 reinterpret_cast<uintptr_t>(event)) < 0 ||
      rpc_request_on(conn, reinterpret_cast<uintptr_t>(hGraphExec)) < 0 ||
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
  if (lupine_route_is_local(route))
    return lupine_call_real_cuda_fn(
        LUPINE_REAL_CUDA_SYMBOL("cuGraphExecEventWaitNodeSetEvent"), hGraphExec,
        hNode, event);
  conn_t *conn = lupine_route_remote_conn(route);
  if (lupine_prepare_rpc(conn) < 0 ||
      rpc_write_start_request_on(conn, RPC_cuGraphExecEventWaitNodeSetEvent,
                                 reinterpret_cast<uintptr_t>(event)) < 0 ||
      rpc_request_on(conn, reinterpret_cast<uintptr_t>(hGraphExec)) < 0 ||
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
  if (lupine_route_is_local(route))
    return lupine_call_real_cuda_fn(
        LUPINE_REAL_CUDA_SYMBOL(
            "cuGraphExecExternalSemaphoresSignalNodeSetParams"),
        hGraphExec, hNode, nodeParams);
  conn_t *conn = lupine_route_remote_conn(route);
  if (nodeParams == nullptr)
    return CUDA_ERROR_INVALID_VALUE;
  if (nodeParams->numExtSems > SIZE_MAX / sizeof(*nodeParams->extSemArray) ||
      (nodeParams->numExtSems != 0 && nodeParams->extSemArray == nullptr))
    return CUDA_ERROR_INVALID_VALUE;
  if (nodeParams->numExtSems > SIZE_MAX / sizeof(*nodeParams->paramsArray) ||
      (nodeParams->numExtSems != 0 && nodeParams->paramsArray == nullptr))
    return CUDA_ERROR_INVALID_VALUE;
  if (lupine_prepare_rpc(conn) < 0 ||
      rpc_write_start_request_on(
          conn, RPC_cuGraphExecExternalSemaphoresSignalNodeSetParams,
          reinterpret_cast<uintptr_t>(hGraphExec)) < 0 ||
      rpc_write(conn, &hGraphExec, sizeof(CUgraphExec)) < 0 ||
      rpc_write(conn, &hNode, sizeof(CUgraphNode)) < 0 ||
      rpc_write(conn, nodeParams, sizeof(*nodeParams)) < 0 ||
      rpc_write(conn, nodeParams->extSemArray,
                nodeParams->numExtSems * sizeof(*nodeParams->extSemArray)) <
          0 ||
      rpc_write(conn, nodeParams->paramsArray,
                nodeParams->numExtSems * sizeof(*nodeParams->paramsArray)) <
          0 ||
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
  if (lupine_route_is_local(route))
    return lupine_call_real_cuda_fn(
        LUPINE_REAL_CUDA_SYMBOL(
            "cuGraphExecExternalSemaphoresWaitNodeSetParams"),
        hGraphExec, hNode, nodeParams);
  conn_t *conn = lupine_route_remote_conn(route);
  if (nodeParams == nullptr)
    return CUDA_ERROR_INVALID_VALUE;
  if (nodeParams->numExtSems > SIZE_MAX / sizeof(*nodeParams->extSemArray) ||
      (nodeParams->numExtSems != 0 && nodeParams->extSemArray == nullptr))
    return CUDA_ERROR_INVALID_VALUE;
  if (nodeParams->numExtSems > SIZE_MAX / sizeof(*nodeParams->paramsArray) ||
      (nodeParams->numExtSems != 0 && nodeParams->paramsArray == nullptr))
    return CUDA_ERROR_INVALID_VALUE;
  if (lupine_prepare_rpc(conn) < 0 ||
      rpc_write_start_request_on(
          conn, RPC_cuGraphExecExternalSemaphoresWaitNodeSetParams,
          reinterpret_cast<uintptr_t>(hGraphExec)) < 0 ||
      rpc_write(conn, &hGraphExec, sizeof(CUgraphExec)) < 0 ||
      rpc_write(conn, &hNode, sizeof(CUgraphNode)) < 0 ||
      rpc_write(conn, nodeParams, sizeof(*nodeParams)) < 0 ||
      rpc_write(conn, nodeParams->extSemArray,
                nodeParams->numExtSems * sizeof(*nodeParams->extSemArray)) <
          0 ||
      rpc_write(conn, nodeParams->paramsArray,
                nodeParams->numExtSems * sizeof(*nodeParams->paramsArray)) <
          0 ||
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
  if (lupine_route_is_local(route))
    return lupine_call_real_cuda_fn(
        LUPINE_REAL_CUDA_SYMBOL("cuGraphNodeSetEnabled"), hGraphExec, hNode,
        isEnabled);
  conn_t *conn = lupine_route_remote_conn(route);
  if (lupine_prepare_rpc(conn) < 0 ||
      rpc_write_start_request_on(conn, RPC_cuGraphNodeSetEnabled,
                                 reinterpret_cast<uintptr_t>(hGraphExec)) < 0 ||
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
  if (lupine_route_is_local(route))
    return lupine_call_real_cuda_fn(
        LUPINE_REAL_CUDA_SYMBOL("cuGraphNodeGetEnabled"), hGraphExec, hNode,
        isEnabled);
  conn_t *conn = lupine_route_remote_conn(route);
  uint8_t isEnabled_present = isEnabled != nullptr;
  if (lupine_prepare_rpc(conn) < 0 ||
      rpc_write_start_request_on(conn, RPC_cuGraphNodeGetEnabled,
                                 reinterpret_cast<uintptr_t>(hGraphExec)) < 0 ||
      rpc_write(conn, &hGraphExec, sizeof(CUgraphExec)) < 0 ||
      rpc_write(conn, &hNode, sizeof(CUgraphNode)) < 0 ||
      rpc_write(conn, &isEnabled_present, sizeof(uint8_t)) < 0 ||
      (isEnabled_present &&
       rpc_write(conn, isEnabled, sizeof(unsigned int)) < 0) ||
      rpc_wait_for_response(conn) < 0 ||
      (isEnabled_present &&
       rpc_read(conn, isEnabled, sizeof(unsigned int)) < 0) ||
      rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
      rpc_read_end(conn) < 0)
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  return return_value;
}

CUresult cuGraphUpload(CUgraphExec hGraphExec, CUstream hStream) {
  lupine_route route = lupine_route_for_graph_exec(hGraphExec);
  CUresult return_value;
  if (lupine_route_is_local(route))
    return lupine_call_real_cuda_fn(LUPINE_REAL_CUDA_SYMBOL("cuGraphUpload"),
                                    hGraphExec, hStream);
  conn_t *conn = lupine_route_remote_conn(route);
  if (lupine_prepare_rpc(conn) < 0 ||
      lupine_rpc_start_on_stream(conn, RPC_cuGraphUpload, hStream) < 0 ||
      rpc_request_on(conn, reinterpret_cast<uintptr_t>(hGraphExec)) < 0 ||
      rpc_write(conn, &hGraphExec, sizeof(CUgraphExec)) < 0 ||
      rpc_write(conn, &hStream, sizeof(CUstream)) < 0 ||
      rpc_write_end(conn) < 0)
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  return_value = CUDA_SUCCESS;
  return return_value;
}

CUresult cuGraphLaunch(CUgraphExec hGraphExec, CUstream hStream) {
  lupine_route route = lupine_route_for_graph_exec(hGraphExec);
  CUresult return_value;
  if (lupine_route_is_local(route)) {
    return_value = lupine_call_real_cuda_fn(
        LUPINE_REAL_CUDA_SYMBOL("cuGraphLaunch"), hGraphExec, hStream);
    // The graph may re-record events it does not name here, so no cached
    // completion can be trusted past the launch.
    lupine_event_invalidate_all_completions();
    return return_value;
  }
  conn_t *conn = lupine_route_remote_conn(route);
  if (lupine_prepare_rpc(conn) < 0 ||
      lupine_rpc_start_on_stream(conn, RPC_cuGraphLaunch, hStream,
                                 RPC_REQUEST_PUBLISH_CONNECTION) < 0 ||
      rpc_request_on(conn, reinterpret_cast<uintptr_t>(hGraphExec)) < 0 ||
      rpc_write(conn, &hGraphExec, sizeof(CUgraphExec)) < 0 ||
      rpc_write(conn, &hStream, sizeof(CUstream)) < 0 ||
      rpc_write_end(conn) < 0)
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  return_value = CUDA_SUCCESS;
  // The graph may re-record events it does not name here, so no cached
  // completion can be trusted past the launch.
  lupine_event_invalidate_all_completions();
  return return_value;
}

CUresult cuGraphExecDestroy(CUgraphExec hGraphExec) {
  lupine_route route = lupine_route_for_graph_exec(hGraphExec);
  CUresult return_value;
  if (lupine_route_is_local(route)) {
    return_value = lupine_call_real_cuda_fn(
        LUPINE_REAL_CUDA_SYMBOL("cuGraphExecDestroy"), hGraphExec);
    if (return_value == CUDA_SUCCESS)
      lupine_forget_graph_exec_owner(hGraphExec);
    return return_value;
  }
  conn_t *conn = lupine_route_remote_conn(route);
  if (lupine_prepare_rpc(conn) < 0 ||
      rpc_write_start_request_on(conn, RPC_cuGraphExecDestroy,
                                 reinterpret_cast<uintptr_t>(hGraphExec)) < 0 ||
      rpc_write(conn, &hGraphExec, sizeof(CUgraphExec)) < 0 ||
      rpc_wait_for_response(conn) < 0 ||
      rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
      rpc_read_end(conn) < 0)
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  if (return_value == CUDA_SUCCESS)
    lupine_forget_graph_exec_owner(hGraphExec);
  return return_value;
}

CUresult cuGraphDestroy(CUgraph hGraph) {
  lupine_route route = lupine_route_for_graph(hGraph);
  auto cached_nodes = lupine_deep_cache_graph_nodes(hGraph);
  CUresult return_value;
  if (lupine_route_is_local(route)) {
    return_value = lupine_call_real_cuda_fn(
        LUPINE_REAL_CUDA_SYMBOL("cuGraphDestroy"), hGraph);
    if (return_value == CUDA_SUCCESS) {
      for (CUgraphNode node : cached_nodes)
        lupine_deep_node_cache_reset(node);
    }
    return return_value;
  }
  conn_t *conn = lupine_route_remote_conn(route);
  if (lupine_prepare_rpc(conn) < 0 ||
      rpc_write_start_request_on(conn, RPC_cuGraphDestroy, RPC_SCOPE_NONE) <
          0 ||
      rpc_write(conn, &hGraph, sizeof(CUgraph)) < 0 || rpc_write_end(conn) < 0)
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  return_value = CUDA_SUCCESS;
  if (return_value == CUDA_SUCCESS) {
    for (CUgraphNode node : cached_nodes)
      lupine_deep_node_cache_reset(node);
  }
  return return_value;
}

CUresult cuGraphExecUpdate_v2(CUgraphExec hGraphExec, CUgraph hGraph,
                              CUgraphExecUpdateResultInfo *resultInfo) {
  lupine_route route = lupine_route_for_graph_exec(hGraphExec);
  CUresult return_value;
  if (lupine_route_is_local(route))
    return lupine_call_real_cuda_fn(
        LUPINE_REAL_CUDA_SYMBOL("cuGraphExecUpdate_v2"), hGraphExec, hGraph,
        resultInfo);
  conn_t *conn = lupine_route_remote_conn(route);
  uint8_t resultInfo_present = resultInfo != nullptr;
  if (lupine_prepare_rpc(conn) < 0 ||
      rpc_write_start_request_on(conn, RPC_cuGraphExecUpdate_v2,
                                 reinterpret_cast<uintptr_t>(hGraphExec)) < 0 ||
      rpc_write(conn, &hGraphExec, sizeof(CUgraphExec)) < 0 ||
      rpc_write(conn, &hGraph, sizeof(CUgraph)) < 0 ||
      rpc_write(conn, &resultInfo_present, sizeof(uint8_t)) < 0 ||
      (resultInfo_present &&
       rpc_write(conn, resultInfo, sizeof(CUgraphExecUpdateResultInfo)) < 0) ||
      rpc_wait_for_response(conn) < 0 ||
      (resultInfo_present &&
       rpc_read(conn, resultInfo, sizeof(CUgraphExecUpdateResultInfo)) < 0) ||
      rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
      rpc_read_end(conn) < 0)
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  return return_value;
}

CUresult cuGraphKernelNodeCopyAttributes(CUgraphNode dst, CUgraphNode src) {
  lupine_route route = lupine_route_for_graph_node(dst);
  CUresult return_value;
  if (lupine_route_is_local(route))
    return lupine_call_real_cuda_fn(
        LUPINE_REAL_CUDA_SYMBOL("cuGraphKernelNodeCopyAttributes"), dst, src);
  conn_t *conn = lupine_route_remote_conn(route);
  if (lupine_prepare_rpc(conn) < 0 ||
      rpc_write_start_request_on(conn, RPC_cuGraphKernelNodeCopyAttributes,
                                 RPC_SCOPE_NONE) < 0 ||
      rpc_write(conn, &dst, sizeof(CUgraphNode)) < 0 ||
      rpc_write(conn, &src, sizeof(CUgraphNode)) < 0 || rpc_write_end(conn) < 0)
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  return_value = CUDA_SUCCESS;
  return return_value;
}

CUresult cuGraphKernelNodeGetAttribute(CUgraphNode hNode,
                                       CUkernelNodeAttrID attr,
                                       CUkernelNodeAttrValue *value_out) {
  lupine_route route = lupine_route_for_graph_node(hNode);
  CUresult return_value;
  if (lupine_route_is_local(route))
    return lupine_call_real_cuda_fn(
        LUPINE_REAL_CUDA_SYMBOL("cuGraphKernelNodeGetAttribute"), hNode, attr,
        value_out);
  conn_t *conn = lupine_route_remote_conn(route);
  uint8_t value_out_present = value_out != nullptr;
  if (lupine_prepare_rpc(conn) < 0 ||
      rpc_write_start_request_on(conn, RPC_cuGraphKernelNodeGetAttribute,
                                 RPC_SCOPE_NONE) < 0 ||
      rpc_write(conn, &hNode, sizeof(CUgraphNode)) < 0 ||
      rpc_write(conn, &attr, sizeof(CUkernelNodeAttrID)) < 0 ||
      rpc_write(conn, &value_out_present, sizeof(uint8_t)) < 0 ||
      (value_out_present &&
       rpc_write(conn, value_out, sizeof(CUkernelNodeAttrValue)) < 0) ||
      rpc_wait_for_response(conn) < 0 ||
      (value_out_present &&
       rpc_read(conn, value_out, sizeof(CUkernelNodeAttrValue)) < 0) ||
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
  if (lupine_route_is_local(route))
    return lupine_call_real_cuda_fn(
        LUPINE_REAL_CUDA_SYMBOL("cuGraphKernelNodeSetAttribute"), hNode, attr,
        value);
  conn_t *conn = lupine_route_remote_conn(route);
  uint8_t value_present = value != nullptr;
  if (lupine_prepare_rpc(conn) < 0 ||
      rpc_write_start_request_on(conn, RPC_cuGraphKernelNodeSetAttribute,
                                 RPC_SCOPE_NONE) < 0 ||
      rpc_write(conn, &hNode, sizeof(CUgraphNode)) < 0 ||
      rpc_write(conn, &attr, sizeof(CUkernelNodeAttrID)) < 0 ||
      rpc_write(conn, &value_present, sizeof(uint8_t)) < 0 ||
      (value_present &&
       rpc_write(conn, value, sizeof(const CUkernelNodeAttrValue)) < 0) ||
      rpc_write_end(conn) < 0)
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  return_value = CUDA_SUCCESS;
  return return_value;
}

CUresult cuGraphDebugDotPrint(CUgraph hGraph, const char *path,
                              unsigned int flags) {
  lupine_route route = lupine_route_for_graph(hGraph);
  CUresult return_value;
  if (lupine_route_is_local(route))
    return lupine_call_real_cuda_fn(
        LUPINE_REAL_CUDA_SYMBOL("cuGraphDebugDotPrint"), hGraph, path, flags);
  conn_t *conn = lupine_route_remote_conn(route);
  std::size_t path_len =
      path == nullptr ? 0 : static_cast<std::size_t>(std::strlen(path) + 1);
  if (lupine_prepare_rpc(conn) < 0 ||
      rpc_write_start_request_on(conn, RPC_cuGraphDebugDotPrint,
                                 RPC_SCOPE_NONE) < 0 ||
      rpc_write(conn, &hGraph, sizeof(CUgraph)) < 0 ||
      rpc_write(conn, &path_len, sizeof(std::size_t)) < 0 ||
      (path_len != 0 && rpc_write(conn, path, path_len) < 0) ||
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
  if (lupine_route_is_local(route))
    return lupine_call_real_cuda_fn(
        LUPINE_REAL_CUDA_SYMBOL("cuUserObjectRetain"), object, count);
  conn_t *conn = lupine_route_remote_conn(route);
  if (lupine_prepare_rpc(conn) < 0 ||
      rpc_write_start_request_on(conn, RPC_cuUserObjectRetain, RPC_SCOPE_NONE) <
          0 ||
      rpc_write(conn, &object, sizeof(CUuserObject)) < 0 ||
      rpc_write(conn, &count, sizeof(unsigned int)) < 0 ||
      rpc_write_end(conn) < 0)
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  return_value = CUDA_SUCCESS;
  return return_value;
}

CUresult cuUserObjectRelease(CUuserObject object, unsigned int count) {
  lupine_route route = lupine_route_for_default();
  CUresult return_value;
  if (lupine_route_is_local(route))
    return lupine_call_real_cuda_fn(
        LUPINE_REAL_CUDA_SYMBOL("cuUserObjectRelease"), object, count);
  conn_t *conn = lupine_route_remote_conn(route);
  if (lupine_prepare_rpc(conn) < 0 ||
      rpc_write_start_request_on(conn, RPC_cuUserObjectRelease,
                                 RPC_SCOPE_NONE) < 0 ||
      rpc_write(conn, &object, sizeof(CUuserObject)) < 0 ||
      rpc_write(conn, &count, sizeof(unsigned int)) < 0 ||
      rpc_write_end(conn) < 0)
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  return_value = CUDA_SUCCESS;
  return return_value;
}

CUresult cuGraphRetainUserObject(CUgraph graph, CUuserObject object,
                                 unsigned int count, unsigned int flags) {
  lupine_route route = lupine_route_for_graph(graph);
  CUresult return_value;
  if (lupine_route_is_local(route))
    return lupine_call_real_cuda_fn(
        LUPINE_REAL_CUDA_SYMBOL("cuGraphRetainUserObject"), graph, object,
        count, flags);
  conn_t *conn = lupine_route_remote_conn(route);
  if (lupine_prepare_rpc(conn) < 0 ||
      rpc_write_start_request_on(conn, RPC_cuGraphRetainUserObject,
                                 RPC_SCOPE_NONE) < 0 ||
      rpc_write(conn, &graph, sizeof(CUgraph)) < 0 ||
      rpc_write(conn, &object, sizeof(CUuserObject)) < 0 ||
      rpc_write(conn, &count, sizeof(unsigned int)) < 0 ||
      rpc_write(conn, &flags, sizeof(unsigned int)) < 0 ||
      rpc_write_end(conn) < 0)
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  return_value = CUDA_SUCCESS;
  return return_value;
}

CUresult cuGraphReleaseUserObject(CUgraph graph, CUuserObject object,
                                  unsigned int count) {
  lupine_route route = lupine_route_for_graph(graph);
  CUresult return_value;
  if (lupine_route_is_local(route))
    return lupine_call_real_cuda_fn(
        LUPINE_REAL_CUDA_SYMBOL("cuGraphReleaseUserObject"), graph, object,
        count);
  conn_t *conn = lupine_route_remote_conn(route);
  if (lupine_prepare_rpc(conn) < 0 ||
      rpc_write_start_request_on(conn, RPC_cuGraphReleaseUserObject,
                                 RPC_SCOPE_NONE) < 0 ||
      rpc_write(conn, &graph, sizeof(CUgraph)) < 0 ||
      rpc_write(conn, &object, sizeof(CUuserObject)) < 0 ||
      rpc_write(conn, &count, sizeof(unsigned int)) < 0 ||
      rpc_write_end(conn) < 0)
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  return_value = CUDA_SUCCESS;
  return return_value;
}

#if CUDA_VERSION >= 12020
CUresult cuGraphNodeSetParams(CUgraphNode hNode,
                              CUgraphNodeParams *nodeParams) {
  lupine_route route = lupine_route_for_graph_node(hNode);
  CUresult return_value;
  if (lupine_route_is_local(route))
    return lupine_call_real_cuda_fn(
        LUPINE_REAL_CUDA_SYMBOL("cuGraphNodeSetParams"), hNode, nodeParams);
  conn_t *conn = lupine_route_remote_conn(route);
  uint8_t nodeParams_present = nodeParams != nullptr;
  if (lupine_prepare_rpc(conn) < 0 ||
      rpc_write_start_request_on(conn, RPC_cuGraphNodeSetParams,
                                 RPC_SCOPE_NONE) < 0 ||
      rpc_write(conn, &hNode, sizeof(CUgraphNode)) < 0 ||
      rpc_write(conn, &nodeParams_present, sizeof(uint8_t)) < 0 ||
      (nodeParams_present &&
       rpc_write(conn, nodeParams, sizeof(CUgraphNodeParams)) < 0) ||
      rpc_wait_for_response(conn) < 0 ||
      rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
      rpc_read_end(conn) < 0)
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  return return_value;
}

#endif

#if CUDA_VERSION >= 12020
CUresult cuGraphExecNodeSetParams(CUgraphExec hGraphExec, CUgraphNode hNode,
                                  CUgraphNodeParams *nodeParams) {
  lupine_route route = lupine_route_for_graph_exec(hGraphExec);
  CUresult return_value;
  if (lupine_route_is_local(route))
    return lupine_call_real_cuda_fn(
        LUPINE_REAL_CUDA_SYMBOL("cuGraphExecNodeSetParams"), hGraphExec, hNode,
        nodeParams);
  conn_t *conn = lupine_route_remote_conn(route);
  uint8_t nodeParams_present = nodeParams != nullptr;
  if (lupine_prepare_rpc(conn) < 0 ||
      rpc_write_start_request_on(conn, RPC_cuGraphExecNodeSetParams,
                                 reinterpret_cast<uintptr_t>(hGraphExec)) < 0 ||
      rpc_write(conn, &hGraphExec, sizeof(CUgraphExec)) < 0 ||
      rpc_write(conn, &hNode, sizeof(CUgraphNode)) < 0 ||
      rpc_write(conn, &nodeParams_present, sizeof(uint8_t)) < 0 ||
      (nodeParams_present &&
       rpc_write(conn, nodeParams, sizeof(CUgraphNodeParams)) < 0) ||
      rpc_wait_for_response(conn) < 0 ||
      rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
      rpc_read_end(conn) < 0)
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  return return_value;
}

#endif

CUresult cuGraphConditionalHandleCreate(CUgraphConditionalHandle *pHandle_out,
                                        CUgraph hGraph, CUcontext ctx,
                                        unsigned int defaultLaunchValue,
                                        unsigned int flags) {
  lupine_route route = (hGraph != nullptr ? lupine_route_for_graph(hGraph)
                                          : lupine_route_for_context(ctx));
  CUresult return_value;
  if (lupine_route_is_local(route))
    return lupine_call_real_cuda_fn(
        LUPINE_REAL_CUDA_SYMBOL("cuGraphConditionalHandleCreate"), pHandle_out,
        hGraph, ctx, defaultLaunchValue, flags);
  conn_t *conn = lupine_route_remote_conn(route);
  uint8_t pHandle_out_present = pHandle_out != nullptr;
  if (lupine_prepare_rpc(conn) < 0 ||
      rpc_write_start_request_on(conn, RPC_cuGraphConditionalHandleCreate,
                                 RPC_SCOPE_NONE) < 0 ||
      rpc_write(conn, &pHandle_out_present, sizeof(uint8_t)) < 0 ||
      (pHandle_out_present &&
       rpc_write(conn, pHandle_out, sizeof(CUgraphConditionalHandle)) < 0) ||
      rpc_write(conn, &hGraph, sizeof(CUgraph)) < 0 ||
      rpc_write(conn, &ctx, sizeof(CUcontext)) < 0 ||
      rpc_write(conn, &defaultLaunchValue, sizeof(unsigned int)) < 0 ||
      rpc_write(conn, &flags, sizeof(unsigned int)) < 0 ||
      rpc_wait_for_response(conn) < 0 ||
      (pHandle_out_present &&
       rpc_read(conn, pHandle_out, sizeof(CUgraphConditionalHandle)) < 0) ||
      rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
      rpc_read_end(conn) < 0)
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  return return_value;
}

CUresult cuOccupancyMaxActiveBlocksPerMultiprocessor(int *numBlocks,
                                                     CUfunction func,
                                                     int blockSize,
                                                     size_t dynamicSMemSize) {
  lupine_route route = lupine_route_for_function(func);
  if (numBlocks == nullptr)
    return CUDA_ERROR_INVALID_VALUE;
  func = lupine_translate_private_function_for_rpc(func);
  route = lupine_route_for_function(func);
  if (!lupine_route_is_local(route) &&
      lupine_try_cached_occupancy(route, func, blockSize, dynamicSMemSize, 0,
                                  false, numBlocks))
    return CUDA_SUCCESS;
  CUresult return_value;
  if (lupine_route_is_local(route)) {
    return_value = lupine_call_real_cuda_fn(
        LUPINE_REAL_CUDA_SYMBOL("cuOccupancyMaxActiveBlocksPerMultiprocessor"),
        numBlocks, func, blockSize, dynamicSMemSize);
    if (return_value == CUDA_SUCCESS && !lupine_route_is_local(route))
      lupine_store_cached_occupancy(route, func, blockSize, dynamicSMemSize, 0,
                                    false, *numBlocks);
    return return_value;
  }
  conn_t *conn = lupine_route_remote_conn(route);
  uint8_t numBlocks_present = numBlocks != nullptr;
  CUfunction func_rpc = lupine_translate_private_function_for_rpc(func);
  if (lupine_prepare_rpc(conn) < 0 ||
      rpc_write_start_request_on(
          conn, RPC_cuOccupancyMaxActiveBlocksPerMultiprocessor,
          RPC_SCOPE_NONE) < 0 ||
      rpc_write(conn, &numBlocks_present, sizeof(uint8_t)) < 0 ||
      (numBlocks_present && rpc_write(conn, numBlocks, sizeof(int)) < 0) ||
      rpc_write(conn, &func_rpc, sizeof(CUfunction)) < 0 ||
      rpc_write(conn, &blockSize, sizeof(int)) < 0 ||
      rpc_write(conn, &dynamicSMemSize, sizeof(size_t)) < 0 ||
      rpc_wait_for_response(conn) < 0 ||
      (numBlocks_present && rpc_read(conn, numBlocks, sizeof(int)) < 0) ||
      rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
      rpc_read_end(conn) < 0)
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  if (return_value == CUDA_SUCCESS && !lupine_route_is_local(route))
    lupine_store_cached_occupancy(route, func, blockSize, dynamicSMemSize, 0,
                                  false, *numBlocks);
  return return_value;
}

CUresult cuOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(
    int *numBlocks, CUfunction func, int blockSize, size_t dynamicSMemSize,
    unsigned int flags) {
  lupine_route route = lupine_route_for_function(func);
  if (numBlocks == nullptr)
    return CUDA_ERROR_INVALID_VALUE;
  func = lupine_translate_private_function_for_rpc(func);
  route = lupine_route_for_function(func);
  if (!lupine_route_is_local(route) &&
      lupine_try_cached_occupancy(route, func, blockSize, dynamicSMemSize,
                                  flags, true, numBlocks))
    return CUDA_SUCCESS;
  CUresult return_value;
  if (lupine_route_is_local(route)) {
    return_value = lupine_call_real_cuda_fn(
        LUPINE_REAL_CUDA_SYMBOL(
            "cuOccupancyMaxActiveBlocksPerMultiprocessorWithFlags"),
        numBlocks, func, blockSize, dynamicSMemSize, flags);
    if (return_value == CUDA_SUCCESS && !lupine_route_is_local(route))
      lupine_store_cached_occupancy(route, func, blockSize, dynamicSMemSize,
                                    flags, true, *numBlocks);
    return return_value;
  }
  conn_t *conn = lupine_route_remote_conn(route);
  uint8_t numBlocks_present = numBlocks != nullptr;
  CUfunction func_rpc = lupine_translate_private_function_for_rpc(func);
  if (lupine_prepare_rpc(conn) < 0 ||
      rpc_write_start_request_on(
          conn, RPC_cuOccupancyMaxActiveBlocksPerMultiprocessorWithFlags,
          RPC_SCOPE_NONE) < 0 ||
      rpc_write(conn, &numBlocks_present, sizeof(uint8_t)) < 0 ||
      (numBlocks_present && rpc_write(conn, numBlocks, sizeof(int)) < 0) ||
      rpc_write(conn, &func_rpc, sizeof(CUfunction)) < 0 ||
      rpc_write(conn, &blockSize, sizeof(int)) < 0 ||
      rpc_write(conn, &dynamicSMemSize, sizeof(size_t)) < 0 ||
      rpc_write(conn, &flags, sizeof(unsigned int)) < 0 ||
      rpc_wait_for_response(conn) < 0 ||
      (numBlocks_present && rpc_read(conn, numBlocks, sizeof(int)) < 0) ||
      rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
      rpc_read_end(conn) < 0)
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  if (return_value == CUDA_SUCCESS && !lupine_route_is_local(route))
    lupine_store_cached_occupancy(route, func, blockSize, dynamicSMemSize,
                                  flags, true, *numBlocks);
  return return_value;
}

CUresult
cuOccupancyMaxPotentialBlockSize(int *minGridSize, int *blockSize,
                                 CUfunction func,
                                 CUoccupancyB2DSize blockSizeToDynamicSMemSize,
                                 size_t dynamicSMemSize, int blockSizeLimit) {
  lupine_route route = lupine_route_for_function(func);
  if (minGridSize == nullptr || blockSize == nullptr)
    return CUDA_ERROR_INVALID_VALUE;
  if (blockSizeToDynamicSMemSize != nullptr)
    return CUDA_ERROR_NOT_SUPPORTED;
  CUresult return_value;
  if (lupine_route_is_local(route))
    return lupine_call_real_cuda_fn(
        LUPINE_REAL_CUDA_SYMBOL("cuOccupancyMaxPotentialBlockSize"),
        minGridSize, blockSize, func, blockSizeToDynamicSMemSize,
        dynamicSMemSize, blockSizeLimit);
  conn_t *conn = lupine_route_remote_conn(route);
  uint8_t minGridSize_present = minGridSize != nullptr;
  uint8_t blockSize_present = blockSize != nullptr;
  CUfunction func_rpc = lupine_translate_private_function_for_rpc(func);
  if (lupine_prepare_rpc(conn) < 0 ||
      rpc_write_start_request_on(conn, RPC_cuOccupancyMaxPotentialBlockSize,
                                 RPC_SCOPE_NONE) < 0 ||
      rpc_write(conn, &minGridSize_present, sizeof(uint8_t)) < 0 ||
      (minGridSize_present && rpc_write(conn, minGridSize, sizeof(int)) < 0) ||
      rpc_write(conn, &blockSize_present, sizeof(uint8_t)) < 0 ||
      (blockSize_present && rpc_write(conn, blockSize, sizeof(int)) < 0) ||
      rpc_write(conn, &func_rpc, sizeof(CUfunction)) < 0 ||
      rpc_write(conn, &dynamicSMemSize, sizeof(size_t)) < 0 ||
      rpc_write(conn, &blockSizeLimit, sizeof(int)) < 0 ||
      rpc_wait_for_response(conn) < 0 ||
      (minGridSize_present && rpc_read(conn, minGridSize, sizeof(int)) < 0) ||
      (blockSize_present && rpc_read(conn, blockSize, sizeof(int)) < 0) ||
      rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
      rpc_read_end(conn) < 0)
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  return return_value;
}

CUresult cuOccupancyMaxPotentialBlockSizeWithFlags(
    int *minGridSize, int *blockSize, CUfunction func,
    CUoccupancyB2DSize blockSizeToDynamicSMemSize, size_t dynamicSMemSize,
    int blockSizeLimit, unsigned int flags) {
  lupine_route route = lupine_route_for_function(func);
  if (minGridSize == nullptr || blockSize == nullptr)
    return CUDA_ERROR_INVALID_VALUE;
  if (blockSizeToDynamicSMemSize != nullptr)
    return CUDA_ERROR_NOT_SUPPORTED;
  CUresult return_value;
  if (lupine_route_is_local(route))
    return lupine_call_real_cuda_fn(
        LUPINE_REAL_CUDA_SYMBOL("cuOccupancyMaxPotentialBlockSizeWithFlags"),
        minGridSize, blockSize, func, blockSizeToDynamicSMemSize,
        dynamicSMemSize, blockSizeLimit, flags);
  conn_t *conn = lupine_route_remote_conn(route);
  uint8_t minGridSize_present = minGridSize != nullptr;
  uint8_t blockSize_present = blockSize != nullptr;
  CUfunction func_rpc = lupine_translate_private_function_for_rpc(func);
  if (lupine_prepare_rpc(conn) < 0 ||
      rpc_write_start_request_on(conn,
                                 RPC_cuOccupancyMaxPotentialBlockSizeWithFlags,
                                 RPC_SCOPE_NONE) < 0 ||
      rpc_write(conn, &minGridSize_present, sizeof(uint8_t)) < 0 ||
      (minGridSize_present && rpc_write(conn, minGridSize, sizeof(int)) < 0) ||
      rpc_write(conn, &blockSize_present, sizeof(uint8_t)) < 0 ||
      (blockSize_present && rpc_write(conn, blockSize, sizeof(int)) < 0) ||
      rpc_write(conn, &func_rpc, sizeof(CUfunction)) < 0 ||
      rpc_write(conn, &dynamicSMemSize, sizeof(size_t)) < 0 ||
      rpc_write(conn, &blockSizeLimit, sizeof(int)) < 0 ||
      rpc_write(conn, &flags, sizeof(unsigned int)) < 0 ||
      rpc_wait_for_response(conn) < 0 ||
      (minGridSize_present && rpc_read(conn, minGridSize, sizeof(int)) < 0) ||
      (blockSize_present && rpc_read(conn, blockSize, sizeof(int)) < 0) ||
      rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
      rpc_read_end(conn) < 0)
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  return return_value;
}

CUresult cuOccupancyAvailableDynamicSMemPerBlock(size_t *dynamicSmemSize,
                                                 CUfunction func, int numBlocks,
                                                 int blockSize) {
  lupine_route route = lupine_route_for_function(func);
  CUresult return_value;
  if (lupine_route_is_local(route))
    return lupine_call_real_cuda_fn(
        LUPINE_REAL_CUDA_SYMBOL("cuOccupancyAvailableDynamicSMemPerBlock"),
        dynamicSmemSize, func, numBlocks, blockSize);
  conn_t *conn = lupine_route_remote_conn(route);
  uint8_t dynamicSmemSize_present = dynamicSmemSize != nullptr;
  CUfunction func_rpc = lupine_translate_private_function_for_rpc(func);
  if (lupine_prepare_rpc(conn) < 0 ||
      rpc_write_start_request_on(conn,
                                 RPC_cuOccupancyAvailableDynamicSMemPerBlock,
                                 RPC_SCOPE_NONE) < 0 ||
      rpc_write(conn, &dynamicSmemSize_present, sizeof(uint8_t)) < 0 ||
      (dynamicSmemSize_present &&
       rpc_write(conn, dynamicSmemSize, sizeof(size_t)) < 0) ||
      rpc_write(conn, &func_rpc, sizeof(CUfunction)) < 0 ||
      rpc_write(conn, &numBlocks, sizeof(int)) < 0 ||
      rpc_write(conn, &blockSize, sizeof(int)) < 0 ||
      rpc_wait_for_response(conn) < 0 ||
      (dynamicSmemSize_present &&
       rpc_read(conn, dynamicSmemSize, sizeof(size_t)) < 0) ||
      rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
      rpc_read_end(conn) < 0)
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  return return_value;
}

CUresult cuOccupancyMaxPotentialClusterSize(int *clusterSize, CUfunction func,
                                            const CUlaunchConfig *config) {
  lupine_route route = lupine_route_for_function(func);
  CUresult return_value;
  if (lupine_route_is_local(route))
    return lupine_call_real_cuda_fn(
        LUPINE_REAL_CUDA_SYMBOL("cuOccupancyMaxPotentialClusterSize"),
        clusterSize, func, config);
  conn_t *conn = lupine_route_remote_conn(route);
  uint8_t clusterSize_present = clusterSize != nullptr;
  CUfunction func_rpc = lupine_translate_private_function_for_rpc(func);
  if (config == nullptr)
    return CUDA_ERROR_INVALID_VALUE;
  if (config->numAttrs > SIZE_MAX / sizeof(*config->attrs) ||
      (config->numAttrs != 0 && config->attrs == nullptr))
    return CUDA_ERROR_INVALID_VALUE;
  if (lupine_prepare_rpc(conn) < 0 ||
      rpc_write_start_request_on(conn, RPC_cuOccupancyMaxPotentialClusterSize,
                                 RPC_SCOPE_NONE) < 0 ||
      rpc_write(conn, &clusterSize_present, sizeof(uint8_t)) < 0 ||
      (clusterSize_present && rpc_write(conn, clusterSize, sizeof(int)) < 0) ||
      rpc_write(conn, &func_rpc, sizeof(CUfunction)) < 0 ||
      rpc_write(conn, config, sizeof(*config)) < 0 ||
      rpc_write(conn, config->attrs,
                config->numAttrs * sizeof(*config->attrs)) < 0 ||
      rpc_wait_for_response(conn) < 0 ||
      (clusterSize_present && rpc_read(conn, clusterSize, sizeof(int)) < 0) ||
      rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
      rpc_read_end(conn) < 0)
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  return return_value;
}

CUresult cuOccupancyMaxActiveClusters(int *numClusters, CUfunction func,
                                      const CUlaunchConfig *config) {
  lupine_route route = lupine_route_for_function(func);
  CUresult return_value;
  if (lupine_route_is_local(route))
    return lupine_call_real_cuda_fn(
        LUPINE_REAL_CUDA_SYMBOL("cuOccupancyMaxActiveClusters"), numClusters,
        func, config);
  conn_t *conn = lupine_route_remote_conn(route);
  uint8_t numClusters_present = numClusters != nullptr;
  CUfunction func_rpc = lupine_translate_private_function_for_rpc(func);
  if (config == nullptr)
    return CUDA_ERROR_INVALID_VALUE;
  if (config->numAttrs > SIZE_MAX / sizeof(*config->attrs) ||
      (config->numAttrs != 0 && config->attrs == nullptr))
    return CUDA_ERROR_INVALID_VALUE;
  if (lupine_prepare_rpc(conn) < 0 ||
      rpc_write_start_request_on(conn, RPC_cuOccupancyMaxActiveClusters,
                                 RPC_SCOPE_NONE) < 0 ||
      rpc_write(conn, &numClusters_present, sizeof(uint8_t)) < 0 ||
      (numClusters_present && rpc_write(conn, numClusters, sizeof(int)) < 0) ||
      rpc_write(conn, &func_rpc, sizeof(CUfunction)) < 0 ||
      rpc_write(conn, config, sizeof(*config)) < 0 ||
      rpc_write(conn, config->attrs,
                config->numAttrs * sizeof(*config->attrs)) < 0 ||
      rpc_wait_for_response(conn) < 0 ||
      (numClusters_present && rpc_read(conn, numClusters, sizeof(int)) < 0) ||
      rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
      rpc_read_end(conn) < 0)
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  return return_value;
}

CUresult cuTexRefSetArray(CUtexref hTexRef, CUarray hArray,
                          unsigned int Flags) {
  lupine_route route = lupine_route_for_default();
  CUresult return_value;
  if (lupine_route_is_local(route))
    return lupine_call_real_cuda_fn(LUPINE_REAL_CUDA_SYMBOL("cuTexRefSetArray"),
                                    hTexRef, hArray, Flags);
  conn_t *conn = lupine_route_remote_conn(route);
  if (lupine_prepare_rpc(conn) < 0 ||
      rpc_write_start_request_on(conn, RPC_cuTexRefSetArray, RPC_SCOPE_NONE) <
          0 ||
      rpc_write(conn, &hTexRef, sizeof(CUtexref)) < 0 ||
      rpc_write(conn, &hArray, sizeof(CUarray)) < 0 ||
      rpc_write(conn, &Flags, sizeof(unsigned int)) < 0 ||
      rpc_write_end(conn) < 0)
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  return_value = CUDA_SUCCESS;
  return return_value;
}

CUresult cuTexRefSetMipmappedArray(CUtexref hTexRef,
                                   CUmipmappedArray hMipmappedArray,
                                   unsigned int Flags) {
  lupine_route route = lupine_route_for_default();
  CUresult return_value;
  if (lupine_route_is_local(route))
    return lupine_call_real_cuda_fn(
        LUPINE_REAL_CUDA_SYMBOL("cuTexRefSetMipmappedArray"), hTexRef,
        hMipmappedArray, Flags);
  conn_t *conn = lupine_route_remote_conn(route);
  if (lupine_prepare_rpc(conn) < 0 ||
      rpc_write_start_request_on(conn, RPC_cuTexRefSetMipmappedArray,
                                 RPC_SCOPE_NONE) < 0 ||
      rpc_write(conn, &hTexRef, sizeof(CUtexref)) < 0 ||
      rpc_write(conn, &hMipmappedArray, sizeof(CUmipmappedArray)) < 0 ||
      rpc_write(conn, &Flags, sizeof(unsigned int)) < 0 ||
      rpc_write_end(conn) < 0)
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  return_value = CUDA_SUCCESS;
  return return_value;
}

CUresult cuTexRefSetAddress_v2(size_t *ByteOffset, CUtexref hTexRef,
                               CUdeviceptr dptr, size_t bytes) {
  lupine_route route = lupine_route_for_deviceptr(dptr);
  CUresult return_value;
  if (lupine_route_is_local(route))
    return lupine_call_real_cuda_fn(
        LUPINE_REAL_CUDA_SYMBOL("cuTexRefSetAddress_v2"), ByteOffset, hTexRef,
        dptr, bytes);
  conn_t *conn = lupine_route_remote_conn(route);
  uint8_t ByteOffset_present = ByteOffset != nullptr;
  if (lupine_prepare_rpc(conn) < 0 ||
      rpc_write_start_request_on(conn, RPC_cuTexRefSetAddress_v2,
                                 RPC_SCOPE_NONE) < 0 ||
      rpc_write(conn, &ByteOffset_present, sizeof(uint8_t)) < 0 ||
      (ByteOffset_present && rpc_write(conn, ByteOffset, sizeof(size_t)) < 0) ||
      rpc_write(conn, &hTexRef, sizeof(CUtexref)) < 0 ||
      rpc_write(conn, &dptr, sizeof(CUdeviceptr)) < 0 ||
      rpc_write(conn, &bytes, sizeof(size_t)) < 0 ||
      rpc_wait_for_response(conn) < 0 ||
      (ByteOffset_present && rpc_read(conn, ByteOffset, sizeof(size_t)) < 0) ||
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
  if (lupine_route_is_local(route))
    return lupine_call_real_cuda_fn(
        LUPINE_REAL_CUDA_SYMBOL("cuTexRefSetAddress2D_v3"), hTexRef, desc, dptr,
        Pitch);
  conn_t *conn = lupine_route_remote_conn(route);
  uint8_t desc_present = desc != nullptr;
  if (lupine_prepare_rpc(conn) < 0 ||
      rpc_write_start_request_on(conn, RPC_cuTexRefSetAddress2D_v3,
                                 RPC_SCOPE_NONE) < 0 ||
      rpc_write(conn, &hTexRef, sizeof(CUtexref)) < 0 ||
      rpc_write(conn, &desc_present, sizeof(uint8_t)) < 0 ||
      (desc_present &&
       rpc_write(conn, desc, sizeof(const CUDA_ARRAY_DESCRIPTOR)) < 0) ||
      rpc_write(conn, &dptr, sizeof(CUdeviceptr)) < 0 ||
      rpc_write(conn, &Pitch, sizeof(size_t)) < 0 || rpc_write_end(conn) < 0)
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  return_value = CUDA_SUCCESS;
  return return_value;
}

CUresult cuTexRefSetFormat(CUtexref hTexRef, CUarray_format fmt,
                           int NumPackedComponents) {
  lupine_route route = lupine_route_for_default();
  CUresult return_value;
  if (lupine_route_is_local(route))
    return lupine_call_real_cuda_fn(
        LUPINE_REAL_CUDA_SYMBOL("cuTexRefSetFormat"), hTexRef, fmt,
        NumPackedComponents);
  conn_t *conn = lupine_route_remote_conn(route);
  if (lupine_prepare_rpc(conn) < 0 ||
      rpc_write_start_request_on(conn, RPC_cuTexRefSetFormat, RPC_SCOPE_NONE) <
          0 ||
      rpc_write(conn, &hTexRef, sizeof(CUtexref)) < 0 ||
      rpc_write(conn, &fmt, sizeof(CUarray_format)) < 0 ||
      rpc_write(conn, &NumPackedComponents, sizeof(int)) < 0 ||
      rpc_write_end(conn) < 0)
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  return_value = CUDA_SUCCESS;
  return return_value;
}

CUresult cuTexRefSetAddressMode(CUtexref hTexRef, int dim, CUaddress_mode am) {
  lupine_route route = lupine_route_for_default();
  CUresult return_value;
  if (lupine_route_is_local(route))
    return lupine_call_real_cuda_fn(
        LUPINE_REAL_CUDA_SYMBOL("cuTexRefSetAddressMode"), hTexRef, dim, am);
  conn_t *conn = lupine_route_remote_conn(route);
  if (lupine_prepare_rpc(conn) < 0 ||
      rpc_write_start_request_on(conn, RPC_cuTexRefSetAddressMode,
                                 RPC_SCOPE_NONE) < 0 ||
      rpc_write(conn, &hTexRef, sizeof(CUtexref)) < 0 ||
      rpc_write(conn, &dim, sizeof(int)) < 0 ||
      rpc_write(conn, &am, sizeof(CUaddress_mode)) < 0 ||
      rpc_write_end(conn) < 0)
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  return_value = CUDA_SUCCESS;
  return return_value;
}

CUresult cuTexRefSetFilterMode(CUtexref hTexRef, CUfilter_mode fm) {
  lupine_route route = lupine_route_for_default();
  CUresult return_value;
  if (lupine_route_is_local(route))
    return lupine_call_real_cuda_fn(
        LUPINE_REAL_CUDA_SYMBOL("cuTexRefSetFilterMode"), hTexRef, fm);
  conn_t *conn = lupine_route_remote_conn(route);
  if (lupine_prepare_rpc(conn) < 0 ||
      rpc_write_start_request_on(conn, RPC_cuTexRefSetFilterMode,
                                 RPC_SCOPE_NONE) < 0 ||
      rpc_write(conn, &hTexRef, sizeof(CUtexref)) < 0 ||
      rpc_write(conn, &fm, sizeof(CUfilter_mode)) < 0 ||
      rpc_write_end(conn) < 0)
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  return_value = CUDA_SUCCESS;
  return return_value;
}

CUresult cuTexRefSetMipmapFilterMode(CUtexref hTexRef, CUfilter_mode fm) {
  lupine_route route = lupine_route_for_default();
  CUresult return_value;
  if (lupine_route_is_local(route))
    return lupine_call_real_cuda_fn(
        LUPINE_REAL_CUDA_SYMBOL("cuTexRefSetMipmapFilterMode"), hTexRef, fm);
  conn_t *conn = lupine_route_remote_conn(route);
  if (lupine_prepare_rpc(conn) < 0 ||
      rpc_write_start_request_on(conn, RPC_cuTexRefSetMipmapFilterMode,
                                 RPC_SCOPE_NONE) < 0 ||
      rpc_write(conn, &hTexRef, sizeof(CUtexref)) < 0 ||
      rpc_write(conn, &fm, sizeof(CUfilter_mode)) < 0 ||
      rpc_write_end(conn) < 0)
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  return_value = CUDA_SUCCESS;
  return return_value;
}

CUresult cuTexRefSetMipmapLevelBias(CUtexref hTexRef, float bias) {
  lupine_route route = lupine_route_for_default();
  CUresult return_value;
  if (lupine_route_is_local(route))
    return lupine_call_real_cuda_fn(
        LUPINE_REAL_CUDA_SYMBOL("cuTexRefSetMipmapLevelBias"), hTexRef, bias);
  conn_t *conn = lupine_route_remote_conn(route);
  if (lupine_prepare_rpc(conn) < 0 ||
      rpc_write_start_request_on(conn, RPC_cuTexRefSetMipmapLevelBias,
                                 RPC_SCOPE_NONE) < 0 ||
      rpc_write(conn, &hTexRef, sizeof(CUtexref)) < 0 ||
      rpc_write(conn, &bias, sizeof(float)) < 0 || rpc_write_end(conn) < 0)
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  return_value = CUDA_SUCCESS;
  return return_value;
}

CUresult cuTexRefSetMipmapLevelClamp(CUtexref hTexRef,
                                     float minMipmapLevelClamp,
                                     float maxMipmapLevelClamp) {
  lupine_route route = lupine_route_for_default();
  CUresult return_value;
  if (lupine_route_is_local(route))
    return lupine_call_real_cuda_fn(
        LUPINE_REAL_CUDA_SYMBOL("cuTexRefSetMipmapLevelClamp"), hTexRef,
        minMipmapLevelClamp, maxMipmapLevelClamp);
  conn_t *conn = lupine_route_remote_conn(route);
  if (lupine_prepare_rpc(conn) < 0 ||
      rpc_write_start_request_on(conn, RPC_cuTexRefSetMipmapLevelClamp,
                                 RPC_SCOPE_NONE) < 0 ||
      rpc_write(conn, &hTexRef, sizeof(CUtexref)) < 0 ||
      rpc_write(conn, &minMipmapLevelClamp, sizeof(float)) < 0 ||
      rpc_write(conn, &maxMipmapLevelClamp, sizeof(float)) < 0 ||
      rpc_write_end(conn) < 0)
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  return_value = CUDA_SUCCESS;
  return return_value;
}

CUresult cuTexRefSetMaxAnisotropy(CUtexref hTexRef, unsigned int maxAniso) {
  lupine_route route = lupine_route_for_default();
  CUresult return_value;
  if (lupine_route_is_local(route))
    return lupine_call_real_cuda_fn(
        LUPINE_REAL_CUDA_SYMBOL("cuTexRefSetMaxAnisotropy"), hTexRef, maxAniso);
  conn_t *conn = lupine_route_remote_conn(route);
  if (lupine_prepare_rpc(conn) < 0 ||
      rpc_write_start_request_on(conn, RPC_cuTexRefSetMaxAnisotropy,
                                 RPC_SCOPE_NONE) < 0 ||
      rpc_write(conn, &hTexRef, sizeof(CUtexref)) < 0 ||
      rpc_write(conn, &maxAniso, sizeof(unsigned int)) < 0 ||
      rpc_write_end(conn) < 0)
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  return_value = CUDA_SUCCESS;
  return return_value;
}

CUresult cuTexRefSetBorderColor(CUtexref hTexRef, float *pBorderColor) {
  lupine_route route = lupine_route_for_default();
  CUresult return_value;
  if (lupine_route_is_local(route))
    return lupine_call_real_cuda_fn(
        LUPINE_REAL_CUDA_SYMBOL("cuTexRefSetBorderColor"), hTexRef,
        pBorderColor);
  conn_t *conn = lupine_route_remote_conn(route);
  uint8_t pBorderColor_null = pBorderColor == nullptr ? 1 : 0;
  if (lupine_prepare_rpc(conn) < 0 ||
      rpc_write_start_request_on(conn, RPC_cuTexRefSetBorderColor,
                                 RPC_SCOPE_NONE) < 0 ||
      rpc_write(conn, &hTexRef, sizeof(CUtexref)) < 0 ||
      rpc_write(conn, &pBorderColor_null, sizeof(uint8_t)) < 0 ||
      (!pBorderColor_null && 16 != 0 &&
       rpc_write(conn, pBorderColor, 16) < 0) ||
      rpc_write_end(conn) < 0)
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  return_value = CUDA_SUCCESS;
  return return_value;
}

CUresult cuTexRefSetFlags(CUtexref hTexRef, unsigned int Flags) {
  lupine_route route = lupine_route_for_default();
  CUresult return_value;
  if (lupine_route_is_local(route))
    return lupine_call_real_cuda_fn(LUPINE_REAL_CUDA_SYMBOL("cuTexRefSetFlags"),
                                    hTexRef, Flags);
  conn_t *conn = lupine_route_remote_conn(route);
  if (lupine_prepare_rpc(conn) < 0 ||
      rpc_write_start_request_on(conn, RPC_cuTexRefSetFlags, RPC_SCOPE_NONE) <
          0 ||
      rpc_write(conn, &hTexRef, sizeof(CUtexref)) < 0 ||
      rpc_write(conn, &Flags, sizeof(unsigned int)) < 0 ||
      rpc_write_end(conn) < 0)
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  return_value = CUDA_SUCCESS;
  return return_value;
}

CUresult cuTexRefGetAddress_v2(CUdeviceptr *pdptr, CUtexref hTexRef) {
  lupine_route route = lupine_route_for_default();
  CUresult return_value;
  if (lupine_route_is_local(route))
    return lupine_call_real_cuda_fn(
        LUPINE_REAL_CUDA_SYMBOL("cuTexRefGetAddress_v2"), pdptr, hTexRef);
  conn_t *conn = lupine_route_remote_conn(route);
  uint8_t pdptr_present = pdptr != nullptr;
  if (lupine_prepare_rpc(conn) < 0 ||
      rpc_write_start_request_on(conn, RPC_cuTexRefGetAddress_v2,
                                 RPC_SCOPE_NONE) < 0 ||
      rpc_write(conn, &pdptr_present, sizeof(uint8_t)) < 0 ||
      (pdptr_present && rpc_write(conn, pdptr, sizeof(CUdeviceptr)) < 0) ||
      rpc_write(conn, &hTexRef, sizeof(CUtexref)) < 0 ||
      rpc_wait_for_response(conn) < 0 ||
      (pdptr_present && rpc_read(conn, pdptr, sizeof(CUdeviceptr)) < 0) ||
      rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
      rpc_read_end(conn) < 0)
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  return return_value;
}

CUresult cuTexRefGetArray(CUarray *phArray, CUtexref hTexRef) {
  lupine_route route = lupine_route_for_default();
  CUresult return_value;
  if (lupine_route_is_local(route))
    return lupine_call_real_cuda_fn(LUPINE_REAL_CUDA_SYMBOL("cuTexRefGetArray"),
                                    phArray, hTexRef);
  conn_t *conn = lupine_route_remote_conn(route);
  uint8_t phArray_present = phArray != nullptr;
  if (lupine_prepare_rpc(conn) < 0 ||
      rpc_write_start_request_on(conn, RPC_cuTexRefGetArray, RPC_SCOPE_NONE) <
          0 ||
      rpc_write(conn, &phArray_present, sizeof(uint8_t)) < 0 ||
      (phArray_present && rpc_write(conn, phArray, sizeof(CUarray)) < 0) ||
      rpc_write(conn, &hTexRef, sizeof(CUtexref)) < 0 ||
      rpc_wait_for_response(conn) < 0 ||
      (phArray_present && rpc_read(conn, phArray, sizeof(CUarray)) < 0) ||
      rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
      rpc_read_end(conn) < 0)
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  return return_value;
}

CUresult cuTexRefGetMipmappedArray(CUmipmappedArray *phMipmappedArray,
                                   CUtexref hTexRef) {
  lupine_route route = lupine_route_for_default();
  CUresult return_value;
  if (lupine_route_is_local(route))
    return lupine_call_real_cuda_fn(
        LUPINE_REAL_CUDA_SYMBOL("cuTexRefGetMipmappedArray"), phMipmappedArray,
        hTexRef);
  conn_t *conn = lupine_route_remote_conn(route);
  uint8_t phMipmappedArray_present = phMipmappedArray != nullptr;
  if (lupine_prepare_rpc(conn) < 0 ||
      rpc_write_start_request_on(conn, RPC_cuTexRefGetMipmappedArray,
                                 RPC_SCOPE_NONE) < 0 ||
      rpc_write(conn, &phMipmappedArray_present, sizeof(uint8_t)) < 0 ||
      (phMipmappedArray_present &&
       rpc_write(conn, phMipmappedArray, sizeof(CUmipmappedArray)) < 0) ||
      rpc_write(conn, &hTexRef, sizeof(CUtexref)) < 0 ||
      rpc_wait_for_response(conn) < 0 ||
      (phMipmappedArray_present &&
       rpc_read(conn, phMipmappedArray, sizeof(CUmipmappedArray)) < 0) ||
      rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
      rpc_read_end(conn) < 0)
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  return return_value;
}

CUresult cuTexRefGetAddressMode(CUaddress_mode *pam, CUtexref hTexRef,
                                int dim) {
  lupine_route route = lupine_route_for_default();
  CUresult return_value;
  if (lupine_route_is_local(route))
    return lupine_call_real_cuda_fn(
        LUPINE_REAL_CUDA_SYMBOL("cuTexRefGetAddressMode"), pam, hTexRef, dim);
  conn_t *conn = lupine_route_remote_conn(route);
  uint8_t pam_present = pam != nullptr;
  if (lupine_prepare_rpc(conn) < 0 ||
      rpc_write_start_request_on(conn, RPC_cuTexRefGetAddressMode,
                                 RPC_SCOPE_NONE) < 0 ||
      rpc_write(conn, &pam_present, sizeof(uint8_t)) < 0 ||
      (pam_present && rpc_write(conn, pam, sizeof(CUaddress_mode)) < 0) ||
      rpc_write(conn, &hTexRef, sizeof(CUtexref)) < 0 ||
      rpc_write(conn, &dim, sizeof(int)) < 0 ||
      rpc_wait_for_response(conn) < 0 ||
      (pam_present && rpc_read(conn, pam, sizeof(CUaddress_mode)) < 0) ||
      rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
      rpc_read_end(conn) < 0)
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  return return_value;
}

CUresult cuTexRefGetFilterMode(CUfilter_mode *pfm, CUtexref hTexRef) {
  lupine_route route = lupine_route_for_default();
  CUresult return_value;
  if (lupine_route_is_local(route))
    return lupine_call_real_cuda_fn(
        LUPINE_REAL_CUDA_SYMBOL("cuTexRefGetFilterMode"), pfm, hTexRef);
  conn_t *conn = lupine_route_remote_conn(route);
  uint8_t pfm_present = pfm != nullptr;
  if (lupine_prepare_rpc(conn) < 0 ||
      rpc_write_start_request_on(conn, RPC_cuTexRefGetFilterMode,
                                 RPC_SCOPE_NONE) < 0 ||
      rpc_write(conn, &pfm_present, sizeof(uint8_t)) < 0 ||
      (pfm_present && rpc_write(conn, pfm, sizeof(CUfilter_mode)) < 0) ||
      rpc_write(conn, &hTexRef, sizeof(CUtexref)) < 0 ||
      rpc_wait_for_response(conn) < 0 ||
      (pfm_present && rpc_read(conn, pfm, sizeof(CUfilter_mode)) < 0) ||
      rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
      rpc_read_end(conn) < 0)
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  return return_value;
}

CUresult cuTexRefGetFormat(CUarray_format *pFormat, int *pNumChannels,
                           CUtexref hTexRef) {
  lupine_route route = lupine_route_for_default();
  CUresult return_value;
  if (lupine_route_is_local(route))
    return lupine_call_real_cuda_fn(
        LUPINE_REAL_CUDA_SYMBOL("cuTexRefGetFormat"), pFormat, pNumChannels,
        hTexRef);
  conn_t *conn = lupine_route_remote_conn(route);
  uint8_t pFormat_present = pFormat != nullptr;
  uint8_t pNumChannels_present = pNumChannels != nullptr;
  if (lupine_prepare_rpc(conn) < 0 ||
      rpc_write_start_request_on(conn, RPC_cuTexRefGetFormat, RPC_SCOPE_NONE) <
          0 ||
      rpc_write(conn, &pFormat_present, sizeof(uint8_t)) < 0 ||
      (pFormat_present &&
       rpc_write(conn, pFormat, sizeof(CUarray_format)) < 0) ||
      rpc_write(conn, &pNumChannels_present, sizeof(uint8_t)) < 0 ||
      (pNumChannels_present &&
       rpc_write(conn, pNumChannels, sizeof(int)) < 0) ||
      rpc_write(conn, &hTexRef, sizeof(CUtexref)) < 0 ||
      rpc_wait_for_response(conn) < 0 ||
      (pFormat_present &&
       rpc_read(conn, pFormat, sizeof(CUarray_format)) < 0) ||
      (pNumChannels_present && rpc_read(conn, pNumChannels, sizeof(int)) < 0) ||
      rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
      rpc_read_end(conn) < 0)
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  return return_value;
}

CUresult cuTexRefGetMipmapFilterMode(CUfilter_mode *pfm, CUtexref hTexRef) {
  lupine_route route = lupine_route_for_default();
  CUresult return_value;
  if (lupine_route_is_local(route))
    return lupine_call_real_cuda_fn(
        LUPINE_REAL_CUDA_SYMBOL("cuTexRefGetMipmapFilterMode"), pfm, hTexRef);
  conn_t *conn = lupine_route_remote_conn(route);
  uint8_t pfm_present = pfm != nullptr;
  if (lupine_prepare_rpc(conn) < 0 ||
      rpc_write_start_request_on(conn, RPC_cuTexRefGetMipmapFilterMode,
                                 RPC_SCOPE_NONE) < 0 ||
      rpc_write(conn, &pfm_present, sizeof(uint8_t)) < 0 ||
      (pfm_present && rpc_write(conn, pfm, sizeof(CUfilter_mode)) < 0) ||
      rpc_write(conn, &hTexRef, sizeof(CUtexref)) < 0 ||
      rpc_wait_for_response(conn) < 0 ||
      (pfm_present && rpc_read(conn, pfm, sizeof(CUfilter_mode)) < 0) ||
      rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
      rpc_read_end(conn) < 0)
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  return return_value;
}

CUresult cuTexRefGetMipmapLevelBias(float *pbias, CUtexref hTexRef) {
  lupine_route route = lupine_route_for_default();
  CUresult return_value;
  if (lupine_route_is_local(route))
    return lupine_call_real_cuda_fn(
        LUPINE_REAL_CUDA_SYMBOL("cuTexRefGetMipmapLevelBias"), pbias, hTexRef);
  conn_t *conn = lupine_route_remote_conn(route);
  uint8_t pbias_present = pbias != nullptr;
  if (lupine_prepare_rpc(conn) < 0 ||
      rpc_write_start_request_on(conn, RPC_cuTexRefGetMipmapLevelBias,
                                 RPC_SCOPE_NONE) < 0 ||
      rpc_write(conn, &pbias_present, sizeof(uint8_t)) < 0 ||
      (pbias_present && rpc_write(conn, pbias, sizeof(float)) < 0) ||
      rpc_write(conn, &hTexRef, sizeof(CUtexref)) < 0 ||
      rpc_wait_for_response(conn) < 0 ||
      (pbias_present && rpc_read(conn, pbias, sizeof(float)) < 0) ||
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
  if (lupine_route_is_local(route))
    return lupine_call_real_cuda_fn(
        LUPINE_REAL_CUDA_SYMBOL("cuTexRefGetMipmapLevelClamp"),
        pminMipmapLevelClamp, pmaxMipmapLevelClamp, hTexRef);
  conn_t *conn = lupine_route_remote_conn(route);
  uint8_t pminMipmapLevelClamp_present = pminMipmapLevelClamp != nullptr;
  uint8_t pmaxMipmapLevelClamp_present = pmaxMipmapLevelClamp != nullptr;
  if (lupine_prepare_rpc(conn) < 0 ||
      rpc_write_start_request_on(conn, RPC_cuTexRefGetMipmapLevelClamp,
                                 RPC_SCOPE_NONE) < 0 ||
      rpc_write(conn, &pminMipmapLevelClamp_present, sizeof(uint8_t)) < 0 ||
      (pminMipmapLevelClamp_present &&
       rpc_write(conn, pminMipmapLevelClamp, sizeof(float)) < 0) ||
      rpc_write(conn, &pmaxMipmapLevelClamp_present, sizeof(uint8_t)) < 0 ||
      (pmaxMipmapLevelClamp_present &&
       rpc_write(conn, pmaxMipmapLevelClamp, sizeof(float)) < 0) ||
      rpc_write(conn, &hTexRef, sizeof(CUtexref)) < 0 ||
      rpc_wait_for_response(conn) < 0 ||
      (pminMipmapLevelClamp_present &&
       rpc_read(conn, pminMipmapLevelClamp, sizeof(float)) < 0) ||
      (pmaxMipmapLevelClamp_present &&
       rpc_read(conn, pmaxMipmapLevelClamp, sizeof(float)) < 0) ||
      rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
      rpc_read_end(conn) < 0)
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  return return_value;
}

CUresult cuTexRefGetMaxAnisotropy(int *pmaxAniso, CUtexref hTexRef) {
  lupine_route route = lupine_route_for_default();
  CUresult return_value;
  if (lupine_route_is_local(route))
    return lupine_call_real_cuda_fn(
        LUPINE_REAL_CUDA_SYMBOL("cuTexRefGetMaxAnisotropy"), pmaxAniso,
        hTexRef);
  conn_t *conn = lupine_route_remote_conn(route);
  uint8_t pmaxAniso_present = pmaxAniso != nullptr;
  if (lupine_prepare_rpc(conn) < 0 ||
      rpc_write_start_request_on(conn, RPC_cuTexRefGetMaxAnisotropy,
                                 RPC_SCOPE_NONE) < 0 ||
      rpc_write(conn, &pmaxAniso_present, sizeof(uint8_t)) < 0 ||
      (pmaxAniso_present && rpc_write(conn, pmaxAniso, sizeof(int)) < 0) ||
      rpc_write(conn, &hTexRef, sizeof(CUtexref)) < 0 ||
      rpc_wait_for_response(conn) < 0 ||
      (pmaxAniso_present && rpc_read(conn, pmaxAniso, sizeof(int)) < 0) ||
      rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
      rpc_read_end(conn) < 0)
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  return return_value;
}

CUresult cuTexRefGetBorderColor(float *pBorderColor, CUtexref hTexRef) {
  lupine_route route = lupine_route_for_default();
  CUresult return_value;
  if (lupine_route_is_local(route))
    return lupine_call_real_cuda_fn(
        LUPINE_REAL_CUDA_SYMBOL("cuTexRefGetBorderColor"), pBorderColor,
        hTexRef);
  conn_t *conn = lupine_route_remote_conn(route);
  uint8_t pBorderColor_null = pBorderColor == nullptr ? 1 : 0;
  if (lupine_prepare_rpc(conn) < 0 ||
      rpc_write_start_request_on(conn, RPC_cuTexRefGetBorderColor,
                                 RPC_SCOPE_NONE) < 0 ||
      rpc_write(conn, &pBorderColor_null, sizeof(uint8_t)) < 0 ||
      rpc_write(conn, &hTexRef, sizeof(CUtexref)) < 0 ||
      rpc_wait_for_response(conn) < 0 ||
      (!pBorderColor_null && 16 != 0 && rpc_read(conn, pBorderColor, 16) < 0) ||
      rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
      rpc_read_end(conn) < 0)
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  return return_value;
}

CUresult cuTexRefGetFlags(unsigned int *pFlags, CUtexref hTexRef) {
  lupine_route route = lupine_route_for_default();
  CUresult return_value;
  if (lupine_route_is_local(route))
    return lupine_call_real_cuda_fn(LUPINE_REAL_CUDA_SYMBOL("cuTexRefGetFlags"),
                                    pFlags, hTexRef);
  conn_t *conn = lupine_route_remote_conn(route);
  uint8_t pFlags_present = pFlags != nullptr;
  if (lupine_prepare_rpc(conn) < 0 ||
      rpc_write_start_request_on(conn, RPC_cuTexRefGetFlags, RPC_SCOPE_NONE) <
          0 ||
      rpc_write(conn, &pFlags_present, sizeof(uint8_t)) < 0 ||
      (pFlags_present && rpc_write(conn, pFlags, sizeof(unsigned int)) < 0) ||
      rpc_write(conn, &hTexRef, sizeof(CUtexref)) < 0 ||
      rpc_wait_for_response(conn) < 0 ||
      (pFlags_present && rpc_read(conn, pFlags, sizeof(unsigned int)) < 0) ||
      rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
      rpc_read_end(conn) < 0)
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  return return_value;
}

CUresult cuTexRefCreate(CUtexref *pTexRef) {
  lupine_route route = lupine_route_for_default();
  CUresult return_value;
  if (lupine_route_is_local(route))
    return lupine_call_real_cuda_fn(LUPINE_REAL_CUDA_SYMBOL("cuTexRefCreate"),
                                    pTexRef);
  conn_t *conn = lupine_route_remote_conn(route);
  uint8_t pTexRef_present = pTexRef != nullptr;
  if (lupine_prepare_rpc(conn) < 0 ||
      rpc_write_start_request_on(conn, RPC_cuTexRefCreate, RPC_SCOPE_NONE) <
          0 ||
      rpc_write(conn, &pTexRef_present, sizeof(uint8_t)) < 0 ||
      (pTexRef_present && rpc_write(conn, pTexRef, sizeof(CUtexref)) < 0) ||
      rpc_wait_for_response(conn) < 0 ||
      (pTexRef_present && rpc_read(conn, pTexRef, sizeof(CUtexref)) < 0) ||
      rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
      rpc_read_end(conn) < 0)
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  return return_value;
}

CUresult cuTexRefDestroy(CUtexref hTexRef) {
  lupine_route route = lupine_route_for_default();
  CUresult return_value;
  if (lupine_route_is_local(route))
    return lupine_call_real_cuda_fn(LUPINE_REAL_CUDA_SYMBOL("cuTexRefDestroy"),
                                    hTexRef);
  conn_t *conn = lupine_route_remote_conn(route);
  if (lupine_prepare_rpc(conn) < 0 ||
      rpc_write_start_request_on(conn, RPC_cuTexRefDestroy, RPC_SCOPE_NONE) <
          0 ||
      rpc_write(conn, &hTexRef, sizeof(CUtexref)) < 0 ||
      rpc_write_end(conn) < 0)
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  return_value = CUDA_SUCCESS;
  return return_value;
}

CUresult cuSurfRefSetArray(CUsurfref hSurfRef, CUarray hArray,
                           unsigned int Flags) {
  lupine_route route = lupine_route_for_default();
  CUresult return_value;
  if (lupine_route_is_local(route))
    return lupine_call_real_cuda_fn(
        LUPINE_REAL_CUDA_SYMBOL("cuSurfRefSetArray"), hSurfRef, hArray, Flags);
  conn_t *conn = lupine_route_remote_conn(route);
  if (lupine_prepare_rpc(conn) < 0 ||
      rpc_write_start_request_on(conn, RPC_cuSurfRefSetArray, RPC_SCOPE_NONE) <
          0 ||
      rpc_write(conn, &hSurfRef, sizeof(CUsurfref)) < 0 ||
      rpc_write(conn, &hArray, sizeof(CUarray)) < 0 ||
      rpc_write(conn, &Flags, sizeof(unsigned int)) < 0 ||
      rpc_write_end(conn) < 0)
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  return_value = CUDA_SUCCESS;
  return return_value;
}

CUresult cuSurfRefGetArray(CUarray *phArray, CUsurfref hSurfRef) {
  lupine_route route = lupine_route_for_default();
  CUresult return_value;
  if (lupine_route_is_local(route))
    return lupine_call_real_cuda_fn(
        LUPINE_REAL_CUDA_SYMBOL("cuSurfRefGetArray"), phArray, hSurfRef);
  conn_t *conn = lupine_route_remote_conn(route);
  uint8_t phArray_present = phArray != nullptr;
  if (lupine_prepare_rpc(conn) < 0 ||
      rpc_write_start_request_on(conn, RPC_cuSurfRefGetArray, RPC_SCOPE_NONE) <
          0 ||
      rpc_write(conn, &phArray_present, sizeof(uint8_t)) < 0 ||
      (phArray_present && rpc_write(conn, phArray, sizeof(CUarray)) < 0) ||
      rpc_write(conn, &hSurfRef, sizeof(CUsurfref)) < 0 ||
      rpc_wait_for_response(conn) < 0 ||
      (phArray_present && rpc_read(conn, phArray, sizeof(CUarray)) < 0) ||
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
  if (lupine_route_is_local(route))
    return lupine_call_real_cuda_fn(
        LUPINE_REAL_CUDA_SYMBOL("cuTexObjectCreate"), pTexObject, pResDesc,
        pTexDesc, pResViewDesc);
  conn_t *conn = lupine_route_remote_conn(route);
  uint8_t pTexObject_present = pTexObject != nullptr;
  uint8_t pResDesc_present = pResDesc != nullptr;
  uint8_t pTexDesc_present = pTexDesc != nullptr;
  uint8_t pResViewDesc_present = pResViewDesc != nullptr;
  if (lupine_prepare_rpc(conn) < 0 ||
      rpc_write_start_request_on(conn, RPC_cuTexObjectCreate, RPC_SCOPE_NONE) <
          0 ||
      rpc_write(conn, &pTexObject_present, sizeof(uint8_t)) < 0 ||
      (pTexObject_present &&
       rpc_write(conn, pTexObject, sizeof(CUtexObject)) < 0) ||
      rpc_write(conn, &pResDesc_present, sizeof(uint8_t)) < 0 ||
      (pResDesc_present &&
       rpc_write(conn, pResDesc, sizeof(const CUDA_RESOURCE_DESC)) < 0) ||
      rpc_write(conn, &pTexDesc_present, sizeof(uint8_t)) < 0 ||
      (pTexDesc_present &&
       rpc_write(conn, pTexDesc, sizeof(const CUDA_TEXTURE_DESC)) < 0) ||
      rpc_write(conn, &pResViewDesc_present, sizeof(uint8_t)) < 0 ||
      (pResViewDesc_present &&
       rpc_write(conn, pResViewDesc, sizeof(const CUDA_RESOURCE_VIEW_DESC)) <
           0) ||
      rpc_wait_for_response(conn) < 0 ||
      (pTexObject_present &&
       rpc_read(conn, pTexObject, sizeof(CUtexObject)) < 0) ||
      rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
      rpc_read_end(conn) < 0)
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  return return_value;
}

CUresult cuTexObjectDestroy(CUtexObject texObject) {
  lupine_route route = lupine_route_for_default();
  CUresult return_value;
  if (lupine_route_is_local(route))
    return lupine_call_real_cuda_fn(
        LUPINE_REAL_CUDA_SYMBOL("cuTexObjectDestroy"), texObject);
  conn_t *conn = lupine_route_remote_conn(route);
  if (lupine_prepare_rpc(conn) < 0 ||
      rpc_write_start_request_on(conn, RPC_cuTexObjectDestroy, RPC_SCOPE_NONE) <
          0 ||
      rpc_write(conn, &texObject, sizeof(CUtexObject)) < 0 ||
      rpc_write_end(conn) < 0)
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  return_value = CUDA_SUCCESS;
  return return_value;
}

CUresult cuTexObjectGetResourceDesc(CUDA_RESOURCE_DESC *pResDesc,
                                    CUtexObject texObject) {
  lupine_route route = lupine_route_for_default();
  CUresult return_value;
  if (lupine_route_is_local(route))
    return lupine_call_real_cuda_fn(
        LUPINE_REAL_CUDA_SYMBOL("cuTexObjectGetResourceDesc"), pResDesc,
        texObject);
  conn_t *conn = lupine_route_remote_conn(route);
  uint8_t pResDesc_present = pResDesc != nullptr;
  if (lupine_prepare_rpc(conn) < 0 ||
      rpc_write_start_request_on(conn, RPC_cuTexObjectGetResourceDesc,
                                 RPC_SCOPE_NONE) < 0 ||
      rpc_write(conn, &pResDesc_present, sizeof(uint8_t)) < 0 ||
      (pResDesc_present &&
       rpc_write(conn, pResDesc, sizeof(CUDA_RESOURCE_DESC)) < 0) ||
      rpc_write(conn, &texObject, sizeof(CUtexObject)) < 0 ||
      rpc_wait_for_response(conn) < 0 ||
      (pResDesc_present &&
       rpc_read(conn, pResDesc, sizeof(CUDA_RESOURCE_DESC)) < 0) ||
      rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
      rpc_read_end(conn) < 0)
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  return return_value;
}

CUresult cuTexObjectGetTextureDesc(CUDA_TEXTURE_DESC *pTexDesc,
                                   CUtexObject texObject) {
  lupine_route route = lupine_route_for_default();
  CUresult return_value;
  if (lupine_route_is_local(route))
    return lupine_call_real_cuda_fn(
        LUPINE_REAL_CUDA_SYMBOL("cuTexObjectGetTextureDesc"), pTexDesc,
        texObject);
  conn_t *conn = lupine_route_remote_conn(route);
  uint8_t pTexDesc_present = pTexDesc != nullptr;
  if (lupine_prepare_rpc(conn) < 0 ||
      rpc_write_start_request_on(conn, RPC_cuTexObjectGetTextureDesc,
                                 RPC_SCOPE_NONE) < 0 ||
      rpc_write(conn, &pTexDesc_present, sizeof(uint8_t)) < 0 ||
      (pTexDesc_present &&
       rpc_write(conn, pTexDesc, sizeof(CUDA_TEXTURE_DESC)) < 0) ||
      rpc_write(conn, &texObject, sizeof(CUtexObject)) < 0 ||
      rpc_wait_for_response(conn) < 0 ||
      (pTexDesc_present &&
       rpc_read(conn, pTexDesc, sizeof(CUDA_TEXTURE_DESC)) < 0) ||
      rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
      rpc_read_end(conn) < 0)
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  return return_value;
}

CUresult cuTexObjectGetResourceViewDesc(CUDA_RESOURCE_VIEW_DESC *pResViewDesc,
                                        CUtexObject texObject) {
  lupine_route route = lupine_route_for_default();
  CUresult return_value;
  if (lupine_route_is_local(route))
    return lupine_call_real_cuda_fn(
        LUPINE_REAL_CUDA_SYMBOL("cuTexObjectGetResourceViewDesc"), pResViewDesc,
        texObject);
  conn_t *conn = lupine_route_remote_conn(route);
  uint8_t pResViewDesc_present = pResViewDesc != nullptr;
  if (lupine_prepare_rpc(conn) < 0 ||
      rpc_write_start_request_on(conn, RPC_cuTexObjectGetResourceViewDesc,
                                 RPC_SCOPE_NONE) < 0 ||
      rpc_write(conn, &pResViewDesc_present, sizeof(uint8_t)) < 0 ||
      (pResViewDesc_present &&
       rpc_write(conn, pResViewDesc, sizeof(CUDA_RESOURCE_VIEW_DESC)) < 0) ||
      rpc_write(conn, &texObject, sizeof(CUtexObject)) < 0 ||
      rpc_wait_for_response(conn) < 0 ||
      (pResViewDesc_present &&
       rpc_read(conn, pResViewDesc, sizeof(CUDA_RESOURCE_VIEW_DESC)) < 0) ||
      rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
      rpc_read_end(conn) < 0)
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  return return_value;
}

CUresult cuSurfObjectCreate(CUsurfObject *pSurfObject,
                            const CUDA_RESOURCE_DESC *pResDesc) {
  lupine_route route = lupine_route_for_default();
  CUresult return_value;
  if (lupine_route_is_local(route))
    return lupine_call_real_cuda_fn(
        LUPINE_REAL_CUDA_SYMBOL("cuSurfObjectCreate"), pSurfObject, pResDesc);
  conn_t *conn = lupine_route_remote_conn(route);
  uint8_t pSurfObject_present = pSurfObject != nullptr;
  uint8_t pResDesc_present = pResDesc != nullptr;
  if (lupine_prepare_rpc(conn) < 0 ||
      rpc_write_start_request_on(conn, RPC_cuSurfObjectCreate, RPC_SCOPE_NONE) <
          0 ||
      rpc_write(conn, &pSurfObject_present, sizeof(uint8_t)) < 0 ||
      (pSurfObject_present &&
       rpc_write(conn, pSurfObject, sizeof(CUsurfObject)) < 0) ||
      rpc_write(conn, &pResDesc_present, sizeof(uint8_t)) < 0 ||
      (pResDesc_present &&
       rpc_write(conn, pResDesc, sizeof(const CUDA_RESOURCE_DESC)) < 0) ||
      rpc_wait_for_response(conn) < 0 ||
      (pSurfObject_present &&
       rpc_read(conn, pSurfObject, sizeof(CUsurfObject)) < 0) ||
      rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
      rpc_read_end(conn) < 0)
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  return return_value;
}

CUresult cuSurfObjectDestroy(CUsurfObject surfObject) {
  lupine_route route = lupine_route_for_default();
  CUresult return_value;
  if (lupine_route_is_local(route))
    return lupine_call_real_cuda_fn(
        LUPINE_REAL_CUDA_SYMBOL("cuSurfObjectDestroy"), surfObject);
  conn_t *conn = lupine_route_remote_conn(route);
  if (lupine_prepare_rpc(conn) < 0 ||
      rpc_write_start_request_on(conn, RPC_cuSurfObjectDestroy,
                                 RPC_SCOPE_NONE) < 0 ||
      rpc_write(conn, &surfObject, sizeof(CUsurfObject)) < 0 ||
      rpc_write_end(conn) < 0)
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  return_value = CUDA_SUCCESS;
  return return_value;
}

CUresult cuSurfObjectGetResourceDesc(CUDA_RESOURCE_DESC *pResDesc,
                                     CUsurfObject surfObject) {
  lupine_route route = lupine_route_for_default();
  CUresult return_value;
  if (lupine_route_is_local(route))
    return lupine_call_real_cuda_fn(
        LUPINE_REAL_CUDA_SYMBOL("cuSurfObjectGetResourceDesc"), pResDesc,
        surfObject);
  conn_t *conn = lupine_route_remote_conn(route);
  uint8_t pResDesc_present = pResDesc != nullptr;
  if (lupine_prepare_rpc(conn) < 0 ||
      rpc_write_start_request_on(conn, RPC_cuSurfObjectGetResourceDesc,
                                 RPC_SCOPE_NONE) < 0 ||
      rpc_write(conn, &pResDesc_present, sizeof(uint8_t)) < 0 ||
      (pResDesc_present &&
       rpc_write(conn, pResDesc, sizeof(CUDA_RESOURCE_DESC)) < 0) ||
      rpc_write(conn, &surfObject, sizeof(CUsurfObject)) < 0 ||
      rpc_wait_for_response(conn) < 0 ||
      (pResDesc_present &&
       rpc_read(conn, pResDesc, sizeof(CUDA_RESOURCE_DESC)) < 0) ||
      rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
      rpc_read_end(conn) < 0)
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  return return_value;
}

#if CUDA_VERSION >= 12000
CUresult cuTensorMapEncodeTiled(
    CUtensorMap *tensorMap, CUtensorMapDataType tensorDataType,
    cuuint32_t tensorRank, void *globalAddress, const cuuint64_t *globalDim,
    const cuuint64_t *globalStrides, const cuuint32_t *boxDim,
    const cuuint32_t *elementStrides, CUtensorMapInterleave interleave,
    CUtensorMapSwizzle swizzle, CUtensorMapL2promotion l2Promotion,
    CUtensorMapFloatOOBfill oobFill) {
  lupine_route route =
      lupine_route_for_deviceptr(reinterpret_cast<CUdeviceptr>(globalAddress));
  if (tensorMap == nullptr || tensorRank == 0 || tensorRank > 5 ||
      globalAddress == nullptr || globalDim == nullptr ||
      globalStrides == nullptr || boxDim == nullptr ||
      elementStrides == nullptr)
    return CUDA_ERROR_INVALID_VALUE;
  std::string key;
  uint64_t generation = 0;
  if (!lupine_route_is_local(route) &&
      lupine_tensor_map_cache_lookup(
          route, tensorDataType, tensorRank, globalAddress, globalDim,
          globalStrides, boxDim, elementStrides, interleave, swizzle,
          l2Promotion, oobFill, &key, &generation, tensorMap))
    return CUDA_SUCCESS;
  CUresult return_value;
  if (lupine_route_is_local(route)) {
    return_value = lupine_call_real_cuda_fn(
        LUPINE_REAL_CUDA_SYMBOL("cuTensorMapEncodeTiled"), tensorMap,
        tensorDataType, tensorRank, globalAddress, globalDim, globalStrides,
        boxDim, elementStrides, interleave, swizzle, l2Promotion, oobFill);
    if (return_value == CUDA_SUCCESS && !lupine_route_is_local(route))
      lupine_tensor_map_cache_insert(std::move(key), generation, *tensorMap);
    return return_value;
  }
  conn_t *conn = lupine_route_remote_conn(route);
  uint8_t tensorMap_present = tensorMap != nullptr;
  uint8_t globalDim_null = globalDim == nullptr ? 1 : 0;
  uint8_t globalStrides_null = globalStrides == nullptr ? 1 : 0;
  uint8_t boxDim_null = boxDim == nullptr ? 1 : 0;
  uint8_t elementStrides_null = elementStrides == nullptr ? 1 : 0;
  if (lupine_prepare_rpc(conn) < 0 ||
      rpc_write_start_request_on(conn, RPC_cuTensorMapEncodeTiled,
                                 RPC_SCOPE_NONE) < 0 ||
      rpc_write(conn, &tensorMap_present, sizeof(uint8_t)) < 0 ||
      (tensorMap_present &&
       rpc_write(conn, tensorMap, sizeof(CUtensorMap)) < 0) ||
      rpc_write(conn, &tensorDataType, sizeof(CUtensorMapDataType)) < 0 ||
      rpc_write(conn, &tensorRank, sizeof(cuuint32_t)) < 0 ||
      rpc_write(conn, &globalAddress, sizeof(void *)) < 0 ||
      rpc_write(conn, &globalDim_null, sizeof(uint8_t)) < 0 ||
      (!globalDim_null && tensorRank * sizeof(const cuuint64_t) != 0 &&
       rpc_write(conn, globalDim, tensorRank * sizeof(const cuuint64_t)) < 0) ||
      rpc_write(conn, &globalStrides_null, sizeof(uint8_t)) < 0 ||
      (!globalStrides_null &&
       (tensorRank <= 1 ? 0 : (tensorRank - 1) * sizeof(const cuuint64_t)) !=
           0 &&
       rpc_write(conn, globalStrides,
                 (tensorRank <= 1
                      ? 0
                      : (tensorRank - 1) * sizeof(const cuuint64_t))) < 0) ||
      rpc_write(conn, &boxDim_null, sizeof(uint8_t)) < 0 ||
      (!boxDim_null && tensorRank * sizeof(const cuuint32_t) != 0 &&
       rpc_write(conn, boxDim, tensorRank * sizeof(const cuuint32_t)) < 0) ||
      rpc_write(conn, &elementStrides_null, sizeof(uint8_t)) < 0 ||
      (!elementStrides_null && tensorRank * sizeof(const cuuint32_t) != 0 &&
       rpc_write(conn, elementStrides, tensorRank * sizeof(const cuuint32_t)) <
           0) ||
      rpc_write(conn, &interleave, sizeof(CUtensorMapInterleave)) < 0 ||
      rpc_write(conn, &swizzle, sizeof(CUtensorMapSwizzle)) < 0 ||
      rpc_write(conn, &l2Promotion, sizeof(CUtensorMapL2promotion)) < 0 ||
      rpc_write(conn, &oobFill, sizeof(CUtensorMapFloatOOBfill)) < 0 ||
      rpc_wait_for_response(conn) < 0 ||
      (tensorMap_present &&
       rpc_read(conn, tensorMap, sizeof(CUtensorMap)) < 0) ||
      rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
      rpc_read_end(conn) < 0)
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  if (return_value == CUDA_SUCCESS && !lupine_route_is_local(route))
    lupine_tensor_map_cache_insert(std::move(key), generation, *tensorMap);
  return return_value;
}

#endif

#if CUDA_VERSION >= 12000
CUresult cuTensorMapEncodeIm2col(
    CUtensorMap *tensorMap, CUtensorMapDataType tensorDataType,
    cuuint32_t tensorRank, void *globalAddress, const cuuint64_t *globalDim,
    const cuuint64_t *globalStrides, const int *pixelBoxLowerCorner,
    const int *pixelBoxUpperCorner, cuuint32_t channelsPerPixel,
    cuuint32_t pixelsPerColumn, const cuuint32_t *elementStrides,
    CUtensorMapInterleave interleave, CUtensorMapSwizzle swizzle,
    CUtensorMapL2promotion l2Promotion, CUtensorMapFloatOOBfill oobFill) {
  lupine_route route =
      lupine_route_for_deviceptr(reinterpret_cast<CUdeviceptr>(globalAddress));
  if (tensorMap == nullptr || tensorRank < 3 || tensorRank > 5 ||
      globalAddress == nullptr || globalDim == nullptr ||
      globalStrides == nullptr || pixelBoxLowerCorner == nullptr ||
      pixelBoxUpperCorner == nullptr || elementStrides == nullptr)
    return CUDA_ERROR_INVALID_VALUE;
  CUresult return_value;
  if (lupine_route_is_local(route))
    return lupine_call_real_cuda_fn(
        LUPINE_REAL_CUDA_SYMBOL("cuTensorMapEncodeIm2col"), tensorMap,
        tensorDataType, tensorRank, globalAddress, globalDim, globalStrides,
        pixelBoxLowerCorner, pixelBoxUpperCorner, channelsPerPixel,
        pixelsPerColumn, elementStrides, interleave, swizzle, l2Promotion,
        oobFill);
  conn_t *conn = lupine_route_remote_conn(route);
  uint8_t tensorMap_present = tensorMap != nullptr;
  uint8_t globalDim_null = globalDim == nullptr ? 1 : 0;
  uint8_t globalStrides_null = globalStrides == nullptr ? 1 : 0;
  uint8_t pixelBoxLowerCorner_null = pixelBoxLowerCorner == nullptr ? 1 : 0;
  uint8_t pixelBoxUpperCorner_null = pixelBoxUpperCorner == nullptr ? 1 : 0;
  uint8_t elementStrides_null = elementStrides == nullptr ? 1 : 0;
  if (lupine_prepare_rpc(conn) < 0 ||
      rpc_write_start_request_on(conn, RPC_cuTensorMapEncodeIm2col,
                                 RPC_SCOPE_NONE) < 0 ||
      rpc_write(conn, &tensorMap_present, sizeof(uint8_t)) < 0 ||
      (tensorMap_present &&
       rpc_write(conn, tensorMap, sizeof(CUtensorMap)) < 0) ||
      rpc_write(conn, &tensorDataType, sizeof(CUtensorMapDataType)) < 0 ||
      rpc_write(conn, &tensorRank, sizeof(cuuint32_t)) < 0 ||
      rpc_write(conn, &globalAddress, sizeof(void *)) < 0 ||
      rpc_write(conn, &globalDim_null, sizeof(uint8_t)) < 0 ||
      (!globalDim_null && tensorRank * sizeof(const cuuint64_t) != 0 &&
       rpc_write(conn, globalDim, tensorRank * sizeof(const cuuint64_t)) < 0) ||
      rpc_write(conn, &globalStrides_null, sizeof(uint8_t)) < 0 ||
      (!globalStrides_null &&
       (tensorRank <= 1 ? 0 : (tensorRank - 1) * sizeof(const cuuint64_t)) !=
           0 &&
       rpc_write(conn, globalStrides,
                 (tensorRank <= 1
                      ? 0
                      : (tensorRank - 1) * sizeof(const cuuint64_t))) < 0) ||
      rpc_write(conn, &pixelBoxLowerCorner_null, sizeof(uint8_t)) < 0 ||
      (!pixelBoxLowerCorner_null &&
       (tensorRank <= 2 ? 0 : (tensorRank - 2) * sizeof(const int)) != 0 &&
       rpc_write(conn, pixelBoxLowerCorner,
                 (tensorRank <= 2 ? 0 : (tensorRank - 2) * sizeof(const int))) <
           0) ||
      rpc_write(conn, &pixelBoxUpperCorner_null, sizeof(uint8_t)) < 0 ||
      (!pixelBoxUpperCorner_null &&
       (tensorRank <= 2 ? 0 : (tensorRank - 2) * sizeof(const int)) != 0 &&
       rpc_write(conn, pixelBoxUpperCorner,
                 (tensorRank <= 2 ? 0 : (tensorRank - 2) * sizeof(const int))) <
           0) ||
      rpc_write(conn, &channelsPerPixel, sizeof(cuuint32_t)) < 0 ||
      rpc_write(conn, &pixelsPerColumn, sizeof(cuuint32_t)) < 0 ||
      rpc_write(conn, &elementStrides_null, sizeof(uint8_t)) < 0 ||
      (!elementStrides_null && tensorRank * sizeof(const cuuint32_t) != 0 &&
       rpc_write(conn, elementStrides, tensorRank * sizeof(const cuuint32_t)) <
           0) ||
      rpc_write(conn, &interleave, sizeof(CUtensorMapInterleave)) < 0 ||
      rpc_write(conn, &swizzle, sizeof(CUtensorMapSwizzle)) < 0 ||
      rpc_write(conn, &l2Promotion, sizeof(CUtensorMapL2promotion)) < 0 ||
      rpc_write(conn, &oobFill, sizeof(CUtensorMapFloatOOBfill)) < 0 ||
      rpc_wait_for_response(conn) < 0 ||
      (tensorMap_present &&
       rpc_read(conn, tensorMap, sizeof(CUtensorMap)) < 0) ||
      rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
      rpc_read_end(conn) < 0)
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  return return_value;
}

#endif

#if CUDA_VERSION >= 12000
CUresult cuTensorMapReplaceAddress(CUtensorMap *tensorMap,
                                   void *globalAddress) {
  lupine_route route =
      lupine_route_for_deviceptr(reinterpret_cast<CUdeviceptr>(globalAddress));
  CUresult return_value;
  if (lupine_route_is_local(route))
    return lupine_call_real_cuda_fn(
        LUPINE_REAL_CUDA_SYMBOL("cuTensorMapReplaceAddress"), tensorMap,
        globalAddress);
  conn_t *conn = lupine_route_remote_conn(route);
  uint8_t tensorMap_present = tensorMap != nullptr;
  if (lupine_prepare_rpc(conn) < 0 ||
      rpc_write_start_request_on(conn, RPC_cuTensorMapReplaceAddress,
                                 RPC_SCOPE_NONE) < 0 ||
      rpc_write(conn, &tensorMap_present, sizeof(uint8_t)) < 0 ||
      (tensorMap_present &&
       rpc_write(conn, tensorMap, sizeof(CUtensorMap)) < 0) ||
      rpc_write(conn, &globalAddress, sizeof(void *)) < 0 ||
      rpc_wait_for_response(conn) < 0 ||
      (tensorMap_present &&
       rpc_read(conn, tensorMap, sizeof(CUtensorMap)) < 0) ||
      rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
      rpc_read_end(conn) < 0)
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  return return_value;
}

#endif

CUresult cuDeviceCanAccessPeer(int *canAccessPeer, CUdevice dev,
                               CUdevice peerDev) {
  lupine_route route = lupine_route_for_default();
  if (canAccessPeer == nullptr)
    return CUDA_ERROR_INVALID_VALUE;
  lupine_route peer_route = lupine_route_for_device(&peerDev);
  route = lupine_route_for_device(&dev);
  if (route.kind == LUPINE_ROUTE_UNKNOWN_DEVICE ||
      peer_route.kind == LUPINE_ROUTE_UNKNOWN_DEVICE)
    return CUDA_ERROR_INVALID_DEVICE;
  if (!lupine_routes_share_server(route, peer_route)) {
    *canAccessPeer = 0;
    return CUDA_SUCCESS;
  }
  CUresult return_value;
  if (lupine_route_is_local(route))
    return lupine_call_real_cuda_fn(
        LUPINE_REAL_CUDA_SYMBOL("cuDeviceCanAccessPeer"), canAccessPeer, dev,
        peerDev);
  conn_t *conn = lupine_route_remote_conn(route);
  uint8_t canAccessPeer_present = canAccessPeer != nullptr;
  if (lupine_prepare_rpc(conn) < 0 ||
      rpc_write_start_request_on(conn, RPC_cuDeviceCanAccessPeer,
                                 RPC_SCOPE_NONE) < 0 ||
      rpc_write(conn, &canAccessPeer_present, sizeof(uint8_t)) < 0 ||
      (canAccessPeer_present &&
       rpc_write(conn, canAccessPeer, sizeof(int)) < 0) ||
      rpc_write(conn, &dev, sizeof(CUdevice)) < 0 ||
      rpc_write(conn, &peerDev, sizeof(CUdevice)) < 0 ||
      rpc_wait_for_response(conn) < 0 ||
      (canAccessPeer_present &&
       rpc_read(conn, canAccessPeer, sizeof(int)) < 0) ||
      rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
      rpc_read_end(conn) < 0)
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  return return_value;
}

CUresult cuCtxEnablePeerAccess(CUcontext peerContext, unsigned int Flags) {
  lupine_route route = lupine_route_for_current_context();
  if (lupine_route_identity(route) !=
      lupine_route_identity(lupine_route_for_context(peerContext)))
    return CUDA_ERROR_PEER_ACCESS_UNSUPPORTED;
  CUresult return_value;
  if (lupine_route_is_local(route))
    return lupine_call_real_cuda_fn(
        LUPINE_REAL_CUDA_SYMBOL("cuCtxEnablePeerAccess"), peerContext, Flags);
  conn_t *conn = lupine_route_remote_conn(route);
  if (lupine_prepare_rpc(conn) < 0 ||
      rpc_write_start_request(conn, RPC_cuCtxEnablePeerAccess) < 0 ||
      rpc_write(conn, &peerContext, sizeof(CUcontext)) < 0 ||
      rpc_write(conn, &Flags, sizeof(unsigned int)) < 0 ||
      rpc_wait_for_response(conn) < 0 ||
      rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
      rpc_read_end(conn) < 0)
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  return return_value;
}

CUresult cuCtxDisablePeerAccess(CUcontext peerContext) {
  lupine_route route = lupine_route_for_current_context();
  if (lupine_route_identity(route) !=
      lupine_route_identity(lupine_route_for_context(peerContext)))
    return CUDA_ERROR_PEER_ACCESS_NOT_ENABLED;
  CUresult return_value;
  if (lupine_route_is_local(route))
    return lupine_call_real_cuda_fn(
        LUPINE_REAL_CUDA_SYMBOL("cuCtxDisablePeerAccess"), peerContext);
  conn_t *conn = lupine_route_remote_conn(route);
  if (lupine_prepare_rpc(conn) < 0 ||
      rpc_write_start_request(conn, RPC_cuCtxDisablePeerAccess) < 0 ||
      rpc_write(conn, &peerContext, sizeof(CUcontext)) < 0 ||
      rpc_wait_for_response(conn) < 0 ||
      rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
      rpc_read_end(conn) < 0)
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  return return_value;
}

CUresult cuDeviceGetP2PAttribute(int *value, CUdevice_P2PAttribute attrib,
                                 CUdevice srcDevice, CUdevice dstDevice) {
  lupine_route route = lupine_route_for_default();
  if (value == nullptr)
    return CUDA_ERROR_INVALID_VALUE;
  if (srcDevice == dstDevice)
    return CUDA_ERROR_INVALID_DEVICE;
  lupine_route dst_route = lupine_route_for_device(&dstDevice);
  route = lupine_route_for_device(&srcDevice);
  if (route.kind == LUPINE_ROUTE_UNKNOWN_DEVICE ||
      dst_route.kind == LUPINE_ROUTE_UNKNOWN_DEVICE)
    return CUDA_ERROR_INVALID_DEVICE;
  if (route.kind == LUPINE_ROUTE_INVALID ||
      dst_route.kind == LUPINE_ROUTE_INVALID)
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  if (!lupine_routes_share_server(route, dst_route)) {
    *value = 0;
    return CUDA_SUCCESS;
  }
  CUresult return_value;
  if (lupine_route_is_local(route))
    return lupine_call_real_cuda_fn(
        LUPINE_REAL_CUDA_SYMBOL("cuDeviceGetP2PAttribute"), value, attrib,
        srcDevice, dstDevice);
  conn_t *conn = lupine_route_remote_conn(route);
  uint8_t value_present = value != nullptr;
  if (lupine_prepare_rpc(conn) < 0 ||
      rpc_write_start_request_on(conn, RPC_cuDeviceGetP2PAttribute,
                                 RPC_SCOPE_NONE) < 0 ||
      rpc_write(conn, &value_present, sizeof(uint8_t)) < 0 ||
      (value_present && rpc_write(conn, value, sizeof(int)) < 0) ||
      rpc_write(conn, &attrib, sizeof(CUdevice_P2PAttribute)) < 0 ||
      rpc_write(conn, &srcDevice, sizeof(CUdevice)) < 0 ||
      rpc_write(conn, &dstDevice, sizeof(CUdevice)) < 0 ||
      rpc_wait_for_response(conn) < 0 ||
      (value_present && rpc_read(conn, value, sizeof(int)) < 0) ||
      rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
      rpc_read_end(conn) < 0)
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  return return_value;
}

CUresult cuGraphicsUnregisterResource(CUgraphicsResource resource) {
  lupine_route route = lupine_route_for_default();
  CUresult return_value;
  if (lupine_route_is_local(route))
    return lupine_call_real_cuda_fn(
        LUPINE_REAL_CUDA_SYMBOL("cuGraphicsUnregisterResource"), resource);
  conn_t *conn = lupine_route_remote_conn(route);
  if (lupine_prepare_rpc(conn) < 0 ||
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
  if (lupine_route_is_local(route))
    return lupine_call_real_cuda_fn(
        LUPINE_REAL_CUDA_SYMBOL("cuGraphicsSubResourceGetMappedArray"), pArray,
        resource, arrayIndex, mipLevel);
  conn_t *conn = lupine_route_remote_conn(route);
  uint8_t pArray_present = pArray != nullptr;
  if (lupine_prepare_rpc(conn) < 0 ||
      rpc_write_start_request_on(conn, RPC_cuGraphicsSubResourceGetMappedArray,
                                 RPC_SCOPE_NONE) < 0 ||
      rpc_write(conn, &pArray_present, sizeof(uint8_t)) < 0 ||
      (pArray_present && rpc_write(conn, pArray, sizeof(CUarray)) < 0) ||
      rpc_write(conn, &resource, sizeof(CUgraphicsResource)) < 0 ||
      rpc_write(conn, &arrayIndex, sizeof(unsigned int)) < 0 ||
      rpc_write(conn, &mipLevel, sizeof(unsigned int)) < 0 ||
      rpc_wait_for_response(conn) < 0 ||
      (pArray_present && rpc_read(conn, pArray, sizeof(CUarray)) < 0) ||
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
  if (lupine_route_is_local(route))
    return lupine_call_real_cuda_fn(
        LUPINE_REAL_CUDA_SYMBOL("cuGraphicsResourceGetMappedMipmappedArray"),
        pMipmappedArray, resource);
  conn_t *conn = lupine_route_remote_conn(route);
  uint8_t pMipmappedArray_present = pMipmappedArray != nullptr;
  if (lupine_prepare_rpc(conn) < 0 ||
      rpc_write_start_request_on(conn,
                                 RPC_cuGraphicsResourceGetMappedMipmappedArray,
                                 RPC_SCOPE_NONE) < 0 ||
      rpc_write(conn, &pMipmappedArray_present, sizeof(uint8_t)) < 0 ||
      (pMipmappedArray_present &&
       rpc_write(conn, pMipmappedArray, sizeof(CUmipmappedArray)) < 0) ||
      rpc_write(conn, &resource, sizeof(CUgraphicsResource)) < 0 ||
      rpc_wait_for_response(conn) < 0 ||
      (pMipmappedArray_present &&
       rpc_read(conn, pMipmappedArray, sizeof(CUmipmappedArray)) < 0) ||
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
  if (lupine_route_is_local(route))
    return lupine_call_real_cuda_fn(
        LUPINE_REAL_CUDA_SYMBOL("cuGraphicsResourceGetMappedPointer_v2"),
        pDevPtr, pSize, resource);
  conn_t *conn = lupine_route_remote_conn(route);
  uint8_t pDevPtr_present = pDevPtr != nullptr;
  uint8_t pSize_present = pSize != nullptr;
  if (lupine_prepare_rpc(conn) < 0 ||
      rpc_write_start_request_on(conn,
                                 RPC_cuGraphicsResourceGetMappedPointer_v2,
                                 RPC_SCOPE_NONE) < 0 ||
      rpc_write(conn, &pDevPtr_present, sizeof(uint8_t)) < 0 ||
      (pDevPtr_present && rpc_write(conn, pDevPtr, sizeof(CUdeviceptr)) < 0) ||
      rpc_write(conn, &pSize_present, sizeof(uint8_t)) < 0 ||
      (pSize_present && rpc_write(conn, pSize, sizeof(size_t)) < 0) ||
      rpc_write(conn, &resource, sizeof(CUgraphicsResource)) < 0 ||
      rpc_wait_for_response(conn) < 0 ||
      (pDevPtr_present && rpc_read(conn, pDevPtr, sizeof(CUdeviceptr)) < 0) ||
      (pSize_present && rpc_read(conn, pSize, sizeof(size_t)) < 0) ||
      rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
      rpc_read_end(conn) < 0)
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  return return_value;
}

CUresult cuGraphicsResourceSetMapFlags_v2(CUgraphicsResource resource,
                                          unsigned int flags) {
  lupine_route route = lupine_route_for_default();
  CUresult return_value;
  if (lupine_route_is_local(route))
    return lupine_call_real_cuda_fn(
        LUPINE_REAL_CUDA_SYMBOL("cuGraphicsResourceSetMapFlags_v2"), resource,
        flags);
  conn_t *conn = lupine_route_remote_conn(route);
  if (lupine_prepare_rpc(conn) < 0 ||
      rpc_write_start_request_on(conn, RPC_cuGraphicsResourceSetMapFlags_v2,
                                 RPC_SCOPE_NONE) < 0 ||
      rpc_write(conn, &resource, sizeof(CUgraphicsResource)) < 0 ||
      rpc_write(conn, &flags, sizeof(unsigned int)) < 0 ||
      rpc_write_end(conn) < 0)
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  return_value = CUDA_SUCCESS;
  return return_value;
}

CUresult cuGraphicsMapResources(unsigned int count,
                                CUgraphicsResource *resources,
                                CUstream hStream) {
  lupine_route route = (hStream != nullptr ? lupine_route_for_stream(hStream)
                                           : lupine_route_for_default());
  CUresult return_value;
  if (lupine_route_is_local(route))
    return lupine_call_real_cuda_fn(
        LUPINE_REAL_CUDA_SYMBOL("cuGraphicsMapResources"), count, resources,
        hStream);
  conn_t *conn = lupine_route_remote_conn(route);
  uint8_t resources_null = resources == nullptr ? 1 : 0;
  if (lupine_prepare_rpc(conn) < 0 ||
      lupine_rpc_start_on_stream(conn, RPC_cuGraphicsMapResources, hStream) <
          0 ||
      rpc_write(conn, &count, sizeof(unsigned int)) < 0 ||
      rpc_write(conn, &resources_null, sizeof(uint8_t)) < 0 ||
      (!resources_null && count * sizeof(CUgraphicsResource) != 0 &&
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
  if (lupine_route_is_local(route))
    return lupine_call_real_cuda_fn(
        LUPINE_REAL_CUDA_SYMBOL("cuGraphicsUnmapResources"), count, resources,
        hStream);
  conn_t *conn = lupine_route_remote_conn(route);
  uint8_t resources_null = resources == nullptr ? 1 : 0;
  if (lupine_prepare_rpc(conn) < 0 ||
      lupine_rpc_start_on_stream(conn, RPC_cuGraphicsUnmapResources, hStream) <
          0 ||
      rpc_write(conn, &count, sizeof(unsigned int)) < 0 ||
      rpc_write(conn, &resources_null, sizeof(uint8_t)) < 0 ||
      (!resources_null && count * sizeof(CUgraphicsResource) != 0 &&
       rpc_write(conn, resources, count * sizeof(CUgraphicsResource)) < 0) ||
      rpc_write(conn, &hStream, sizeof(CUstream)) < 0 ||
      rpc_wait_for_response(conn) < 0 ||
      rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
      rpc_read_end(conn) < 0)
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  return return_value;
}

#if CUDA_VERSION >= 12010
CUresult cuCoredumpGetAttributeGlobal(CUcoredumpSettings attrib, void *value,
                                      size_t *size) {
  lupine_route route = lupine_route_for_default();
  CUresult return_value;
  if (lupine_route_is_local(route))
    return lupine_call_real_cuda_fn(
        LUPINE_REAL_CUDA_SYMBOL("cuCoredumpGetAttributeGlobal"), attrib, value,
        size);
  conn_t *conn = lupine_route_remote_conn(route);
  uint8_t size_present = size != nullptr;
  uint8_t value_null = value == nullptr ? 1 : 0;
  if (lupine_prepare_rpc(conn) < 0 ||
      rpc_write_start_request_on(conn, RPC_cuCoredumpGetAttributeGlobal,
                                 RPC_SCOPE_NONE) < 0 ||
      rpc_write(conn, &attrib, sizeof(CUcoredumpSettings)) < 0 ||
      rpc_write(conn, &size_present, sizeof(uint8_t)) < 0 ||
      (size_present && rpc_write(conn, size, sizeof(size_t)) < 0) ||
      rpc_write(conn, &value_null, sizeof(uint8_t)) < 0 ||
      rpc_wait_for_response(conn) < 0 ||
      (size_present && rpc_read(conn, size, sizeof(size_t)) < 0) ||
      (!value_null && (size != nullptr ? *size : 0) != 0 &&
       rpc_read(conn, value, (size != nullptr ? *size : 0)) < 0) ||
      rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
      rpc_read_end(conn) < 0)
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  return return_value;
}

#endif

#if CUDA_VERSION >= 12010
CUresult cuCoredumpSetAttributeGlobal(CUcoredumpSettings attrib, void *value,
                                      size_t *size) {
  lupine_route route = lupine_route_for_default();
  CUresult return_value;
  if (lupine_route_is_local(route))
    return lupine_call_real_cuda_fn(
        LUPINE_REAL_CUDA_SYMBOL("cuCoredumpSetAttributeGlobal"), attrib, value,
        size);
  conn_t *conn = lupine_route_remote_conn(route);
  uint8_t size_present = size != nullptr;
  uint8_t value_null = value == nullptr ? 1 : 0;
  if (lupine_prepare_rpc(conn) < 0 ||
      rpc_write_start_request_on(conn, RPC_cuCoredumpSetAttributeGlobal,
                                 RPC_SCOPE_NONE) < 0 ||
      rpc_write(conn, &attrib, sizeof(CUcoredumpSettings)) < 0 ||
      rpc_write(conn, &size_present, sizeof(uint8_t)) < 0 ||
      (size_present && rpc_write(conn, size, sizeof(size_t)) < 0) ||
      rpc_write(conn, &value_null, sizeof(uint8_t)) < 0 ||
      (!value_null && (size != nullptr ? *size : 0) != 0 &&
       rpc_write(conn, value, (size != nullptr ? *size : 0)) < 0) ||
      rpc_wait_for_response(conn) < 0 ||
      (size_present && rpc_read(conn, size, sizeof(size_t)) < 0) ||
      rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
      rpc_read_end(conn) < 0)
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  return return_value;
}

#endif

#if CUDA_VERSION >= 12040
CUresult cuGreenCtxCreate(CUgreenCtx *phCtx, CUdevResourceDesc desc,
                          CUdevice dev, unsigned int flags) {
  lupine_route route = lupine_route_for_device(&dev);
  if (route.kind == LUPINE_ROUTE_UNKNOWN_DEVICE)
    return CUDA_ERROR_INVALID_DEVICE;
  CUresult return_value;
  if (lupine_route_is_local(route))
    return lupine_call_real_cuda_fn(LUPINE_REAL_CUDA_SYMBOL("cuGreenCtxCreate"),
                                    phCtx, desc, dev, flags);
  conn_t *conn = lupine_route_remote_conn(route);
  uint8_t phCtx_present = phCtx != nullptr;
  if (lupine_prepare_rpc(conn) < 0 ||
      rpc_write_start_request_on(conn, RPC_cuGreenCtxCreate, RPC_SCOPE_NONE) <
          0 ||
      rpc_write(conn, &phCtx_present, sizeof(uint8_t)) < 0 ||
      (phCtx_present && rpc_write(conn, phCtx, sizeof(CUgreenCtx)) < 0) ||
      rpc_write(conn, &desc, sizeof(CUdevResourceDesc)) < 0 ||
      rpc_write(conn, &dev, sizeof(CUdevice)) < 0 ||
      rpc_write(conn, &flags, sizeof(unsigned int)) < 0 ||
      rpc_wait_for_response(conn) < 0 ||
      (phCtx_present && rpc_read(conn, phCtx, sizeof(CUgreenCtx)) < 0) ||
      rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
      rpc_read_end(conn) < 0)
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  return return_value;
}

#endif

#if CUDA_VERSION >= 12040
CUresult cuGreenCtxDestroy(CUgreenCtx hCtx) {
  lupine_route route = lupine_route_for_current_context();
  CUresult return_value;
  if (lupine_route_is_local(route)) {
    return_value = lupine_call_real_cuda_fn(
        LUPINE_REAL_CUDA_SYMBOL("cuGreenCtxDestroy"), hCtx);
    if (return_value == CUDA_SUCCESS)
      lupine_forget_destroyed_context(reinterpret_cast<CUcontext>(hCtx));
    return return_value;
  }
  conn_t *conn = lupine_route_remote_conn(route);
  if (lupine_prepare_rpc(conn) < 0 ||
      rpc_write_start_request(conn, RPC_cuGreenCtxDestroy) < 0 ||
      rpc_write(conn, &hCtx, sizeof(CUgreenCtx)) < 0 ||
      rpc_wait_for_response(conn) < 0 ||
      rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
      rpc_read_end(conn) < 0)
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  if (return_value == CUDA_SUCCESS)
    lupine_forget_destroyed_context(reinterpret_cast<CUcontext>(hCtx));
  return return_value;
}

#endif

#if CUDA_VERSION >= 12040
CUresult cuCtxFromGreenCtx(CUcontext *pContext, CUgreenCtx hCtx) {
  lupine_route route = lupine_route_for_current_context();
  CUresult return_value;
  if (lupine_route_is_local(route)) {
    return_value = lupine_call_real_cuda_fn(
        LUPINE_REAL_CUDA_SYMBOL("cuCtxFromGreenCtx"), pContext, hCtx);
    if (return_value == CUDA_SUCCESS && pContext != nullptr) {
      lupine_note_context_owner_route(*pContext, route);
    }
    if (return_value == CUDA_SUCCESS && pContext != nullptr)
      lupine_mark_context_green(*pContext);
    return return_value;
  }
  conn_t *conn = lupine_route_remote_conn(route);
  uint8_t pContext_present = pContext != nullptr;
  if (lupine_prepare_rpc(conn) < 0 ||
      rpc_write_start_request_on(conn, RPC_cuCtxFromGreenCtx, RPC_SCOPE_NONE) <
          0 ||
      rpc_write(conn, &pContext_present, sizeof(uint8_t)) < 0 ||
      (pContext_present && rpc_write(conn, pContext, sizeof(CUcontext)) < 0) ||
      rpc_write(conn, &hCtx, sizeof(CUgreenCtx)) < 0 ||
      rpc_wait_for_response(conn) < 0 ||
      (pContext_present && rpc_read(conn, pContext, sizeof(CUcontext)) < 0) ||
      rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
      rpc_read_end(conn) < 0)
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  if (return_value == CUDA_SUCCESS && pContext != nullptr) {
    lupine_note_context_owner_route(*pContext, route);
  }
  if (return_value == CUDA_SUCCESS && pContext != nullptr)
    lupine_mark_context_green(*pContext);
  return return_value;
}

#endif

#if CUDA_VERSION >= 12040
CUresult cuDeviceGetDevResource(CUdevice device, CUdevResource *resource,
                                CUdevResourceType type) {
  lupine_route route = lupine_route_for_device(&device);
  if (route.kind == LUPINE_ROUTE_UNKNOWN_DEVICE)
    return CUDA_ERROR_INVALID_DEVICE;
  CUresult return_value;
  if (lupine_route_is_local(route))
    return lupine_call_real_cuda_fn(
        LUPINE_REAL_CUDA_SYMBOL("cuDeviceGetDevResource"), device, resource,
        type);
  conn_t *conn = lupine_route_remote_conn(route);
  uint8_t resource_present = resource != nullptr;
  if (lupine_prepare_rpc(conn) < 0 ||
      rpc_write_start_request_on(conn, RPC_cuDeviceGetDevResource,
                                 RPC_SCOPE_NONE) < 0 ||
      rpc_write(conn, &device, sizeof(CUdevice)) < 0 ||
      rpc_write(conn, &resource_present, sizeof(uint8_t)) < 0 ||
      (resource_present &&
       rpc_write(conn, resource, sizeof(CUdevResource)) < 0) ||
      rpc_write(conn, &type, sizeof(CUdevResourceType)) < 0 ||
      rpc_wait_for_response(conn) < 0 ||
      (resource_present &&
       rpc_read(conn, resource, sizeof(CUdevResource)) < 0) ||
      rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
      rpc_read_end(conn) < 0)
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  return return_value;
}

#endif

#if CUDA_VERSION >= 12040
CUresult cuCtxGetDevResource(CUcontext hCtx, CUdevResource *resource,
                             CUdevResourceType type) {
  lupine_route route = lupine_route_for_context(hCtx);
  CUresult return_value;
  if (lupine_route_is_local(route))
    return lupine_call_real_cuda_fn(
        LUPINE_REAL_CUDA_SYMBOL("cuCtxGetDevResource"), hCtx, resource, type);
  conn_t *conn = lupine_route_remote_conn(route);
  uint8_t resource_present = resource != nullptr;
  if (lupine_prepare_rpc(conn) < 0 ||
      rpc_write_start_request_on(conn, RPC_cuCtxGetDevResource,
                                 RPC_SCOPE_NONE) < 0 ||
      rpc_write(conn, &hCtx, sizeof(CUcontext)) < 0 ||
      rpc_write(conn, &resource_present, sizeof(uint8_t)) < 0 ||
      (resource_present &&
       rpc_write(conn, resource, sizeof(CUdevResource)) < 0) ||
      rpc_write(conn, &type, sizeof(CUdevResourceType)) < 0 ||
      rpc_wait_for_response(conn) < 0 ||
      (resource_present &&
       rpc_read(conn, resource, sizeof(CUdevResource)) < 0) ||
      rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
      rpc_read_end(conn) < 0)
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  return return_value;
}

#endif

#if CUDA_VERSION >= 12040
CUresult cuGreenCtxGetDevResource(CUgreenCtx hCtx, CUdevResource *resource,
                                  CUdevResourceType type) {
  lupine_route route = lupine_route_for_current_context();
  CUresult return_value;
  if (lupine_route_is_local(route))
    return lupine_call_real_cuda_fn(
        LUPINE_REAL_CUDA_SYMBOL("cuGreenCtxGetDevResource"), hCtx, resource,
        type);
  conn_t *conn = lupine_route_remote_conn(route);
  uint8_t resource_present = resource != nullptr;
  if (lupine_prepare_rpc(conn) < 0 ||
      rpc_write_start_request_on(conn, RPC_cuGreenCtxGetDevResource,
                                 RPC_SCOPE_NONE) < 0 ||
      rpc_write(conn, &hCtx, sizeof(CUgreenCtx)) < 0 ||
      rpc_write(conn, &resource_present, sizeof(uint8_t)) < 0 ||
      (resource_present &&
       rpc_write(conn, resource, sizeof(CUdevResource)) < 0) ||
      rpc_write(conn, &type, sizeof(CUdevResourceType)) < 0 ||
      rpc_wait_for_response(conn) < 0 ||
      (resource_present &&
       rpc_read(conn, resource, sizeof(CUdevResource)) < 0) ||
      rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
      rpc_read_end(conn) < 0)
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  return return_value;
}

#endif

#if CUDA_VERSION >= 12040
CUresult cuDevSmResourceSplitByCount(
    CUdevResource *result, unsigned int *nbGroups, const CUdevResource *input,
    CUdevResource *remainder, unsigned int flags, unsigned int minCount) {
  lupine_route route = lupine_route_for_current_context();
  CUresult return_value;
  if (lupine_route_is_local(route))
    return lupine_call_real_cuda_fn(
        LUPINE_REAL_CUDA_SYMBOL("cuDevSmResourceSplitByCount"), result,
        nbGroups, input, remainder, flags, minCount);
  conn_t *conn = lupine_route_remote_conn(route);
  uint8_t nbGroups_present = nbGroups != nullptr;
  unsigned int nbGroups_requested =
      (nbGroups_present && result != nullptr) ? *nbGroups : 0;
  uint8_t result_null = result == nullptr ? 1 : 0;
  uint8_t input_present = input != nullptr;
  uint8_t remainder_present = remainder != nullptr;
  if (lupine_prepare_rpc(conn) < 0 ||
      rpc_write_start_request_on(conn, RPC_cuDevSmResourceSplitByCount,
                                 RPC_SCOPE_NONE) < 0 ||
      rpc_write(conn, &nbGroups_present, sizeof(uint8_t)) < 0 ||
      rpc_write(conn, &nbGroups_requested, sizeof(unsigned int)) < 0 ||
      rpc_write(conn, &result_null, sizeof(uint8_t)) < 0 ||
      rpc_write(conn, &input_present, sizeof(uint8_t)) < 0 ||
      (input_present &&
       rpc_write(conn, input, sizeof(const CUdevResource)) < 0) ||
      rpc_write(conn, &remainder_present, sizeof(uint8_t)) < 0 ||
      (remainder_present &&
       rpc_write(conn, remainder, sizeof(CUdevResource)) < 0) ||
      rpc_write(conn, &flags, sizeof(unsigned int)) < 0 ||
      rpc_write(conn, &minCount, sizeof(unsigned int)) < 0 ||
      rpc_wait_for_response(conn) < 0 ||
      (nbGroups_present &&
       rpc_read(conn, nbGroups, sizeof(unsigned int)) < 0) ||
      (result != nullptr && nbGroups_requested != 0 && nbGroups_present &&
       *nbGroups != 0 &&
       rpc_read(conn, result,
                ((nbGroups_present ? *nbGroups : 0) < nbGroups_requested
                     ? (nbGroups_present ? *nbGroups : 0)
                     : nbGroups_requested) *
                    sizeof(CUdevResource)) < 0) ||
      (remainder_present &&
       rpc_read(conn, remainder, sizeof(CUdevResource)) < 0) ||
      rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
      rpc_read_end(conn) < 0)
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  return return_value;
}

#endif

#if CUDA_VERSION >= 13010
CUresult cuDevSmResourceSplit(CUdevResource *result, unsigned int nbGroups,
                              const CUdevResource *input,
                              CUdevResource *remainder, unsigned int flags,
                              CU_DEV_SM_RESOURCE_GROUP_PARAMS *groupParams) {
  lupine_route route = lupine_route_for_current_context();
  CUresult return_value;
  if (lupine_route_is_local(route))
    return lupine_call_real_cuda_fn(
        LUPINE_REAL_CUDA_SYMBOL("cuDevSmResourceSplit"), result, nbGroups,
        input, remainder, flags, groupParams);
  conn_t *conn = lupine_route_remote_conn(route);
  uint8_t result_null = result == nullptr ? 1 : 0;
  uint8_t input_present = input != nullptr;
  uint8_t remainder_present = remainder != nullptr;
  uint8_t groupParams_null = groupParams == nullptr ? 1 : 0;
  if (lupine_prepare_rpc(conn) < 0 ||
      rpc_write_start_request_on(conn, RPC_cuDevSmResourceSplit,
                                 RPC_SCOPE_NONE) < 0 ||
      rpc_write(conn, &nbGroups, sizeof(unsigned int)) < 0 ||
      rpc_write(conn, &result_null, sizeof(uint8_t)) < 0 ||
      rpc_write(conn, &input_present, sizeof(uint8_t)) < 0 ||
      (input_present &&
       rpc_write(conn, input, sizeof(const CUdevResource)) < 0) ||
      rpc_write(conn, &remainder_present, sizeof(uint8_t)) < 0 ||
      (remainder_present &&
       rpc_write(conn, remainder, sizeof(CUdevResource)) < 0) ||
      rpc_write(conn, &flags, sizeof(unsigned int)) < 0 ||
      rpc_write(conn, &groupParams_null, sizeof(uint8_t)) < 0 ||
      (!groupParams_null &&
       nbGroups * sizeof(CU_DEV_SM_RESOURCE_GROUP_PARAMS) != 0 &&
       rpc_write(conn, groupParams,
                 nbGroups * sizeof(CU_DEV_SM_RESOURCE_GROUP_PARAMS)) < 0) ||
      rpc_wait_for_response(conn) < 0 ||
      (result != nullptr && nbGroups != 0 &&
       rpc_read(conn, result, nbGroups * sizeof(CUdevResource)) < 0) ||
      (remainder_present &&
       rpc_read(conn, remainder, sizeof(CUdevResource)) < 0) ||
      (!groupParams_null &&
       nbGroups * sizeof(CU_DEV_SM_RESOURCE_GROUP_PARAMS) != 0 &&
       rpc_read(conn, groupParams,
                nbGroups * sizeof(CU_DEV_SM_RESOURCE_GROUP_PARAMS)) < 0) ||
      rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
      rpc_read_end(conn) < 0)
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  return return_value;
}

#endif

#if CUDA_VERSION >= 12040
CUresult cuDevResourceGenerateDesc(CUdevResourceDesc *phDesc,
                                   CUdevResource *resources,
                                   unsigned int nbResources) {
  lupine_route route = lupine_route_for_current_context();
  CUresult return_value;
  if (lupine_route_is_local(route))
    return lupine_call_real_cuda_fn(
        LUPINE_REAL_CUDA_SYMBOL("cuDevResourceGenerateDesc"), phDesc, resources,
        nbResources);
  conn_t *conn = lupine_route_remote_conn(route);
  uint8_t resources_null = resources == nullptr ? 1 : 0;
  uint8_t phDesc_present = phDesc != nullptr;
  if (lupine_prepare_rpc(conn) < 0 ||
      rpc_write_start_request_on(conn, RPC_cuDevResourceGenerateDesc,
                                 RPC_SCOPE_NONE) < 0 ||
      rpc_write(conn, &nbResources, sizeof(unsigned int)) < 0 ||
      rpc_write(conn, &resources_null, sizeof(uint8_t)) < 0 ||
      (!resources_null && nbResources * sizeof(CUdevResource) != 0 &&
       rpc_write(conn, resources, nbResources * sizeof(CUdevResource)) < 0) ||
      rpc_write(conn, &phDesc_present, sizeof(uint8_t)) < 0 ||
      (phDesc_present &&
       rpc_write(conn, phDesc, sizeof(CUdevResourceDesc)) < 0) ||
      rpc_wait_for_response(conn) < 0 ||
      (phDesc_present &&
       rpc_read(conn, phDesc, sizeof(CUdevResourceDesc)) < 0) ||
      rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
      rpc_read_end(conn) < 0)
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  return return_value;
}

#endif

#if CUDA_VERSION >= 12040
CUresult cuGreenCtxRecordEvent(CUgreenCtx hCtx, CUevent hEvent) {
  lupine_route route = lupine_route_for_current_context();
  CUresult return_value;
  if (lupine_route_is_local(route)) {
    return_value = lupine_call_real_cuda_fn(
        LUPINE_REAL_CUDA_SYMBOL("cuGreenCtxRecordEvent"), hCtx, hEvent);
    // Re-recording re-arms the event, so its cached completion is stale.
    // Invalidate after enqueue so an overlapping query cannot retain old state.
    lupine_event_invalidate_completion(hEvent);
    return return_value;
  }
  conn_t *conn = lupine_route_remote_conn(route);
  if (lupine_prepare_rpc(conn) < 0 ||
      rpc_write_start_request(conn, RPC_cuGreenCtxRecordEvent) < 0 ||
      rpc_request_on(conn, reinterpret_cast<uintptr_t>(hEvent)) < 0 ||
      rpc_write(conn, &hCtx, sizeof(CUgreenCtx)) < 0 ||
      rpc_write(conn, &hEvent, sizeof(CUevent)) < 0 || rpc_write_end(conn) < 0)
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  return_value = CUDA_SUCCESS;
  // Re-recording re-arms the event, so its cached completion is stale.
  // Invalidate after enqueue so an overlapping query cannot retain old state.
  lupine_event_invalidate_completion(hEvent);
  return return_value;
}

#endif

#if CUDA_VERSION >= 12040
CUresult cuGreenCtxWaitEvent(CUgreenCtx hCtx, CUevent hEvent) {
  lupine_route route = lupine_route_for_current_context();
  CUresult return_value;
  if (lupine_route_is_local(route))
    return lupine_call_real_cuda_fn(
        LUPINE_REAL_CUDA_SYMBOL("cuGreenCtxWaitEvent"), hCtx, hEvent);
  conn_t *conn = lupine_route_remote_conn(route);
  if (lupine_prepare_rpc(conn) < 0 ||
      rpc_write_start_request(conn, RPC_cuGreenCtxWaitEvent) < 0 ||
      rpc_request_on(conn, reinterpret_cast<uintptr_t>(hEvent)) < 0 ||
      rpc_write(conn, &hCtx, sizeof(CUgreenCtx)) < 0 ||
      rpc_write(conn, &hEvent, sizeof(CUevent)) < 0 || rpc_write_end(conn) < 0)
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  return_value = CUDA_SUCCESS;
  return return_value;
}

#endif

#if CUDA_VERSION >= 12040
CUresult cuStreamGetGreenCtx(CUstream hStream, CUgreenCtx *phCtx) {
  lupine_route route = (hStream != nullptr ? lupine_route_for_stream(hStream)
                                           : lupine_route_for_default());
  CUresult return_value;
  if (lupine_route_is_local(route))
    return lupine_call_real_cuda_fn(
        LUPINE_REAL_CUDA_SYMBOL("cuStreamGetGreenCtx"), hStream, phCtx);
  conn_t *conn = lupine_route_remote_conn(route);
  uint8_t phCtx_present = phCtx != nullptr;
  if (lupine_prepare_rpc(conn) < 0 ||
      lupine_rpc_start_on_stream(conn, RPC_cuStreamGetGreenCtx, hStream) < 0 ||
      rpc_write(conn, &hStream, sizeof(CUstream)) < 0 ||
      rpc_write(conn, &phCtx_present, sizeof(uint8_t)) < 0 ||
      (phCtx_present && rpc_write(conn, phCtx, sizeof(CUgreenCtx)) < 0) ||
      rpc_wait_for_response(conn) < 0 ||
      (phCtx_present && rpc_read(conn, phCtx, sizeof(CUgreenCtx)) < 0) ||
      rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
      rpc_read_end(conn) < 0)
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  return return_value;
}

#endif

#if CUDA_VERSION >= 12050
CUresult cuGreenCtxStreamCreate(CUstream *phStream, CUgreenCtx greenCtx,
                                unsigned int flags, int priority) {
  lupine_route route = lupine_route_for_current_context();
  CUresult return_value;
  if (lupine_route_is_local(route)) {
    return_value = lupine_call_real_cuda_fn(
        LUPINE_REAL_CUDA_SYMBOL("cuGreenCtxStreamCreate"), phStream, greenCtx,
        flags, priority);
    if (return_value == CUDA_SUCCESS && phStream != nullptr) {
      lupine_note_stream_owner_route(*phStream, route);
    }
    return return_value;
  }
  conn_t *conn = lupine_route_remote_conn(route);
  uint8_t phStream_present = phStream != nullptr;
  if (lupine_prepare_rpc(conn) < 0 ||
      rpc_write_start_request_on(conn, RPC_cuGreenCtxStreamCreate,
                                 RPC_SCOPE_NONE) < 0 ||
      rpc_write(conn, &phStream_present, sizeof(uint8_t)) < 0 ||
      (phStream_present && rpc_write(conn, phStream, sizeof(CUstream)) < 0) ||
      rpc_write(conn, &greenCtx, sizeof(CUgreenCtx)) < 0 ||
      rpc_write(conn, &flags, sizeof(unsigned int)) < 0 ||
      rpc_write(conn, &priority, sizeof(int)) < 0 ||
      rpc_wait_for_response(conn) < 0 ||
      (phStream_present && rpc_read(conn, phStream, sizeof(CUstream)) < 0) ||
      rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
      rpc_read_end(conn) < 0)
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  if (return_value == CUDA_SUCCESS && phStream != nullptr) {
    lupine_note_stream_owner_route(*phStream, route);
  }
  return return_value;
}

#endif

#if CUDA_VERSION >= 13000
CUresult cuGreenCtxGetId(CUgreenCtx greenCtx, unsigned long long *greenCtxId) {
  lupine_route route = lupine_route_for_current_context();
  CUresult return_value;
  if (lupine_route_is_local(route))
    return lupine_call_real_cuda_fn(LUPINE_REAL_CUDA_SYMBOL("cuGreenCtxGetId"),
                                    greenCtx, greenCtxId);
  conn_t *conn = lupine_route_remote_conn(route);
  uint8_t greenCtxId_present = greenCtxId != nullptr;
  if (lupine_prepare_rpc(conn) < 0 ||
      rpc_write_start_request_on(conn, RPC_cuGreenCtxGetId, RPC_SCOPE_NONE) <
          0 ||
      rpc_write(conn, &greenCtx, sizeof(CUgreenCtx)) < 0 ||
      rpc_write(conn, &greenCtxId_present, sizeof(uint8_t)) < 0 ||
      (greenCtxId_present &&
       rpc_write(conn, greenCtxId, sizeof(unsigned long long)) < 0) ||
      rpc_wait_for_response(conn) < 0 ||
      (greenCtxId_present &&
       rpc_read(conn, greenCtxId, sizeof(unsigned long long)) < 0) ||
      rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
      rpc_read_end(conn) < 0)
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  return return_value;
}

#endif

#if CUDA_VERSION >= 13010
CUresult cuStreamGetDevResource(CUstream hStream, CUdevResource *resource,
                                CUdevResourceType type) {
  lupine_route route = (hStream != nullptr ? lupine_route_for_stream(hStream)
                                           : lupine_route_for_default());
  CUresult return_value;
  if (lupine_route_is_local(route))
    return lupine_call_real_cuda_fn(
        LUPINE_REAL_CUDA_SYMBOL("cuStreamGetDevResource"), hStream, resource,
        type);
  conn_t *conn = lupine_route_remote_conn(route);
  uint8_t resource_present = resource != nullptr;
  if (lupine_prepare_rpc(conn) < 0 ||
      lupine_rpc_start_on_stream(conn, RPC_cuStreamGetDevResource, hStream) <
          0 ||
      rpc_write(conn, &hStream, sizeof(CUstream)) < 0 ||
      rpc_write(conn, &resource_present, sizeof(uint8_t)) < 0 ||
      (resource_present &&
       rpc_write(conn, resource, sizeof(CUdevResource)) < 0) ||
      rpc_write(conn, &type, sizeof(CUdevResourceType)) < 0 ||
      rpc_wait_for_response(conn) < 0 ||
      (resource_present &&
       rpc_read(conn, resource, sizeof(CUdevResource)) < 0) ||
      rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
      rpc_read_end(conn) < 0)
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  return return_value;
}

#endif

#if CUDA_VERSION >= 12090
CUresult cuLogsCurrent(CUlogIterator *iterator_out, unsigned int flags) {
  lupine_route route = lupine_route_for_default();
  CUresult return_value;
  if (lupine_route_is_local(route))
    return lupine_call_real_cuda_fn(LUPINE_REAL_CUDA_SYMBOL("cuLogsCurrent"),
                                    iterator_out, flags);
  conn_t *conn = lupine_route_remote_conn(route);
  uint8_t iterator_out_present = iterator_out != nullptr;
  if (lupine_prepare_rpc(conn) < 0 ||
      rpc_write_start_request_on(conn, RPC_cuLogsCurrent, RPC_SCOPE_NONE) < 0 ||
      rpc_write(conn, &iterator_out_present, sizeof(uint8_t)) < 0 ||
      (iterator_out_present &&
       rpc_write(conn, iterator_out, sizeof(CUlogIterator)) < 0) ||
      rpc_write(conn, &flags, sizeof(unsigned int)) < 0 ||
      rpc_wait_for_response(conn) < 0 ||
      (iterator_out_present &&
       rpc_read(conn, iterator_out, sizeof(CUlogIterator)) < 0) ||
      rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
      rpc_read_end(conn) < 0)
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  return return_value;
}

#endif

#if CUDA_VERSION >= 12090
CUresult cuLogsDumpToFile(CUlogIterator *iterator, const char *pathToFile,
                          unsigned int flags) {
  lupine_route route = lupine_route_for_default();
  CUresult return_value;
  if (lupine_route_is_local(route))
    return lupine_call_real_cuda_fn(LUPINE_REAL_CUDA_SYMBOL("cuLogsDumpToFile"),
                                    iterator, pathToFile, flags);
  conn_t *conn = lupine_route_remote_conn(route);
  uint8_t iterator_present = iterator != nullptr;
  std::size_t pathToFile_len =
      pathToFile == nullptr
          ? 0
          : static_cast<std::size_t>(std::strlen(pathToFile) + 1);
  if (lupine_prepare_rpc(conn) < 0 ||
      rpc_write_start_request_on(conn, RPC_cuLogsDumpToFile, RPC_SCOPE_NONE) <
          0 ||
      rpc_write(conn, &iterator_present, sizeof(uint8_t)) < 0 ||
      (iterator_present &&
       rpc_write(conn, iterator, sizeof(CUlogIterator)) < 0) ||
      rpc_write(conn, &pathToFile_len, sizeof(std::size_t)) < 0 ||
      (pathToFile_len != 0 &&
       rpc_write(conn, pathToFile, pathToFile_len) < 0) ||
      rpc_write(conn, &flags, sizeof(unsigned int)) < 0 ||
      rpc_wait_for_response(conn) < 0 ||
      (iterator_present &&
       rpc_read(conn, iterator, sizeof(CUlogIterator)) < 0) ||
      rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
      rpc_read_end(conn) < 0)
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  return return_value;
}

#endif

#if CUDA_VERSION >= 12090
CUresult cuLogsDumpToMemory(CUlogIterator *iterator, char *buffer, size_t *size,
                            unsigned int flags) {
  lupine_route route = lupine_route_for_default();
  size_t buffer_capacity = size != nullptr ? *size : 0;
  CUresult return_value;
  if (lupine_route_is_local(route)) {
    return_value =
        lupine_call_real_cuda_fn(LUPINE_REAL_CUDA_SYMBOL("cuLogsDumpToMemory"),
                                 iterator, buffer, size, flags);
    if (return_value == CUDA_SUCCESS && buffer != nullptr && size != nullptr &&
        buffer_capacity > *size)
      buffer[*size] = '\0';
    return return_value;
  }
  conn_t *conn = lupine_route_remote_conn(route);
  uint8_t iterator_present = iterator != nullptr;
  uint8_t size_present = size != nullptr;
  size_t size_requested = (size_present && buffer != nullptr) ? *size : 0;
  uint8_t buffer_null = buffer == nullptr ? 1 : 0;
  if (lupine_prepare_rpc(conn) < 0 ||
      rpc_write_start_request_on(conn, RPC_cuLogsDumpToMemory, RPC_SCOPE_NONE) <
          0 ||
      rpc_write(conn, &iterator_present, sizeof(uint8_t)) < 0 ||
      (iterator_present &&
       rpc_write(conn, iterator, sizeof(CUlogIterator)) < 0) ||
      rpc_write(conn, &size_present, sizeof(uint8_t)) < 0 ||
      rpc_write(conn, &size_requested, sizeof(size_t)) < 0 ||
      rpc_write(conn, &buffer_null, sizeof(uint8_t)) < 0 ||
      rpc_write(conn, &flags, sizeof(unsigned int)) < 0 ||
      rpc_wait_for_response(conn) < 0 ||
      (iterator_present &&
       rpc_read(conn, iterator, sizeof(CUlogIterator)) < 0) ||
      (size_present && rpc_read(conn, size, sizeof(size_t)) < 0) ||
      (buffer != nullptr && size_requested != 0 && size_present && *size != 0 &&
       rpc_read(conn, buffer,
                ((size_present ? *size : 0) < size_requested
                     ? (size_present ? *size : 0)
                     : size_requested) *
                    sizeof(char)) < 0) ||
      rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
      rpc_read_end(conn) < 0)
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  if (return_value == CUDA_SUCCESS && buffer != nullptr && size != nullptr &&
      buffer_capacity > *size)
    buffer[*size] = '\0';
  return return_value;
}

#endif

CUresult cuMemPrefetchAsync(CUdeviceptr devPtr, size_t count,
                            CUdevice dstDevice, CUstream hStream) {
  lupine_route route = lupine_route_for_deviceptr(devPtr);
  if (dstDevice != CU_DEVICE_CPU) {
    route = lupine_route_for_device(&dstDevice);
    if (route.kind == LUPINE_ROUTE_UNKNOWN_DEVICE)
      return CUDA_ERROR_INVALID_DEVICE;
  } else if (hStream != nullptr) {
    route = lupine_route_for_stream(hStream);
  }
  if (hStream != nullptr &&
      !lupine_routes_share_server(route, lupine_route_for_stream(hStream)))
    return CUDA_ERROR_INVALID_VALUE;
  if (!lupine_route_is_local(route) && lupine_is_managed_host_alias(devPtr))
    return CUDA_SUCCESS;
  CUresult return_value;
  if (lupine_route_is_local(route))
    return lupine_call_real_cuda_fn(
        LUPINE_REAL_CUDA_SYMBOL("cuMemPrefetchAsync"), devPtr, count, dstDevice,
        hStream);
  conn_t *conn = lupine_route_remote_conn(route);
  if (lupine_prepare_rpc(conn) < 0 ||
      lupine_rpc_start_on_stream(conn, RPC_cuMemPrefetchAsync, hStream) < 0 ||
      rpc_write(conn, &devPtr, sizeof(CUdeviceptr)) < 0 ||
      rpc_write(conn, &count, sizeof(size_t)) < 0 ||
      rpc_write(conn, &dstDevice, sizeof(CUdevice)) < 0 ||
      rpc_write(conn, &hStream, sizeof(CUstream)) < 0 ||
      rpc_write_end(conn) < 0)
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  return_value = CUDA_SUCCESS;
  return return_value;
}

CUresult cuMemAdvise(CUdeviceptr devPtr, size_t count, CUmem_advise advice,
                     CUdevice device) {
  lupine_route route = lupine_route_for_deviceptr(devPtr);
  CUdevice virtual_device = device;
  if (device != CU_DEVICE_CPU) {
    route = lupine_route_for_device(&device);
    if (route.kind == LUPINE_ROUTE_UNKNOWN_DEVICE)
      return CUDA_ERROR_INVALID_DEVICE;
  }
  if (!lupine_route_is_local(route) && lupine_is_managed_host_alias(devPtr))
    return lupine_managed_mem_advise(devPtr, count, advice, virtual_device);
  CUresult return_value;
  if (lupine_route_is_local(route))
    return lupine_call_real_cuda_fn(LUPINE_REAL_CUDA_SYMBOL("cuMemAdvise"),
                                    devPtr, count, advice, device);
  conn_t *conn = lupine_route_remote_conn(route);
  if (lupine_prepare_rpc(conn) < 0 ||
      rpc_write_start_request_on(conn, RPC_cuMemAdvise, RPC_SCOPE_NONE) < 0 ||
      rpc_write(conn, &devPtr, sizeof(CUdeviceptr)) < 0 ||
      rpc_write(conn, &count, sizeof(size_t)) < 0 ||
      rpc_write(conn, &advice, sizeof(CUmem_advise)) < 0 ||
      rpc_write(conn, &device, sizeof(CUdevice)) < 0 || rpc_write_end(conn) < 0)
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  return_value = CUDA_SUCCESS;
  return return_value;
}

CUresult cuGraphInstantiate_v2(CUgraphExec *phGraphExec, CUgraph hGraph,
                               CUgraphNode *phErrorNode, char *logBuffer,
                               size_t bufferSize) {
  lupine_route route = lupine_route_for_graph(hGraph);
  CUresult return_value;
  if (lupine_route_is_local(route)) {
    return_value = lupine_call_real_cuda_fn(
        LUPINE_REAL_CUDA_SYMBOL("cuGraphInstantiate_v2"), phGraphExec, hGraph,
        phErrorNode, logBuffer, bufferSize);
    if (return_value == CUDA_SUCCESS && phGraphExec != nullptr) {
      lupine_note_graph_exec_owner_route(*phGraphExec, route);
    }
    return return_value;
  }
  conn_t *conn = lupine_route_remote_conn(route);
  if (phGraphExec == nullptr)
    return CUDA_ERROR_INVALID_VALUE;
  uint8_t phErrorNode_present = phErrorNode != nullptr;
  uint8_t logBuffer_null = logBuffer == nullptr ? 1 : 0;
  if (lupine_prepare_rpc(conn) < 0 ||
      rpc_write_start_request_on(conn, RPC_cuGraphInstantiate_v2,
                                 RPC_SCOPE_NONE) < 0 ||
      rpc_write(conn, &hGraph, sizeof(CUgraph)) < 0 ||
      rpc_write(conn, &phErrorNode_present, sizeof(uint8_t)) < 0 ||
      (phErrorNode_present &&
       rpc_write(conn, phErrorNode, sizeof(CUgraphNode)) < 0) ||
      rpc_write(conn, &bufferSize, sizeof(size_t)) < 0 ||
      rpc_write(conn, &logBuffer_null, sizeof(uint8_t)) < 0 ||
      rpc_wait_for_response(conn) < 0 ||
      rpc_read(conn, phGraphExec, sizeof(CUgraphExec)) < 0 ||
      (phErrorNode_present &&
       rpc_read(conn, phErrorNode, sizeof(CUgraphNode)) < 0) ||
      ([&]() {
        uint8_t logBuffer_has_data = 0;
        if (rpc_read(conn, &logBuffer_has_data, sizeof(uint8_t)) < 0)
          return true;
        if (logBuffer_has_data == 0)
          return false;
        if (logBuffer_has_data != 1 || logBuffer == nullptr ||
            bufferSize > size_t(-1) / sizeof(char)) {
          rpc_close_transport_socket(conn);
          return true;
        }
        return bufferSize != 0 &&
               rpc_read(conn, logBuffer, bufferSize * sizeof(char)) < 0;
      }()) ||
      rpc_read(conn, &return_value, sizeof(CUresult)) < 0 ||
      rpc_read_end(conn) < 0)
    return CUDA_ERROR_DEVICE_UNAVAILABLE;
  if (return_value == CUDA_SUCCESS && phGraphExec != nullptr) {
    lupine_note_graph_exec_owner_route(*phGraphExec, route);
  }
  return return_value;
}

CUresult cuGraphExecUpdate(CUgraphExec hGraphExec, CUgraph hGraph,
                           CUgraphNode *hErrorNode_out,
                           CUgraphExecUpdateResult *updateResult_out) {
  lupine_route route = lupine_route_for_graph_exec(hGraphExec);
  CUresult return_value;
  if (lupine_route_is_local(route))
    return lupine_call_real_cuda_fn(
        LUPINE_REAL_CUDA_SYMBOL("cuGraphExecUpdate"), hGraphExec, hGraph,
        hErrorNode_out, updateResult_out);
  conn_t *conn = lupine_route_remote_conn(route);
  uint8_t hErrorNode_out_present = hErrorNode_out != nullptr;
  uint8_t updateResult_out_present = updateResult_out != nullptr;
  if (lupine_prepare_rpc(conn) < 0 ||
      rpc_write_start_request_on(conn, RPC_cuGraphExecUpdate,
                                 reinterpret_cast<uintptr_t>(hGraphExec)) < 0 ||
      rpc_write(conn, &hGraphExec, sizeof(CUgraphExec)) < 0 ||
      rpc_write(conn, &hGraph, sizeof(CUgraph)) < 0 ||
      rpc_write(conn, &hErrorNode_out_present, sizeof(uint8_t)) < 0 ||
      (hErrorNode_out_present &&
       rpc_write(conn, hErrorNode_out, sizeof(CUgraphNode)) < 0) ||
      rpc_write(conn, &updateResult_out_present, sizeof(uint8_t)) < 0 ||
      (updateResult_out_present &&
       rpc_write(conn, updateResult_out, sizeof(CUgraphExecUpdateResult)) <
           0) ||
      rpc_wait_for_response(conn) < 0 ||
      (hErrorNode_out_present &&
       rpc_read(conn, hErrorNode_out, sizeof(CUgraphNode)) < 0) ||
      (updateResult_out_present &&
       rpc_read(conn, updateResult_out, sizeof(CUgraphExecUpdateResult)) < 0) ||
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

#ifdef cuMemcpyHtoD
#undef cuMemcpyHtoD
#endif
extern "C" CUresult cuMemcpyHtoD(CUdeviceptr dstDevice, const void *srcHost,
                                 size_t ByteCount) {
  return cuMemcpyHtoD_v2(dstDevice, srcHost, ByteCount);
}

#ifdef cuMemcpyHtoDAsync
#undef cuMemcpyHtoDAsync
#endif
extern "C" CUresult cuMemcpyHtoDAsync(CUdeviceptr dstDevice,
                                      const void *srcHost, size_t ByteCount,
                                      CUstream hStream) {
  return cuMemcpyHtoDAsync_v2(dstDevice, srcHost, ByteCount, hStream);
}

#ifdef cuMemcpyDtoH
#undef cuMemcpyDtoH
#endif
extern "C" CUresult cuMemcpyDtoH(void *dstHost, CUdeviceptr srcDevice,
                                 size_t ByteCount) {
  return cuMemcpyDtoH_v2(dstHost, srcDevice, ByteCount);
}

#ifdef cuMemcpyDtoHAsync
#undef cuMemcpyDtoHAsync
#endif
extern "C" CUresult cuMemcpyDtoHAsync(void *dstHost, CUdeviceptr srcDevice,
                                      size_t ByteCount, CUstream hStream) {
  return cuMemcpyDtoHAsync_v2(dstHost, srcDevice, ByteCount, hStream);
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

#ifdef cuStreamDestroy
#undef cuStreamDestroy
#endif
extern "C" CUresult cuStreamDestroy(CUstream hStream) {
  return cuStreamDestroy_v2(hStream);
}

#ifdef cuEventDestroy
#undef cuEventDestroy
#endif
extern "C" CUresult cuEventDestroy(CUevent hEvent) {
  return cuEventDestroy_v2(hEvent);
}

#ifdef cuEventElapsedTime
#undef cuEventElapsedTime
#endif
extern "C" CUresult cuEventElapsedTime(float *pMilliseconds, CUevent hStart,
                                       CUevent hEnd) {
  return cuEventElapsedTime_v2(pMilliseconds, hStart, hEnd);
}

#ifdef cuStreamGetDevice_ptsz
#undef cuStreamGetDevice_ptsz
#endif
extern "C" CUresult cuStreamGetDevice_ptsz(CUstream hStream, CUdevice *device) {
  return cuStreamGetDevice(hStream, device);
}

#ifdef cuMemGetAddressRange
#undef cuMemGetAddressRange
#endif
extern "C" CUresult cuMemGetAddressRange(CUdeviceptr *pbase, size_t *psize,
                                         CUdeviceptr dptr) {
  return cuMemGetAddressRange_v2(pbase, psize, dptr);
}

#ifdef cuMemGetInfo
#undef cuMemGetInfo
#endif
extern "C" CUresult cuMemGetInfo(size_t *free, size_t *total) {
  return cuMemGetInfo_v2(free, total);
}

#ifdef cuLinkCreate
#undef cuLinkCreate
#endif
extern "C" CUresult cuLinkCreate(unsigned int numOptions, CUjit_option *options,
                                 void **optionValues, CUlinkState *stateOut) {
  return cuLinkCreate_v2(numOptions, options, optionValues, stateOut);
}

#ifdef cuLinkAddData
#undef cuLinkAddData
#endif
extern "C" CUresult cuLinkAddData(CUlinkState state, CUjitInputType type,
                                  void *data, size_t size, const char *name,
                                  unsigned int numOptions,
                                  CUjit_option *options, void **optionValues) {
  return cuLinkAddData_v2(state, type, data, size, name, numOptions, options,
                          optionValues);
}

#ifdef cuLinkAddFile
#undef cuLinkAddFile
#endif
extern "C" CUresult cuLinkAddFile(CUlinkState state, CUjitInputType type,
                                  const char *path, unsigned int numOptions,
                                  CUjit_option *options, void **optionValues) {
  return cuLinkAddFile_v2(state, type, path, numOptions, options, optionValues);
}

#ifdef cuLaunchCooperativeKernel_ptsz
#undef cuLaunchCooperativeKernel_ptsz
#endif
extern "C" CUresult cuLaunchCooperativeKernel_ptsz(
    CUfunction f, unsigned int gridDimX, unsigned int gridDimY,
    unsigned int gridDimZ, unsigned int blockDimX, unsigned int blockDimY,
    unsigned int blockDimZ, unsigned int sharedMemBytes, CUstream hStream,
    void **kernelParams) {
  return cuLaunchCooperativeKernel(f, gridDimX, gridDimY, gridDimZ, blockDimX,
                                   blockDimY, blockDimZ, sharedMemBytes,
                                   hStream, kernelParams);
}

#ifdef cuGraphKernelNodeGetParams
#undef cuGraphKernelNodeGetParams
#endif
extern "C" CUresult
cuGraphKernelNodeGetParams(CUgraphNode hNode,
                           CUDA_KERNEL_NODE_PARAMS *nodeParams) {
  return cuGraphKernelNodeGetParams_v2(hNode, nodeParams);
}

#ifdef cuGraphKernelNodeSetParams
#undef cuGraphKernelNodeSetParams
#endif
extern "C" CUresult
cuGraphKernelNodeSetParams(CUgraphNode hNode,
                           const CUDA_KERNEL_NODE_PARAMS *nodeParams) {
  return cuGraphKernelNodeSetParams_v2(hNode, nodeParams);
}

#ifdef cuGraphAddKernelNode
#undef cuGraphAddKernelNode
#endif
extern "C" CUresult
cuGraphAddKernelNode(CUgraphNode *phGraphNode, CUgraph hGraph,
                     const CUgraphNode *dependencies, size_t numDependencies,
                     const CUDA_KERNEL_NODE_PARAMS *nodeParams) {
  return cuGraphAddKernelNode_v2(phGraphNode, hGraph, dependencies,
                                 numDependencies, nodeParams);
}

#ifdef cuGraphExecKernelNodeSetParams
#undef cuGraphExecKernelNodeSetParams
#endif
extern "C" CUresult
cuGraphExecKernelNodeSetParams(CUgraphExec hGraphExec, CUgraphNode hNode,
                               const CUDA_KERNEL_NODE_PARAMS *nodeParams) {
  return cuGraphExecKernelNodeSetParams_v2(hGraphExec, hNode, nodeParams);
}

#ifdef cuStreamWaitValue32
#undef cuStreamWaitValue32
#endif
extern "C" CUresult cuStreamWaitValue32(CUstream stream, CUdeviceptr addr,
                                        cuuint32_t value, unsigned int flags) {
  return cuStreamWaitValue32_v2(stream, addr, value, flags);
}

#ifdef cuStreamWaitValue64
#undef cuStreamWaitValue64
#endif
extern "C" CUresult cuStreamWaitValue64(CUstream stream, CUdeviceptr addr,
                                        cuuint64_t value, unsigned int flags) {
  return cuStreamWaitValue64_v2(stream, addr, value, flags);
}

#ifdef cuStreamWriteValue32
#undef cuStreamWriteValue32
#endif
extern "C" CUresult cuStreamWriteValue32(CUstream stream, CUdeviceptr addr,
                                         cuuint32_t value, unsigned int flags) {
  return cuStreamWriteValue32_v2(stream, addr, value, flags);
}

#ifdef cuStreamWriteValue64
#undef cuStreamWriteValue64
#endif
extern "C" CUresult cuStreamWriteValue64(CUstream stream, CUdeviceptr addr,
                                         cuuint64_t value, unsigned int flags) {
  return cuStreamWriteValue64_v2(stream, addr, value, flags);
}

#ifdef cuStreamBatchMemOp
#undef cuStreamBatchMemOp
#endif
extern "C" CUresult cuStreamBatchMemOp(CUstream stream, unsigned int count,
                                       CUstreamBatchMemOpParams *paramArray,
                                       unsigned int flags) {
  return cuStreamBatchMemOp_v2(stream, count, paramArray, flags);
}

#ifdef cuArrayCreate
#undef cuArrayCreate
#endif
extern "C" CUresult cuArrayCreate(CUarray *pHandle,
                                  const CUDA_ARRAY_DESCRIPTOR *pAllocateArray) {
  return cuArrayCreate_v2(pHandle, pAllocateArray);
}

#ifdef cuArray3DCreate
#undef cuArray3DCreate
#endif
extern "C" CUresult
cuArray3DCreate(CUarray *pHandle,
                const CUDA_ARRAY3D_DESCRIPTOR *pAllocateArray) {
  return cuArray3DCreate_v2(pHandle, pAllocateArray);
}

#ifdef cuArray3DGetDescriptor
#undef cuArray3DGetDescriptor
#endif
extern "C" CUresult
cuArray3DGetDescriptor(CUDA_ARRAY3D_DESCRIPTOR *pArrayDescriptor,
                       CUarray hArray) {
  return cuArray3DGetDescriptor_v2(pArrayDescriptor, hArray);
}

#ifdef cuMemAllocHost
#undef cuMemAllocHost
#endif
extern "C" CUresult cuMemAllocHost(void **pp, size_t bytesize) {
  return cuMemAllocHost_v2(pp, bytesize);
}

#ifdef cuMemHostGetDevicePointer
#undef cuMemHostGetDevicePointer
#endif
extern "C" CUresult cuMemHostGetDevicePointer(CUdeviceptr *pdptr, void *p,
                                              unsigned int Flags) {
  return cuMemHostGetDevicePointer_v2(pdptr, p, Flags);
}

#ifdef cuMemFree
#undef cuMemFree
#endif
extern "C" CUresult cuMemFree(CUdeviceptr dptr) { return cuMemFree_v2(dptr); }

#ifdef cuMemcpyDtoA
#undef cuMemcpyDtoA
#endif
extern "C" CUresult cuMemcpyDtoA(CUarray dstArray, size_t dstOffset,
                                 CUdeviceptr srcDevice, size_t ByteCount) {
  return cuMemcpyDtoA_v2(dstArray, dstOffset, srcDevice, ByteCount);
}

#ifdef cuMemcpyAtoD
#undef cuMemcpyAtoD
#endif
extern "C" CUresult cuMemcpyAtoD(CUdeviceptr dstDevice, CUarray srcArray,
                                 size_t srcOffset, size_t ByteCount) {
  return cuMemcpyAtoD_v2(dstDevice, srcArray, srcOffset, ByteCount);
}

#ifdef cuMemcpyAtoA
#undef cuMemcpyAtoA
#endif
extern "C" CUresult cuMemcpyAtoA(CUarray dstArray, size_t dstOffset,
                                 CUarray srcArray, size_t srcOffset,
                                 size_t ByteCount) {
  return cuMemcpyAtoA_v2(dstArray, dstOffset, srcArray, srcOffset, ByteCount);
}

#ifdef cuMemcpyHtoA
#undef cuMemcpyHtoA
#endif
extern "C" CUresult cuMemcpyHtoA(CUarray dstArray, size_t dstOffset,
                                 const void *srcHost, size_t ByteCount) {
  return cuMemcpyHtoA_v2(dstArray, dstOffset, srcHost, ByteCount);
}

#ifdef cuMemcpyHtoA_v2_ptds
#undef cuMemcpyHtoA_v2_ptds
#endif
extern "C" CUresult cuMemcpyHtoA_v2_ptds(CUarray dstArray, size_t dstOffset,
                                         const void *srcHost,
                                         size_t ByteCount) {
  return cuMemcpyHtoA_v2(dstArray, dstOffset, srcHost, ByteCount);
}

#ifdef cuMemcpyHtoAAsync
#undef cuMemcpyHtoAAsync
#endif
extern "C" CUresult cuMemcpyHtoAAsync(CUarray dstArray, size_t dstOffset,
                                      const void *srcHost, size_t ByteCount,
                                      CUstream hStream) {
  return cuMemcpyHtoAAsync_v2(dstArray, dstOffset, srcHost, ByteCount, hStream);
}

#ifdef cuMemcpyHtoAAsync_v2_ptsz
#undef cuMemcpyHtoAAsync_v2_ptsz
#endif
extern "C" CUresult cuMemcpyHtoAAsync_v2_ptsz(CUarray dstArray,
                                              size_t dstOffset,
                                              const void *srcHost,
                                              size_t ByteCount,
                                              CUstream hStream) {
  return cuMemcpyHtoAAsync_v2(dstArray, dstOffset, srcHost, ByteCount, hStream);
}

#ifdef cuMemcpyAtoH
#undef cuMemcpyAtoH
#endif
extern "C" CUresult cuMemcpyAtoH(void *dstHost, CUarray srcArray,
                                 size_t srcOffset, size_t ByteCount) {
  return cuMemcpyAtoH_v2(dstHost, srcArray, srcOffset, ByteCount);
}

#ifdef cuMemcpyAtoH_v2_ptds
#undef cuMemcpyAtoH_v2_ptds
#endif
extern "C" CUresult cuMemcpyAtoH_v2_ptds(void *dstHost, CUarray srcArray,
                                         size_t srcOffset, size_t ByteCount) {
  return cuMemcpyAtoH_v2(dstHost, srcArray, srcOffset, ByteCount);
}

#ifdef cuMemcpyAtoHAsync
#undef cuMemcpyAtoHAsync
#endif
extern "C" CUresult cuMemcpyAtoHAsync(void *dstHost, CUarray srcArray,
                                      size_t srcOffset, size_t ByteCount,
                                      CUstream hStream) {
  return cuMemcpyAtoHAsync_v2(dstHost, srcArray, srcOffset, ByteCount, hStream);
}

#ifdef cuMemcpyAtoHAsync_v2_ptsz
#undef cuMemcpyAtoHAsync_v2_ptsz
#endif
extern "C" CUresult cuMemcpyAtoHAsync_v2_ptsz(void *dstHost, CUarray srcArray,
                                              size_t srcOffset,
                                              size_t ByteCount,
                                              CUstream hStream) {
  return cuMemcpyAtoHAsync_v2(dstHost, srcArray, srcOffset, ByteCount, hStream);
}

#ifdef cuMemcpy2D
#undef cuMemcpy2D
#endif
extern "C" CUresult cuMemcpy2D(const CUDA_MEMCPY2D *pCopy) {
  return cuMemcpy2D_v2(pCopy);
}

#ifdef cuMemcpy2DUnaligned
#undef cuMemcpy2DUnaligned
#endif
extern "C" CUresult cuMemcpy2DUnaligned(const CUDA_MEMCPY2D *pCopy) {
  return cuMemcpy2DUnaligned_v2(pCopy);
}

#ifdef cuMemcpy2DAsync
#undef cuMemcpy2DAsync
#endif
extern "C" CUresult cuMemcpy2DAsync(const CUDA_MEMCPY2D *pCopy,
                                    CUstream hStream) {
  return cuMemcpy2DAsync_v2(pCopy, hStream);
}

#ifdef cuMemcpy2DAsync_ptsz
#undef cuMemcpy2DAsync_ptsz
#endif
extern "C" CUresult cuMemcpy2DAsync_ptsz(const CUDA_MEMCPY2D *pCopy,
                                         CUstream hStream) {
  return cuMemcpy2DAsync_v2(pCopy, hStream);
}

#ifdef cuMemcpy3D
#undef cuMemcpy3D
#endif
extern "C" CUresult cuMemcpy3D(const CUDA_MEMCPY3D *pCopy) {
  return cuMemcpy3D_v2(pCopy);
}

#ifdef cuMemcpy3DAsync
#undef cuMemcpy3DAsync
#endif
extern "C" CUresult cuMemcpy3DAsync(const CUDA_MEMCPY3D *pCopy,
                                    CUstream hStream) {
  return cuMemcpy3DAsync_v2(pCopy, hStream);
}

#ifdef cuMemcpy3DAsync_ptsz
#undef cuMemcpy3DAsync_ptsz
#endif
extern "C" CUresult cuMemcpy3DAsync_ptsz(const CUDA_MEMCPY3D *pCopy,
                                         CUstream hStream) {
  return cuMemcpy3DAsync_v2(pCopy, hStream);
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

#ifdef cuMemcpy3DPeer_ptds
#undef cuMemcpy3DPeer_ptds
#endif
extern "C" CUresult cuMemcpy3DPeer_ptds(const CUDA_MEMCPY3D_PEER *pCopy) {
  return cuMemcpy3DPeer(pCopy);
}

#ifdef cuMemcpy3DPeerAsync_ptsz
#undef cuMemcpy3DPeerAsync_ptsz
#endif
extern "C" CUresult cuMemcpy3DPeerAsync_ptsz(const CUDA_MEMCPY3D_PEER *pCopy,
                                             CUstream hStream) {
  return cuMemcpy3DPeerAsync(pCopy, hStream);
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

#ifdef cuStreamWaitEvent_ptsz
#undef cuStreamWaitEvent_ptsz
#endif
extern "C" CUresult cuStreamWaitEvent_ptsz(CUstream hStream, CUevent hEvent,
                                           unsigned int Flags) {
  return cuStreamWaitEvent(hStream, hEvent, Flags);
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

#ifdef cuStreamAddCallback_ptsz
#undef cuStreamAddCallback_ptsz
#endif
extern "C" CUresult cuStreamAddCallback_ptsz(CUstream hStream,
                                             CUstreamCallback callback,
                                             void *userData,
                                             unsigned int flags) {
  return cuStreamAddCallback(hStream, callback, userData, flags);
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

#ifdef cuLaunchKernel_ptsz
#undef cuLaunchKernel_ptsz
#endif
extern "C" CUresult
cuLaunchKernel_ptsz(CUfunction f, unsigned int gridDimX, unsigned int gridDimY,
                    unsigned int gridDimZ, unsigned int blockDimX,
                    unsigned int blockDimY, unsigned int blockDimZ,
                    unsigned int sharedMemBytes, CUstream hStream,
                    void **kernelParams, void **extra) {
  return cuLaunchKernel(f, gridDimX, gridDimY, gridDimZ, blockDimX, blockDimY,
                        blockDimZ, sharedMemBytes, hStream, kernelParams,
                        extra);
}

#ifdef cuLaunchKernelEx_ptsz
#undef cuLaunchKernelEx_ptsz
#endif
extern "C" CUresult cuLaunchKernelEx_ptsz(const CUlaunchConfig *config,
                                          CUfunction f, void **kernelParams,
                                          void **extra) {
  return cuLaunchKernelEx(config, f, kernelParams, extra);
}

#ifdef cuLaunchHostFunc_ptsz
#undef cuLaunchHostFunc_ptsz
#endif
extern "C" CUresult cuLaunchHostFunc_ptsz(CUstream hStream, CUhostFn fn,
                                          void *userData) {
  return cuLaunchHostFunc(hStream, fn, userData);
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

#ifdef cuStreamUpdateCaptureDependencies_v2_ptsz
#undef cuStreamUpdateCaptureDependencies_v2_ptsz
#endif
extern "C" CUresult cuStreamUpdateCaptureDependencies_v2_ptsz(
    CUstream hStream, CUgraphNode *dependencies,
    const CUgraphEdgeData *dependencyData, size_t numDependencies,
    unsigned int flags) {
  return cuStreamUpdateCaptureDependencies_v2(
      hStream, dependencies, dependencyData, numDependencies, flags);
}

namespace {
struct lupine_cuda_function_entry {
  const char *name;
  void *pointer;
};

const lupine_cuda_function_entry kFunctionMap[] = {
    {"cuArray3DCreate", (void *)cuArray3DCreate_v2},
    {"cuArray3DCreate_v2", (void *)cuArray3DCreate_v2},
    {"cuArray3DGetDescriptor", (void *)cuArray3DGetDescriptor_v2},
    {"cuArray3DGetDescriptor_v2", (void *)cuArray3DGetDescriptor_v2},
    {"cuArrayCreate", (void *)cuArrayCreate_v2},
    {"cuArrayCreate_v2", (void *)cuArrayCreate_v2},
    {"cuArrayDestroy", (void *)cuArrayDestroy},
    {"cuArrayGetDescriptor_v2", (void *)cuArrayGetDescriptor_v2},
    {"cuArrayGetMemoryRequirements", (void *)cuArrayGetMemoryRequirements},
    {"cuArrayGetPlane", (void *)cuArrayGetPlane},
    {"cuArrayGetSparseProperties", (void *)cuArrayGetSparseProperties},
#if CUDA_VERSION >= 12010
    {"cuCoredumpGetAttributeGlobal", (void *)cuCoredumpGetAttributeGlobal},
#endif
#if CUDA_VERSION >= 12010
    {"cuCoredumpSetAttributeGlobal", (void *)cuCoredumpSetAttributeGlobal},
#endif
    {"cuCtxAttach", (void *)cuCtxAttach},
    {"cuCtxDestroy", (void *)cuCtxDestroy_v2},
    {"cuCtxDestroy_v2", (void *)cuCtxDestroy_v2},
    {"cuCtxDetach", (void *)cuCtxDetach},
    {"cuCtxDisablePeerAccess", (void *)cuCtxDisablePeerAccess},
    {"cuCtxEnablePeerAccess", (void *)cuCtxEnablePeerAccess},
#if CUDA_VERSION >= 12040
    {"cuCtxFromGreenCtx", (void *)cuCtxFromGreenCtx},
#endif
    {"cuCtxGetApiVersion", (void *)cuCtxGetApiVersion},
    {"cuCtxGetCacheConfig", (void *)cuCtxGetCacheConfig},
    {"cuCtxGetCurrent", (void *)cuCtxGetCurrent},
#if CUDA_VERSION >= 12040
    {"cuCtxGetDevResource", (void *)cuCtxGetDevResource},
#endif
    {"cuCtxGetDevice", (void *)cuCtxGetDevice},
#if CUDA_VERSION >= 13000
    {"cuCtxGetDevice_v2", (void *)cuCtxGetDevice_v2},
#endif
    {"cuCtxGetExecAffinity", (void *)cuCtxGetExecAffinity},
    {"cuCtxGetFlags", (void *)cuCtxGetFlags},
    {"cuCtxGetId", (void *)cuCtxGetId},
    {"cuCtxGetLimit", (void *)cuCtxGetLimit},
    {"cuCtxGetSharedMemConfig", (void *)cuCtxGetSharedMemConfig},
    {"cuCtxGetStreamPriorityRange", (void *)cuCtxGetStreamPriorityRange},
    {"cuCtxPopCurrent", (void *)cuCtxPopCurrent_v2},
    {"cuCtxPopCurrent_v2", (void *)cuCtxPopCurrent_v2},
    {"cuCtxPushCurrent", (void *)cuCtxPushCurrent_v2},
    {"cuCtxPushCurrent_v2", (void *)cuCtxPushCurrent_v2},
#if CUDA_VERSION >= 12050
    {"cuCtxRecordEvent", (void *)cuCtxRecordEvent},
#endif
    {"cuCtxResetPersistingL2Cache", (void *)cuCtxResetPersistingL2Cache},
    {"cuCtxSetCacheConfig", (void *)cuCtxSetCacheConfig},
    {"cuCtxSetCurrent", (void *)cuCtxSetCurrent},
    {"cuCtxSetLimit", (void *)cuCtxSetLimit},
    {"cuCtxSetSharedMemConfig", (void *)cuCtxSetSharedMemConfig},
    {"cuCtxSynchronize", (void *)cuCtxSynchronize},
#if CUDA_VERSION >= 13000
    {"cuCtxSynchronize_v2", (void *)cuCtxSynchronize_v2},
#endif
#if CUDA_VERSION >= 12050
    {"cuCtxWaitEvent", (void *)cuCtxWaitEvent},
#endif
    {"cuDestroyExternalMemory", (void *)cuDestroyExternalMemory},
    {"cuDestroyExternalSemaphore", (void *)cuDestroyExternalSemaphore},
#if CUDA_VERSION >= 12040
    {"cuDevResourceGenerateDesc", (void *)cuDevResourceGenerateDesc},
#endif
#if CUDA_VERSION >= 13010
    {"cuDevSmResourceSplit", (void *)cuDevSmResourceSplit},
#endif
#if CUDA_VERSION >= 12040
    {"cuDevSmResourceSplitByCount", (void *)cuDevSmResourceSplitByCount},
#endif
    {"cuDeviceCanAccessPeer", (void *)cuDeviceCanAccessPeer},
    {"cuDeviceComputeCapability", (void *)cuDeviceComputeCapability},
    {"cuDeviceGet", (void *)cuDeviceGet},
    {"cuDeviceGetAttribute", (void *)cuDeviceGetAttribute},
    {"cuDeviceGetByPCIBusId", (void *)cuDeviceGetByPCIBusId},
    {"cuDeviceGetCount", (void *)cuDeviceGetCount},
    {"cuDeviceGetDefaultMemPool", (void *)cuDeviceGetDefaultMemPool},
#if CUDA_VERSION >= 12040
    {"cuDeviceGetDevResource", (void *)cuDeviceGetDevResource},
#endif
    {"cuDeviceGetExecAffinitySupport", (void *)cuDeviceGetExecAffinitySupport},
    {"cuDeviceGetGraphMemAttribute", (void *)cuDeviceGetGraphMemAttribute},
    {"cuDeviceGetLuid", (void *)cuDeviceGetLuid},
    {"cuDeviceGetMemPool", (void *)cuDeviceGetMemPool},
    {"cuDeviceGetName", (void *)cuDeviceGetName},
    {"cuDeviceGetP2PAttribute", (void *)cuDeviceGetP2PAttribute},
    {"cuDeviceGetPCIBusId", (void *)cuDeviceGetPCIBusId},
    {"cuDeviceGetProperties", (void *)cuDeviceGetProperties},
    {"cuDeviceGetTexture1DLinearMaxWidth",
     (void *)cuDeviceGetTexture1DLinearMaxWidth},
    {"cuDeviceGetUuid", (void *)cuDeviceGetUuid_v2},
    {"cuDeviceGetUuid_v2", (void *)cuDeviceGetUuid_v2},
    {"cuDeviceGraphMemTrim", (void *)cuDeviceGraphMemTrim},
    {"cuDevicePrimaryCtxGetState", (void *)cuDevicePrimaryCtxGetState},
    {"cuDevicePrimaryCtxRelease", (void *)cuDevicePrimaryCtxRelease_v2},
    {"cuDevicePrimaryCtxRelease_v2", (void *)cuDevicePrimaryCtxRelease_v2},
    {"cuDevicePrimaryCtxReset", (void *)cuDevicePrimaryCtxReset_v2},
    {"cuDevicePrimaryCtxReset_v2", (void *)cuDevicePrimaryCtxReset_v2},
    {"cuDevicePrimaryCtxRetain", (void *)cuDevicePrimaryCtxRetain},
    {"cuDevicePrimaryCtxSetFlags", (void *)cuDevicePrimaryCtxSetFlags_v2},
    {"cuDevicePrimaryCtxSetFlags_v2", (void *)cuDevicePrimaryCtxSetFlags_v2},
    {"cuDeviceSetGraphMemAttribute", (void *)cuDeviceSetGraphMemAttribute},
    {"cuDeviceSetMemPool", (void *)cuDeviceSetMemPool},
    {"cuDeviceTotalMem", (void *)cuDeviceTotalMem_v2},
    {"cuDeviceTotalMem_v2", (void *)cuDeviceTotalMem_v2},
    {"cuDriverGetVersion", (void *)cuDriverGetVersion},
    {"cuEventCreate", (void *)cuEventCreate},
    {"cuEventDestroy", (void *)cuEventDestroy_v2},
    {"cuEventDestroy_v2", (void *)cuEventDestroy_v2},
    {"cuEventElapsedTime", (void *)cuEventElapsedTime_v2},
    {"cuEventElapsedTime_v2", (void *)cuEventElapsedTime_v2},
    {"cuEventQuery", (void *)cuEventQuery},
    {"cuEventRecord", (void *)cuEventRecord},
    {"cuEventRecordWithFlags", (void *)cuEventRecordWithFlags},
    {"cuEventRecordWithFlags_ptsz", (void *)cuEventRecordWithFlags},
    {"cuEventRecord_ptsz", (void *)cuEventRecord},
    {"cuEventSynchronize", (void *)cuEventSynchronize},
    {"cuExternalMemoryGetMappedBuffer",
     (void *)cuExternalMemoryGetMappedBuffer},
    {"cuExternalMemoryGetMappedMipmappedArray",
     (void *)cuExternalMemoryGetMappedMipmappedArray},
    {"cuFlushGPUDirectRDMAWrites", (void *)cuFlushGPUDirectRDMAWrites},
    {"cuFuncGetAttribute", (void *)cuFuncGetAttribute},
    {"cuFuncGetModule", (void *)cuFuncGetModule},
    {"cuFuncGetName", (void *)cuFuncGetName},
    {"cuFuncGetParamInfo", (void *)cuFuncGetParamInfo},
    {"cuFuncSetAttribute", (void *)cuFuncSetAttribute},
    {"cuFuncSetBlockShape", (void *)cuFuncSetBlockShape},
    {"cuFuncSetCacheConfig", (void *)cuFuncSetCacheConfig},
    {"cuFuncSetSharedMemConfig", (void *)cuFuncSetSharedMemConfig},
    {"cuFuncSetSharedSize", (void *)cuFuncSetSharedSize},
    {"cuGetErrorName", (void *)cuGetErrorName},
    {"cuGetErrorString", (void *)cuGetErrorString},
    {"cuGraphAddBatchMemOpNode", (void *)cuGraphAddBatchMemOpNode},
    {"cuGraphAddChildGraphNode", (void *)cuGraphAddChildGraphNode},
#if CUDA_VERSION >= 12030
    {"cuGraphAddDependencies", (void *)cuGraphAddDependencies_v2},
#endif
    {"cuGraphAddDependencies_v2", (void *)cuGraphAddDependencies_v2},
    {"cuGraphAddEmptyNode", (void *)cuGraphAddEmptyNode},
    {"cuGraphAddEventRecordNode", (void *)cuGraphAddEventRecordNode},
    {"cuGraphAddEventWaitNode", (void *)cuGraphAddEventWaitNode},
    {"cuGraphAddExternalSemaphoresSignalNode",
     (void *)cuGraphAddExternalSemaphoresSignalNode},
    {"cuGraphAddExternalSemaphoresWaitNode",
     (void *)cuGraphAddExternalSemaphoresWaitNode},
    {"cuGraphAddHostNode", (void *)cuGraphAddHostNode},
    {"cuGraphAddKernelNode", (void *)cuGraphAddKernelNode_v2},
    {"cuGraphAddKernelNode_v2", (void *)cuGraphAddKernelNode_v2},
    {"cuGraphAddMemAllocNode", (void *)cuGraphAddMemAllocNode},
    {"cuGraphAddMemFreeNode", (void *)cuGraphAddMemFreeNode},
    {"cuGraphAddMemcpyNode", (void *)cuGraphAddMemcpyNode},
    {"cuGraphAddMemsetNode", (void *)cuGraphAddMemsetNode},
    {"cuGraphBatchMemOpNodeGetParams", (void *)cuGraphBatchMemOpNodeGetParams},
    {"cuGraphBatchMemOpNodeSetParams", (void *)cuGraphBatchMemOpNodeSetParams},
    {"cuGraphChildGraphNodeGetGraph", (void *)cuGraphChildGraphNodeGetGraph},
    {"cuGraphClone", (void *)cuGraphClone},
    {"cuGraphConditionalHandleCreate", (void *)cuGraphConditionalHandleCreate},
    {"cuGraphCreate", (void *)cuGraphCreate},
    {"cuGraphDebugDotPrint", (void *)cuGraphDebugDotPrint},
    {"cuGraphDestroy", (void *)cuGraphDestroy},
    {"cuGraphDestroyNode", (void *)cuGraphDestroyNode},
    {"cuGraphEventRecordNodeGetEvent", (void *)cuGraphEventRecordNodeGetEvent},
    {"cuGraphEventRecordNodeSetEvent", (void *)cuGraphEventRecordNodeSetEvent},
    {"cuGraphEventWaitNodeGetEvent", (void *)cuGraphEventWaitNodeGetEvent},
    {"cuGraphEventWaitNodeSetEvent", (void *)cuGraphEventWaitNodeSetEvent},
    {"cuGraphExecBatchMemOpNodeSetParams",
     (void *)cuGraphExecBatchMemOpNodeSetParams},
    {"cuGraphExecChildGraphNodeSetParams",
     (void *)cuGraphExecChildGraphNodeSetParams},
    {"cuGraphExecDestroy", (void *)cuGraphExecDestroy},
    {"cuGraphExecEventRecordNodeSetEvent",
     (void *)cuGraphExecEventRecordNodeSetEvent},
    {"cuGraphExecEventWaitNodeSetEvent",
     (void *)cuGraphExecEventWaitNodeSetEvent},
    {"cuGraphExecExternalSemaphoresSignalNodeSetParams",
     (void *)cuGraphExecExternalSemaphoresSignalNodeSetParams},
    {"cuGraphExecExternalSemaphoresWaitNodeSetParams",
     (void *)cuGraphExecExternalSemaphoresWaitNodeSetParams},
    {"cuGraphExecGetFlags", (void *)cuGraphExecGetFlags},
#if CUDA_VERSION >= 13010
    {"cuGraphExecGetId", (void *)cuGraphExecGetId},
#endif
    {"cuGraphExecHostNodeSetParams", (void *)cuGraphExecHostNodeSetParams},
    {"cuGraphExecKernelNodeSetParams",
     (void *)cuGraphExecKernelNodeSetParams_v2},
    {"cuGraphExecKernelNodeSetParams_v2",
     (void *)cuGraphExecKernelNodeSetParams_v2},
    {"cuGraphExecMemcpyNodeSetParams", (void *)cuGraphExecMemcpyNodeSetParams},
    {"cuGraphExecMemsetNodeSetParams", (void *)cuGraphExecMemsetNodeSetParams},
#if CUDA_VERSION >= 12020
    {"cuGraphExecNodeSetParams", (void *)cuGraphExecNodeSetParams},
#endif
    {"cuGraphExecUpdate", (void *)cuGraphExecUpdate},
    {"cuGraphExecUpdate_v2", (void *)cuGraphExecUpdate_v2},
    {"cuGraphExternalSemaphoresSignalNodeGetParams",
     (void *)cuGraphExternalSemaphoresSignalNodeGetParams},
    {"cuGraphExternalSemaphoresSignalNodeSetParams",
     (void *)cuGraphExternalSemaphoresSignalNodeSetParams},
    {"cuGraphExternalSemaphoresWaitNodeGetParams",
     (void *)cuGraphExternalSemaphoresWaitNodeGetParams},
    {"cuGraphExternalSemaphoresWaitNodeSetParams",
     (void *)cuGraphExternalSemaphoresWaitNodeSetParams},
#if CUDA_VERSION >= 12030
    {"cuGraphGetEdges", (void *)cuGraphGetEdges_v2},
#endif
    {"cuGraphGetEdges_v2", (void *)cuGraphGetEdges_v2},
#if CUDA_VERSION >= 13010
    {"cuGraphGetId", (void *)cuGraphGetId},
#endif
    {"cuGraphGetNodes", (void *)cuGraphGetNodes},
    {"cuGraphGetRootNodes", (void *)cuGraphGetRootNodes},
    {"cuGraphHostNodeGetParams", (void *)cuGraphHostNodeGetParams},
    {"cuGraphHostNodeSetParams", (void *)cuGraphHostNodeSetParams},
    {"cuGraphInstantiateWithFlags", (void *)cuGraphInstantiateWithFlags},
    {"cuGraphInstantiateWithParams", (void *)cuGraphInstantiateWithParams},
    {"cuGraphInstantiateWithParams_ptsz", (void *)cuGraphInstantiateWithParams},
    {"cuGraphInstantiate_v2", (void *)cuGraphInstantiate_v2},
    {"cuGraphKernelNodeCopyAttributes",
     (void *)cuGraphKernelNodeCopyAttributes},
    {"cuGraphKernelNodeGetAttribute", (void *)cuGraphKernelNodeGetAttribute},
    {"cuGraphKernelNodeGetParams", (void *)cuGraphKernelNodeGetParams_v2},
    {"cuGraphKernelNodeGetParams_v2", (void *)cuGraphKernelNodeGetParams_v2},
    {"cuGraphKernelNodeSetAttribute", (void *)cuGraphKernelNodeSetAttribute},
    {"cuGraphKernelNodeSetParams", (void *)cuGraphKernelNodeSetParams_v2},
    {"cuGraphKernelNodeSetParams_v2", (void *)cuGraphKernelNodeSetParams_v2},
    {"cuGraphLaunch", (void *)cuGraphLaunch},
    {"cuGraphLaunch_ptsz", (void *)cuGraphLaunch},
    {"cuGraphMemAllocNodeGetParams", (void *)cuGraphMemAllocNodeGetParams},
    {"cuGraphMemFreeNodeGetParams", (void *)cuGraphMemFreeNodeGetParams},
    {"cuGraphMemcpyNodeGetParams", (void *)cuGraphMemcpyNodeGetParams},
    {"cuGraphMemcpyNodeSetParams", (void *)cuGraphMemcpyNodeSetParams},
    {"cuGraphMemsetNodeGetParams", (void *)cuGraphMemsetNodeGetParams},
    {"cuGraphMemsetNodeSetParams", (void *)cuGraphMemsetNodeSetParams},
    {"cuGraphNodeFindInClone", (void *)cuGraphNodeFindInClone},
#if CUDA_VERSION >= 13010
    {"cuGraphNodeGetContainingGraph", (void *)cuGraphNodeGetContainingGraph},
#endif
#if CUDA_VERSION >= 12030
    {"cuGraphNodeGetDependencies", (void *)cuGraphNodeGetDependencies_v2},
#endif
    {"cuGraphNodeGetDependencies_v2", (void *)cuGraphNodeGetDependencies_v2},
#if CUDA_VERSION >= 12030
    {"cuGraphNodeGetDependentNodes", (void *)cuGraphNodeGetDependentNodes_v2},
#endif
    {"cuGraphNodeGetDependentNodes_v2",
     (void *)cuGraphNodeGetDependentNodes_v2},
    {"cuGraphNodeGetEnabled", (void *)cuGraphNodeGetEnabled},
#if CUDA_VERSION >= 13010
    {"cuGraphNodeGetLocalId", (void *)cuGraphNodeGetLocalId},
#endif
#if CUDA_VERSION >= 13010
    {"cuGraphNodeGetToolsId", (void *)cuGraphNodeGetToolsId},
#endif
    {"cuGraphNodeGetType", (void *)cuGraphNodeGetType},
    {"cuGraphNodeSetEnabled", (void *)cuGraphNodeSetEnabled},
#if CUDA_VERSION >= 12020
    {"cuGraphNodeSetParams", (void *)cuGraphNodeSetParams},
#endif
    {"cuGraphReleaseUserObject", (void *)cuGraphReleaseUserObject},
#if CUDA_VERSION >= 12030
    {"cuGraphRemoveDependencies", (void *)cuGraphRemoveDependencies_v2},
#endif
    {"cuGraphRemoveDependencies_v2", (void *)cuGraphRemoveDependencies_v2},
    {"cuGraphRetainUserObject", (void *)cuGraphRetainUserObject},
    {"cuGraphUpload", (void *)cuGraphUpload},
    {"cuGraphUpload_ptsz", (void *)cuGraphUpload},
    {"cuGraphicsMapResources", (void *)cuGraphicsMapResources},
    {"cuGraphicsMapResources_ptsz", (void *)cuGraphicsMapResources},
    {"cuGraphicsResourceGetMappedMipmappedArray",
     (void *)cuGraphicsResourceGetMappedMipmappedArray},
    {"cuGraphicsResourceGetMappedPointer_v2",
     (void *)cuGraphicsResourceGetMappedPointer_v2},
    {"cuGraphicsResourceSetMapFlags_v2",
     (void *)cuGraphicsResourceSetMapFlags_v2},
    {"cuGraphicsSubResourceGetMappedArray",
     (void *)cuGraphicsSubResourceGetMappedArray},
    {"cuGraphicsUnmapResources", (void *)cuGraphicsUnmapResources},
    {"cuGraphicsUnmapResources_ptsz", (void *)cuGraphicsUnmapResources},
    {"cuGraphicsUnregisterResource", (void *)cuGraphicsUnregisterResource},
#if CUDA_VERSION >= 12040
    {"cuGreenCtxCreate", (void *)cuGreenCtxCreate},
#endif
#if CUDA_VERSION >= 12040
    {"cuGreenCtxDestroy", (void *)cuGreenCtxDestroy},
#endif
#if CUDA_VERSION >= 12040
    {"cuGreenCtxGetDevResource", (void *)cuGreenCtxGetDevResource},
#endif
#if CUDA_VERSION >= 13000
    {"cuGreenCtxGetId", (void *)cuGreenCtxGetId},
#endif
#if CUDA_VERSION >= 12040
    {"cuGreenCtxRecordEvent", (void *)cuGreenCtxRecordEvent},
#endif
#if CUDA_VERSION >= 12050
    {"cuGreenCtxStreamCreate", (void *)cuGreenCtxStreamCreate},
#endif
#if CUDA_VERSION >= 12040
    {"cuGreenCtxWaitEvent", (void *)cuGreenCtxWaitEvent},
#endif
    {"cuImportExternalMemory", (void *)cuImportExternalMemory},
    {"cuImportExternalSemaphore", (void *)cuImportExternalSemaphore},
    {"cuInit", (void *)cuInit},
    {"cuIpcCloseMemHandle", (void *)cuIpcCloseMemHandle},
    {"cuIpcGetEventHandle", (void *)cuIpcGetEventHandle},
    {"cuIpcGetMemHandle", (void *)cuIpcGetMemHandle},
    {"cuIpcOpenEventHandle", (void *)cuIpcOpenEventHandle},
    {"cuIpcOpenMemHandle", (void *)cuIpcOpenMemHandle_v2},
    {"cuIpcOpenMemHandle_v2", (void *)cuIpcOpenMemHandle_v2},
    {"cuKernelGetAttribute", (void *)cuKernelGetAttribute},
    {"cuKernelGetFunction", (void *)cuKernelGetFunction},
    {"cuKernelGetLibrary", (void *)cuKernelGetLibrary},
    {"cuKernelGetName", (void *)cuKernelGetName},
    {"cuKernelGetParamInfo", (void *)cuKernelGetParamInfo},
    {"cuKernelSetAttribute", (void *)cuKernelSetAttribute},
    {"cuKernelSetCacheConfig", (void *)cuKernelSetCacheConfig},
    {"cuLaunch", (void *)cuLaunch},
    {"cuLaunchCooperativeKernel", (void *)cuLaunchCooperativeKernel},
    {"cuLaunchCooperativeKernelMultiDevice",
     (void *)cuLaunchCooperativeKernelMultiDevice},
    {"cuLaunchCooperativeKernel_ptsz", (void *)cuLaunchCooperativeKernel},
    {"cuLaunchGrid", (void *)cuLaunchGrid},
    {"cuLaunchGridAsync", (void *)cuLaunchGridAsync},
    {"cuLaunchHostFunc", (void *)cuLaunchHostFunc},
    {"cuLaunchHostFunc_ptsz", (void *)cuLaunchHostFunc},
    {"cuLaunchKernel", (void *)cuLaunchKernel},
    {"cuLaunchKernelEx", (void *)cuLaunchKernelEx},
    {"cuLaunchKernelEx_ptsz", (void *)cuLaunchKernelEx},
    {"cuLaunchKernel_ptsz", (void *)cuLaunchKernel},
    {"cuLibraryGetGlobal", (void *)cuLibraryGetGlobal},
    {"cuLibraryGetKernel", (void *)cuLibraryGetKernel},
    {"cuLibraryGetManaged", (void *)cuLibraryGetManaged},
    {"cuLibraryGetModule", (void *)cuLibraryGetModule},
    {"cuLibraryGetUnifiedFunction", (void *)cuLibraryGetUnifiedFunction},
    {"cuLibraryLoadData", (void *)cuLibraryLoadData},
    {"cuLibraryUnload", (void *)cuLibraryUnload},
    {"cuLinkAddData", (void *)cuLinkAddData_v2},
    {"cuLinkAddData_v2", (void *)cuLinkAddData_v2},
    {"cuLinkAddFile", (void *)cuLinkAddFile_v2},
    {"cuLinkAddFile_v2", (void *)cuLinkAddFile_v2},
    {"cuLinkComplete", (void *)cuLinkComplete},
    {"cuLinkCreate", (void *)cuLinkCreate_v2},
    {"cuLinkCreate_v2", (void *)cuLinkCreate_v2},
    {"cuLinkDestroy", (void *)cuLinkDestroy},
#if CUDA_VERSION >= 12090
    {"cuLogsCurrent", (void *)cuLogsCurrent},
#endif
#if CUDA_VERSION >= 12090
    {"cuLogsDumpToFile", (void *)cuLogsDumpToFile},
#endif
#if CUDA_VERSION >= 12090
    {"cuLogsDumpToMemory", (void *)cuLogsDumpToMemory},
#endif
#if CUDA_VERSION >= 12090
    {"cuLogsRegisterCallback", (void *)cuLogsRegisterCallback},
#endif
#if CUDA_VERSION >= 12090
    {"cuLogsUnregisterCallback", (void *)cuLogsUnregisterCallback},
#endif
    {"cuMemAddressFree", (void *)cuMemAddressFree},
    {"cuMemAddressReserve", (void *)cuMemAddressReserve},
    {"cuMemAdvise", (void *)cuMemAdvise},
#if CUDA_VERSION >= 12020
    {"cuMemAdvise_v2", (void *)cuMemAdvise_v2},
#endif
    {"cuMemAlloc", (void *)cuMemAlloc_v2},
    {"cuMemAllocAsync", (void *)cuMemAllocAsync},
    {"cuMemAllocAsync_ptsz", (void *)cuMemAllocAsync},
    {"cuMemAllocFromPoolAsync", (void *)cuMemAllocFromPoolAsync},
    {"cuMemAllocFromPoolAsync_ptsz", (void *)cuMemAllocFromPoolAsync},
    {"cuMemAllocHost", (void *)cuMemAllocHost_v2},
    {"cuMemAllocHost_v2", (void *)cuMemAllocHost_v2},
    {"cuMemAllocManaged", (void *)cuMemAllocManaged},
    {"cuMemAllocPitch", (void *)cuMemAllocPitch_v2},
    {"cuMemAllocPitch_v2", (void *)cuMemAllocPitch_v2},
    {"cuMemAlloc_v2", (void *)cuMemAlloc_v2},
    {"cuMemCreate", (void *)cuMemCreate},
    {"cuMemExportToShareableHandle", (void *)cuMemExportToShareableHandle},
    {"cuMemFree", (void *)cuMemFree_v2},
    {"cuMemFreeAsync", (void *)cuMemFreeAsync},
    {"cuMemFreeAsync_ptsz", (void *)cuMemFreeAsync},
    {"cuMemFreeHost", (void *)cuMemFreeHost},
    {"cuMemFree_v2", (void *)cuMemFree_v2},
    {"cuMemGetAccess", (void *)cuMemGetAccess},
    {"cuMemGetAddressRange", (void *)cuMemGetAddressRange_v2},
    {"cuMemGetAddressRange_v2", (void *)cuMemGetAddressRange_v2},
    {"cuMemGetAllocationGranularity", (void *)cuMemGetAllocationGranularity},
    {"cuMemGetAllocationPropertiesFromHandle",
     (void *)cuMemGetAllocationPropertiesFromHandle},
    {"cuMemGetHandleForAddressRange", (void *)cuMemGetHandleForAddressRange},
    {"cuMemGetInfo", (void *)cuMemGetInfo_v2},
    {"cuMemGetInfo_v2", (void *)cuMemGetInfo_v2},
    {"cuMemHostAlloc", (void *)cuMemHostAlloc},
    {"cuMemHostGetDevicePointer", (void *)cuMemHostGetDevicePointer_v2},
    {"cuMemHostGetDevicePointer_v2", (void *)cuMemHostGetDevicePointer_v2},
    {"cuMemHostGetFlags", (void *)cuMemHostGetFlags},
    {"cuMemImportFromShareableHandle", (void *)cuMemImportFromShareableHandle},
    {"cuMemMap", (void *)cuMemMap},
    {"cuMemMapArrayAsync", (void *)cuMemMapArrayAsync},
    {"cuMemMapArrayAsync_ptsz", (void *)cuMemMapArrayAsync},
    {"cuMemPoolCreate", (void *)cuMemPoolCreate},
    {"cuMemPoolDestroy", (void *)cuMemPoolDestroy},
    {"cuMemPoolExportPointer", (void *)cuMemPoolExportPointer},
    {"cuMemPoolExportToShareableHandle",
     (void *)cuMemPoolExportToShareableHandle},
    {"cuMemPoolGetAccess", (void *)cuMemPoolGetAccess},
    {"cuMemPoolGetAttribute", (void *)cuMemPoolGetAttribute},
    {"cuMemPoolImportFromShareableHandle",
     (void *)cuMemPoolImportFromShareableHandle},
    {"cuMemPoolImportPointer", (void *)cuMemPoolImportPointer},
    {"cuMemPoolSetAccess", (void *)cuMemPoolSetAccess},
    {"cuMemPoolSetAttribute", (void *)cuMemPoolSetAttribute},
    {"cuMemPoolTrimTo", (void *)cuMemPoolTrimTo},
    {"cuMemPrefetchAsync", (void *)cuMemPrefetchAsync},
#if CUDA_VERSION >= 12020
    {"cuMemPrefetchAsync_v2", (void *)cuMemPrefetchAsync_v2},
#endif
    {"cuMemRangeGetAttribute", (void *)cuMemRangeGetAttribute},
    {"cuMemRangeGetAttributes", (void *)cuMemRangeGetAttributes},
    {"cuMemRelease", (void *)cuMemRelease},
    {"cuMemRetainAllocationHandle", (void *)cuMemRetainAllocationHandle},
    {"cuMemSetAccess", (void *)cuMemSetAccess},
    {"cuMemUnmap", (void *)cuMemUnmap},
    {"cuMemcpy", (void *)cuMemcpy},
    {"cuMemcpy2D", (void *)cuMemcpy2D_v2},
    {"cuMemcpy2DAsync", (void *)cuMemcpy2DAsync_v2},
    {"cuMemcpy2DAsync_ptsz", (void *)cuMemcpy2DAsync_v2},
    {"cuMemcpy2DAsync_v2", (void *)cuMemcpy2DAsync_v2},
    {"cuMemcpy2DUnaligned", (void *)cuMemcpy2DUnaligned_v2},
    {"cuMemcpy2DUnaligned_v2", (void *)cuMemcpy2DUnaligned_v2},
    {"cuMemcpy2D_v2", (void *)cuMemcpy2D_v2},
    {"cuMemcpy3D", (void *)cuMemcpy3D_v2},
    {"cuMemcpy3DAsync", (void *)cuMemcpy3DAsync_v2},
    {"cuMemcpy3DAsync_ptsz", (void *)cuMemcpy3DAsync_v2},
    {"cuMemcpy3DAsync_v2", (void *)cuMemcpy3DAsync_v2},
    {"cuMemcpy3DPeer", (void *)cuMemcpy3DPeer},
    {"cuMemcpy3DPeerAsync", (void *)cuMemcpy3DPeerAsync},
    {"cuMemcpy3DPeerAsync_ptsz", (void *)cuMemcpy3DPeerAsync},
    {"cuMemcpy3DPeer_ptds", (void *)cuMemcpy3DPeer},
    {"cuMemcpy3D_v2", (void *)cuMemcpy3D_v2},
    {"cuMemcpyAtoA", (void *)cuMemcpyAtoA_v2},
    {"cuMemcpyAtoA_v2", (void *)cuMemcpyAtoA_v2},
    {"cuMemcpyAtoD", (void *)cuMemcpyAtoD_v2},
    {"cuMemcpyAtoD_v2", (void *)cuMemcpyAtoD_v2},
    {"cuMemcpyAtoH", (void *)cuMemcpyAtoH_v2},
    {"cuMemcpyAtoHAsync", (void *)cuMemcpyAtoHAsync_v2},
    {"cuMemcpyAtoHAsync_v2", (void *)cuMemcpyAtoHAsync_v2},
    {"cuMemcpyAtoHAsync_v2_ptsz", (void *)cuMemcpyAtoHAsync_v2},
    {"cuMemcpyAtoH_v2", (void *)cuMemcpyAtoH_v2},
    {"cuMemcpyAtoH_v2_ptds", (void *)cuMemcpyAtoH_v2},
    {"cuMemcpyDtoA", (void *)cuMemcpyDtoA_v2},
    {"cuMemcpyDtoA_v2", (void *)cuMemcpyDtoA_v2},
    {"cuMemcpyDtoD", (void *)cuMemcpyDtoD_v2},
    {"cuMemcpyDtoDAsync", (void *)cuMemcpyDtoDAsync_v2},
    {"cuMemcpyDtoDAsync_v2", (void *)cuMemcpyDtoDAsync_v2},
    {"cuMemcpyDtoD_v2", (void *)cuMemcpyDtoD_v2},
    {"cuMemcpyDtoH", (void *)cuMemcpyDtoH_v2},
    {"cuMemcpyDtoHAsync", (void *)cuMemcpyDtoHAsync_v2},
    {"cuMemcpyDtoHAsync_v2", (void *)cuMemcpyDtoHAsync_v2},
    {"cuMemcpyDtoH_v2", (void *)cuMemcpyDtoH_v2},
    {"cuMemcpyHtoA", (void *)cuMemcpyHtoA_v2},
    {"cuMemcpyHtoAAsync", (void *)cuMemcpyHtoAAsync_v2},
    {"cuMemcpyHtoAAsync_v2", (void *)cuMemcpyHtoAAsync_v2},
    {"cuMemcpyHtoAAsync_v2_ptsz", (void *)cuMemcpyHtoAAsync_v2},
    {"cuMemcpyHtoA_v2", (void *)cuMemcpyHtoA_v2},
    {"cuMemcpyHtoA_v2_ptds", (void *)cuMemcpyHtoA_v2},
    {"cuMemcpyHtoD", (void *)cuMemcpyHtoD_v2},
    {"cuMemcpyHtoDAsync", (void *)cuMemcpyHtoDAsync_v2},
    {"cuMemcpyHtoDAsync_v2", (void *)cuMemcpyHtoDAsync_v2},
    {"cuMemcpyHtoD_v2", (void *)cuMemcpyHtoD_v2},
    {"cuMemcpyPeer", (void *)cuMemcpyPeer},
    {"cuMemcpyPeerAsync", (void *)cuMemcpyPeerAsync},
    {"cuMemcpyPeerAsync_ptsz", (void *)cuMemcpyPeerAsync},
    {"cuMemcpyPeer_ptds", (void *)cuMemcpyPeer},
    {"cuMemsetD16Async", (void *)cuMemsetD16Async},
    {"cuMemsetD16Async_ptsz", (void *)cuMemsetD16Async},
    {"cuMemsetD16_v2", (void *)cuMemsetD16_v2},
    {"cuMemsetD2D16", (void *)cuMemsetD2D16_v2},
    {"cuMemsetD2D16Async", (void *)cuMemsetD2D16Async},
    {"cuMemsetD2D16Async_ptsz", (void *)cuMemsetD2D16Async},
    {"cuMemsetD2D16_v2", (void *)cuMemsetD2D16_v2},
    {"cuMemsetD2D32", (void *)cuMemsetD2D32_v2},
    {"cuMemsetD2D32Async", (void *)cuMemsetD2D32Async},
    {"cuMemsetD2D32Async_ptsz", (void *)cuMemsetD2D32Async},
    {"cuMemsetD2D32_v2", (void *)cuMemsetD2D32_v2},
    {"cuMemsetD2D8", (void *)cuMemsetD2D8_v2},
    {"cuMemsetD2D8Async", (void *)cuMemsetD2D8Async},
    {"cuMemsetD2D8Async_ptsz", (void *)cuMemsetD2D8Async},
    {"cuMemsetD2D8_v2", (void *)cuMemsetD2D8_v2},
    {"cuMemsetD32Async", (void *)cuMemsetD32Async},
    {"cuMemsetD32Async_ptsz", (void *)cuMemsetD32Async},
    {"cuMemsetD32_v2", (void *)cuMemsetD32_v2},
    {"cuMemsetD8", (void *)cuMemsetD8_v2},
    {"cuMemsetD8Async", (void *)cuMemsetD8Async},
    {"cuMemsetD8Async_ptsz", (void *)cuMemsetD8Async},
    {"cuMemsetD8_v2", (void *)cuMemsetD8_v2},
    {"cuMipmappedArrayCreate", (void *)cuMipmappedArrayCreate},
    {"cuMipmappedArrayDestroy", (void *)cuMipmappedArrayDestroy},
    {"cuMipmappedArrayGetLevel", (void *)cuMipmappedArrayGetLevel},
    {"cuMipmappedArrayGetMemoryRequirements",
     (void *)cuMipmappedArrayGetMemoryRequirements},
    {"cuMipmappedArrayGetSparseProperties",
     (void *)cuMipmappedArrayGetSparseProperties},
    {"cuModuleGetFunction", (void *)cuModuleGetFunction},
    {"cuModuleGetGlobal", (void *)cuModuleGetGlobal_v2},
    {"cuModuleGetGlobal_v2", (void *)cuModuleGetGlobal_v2},
    {"cuModuleGetLoadingMode", (void *)cuModuleGetLoadingMode},
    {"cuModuleGetSurfRef", (void *)cuModuleGetSurfRef},
    {"cuModuleGetTexRef", (void *)cuModuleGetTexRef},
    {"cuModuleLoad", (void *)cuModuleLoad},
    {"cuModuleLoadData", (void *)cuModuleLoadData},
    {"cuModuleLoadDataEx", (void *)cuModuleLoadDataEx},
    {"cuModuleUnload", (void *)cuModuleUnload},
    {"cuOccupancyAvailableDynamicSMemPerBlock",
     (void *)cuOccupancyAvailableDynamicSMemPerBlock},
    {"cuOccupancyMaxActiveBlocksPerMultiprocessor",
     (void *)cuOccupancyMaxActiveBlocksPerMultiprocessor},
    {"cuOccupancyMaxActiveBlocksPerMultiprocessorWithFlags",
     (void *)cuOccupancyMaxActiveBlocksPerMultiprocessorWithFlags},
    {"cuOccupancyMaxActiveClusters", (void *)cuOccupancyMaxActiveClusters},
    {"cuOccupancyMaxPotentialBlockSize",
     (void *)cuOccupancyMaxPotentialBlockSize},
    {"cuOccupancyMaxPotentialBlockSizeWithFlags",
     (void *)cuOccupancyMaxPotentialBlockSizeWithFlags},
    {"cuOccupancyMaxPotentialClusterSize",
     (void *)cuOccupancyMaxPotentialClusterSize},
    {"cuParamSetSize", (void *)cuParamSetSize},
    {"cuParamSetTexRef", (void *)cuParamSetTexRef},
    {"cuParamSetf", (void *)cuParamSetf},
    {"cuParamSeti", (void *)cuParamSeti},
    {"cuParamSetv", (void *)cuParamSetv},
    {"cuPointerGetAttribute", (void *)cuPointerGetAttribute},
    {"cuPointerGetAttributes", (void *)cuPointerGetAttributes},
    {"cuPointerSetAttribute", (void *)cuPointerSetAttribute},
    {"cuSignalExternalSemaphoresAsync",
     (void *)cuSignalExternalSemaphoresAsync},
    {"cuSignalExternalSemaphoresAsync_ptsz",
     (void *)cuSignalExternalSemaphoresAsync},
    {"cuStreamAddCallback", (void *)cuStreamAddCallback},
    {"cuStreamAddCallback_ptsz", (void *)cuStreamAddCallback},
    {"cuStreamAttachMemAsync", (void *)cuStreamAttachMemAsync},
    {"cuStreamAttachMemAsync_ptsz", (void *)cuStreamAttachMemAsync},
    {"cuStreamBatchMemOp", (void *)cuStreamBatchMemOp_v2},
    {"cuStreamBatchMemOp_v2", (void *)cuStreamBatchMemOp_v2},
    {"cuStreamBeginCapture", (void *)cuStreamBeginCapture_v2},
    {"cuStreamBeginCapture_v2", (void *)cuStreamBeginCapture_v2},
    {"cuStreamCopyAttributes", (void *)cuStreamCopyAttributes},
    {"cuStreamCopyAttributes_ptsz", (void *)cuStreamCopyAttributes},
    {"cuStreamCreate", (void *)cuStreamCreate},
    {"cuStreamCreateWithPriority", (void *)cuStreamCreateWithPriority},
    {"cuStreamDestroy", (void *)cuStreamDestroy_v2},
    {"cuStreamDestroy_v2", (void *)cuStreamDestroy_v2},
    {"cuStreamEndCapture", (void *)cuStreamEndCapture},
    {"cuStreamEndCapture_ptsz", (void *)cuStreamEndCapture},
    {"cuStreamGetAttribute", (void *)cuStreamGetAttribute},
    {"cuStreamGetAttribute_ptsz", (void *)cuStreamGetAttribute},
    {"cuStreamGetCtx", (void *)cuStreamGetCtx},
    {"cuStreamGetCtx_ptsz", (void *)cuStreamGetCtx},
#if CUDA_VERSION >= 13010
    {"cuStreamGetDevResource", (void *)cuStreamGetDevResource},
#endif
    {"cuStreamGetDevice", (void *)cuStreamGetDevice},
    {"cuStreamGetDevice_ptsz", (void *)cuStreamGetDevice},
    {"cuStreamGetFlags", (void *)cuStreamGetFlags},
    {"cuStreamGetFlags_ptsz", (void *)cuStreamGetFlags},
#if CUDA_VERSION >= 12040
    {"cuStreamGetGreenCtx", (void *)cuStreamGetGreenCtx},
#endif
    {"cuStreamGetId", (void *)cuStreamGetId},
    {"cuStreamGetId_ptsz", (void *)cuStreamGetId},
    {"cuStreamGetPriority", (void *)cuStreamGetPriority},
    {"cuStreamGetPriority_ptsz", (void *)cuStreamGetPriority},
    {"cuStreamIsCapturing", (void *)cuStreamIsCapturing},
    {"cuStreamIsCapturing_ptsz", (void *)cuStreamIsCapturing},
    {"cuStreamQuery", (void *)cuStreamQuery},
    {"cuStreamQuery_ptsz", (void *)cuStreamQuery},
    {"cuStreamSetAttribute", (void *)cuStreamSetAttribute},
    {"cuStreamSetAttribute_ptsz", (void *)cuStreamSetAttribute},
    {"cuStreamSynchronize", (void *)cuStreamSynchronize},
    {"cuStreamSynchronize_ptsz", (void *)cuStreamSynchronize},
#if CUDA_VERSION >= 12030
    {"cuStreamUpdateCaptureDependencies",
     (void *)cuStreamUpdateCaptureDependencies_v2},
#endif
#if CUDA_VERSION >= 12030
    {"cuStreamUpdateCaptureDependencies_ptsz",
     (void *)cuStreamUpdateCaptureDependencies_v2},
#endif
    {"cuStreamUpdateCaptureDependencies_v2",
     (void *)cuStreamUpdateCaptureDependencies_v2},
    {"cuStreamUpdateCaptureDependencies_v2_ptsz",
     (void *)cuStreamUpdateCaptureDependencies_v2},
    {"cuStreamWaitEvent", (void *)cuStreamWaitEvent},
    {"cuStreamWaitEvent_ptsz", (void *)cuStreamWaitEvent},
    {"cuStreamWaitValue32", (void *)cuStreamWaitValue32_v2},
    {"cuStreamWaitValue32_v2", (void *)cuStreamWaitValue32_v2},
    {"cuStreamWaitValue64", (void *)cuStreamWaitValue64_v2},
    {"cuStreamWaitValue64_v2", (void *)cuStreamWaitValue64_v2},
    {"cuStreamWriteValue32", (void *)cuStreamWriteValue32_v2},
    {"cuStreamWriteValue32_v2", (void *)cuStreamWriteValue32_v2},
    {"cuStreamWriteValue64", (void *)cuStreamWriteValue64_v2},
    {"cuStreamWriteValue64_v2", (void *)cuStreamWriteValue64_v2},
    {"cuSurfObjectCreate", (void *)cuSurfObjectCreate},
    {"cuSurfObjectDestroy", (void *)cuSurfObjectDestroy},
    {"cuSurfObjectGetResourceDesc", (void *)cuSurfObjectGetResourceDesc},
    {"cuSurfRefGetArray", (void *)cuSurfRefGetArray},
    {"cuSurfRefSetArray", (void *)cuSurfRefSetArray},
#if CUDA_VERSION >= 12000
    {"cuTensorMapEncodeIm2col", (void *)cuTensorMapEncodeIm2col},
#endif
#if CUDA_VERSION >= 12000
    {"cuTensorMapEncodeTiled", (void *)cuTensorMapEncodeTiled},
#endif
#if CUDA_VERSION >= 12000
    {"cuTensorMapReplaceAddress", (void *)cuTensorMapReplaceAddress},
#endif
    {"cuTexObjectCreate", (void *)cuTexObjectCreate},
    {"cuTexObjectDestroy", (void *)cuTexObjectDestroy},
    {"cuTexObjectGetResourceDesc", (void *)cuTexObjectGetResourceDesc},
    {"cuTexObjectGetResourceViewDesc", (void *)cuTexObjectGetResourceViewDesc},
    {"cuTexObjectGetTextureDesc", (void *)cuTexObjectGetTextureDesc},
    {"cuTexRefCreate", (void *)cuTexRefCreate},
    {"cuTexRefDestroy", (void *)cuTexRefDestroy},
    {"cuTexRefGetAddressMode", (void *)cuTexRefGetAddressMode},
    {"cuTexRefGetAddress_v2", (void *)cuTexRefGetAddress_v2},
    {"cuTexRefGetArray", (void *)cuTexRefGetArray},
    {"cuTexRefGetBorderColor", (void *)cuTexRefGetBorderColor},
    {"cuTexRefGetFilterMode", (void *)cuTexRefGetFilterMode},
    {"cuTexRefGetFlags", (void *)cuTexRefGetFlags},
    {"cuTexRefGetFormat", (void *)cuTexRefGetFormat},
    {"cuTexRefGetMaxAnisotropy", (void *)cuTexRefGetMaxAnisotropy},
    {"cuTexRefGetMipmapFilterMode", (void *)cuTexRefGetMipmapFilterMode},
    {"cuTexRefGetMipmapLevelBias", (void *)cuTexRefGetMipmapLevelBias},
    {"cuTexRefGetMipmapLevelClamp", (void *)cuTexRefGetMipmapLevelClamp},
    {"cuTexRefGetMipmappedArray", (void *)cuTexRefGetMipmappedArray},
    {"cuTexRefSetAddress2D_v3", (void *)cuTexRefSetAddress2D_v3},
    {"cuTexRefSetAddressMode", (void *)cuTexRefSetAddressMode},
    {"cuTexRefSetAddress_v2", (void *)cuTexRefSetAddress_v2},
    {"cuTexRefSetArray", (void *)cuTexRefSetArray},
    {"cuTexRefSetBorderColor", (void *)cuTexRefSetBorderColor},
    {"cuTexRefSetFilterMode", (void *)cuTexRefSetFilterMode},
    {"cuTexRefSetFlags", (void *)cuTexRefSetFlags},
    {"cuTexRefSetFormat", (void *)cuTexRefSetFormat},
    {"cuTexRefSetMaxAnisotropy", (void *)cuTexRefSetMaxAnisotropy},
    {"cuTexRefSetMipmapFilterMode", (void *)cuTexRefSetMipmapFilterMode},
    {"cuTexRefSetMipmapLevelBias", (void *)cuTexRefSetMipmapLevelBias},
    {"cuTexRefSetMipmapLevelClamp", (void *)cuTexRefSetMipmapLevelClamp},
    {"cuTexRefSetMipmappedArray", (void *)cuTexRefSetMipmappedArray},
    {"cuThreadExchangeStreamCaptureMode",
     (void *)cuThreadExchangeStreamCaptureMode},
    {"cuUserObjectRelease", (void *)cuUserObjectRelease},
    {"cuUserObjectRetain", (void *)cuUserObjectRetain},
    {"cuWaitExternalSemaphoresAsync", (void *)cuWaitExternalSemaphoresAsync},
    {"cuWaitExternalSemaphoresAsync_ptsz",
     (void *)cuWaitExternalSemaphoresAsync},
};
} // namespace

void *get_function_pointer(const char *name) {
  if (name == nullptr)
    return nullptr;
  constexpr size_t count = sizeof(kFunctionMap) / sizeof(kFunctionMap[0]);
  const auto *it = std::lower_bound(
      kFunctionMap, kFunctionMap + count, name,
      [](const lupine_cuda_function_entry &entry, const char *key) {
        return std::strcmp(entry.name, key) < 0;
      });
  if (it == kFunctionMap + count || std::strcmp(it->name, name) != 0)
    return nullptr;
  return it->pointer;
}
