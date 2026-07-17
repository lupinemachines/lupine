#pragma once

#include <cstddef>

#define LUPINE_CUDA_COMPAT_TYPES_ONLY
#include "cuda_compat.h"
#undef LUPINE_CUDA_COMPAT_TYPES_ONLY

#include "rpc.h"
#include "pointer_translation.h"

struct lupine_kernel_param_layout;

static constexpr int LUPINE_ROUTE_REMOTE = 0;
static constexpr int LUPINE_ROUTE_LOCAL = 1;
static constexpr int LUPINE_ROUTE_INVALID = 2;

struct lupine_route {
  int kind = LUPINE_ROUTE_INVALID;
  conn_t *conn = nullptr;
};

lupine_route lupine_remote_route_for_conn(conn_t *conn);
int lupine_route_identity(lupine_route route);
lupine_route lupine_route_from_identity(int route_id);
int lupine_known_deviceptr_route_id(CUdeviceptr ptr);
lupine_route lupine_route_from_known_kernel_deviceptr_args(
    void *const *kernel_params, const lupine_kernel_param_layout &layout,
    lupine_route fallback);
conn_t *lupine_thread_conn_by_index(unsigned int index);
CUresult lupine_virtual_device_count(int *count);
CUresult lupine_virtual_device_for_ordinal(CUdevice *device, int ordinal);
CUresult lupine_set_current_context_on_route(lupine_route route, CUcontext ctx);
bool lupine_local_cuda_available();
CUcontext lupine_current_context_hint();
CUcontext lupine_default_context_hint_value();
CUcontext lupine_global_default_context_hint_value();
void lupine_accept_current_context_hint(CUcontext ctx);

extern "C" lupine_route lupine_route_for_default();
extern "C" lupine_route lupine_route_for_device(CUdevice *device);
extern "C" lupine_route lupine_route_for_current_context();
extern "C" lupine_route lupine_route_for_context(CUcontext ctx);
extern "C" lupine_route lupine_route_for_module(CUmodule module);
extern "C" lupine_route lupine_route_for_library(CUlibrary library);
extern "C" lupine_route lupine_route_for_function(CUfunction function);
extern "C" lupine_route lupine_route_for_stream(CUstream stream);
extern "C" lupine_route lupine_route_for_event(CUevent event);
extern "C" lupine_route lupine_route_for_memory_pool(CUmemoryPool pool);
extern "C" lupine_route lupine_route_for_graph(CUgraph graph);
extern "C" lupine_route lupine_route_for_graph_node(CUgraphNode node);
extern "C" lupine_route lupine_route_for_graph_exec(CUgraphExec exec);
extern "C" lupine_route lupine_route_for_deviceptr(CUdeviceptr ptr);

extern "C" conn_t *lupine_rpc_conn_for_device(CUdevice *device);
extern "C" conn_t *lupine_rpc_conn_for_current_context();
extern "C" conn_t *lupine_rpc_conn_for_context(CUcontext ctx);
extern "C" conn_t *lupine_rpc_conn_for_module(CUmodule module);
extern "C" conn_t *lupine_rpc_conn_for_function(CUfunction function);
extern "C" conn_t *lupine_rpc_conn_for_stream(CUstream stream);
extern "C" conn_t *lupine_rpc_conn_for_event(CUevent event);
extern "C" conn_t *lupine_rpc_conn_for_deviceptr(CUdeviceptr ptr);

extern "C" bool lupine_route_is_local(lupine_route route);
extern "C" conn_t *lupine_route_remote_conn(lupine_route route);
extern "C" bool lupine_routes_share_server(lupine_route first,
                                           lupine_route second);
extern "C" bool lupine_deviceptrs_share_route(CUdeviceptr first,
                                              CUdeviceptr second);
extern "C" bool lupine_translate_device_for_conn(conn_t *conn,
                                                 CUdevice *device);
extern "C" CUdevice lupine_local_device_for_remote(conn_t *conn,
                                                   CUdevice remote_device);

using lupine_device_lookup_callback = CUresult (*)(void *context,
                                                   lupine_route route,
                                                   CUdevice *route_device);
extern "C" CUresult
lupine_lookup_device_on_all_routes_impl(CUdevice *device, void *context,
                                        lupine_device_lookup_callback lookup);

template <typename Lookup>
static CUresult lupine_lookup_device_on_all_routes(CUdevice *device,
                                                   Lookup lookup) {
  return lupine_lookup_device_on_all_routes_impl(
      device, &lookup,
      [](void *context, lupine_route route, CUdevice *route_device) {
        return (*static_cast<Lookup *>(context))(route, route_device);
      });
}

extern "C" void *lupine_real_cuda_symbol(const char *name);
extern "C" bool lupine_local_cuda_symbol_if_routed(lupine_route route,
                                                   const char *symbol,
                                                   void **symbol_out);

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

extern "C" void lupine_note_context_owner_route(CUcontext ctx,
                                                lupine_route route);
extern "C" void lupine_note_module_owner_route(CUmodule module,
                                               lupine_route route);
extern "C" void lupine_note_library_owner_route(CUlibrary library,
                                                lupine_route route);
extern "C" void lupine_note_function_owner_route(CUfunction function,
                                                 lupine_route route);
extern "C" void lupine_note_stream_owner_route(CUstream stream,
                                               lupine_route route);
extern "C" void lupine_note_event_owner_route(CUevent event,
                                              lupine_route route);
extern "C" void lupine_note_memory_pool_owner_route(CUmemoryPool pool,
                                                    lupine_route route);
extern "C" void lupine_note_graph_owner_route(CUgraph graph,
                                              lupine_route route);
extern "C" void lupine_note_graph_node_owner_route(CUgraphNode node,
                                                   lupine_route route);
extern "C" void lupine_note_graph_exec_owner_route(CUgraphExec exec,
                                                   lupine_route route);
extern "C" void lupine_note_deviceptr_owner_route(CUdeviceptr ptr,
                                                  lupine_route route);
extern "C" void lupine_note_deviceptr_allocation_route(CUdeviceptr ptr,
                                                       size_t size,
                                                       lupine_route route);

template <typename Fn> static Fn lupine_real_cuda_fn(const char *name) {
  return reinterpret_cast<Fn>(lupine_real_cuda_symbol(name));
}

template <typename T> static T lupine_translate_local_call_arg(T value) {
  return value;
}

static CUdeviceptr lupine_translate_local_call_arg(CUdeviceptr value) {
  CUdeviceptr translated = value;
  (void)lupine_translate_device_pointer(value, &translated);
  return translated;
}

template <typename Fn, typename... Args>
static bool lupine_call_local_cuda_if_routed(lupine_route route,
                                             const char *symbol,
                                             CUresult *result, Args... args) {
  void *local_symbol = nullptr;
  if (!lupine_local_cuda_symbol_if_routed(route, symbol, &local_symbol)) {
    return false;
  }
  auto real = reinterpret_cast<Fn>(local_symbol);
  *result = real == nullptr
                ? CUDA_ERROR_DEVICE_UNAVAILABLE
                : real(lupine_translate_local_call_arg(args)...);
  return true;
}
