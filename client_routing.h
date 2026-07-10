#pragma once

#include <cstddef>

#define LUPINE_CUDA_COMPAT_TYPES_ONLY
#include "cuda_compat.h"
#undef LUPINE_CUDA_COMPAT_TYPES_ONLY

#include "rpc.h"

static constexpr int LUPINE_ROUTE_REMOTE = 0;
static constexpr int LUPINE_ROUTE_LOCAL = 1;
static constexpr int LUPINE_ROUTE_INVALID = 2;

struct lupine_route {
  int kind = LUPINE_ROUTE_INVALID;
  conn_t *conn = nullptr;
};

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

extern "C" bool lupine_route_is_local(lupine_route route);
extern "C" conn_t *lupine_route_remote_conn(lupine_route route);

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

template <typename Fn, typename... Args>
static bool lupine_call_local_cuda_if_routed(lupine_route route,
                                             const char *symbol,
                                             CUresult *result, Args... args) {
  void *local_symbol = nullptr;
  if (!lupine_local_cuda_symbol_if_routed(route, symbol, &local_symbol)) {
    return false;
  }
  auto real = reinterpret_cast<Fn>(local_symbol);
  *result = real == nullptr ? CUDA_ERROR_DEVICE_UNAVAILABLE : real(args...);
  return true;
}
