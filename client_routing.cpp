#include "client_routing.h"

extern "C" bool lupine_route_is_local(lupine_route route) {
  return route.kind == LUPINE_ROUTE_LOCAL;
}

extern "C" conn_t *lupine_route_remote_conn(lupine_route route) {
  return route.kind == LUPINE_ROUTE_REMOTE ? route.conn : nullptr;
}

extern "C" bool lupine_local_cuda_symbol_if_routed(lupine_route route,
                                                   const char *symbol,
                                                   void **symbol_out) {
  if (!lupine_route_is_local(route)) {
    return false;
  }
  if (symbol_out != nullptr) {
    *symbol_out = lupine_real_cuda_symbol(symbol);
  }
  return true;
}
