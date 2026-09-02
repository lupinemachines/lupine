#include "cublas_bridge.h"

#include <cuda.h>

#include "client_routing.h"
#include "rpc.h"

static int default_route() {
  return lupine_route_identity(lupine_route_for_default());
}

static int stream_route(void *stream) {
  return lupine_route_identity(
      lupine_route_for_stream(reinterpret_cast<CUstream>(stream)));
}

static conn_t *connection(int route_id) {
  return lupine_route_remote_conn(lupine_route_from_identity(route_id));
}

static const lupine_cublas_rpc_api api = {
    LUPINE_CUBLAS_RPC_API_VERSION,
    default_route,
    stream_route,
    connection,
    lupine_prepare_rpc,
    rpc_write_start_request,
    rpc_write,
    rpc_wait_for_response,
    rpc_read,
    rpc_read_end,
};

extern "C" LUPINE_CUBLAS_BRIDGE_API const lupine_cublas_rpc_api *
lupine_cublas_rpc_api_v1() {
  return &api;
}
