#ifndef LUPINE_RPC_SERVER_H
#define LUPINE_RPC_SERVER_H

#include "rpc.h"

#include <unordered_map>

typedef int (*RequestHandler)(conn_t *conn);

// cudart, cublas and cufft handlers run in the CUDA connection process: the
// same child owns the driver state the runtime and library calls operate on.
enum class rpc_backend { cuda, cudart, cublas, cufft, nvml, hip };

struct rpc_handler {
  RequestHandler handler = nullptr;
  rpc_backend backend = rpc_backend::cuda;
};

using rpc_handler_registry = std::unordered_map<int, rpc_handler>;

bool rpc_server_validate(const rpc_handler_registry &handlers);
int rpc_server_dispatch(const rpc_handler_registry &handlers, conn_t *conn,
                        int op);

const rpc_handler_registry &lupine_rpc_handlers();

#endif
