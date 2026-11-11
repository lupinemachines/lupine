#ifndef LUPINE_RPC_SERVER_H
#define LUPINE_RPC_SERVER_H

#include "rpc.h"

typedef int (*RequestHandler)(conn_t *conn);

// The NCCL handlers run in the CUDA connection process: the same child owns
// the driver state the collectives operate on.
enum class rpc_backend { cuda, nvml, hip, nccl };

struct rpc_handler {
  RequestHandler handler = nullptr;
  rpc_backend backend = rpc_backend::cuda;
};

const rpc_handler *lupine_rpc_handler_lookup(int op);

#endif
