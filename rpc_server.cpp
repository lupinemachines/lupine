#include "rpc_server.h"

#include "checkpoint.h"
#include "lupine_log.h"

bool rpc_server_validate(const rpc_handler_registry &handlers) {
  for (const auto &entry : handlers) {
    if (entry.second.handler == nullptr) {
      return false;
    }
    switch (entry.second.backend) {
    case rpc_backend::cuda:
    case rpc_backend::nvml:
      break;
    default:
      return false;
    }
  }
  return true;
}

int rpc_server_dispatch(const rpc_handler_registry &handlers, conn_t *conn,
                        int op) {
  LUPINE_TRACE_LOG("LUPINE server handling op " << op);
  auto it = handlers.find(op);
  if (it == handlers.end()) {
    LUPINE_LOG_ERROR("No RPC handler for op " << op << "; closing client.");
    return -1;
  }
  const rpc_handler &handler = it->second;
  const char *backend_name;
  int result;
  switch (handler.backend) {
  case rpc_backend::cuda: {
    backend_name = "CUDA";
    lupine_checkpoint::cuda_call_guard guard;
    result = handler.handler(conn);
    break;
  }
  case rpc_backend::nvml:
    backend_name = "NVML";
    result = handler.handler(conn);
    break;
  default:
    LUPINE_LOG_ERROR("Invalid RPC backend for op " << op << ".");
    return -1;
  }
  if (result >= 0) {
    return 0;
  }
  LUPINE_LOG_ERROR("Error handling " << backend_name << " request for op " << op
                                     << ".");
  return -1;
}
