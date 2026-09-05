#include "rpc_server.h"

bool rpc_server_validate(const rpc_handler_registry &handlers) {
  for (const auto &entry : handlers) {
    if (entry.second.handler == nullptr) {
      return false;
    }
    switch (entry.second.backend) {
    case rpc_backend::cuda:
    case rpc_backend::cudart:
    case rpc_backend::cublas:
    case rpc_backend::cufft:
    case rpc_backend::nvml:
    case rpc_backend::hip:
      break;
    default:
      return false;
    }
  }
  return true;
}
