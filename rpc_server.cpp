#include "rpc_server.h"

bool rpc_server_validate(const rpc_handler_registry &handlers) {
  for (const auto &entry : handlers) {
    if (entry.second.handler == nullptr) {
      return false;
    }
    switch (entry.second.backend) {
    case rpc_backend::cuda:
    case rpc_backend::cublas:
    case rpc_backend::cufft:
    case rpc_backend::cudnn:
    case rpc_backend::curand:
    case rpc_backend::cusparse:
    case rpc_backend::cusolver:
    case rpc_backend::nvrtc:
    case rpc_backend::nccl:
    case rpc_backend::nvjitlink:
    case rpc_backend::nvjpeg:
    case rpc_backend::npp:
    case rpc_backend::nvml:
    case rpc_backend::hip:
    case rpc_backend::cudart:
      break;
    default:
      return false;
    }
  }
  return true;
}
