#include "rpc_server.h"

#include "checkpoint.h"
#include "codegen/gen_server.h"
#include "copy_pipeline.h"
#include "server_checkpoint.h"

#include <array>

rpc_handler lupine_cuda_server_handler(int op);
const int *lupine_cuda_server_operations(size_t *count);

namespace {

int dispatch_cuda(const rpc_handler &handler, conn_t *conn) {
  lupine_checkpoint::cuda_call_guard guard;
  return handler.handler(conn);
}

int dispatch_direct(const rpc_handler &handler, conn_t *conn) {
  return handler.handler(conn);
}

bool cuda_connection_ready(conn_t *, const char *session_id) {
  return lupine_server_checkpoint_connection_ready(session_id);
}

const rpc_backend cuda_backend = {
    "CUDA",
    lupine_cuda_server_handler,
    lupine_cuda_server_operations,
    lupine_server_checkpoint_child_start,
    lupine_server_checkpoint_child_finish,
    lupine_server_initialize_connection,
    cuda_connection_ready,
    dispatch_cuda,
    lupine_server_cleanup_connection,
};

const rpc_backend nvml_backend = {
    "NVML",  get_nvml_handler, get_nvml_handler_operations,
    nullptr, nullptr,          nullptr,
    nullptr, dispatch_direct,  nullptr,
};

const std::array<const rpc_backend *, 2> compiled_backends = {
    &cuda_backend,
    &nvml_backend,
};

} // namespace

const rpc_backend *const *lupine_compiled_backends(size_t *count) {
  *count = compiled_backends.size();
  return compiled_backends.data();
}
