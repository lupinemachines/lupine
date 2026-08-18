#include "cuda_rpc.h"

#include <algorithm>

static const rpc_jit_output_binding *
rpc_find_jit_output_binding(const std::vector<rpc_jit_output_binding> &bindings,
                            CUjit_option option) {
  for (const auto &binding : bindings) {
    if (binding.option == option && binding.dst != nullptr) {
      return &binding;
    }
  }
  return nullptr;
}

int rpc_read_jit_outputs(conn_t *conn,
                         const std::vector<rpc_jit_output_binding> &bindings) {
  const auto *binding = rpc_find_jit_output_binding(bindings, CU_JIT_WALL_TIME);
  size_t direct_size =
      binding == nullptr ? 0 : std::min(binding->size, sizeof(float));
  if ((direct_size != 0 && rpc_read(conn, binding->dst, direct_size) < 0) ||
      rpc_drain(conn, sizeof(float) - direct_size) < 0) {
    return -1;
  }

  size_t payload_size = 0;
  if (rpc_read(conn, &payload_size, sizeof(payload_size)) < 0) {
    return -1;
  }
  binding = rpc_find_jit_output_binding(bindings, CU_JIT_INFO_LOG_BUFFER);
  direct_size = binding == nullptr ? 0 : std::min(binding->size, payload_size);
  if ((direct_size != 0 && rpc_read(conn, binding->dst, direct_size) < 0) ||
      rpc_drain(conn, payload_size - direct_size) < 0) {
    return -1;
  }

  if (rpc_read(conn, &payload_size, sizeof(payload_size)) < 0) {
    return -1;
  }
  binding = rpc_find_jit_output_binding(bindings, CU_JIT_ERROR_LOG_BUFFER);
  direct_size = binding == nullptr ? 0 : std::min(binding->size, payload_size);
  if ((direct_size != 0 && rpc_read(conn, binding->dst, direct_size) < 0) ||
      rpc_drain(conn, payload_size - direct_size) < 0) {
    return -1;
  }
  return 0;
}
