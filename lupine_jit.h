#pragma once

#include <cstdlib>

#include <cuda.h>

#include "rpc.h"

// JIT option marshalling, shared by the hand-written loaders in
// cuda_server.cpp and the generated handlers in gen_cuda_server.cpp. The
// caller's log buffer pointers are client addresses, so they are replaced with
// server-side storage before the driver sees them; every option value word the
// driver rewrites in place is echoed back untouched.
struct lupine_jit_state {
  unsigned int num_options = 0;
  CUjit_option *options = nullptr;
  void **option_values = nullptr;
  size_t info_log_size = 0;
  char *info_log = nullptr;
  size_t error_log_size = 0;
  char *error_log = nullptr;
};

int lupine_read_jit_options(conn_t *conn, lupine_jit_state *jit);
int lupine_write_jit_outputs(conn_t *conn, lupine_jit_state *jit);

// Owns what lupine_read_jit_options allocates, so a handler can leave through
// an error label without a cleanup list.
struct lupine_jit_options {
  lupine_jit_state state;

  ~lupine_jit_options() {
    std::free(state.options);
    std::free(state.option_values);
    std::free(state.info_log);
    std::free(state.error_log);
  }
};
