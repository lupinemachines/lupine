#ifndef LUPINE_LIBRARY_LOGGING_CLIENT_H
#define LUPINE_LIBRARY_LOGGING_CLIENT_H

#include "cuda_client_rpc.h"
#include "library_logging.h"

inline cublasStatus_t lupine_set_library_log_target(conn_t *conn, int op,
                                                    library_log_target target) {
  cublasStatus_t status = CUBLAS_STATUS_NOT_INITIALIZED;
  if (conn == nullptr || lupine_rpc_write_start_request(conn, op) < 0 ||
      lupine_rpc_write(conn, &target.callback, sizeof(target.callback)) < 0 ||
      lupine_rpc_write(conn, &target.user_data, sizeof(target.user_data)) < 0 ||
      lupine_rpc_wait_for_response(conn) < 0 ||
      lupine_rpc_read(conn, &status, sizeof(status)) < 0 ||
      lupine_rpc_read_end(conn) < 0) {
    return CUBLAS_STATUS_NOT_INITIALIZED;
  }
  return status;
}

#endif
