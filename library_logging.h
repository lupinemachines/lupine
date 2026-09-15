#ifndef LUPINE_LIBRARY_LOGGING_H
#define LUPINE_LIBRARY_LOGGING_H

#include <cstddef>
#include <cstdint>

struct conn_t;

// Only the client invokes this function and dereferences user_data. The
// server carries both as opaque tokens, just like CUDA's log callback bridge.
using library_log_callback = void (*)(void *user_data, int level,
                                      const char *function, const char *message,
                                      size_t length);
struct library_log_target {
  library_log_callback callback = nullptr;
  void *user_data = nullptr;
};

constexpr uint32_t LUPINE_MAX_LIBRARY_LOG_BYTES = 4 * 1024 * 1024;

class pending_log_callbacks;
int lupine_read_library_log(conn_t *conn, pending_log_callbacks &pending);

int lupine_send_library_log(conn_t *conn, library_log_target target, int level,
                            const char *function, const char *message,
                            size_t length, bool file_output = false);
int lupine_cublas_flush_logs(conn_t *conn);
void lupine_cublas_cleanup_logs(conn_t *conn);
void lupine_cublaslt_cleanup_logs(conn_t *conn);

#endif
