#ifndef LUPINE_LIBRARY_LOGGING_SERVER_H
#define LUPINE_LIBRARY_LOGGING_SERVER_H

#include "library_logging.h"
#include "rpc.h"
#include <mutex>

// NVIDIA's library callbacks have no user-data argument. Each server child
// owns one library instance, so its process-wide registration holds the route.
struct library_logger {
  std::mutex update_mutex;
  std::mutex mutex;
  conn_t *conn = nullptr;
  library_log_target target;

  void set(conn_t *connection, library_log_target next) {
    std::lock_guard<std::mutex> lock(mutex);
    conn = connection;
    target = next;
  }

  void emit(int level, const char *function, const char *message,
            size_t length) {
    std::lock_guard<std::mutex> lock(mutex);
    if (conn != nullptr && target.callback != nullptr) {
      lupine_send_library_log(conn, target, level, function, message, length);
    }
  }

  void clear(conn_t *connection) {
    std::lock_guard<std::mutex> lock(mutex);
    if (conn == connection) {
      conn = nullptr;
      target = {};
    }
  }
};

inline int lupine_read_library_log_target(conn_t *conn,
                                          library_log_target *target) {
  if (rpc_read(conn, &target->callback, sizeof(target->callback)) < 0 ||
      rpc_read(conn, &target->user_data, sizeof(target->user_data)) < 0) {
    return -1;
  }
  return rpc_read_end(conn);
}

#endif
