#include "library_logging.h"
#include "pending_log_callbacks.h"
#include "rpc.h"
#include <string>

int lupine_read_library_log(conn_t *conn, pending_log_callbacks &pending) {
  library_log_target target;
  int level = 0;
  int32_t origin_stream = -1;
  uint8_t before_callbacks = 0;
  uint32_t function_length = 0, length = 0;
  if (rpc_read(conn, &target.callback, sizeof(target.callback)) < 0 ||
      rpc_read(conn, &target.user_data, sizeof(target.user_data)) < 0 ||
      rpc_read(conn, &origin_stream, sizeof(origin_stream)) < 0 ||
      rpc_read(conn, &before_callbacks, sizeof(before_callbacks)) < 0 ||
      rpc_read(conn, &level, sizeof(level)) < 0 ||
      rpc_read(conn, &function_length, sizeof(function_length)) < 0 ||
      rpc_read(conn, &length, sizeof(length)) < 0 ||
      function_length > LUPINE_MAX_LIBRARY_LOG_BYTES ||
      length > LUPINE_MAX_LIBRARY_LOG_BYTES) {
    return -1;
  }
  std::string function, message;
  try {
    function.resize(function_length);
    message.resize(length);
  } catch (...) {
    return -1;
  }
  if (rpc_read(conn, function.data(), function_length) < 0 ||
      rpc_read(conn, message.data(), length) < 0) {
    return -1;
  }
  int32_t callback_stream = rpc_current_http2_stream(conn);
  int request_id = rpc_read_end(conn);
  if (request_id < 0 || target.callback == nullptr) {
    return -1;
  }
  auto invoke = [target, level, function = std::move(function),
                 message = std::move(message)] {
    target.callback(target.user_data, level, function.c_str(), message.c_str(),
                    message.size());
  };
  if (origin_stream < 0 || origin_stream == callback_stream) {
    invoke();
  } else if (!pending.enqueue(conn, origin_stream, std::move(invoke),
                              before_callbacks != 0)) {
    return -1;
  }
  void *response = nullptr;
  if (rpc_write_start_response(conn, request_id) < 0 ||
      rpc_write(conn, &response, sizeof(response)) < 0 ||
      rpc_write_end(conn) < 0) {
    return -1;
  }
  return 0;
}
