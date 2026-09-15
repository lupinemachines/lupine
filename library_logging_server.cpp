#include "library_logging.h"
#include "rpc.h"
#include <cstring>

int lupine_send_library_log(conn_t *conn, library_log_target target, int level,
                            const char *function, const char *message,
                            size_t length, bool file_output) {
  size_t function_size = function == nullptr ? 0 : std::strlen(function);
  if (length > LUPINE_MAX_LIBRARY_LOG_BYTES ||
      function_size > LUPINE_MAX_LIBRARY_LOG_BYTES) {
    return -1;
  }
  uint32_t function_length = static_cast<uint32_t>(function_size);
  uint32_t message_length = static_cast<uint32_t>(length);
  int32_t origin_stream = rpc_current_http2_stream(conn);
  uint8_t before_callbacks = file_output;
  void *response = nullptr;
  if (rpc_write_start_request(conn, LUPINE_SIDE_EFFECT_LIBRARY_LOG) < 0 ||
      rpc_write(conn, &target.callback, sizeof(target.callback)) < 0 ||
      rpc_write(conn, &target.user_data, sizeof(target.user_data)) < 0 ||
      rpc_write(conn, &origin_stream, sizeof(origin_stream)) < 0 ||
      rpc_write(conn, &before_callbacks, sizeof(before_callbacks)) < 0 ||
      rpc_write(conn, &level, sizeof(level)) < 0 ||
      rpc_write(conn, &function_length, sizeof(function_length)) < 0 ||
      rpc_write(conn, &message_length, sizeof(message_length)) < 0 ||
      rpc_write(conn, function, function_length) < 0 ||
      rpc_write(conn, message, message_length) < 0 ||
      rpc_wait_for_response(conn) < 0 ||
      rpc_read(conn, &response, sizeof(response)) < 0 ||
      rpc_read_end(conn) < 0) {
    return -1;
  }
  return 0;
}
