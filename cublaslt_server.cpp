// cublasLt.h brings cublas_api.h with the umbrella marker it wants, and the
// runtime header supplies the stream type.
#include <cublasLt.h>
#include <cuda_runtime_api.h>

#include <cstdint>
#include <cstring>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#ifdef _WIN32
#include <windows.h>
#else
#include <dlfcn.h>
#endif

#include "codegen/gen_rpc_ids.h"
#include "cuda_client_rpc.h"
#include "rpc.h"

namespace {

cublasStatus_t function_not_found() { return CUBLAS_STATUS_NOT_SUPPORTED; }

// The library this server was compiled against: descriptors travel as the
// opaque structs of that header, so another major on the machine is not a
// match.
void *cublaslt_library() {
  static void *library = []() -> void * {
    const std::string major = std::to_string(CUBLAS_VER_MAJOR);
#ifdef _WIN32
    const std::string name = "cublasLt64_" + major + ".dll";
    return reinterpret_cast<void *>(LoadLibraryA(name.c_str()));
#else
    const std::string name = "libcublasLt.so." + major;
    return dlopen(name.c_str(), RTLD_LAZY | RTLD_LOCAL);
#endif
  }();
  return library;
}

template <typename Fn> Fn cublaslt_symbol(const char *name) {
  void *lib = cublaslt_library();
  if (lib == nullptr) {
    return nullptr;
  }
#ifdef _WIN32
  return reinterpret_cast<Fn>(GetProcAddress(static_cast<HMODULE>(lib), name));
#else
  return reinterpret_cast<Fn>(dlsym(lib, name));
#endif
}

// cublasLtGetStatusName and cublasLtGetStatusString return static strings, so
// the response carries the bytes length-prefixed instead of a status.
int handle_status_text(conn_t *conn, const char *symbol) {
  cublasStatus_t status;
  if (rpc_read(conn, &status, sizeof(status)) < 0) {
    return -1;
  }
  int request_id = rpc_read_end(conn);
  if (request_id < 0) {
    return -1;
  }
  using fn_t = const char *(*)(cublasStatus_t);
  fn_t fn = cublaslt_symbol<fn_t>(symbol);
  const char *text = fn == nullptr ? nullptr : fn(status);
  uint32_t length = text == nullptr ? 0 : static_cast<uint32_t>(strlen(text));
  if (rpc_write_start_response(conn, request_id) < 0 ||
      rpc_write(conn, &length, sizeof(length)) < 0 ||
      rpc_write(conn, text, length) < 0 || rpc_write_end(conn) < 0) {
    return -1;
  }
  return 0;
}

} // namespace

int handle_cublasLtGetStatusName(conn_t *conn) {
  return handle_status_text(conn, "cublasLtGetStatusName");
}

int handle_cublasLtGetStatusString(conn_t *conn) {
  return handle_status_text(conn, "cublasLtGetStatusString");
}

namespace {
std::mutex logger_update_mutex;
std::mutex logger_mutex;
conn_t *logger_conn = nullptr;
library_log_target logger_target;
library_log_target logger_file_target;

void log_callback(int level, const char *function, const char *message) {
  std::lock_guard<std::mutex> lock(logger_mutex);
  conn_t *conn = logger_conn;
  if (conn == nullptr) {
    return;
  }
  size_t length = message == nullptr ? 0 : std::strlen(message);
  library_log_target targets[] = {logger_file_target, logger_target};
  for (int sink = 0; sink != 2; ++sink) {
    auto target = targets[sink];
    if (target.callback == nullptr) {
      continue;
    }
    size_t function_size = function == nullptr ? 0 : std::strlen(function);
    if (length > LUPINE_MAX_LIBRARY_LOG_BYTES ||
        function_size > LUPINE_MAX_LIBRARY_LOG_BYTES) {
      return;
    }
    uint32_t function_length = static_cast<uint32_t>(function_size);
    uint32_t message_length = static_cast<uint32_t>(length);
    int32_t origin_stream = rpc_current_http2_stream(conn);
    uint8_t before_callbacks = sink == 0;
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
      return;
    }
  }
}

int set_log_target(conn_t *conn, bool file_output) {
  library_log_target target;
  if (rpc_read(conn, &target.callback, sizeof(target.callback)) < 0 ||
      rpc_read(conn, &target.user_data, sizeof(target.user_data)) < 0) {
    return -1;
  }
  int request_id = rpc_read_end(conn);
  if (request_id < 0) {
    return -1;
  }
  cublasStatus_t status = function_not_found();
  {
    std::lock_guard<std::mutex> update(logger_update_mutex);
    auto set_callback = cublaslt_symbol<decltype(&cublasLtLoggerSetCallback)>(
        "cublasLtLoggerSetCallback");
    auto set_file = cublaslt_symbol<decltype(&cublasLtLoggerSetFile)>(
        "cublasLtLoggerSetFile");
    bool enabled = target.callback != nullptr;
    {
      std::lock_guard<std::mutex> lock(logger_mutex);
      enabled |= (file_output ? logger_target.callback
                              : logger_file_target.callback) != nullptr;
    }
    if (set_callback != nullptr && (!file_output || set_file != nullptr)) {
      // Disable NVIDIA's server-side file sink. Its callback carries both
      // application callbacks and client file output, independently.
      status = file_output ? set_file(nullptr) : CUBLAS_STATUS_SUCCESS;
      if (status == CUBLAS_STATUS_SUCCESS) {
        status = set_callback(enabled ? log_callback : nullptr);
      }
      if (status == CUBLAS_STATUS_SUCCESS) {
        std::lock_guard<std::mutex> lock(logger_mutex);
        logger_conn = conn;
        if (file_output) {
          logger_file_target = target;
        } else {
          logger_target = target;
        }
      }
    }
  }
  if (rpc_write_start_response(conn, request_id) < 0 ||
      rpc_write(conn, &status, sizeof(status)) < 0 || rpc_write_end(conn) < 0) {
    return -1;
  }
  return 0;
}
} // namespace

int handle_cublasLtLoggerSetCallback(conn_t *conn) {
  return set_log_target(conn, false);
}

int handle_cublasLtLoggerSetFile(conn_t *conn) {
  return set_log_target(conn, true);
}

void lupine_cublaslt_cleanup_logs(conn_t *conn) {
  std::lock_guard<std::mutex> update(logger_update_mutex);
  {
    std::lock_guard<std::mutex> lock(logger_mutex);
    if (logger_conn == conn) {
      logger_conn = nullptr;
      logger_target = {};
      logger_file_target = {};
    }
  }
}

#if CUBLAS_VERSION >= 130100
namespace {

// The emulation descriptors clients attach to matmul descriptors, by the
// client's address. A matmul descriptor holds its copy's address, so a copy is
// refreshed in place and kept for the life of the connection process.
std::mutex emulation_copies_mutex;
std::unordered_map<uint64_t, std::unique_ptr<cublasLtEmulationDescOpaque_t>>
    emulation_copies;

} // namespace

int handle_lupineCublasLtEmulationDescCopy(conn_t *conn) {
  uint64_t original;
  cublasLtEmulationDescOpaque_t contents;
  if (rpc_read(conn, &original, sizeof(original)) < 0 ||
      rpc_read(conn, &contents, sizeof(contents)) < 0) {
    return -1;
  }
  int request_id = rpc_read_end(conn);
  if (request_id < 0) {
    return -1;
  }
  cublasLtEmulationDescOpaque_t *copy;
  {
    std::lock_guard<std::mutex> lock(emulation_copies_mutex);
    std::unique_ptr<cublasLtEmulationDescOpaque_t> &slot =
        emulation_copies[original];
    if (!slot) {
      slot.reset(new cublasLtEmulationDescOpaque_t());
    }
    *slot = contents;
    copy = slot.get();
  }
  if (rpc_write_start_response(conn, request_id) < 0 ||
      rpc_write(conn, &copy, sizeof(copy)) < 0 || rpc_write_end(conn) < 0) {
    return -1;
  }
  return 0;
}
#endif

#include "codegen/gen_cublaslt_server.inc"
