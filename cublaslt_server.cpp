// cublasLt.h brings cublas_api.h with the umbrella marker it wants, and the
// runtime header supplies the stream type.
#include <cublasLt.h>
#include <cuda_runtime_api.h>

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

#ifdef _WIN32
#include <windows.h>
#else
#include <dlfcn.h>
#endif

#include "codegen/gen_rpc_ids.h"
#include "library_logging_server.h"
#include "rpc.h"
#include <cstdio>
#ifdef _WIN32
#include <io.h>
#else
#include <unistd.h>
#endif

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
  if (lupine_cublas_flush_logs(conn) < 0 ||
      rpc_write_start_response(conn, request_id) < 0 ||
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
library_logger logger;
std::mutex log_file_mutex;
FILE *log_staging = nullptr;
std::atomic<conn_t *> log_file_conn{nullptr};
library_log_target log_file_target;

void log_callback(int level, const char *function, const char *message) {
  logger.emit(level, function, message,
              message == nullptr ? 0 : std::strlen(message));
}

class file_lock {
public:
  explicit file_lock(FILE *file) : file_(file) {
#ifdef _WIN32
    _lock_file(file_);
#else
    flockfile(file_);
#endif
  }
  ~file_lock() {
#ifdef _WIN32
    _unlock_file(file_);
#else
    funlockfile(file_);
#endif
  }

private:
  FILE *file_;
};

// Called before a library response is sent, including cuBLAS calls that use
// cuBLASLt internally. Keep NVIDIA's exact formatted output, and truncate the
// staging file after every drain so long-running clients don't grow it forever.
// The FILE lock also excludes writes from NVIDIA's own worker threads. File
// bytes are delivered before user callbacks, which may replace and close the
// file when the API response is consumed.
int flush_log_file_locked(conn_t *conn) {
  if (log_staging == nullptr || log_file_conn != conn) {
    return 0;
  }
  file_lock lock(log_staging);
  if (std::fflush(log_staging) != 0) {
    return -1;
  }
  long length = std::ftell(log_staging);
  if (length < 0 || std::fseek(log_staging, 0, SEEK_SET) != 0) {
    return -1;
  }
  char buffer[64 * 1024];
  while (length > 0) {
    size_t chunk = std::min(static_cast<size_t>(length), sizeof(buffer));
    if (std::fread(buffer, 1, chunk, log_staging) != chunk ||
        lupine_send_library_log(conn, log_file_target, 0, "", buffer, chunk,
                                true) < 0) {
      return -1;
    }
    length -= static_cast<long>(chunk);
  }
#ifdef _WIN32
  if (_chsize_s(_fileno(log_staging), 0) != 0) {
#else
  if (ftruncate(fileno(log_staging), 0) != 0) {
#endif
    return -1;
  }
  return std::fseek(log_staging, 0, SEEK_SET);
}

int write_log_status(conn_t *conn, int request_id, cublasStatus_t status) {
  if (lupine_cublas_flush_logs(conn) < 0 ||
      rpc_write_start_response(conn, request_id) < 0 ||
      rpc_write(conn, &status, sizeof(status)) < 0 || rpc_write_end(conn) < 0) {
    return -1;
  }
  return 0;
}
} // namespace

int lupine_cublas_flush_logs(conn_t *conn) {
  if (log_file_conn.load(std::memory_order_acquire) != conn) {
    return 0;
  }
  std::lock_guard<std::mutex> lock(log_file_mutex);
  return flush_log_file_locked(conn);
}

int handle_cublasLtLoggerSetCallback(conn_t *conn) {
  library_log_target target;
  int request_id = lupine_read_library_log_target(conn, &target);
  if (request_id < 0) {
    return -1;
  }
  cublasStatus_t status;
  {
    std::lock_guard<std::mutex> update(logger.update_mutex);
    auto fn = cublaslt_symbol<decltype(&cublasLtLoggerSetCallback)>(
        "cublasLtLoggerSetCallback");
    status = fn == nullptr
                 ? function_not_found()
                 : fn(target.callback == nullptr ? nullptr : log_callback);
    if (status == CUBLAS_STATUS_SUCCESS) {
      logger.set(conn, target);
    }
  }
  return write_log_status(conn, request_id, status);
}

int handle_cublasLtLoggerSetFile(conn_t *conn) {
  library_log_target target;
  int request_id = lupine_read_library_log_target(conn, &target);
  if (request_id < 0) {
    return -1;
  }
  cublasStatus_t status = function_not_found();
  {
    std::lock_guard<std::mutex> lock(log_file_mutex);
    auto fn = cublaslt_symbol<decltype(&cublasLtLoggerSetFile)>(
        "cublasLtLoggerSetFile");
    if (fn != nullptr) {
      FILE *next = target.callback == nullptr ? nullptr : std::tmpfile();
      if (target.callback != nullptr && next == nullptr) {
        status = CUBLAS_STATUS_ALLOC_FAILED;
      } else {
        status = fn(next);
        // SetFile itself can log to the old sink. Deliver that before allowing
        // the client to close its previous FILE, even when the setter fails.
        int flushed = flush_log_file_locked(conn);
        if (status == CUBLAS_STATUS_SUCCESS) {
          if (log_staging != nullptr) {
            std::fclose(log_staging);
          }
          log_staging = next;
          log_file_conn = next == nullptr ? nullptr : conn;
          log_file_target = target;
        } else if (next != nullptr) {
          std::fclose(next);
        }
        if (flushed < 0) {
          return -1;
        }
      }
    }
  }
  return write_log_status(conn, request_id, status);
}

int handle_cublasLtLoggerOpenFile(conn_t *conn) {
  uint8_t has_name = 0;
  uint32_t length = 0;
  if (rpc_read(conn, &has_name, sizeof(has_name)) < 0 ||
      rpc_read(conn, &length, sizeof(length)) < 0 ||
      length > LUPINE_MAX_LIBRARY_LOG_BYTES) {
    return -1;
  }
  std::string name(length, '\0');
  if (rpc_read(conn, name.data(), length) < 0) {
    return -1;
  }
  int request_id = rpc_read_end(conn);
  if (request_id < 0) {
    return -1;
  }
  cublasStatus_t status;
  {
    std::lock_guard<std::mutex> lock(log_file_mutex);
    auto fn = cublaslt_symbol<decltype(&cublasLtLoggerOpenFile)>(
        "cublasLtLoggerOpenFile");
    status = fn == nullptr ? function_not_found()
                           : fn(has_name ? name.c_str() : nullptr);
    int flushed = flush_log_file_locked(conn);
    if (status == CUBLAS_STATUS_SUCCESS && log_staging != nullptr) {
      std::fclose(log_staging);
      log_staging = nullptr;
      log_file_conn = nullptr;
      log_file_target = {};
    }
    if (flushed < 0) {
      return -1;
    }
  }
  return write_log_status(conn, request_id, status);
}

void lupine_cublaslt_cleanup_logs(conn_t *conn) {
  std::lock_guard<std::mutex> update(logger.update_mutex);
  logger.clear(conn);
  std::lock_guard<std::mutex> lock(log_file_mutex);
  if (log_file_conn == conn) {
    auto fn = cublaslt_symbol<decltype(&cublasLtLoggerSetFile)>(
        "cublasLtLoggerSetFile");
    if (fn != nullptr && fn(nullptr) == CUBLAS_STATUS_SUCCESS) {
      if (log_staging != nullptr) {
        std::fclose(log_staging);
      }
      log_staging = nullptr;
    }
    log_file_conn = nullptr;
    log_file_target = {};
  }
}

#include "codegen/gen_cublaslt_server.inc"
