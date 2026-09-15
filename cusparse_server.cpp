#include <cuda_runtime_api.h>

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#ifdef _WIN32
#include <windows.h>
#else
#include <dlfcn.h>
#endif

#include "codegen/gen_rpc_ids.h"
#include "cusparse_compat.h"
#include "rpc.h"

namespace {

cusparseStatus_t function_not_found() { return CUSPARSE_STATUS_NOT_SUPPORTED; }

// The library this server was compiled against; the handlers marshal its
// enums and pointer-mode scalars, so another major on the machine is not a
// match.
void *cusparse_library() {
  static void *library = []() -> void * {
    const std::string major = std::to_string(CUSPARSE_VER_MAJOR);
#ifdef _WIN32
    const std::string name = "cusparse64_" + major + ".dll";
    return reinterpret_cast<void *>(LoadLibraryA(name.c_str()));
#else
    const std::string name = "libcusparse.so." + major;
    return dlopen(name.c_str(), RTLD_LAZY | RTLD_LOCAL);
#endif
  }();
  return library;
}

template <typename Fn> Fn cusparse_symbol(const char *name) {
  void *lib = cusparse_library();
  if (lib == nullptr) {
    return nullptr;
  }
#ifdef _WIN32
  return reinterpret_cast<Fn>(GetProcAddress(static_cast<HMODULE>(lib), name));
#else
  return reinterpret_cast<Fn>(dlsym(lib, name));
#endif
}

// cusparseGetErrorName and cusparseGetErrorString return static strings, so
// the response carries the bytes length-prefixed instead of a status.
int handle_status_text(conn_t *conn, const char *symbol) {
  cusparseStatus_t status;
  if (rpc_read(conn, &status, sizeof(status)) < 0) {
    return -1;
  }
  int request_id = rpc_read_end(conn);
  if (request_id < 0) {
    return -1;
  }
  using fn_t = const char *(*)(cusparseStatus_t);
  fn_t fn = cusparse_symbol<fn_t>(symbol);
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

int handle_cusparseGetErrorName(conn_t *conn) {
  return handle_status_text(conn, "cusparseGetErrorName");
}

int handle_cusparseGetErrorString(conn_t *conn) {
  return handle_status_text(conn, "cusparseGetErrorString");
}

#include "codegen/gen_cusparse_server.inc"
