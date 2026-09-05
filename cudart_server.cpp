// cuda.h supplies CUDA_VERSION, which the generated guards and the registry
// both test; the runtime header alone only defines CUDART_VERSION.
#include <cuda.h>
#include <cuda_runtime_api.h>

#include <cstdint>
#include <cstring>
#include <string>

#ifdef _WIN32
#include <windows.h>
#else
#include <dlfcn.h>
#endif

#include "codegen/gen_rpc_ids.h"
#include "rpc.h"

namespace {

cudaError_t function_not_found() { return cudaErrorNotSupported; }

// The runtime this server was compiled against; the handlers marshal its
// struct layouts, so an older or newer runtime on the machine is not a match.
void *cudart_library() {
  static void *library = []() -> void * {
    const std::string major = std::to_string(CUDART_VERSION / 1000);
#ifdef _WIN32
    const std::string name = "cudart64_" + major + ".dll";
    return reinterpret_cast<void *>(LoadLibraryA(name.c_str()));
#else
    const std::string name = "libcudart.so." + major;
    return dlopen(name.c_str(), RTLD_LAZY | RTLD_LOCAL);
#endif
  }();
  return library;
}

template <typename Fn> Fn cudart_symbol(const char *name) {
  void *lib = cudart_library();
  if (lib == nullptr) {
    return nullptr;
  }
#ifdef _WIN32
  return reinterpret_cast<Fn>(GetProcAddress(static_cast<HMODULE>(lib), name));
#else
  return reinterpret_cast<Fn>(dlsym(lib, name));
#endif
}

// cudaGetErrorName and cudaGetErrorString return static strings, so the
// response carries the bytes length-prefixed instead of a result code.
int handle_error_text(conn_t *conn, const char *symbol) {
  cudaError_t error;
  if (rpc_read(conn, &error, sizeof(error)) < 0) {
    return -1;
  }
  int request_id = rpc_read_end(conn);
  if (request_id < 0) {
    return -1;
  }
  using fn_t = const char *(*)(cudaError_t);
  fn_t fn = cudart_symbol<fn_t>(symbol);
  const char *text = fn == nullptr ? nullptr : fn(error);
  uint32_t length = text == nullptr ? 0 : static_cast<uint32_t>(strlen(text));
  if (rpc_write_start_response(conn, request_id) < 0 ||
      rpc_write(conn, &length, sizeof(length)) < 0 ||
      rpc_write(conn, text, length) < 0 || rpc_write_end(conn) < 0) {
    return -1;
  }
  return 0;
}

} // namespace

int handle_cudaGetErrorName(conn_t *conn) {
  return handle_error_text(conn, "cudaGetErrorName");
}

int handle_cudaGetErrorString(conn_t *conn) {
  return handle_error_text(conn, "cudaGetErrorString");
}

#include "codegen/gen_cudart_server.inc"
