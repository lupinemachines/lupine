// cusparseLt.h brings cusparse.h for its status type, and the runtime header
// supplies the stream type.
#include <cuda_runtime_api.h>
#include <cusparseLt.h>

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
#include "rpc.h"

namespace {

cusparseStatus_t function_not_found() { return CUSPARSE_STATUS_NOT_SUPPORTED; }

// The machine's library, whose SONAME major is 0 for every cuSPARSELt release.
void *cusparselt_library() {
  static void *library = []() -> void * {
#ifdef _WIN32
    return reinterpret_cast<void *>(LoadLibraryA("cusparseLt.dll"));
#else
    return dlopen("libcusparseLt.so.0", RTLD_LAZY | RTLD_LOCAL);
#endif
  }();
  return library;
}

template <typename Fn> Fn cusparselt_symbol(const char *name) {
  void *lib = cusparselt_library();
  if (lib == nullptr) {
    return nullptr;
  }
#ifdef _WIN32
  return reinterpret_cast<Fn>(GetProcAddress(static_cast<HMODULE>(lib), name));
#else
  return reinterpret_cast<Fn>(dlsym(lib, name));
#endif
}

// An object is this server's allocation, named by its address: the library
// releases what it holds and the allocation goes with it.
template <typename Object>
int handle_destroy(conn_t *conn, const char *symbol) {
  Object *object = nullptr;
  if (rpc_read(conn, &object, sizeof(void *)) < 0) {
    return -1;
  }
  int request_id = rpc_read_end(conn);
  if (request_id < 0) {
    return -1;
  }
  using fn_t = cusparseStatus_t (*)(const Object *);
  fn_t fn = cusparselt_symbol<fn_t>(symbol);
  const cusparseStatus_t status =
      fn == nullptr ? function_not_found() : fn(object);
  delete object;
  if (rpc_write_start_response(conn, request_id) < 0 ||
      rpc_write(conn, &status, sizeof(status)) < 0 || rpc_write_end(conn) < 0) {
    return -1;
  }
  return 0;
}

#if CUSPARSELT_VERSION >= 700
// cusparseLtGetErrorName and cusparseLtGetErrorString return static strings,
// so the response carries the bytes length-prefixed instead of a status.
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
  fn_t fn = cusparselt_symbol<fn_t>(symbol);
  const char *text = fn == nullptr ? nullptr : fn(status);
  uint32_t length = text == nullptr ? 0 : static_cast<uint32_t>(strlen(text));
  if (rpc_write_start_response(conn, request_id) < 0 ||
      rpc_write(conn, &length, sizeof(length)) < 0 ||
      rpc_write(conn, text, length) < 0 || rpc_write_end(conn) < 0) {
    return -1;
  }
  return 0;
}
#endif

} // namespace

int handle_cusparseLtDestroy(conn_t *conn) {
  return handle_destroy<cusparseLtHandle_t>(conn, "cusparseLtDestroy");
}

int handle_cusparseLtMatDescriptorDestroy(conn_t *conn) {
  return handle_destroy<cusparseLtMatDescriptor_t>(
      conn, "cusparseLtMatDescriptorDestroy");
}

#if CUSPARSELT_VERSION >= 800
int handle_cusparseLtMatmulAlgSelectionDestroy(conn_t *conn) {
  return handle_destroy<cusparseLtMatmulAlgSelection_t>(
      conn, "cusparseLtMatmulAlgSelectionDestroy");
}
#endif

int handle_cusparseLtMatmulPlanDestroy(conn_t *conn) {
  return handle_destroy<cusparseLtMatmulPlan_t>(conn,
                                                "cusparseLtMatmulPlanDestroy");
}

#if CUSPARSELT_VERSION >= 700
int handle_cusparseLtGetErrorName(conn_t *conn) {
  return handle_status_text(conn, "cusparseLtGetErrorName");
}

int handle_cusparseLtGetErrorString(conn_t *conn) {
  return handle_status_text(conn, "cusparseLtGetErrorString");
}
#endif

#include "codegen/gen_cusparselt_server.inc"
