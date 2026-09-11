#include <cuda_runtime_api.h>
#include <cufftXt.h>

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
#include "rpc.h"

namespace {

cufftResult function_not_found() { return CUFFT_NOT_SUPPORTED; }

// The library this server was compiled against; the handlers marshal its
// enums and descriptors, so another major on the machine is not a match.
void *cufft_library() {
  static void *library = []() -> void * {
    const std::string major = std::to_string(CUFFT_VER_MAJOR);
#ifdef _WIN32
    const std::string name = "cufft64_" + major + ".dll";
    return reinterpret_cast<void *>(LoadLibraryA(name.c_str()));
#else
    const std::string name = "libcufft.so." + major;
    return dlopen(name.c_str(), RTLD_LAZY | RTLD_LOCAL);
#endif
  }();
  return library;
}

template <typename Fn> Fn cufft_symbol(const char *name) {
  void *lib = cufft_library();
  if (lib == nullptr) {
    return nullptr;
  }
#ifdef _WIN32
  return reinterpret_cast<Fn>(GetProcAddress(static_cast<HMODULE>(lib), name));
#else
  return reinterpret_cast<Fn>(dlsym(lib, name));
#endif
}

int write_status(conn_t *conn, int request_id, cufftResult status) {
  if (rpc_write_start_response(conn, request_id) < 0 ||
      rpc_write(conn, &status, sizeof(status)) < 0 || rpc_write_end(conn) < 0) {
    return -1;
  }
  return 0;
}

// A per-GPU pointer array: its count, then that many pointers. Zero means
// the caller passed no array.
int read_pointer_array(conn_t *conn, std::vector<void *> &pointers) {
  uint32_t n = 0;
  if (rpc_read(conn, &n, sizeof(n)) < 0) {
    return -1;
  }
  pointers.resize(n);
  if (n != 0 && rpc_read(conn, pointers.data(), n * sizeof(void *)) < 0) {
    return -1;
  }
  return 0;
}

void **array_or_null(std::vector<void *> &pointers) {
  return pointers.empty() ? nullptr : pointers.data();
}

} // namespace

// ---------------------------------------------------------------------------
// Multi-GPU descriptors
// ---------------------------------------------------------------------------

// The client keeps a copy of the descriptor, so a successful allocation
// answers with the descriptor's address and both structs behind it.
int handle_cufftXtMalloc(conn_t *conn) {
  cufftHandle plan;
  cufftXtSubFormat format;
  if (rpc_read(conn, &plan, sizeof(plan)) < 0 ||
      rpc_read(conn, &format, sizeof(format)) < 0) {
    return -1;
  }
  int request_id = rpc_read_end(conn);
  if (request_id < 0) {
    return -1;
  }
  using fn_t = cufftResult (*)(cufftHandle, cudaLibXtDesc **, cufftXtSubFormat);
  fn_t fn = cufft_symbol<fn_t>("cufftXtMalloc");
  cudaLibXtDesc *descriptor = nullptr;
  cufftResult status =
      fn == nullptr ? function_not_found() : fn(plan, &descriptor, format);
  if (status == CUFFT_SUCCESS && descriptor->descriptor == nullptr) {
    status = CUFFT_INTERNAL_ERROR;
  }
  if (rpc_write_start_response(conn, request_id) < 0 ||
      rpc_write(conn, &status, sizeof(status)) < 0 ||
      (status == CUFFT_SUCCESS &&
       (rpc_write(conn, &descriptor, sizeof(descriptor)) < 0 ||
        rpc_write(conn, descriptor, sizeof(*descriptor)) < 0 ||
        rpc_write(conn, descriptor->descriptor,
                  sizeof(*descriptor->descriptor)) < 0)) ||
      rpc_write_end(conn) < 0) {
    return -1;
  }
  return 0;
}

int handle_cufftXtFree(conn_t *conn) {
  cudaLibXtDesc *descriptor;
  if (rpc_read(conn, &descriptor, sizeof(descriptor)) < 0) {
    return -1;
  }
  int request_id = rpc_read_end(conn);
  if (request_id < 0) {
    return -1;
  }
  using fn_t = cufftResult (*)(cudaLibXtDesc *);
  fn_t fn = cufft_symbol<fn_t>("cufftXtFree");
  return write_status(conn, request_id,
                      fn == nullptr ? function_not_found() : fn(descriptor));
}

// The host side of a copy is staged here for the length the client's mirror
// of the descriptor computed; cufftXtMemcpy is synchronous, so the staging
// buffer never has to outlive the handler.
int handle_cufftXtMemcpy(conn_t *conn) {
  cufftHandle plan;
  cufftXtCopyType type;
  cudaLibXtDesc *dst;
  cudaLibXtDesc *src;
  size_t bytes;
  if (rpc_read(conn, &plan, sizeof(plan)) < 0 ||
      rpc_read(conn, &type, sizeof(type)) < 0 ||
      rpc_read(conn, &dst, sizeof(dst)) < 0 ||
      rpc_read(conn, &src, sizeof(src)) < 0 ||
      rpc_read(conn, &bytes, sizeof(bytes)) < 0) {
    return -1;
  }
  std::vector<unsigned char> host(type == CUFFT_COPY_DEVICE_TO_DEVICE ? 0
                                                                      : bytes);
  if (type == CUFFT_COPY_HOST_TO_DEVICE && bytes != 0 &&
      rpc_read(conn, host.data(), bytes) < 0) {
    return -1;
  }
  int request_id = rpc_read_end(conn);
  if (request_id < 0) {
    return -1;
  }
  using fn_t = cufftResult (*)(cufftHandle, void *, void *, cufftXtCopyType);
  fn_t fn = cufft_symbol<fn_t>("cufftXtMemcpy");
  cufftResult status = function_not_found();
  if (fn != nullptr) {
    switch (type) {
    case CUFFT_COPY_HOST_TO_DEVICE:
      status = fn(plan, dst, host.data(), type);
      break;
    case CUFFT_COPY_DEVICE_TO_HOST:
      status = fn(plan, host.data(), src, type);
      break;
    default:
      status = fn(plan, dst, src, type);
      break;
    }
  }
  if (rpc_write_start_response(conn, request_id) < 0 ||
      rpc_write(conn, &status, sizeof(status)) < 0 ||
      (type == CUFFT_COPY_DEVICE_TO_HOST && status == CUFFT_SUCCESS &&
       bytes != 0 && rpc_write(conn, host.data(), bytes) < 0) ||
      rpc_write_end(conn) < 0) {
    return -1;
  }
  return 0;
}

// ---------------------------------------------------------------------------
// Arrays with one entry per GPU of the plan
// ---------------------------------------------------------------------------

int handle_cufftXtSetWorkArea(conn_t *conn) {
  cufftHandle plan;
  std::vector<void *> work_area;
  if (rpc_read(conn, &plan, sizeof(plan)) < 0 ||
      read_pointer_array(conn, work_area) < 0) {
    return -1;
  }
  int request_id = rpc_read_end(conn);
  if (request_id < 0) {
    return -1;
  }
  using fn_t = cufftResult (*)(cufftHandle, void **);
  fn_t fn = cufft_symbol<fn_t>("cufftXtSetWorkArea");
  return write_status(conn, request_id,
                      fn == nullptr ? function_not_found()
                                    : fn(plan, array_or_null(work_area)));
}

int handle_cufftXtSetCallback(conn_t *conn) {
  cufftHandle plan;
  cufftXtCallbackType type;
  std::vector<void *> routines;
  std::vector<void *> caller_info;
  if (rpc_read(conn, &plan, sizeof(plan)) < 0 ||
      rpc_read(conn, &type, sizeof(type)) < 0 ||
      read_pointer_array(conn, routines) < 0 ||
      read_pointer_array(conn, caller_info) < 0) {
    return -1;
  }
  int request_id = rpc_read_end(conn);
  if (request_id < 0) {
    return -1;
  }
  using fn_t =
      cufftResult (*)(cufftHandle, void **, cufftXtCallbackType, void **);
  fn_t fn = cufft_symbol<fn_t>("cufftXtSetCallback");
  return write_status(conn, request_id,
                      fn == nullptr ? function_not_found()
                                    : fn(plan, array_or_null(routines), type,
                                         array_or_null(caller_info)));
}

namespace {

int handle_set_jit_callback(conn_t *conn, const char *symbol) {
  cufftHandle plan;
  uint32_t name_length;
  uint64_t fatbin_size;
  cufftXtCallbackType type;
  std::string name;
  std::vector<unsigned char> fatbin;
  std::vector<void *> caller_info;
  if (rpc_read(conn, &plan, sizeof(plan)) < 0 ||
      rpc_read(conn, &name_length, sizeof(name_length)) < 0) {
    return -1;
  }
  name.resize(name_length);
  if ((name_length != 0 && rpc_read(conn, &name[0], name_length) < 0) ||
      rpc_read(conn, &fatbin_size, sizeof(fatbin_size)) < 0) {
    return -1;
  }
  fatbin.resize(fatbin_size);
  if ((fatbin_size != 0 && rpc_read(conn, fatbin.data(), fatbin_size) < 0) ||
      rpc_read(conn, &type, sizeof(type)) < 0 ||
      read_pointer_array(conn, caller_info) < 0) {
    return -1;
  }
  int request_id = rpc_read_end(conn);
  if (request_id < 0) {
    return -1;
  }
  using fn_t = cufftResult (*)(cufftHandle, const char *, const void *, size_t,
                               cufftXtCallbackType, void **);
  fn_t fn = cufft_symbol<fn_t>(symbol);
  return write_status(conn, request_id,
                      fn == nullptr
                          ? function_not_found()
                          : fn(plan, name_length != 0 ? name.c_str() : nullptr,
                               fatbin_size != 0 ? fatbin.data() : nullptr,
                               fatbin_size, type, array_or_null(caller_info)));
}

} // namespace

#if CUFFT_VERSION >= 12000
int handle_cufftXtSetJITCallback(conn_t *conn) {
  return handle_set_jit_callback(conn, "cufftXtSetJITCallback");
}
#endif

#if CUFFT_VERSION >= 11300 && CUFFT_VERSION < 12000
int handle___cufftXtSetJITCallback_12_7(conn_t *conn) {
  return handle_set_jit_callback(conn, "__cufftXtSetJITCallback_12_7");
}
#endif

#include "codegen/gen_cufft_server.inc"
