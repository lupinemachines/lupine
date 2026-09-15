#include <curand.h>

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

// The library documents this for a header that does not match the library,
// which includes a server with no libcurand at all.
curandStatus_t function_not_found() { return CURAND_STATUS_VERSION_MISMATCH; }

// The library this server was compiled against; the handlers marshal its
// enums, so another major on the machine is not a match.
void *curand_library() {
  static void *library = []() -> void * {
    const std::string major = std::to_string(CURAND_VER_MAJOR);
#ifdef _WIN32
    const std::string name = "curand64_" + major + ".dll";
    return reinterpret_cast<void *>(LoadLibraryA(name.c_str()));
#else
    const std::string name = "libcurand.so." + major;
    return dlopen(name.c_str(), RTLD_LAZY | RTLD_LOCAL);
#endif
  }();
  return library;
}

template <typename Fn> Fn curand_symbol(const char *name) {
  void *lib = curand_library();
  if (lib == nullptr) {
    return nullptr;
  }
#ifdef _WIN32
  return reinterpret_cast<Fn>(GetProcAddress(static_cast<HMODULE>(lib), name));
#else
  return reinterpret_cast<Fn>(dlsym(lib, name));
#endif
}

// ---------------------------------------------------------------------------
// Static tables
// ---------------------------------------------------------------------------

// The library only hands out a pointer; its Sobol tables cover the 20,000
// dimensions it documents. A table travels as its byte count, zero after a
// failure, then its bytes.
constexpr uint64_t sobol_dimensions = 20000;

template <typename T>
int handle_table(conn_t *conn, const char *symbol, bool takes_set) {
  curandDirectionVectorSet_t set;
  if (rpc_read(conn, &set, sizeof(set)) < 0) {
    return -1;
  }
  int request_id = rpc_read_end(conn);
  if (request_id < 0) {
    return -1;
  }
  T *table = nullptr;
  curandStatus_t status = function_not_found();
  if (takes_set) {
    using fn_t = curandStatus_t (*)(T **, curandDirectionVectorSet_t);
    if (fn_t fn = curand_symbol<fn_t>(symbol)) {
      status = fn(&table, set);
    }
  } else {
    using fn_t = curandStatus_t (*)(T **);
    if (fn_t fn = curand_symbol<fn_t>(symbol)) {
      status = fn(&table);
    }
  }
  const uint64_t bytes =
      status == CURAND_STATUS_SUCCESS ? sobol_dimensions * sizeof(T) : 0;
  if (rpc_write_start_response(conn, request_id) < 0 ||
      rpc_write(conn, &status, sizeof(status)) < 0 ||
      rpc_write(conn, &bytes, sizeof(bytes)) < 0 ||
      (bytes != 0 && rpc_write(conn, table, bytes) < 0) ||
      rpc_write_end(conn) < 0) {
    return -1;
  }
  return 0;
}

} // namespace

int handle_curandGetDirectionVectors32(conn_t *conn) {
  return handle_table<curandDirectionVectors32_t>(
      conn, "curandGetDirectionVectors32", true);
}

int handle_curandGetDirectionVectors64(conn_t *conn) {
  return handle_table<curandDirectionVectors64_t>(
      conn, "curandGetDirectionVectors64", true);
}

int handle_curandGetScrambleConstants32(conn_t *conn) {
  return handle_table<unsigned int>(conn, "curandGetScrambleConstants32",
                                    false);
}

int handle_curandGetScrambleConstants64(conn_t *conn) {
  return handle_table<unsigned long long>(conn, "curandGetScrambleConstants64",
                                          false);
}

#include "codegen/gen_curand_server.inc"
