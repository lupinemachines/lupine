#include <cuda_runtime_api.h>

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <string>

#ifdef _WIN32
#include <windows.h>
#else
#include <dlfcn.h>
#endif

#include "codegen/gen_rpc_ids.h"
#include "cusolver_compat.h"
#include "rpc.h"

namespace {

cusolverStatus_t function_not_found() { return CUSOLVER_STATUS_NOT_SUPPORTED; }

// cuSOLVERMg shares cuSOLVER's version, so the matching major is the one this
// server was compiled against.
void *cusolvermg_library() {
  static void *library = []() -> void * {
    const std::string major = std::to_string(CUSOLVER_VER_MAJOR);
#ifdef _WIN32
    const std::string name = "cusolverMg64_" + major + ".dll";
    return reinterpret_cast<void *>(LoadLibraryA(name.c_str()));
#else
    const std::string name = "libcusolverMg.so." + major;
    return dlopen(name.c_str(), RTLD_LAZY | RTLD_LOCAL);
#endif
  }();
  return library;
}

template <typename Fn> Fn cusolvermg_symbol(const char *name) {
  void *lib = cusolvermg_library();
  if (lib == nullptr) {
    return nullptr;
  }
#ifdef _WIN32
  return reinterpret_cast<Fn>(GetProcAddress(static_cast<HMODULE>(lib), name));
#else
  return reinterpret_cast<Fn>(dlsym(lib, name));
#endif
}

} // namespace

#include "codegen/gen_cusolvermg_server.inc"
