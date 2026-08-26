#include <hip/hip_runtime_api.h>

#include <cstring>

#ifdef _WIN32
#include <windows.h>
#else
#include <dlfcn.h>
#endif

#include "codegen/gen_rpc_ids.h"
#include "rpc.h"

namespace {

hipError_t function_not_found() { return hipErrorUnknown; }

void *hip_library() {
  static void *library = []() -> void * {
#ifdef _WIN32
    // ROCm uses a versioned runtime name in normal installations, while some
    // SDK and development layouts expose only the unversioned alias.
    static const char *const names[] = {"amdhip64_6.dll", "amdhip64.dll"};
    for (const char *name : names) {
      HMODULE candidate = LoadLibraryA(name);
      if (candidate != nullptr) {
        return reinterpret_cast<void *>(candidate);
      }
    }
#else
    static const char *const names[] = {"libamdhip64.so.6", "libamdhip64.so"};
    for (const char *name : names) {
      void *candidate = dlopen(name, RTLD_LAZY | RTLD_LOCAL);
      if (candidate != nullptr) {
        return candidate;
      }
    }
#endif
    return nullptr;
  }();
  return library;
}

template <typename Fn> Fn hip_symbol(const char *name) {
  void *lib = hip_library();
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

#include "codegen/gen_hip_server.inc"
