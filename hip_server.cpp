#include "hip_server.h"

#include "hip_compat.h"

#ifdef _WIN32
#include <windows.h>
#else
#include <dlfcn.h>
#endif

#include "codegen/gen_api.h"

namespace {

hipError_t function_not_found() { return hipErrorUnknown; }

void *hip_library() {
#ifdef _WIN32
  // amdhip64_6.dll on ROCm 6.x. The 6.x runtime ships this name on Windows.
  static HMODULE lib = LoadLibraryA("amdhip64_6.dll");
  if (lib == nullptr) {
    // Fall back to the unversioned name used by some SDK layouts.
    static HMODULE fallback = LoadLibraryA("amdhip64.dll");
    return fallback;
  }
  return lib;
#else
  static void *lib = dlopen("libamdhip64.so", RTLD_LAZY | RTLD_LOCAL);
  return lib;
#endif
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