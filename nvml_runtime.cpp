#include "nvml_runtime.h"

#ifdef _WIN32
#include <windows.h>
#else
#include <dlfcn.h>
#endif

uintptr_t lupine_nvml_symbol_address(const char *name) {
#ifdef _WIN32
  static HMODULE library = LoadLibraryA("nvml.dll");
  return library == nullptr
             ? 0
             : reinterpret_cast<uintptr_t>(GetProcAddress(library, name));
#else
  static void *library = dlopen("libnvidia-ml.so.1", RTLD_LAZY | RTLD_LOCAL);
  return library == nullptr ? 0
                            : reinterpret_cast<uintptr_t>(dlsym(library, name));
#endif
}
