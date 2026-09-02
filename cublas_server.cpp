#include "cublas_server.h"

#include <cublas_v2.h>
#include <cuda.h>

#include <cstdlib>

#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#else
#include <dlfcn.h>
#endif

namespace {

#ifdef _WIN32
using library_handle = HMODULE;

library_handle open_library(const char *name) { return LoadLibraryA(name); }

void *load_symbol(library_handle library, const char *name) {
  return reinterpret_cast<void *>(GetProcAddress(library, name));
}
#else
using library_handle = void *;

library_handle open_library(const char *name) {
  return dlopen(name, RTLD_NOW | RTLD_LOCAL);
}

void *load_symbol(library_handle library, const char *name) {
  return dlsym(library, name);
}
#endif

library_handle cublas_library() {
  static library_handle library = [] {
    const char *override_name = std::getenv("LUPINE_CUBLAS_LIBRARY");
    if (override_name != nullptr && override_name[0] != '\0') {
      if (auto handle = open_library(override_name)) {
        return handle;
      }
    }
#ifdef _WIN32
    const char *names[] = {"cublas64_13.dll", "cublas64_12.dll",
                           "cublas64_11.dll"};
#else
    const char *names[] = {"libcublas.so.13", "libcublas.so.12",
                           "libcublas.so.11", "libcublas.so"};
#endif
    for (const char *name : names) {
      if (auto handle = open_library(name)) {
        return handle;
      }
    }
    return library_handle{};
  }();
  return library;
}

} // namespace

template <typename Function> Function cublas_symbol(const char *name) {
  const auto library = cublas_library();
  return library == library_handle{}
             ? nullptr
             : reinterpret_cast<Function>(load_symbol(library, name));
}

#include "codegen/gen_cublas_server.inc"
