#ifndef LUPINE_NPP_SERVER_H
#define LUPINE_NPP_SERVER_H

// What every NPP library's server handlers share. The build compiles one
// translation unit per library and names it in LUPINE_NPP_LIBRARY.

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
#include "npp_compat.h"
#include "rpc.h"

namespace {

NppStatus function_not_found() { return NPP_NOT_IMPLEMENTED_ERROR; }

// The library this server was compiled against; the handlers marshal its
// structures and parameter widths, so another major on the machine is not a
// match.
void *npp_library() {
  static void *library = []() -> void * {
    const std::string major = std::to_string(NPP_VER_MAJOR);
#ifdef _WIN32
    const std::string name = LUPINE_NPP_LIBRARY "64_" + major + ".dll";
    return reinterpret_cast<void *>(LoadLibraryA(name.c_str()));
#else
    const std::string name = "lib" LUPINE_NPP_LIBRARY ".so." + major;
    return dlopen(name.c_str(), RTLD_LAZY | RTLD_LOCAL);
#endif
  }();
  return library;
}

template <typename Fn> Fn npp_symbol(const char *name) {
  void *lib = npp_library();
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

#endif
