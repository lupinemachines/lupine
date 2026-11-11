#include <hip/hip_runtime_api.h>

#include <cstring>
#include <mutex>
#include <unordered_map>
#include <vector>

#ifdef _WIN32
#include <windows.h>
#else
#include <dlfcn.h>
#endif

#include "codegen/gen_rpc_ids.h"
#include "device_stdout.h"
#include "hip_fatbin.h"
#include "rpc.h"

namespace {

hipError_t function_not_found() { return hipErrorUnknown; }

std::mutex module_images_mutex;
std::unordered_map<hipModule_t, std::vector<unsigned char>> module_images;

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
    static const char *const names[] = {"libamdhip64.so.7", "libamdhip64.so.6",
                                        "libamdhip64.so"};
    for (const char *name : names) {
      void *candidate = dlopen(name, RTLD_LAZY | RTLD_LOCAL);
      if (candidate != nullptr) {
        if (dlsym(candidate, "lupine_hip_client_marker") != nullptr) {
          dlclose(candidate);
          continue;
        }
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

int handle_hipModuleLoadData(conn_t *conn) {
  uint64_t size = 0;
  if (rpc_read(conn, &size, sizeof(size)) < 0 || size == 0 ||
      size > lupine_hip_max_image_size)
    return -1;
  std::vector<unsigned char> image(static_cast<size_t>(size));
  int request_id;
  if (rpc_read(conn, image.data(), image.size()) < 0 ||
      (request_id = rpc_read_end(conn)) < 0)
    return -1;
  static const auto load =
      hip_symbol<decltype(&hipModuleLoadData)>("hipModuleLoadData");
  hipModule_t module = nullptr;
  hipError_t result = hipErrorInvalidImage;
  if (lupine_hip_bundle_size(image.data(), image.size()) == image.size()) {
    result =
        load == nullptr ? function_not_found() : load(&module, image.data());
  }
  if (result == hipSuccess) {
    // Native HIP may retain references to its input for later symbol lookup.
    std::lock_guard<std::mutex> lock(module_images_mutex);
    module_images.insert_or_assign(module, std::move(image));
  }
  return rpc_write_start_response(conn, request_id) < 0 ||
                 rpc_write(conn, &module, sizeof(module)) < 0 ||
                 rpc_write(conn, &result, sizeof(result)) < 0 ||
                 rpc_write_end(conn) < 0
             ? -1
             : 0;
}

int handle_hipModuleUnload(conn_t *conn) {
  hipModule_t module = nullptr;
  int request_id;
  if (rpc_read(conn, &module, sizeof(module)) < 0 ||
      (request_id = rpc_read_end(conn)) < 0)
    return -1;
  static const auto unload =
      hip_symbol<decltype(&hipModuleUnload)>("hipModuleUnload");
  hipError_t result = unload == nullptr ? function_not_found() : unload(module);
  if (result == hipSuccess) {
    std::lock_guard<std::mutex> lock(module_images_mutex);
    module_images.erase(module);
  }
  return rpc_write_start_response(conn, request_id) < 0 ||
                 rpc_write(conn, &result, sizeof(result)) < 0 ||
                 rpc_write_end(conn) < 0
             ? -1
             : 0;
}

int handle_hipModuleLaunchKernel(conn_t *conn) {
  hipFunction_t function = nullptr;
  unsigned int dimensions[6] = {};
  unsigned int shared_memory = 0;
  hipStream_t stream = nullptr;
  int request_id;
  if (rpc_read(conn, &function, sizeof(function)) < 0 ||
      rpc_read(conn, dimensions, sizeof(dimensions)) < 0 ||
      rpc_read(conn, &shared_memory, sizeof(shared_memory)) < 0 ||
      rpc_read(conn, &stream, sizeof(stream)) < 0 ||
      (request_id = rpc_read_end(conn)) < 0)
    return -1;
  static const auto launch =
      hip_symbol<decltype(&hipModuleLaunchKernel)>("hipModuleLaunchKernel");
  static const auto synchronize =
      hip_symbol<decltype(&hipStreamSynchronize)>("hipStreamSynchronize");
  hipError_t result = function_not_found();
  lupine_captured_stdout capture;
  if (launch != nullptr && synchronize != nullptr &&
      lupine_start_stdout_capture(&capture, true)) {
    // HIP can print before the launch returns, unlike CUDA's deferred printf.
    // Keep capture active through stream completion. This first, argument-free
    // launch path is deliberately synchronous so output cannot escape between
    // the launch RPC and a later synchronization RPC.
    result = launch(function, dimensions[0], dimensions[1], dimensions[2],
                    dimensions[3], dimensions[4], dimensions[5], shared_memory,
                    stream, nullptr, nullptr);
    // Null kernelParams/extra make native HIP validate that the kernel takes
    // no arguments. Never manufacture a buffer from untyped client addresses.
    if (result == hipSuccess) {
      result = synchronize(stream);
    }
    lupine_finish_stdout_capture(&capture);
  }
  return rpc_write_start_response(conn, request_id) < 0 ||
                 rpc_write(conn, &result, sizeof(result)) < 0 ||
                 lupine_write_captured_stdout(conn, capture) < 0 ||
                 rpc_write_end(conn) < 0
             ? -1
             : 0;
}

namespace {

int handle_error_text(conn_t *conn, const char *(*describe)(hipError_t),
                      const char *fallback) {
  hipError_t error;
  int request_id;
  if (rpc_read(conn, &error, sizeof(error)) < 0 ||
      (request_id = rpc_read_end(conn)) < 0)
    return -1;
  const char *text = describe == nullptr ? nullptr : describe(error);
  if (text == nullptr)
    text = fallback;
  uint64_t size = strlen(text) + 1;
  return rpc_write_start_response(conn, request_id) < 0 ||
                 rpc_write(conn, &size, sizeof(size)) < 0 ||
                 rpc_write(conn, text, static_cast<size_t>(size)) < 0 ||
                 rpc_write_end(conn) < 0
             ? -1
             : 0;
}

} // namespace

int handle_hipGetErrorString(conn_t *conn) {
  static const auto describe =
      hip_symbol<const char *(*)(hipError_t)>("hipGetErrorString");
  return handle_error_text(conn, describe, "unknown error");
}

int handle_hipGetErrorName(conn_t *conn) {
  static const auto describe =
      hip_symbol<const char *(*)(hipError_t)>("hipGetErrorName");
  return handle_error_text(conn, describe, "hipErrorUnknown");
}
