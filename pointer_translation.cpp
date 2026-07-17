#include "pointer_translation.h"

#include <dlfcn.h>
#include <cstdlib>
#include <cstring>
#include <mutex>

namespace {

struct provider_state {
  void *library = nullptr;
  lupine_pointer_translate_fn translate = nullptr;
  void *context = nullptr;
  bool initialized = false;
};

provider_state &state() {
  static provider_state *value = new provider_state();
  return *value;
}

std::once_flag &init_once() {
  static auto *flag = new std::once_flag();
  return *flag;
}

void initialize() {
  provider_state &value = state();
  value.initialized = true;
  const char *path = std::getenv("LUPINE_POINTER_TRANSLATOR");
  if (path == nullptr || *path == '\0') {
    return;
  }
  value.library = dlopen(path, RTLD_NOW | RTLD_LOCAL);
  if (value.library == nullptr) {
    value.library = nullptr;
    return;
  }
  auto provider = reinterpret_cast<lupine_pointer_provider_fn>(
      dlsym(value.library, "lupine_pointer_translation_provider_v1"));
  if (provider == nullptr || provider(&value.translate, &value.context) != 0 ||
      value.translate == nullptr) {
    value.translate = nullptr;
    value.context = nullptr;
  }
}

} // namespace

bool lupine_translate_device_pointer(CUdeviceptr old_pointer,
                                     CUdeviceptr *new_pointer) {
#ifdef LUPINE_RPC_CLIENT
  if (new_pointer == nullptr) {
    return false;
  }
  std::call_once(init_once(), initialize);
  provider_state &value = state();
  if (!value.initialized || value.translate == nullptr) {
    return false;
  }
  CUdeviceptr translated = 0;
  if (value.translate(value.context, old_pointer, &translated) <= 0 ||
      translated == old_pointer) {
    return false;
  }
  *new_pointer = translated;
  return true;
#else
  (void)old_pointer;
  (void)new_pointer;
  return false;
#endif
}

bool lupine_translate_rpc_pointer(const void *data, size_t size,
                                  void **owned_copy) {
  if (data == nullptr || owned_copy == nullptr || size != sizeof(CUdeviceptr)) {
    return false;
  }
  CUdeviceptr old_pointer = 0;
  CUdeviceptr new_pointer = 0;
  std::memcpy(&old_pointer, data, sizeof(old_pointer));
  if (!lupine_translate_device_pointer(old_pointer, &new_pointer)) {
    return false;
  }
  void *copy = std::malloc(sizeof(new_pointer));
  if (copy == nullptr) {
    return false;
  }
  std::memcpy(copy, &new_pointer, sizeof(new_pointer));
  *owned_copy = copy;
  return true;
}
