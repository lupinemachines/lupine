// A RECV_ONLY LENGTH: out-buffer is allocated on the server at the caller's
// advertised length and the whole of it goes back on the wire, so every byte
// the library leaves untouched has to come from the allocation. This drives
// nvmlDeviceGetUUID, whose string is far shorter than the length torch asks
// for, and primes the server's heap first so an untouched tail would arrive
// as the primer's bytes rather than as zeros.

#include <nvml.h>

#include <dlfcn.h>

#include <cstdio>
#include <cstring>

namespace {

// The primer and the out-buffer ask for the same number of bytes so the
// allocator hands the second call the chunk the first one just released.
constexpr unsigned int kLength = 512;
constexpr char kPrimerByte = '\x80';
constexpr int kAttempts = 8;

using init_fn = nvmlReturn_t (*)();
using shutdown_fn = nvmlReturn_t (*)();
using handle_by_index_fn = nvmlReturn_t (*)(unsigned int, nvmlDevice_t *);
using handle_by_uuid_fn = nvmlReturn_t (*)(const char *, nvmlDevice_t *);
using get_uuid_fn = nvmlReturn_t (*)(nvmlDevice_t, char *, unsigned int);

template <typename T> T lookup(void *library, const char *name) {
  auto symbol = reinterpret_cast<T>(dlsym(library, name));
  if (symbol == nullptr) {
    std::fprintf(stderr, "missing NVML symbol %s\n", name);
  }
  return symbol;
}

} // namespace

int main() {
  void *library = dlopen("libnvidia-ml.so.1", RTLD_NOW);
  if (library == nullptr) {
    std::fprintf(stderr, "dlopen(libnvidia-ml.so.1) failed: %s\n", dlerror());
    return 1;
  }
  auto nvml_init = lookup<init_fn>(library, "nvmlInit_v2");
  auto nvml_shutdown = lookup<shutdown_fn>(library, "nvmlShutdown");
  auto handle_by_index =
      lookup<handle_by_index_fn>(library, "nvmlDeviceGetHandleByIndex_v2");
  auto handle_by_uuid =
      lookup<handle_by_uuid_fn>(library, "nvmlDeviceGetHandleByUUID");
  auto get_uuid = lookup<get_uuid_fn>(library, "nvmlDeviceGetUUID");
  if (nvml_init == nullptr || nvml_shutdown == nullptr ||
      handle_by_index == nullptr || handle_by_uuid == nullptr ||
      get_uuid == nullptr) {
    return 1;
  }

  nvmlReturn_t status = nvml_init();
  if (status != NVML_SUCCESS) {
    std::fprintf(stderr, "nvmlInit_v2 failed: %d\n", static_cast<int>(status));
    return 1;
  }

  nvmlDevice_t device = nullptr;
  status = handle_by_index(0, &device);
  if (status != NVML_SUCCESS) {
    std::fprintf(stderr, "nvmlDeviceGetHandleByIndex_v2 failed: %d\n",
                 static_cast<int>(status));
    return 1;
  }

  int failures = 0;
  for (int attempt = 0; attempt < kAttempts; ++attempt) {
    char primer[kLength];
    std::memset(primer, kPrimerByte, sizeof(primer) - 1);
    primer[sizeof(primer) - 1] = '\0';
    nvmlDevice_t ignored = nullptr;
    // The lookup is expected to fail; what matters is that the server read
    // kLength bytes of the primer into a buffer of exactly the size the next
    // call asks for, then freed it.
    handle_by_uuid(primer, &ignored);

    char uuid[kLength];
    std::memset(uuid, kPrimerByte, sizeof(uuid));
    status = get_uuid(device, uuid, kLength);
    if (status != NVML_SUCCESS) {
      std::fprintf(stderr, "nvmlDeviceGetUUID failed: %d\n",
                   static_cast<int>(status));
      return 1;
    }
    size_t used = strnlen(uuid, sizeof(uuid));
    if (used == sizeof(uuid)) {
      std::fprintf(stderr, "attempt %d: UUID is not terminated\n", attempt);
      ++failures;
      continue;
    }
    for (size_t i = used + 1; i < sizeof(uuid); ++i) {
      if (uuid[i] != '\0') {
        std::fprintf(
            stderr,
            "attempt %d: byte %zu past the UUID's terminator is 0x%02x, not"
            " zero\n",
            attempt, i, static_cast<unsigned char>(uuid[i]));
        ++failures;
        break;
      }
    }
  }

  nvml_shutdown();
  if (failures != 0) {
    std::fprintf(stderr, "%d of %d attempts returned uninitialized bytes\n",
                 failures, kAttempts);
    return 1;
  }
  std::printf("nvmlDeviceGetUUID out-buffer tail is zeroed\n");
  return 0;
}
