// torch calls nvmlInit on every device query, from whichever thread asks, while
// other threads keep using handles they already hold. Real NVML is thread-safe
// and a repeated init leaves existing handles valid.

#include <nvml.h>

#include <dlfcn.h>

#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

namespace {

constexpr int kThreads = 8;
constexpr auto kDuration = std::chrono::seconds(5);

using init_fn = nvmlReturn_t (*)();
using shutdown_fn = nvmlReturn_t (*)();
using handle_by_index_fn = nvmlReturn_t (*)(unsigned int, nvmlDevice_t *);
using get_name_fn = nvmlReturn_t (*)(nvmlDevice_t, char *, unsigned int);

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
  auto get_name = lookup<get_name_fn>(library, "nvmlDeviceGetName");
  if (nvml_init == nullptr || nvml_shutdown == nullptr ||
      handle_by_index == nullptr || get_name == nullptr) {
    return 1;
  }

  nvmlDevice_t device = nullptr;
  char expected[NVML_DEVICE_NAME_BUFFER_SIZE] = {};
  if (nvml_init() != NVML_SUCCESS || handle_by_index(0, &device) != NVML_SUCCESS ||
      get_name(device, expected, sizeof(expected)) != NVML_SUCCESS) {
    std::fprintf(stderr, "single-threaded NVML setup failed\n");
    return 1;
  }

  std::atomic<long> iterations{0};
  std::atomic<long> failures{0};
  auto deadline = std::chrono::steady_clock::now() + kDuration;
  std::vector<std::thread> threads;
  for (int t = 0; t < kThreads; ++t) {
    threads.emplace_back([&, t] {
      while (std::chrono::steady_clock::now() < deadline) {
        nvmlReturn_t init = nvml_init();
        nvmlDevice_t handle = nullptr;
        nvmlReturn_t lookup_status = handle_by_index(0, &handle);
        char name[NVML_DEVICE_NAME_BUFFER_SIZE] = {};
        nvmlReturn_t name_status =
            lookup_status == NVML_SUCCESS
                ? get_name(handle, name, sizeof(name))
                : lookup_status;
        // The handle taken before the threads started must stay valid too.
        char held[NVML_DEVICE_NAME_BUFFER_SIZE] = {};
        nvmlReturn_t held_status = get_name(device, held, sizeof(held));
        if (init != NVML_SUCCESS || name_status != NVML_SUCCESS ||
            held_status != NVML_SUCCESS || std::strcmp(name, expected) != 0 ||
            std::strcmp(held, expected) != 0) {
          if (failures.fetch_add(1) < 5) {
            std::fprintf(stderr,
                         "thread %d: init %d lookup %d name %d \"%s\" held %d "
                         "\"%s\"\n",
                         t, init, lookup_status, name_status, name, held_status,
                         held);
          }
        }
        iterations.fetch_add(1);
      }
    });
  }
  for (auto &thread : threads) {
    thread.join();
  }
  for (long i = 0; i <= iterations.load(); ++i) {
    nvml_shutdown();
  }
  std::printf("%ld iterations, %ld failures\n", iterations.load(),
              failures.load());
  return failures.load() == 0 ? 0 : 1;
}
