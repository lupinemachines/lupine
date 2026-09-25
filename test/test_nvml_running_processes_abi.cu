// The unversioned nvmlDevice*RunningProcesses entry points are the v1 ABI:
// their entries are the 16-byte nvmlProcessInfo_v1_t, while _v2 and _v3 fill
// the 24-byte nvmlProcessInfo_t. nvml.h maps the unversioned names to _v3, so
// only binaries built against old headers (and dlsym callers) reach v1. This
// holds a context through the shim so the server's own process shows up, then
// checks that a v1 call sized exactly to the process count leaves the bytes
// past the caller's array alone and agrees with _v3.

#include <cuda.h>
#include <nvml.h>

#include <dlfcn.h>

#include <cstdio>
#include <cstring>
#include <vector>

namespace {

constexpr unsigned char kGuard = 0xab;
constexpr size_t kGuardBytes = 256;

using init_fn = nvmlReturn_t (*)();
using shutdown_fn = nvmlReturn_t (*)();
using count_fn = nvmlReturn_t (*)(unsigned int *);
using handle_by_index_fn = nvmlReturn_t (*)(unsigned int, nvmlDevice_t *);
using processes_v1_fn = nvmlReturn_t (*)(nvmlDevice_t, unsigned int *,
                                         nvmlProcessInfo_v1_t *);
using processes_v3_fn = nvmlReturn_t (*)(nvmlDevice_t, unsigned int *,
                                         nvmlProcessInfo_t *);

template <typename T> T lookup(void *library, const char *name) {
  auto symbol = reinterpret_cast<T>(dlsym(library, name));
  if (symbol == nullptr) {
    std::fprintf(stderr, "missing NVML symbol %s\n", name);
  }
  return symbol;
}

} // namespace

int main() {
  CUdevice cu_device = 0;
  CUcontext context = nullptr;
  CUdeviceptr allocation = 0;
  if (cuInit(0) != CUDA_SUCCESS || cuDeviceGet(&cu_device, 0) != CUDA_SUCCESS ||
      cuDevicePrimaryCtxRetain(&context, cu_device) != CUDA_SUCCESS ||
      cuCtxSetCurrent(context) != CUDA_SUCCESS ||
      cuMemAlloc(&allocation, 64 << 20) != CUDA_SUCCESS) {
    std::fprintf(stderr, "failed to hold a CUDA context\n");
    return 1;
  }

  void *library = dlopen("libnvidia-ml.so.1", RTLD_NOW);
  if (library == nullptr) {
    std::fprintf(stderr, "dlopen(libnvidia-ml.so.1) failed: %s\n", dlerror());
    return 1;
  }
  auto nvml_init = lookup<init_fn>(library, "nvmlInit_v2");
  auto nvml_shutdown = lookup<shutdown_fn>(library, "nvmlShutdown");
  auto device_count = lookup<count_fn>(library, "nvmlDeviceGetCount_v2");
  auto handle_by_index =
      lookup<handle_by_index_fn>(library, "nvmlDeviceGetHandleByIndex_v2");
  auto processes_v1 = lookup<processes_v1_fn>(
      library, "nvmlDeviceGetComputeRunningProcesses");
  auto processes_v3 = lookup<processes_v3_fn>(
      library, "nvmlDeviceGetComputeRunningProcesses_v3");
  if (nvml_init == nullptr || nvml_shutdown == nullptr ||
      device_count == nullptr || handle_by_index == nullptr ||
      processes_v1 == nullptr) {
    return 1;
  }
  if (nvml_init() != NVML_SUCCESS) {
    std::fprintf(stderr, "nvmlInit_v2 failed\n");
    return 1;
  }

  unsigned int devices = 0;
  if (device_count(&devices) != NVML_SUCCESS) {
    std::fprintf(stderr, "nvmlDeviceGetCount_v2 failed\n");
    return 1;
  }
  int failures = processes_v3 == nullptr ? 1 : 0;
  unsigned int seen = 0;
  for (unsigned int index = 0; index < devices; ++index) {
    nvmlDevice_t device = nullptr;
    if (handle_by_index(index, &device) != NVML_SUCCESS) {
      std::fprintf(stderr, "nvmlDeviceGetHandleByIndex_v2(%u) failed\n", index);
      return 1;
    }
    std::vector<unsigned char> storage;
    unsigned int count = 0;
    nvmlReturn_t status = NVML_ERROR_INSUFFICIENT_SIZE;
    // The process list can grow between the size query and the fill.
    while (status == NVML_ERROR_INSUFFICIENT_SIZE) {
      status = processes_v1(device, &count, nullptr);
      if (status != NVML_SUCCESS && status != NVML_ERROR_INSUFFICIENT_SIZE) {
        break;
      }
      storage.assign(count * sizeof(nvmlProcessInfo_v1_t) + kGuardBytes,
                     kGuard);
      status = processes_v1(
          device, &count,
          reinterpret_cast<nvmlProcessInfo_v1_t *>(storage.data()));
    }
    if (status != NVML_SUCCESS) {
      std::fprintf(stderr, "device %u: v1 query failed: %d\n", index,
                   static_cast<int>(status));
      return 1;
    }
    for (size_t i = count * sizeof(nvmlProcessInfo_v1_t); i < storage.size();
         ++i) {
      if (storage[i] != kGuard) {
        std::fprintf(stderr,
                     "device %u: %u v1 entries wrote past the array (guard "
                     "byte %zu is 0x%02x)\n",
                     index, count, i - count * sizeof(nvmlProcessInfo_v1_t),
                     storage[i]);
        ++failures;
        break;
      }
    }

    const auto *v1 =
        reinterpret_cast<const nvmlProcessInfo_v1_t *>(storage.data());
    for (unsigned int i = 0; i < count; ++i) {
      std::printf("device %u: pid %u used %llu\n", index, v1[i].pid,
                  static_cast<unsigned long long>(v1[i].usedGpuMemory));
    }
    seen += count;
    if (processes_v3 == nullptr) {
      continue;
    }

    std::vector<nvmlProcessInfo_t> v3(count + 4);
    unsigned int v3_count = static_cast<unsigned int>(v3.size());
    status = processes_v3(device, &v3_count, v3.data());
    if (status != NVML_SUCCESS) {
      std::fprintf(stderr, "device %u: v3 query failed: %d\n", index,
                   static_cast<int>(status));
      return 1;
    }
    for (unsigned int i = 0; i < count; ++i) {
      bool found = false;
      for (unsigned int j = 0; j < v3_count; ++j) {
        found |= v3[j].pid == v1[i].pid;
      }
      if (!found) {
        std::fprintf(stderr, "device %u: v1 pid %u is not in the v3 list\n",
                     index, v1[i].pid);
        ++failures;
      }
    }
  }

  nvml_shutdown();
  cuMemFree(allocation);
  cuDevicePrimaryCtxRelease(cu_device);
  if (seen == 0) {
    std::fprintf(stderr, "no compute process was reported\n");
    return 1;
  }
  if (failures != 0) {
    return 1;
  }
  std::printf("v1 running-process entries stay inside the caller's array\n");
  return 0;
}
