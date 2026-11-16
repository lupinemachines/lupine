#include "ops/smemcpy_module.h"

#include <cstring>

extern "C" const void *lupine_cuda_smemcpy_image();

CUresult
lupine_smemcpy_cached_module_load(lupine_smemcpy_cached_module *cached) {
  CUdevice device = 0;
  char name[128] = {};
  lupine_smemcpy_device &features = cached->device;
  CUresult result;
  if ((result = cuCtxGetDevice(&device)) != CUDA_SUCCESS ||
      (result = cuDeviceGetAttribute(
           &features.major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR,
           device)) != CUDA_SUCCESS ||
      (result = cuDeviceGetAttribute(
           &features.minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR,
           device)) != CUDA_SUCCESS ||
      (result = cuDeviceGetName(name, sizeof(name), device)) != CUDA_SUCCESS) {
    return result;
  }
  features.is_l4 = std::strcmp(name, "NVIDIA L4") == 0;
  return cuModuleLoadData(&cached->module, lupine_cuda_smemcpy_image());
}

void lupine_smemcpy_cached_module_unload(lupine_smemcpy_cached_module *cached) {
  if (cached->module != nullptr) {
    (void)cuModuleUnload(cached->module);
  }
}

CUresult
lupine_smemcpy_cached_function(lupine_smemcpy_cached_module *cached,
                               const lupine_smemcpy_launch_descriptor *launch,
                               CUfunction *function) {
  int index = lupine_smemcpy_kernel_index(launch);
  if (index < 0) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  CUfunction resolved =
      cached->functions[index].load(std::memory_order_acquire);
  if (resolved == nullptr) {
    CUresult result = cuModuleGetFunction(&resolved, cached->module,
                                          lupine_smemcpy_kernel_name_at(index));
    if (result != CUDA_SUCCESS) {
      return result;
    }
    cached->functions[index].store(resolved, std::memory_order_release);
  }
  *function = resolved;
  return CUDA_SUCCESS;
}

bool lupine_smemcpy_cached_contains_function(
    const lupine_smemcpy_cached_module *cached, CUfunction function) {
  for (const auto &entry : cached->functions) {
    if (entry.load(std::memory_order_acquire) == function) {
      return true;
    }
  }
  return false;
}
