#include "ops/smemcpy_module.h"

#include <cstring>

extern "C" const void *lupine_cuda_smemcpy_image();

namespace {

CUresult query_current_smemcpy_device(lupine_smemcpy_device *features) {
  CUdevice device = 0;
  CUresult result = cuCtxGetDevice(&device);
  if (result != CUDA_SUCCESS) {
    return result;
  }
  *features = {};
  result = cuDeviceGetAttribute(
      &features->major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, device);
  if (result == CUDA_SUCCESS) {
    result = cuDeviceGetAttribute(
        &features->minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, device);
  }
  char name[128] = {};
  if (result == CUDA_SUCCESS) {
    result = cuDeviceGetName(name, sizeof(name), device);
  }
  if (result != CUDA_SUCCESS) {
    return result;
  }
  features->is_l4 = std::strcmp(name, "NVIDIA L4") == 0;
  return CUDA_SUCCESS;
}

} // namespace

extern "C" CUresult lupine_smemcpy_module_load(CUmodule *module) {
  if (module == nullptr) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  return cuModuleLoadData(module, lupine_cuda_smemcpy_image());
}

extern "C" CUresult
lupine_smemcpy_prepare_driver_launch(const lupine_smemcpy_params *params,
                                     lupine_smemcpy_launch_descriptor *launch) {
  CUresult result = lupine_smemcpy_prepare_descriptor(params, nullptr, launch);
  if (result != CUDA_SUCCESS || params->bytes == 0 || params->width > 3 ||
      launch->use_cuda_memcpy) {
    return result;
  }

  lupine_smemcpy_device features = {};
  result = query_current_smemcpy_device(&features);
  if (result != CUDA_SUCCESS) {
    return result;
  }
  return lupine_smemcpy_prepare_descriptor(params, &features, launch);
}

extern "C" CUresult
lupine_smemcpy_module_function(CUmodule module,
                               const lupine_smemcpy_launch_descriptor *launch,
                               CUfunction *function) {
  if (module == nullptr || function == nullptr) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  const char *name = lupine_smemcpy_kernel_name(launch);
  return name == nullptr ? CUDA_ERROR_INVALID_VALUE
                         : cuModuleGetFunction(function, module, name);
}

CUresult
lupine_smemcpy_cached_module_load(lupine_smemcpy_cached_module *cached) {
  if (cached == nullptr) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  lupine_smemcpy_device features = {};
  CUresult result = query_current_smemcpy_device(&features);
  if (result != CUDA_SUCCESS) {
    return result;
  }
  CUmodule module = nullptr;
  result = lupine_smemcpy_module_load(&module);
  if (result != CUDA_SUCCESS) {
    return result;
  }
  cached->module = module;
  cached->device = features;
  for (auto &fn : cached->functions) {
    fn.store(nullptr, std::memory_order_relaxed);
  }
  return CUDA_SUCCESS;
}

void lupine_smemcpy_cached_module_unload(lupine_smemcpy_cached_module *cached) {
  if (cached == nullptr || cached->module == nullptr) {
    return;
  }
  (void)cuModuleUnload(cached->module);
  cached->module = nullptr;
  for (auto &fn : cached->functions) {
    fn.store(nullptr, std::memory_order_relaxed);
  }
}

CUresult
lupine_smemcpy_cached_prepare_launch(const lupine_smemcpy_cached_module *cached,
                                     const lupine_smemcpy_params *params,
                                     lupine_smemcpy_launch_descriptor *launch) {
  if (cached == nullptr) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  return lupine_smemcpy_prepare_descriptor(params, &cached->device, launch);
}

CUresult
lupine_smemcpy_cached_function(lupine_smemcpy_cached_module *cached,
                               const lupine_smemcpy_launch_descriptor *launch,
                               CUfunction *function) {
  if (cached == nullptr || cached->module == nullptr || function == nullptr) {
    return CUDA_ERROR_INVALID_VALUE;
  }
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
  if (cached == nullptr || function == nullptr) {
    return false;
  }
  for (const auto &entry : cached->functions) {
    if (entry.load(std::memory_order_acquire) == function) {
      return true;
    }
  }
  return false;
}
