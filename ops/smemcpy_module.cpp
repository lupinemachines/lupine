#include "ops/smemcpy_module.h"

#include "smemcpy.inc"

#include <cstring>

extern "C" CUresult lupine_smemcpy_module_load(CUmodule *module) {
  if (module == nullptr) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  return cuModuleLoadData(module, lupine_smemcpy_fatbin);
}

extern "C" CUresult
lupine_smemcpy_prepare_driver_launch(const lupine_smemcpy_params *params,
                                     lupine_smemcpy_launch_descriptor *launch) {
  CUresult result = lupine_smemcpy_prepare_descriptor(params, nullptr, launch);
  if (result != CUDA_SUCCESS || params->bytes == 0 || params->width > 3 ||
      launch->use_cuda_memcpy) {
    return result;
  }

  CUdevice device = 0;
  result = cuCtxGetDevice(&device);
  if (result != CUDA_SUCCESS) {
    return result;
  }
  lupine_smemcpy_device features = {};
  result = cuDeviceGetAttribute(
      &features.major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, device);
  if (result == CUDA_SUCCESS) {
    result = cuDeviceGetAttribute(
        &features.minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, device);
  }
  char name[128] = {};
  if (result == CUDA_SUCCESS) {
    result = cuDeviceGetName(name, sizeof(name), device);
  }
  if (result != CUDA_SUCCESS) {
    return result;
  }
  features.is_l4 = std::strcmp(name, "NVIDIA L4") == 0;
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
