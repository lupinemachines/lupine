#ifndef LUPINE_OPS_SMEMCPY_MODULE_H
#define LUPINE_OPS_SMEMCPY_MODULE_H

#include "ops/smemcpy_dispatch.h"

#ifdef __cplusplus
extern "C" {
#endif

// Loads the precompiled scatter kernels into the current CUDA context. The
// returned module is owned by the caller and must be unloaded before its
// context is destroyed.
CUresult lupine_smemcpy_module_load(CUmodule *module);

// Selects a launch using Driver API device properties only.
CUresult
lupine_smemcpy_prepare_driver_launch(const lupine_smemcpy_params *params,
                                     lupine_smemcpy_launch_descriptor *launch);

// Resolves the selected stable kernel entry point in a loaded module.
CUresult
lupine_smemcpy_module_function(CUmodule module,
                               const lupine_smemcpy_launch_descriptor *launch,
                               CUfunction *function);

#ifdef __cplusplus
}

#include <atomic>

struct lupine_smemcpy_cached_module {
  CUmodule module = nullptr;
  lupine_smemcpy_device device = {};
  std::atomic<CUfunction> functions[LUPINE_SMEMCPY_KERNEL_COUNT] = {};
};

CUresult
lupine_smemcpy_cached_module_load(lupine_smemcpy_cached_module *cached);
void lupine_smemcpy_cached_module_unload(lupine_smemcpy_cached_module *cached);
CUresult
lupine_smemcpy_cached_prepare_launch(const lupine_smemcpy_cached_module *cached,
                                     const lupine_smemcpy_params *params,
                                     lupine_smemcpy_launch_descriptor *launch);
CUresult
lupine_smemcpy_cached_function(lupine_smemcpy_cached_module *cached,
                               const lupine_smemcpy_launch_descriptor *launch,
                               CUfunction *function);
bool lupine_smemcpy_cached_contains_function(
    const lupine_smemcpy_cached_module *cached, CUfunction function);
#endif

#endif
