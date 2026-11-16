#ifndef LUPINE_OPS_SMEMCPY_MODULE_H
#define LUPINE_OPS_SMEMCPY_MODULE_H

#include "ops/smemcpy_dispatch.h"

#include <atomic>

// The precompiled scatter kernels loaded into one CUDA context, which must
// unload them before it is destroyed.
struct lupine_smemcpy_cached_module {
  CUmodule module = nullptr;
  lupine_smemcpy_device device = {};
  std::atomic<CUfunction> functions[LUPINE_SMEMCPY_KERNEL_COUNT] = {};
};

CUresult
lupine_smemcpy_cached_module_load(lupine_smemcpy_cached_module *cached);
void lupine_smemcpy_cached_module_unload(lupine_smemcpy_cached_module *cached);
CUresult
lupine_smemcpy_cached_function(lupine_smemcpy_cached_module *cached,
                               const lupine_smemcpy_launch_descriptor *launch,
                               CUfunction *function);
bool lupine_smemcpy_cached_contains_function(
    const lupine_smemcpy_cached_module *cached, CUfunction function);

#endif
