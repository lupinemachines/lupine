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
#endif

#endif
