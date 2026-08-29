#ifndef LUPINE_OPS_SMEMCPY_H
#define LUPINE_OPS_SMEMCPY_H

#include <cuda_runtime_api.h>

#include "ops/smemcpy_dispatch.h"

#include <stdbool.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

// Describes the operation selected for a set of parameters. A contiguous
// fragment uses CUDA memcpy; otherwise pass the address of params as the
// selected kernel's sole argument. The prepared parameters may be rebased
// relative to the input.
typedef struct lupine_smemcpy_launch {
  lupine_smemcpy_params params;
  bool use_cuda_memcpy;
  const void *kernel;
  unsigned int blocks;
  unsigned int threads;
} lupine_smemcpy_launch;

// Selects the best safe implementation and launch geometry for the current
// CUDA device. This is useful to graph code constructing or updating an
// equivalent kernel node.
cudaError_t lupine_smemcpy_prepare_launch(const lupine_smemcpy_params *params,
                                          lupine_smemcpy_launch *launch);

// Enqueues one scatter copy. Source byte i is written to the pitched
// destination coordinate represented by logical_offset + i.
cudaError_t lupine_smemcpy_async(const lupine_smemcpy_params *params,
                                 cudaStream_t stream);

// Returns the fully general scalar kernel. Prefer prepare_launch: this symbol
// is retained as a compatibility fallback for graph code that controls its own
// launch geometry.
const void *lupine_smemcpy_kernel(void);

#ifdef __cplusplus
}
#endif

#endif
