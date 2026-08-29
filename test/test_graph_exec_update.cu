// Exercise the cudaGraphExecUpdate ABI selected by the active CUDA runtime.
// CUDA 11 uses separate error-node and update-result outputs, while CUDA 12+
// uses cudaGraphExecUpdateResultInfo.
#include <cstdio>
#include <cuda_runtime_api.h>

static int check(cudaError_t result, const char *operation) {
  if (result == cudaSuccess)
    return 0;
  fprintf(stderr, "FAIL: %s: %s\n", operation, cudaGetErrorString(result));
  return 1;
}

int main() {
  cudaGraph_t original = nullptr;
  cudaGraph_t update = nullptr;
  cudaGraphExec_t exec = nullptr;
  cudaGraphNode_t original_node = nullptr;
  cudaGraphNode_t update_node = nullptr;
  int failed = 0;

  failed |= check(cudaGraphCreate(&original, 0), "cudaGraphCreate(original)");
  failed |= check(cudaGraphAddEmptyNode(&original_node, original, nullptr, 0),
                  "cudaGraphAddEmptyNode(original)");
  failed |= check(cudaGraphInstantiate(&exec, original, nullptr, nullptr, 0),
                  "cudaGraphInstantiate");
  failed |= check(cudaGraphCreate(&update, 0), "cudaGraphCreate(update)");
  failed |= check(cudaGraphAddEmptyNode(&update_node, update, nullptr, 0),
                  "cudaGraphAddEmptyNode(update)");
  if (failed)
    return 1;

  cudaGraphExecUpdateResult update_result = cudaGraphExecUpdateError;
#if CUDART_VERSION >= 12000
  cudaGraphExecUpdateResultInfo result_info = {};
  cudaError_t result = cudaGraphExecUpdate(exec, update, &result_info);
  update_result = result_info.result;
#else
  cudaGraphNode_t error_node = nullptr;
  cudaError_t result =
      cudaGraphExecUpdate(exec, update, &error_node, &update_result);
#endif
  if (result != cudaSuccess || update_result != cudaGraphExecUpdateSuccess) {
    fprintf(stderr, "FAIL: cudaGraphExecUpdate: %s, update result %d\n",
            cudaGetErrorString(result), static_cast<int>(update_result));
    failed = 1;
  }

  failed |= check(cudaGraphExecDestroy(exec), "cudaGraphExecDestroy");
  failed |= check(cudaGraphDestroy(update), "cudaGraphDestroy(update)");
  failed |= check(cudaGraphDestroy(original), "cudaGraphDestroy(original)");

  if (!failed)
    fprintf(stderr, "PASS: cudaGraphExecUpdate\n");
  return failed ? 1 : 0;
}
