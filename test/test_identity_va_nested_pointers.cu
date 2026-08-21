#include <cuda_runtime.h>

#include <cstdio>

struct nested_pointer {
  int *value;
};

__global__ void increment_nested(nested_pointer *nested) {
  if (blockIdx.x == 0 && threadIdx.x == 0) {
    *nested->value += 1;
  }
}

static bool check(cudaError_t result, const char *call) {
  if (result == cudaSuccess) {
    return true;
  }
  std::fprintf(stderr, "%s failed: %d (%s)\n", call, static_cast<int>(result),
               cudaGetErrorString(result));
  return false;
}

int main() {
  nested_pointer *nested = nullptr;
  int *value = nullptr;
  if (!check(cudaMallocManaged(&nested, sizeof(*nested)),
             "cudaMallocManaged nested") ||
      !check(cudaMallocManaged(&value, sizeof(*value)),
             "cudaMallocManaged value")) {
    return 1;
  }

  *value = 41;
  nested->value = value;
  increment_nested<<<1, 1>>>(nested);
  if (!check(cudaGetLastError(), "increment_nested launch") ||
      !check(cudaDeviceSynchronize(), "cudaDeviceSynchronize")) {
    return 1;
  }
  if (*value != 42) {
    std::fprintf(stderr, "nested managed pointer produced %d, expected 42\n",
                 *value);
    return 1;
  }

  if (!check(cudaFree(value), "cudaFree value") ||
      !check(cudaFree(nested), "cudaFree nested")) {
    return 1;
  }
  std::printf("nested managed pointer preserved across RPC\n");
  return 0;
}
