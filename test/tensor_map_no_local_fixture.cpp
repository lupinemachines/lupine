// Hide local devices from the client so the benchmark must use RPC.
#include <cuda.h>

extern "C" CUresult CUDAAPI cuInit(unsigned int) { return CUDA_SUCCESS; }
extern "C" CUresult CUDAAPI cuDeviceGetCount(int *count) {
  *count = 0;
  return CUDA_SUCCESS;
}
