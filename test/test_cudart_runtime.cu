// Exercises the CUDA runtime shim end to end with code nvcc compiled: the
// registration ABI forwards the embedded fatbin to the server's runtime, the
// kernel launch names the host entry point the server registered, and the
// copies, symbols, streams, and events run against the remote device.
#include <cuda_runtime.h>

#include <cstdio>
#include <cstring>
#include <vector>

#define CHECK(call)                                                            \
  do {                                                                         \
    cudaError_t status = (call);                                               \
    if (status != cudaSuccess) {                                               \
      fprintf(stderr, "%s failed: %s (%s)\n", #call, cudaGetErrorName(status), \
              cudaGetErrorString(status));                                     \
      return 1;                                                                \
    }                                                                          \
  } while (0)

#define EXPECT(condition)                                                      \
  do {                                                                         \
    if (!(condition)) {                                                        \
      fprintf(stderr, "expectation failed: %s\n", #condition);                 \
      return 1;                                                                \
    }                                                                          \
  } while (0)

__device__ int device_scale;

__global__ void scale_and_add(const int *input, int *output, int addend,
                              int count) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < count) {
    output[index] = input[index] * device_scale + addend;
  }
}

int main() {
  int count = 0;
  CHECK(cudaGetDeviceCount(&count));
  EXPECT(count > 0);

  cudaDeviceProp properties;
  CHECK(cudaGetDeviceProperties(&properties, 0));
  printf("device 0: %s (sm %d.%d)\n", properties.name, properties.major,
         properties.minor);

  constexpr int kCount = 4096;
  std::vector<int> host_input(kCount);
  std::vector<int> host_output(kCount, 0);
  for (int i = 0; i < kCount; ++i) {
    host_input[i] = i;
  }

  int *input = nullptr;
  int *output = nullptr;
  CHECK(cudaMalloc(&input, kCount * sizeof(int)));
  CHECK(cudaMalloc(&output, kCount * sizeof(int)));
  CHECK(cudaMemset(output, 0, kCount * sizeof(int)));

  cudaStream_t stream;
  CHECK(cudaStreamCreate(&stream));
  cudaEvent_t start;
  cudaEvent_t stop;
  CHECK(cudaEventCreate(&start));
  CHECK(cudaEventCreate(&stop));

  const int scale = 3;
  CHECK(cudaMemcpyToSymbol(device_scale, &scale, sizeof(scale)));
  int scale_readback = 0;
  CHECK(cudaMemcpyFromSymbol(&scale_readback, device_scale, sizeof(scale)));
  EXPECT(scale_readback == scale);

  void *symbol_address = nullptr;
  size_t symbol_size = 0;
  CHECK(cudaGetSymbolAddress(&symbol_address, device_scale));
  CHECK(cudaGetSymbolSize(&symbol_size, device_scale));
  EXPECT(symbol_address != nullptr && symbol_size == sizeof(int));

  CHECK(cudaMemcpyAsync(input, host_input.data(), kCount * sizeof(int),
                        cudaMemcpyHostToDevice, stream));
  CHECK(cudaEventRecord(start, stream));
  scale_and_add<<<(kCount + 255) / 256, 256, 0, stream>>>(input, output, 7,
                                                          kCount);
  CHECK(cudaGetLastError());
  CHECK(cudaEventRecord(stop, stream));
  CHECK(cudaMemcpyAsync(host_output.data(), output, kCount * sizeof(int),
                        cudaMemcpyDeviceToHost, stream));
  CHECK(cudaStreamSynchronize(stream));

  float elapsed = -1.0f;
  CHECK(cudaEventElapsedTime(&elapsed, start, stop));
  EXPECT(elapsed >= 0.0f);

  for (int i = 0; i < kCount; ++i) {
    if (host_output[i] != i * scale + 7) {
      fprintf(stderr, "output[%d] = %d, expected %d\n", i, host_output[i],
              i * scale + 7);
      return 1;
    }
  }

  cudaFuncAttributes attributes;
  CHECK(cudaFuncGetAttributes(&attributes, (const void *)scale_and_add));
  EXPECT(attributes.maxThreadsPerBlock > 0);
  int blocks = 0;
  CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &blocks, (const void *)scale_and_add, 256, 0));
  EXPECT(blocks > 0);

  // cudaMemcpyDefault has the shim ask the server which side is its memory.
  std::vector<int> readback(kCount, 0);
  CHECK(cudaMemcpy(readback.data(), output, kCount * sizeof(int),
                   cudaMemcpyDefault));
  EXPECT(readback[kCount - 1] == (kCount - 1) * scale + 7);
  CHECK(cudaMemcpy(input, readback.data(), kCount * sizeof(int),
                   cudaMemcpyDefault));

  cudaPointerAttributes pointer_attributes;
  CHECK(cudaPointerGetAttributes(&pointer_attributes, output));
  EXPECT(pointer_attributes.type == cudaMemoryTypeDevice);

  CHECK(cudaEventDestroy(start));
  CHECK(cudaEventDestroy(stop));
  CHECK(cudaStreamDestroy(stream));
  CHECK(cudaFree(input));
  CHECK(cudaFree(output));
  CHECK(cudaDeviceSynchronize());

  // A failure raised by the shim itself has to reach the sticky error too.
  EXPECT(cudaSetDevice(count + 7) == cudaErrorInvalidDevice);
  EXPECT(cudaGetLastError() != cudaSuccess);
  EXPECT(cudaGetLastError() == cudaSuccess);

  printf("test_cudart_runtime passed\n");
  return 0;
}
