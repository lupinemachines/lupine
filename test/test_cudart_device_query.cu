// Exercises the CUDA runtime shim's forwarded device, version, and error
// surface through the driver shim's connection.
#include <cuda_runtime.h>

#include <cstdio>
#include <cstring>

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

int main() {
  int count = 0;
  CHECK(cudaGetDeviceCount(&count));
  EXPECT(count > 0);

  int runtime_version = 0;
  int driver_version = 0;
  CHECK(cudaRuntimeGetVersion(&runtime_version));
  CHECK(cudaDriverGetVersion(&driver_version));
  printf("runtime %d driver %d devices %d\n", runtime_version, driver_version,
         count);
  EXPECT(runtime_version >= 11000 && driver_version >= runtime_version);

  for (int device = 0; device < count; ++device) {
    CHECK(cudaSetDevice(device));
    int current = -1;
    CHECK(cudaGetDevice(&current));
    EXPECT(current == device);

    cudaDeviceProp prop;
    memset(&prop, 0, sizeof(prop));
    CHECK(cudaGetDeviceProperties(&prop, device));
    int multiprocessors = 0;
    CHECK(cudaDeviceGetAttribute(&multiprocessors,
                                 cudaDevAttrMultiProcessorCount, device));
    EXPECT(strlen(prop.name) > 0);
    EXPECT(prop.multiProcessorCount == multiprocessors);
    EXPECT(prop.totalGlobalMem > 0);

    char bus_id[32] = {};
    CHECK(cudaDeviceGetPCIBusId(bus_id, sizeof(bus_id), device));
    EXPECT(strlen(bus_id) > 0);

    size_t stack_size = 0;
    CHECK(cudaDeviceGetLimit(&stack_size, cudaLimitStackSize));
    EXPECT(stack_size > 0);

    int least = 0;
    int greatest = 0;
    CHECK(cudaDeviceGetStreamPriorityRange(&least, &greatest));
    EXPECT(greatest <= least);

    CHECK(cudaDeviceSynchronize());
    printf("device %d: %s sm=%d mem=%zu pci=%s\n", device, prop.name,
           prop.multiProcessorCount, prop.totalGlobalMem, bus_id);
  }

  // The sticky error lives in the server's runtime, on this thread's lane.
  EXPECT(cudaSetDevice(count + 7) == cudaErrorInvalidDevice);
  int value = 0;
  EXPECT(cudaDeviceGetAttribute(&value, (cudaDeviceAttr)999999, 0) ==
         cudaErrorInvalidValue);
  EXPECT(cudaPeekAtLastError() == cudaErrorInvalidValue);
  EXPECT(cudaGetLastError() == cudaErrorInvalidValue);
  EXPECT(cudaGetLastError() == cudaSuccess);
  EXPECT(strcmp(cudaGetErrorName(cudaErrorInvalidValue),
                "cudaErrorInvalidValue") == 0);
  EXPECT(strcmp(cudaGetErrorString(cudaErrorInvalidValue),
                "invalid argument") == 0);
  EXPECT(strcmp(cudaGetErrorString(cudaSuccess), "no error") == 0);

  printf("test_cudart_device_query passed\n");
  return 0;
}
