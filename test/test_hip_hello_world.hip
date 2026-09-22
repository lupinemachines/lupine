#include <hip/hip_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>

#define CHECK(call)                                                            \
  do {                                                                         \
    hipError_t error = (call);                                                 \
    if (error != hipSuccess) {                                                 \
      fprintf(stderr, "%s: %s (%s)\n", #call, hipGetErrorName(error),          \
              hipGetErrorString(error));                                       \
      return 1;                                                                \
    }                                                                          \
  } while (0)

__global__ void hello_world() {
  printf("Hello from HIP block %u thread %u!\n", blockIdx.x, threadIdx.x);
}

__global__ void requires_argument(int value) {
  printf("argument kernel executed: %d\n", value);
}

int main() {
  puts("Hello from the host!");
  // Exercise the actual compiler-generated push/pop and registration ABI.
  hello_world<<<2, 2>>>();
  CHECK(hipGetLastError());
  CHECK(hipDeviceSynchronize());

  hipStream_t stream = nullptr;
  CHECK(hipStreamCreate(&stream));
  CHECK(hipLaunchKernel(reinterpret_cast<const void *>(hello_world), dim3(1),
                        dim3(1), nullptr, 0, stream));
  CHECK(hipStreamSynchronize(stream));
  CHECK(hipStreamDestroy(stream));

  if (getenv("LUPINE_HIP_SHIM") != nullptr) {
    int value = 42;
    void *args[] = {&value};
    auto error =
        hipLaunchKernel(reinterpret_cast<const void *>(requires_argument),
                        dim3(1), dim3(1), args, 0, nullptr);
    if (error != hipErrorInvalidValue) {
      fprintf(stderr, "argument kernel was not rejected: %d\n", error);
      return 1;
    }
    CHECK(hipGetLastError() == hipErrorInvalidValue ? hipSuccess
                                                    : hipErrorUnknown);
  }
  if (strcmp(hipGetErrorName(hipErrorInvalidValue), "hipErrorInvalidValue") !=
          0 ||
      strcmp(hipGetErrorString(hipErrorInvalidValue), "invalid argument") !=
          0) {
    fputs("incorrect error description\n", stderr);
    return 1;
  }

  // A reset invalidates the module cache. The next launch must upload again.
  CHECK(hipDeviceReset());
  CHECK(hipLaunchKernel(reinterpret_cast<const void *>(hello_world), dim3(1),
                        dim3(1), nullptr, 0, nullptr));
  CHECK(hipDeviceSynchronize());
  puts("PASS");
  return 0;
}
