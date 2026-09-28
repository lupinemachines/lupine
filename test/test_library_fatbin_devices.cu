// Build with -arch=all (the custom runner's default) to exercise the uploaded
// fatbin on every exposed device, including heterogeneous servers.
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

__global__ void compiled_architecture(int *out) {
#ifdef __CUDA_ARCH__
  *out = __CUDA_ARCH__;
#endif
}

static void check(cudaError_t result, const char *operation) {
  if (result != cudaSuccess) {
    std::fprintf(stderr, "%s: %s\n", operation, cudaGetErrorString(result));
    std::exit(1);
  }
}

int main() {
  int count = 0;
  check(cudaGetDeviceCount(&count), "device count");
  if (count == 0) {
    return 1;
  }
  for (int device = 0; device < count; ++device) {
    check(cudaSetDevice(device), "set device");
    cudaDeviceProp properties;
    check(cudaGetDeviceProperties(&properties, device), "device properties");
    int *output = nullptr;
    check(cudaMalloc(&output, sizeof(int)), "allocate");
    compiled_architecture<<<1, 1>>>(output);
    int architecture = 0;
    check(
        cudaMemcpy(&architecture, output, sizeof(int), cudaMemcpyDeviceToHost),
        "read architecture");
    if (architecture <= 0 ||
        architecture > properties.major * 100 + properties.minor * 10) {
      std::fprintf(stderr, "device %d executed incompatible code %d\n", device,
                   architecture);
      return 1;
    }
    check(cudaFree(output), "free");
    std::printf("device %d SM%d%d executes SM%d code\n", device,
                properties.major, properties.minor, architecture / 10);
  }
  std::puts("test_library_fatbin_devices: PASS");
}
