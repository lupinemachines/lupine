#include <cuda_runtime_api.h>

#include <cstdio>
#include <cstdlib>
#include <thread>

extern "C" unsigned __cudaPushCallConfiguration(dim3, dim3, size_t,
                                                cudaStream_t);
extern "C" cudaError_t __cudaPopCallConfiguration(dim3 *, dim3 *, size_t *,
                                                  void *);

static void require(bool condition, const char *message) {
  if (!condition) {
    std::fprintf(stderr, "%s\n", message);
    std::exit(1);
  }
}

static void check_stack(unsigned offset) {
  for (unsigned i = 1; i <= 100; ++i) {
    require(__cudaPushCallConfiguration(dim3(i + offset, 2, 3), dim3(4, 5, 6),
                                        i * 16, cudaStreamPerThread) == 0,
            "push failed");
  }
  for (unsigned i = 100; i != 0; --i) {
    dim3 grid, block;
    size_t shared = 0;
    cudaStream_t stream = nullptr;
    require(__cudaPopCallConfiguration(&grid, &block, &shared, &stream) ==
                cudaSuccess,
            "pop failed");
    require(grid.x == i + offset && grid.y == 2 && grid.z == 3 &&
                block.x == 4 && block.y == 5 && block.z == 6 &&
                shared == i * 16 && stream == cudaStreamPerThread,
            "configuration stack lost or reordered a field");
  }
}

__global__ void write_value(int *output) { *output = 42; }

int main() {
  require(__cudaPushCallConfiguration(dim3(7), dim3(8), 9, cudaStreamLegacy) ==
              0,
          "outer push failed");
  std::thread other([] { check_stack(1000); });
  check_stack(0);
  other.join();
  dim3 grid, block;
  size_t shared = 0;
  cudaStream_t stream = nullptr;
  require(__cudaPopCallConfiguration(&grid, &block, &shared, &stream) ==
                  cudaSuccess &&
              grid.x == 7 && block.x == 8 && shared == 9 &&
              stream == cudaStreamLegacy,
          "thread-local outer configuration was overwritten");

  (void)cudaGetLastError();
  require(__cudaPopCallConfiguration(&grid, &block, &shared, &stream) ==
              cudaErrorMissingConfiguration,
          "empty pop returned the wrong error");
  require(cudaPeekAtLastError() == cudaErrorMissingConfiguration &&
              cudaGetLastError() == cudaErrorMissingConfiguration &&
              cudaGetLastError() == cudaSuccess,
          "empty pop did not preserve native last-error semantics");

  int *output = nullptr;
  require(cudaMalloc(&output, sizeof(int)) == cudaSuccess, "allocation failed");
  write_value<<<1, 1>>>(output);
  require(cudaGetLastError() == cudaSuccess, "nvcc launch failed");
  int value = 0;
  require(cudaMemcpy(&value, output, sizeof(value), cudaMemcpyDeviceToHost) ==
                  cudaSuccess &&
              value == 42,
          "nvcc launch did not receive its configuration");
  require(cudaFree(output) == cudaSuccess, "free failed");
  std::puts("runtime launch configuration preserves nesting, threads, errors, "
            "and nvcc launches");
}
