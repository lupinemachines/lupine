// The runtime's per-thread last error must survive being answered without
// asking the server: a failure that is never read stays until cudaGetLastError
// takes it, and an asynchronous launch failure still reaches the caller from
// the call that synchronizes with the device.
#include <cuda_runtime_api.h>

#include <cstdio>
#include <cstdlib>
#include <thread>

static void require(bool condition, const char *message) {
  if (!condition) {
    std::fprintf(stderr, "%s\n", message);
    std::exit(1);
  }
}

__global__ void store_out_of_range(int *values, long long index) {
  values[index] = 1;
}

__global__ void spin(long long iterations) {
  for (long long i = 0; i < iterations; ++i) {
    __nanosleep(1000);
  }
}

int main() {
  (void)cudaGetLastError();

  void *unusable = nullptr;
  const cudaError_t allocation_error =
      cudaMalloc(&unusable, static_cast<size_t>(1) << 60);
  require(allocation_error != cudaSuccess, "an impossible allocation succeeded");
  require(cudaPeekAtLastError() == allocation_error,
          "peek did not report the failed call's error");
  require(cudaPeekAtLastError() == allocation_error, "peek cleared the error");

  int *values = nullptr;
  require(cudaMalloc(&values, sizeof(int)) == cudaSuccess,
          "allocation failed");
  require(cudaPeekAtLastError() == allocation_error,
          "a successful call cleared the error");
  require(cudaGetLastError() == allocation_error,
          "cudaGetLastError did not report the failed call's error");
  require(cudaGetLastError() == cudaSuccess,
          "cudaGetLastError did not clear the error");

  const char *name = cudaGetErrorName(allocation_error);
  const char *description = cudaGetErrorString(allocation_error);
  require(name != nullptr && name[0] != '\0' && description != nullptr &&
              description[0] != '\0',
          "the error has no name or description");
  require(cudaPeekAtLastError() == cudaSuccess,
          "naming an error recorded one");

  std::thread other([] {
    void *unreachable = nullptr;
    require(cudaMalloc(&unreachable, static_cast<size_t>(1) << 60) !=
                cudaSuccess,
            "an impossible allocation succeeded on the second thread");
  });
  other.join();
  require(cudaPeekAtLastError() == cudaSuccess,
          "another thread's error reached this thread");

  // A query's cudaErrorNotReady is an answer, not a failure: the runtime hands
  // it back without recording it.
  cudaStream_t stream = nullptr;
  require(cudaStreamCreate(&stream) == cudaSuccess, "stream creation failed");
  spin<<<1, 1, 0, stream>>>(4000000);
  require(cudaStreamQuery(stream) == cudaErrorNotReady,
          "a busy stream did not report itself busy");
  require(cudaPeekAtLastError() == cudaSuccess,
          "a busy stream query recorded an error");
  cudaEvent_t event = nullptr;
  require(cudaEventCreate(&event) == cudaSuccess, "event creation failed");
  require(cudaEventRecord(event, stream) == cudaSuccess, "event record failed");
  require(cudaEventQuery(event) == cudaErrorNotReady,
          "a pending event did not report itself pending");
  require(cudaPeekAtLastError() == cudaSuccess,
          "a pending event query recorded an error");
  require(cudaStreamSynchronize(stream) == cudaSuccess,
          "the spin kernel failed");
  require(cudaGetLastError() == cudaSuccess,
          "the spin kernel recorded an error");

  // A launch is fire-and-forget, so the failure lands on the first call that
  // waits for the device, and stays readable there.
  store_out_of_range<<<1, 1>>>(values, 1LL << 36);
  const cudaError_t launch_error = cudaDeviceSynchronize();
  require(launch_error != cudaSuccess,
          "an out-of-range store did not fail the device synchronize");
  require(cudaGetLastError() == launch_error,
          "the asynchronous launch failure never reached cudaGetLastError");

  std::puts("runtime last error survives being answered on the client");
}
