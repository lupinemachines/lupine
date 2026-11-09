// An async HtoD from page-locked memory returns before its stream reaches the
// copy, so another thread may issue the work that lets the stream get there.
// Here a kernel queued ahead of the copy spins until a second thread sets a
// flag from another stream. Natively the flag always lands; Lupine must not
// make the second thread's call wait for the first thread's copy, whether the
// source is a cuMemHostAlloc buffer (mirrored on the server), a registered one
// (pulled from the client when the stream reaches the copy) or small pageable
// memory (sent in the request). A pageable source is free to reuse once the
// call returns, so that case also overwrites it straight away.
#include <cuda_runtime.h>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <thread>

namespace {

#define CHECK(call)                                                            \
  do {                                                                         \
    cudaError_t check_result_ = (call);                                        \
    if (check_result_ != cudaSuccess) {                                        \
      std::fprintf(stderr, "FAIL: %s: %s\n", #call,                            \
                   cudaGetErrorString(check_result_));                         \
      return 1;                                                                \
    }                                                                          \
  } while (0)

// Spins until *flag is set or about two seconds of GPU clock pass, so a
// deadlock reports instead of hanging. *timed_out records which one ended it.
__global__ void wait_for_flag(volatile unsigned int *flag,
                              unsigned int *timed_out, long long budget) {
  long long start = clock64();
  while (*flag == 0) {
    if (clock64() - start > budget) {
      *timed_out = 1;
      return;
    }
  }
}

int run(const char *label, unsigned int *source, bool pageable,
        long long budget) {
  cudaStream_t waiting = nullptr;
  cudaStream_t signalling = nullptr;
  CHECK(cudaStreamCreateWithFlags(&waiting, cudaStreamNonBlocking));
  CHECK(cudaStreamCreateWithFlags(&signalling, cudaStreamNonBlocking));
  unsigned int *flag = nullptr;
  unsigned int *timed_out = nullptr;
  unsigned int *target = nullptr;
  CHECK(cudaMalloc(&flag, sizeof(*flag)));
  CHECK(cudaMalloc(&timed_out, sizeof(*timed_out)));
  CHECK(cudaMalloc(&target, sizeof(*target)));
  CHECK(cudaMemset(flag, 0, sizeof(*flag)));
  CHECK(cudaMemset(timed_out, 0, sizeof(*timed_out)));

  // Warm the copy path first: its first use in a context may load a module,
  // which waits for running kernels on its own.
  *source = 0;
  CHECK(cudaMemcpyAsync(target, source, sizeof(*source),
                        cudaMemcpyHostToDevice, waiting));
  CHECK(cudaStreamSynchronize(waiting));
  *source = 0x1234abcd;

  cudaError_t copy_result = cudaSuccess;
  std::thread producer([&] {
    wait_for_flag<<<1, 1, 0, waiting>>>(flag, timed_out, budget);
    copy_result = cudaMemcpyAsync(target, source, sizeof(*source),
                                  cudaMemcpyHostToDevice, waiting);
    if (pageable) {
      *source = 0xdeadbeef;
    }
  });
  // Give the producer time to enqueue both without depending on its copy call
  // returning: where that call is held until the stream reaches the copy, the
  // flag still has to get through.
  std::this_thread::sleep_for(std::chrono::milliseconds(200));
  cudaError_t signal_result =
      cudaMemsetAsync(flag, 1, sizeof(*flag), signalling);
  producer.join();
  CHECK(signal_result);
  CHECK(copy_result);
  CHECK(cudaStreamSynchronize(signalling));
  CHECK(cudaStreamSynchronize(waiting));

  unsigned int expired = 0;
  unsigned int copied = 0;
  CHECK(cudaMemcpy(&expired, timed_out, sizeof(expired),
                   cudaMemcpyDeviceToHost));
  CHECK(cudaMemcpy(&copied, target, sizeof(copied), cudaMemcpyDeviceToHost));
  if (expired != 0) {
    std::fprintf(stderr,
                 "FAIL: %s: the flag never reached the spinning kernel; the "
                 "signalling call waited behind the copy\n",
                 label);
    return 1;
  }
  if (copied != 0x1234abcd) {
    std::fprintf(stderr, "FAIL: %s: copy delivered 0x%08x\n", label, copied);
    return 1;
  }
  CHECK(cudaFree(flag));
  CHECK(cudaFree(timed_out));
  CHECK(cudaFree(target));
  CHECK(cudaStreamDestroy(waiting));
  CHECK(cudaStreamDestroy(signalling));
  std::printf("%s: ok\n", label);
  return 0;
}

} // namespace

int main() {
  CHECK(cudaSetDevice(0));
  int clock_khz = 0;
  CHECK(cudaDeviceGetAttribute(&clock_khz, cudaDevAttrClockRate, 0));
  long long budget = static_cast<long long>(clock_khz) * 2000; // ~2 s

  unsigned int *allocated = nullptr;
  CHECK(cudaMallocHost(&allocated, sizeof(*allocated)));
  if (run("cudaMallocHost source", allocated, false, budget) != 0) {
    return 1;
  }
  CHECK(cudaFreeHost(allocated));

  auto *registered = static_cast<unsigned int *>(std::aligned_alloc(4096, 4096));
  if (registered == nullptr) {
    return 1;
  }
  CHECK(cudaHostRegister(registered, 4096, cudaHostRegisterDefault));
  if (run("registered source", registered, false, budget) != 0) {
    return 1;
  }
  CHECK(cudaHostUnregister(registered));
  std::free(registered);

  static unsigned int pageable = 0;
  if (run("pageable source", &pageable, true, budget) != 0) {
    return 1;
  }
  return 0;
}
