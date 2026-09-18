// Integration test for a mapped-host demand fetch that originates inside a
// cuLaunchHostFunc callback, i.e. on the client's RPC dispatch thread.
//
// A synchronization point invalidates the pinned mapped allocation instead of
// copying it back, so the first host touch has to fetch the affected chunks.
// When that touch happens in a host-func callback the fault handler runs on the
// dispatch thread, whose server lane is not the one the application bound to a
// context; a fetch that assumes the caller's lane already has a current context
// comes back CUDA_ERROR_INVALID_CONTEXT and the handler turns a recoverable
// fault into SIGSEGV.
#include <atomic>
#include <cuda_runtime.h>
#include <stdio.h>

static const size_t kBytes = 1ull << 20;
static const unsigned char kDeviceValue = 0x5a;

// 0 = not run, 1 = pass, -1 = mismatch.
static std::atomic<int> g_result{0};
static unsigned char *g_host = nullptr;

__global__ void write_bytes(unsigned char *dst, unsigned char value,
                            size_t count) {
  size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < count) {
    dst[idx] = value;
  }
}

static void CUDART_CB host_fn(void *) {
  int ok = 1;
  for (size_t i = 0; i < kBytes; ++i) {
    if (g_host[i] != kDeviceValue) {
      ok = -1;
      break;
    }
  }
  g_result.store(ok, std::memory_order_release);
}

static int fatal(cudaError_t err, const char *what) {
  if (err == cudaSuccess) {
    return 0;
  }
  printf("RESULT: ERROR %s %s\n", what, cudaGetErrorString(err));
  return 1;
}

int main() {
  unsigned char *device = nullptr;
  cudaStream_t stream = nullptr;
  if (fatal(cudaHostAlloc((void **)&g_host, kBytes, cudaHostAllocMapped),
            "cudaHostAlloc") ||
      fatal(cudaHostGetDevicePointer((void **)&device, g_host, 0),
            "cudaHostGetDevicePointer") ||
      fatal(cudaStreamCreate(&stream), "cudaStreamCreate")) {
    return 2;
  }

  write_bytes<<<(unsigned)(kBytes / 256), 256, 0, stream>>>(
      device, kDeviceValue, kBytes);
  // Invalidates the client's view: the pages go PROT_NONE and stay that way
  // until something touches them.
  if (fatal(cudaStreamSynchronize(stream), "sync after kernel")) {
    return 2;
  }

  // The first touch of the invalidated range now happens on the dispatch
  // thread, inside the callback.
  if (fatal(cudaLaunchHostFunc(stream, host_fn, nullptr),
            "cudaLaunchHostFunc") ||
      fatal(cudaStreamSynchronize(stream), "sync after host func")) {
    return 2;
  }

  int result = g_result.load(std::memory_order_acquire);
  cudaStreamDestroy(stream);
  cudaFreeHost(g_host);

  if (result == 1) {
    printf("RESULT: PASS (host func fetched the invalidated mapped range)\n");
    return 0;
  }
  if (result == -1) {
    printf("RESULT: FAIL (host func read stale mapped bytes)\n");
    return 1;
  }
  printf("RESULT: FAIL (host func never ran)\n");
  return 1;
}
