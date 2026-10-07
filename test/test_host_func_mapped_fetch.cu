// Fetch invalidated pinned bytes inside a host callback while cudaFree waits
// for that callback. The fetch must not enter CUDA behind the free's lock.
#include <atomic>
#include <chrono>
#include <cuda_runtime.h>
#include <stdio.h>
#include <thread>

static const size_t kBytes = 1ull << 20;
static const unsigned char kDeviceValue = 0x5a;

// 0 = not run, 1 = pass, -1 = mismatch.
static std::atomic<int> g_result{0};
static std::atomic<bool> g_started{false};
static unsigned char *g_host = nullptr;

__global__ void write_bytes(unsigned char *dst, unsigned char value,
                            size_t count) {
  size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < count) {
    dst[idx] = value;
  }
}

static void CUDART_CB host_fn(void *) {
  g_started.store(true, std::memory_order_release);
  // Let cudaFree enter its implicit synchronization before the first fetch.
  std::this_thread::sleep_for(std::chrono::milliseconds(50));
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
  void *unrelated = nullptr;
  unsigned char *device = nullptr;
  cudaStream_t stream = nullptr;
  if (fatal(cudaMalloc(&unrelated, 1), "cudaMalloc") ||
      fatal(cudaHostAlloc((void **)&g_host, kBytes, cudaHostAllocMapped),
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
            "cudaLaunchHostFunc")) {
    return 2;
  }
  while (!g_started.load(std::memory_order_acquire)) {
    std::this_thread::yield();
  }
  // cuMemFree can hold the CUDA driver's write lock until the callback exits.
  // Fetching pinned bytes must not require another CUDA call behind that lock.
  if (fatal(cudaFree(unrelated), "cudaFree during host func") ||
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
