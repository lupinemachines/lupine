// A pageable async DtoH copy blocks until its bytes land unless its own stream
// is capturing. Another thread holding a capture open on an unrelated stream
// must not turn this thread's copy into a deferred one: the caller reads the
// buffer as soon as the call returns, as it would natively.
#include <cuda_runtime.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <thread>
#include <vector>

namespace {

constexpr size_t kElements = 4096;
constexpr size_t kBytes = kElements * sizeof(int);
constexpr int kUntouched = -1;

bool check(cudaError_t result, const char *operation) {
  if (result == cudaSuccess) {
    return true;
  }
  std::fprintf(stderr, "FAIL: %s: %s\n", operation, cudaGetErrorString(result));
  return false;
}

// Spins first so a copy that returns early reads the buffer before the value
// is written.
__global__ void slow_fill(int *out, int value, size_t elements,
                          long long spin_cycles) {
  long long start = clock64();
  while (clock64() - start < spin_cycles) {
  }
  size_t index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < elements) {
    out[index] = value;
  }
}

bool all_equal(const std::vector<int> &host, int value) {
  return std::all_of(host.begin(), host.end(),
                     [value](int seen) { return seen == value; });
}

} // namespace

int main() {
  int *captured_device = nullptr;
  int *device = nullptr;
  cudaStream_t capture_stream = nullptr;
  cudaStream_t copy_stream = nullptr;
  if (!check(cudaMalloc(reinterpret_cast<void **>(&captured_device), kBytes),
             "cudaMalloc captured") ||
      !check(cudaMalloc(reinterpret_cast<void **>(&device), kBytes),
             "cudaMalloc") ||
      !check(cudaStreamCreateWithFlags(&capture_stream, cudaStreamNonBlocking),
             "cudaStreamCreate capture") ||
      !check(cudaStreamCreateWithFlags(&copy_stream, cudaStreamNonBlocking),
             "cudaStreamCreate copy")) {
    return 1;
  }
  const unsigned int blocks = (kElements + 255) / 256;
  std::vector<int> captured_host(kElements, kUntouched);

  std::atomic<bool> capturing{false};
  std::atomic<bool> copied{false};
  bool capture_ok = true;
  cudaGraph_t graph = nullptr;
  std::thread capturer([&] {
    capture_ok =
        check(cudaStreamBeginCapture(capture_stream,
                                     cudaStreamCaptureModeThreadLocal),
              "cudaStreamBeginCapture");
    capturing = true;
    if (capture_ok) {
      slow_fill<<<blocks, 256, 0, capture_stream>>>(captured_device, 5,
                                                    kElements, 0);
      // Inside the capture the pageable copy only becomes a graph node.
      capture_ok = check(cudaGetLastError(), "captured launch") &&
                   check(cudaMemcpyAsync(captured_host.data(), captured_device,
                                         kBytes, cudaMemcpyDeviceToHost,
                                         capture_stream),
                         "captured DtoH");
    }
    auto deadline =
        std::chrono::steady_clock::now() + std::chrono::seconds(2);
    while (!copied && std::chrono::steady_clock::now() < deadline) {
      std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    capture_ok = check(cudaStreamEndCapture(capture_stream, &graph),
                       "cudaStreamEndCapture") &&
                 capture_ok;
  });

  while (!capturing) {
    std::this_thread::yield();
  }
  std::vector<int> host(kElements, kUntouched);
  slow_fill<<<blocks, 256, 0, copy_stream>>>(device, 7, kElements,
                                             500000000LL);
  bool ok = check(cudaGetLastError(), "launch slow_fill") &&
            check(cudaMemcpyAsync(host.data(), device, kBytes,
                                  cudaMemcpyDeviceToHost, copy_stream),
                  "pageable DtoH beside a capture");
  bool landed = ok && all_equal(host, 7);
  copied = true;
  capturer.join();
  if (ok && !landed) {
    std::fprintf(stderr, "FAIL: pageable DtoH on a non-capturing stream "
                         "returned before its bytes landed while another "
                         "thread was capturing\n");
  }
  ok = ok && landed && capture_ok;

  if (ok) {
    cudaGraphExec_t exec = nullptr;
    ok = check(cudaGraphInstantiate(&exec, graph, nullptr, nullptr, 0),
               "cudaGraphInstantiate") &&
         check(cudaGraphLaunch(exec, capture_stream), "cudaGraphLaunch") &&
         check(cudaStreamSynchronize(capture_stream), "cudaStreamSynchronize");
    if (ok && !all_equal(captured_host, 5)) {
      std::fprintf(stderr, "FAIL: captured pageable DtoH did not deliver on "
                           "replay\n");
      ok = false;
    }
    cudaGraphExecDestroy(exec);
  }
  if (graph != nullptr) {
    cudaGraphDestroy(graph);
  }
  cudaStreamDestroy(copy_stream);
  cudaStreamDestroy(capture_stream);
  cudaFree(device);
  cudaFree(captured_device);
  if (!ok) {
    return 1;
  }
  std::printf("PASS: pageable async DtoH blocks beside another thread's "
              "capture\n");
  return 0;
}
