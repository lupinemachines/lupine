// A stream-captured DtoH copy into pinned or mapped host memory must target
// the server's backing host buffer and deliver to the client's writable alias
// rather than scratch staging. Otherwise a subsequent kernel in the same graph
// reading or updating mapped memory never sees the copied bytes, and delivering
// scratch to the application pointer marks the client page dirty and overwrites
// the kernel's result on synchronization and subsequent replays.
#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>

namespace {

constexpr int kCount = 16;

__global__ void scale_and_bias(unsigned int *buf, int n, unsigned int mul,
                               unsigned int bias) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    buf[idx] = buf[idx] * mul + bias + static_cast<unsigned int>(idx);
  }
}

void check(cudaError_t err, const char *what) {
  if (err == cudaSuccess) {
    return;
  }
  std::fprintf(stderr, "FAIL: %s: %s\n", what, cudaGetErrorString(err));
  std::exit(EXIT_FAILURE);
}

} // namespace

int main() {
  int device_count = 0;
  check(cudaGetDeviceCount(&device_count), "cudaGetDeviceCount");
  if (device_count == 0) {
    std::printf("SKIP: no CUDA devices found\n");
    return EXIT_SUCCESS;
  }

  check(cudaSetDeviceFlags(cudaDeviceMapHost), "cudaSetDeviceFlags");

  unsigned int *dev_src = nullptr;
  unsigned int *mapped_host = nullptr;
  unsigned int *mapped_dev = nullptr;
  unsigned int *pinned_host = nullptr;
  cudaStream_t stream = nullptr;

  check(cudaMalloc(&dev_src, kCount * sizeof(unsigned int)), "cudaMalloc");
  check(cudaHostAlloc(&mapped_host, kCount * sizeof(unsigned int),
                      cudaHostAllocMapped),
        "cudaHostAlloc(mapped)");
  check(cudaHostGetDevicePointer(&mapped_dev, mapped_host, 0),
        "cudaHostGetDevicePointer");
  check(cudaMallocHost(&pinned_host, kCount * sizeof(unsigned int)),
        "cudaMallocHost");
  check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking),
        "cudaStreamCreateWithFlags");

  for (int i = 0; i < kCount; ++i) {
    mapped_host[i] = 0xdeadbeefu;
    pinned_host[i] = 0xbaadf00du;
  }

  int failures = 0;

  // 1. Captured DtoH into mapped host memory followed by a kernel updating the
  // mapped buffer in-place inside the same graph, replayed across changing
  // device inputs and host overwrites.
  cudaGraph_t mapped_graph = nullptr;
  cudaGraphExec_t mapped_exec = nullptr;
  check(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal),
        "cudaStreamBeginCapture(mapped)");
  check(cudaMemcpyAsync(mapped_host, dev_src, kCount * sizeof(unsigned int),
                        cudaMemcpyDeviceToHost, stream),
        "captured DtoH into mapped");
  scale_and_bias<<<1, kCount, 0, stream>>>(mapped_dev, kCount, 3u, 100u);
  check(cudaGetLastError(), "captured scale_and_bias");
  check(cudaStreamEndCapture(stream, &mapped_graph),
        "cudaStreamEndCapture(mapped)");
  check(cudaGraphInstantiate(&mapped_exec, mapped_graph, nullptr, nullptr, 0),
        "cudaGraphInstantiate(mapped)");

  for (int replay = 0; replay < 4; ++replay) {
    unsigned int host_init[kCount];
    for (int i = 0; i < kCount; ++i) {
      host_init[i] = static_cast<unsigned int>((replay + 1) * 10 + i);
      mapped_host[i] = 999999u;
    }
    check(cudaMemcpyAsync(dev_src, host_init, sizeof(host_init),
                          cudaMemcpyHostToDevice, stream),
          "HtoD init");
    check(cudaGraphLaunch(mapped_exec, stream), "cudaGraphLaunch(mapped)");
    check(cudaStreamSynchronize(stream), "cudaStreamSynchronize(mapped)");

    for (int i = 0; i < kCount; ++i) {
      unsigned int want = host_init[i] * 3u + 100u + static_cast<unsigned int>(i);
      unsigned int got = mapped_host[i];
      if (got != want) {
        std::fprintf(stderr,
                     "FAIL: mapped replay %d[%d]: got %u, want %u\n", replay, i,
                     got, want);
        ++failures;
        break;
      }
    }
  }

  // 2. Captured DtoH into pinned memory replayed multiple times, followed by
  // freeing pinned memory BEFORE destroying the graph/exec (borrowed storage
  // must not be double-freed on graph teardown).
  cudaGraph_t pinned_graph = nullptr;
  cudaGraphExec_t pinned_exec = nullptr;
  check(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal),
        "cudaStreamBeginCapture(pinned)");
  check(cudaMemcpyAsync(pinned_host, dev_src, kCount * sizeof(unsigned int),
                        cudaMemcpyDeviceToHost, stream),
        "captured DtoH into pinned");
  check(cudaStreamEndCapture(stream, &pinned_graph),
        "cudaStreamEndCapture(pinned)");
  check(cudaGraphInstantiate(&pinned_exec, pinned_graph, nullptr, nullptr, 0),
        "cudaGraphInstantiate(pinned)");

  for (int replay = 0; replay < 3; ++replay) {
    unsigned int host_init[kCount];
    for (int i = 0; i < kCount; ++i) {
      host_init[i] = static_cast<unsigned int>((replay + 5) * 100 + i);
    }
    check(cudaMemcpyAsync(dev_src, host_init, sizeof(host_init),
                          cudaMemcpyHostToDevice, stream),
          "HtoD init pinned");
    check(cudaGraphLaunch(pinned_exec, stream), "cudaGraphLaunch(pinned)");
    check(cudaStreamSynchronize(stream), "cudaStreamSynchronize(pinned)");
    for (int i = 0; i < kCount; ++i) {
      if (pinned_host[i] != host_init[i]) {
        std::fprintf(stderr,
                     "FAIL: pinned replay %d[%d]: got %u, want %u\n", replay, i,
                     pinned_host[i], host_init[i]);
        ++failures;
        break;
      }
    }
  }

  // Free pinned_host BEFORE destroying pinned_exec and pinned_graph.
  check(cudaFreeHost(pinned_host), "cudaFreeHost(pinned_host)");
  check(cudaGraphExecDestroy(pinned_exec), "cudaGraphExecDestroy(pinned)");
  check(cudaGraphDestroy(pinned_graph), "cudaGraphDestroy(pinned)");

  check(cudaGraphExecDestroy(mapped_exec), "cudaGraphExecDestroy(mapped)");
  check(cudaGraphDestroy(mapped_graph), "cudaGraphDestroy(mapped)");
  check(cudaFreeHost(mapped_host), "cudaFreeHost(mapped_host)");
  check(cudaFree(dev_src), "cudaFree(dev_src)");
  check(cudaStreamDestroy(stream), "cudaStreamDestroy");

  if (failures != 0) {
    std::fprintf(stderr, "FAIL: %d captured pinned/mapped DtoH checks failed\n",
                 failures);
    return EXIT_FAILURE;
  }
  std::printf("PASS: captured pinned/mapped DtoH copies preserve graph ordering\n");
  return EXIT_SUCCESS;
}
