// Stream-captured host-to-host copies (cuMemcpyAsync, cudaMemcpyAsync with
// cudaMemcpyHostToHost or cudaMemcpyDefault, and 2D host-to-host copies) must
// record graph work rather than mutating the destination during capture, and
// must execute in graph order on every replay across pageable, pinned, and
// mapped host memory.
#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>

namespace {

__global__ void scale_add(unsigned int *buf, int n, unsigned int mul,
                          unsigned int addend) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    buf[idx] = buf[idx] * mul + addend;
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

  constexpr int kCount = 8;
  unsigned int pageable_src[kCount] = {};
  unsigned int pageable_dst[kCount] = {};
  unsigned int *pinned_mid = nullptr;
  unsigned int *mapped_buf = nullptr;
  unsigned int *mapped_dev = nullptr;
  cudaStream_t stream = nullptr;

  check(cudaMallocHost(&pinned_mid, kCount * sizeof(unsigned int)),
        "cudaMallocHost");
  check(cudaHostAlloc(&mapped_buf, kCount * sizeof(unsigned int),
                      cudaHostAllocMapped),
        "cudaHostAlloc(mapped)");
  check(cudaHostGetDevicePointer(&mapped_dev, mapped_buf, 0),
        "cudaHostGetDevicePointer");
  check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking),
        "cudaStreamCreateWithFlags");

  for (int i = 0; i < kCount; ++i) {
    pageable_src[i] = static_cast<unsigned int>(i + 1);
    pinned_mid[i] = 0xaaaa0000u;
    mapped_buf[i] = 0xbbbb0000u;
    pageable_dst[i] = 0xcccc0000u;
  }

  int failures = 0;

  // 1. Capture pipeline:
  //    pageable_src -> pinned_mid (HtoH)
  //    pinned_mid   -> mapped_buf (cudaMemcpyDefault HtoH)
  //    kernel updates mapped_buf in-place on GPU
  //    mapped_buf   -> pageable_dst (2D pitched HtoH)
  cudaGraph_t graph = nullptr;
  cudaGraphExec_t exec = nullptr;
  check(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal),
        "cudaStreamBeginCapture");
  check(cudaMemcpyAsync(pinned_mid, pageable_src, sizeof(pageable_src),
                        cudaMemcpyHostToHost, stream),
        "captured HtoH pageable->pinned");
  check(cudaMemcpyAsync(mapped_buf, pinned_mid, sizeof(pageable_src),
                        cudaMemcpyDefault, stream),
        "captured default HtoH pinned->mapped");
  scale_add<<<1, kCount, 0, stream>>>(mapped_dev, kCount, 2u, 50u);
  check(cudaGetLastError(), "captured scale_add");
  check(cudaMemcpy2DAsync(pageable_dst, 2 * sizeof(unsigned int), mapped_buf,
                          2 * sizeof(unsigned int), 2 * sizeof(unsigned int),
                          kCount / 2, cudaMemcpyHostToHost, stream),
        "captured 2D HtoH mapped->pageable");
  check(cudaStreamEndCapture(stream, &graph), "cudaStreamEndCapture");

  // Destination buffers must NOT have been mutated during capture.
  for (int i = 0; i < kCount; ++i) {
    if (pinned_mid[i] != 0xaaaa0000u || mapped_buf[i] != 0xbbbb0000u ||
        pageable_dst[i] != 0xcccc0000u) {
      std::fprintf(stderr,
                   "FAIL: destination mutated during capture at %d: "
                   "pinned=0x%x mapped=0x%x dst=0x%x\n",
                   i, pinned_mid[i], mapped_buf[i], pageable_dst[i]);
      ++failures;
      break;
    }
  }

  check(cudaGraphInstantiate(&exec, graph, nullptr, nullptr, 0),
        "cudaGraphInstantiate");

  for (int replay = 0; replay < 4; ++replay) {
    for (int i = 0; i < kCount; ++i) {
      pageable_src[i] = static_cast<unsigned int>((replay + 1) * 10 + i);
      pageable_dst[i] = 0u;
    }
    check(cudaGraphLaunch(exec, stream), "cudaGraphLaunch");
    check(cudaStreamSynchronize(stream), "cudaStreamSynchronize");

    for (int i = 0; i < kCount; ++i) {
      unsigned int want = pageable_src[i] * 2u + 50u;
      if (pinned_mid[i] != pageable_src[i] || pageable_dst[i] != want) {
        std::fprintf(stderr,
                     "FAIL: replay %d[%d]: pinned=%u(want %u) dst=%u(want %u)\n",
                     replay, i, pinned_mid[i], pageable_src[i], pageable_dst[i],
                     want);
        ++failures;
        break;
      }
    }
  }

  check(cudaGraphExecDestroy(exec), "cudaGraphExecDestroy");
  check(cudaGraphDestroy(graph), "cudaGraphDestroy");
  check(cudaStreamDestroy(stream), "cudaStreamDestroy");
  check(cudaFreeHost(mapped_buf), "cudaFreeHost(mapped)");
  check(cudaFreeHost(pinned_mid), "cudaFreeHost(pinned)");

  if (failures != 0) {
    std::fprintf(stderr, "FAIL: %d captured HtoH checks failed\n", failures);
    return EXIT_FAILURE;
  }
  std::printf("PASS: captured host-to-host copies run in graph order\n");
  return EXIT_SUCCESS;
}
