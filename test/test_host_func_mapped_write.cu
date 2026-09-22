// Integration test for a host callback that writes pinned mapped memory which a
// kernel queued behind it on the same stream then reads through the device
// pointer.
//
// The stream resumes as soon as the callback returns, so the callback's host
// writes must be on the server before the callback is answered. nvCOMP's
// batched LZ4 decompression fills its pointer arrays this way, and a kernel
// that reads the previous contents dereferences stale pointers.
#include <chrono>
#include <cuda_runtime.h>
#include <stdio.h>
#include <thread>

static const int kCount = 1 << 16;
static const int kRounds = 8;

struct fill_request {
  int *host = nullptr;
  int value = 0;
};

__global__ void copy_values(const int *src, int *dst, int count) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < count) {
    dst[idx] = src[idx];
  }
}

static void fill(fill_request *request) {
  for (int i = 0; i < kCount; ++i) {
    request->host[i] = request->value;
  }
}

static void CUDART_CB delay_host_fn(void *) {
  std::this_thread::sleep_for(std::chrono::milliseconds(50));
}

static void CUDART_CB fill_host_fn(void *opaque) {
  fill(static_cast<fill_request *>(opaque));
}

static void CUDART_CB fill_stream_callback(cudaStream_t, cudaError_t,
                                           void *opaque) {
  fill(static_cast<fill_request *>(opaque));
}

static int fatal(cudaError_t err, const char *what) {
  if (err == cudaSuccess) {
    return 0;
  }
  printf("RESULT: ERROR %s %s\n", what, cudaGetErrorString(err));
  return 1;
}

int main() {
  int *host = nullptr;
  int *mapped = nullptr;
  int *device = nullptr;
  cudaStream_t stream = nullptr;
  static int readback[kCount];
  if (fatal(cudaHostAlloc((void **)&host, kCount * sizeof(int),
                          cudaHostAllocMapped),
            "cudaHostAlloc") ||
      fatal(cudaHostGetDevicePointer((void **)&mapped, host, 0),
            "cudaHostGetDevicePointer") ||
      fatal(cudaMalloc((void **)&device, kCount * sizeof(int)),
            "cudaMalloc") ||
      fatal(cudaStreamCreate(&stream), "cudaStreamCreate")) {
    return 2;
  }

  fill_request request{host, 0};
  for (int round = 1; round <= kRounds; ++round) {
    request.value = round;
    // The delay keeps the stream busy until every call below has been issued,
    // so no later call of this thread can carry the callback's writes over.
    if (fatal(cudaLaunchHostFunc(stream, delay_host_fn, nullptr),
              "delay host func")) {
      return 2;
    }
    cudaError_t queued =
        round % 2 == 0
            ? cudaStreamAddCallback(stream, fill_stream_callback, &request, 0)
            : cudaLaunchHostFunc(stream, fill_host_fn, &request);
    if (fatal(queued, "fill callback")) {
      return 2;
    }
    copy_values<<<kCount / 256, 256, 0, stream>>>(mapped, device, kCount);
    if (fatal(cudaMemcpyAsync(readback, device, sizeof(readback),
                              cudaMemcpyDeviceToHost, stream),
              "cudaMemcpyAsync") ||
        fatal(cudaStreamSynchronize(stream), "cudaStreamSynchronize")) {
      return 2;
    }
    for (int i = 0; i < kCount; ++i) {
      if (readback[i] != round) {
        printf("RESULT: FAIL (round %d: kernel read %d at %d, callback wrote "
               "%d)\n",
               round, readback[i], i, round);
        return 1;
      }
    }
  }

  cudaStreamDestroy(stream);
  cudaFree(device);
  cudaFreeHost(host);
  printf("RESULT: PASS (kernels saw every callback's mapped writes)\n");
  return 0;
}
