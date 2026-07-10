#include <cuda.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

static void check_cuda(cudaError_t result, const char *call, int line) {
  if (result == cudaSuccess) {
    return;
  }
  std::fprintf(stderr, "%s failed at line %d: %s (%d)\n", call, line,
               cudaGetErrorString(result), static_cast<int>(result));
  std::exit(EXIT_FAILURE);
}

static void check_driver(CUresult result, const char *call, int line) {
  if (result == CUDA_SUCCESS) {
    return;
  }
  const char *message = nullptr;
  cuGetErrorString(result, &message);
  std::fprintf(stderr, "%s failed at line %d: %s (%d)\n", call, line,
               message == nullptr ? "unknown" : message,
               static_cast<int>(result));
  std::exit(EXIT_FAILURE);
}

#define CHECK_CUDA(call) check_cuda((call), #call, __LINE__)
#define CHECK_DRV(call) check_driver((call), #call, __LINE__)

__global__ void delayed_fill(unsigned char *dst, size_t bytes,
                             unsigned long long cycles, unsigned char value) {
  if (threadIdx.x == 0) {
    unsigned long long start = clock64();
    while (clock64() - start < cycles) {
    }
  }
  __syncthreads();
  for (size_t i = threadIdx.x; i < bytes; i += blockDim.x) {
    dst[i] = value;
  }
}

static CUdeviceptr as_device_ptr(void *ptr) {
  return static_cast<CUdeviceptr>(reinterpret_cast<uintptr_t>(ptr));
}

static unsigned long long delay_cycles(int milliseconds) {
  int clock_khz = 0;
  CHECK_CUDA(cudaDeviceGetAttribute(&clock_khz, cudaDevAttrClockRate, 0));
  return static_cast<unsigned long long>(clock_khz) * milliseconds;
}

static void require_bytes(const std::vector<unsigned char> &actual,
                          const std::vector<unsigned char> &expected,
                          const char *message) {
  if (actual == expected) {
    return;
  }
  std::fprintf(stderr, "%s\n", message);
  std::exit(EXIT_FAILURE);
}

int main() {
  constexpr size_t kSlotBytes = 8 * 1024 * 1024;
  constexpr size_t kLargeBytes = 3 * kSlotBytes + 123;

  CHECK_CUDA(cudaSetDevice(0));
  CHECK_CUDA(cudaFree(nullptr));

  unsigned char *device = nullptr;
  CHECK_CUDA(cudaMalloc(&device, kLargeBytes));
  CUdeviceptr device_ptr = as_device_ptr(device);

  std::vector<unsigned char> source(kLargeBytes);
  std::vector<unsigned char> destination(kLargeBytes);
  uint32_t random = 0x12345678u;
  for (unsigned char &byte : source) {
    random = random * 1664525u + 1013904223u;
    byte = static_cast<unsigned char>(random >> 24);
  }

  // Cover the unframed/framed threshold and both sides of the retained-slot
  // boundary, including enough data to reuse a slot after its event fires.
  const size_t sizes[] = {0,
                          1,
                          64 * 1024 - 1,
                          64 * 1024,
                          kSlotBytes - 1,
                          kSlotBytes,
                          kSlotBytes + 1,
                          2 * kSlotBytes,
                          kLargeBytes};
  for (size_t bytes : sizes) {
    const void *src = bytes == 0 ? nullptr : source.data();
    CHECK_DRV(cuMemcpyHtoD(device_ptr, src, bytes));
    if (bytes == 0) {
      continue;
    }
    CHECK_DRV(cuMemcpyDtoH(destination.data(), device_ptr, bytes));
    if (std::memcmp(destination.data(), source.data(), bytes) != 0) {
      std::fprintf(stderr, "HtoD boundary payload mismatch at %zu bytes\n",
                   bytes);
      return 1;
    }
  }

  // Synchronous CUDA copies use the legacy stream. Work already queued on a
  // blocking stream must finish before the HtoD copy, otherwise this delayed
  // fill would overwrite the copied bytes.
  cudaStream_t blocking = nullptr;
  CHECK_CUDA(cudaStreamCreateWithFlags(&blocking, cudaStreamDefault));
  std::fill(source.begin(), source.end(), 0xa5);
  delayed_fill<<<1, 256, 0, blocking>>>(device, kLargeBytes, delay_cycles(500),
                                        0x3c);
  CHECK_CUDA(cudaGetLastError());
  CHECK_DRV(cuMemcpyHtoD(device_ptr, source.data(), kLargeBytes));
  CHECK_CUDA(cudaStreamSynchronize(blocking));
  CHECK_DRV(cuMemcpyDtoH(destination.data(), device_ptr, kLargeBytes));
  require_bytes(destination, source,
                "legacy-stream HtoD did not wait for a blocking stream");
  CHECK_CUDA(cudaStreamDestroy(blocking));

  // Non-blocking streams are explicitly excluded from legacy-stream implicit
  // synchronization. A broad context synchronization in the fast path would
  // incorrectly make this event ready before the synchronous HtoD returns.
  cudaStream_t nonblocking = nullptr;
  CHECK_CUDA(cudaStreamCreateWithFlags(&nonblocking, cudaStreamNonBlocking));
  unsigned char *marker = nullptr;
  CHECK_CUDA(cudaMalloc(&marker, 1));
  delayed_fill<<<1, 1, 0, nonblocking>>>(marker, 1, delay_cycles(1500), 1);
  CHECK_CUDA(cudaGetLastError());
  CUevent delayed = nullptr;
  CHECK_DRV(cuEventCreate(&delayed, CU_EVENT_DISABLE_TIMING));
  CHECK_DRV(cuEventRecord(delayed, reinterpret_cast<CUstream>(nonblocking)));

  CHECK_DRV(cuMemcpyHtoD(device_ptr, source.data(), kLargeBytes));
  CUresult query = cuEventQuery(delayed);
  if (query != CUDA_ERROR_NOT_READY) {
    std::fprintf(
        stderr, "sync HtoD unexpectedly waited for a non-blocking stream: %d\n",
        static_cast<int>(query));
    return 1;
  }
  CHECK_DRV(cuEventSynchronize(delayed));
  CHECK_DRV(cuEventDestroy(delayed));
  CHECK_CUDA(cudaFree(marker));
  CHECK_CUDA(cudaStreamDestroy(nonblocking));

  // A CUDA failure after receiving the first pipeline slot must drain the rest
  // of the framed request so the next RPC remains aligned. Keep this after the
  // ordering checks so an implementation that reports an asynchronous device
  // fault cannot make those checks fall back to the serial path.
  CUresult invalid = cuMemcpyHtoD(0, source.data(), kLargeBytes);
  if (invalid == CUDA_SUCCESS) {
    std::fprintf(stderr, "invalid large HtoD unexpectedly succeeded\n");
    return 1;
  }
  CHECK_DRV(cuMemcpyHtoD(device_ptr, source.data(), kSlotBytes + 1));
  CHECK_DRV(cuMemcpyDtoH(destination.data(), device_ptr, kSlotBytes + 1));
  if (std::memcmp(destination.data(), source.data(), kSlotBytes + 1) != 0) {
    std::fprintf(stderr, "RPC payload was misaligned after failed HtoD\n");
    return 1;
  }

  CHECK_CUDA(cudaFree(device));
  std::printf(
      "PASS: synchronous HtoD pipeline preserves data and stream ordering\n");
  return 0;
}
