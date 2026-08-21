// Unified-memory integration coverage for the runtime API. This deliberately
// crosses several host/device ownership boundaries and CUDA streams so a test
// cannot pass solely because the first host-to-device upload worked.
#include <cuda_runtime.h>

#include <cstdio>

namespace {

constexpr size_t kElements = 1u << 20;
constexpr size_t kBytes = kElements * sizeof(int);

__host__ __device__ int initial_value(size_t index) {
  return static_cast<int>((index * 17u + 11u) & 0x3fffffff);
}

__global__ void transform(int *values, size_t count, int addend) {
  size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index < count) {
    values[index] = values[index] * 2 + addend;
  }
}

int check(cudaError_t status, const char *operation) {
  if (status == cudaSuccess) {
    return 0;
  }
  std::fprintf(stderr, "%s failed: %d (%s)\n", operation,
               static_cast<int>(status), cudaGetErrorString(status));
  return 1;
}

int expect(bool condition, const char *message) {
  if (condition) {
    return 0;
  }
  std::fprintf(stderr, "FAIL: %s\n", message);
  return 1;
}

cudaError_t prefetch(const void *pointer, size_t bytes, int device,
                     cudaStream_t stream) {
#if CUDART_VERSION >= 12020
  cudaMemLocation location = {};
  location.type = device == cudaCpuDeviceId ? cudaMemLocationTypeHost
                                            : cudaMemLocationTypeDevice;
  location.id = device == cudaCpuDeviceId ? 0 : device;
  return cudaMemPrefetchAsync(pointer, bytes, location, 0, stream);
#else
  return cudaMemPrefetchAsync(pointer, bytes, device, stream);
#endif
}

cudaError_t advise(const void *pointer, size_t bytes,
                   cudaMemoryAdvise advice_kind, int device) {
#if CUDART_VERSION >= 12020
  cudaMemLocation location = {};
  location.type = device == cudaCpuDeviceId ? cudaMemLocationTypeHost
                                            : cudaMemLocationTypeDevice;
  location.id = device == cudaCpuDeviceId ? 0 : device;
  return cudaMemAdvise(pointer, bytes, advice_kind, location);
#else
  return cudaMemAdvise(pointer, bytes, advice_kind, device);
#endif
}

} // namespace

int main() {
  int device_count = 0;
  if (check(cudaGetDeviceCount(&device_count), "cudaGetDeviceCount") != 0 ||
      expect(device_count > 0, "no CUDA devices visible") != 0) {
    return 1;
  }

  int device = 0;
  if (check(cudaGetDevice(&device), "cudaGetDevice") != 0) {
    return 2;
  }

  int *managed = nullptr;
  if (check(cudaMallocManaged(&managed, kBytes, cudaMemAttachGlobal),
            "cudaMallocManaged") != 0) {
    return 3;
  }

  cudaPointerAttributes attributes = {};
  if (check(cudaPointerGetAttributes(&attributes, managed),
            "cudaPointerGetAttributes") != 0 ||
      expect(attributes.type == cudaMemoryTypeManaged,
             "managed allocation has the wrong memory type") != 0 ||
      expect(attributes.hostPointer == managed,
             "managed allocation has the wrong host alias") != 0 ||
      expect(attributes.devicePointer != nullptr,
             "managed allocation has no device alias") != 0) {
    cudaFree(managed);
    return 4;
  }
  if (check(
          advise(managed, kBytes, cudaMemAdviseSetReadMostly, cudaCpuDeviceId),
          "cudaMemAdvise(SetReadMostly)") != 0 ||
      check(advise(managed, kBytes, cudaMemAdviseSetPreferredLocation, device),
            "cudaMemAdvise(SetPreferredLocation)") != 0 ||
      check(advise(managed, kBytes, cudaMemAdviseUnsetPreferredLocation,
                   cudaCpuDeviceId),
            "cudaMemAdvise(UnsetPreferredLocation)") != 0 ||
      check(advise(managed, kBytes, cudaMemAdviseUnsetReadMostly,
                   cudaCpuDeviceId),
            "cudaMemAdvise(UnsetReadMostly)") != 0) {
    cudaFree(managed);
    return 5;
  }

  for (size_t i = 0; i < kElements; ++i) {
    managed[i] = initial_value(i);
  }

  cudaStream_t stream = nullptr;
  if (check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking),
            "cudaStreamCreateWithFlags") != 0 ||
      check(cudaStreamAttachMemAsync(stream, managed, 0, cudaMemAttachSingle),
            "cudaStreamAttachMemAsync") != 0 ||
      check(prefetch(managed + 37, kBytes - 79 * sizeof(int), device, stream),
            "cudaMemPrefetchAsync(device)") != 0) {
    cudaFree(managed);
    return 6;
  }

  constexpr int kFirstAddend = 5;
  transform<<<(kElements + 255) / 256, 256, 0, stream>>>(managed, kElements,
                                                         kFirstAddend);
  if (check(cudaGetLastError(), "first transform launch") != 0 ||
      check(prefetch(managed, kBytes, cudaCpuDeviceId, stream),
            "cudaMemPrefetchAsync(host)") != 0 ||
      check(cudaStreamSynchronize(stream), "first cudaStreamSynchronize") !=
          0) {
    cudaStreamDestroy(stream);
    cudaFree(managed);
    return 7;
  }

  for (size_t i = 0; i < kElements; ++i) {
    int expected = initial_value(i) * 2 + kFirstAddend;
    if (managed[i] != expected) {
      std::fprintf(stderr,
                   "first device-to-host mismatch at %zu: got %d expected %d\n",
                   i, managed[i], expected);
      cudaStreamDestroy(stream);
      cudaFree(managed);
      return 8;
    }
  }

  // Dirty pages after the mirror has been demand-fetched from the server.
  // The next kernel must observe both dense and sparse host writes.
  constexpr size_t kSpots[] = {0, 1, (64u << 10) / sizeof(int),
                               (2u << 20) / sizeof(int), kElements - 1};
  for (size_t spot : kSpots) {
    managed[spot] = -static_cast<int>(spot) - 100;
  }

  constexpr int kSecondAddend = 9;
  transform<<<(kElements + 255) / 256, 256>>>(managed, kElements,
                                              kSecondAddend);
  if (check(cudaGetLastError(), "second transform launch") != 0 ||
      check(cudaDeviceSynchronize(), "second cudaDeviceSynchronize") != 0) {
    cudaStreamDestroy(stream);
    cudaFree(managed);
    return 9;
  }

  for (size_t i = 0; i < kElements; ++i) {
    int before = initial_value(i) * 2 + kFirstAddend;
    for (size_t spot : kSpots) {
      if (i == spot) {
        before = -static_cast<int>(spot) - 100;
      }
    }
    int expected = before * 2 + kSecondAddend;
    if (managed[i] != expected) {
      std::fprintf(
          stderr, "second device-to-host mismatch at %zu: got %d expected %d\n",
          i, managed[i], expected);
      cudaStreamDestroy(stream);
      cudaFree(managed);
      return 10;
    }
  }

  // Exercise managed pointers as both source and destination of inferred
  // cudaMemcpy operations, plus a device-side memset of an interior range.
  constexpr size_t kCopyElements = 4096;
  constexpr size_t kSource = 12345;
  constexpr size_t kDestination = 700000;
  int expected_copy[kCopyElements];
  for (size_t i = 0; i < kCopyElements; ++i) {
    expected_copy[i] = managed[kSource + i];
  }
  int *device_scratch = nullptr;
  if (check(cudaMalloc(&device_scratch, kCopyElements * sizeof(int)),
            "cudaMalloc(device_scratch)") != 0 ||
      check(cudaMemcpyAsync(device_scratch, managed + kSource,
                            kCopyElements * sizeof(int), cudaMemcpyDefault,
                            stream),
            "cudaMemcpyAsync(managed to device)") != 0 ||
      check(cudaMemcpyAsync(managed + kDestination, device_scratch,
                            kCopyElements * sizeof(int), cudaMemcpyDefault,
                            stream),
            "cudaMemcpyAsync(device to managed)") != 0 ||
      check(cudaMemsetAsync(managed + kSource, 0, kCopyElements * sizeof(int),
                            stream),
            "cudaMemsetAsync(managed)") != 0 ||
      check(cudaStreamSynchronize(stream), "copy cudaStreamSynchronize") != 0) {
    cudaFree(device_scratch);
    cudaStreamDestroy(stream);
    cudaFree(managed);
    return 11;
  }

  for (size_t i = 0; i < kCopyElements; ++i) {
    if (managed[kSource + i] != 0 ||
        managed[kDestination + i] != expected_copy[i]) {
      std::fprintf(
          stderr,
          "copy/memset mismatch at %zu: source=%d dest=%d expected=%d\n", i,
          managed[kSource + i], managed[kDestination + i], expected_copy[i]);
      cudaFree(device_scratch);
      cudaStreamDestroy(stream);
      cudaFree(managed);
      return 12;
    }
  }

  constexpr size_t kHostCopyElements = 2048;
  constexpr size_t kHostDestination = 500000;
  int host_input[kHostCopyElements];
  int host_output[kHostCopyElements] = {};
  for (size_t i = 0; i < kHostCopyElements; ++i) {
    host_input[i] = -static_cast<int>(i * 29u + 3u);
  }
  if (check(cudaMemcpyAsync(managed + kHostDestination, host_input,
                            sizeof(host_input), cudaMemcpyDefault, stream),
            "cudaMemcpyAsync(host to managed)") != 0 ||
      check(cudaMemcpyAsync(host_output, managed + kHostDestination,
                            sizeof(host_output), cudaMemcpyDefault, stream),
            "cudaMemcpyAsync(managed to host)") != 0 ||
      check(cudaStreamSynchronize(stream), "host copy cudaStreamSynchronize") !=
          0) {
    cudaFree(device_scratch);
    cudaStreamDestroy(stream);
    cudaFree(managed);
    return 13;
  }
  for (size_t i = 0; i < kHostCopyElements; ++i) {
    if (managed[kHostDestination + i] != host_input[i] ||
        host_output[i] != host_input[i]) {
      std::fprintf(stderr,
                   "host/managed copy mismatch at %zu: managed=%d host=%d "
                   "expected=%d\n",
                   i, managed[kHostDestination + i], host_output[i],
                   host_input[i]);
      cudaFree(device_scratch);
      cudaStreamDestroy(stream);
      cudaFree(managed);
      return 14;
    }
  }

  if (check(cudaFree(device_scratch), "cudaFree(device_scratch)") != 0 ||
      check(cudaStreamDestroy(stream), "cudaStreamDestroy") != 0 ||
      check(cudaFree(managed), "cudaFree(managed)") != 0) {
    return 15;
  }

  std::printf("RESULT: PASS runtime unified-memory coherence\n");
  return 0;
}
