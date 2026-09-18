// Kernel-originated writes must work in ordinary pinned buffers, including
// after CUDA copies and RPC output writes. Once exposed to device code, writes
// made before dirty tracking started must reach the device too (#598).
#include <cuda.h>
#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

static constexpr size_t kBytes = 128 * 1024;

static void check(cudaError_t result) {
  if (result != cudaSuccess) {
    fprintf(stderr, "CUDA failure: %s\n", cudaGetErrorString(result));
    exit(1);
  }
}

static void check_driver(CUresult result) {
  if (result != CUDA_SUCCESS) {
    fprintf(stderr, "driver failure: %d\n", static_cast<int>(result));
    exit(1);
  }
}

static void expect(bool condition, const char *message) {
  if (!condition) {
    fprintf(stderr, "FAIL: %s\n", message);
    exit(1);
  }
}

__global__ void increment(unsigned char *bytes) {
  size_t index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < kBytes) {
    ++bytes[index];
  }
}

static void read_pattern(FILE *file, unsigned char *host,
                         const std::vector<unsigned char> &expected) {
  rewind(file);
  // The unbuffered stream writes straight into host, without a userspace copy
  // that could accidentally make a protected page writable first.
  expect(fread(host, 1, kBytes, file) == kBytes, "fread into pinned memory");
  expect(memcmp(host, expected.data(), kBytes) == 0, "file bytes");
}

static void exercise(int kind, bool implicit_mapping) {
  printf("allocation kind %d, implicit mapping %d\n", kind, implicit_mapping);
  fflush(stdout);
  std::vector<unsigned char> expected(kBytes);
  for (size_t i = 0; i < kBytes; ++i) {
    expected[i] = static_cast<unsigned char>(i * 13 + 7);
  }
  FILE *file = tmpfile();
  expect(file != nullptr, "tmpfile");
  expect(setvbuf(file, nullptr, _IONBF, 0) == 0, "setvbuf");
  expect(fwrite(expected.data(), 1, kBytes, file) == kBytes, "write fixture");

  unsigned char *host = nullptr;
  if (kind == 0) {
    check(cudaMallocHost(&host, kBytes));
  } else if (kind == 1) {
    check(cudaHostAlloc(&host, kBytes, cudaHostAllocPortable | cudaHostAllocMapped));
  } else if (kind == 2) {
    check_driver(cuMemAllocHost(reinterpret_cast<void **>(&host), kBytes));
  } else {
    check_driver(cuMemHostAlloc(reinterpret_cast<void **>(&host), kBytes,
                                CU_MEMHOSTALLOC_PORTABLE | CU_MEMHOSTALLOC_DEVICEMAP));
  }
  // Intervening CUDA work must not protect a buffer used only for host I/O.
  unsigned char *device = nullptr;
  check(cudaMalloc(&device, kBytes));
  cudaStream_t stream = nullptr;
  check(cudaStreamCreate(&stream));
  read_pattern(file, host, expected);

  std::vector<unsigned char> actual(kBytes);
  check(cudaMemcpy(device, host, kBytes, cudaMemcpyHostToDevice));
  check(cudaMemcpy(actual.data(), device, kBytes, cudaMemcpyDeviceToHost));
  expect(actual == expected, "HtoD sees file contents");

  check(cudaMemset(device, 0, kBytes));
  check(cudaMemcpyAsync(host, device, kBytes, cudaMemcpyDeviceToHost, stream));
  check(cudaStreamSynchronize(stream));
  for (size_t i = 0; i < kBytes; ++i) {
    expect(host[i] == 0, "async DtoH bytes");
  }
  read_pattern(file, host, expected);
  check(cudaMemcpyAsync(device, host, kBytes, cudaMemcpyHostToDevice, stream));
  check(cudaMemcpyAsync(actual.data(), device, kBytes, cudaMemcpyDeviceToHost, stream));
  check(cudaStreamSynchronize(stream));
  expect(actual == expected, "async HtoD sees replacement file contents");

  // Exercise generic RPC output into a pinned read view, not just memcpy's
  // direct pinned destination path.
  auto *sizes = reinterpret_cast<size_t *>(host);
  check(cudaMemGetInfo(sizes, sizes + 1));
  expect(sizes[1] > 0 && sizes[0] <= sizes[1], "memory info output");
  read_pattern(file, host, expected);

  check(cudaMemset(device, 0, kBytes));
  check(cudaMemcpy2D(device, 512, host, 512, 512, kBytes / 512, cudaMemcpyHostToDevice));
  check(cudaMemcpy(actual.data(), device, kBytes, cudaMemcpyDeviceToHost));
  expect(actual == expected, "2D HtoD sees file contents");

  unsigned char *mapped = host;
  if (!implicit_mapping) {
    check(cudaHostGetDevicePointer(&mapped, host, 0));
  }
  increment<<<kBytes / 256, 256, 0, stream>>>(mapped);
  check(cudaGetLastError());
  check(cudaStreamSynchronize(stream));
  for (size_t i = 0; i < kBytes; ++i) {
    if (host[i] != static_cast<unsigned char>(expected[i] + 1)) {
      fprintf(stderr, "offset %zu: got %u, expected %u\n", i, host[i],
              static_cast<unsigned char>(expected[i] + 1));
    }
    expect(host[i] == static_cast<unsigned char>(expected[i] + 1),
           "mapping preserves bytes written before tracking");
    host[i] = static_cast<unsigned char>(expected[i] + 2);
  }
  increment<<<kBytes / 256, 256, 0, stream>>>(mapped);
  check(cudaGetLastError());
  check(cudaStreamSynchronize(stream));
  for (size_t i = 0; i < kBytes; ++i) {
    expect(host[i] == static_cast<unsigned char>(expected[i] + 3),
           "mapped writes remain coherent after tracking starts");
  }

  check(cudaStreamDestroy(stream));
  check(cudaFree(device));
  if (kind < 2) {
    check(cudaFreeHost(host));
  } else {
    check_driver(cuMemFreeHost(host));
  }
  fclose(file);
}

int main() {
  check(cudaSetDevice(0));
  for (int kind = 0; kind < 4; ++kind) {
    exercise(kind, false);
    exercise(kind, true);
  }
  puts("PASS: pinned file reads, copy reuse, and explicit/implicit device mapping");
}
