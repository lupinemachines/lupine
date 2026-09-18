// A thread whose very first CUDA call is a host-to-device copy. The lane the
// copy opens has never carried a CUDA call, so nothing has put a context on
// the server thread that serves it; the copy itself has to bind one.
#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <thread>
#include <vector>

static const char *error_name(cudaError_t result) {
  const char *name = cudaGetErrorName(result);
  return name == nullptr ? "unknown" : name;
}

static void check(cudaError_t result, const char *expr, int line) {
  if (result != cudaSuccess) {
    std::fprintf(stderr, "%s failed at line %d: %s (%d)\n", expr, line,
                 error_name(result), static_cast<int>(result));
    std::exit(EXIT_FAILURE);
  }
}

#define CHECK(expr) check((expr), #expr, __LINE__)

static bool copy_from_fresh_thread(size_t bytes, unsigned char seed) {
  std::vector<unsigned char> source(bytes);
  for (size_t i = 0; i < bytes; ++i) {
    source[i] = static_cast<unsigned char>(seed + i);
  }
  void *device = nullptr;
  CHECK(cudaMalloc(&device, bytes));
  CHECK(cudaMemset(device, 0, bytes));
  CHECK(cudaDeviceSynchronize());

  cudaError_t copy_result = cudaErrorUnknown;
  // No cudaSetDevice, no allocation, nothing: the copy is this thread's first
  // CUDA call, so it is what has to establish the lane's binding.
  std::thread writer([&]() {
    copy_result = cudaMemcpy(device, source.data(), bytes,
                             cudaMemcpyHostToDevice);
  });
  writer.join();

  bool ok = true;
  if (copy_result != cudaSuccess) {
    std::fprintf(stderr,
                 "FAIL: %zu-byte copy on a fresh thread returned %s (%d)\n",
                 bytes, error_name(copy_result),
                 static_cast<int>(copy_result));
    ok = false;
  } else {
    std::vector<unsigned char> readback(bytes, 0);
    CHECK(cudaMemcpy(readback.data(), device, bytes, cudaMemcpyDeviceToHost));
    if (std::memcmp(readback.data(), source.data(), bytes) != 0) {
      size_t index = 0;
      while (index < bytes && readback[index] == source[index]) {
        ++index;
      }
      std::fprintf(stderr,
                   "FAIL: %zu-byte copy delivered wrong bytes: index %zu is "
                   "0x%02x, expected 0x%02x\n",
                   bytes, index, readback[index], source[index]);
      ok = false;
    }
  }
  CHECK(cudaFree(device));
  return ok;
}

// The same shape with an async copy and an explicit stream: the stream is
// created on the owning thread, so the writer still makes no call that would
// bind its own lane before the copy.
static bool async_copy_from_fresh_thread(size_t bytes, unsigned char seed) {
  std::vector<unsigned char> source(bytes);
  for (size_t i = 0; i < bytes; ++i) {
    source[i] = static_cast<unsigned char>(seed + 3 * i);
  }
  void *device = nullptr;
  cudaStream_t stream = nullptr;
  CHECK(cudaMalloc(&device, bytes));
  CHECK(cudaMemset(device, 0, bytes));
  CHECK(cudaStreamCreate(&stream));
  CHECK(cudaDeviceSynchronize());

  cudaError_t copy_result = cudaErrorUnknown;
  cudaError_t sync_result = cudaErrorUnknown;
  std::thread writer([&]() {
    copy_result = cudaMemcpyAsync(device, source.data(), bytes,
                                  cudaMemcpyHostToDevice, stream);
    sync_result = cudaStreamSynchronize(stream);
  });
  writer.join();

  bool ok = true;
  if (copy_result != cudaSuccess || sync_result != cudaSuccess) {
    std::fprintf(stderr,
                 "FAIL: %zu-byte async copy on a fresh thread returned %s "
                 "(%d), synchronize returned %s (%d)\n",
                 bytes, error_name(copy_result),
                 static_cast<int>(copy_result), error_name(sync_result),
                 static_cast<int>(sync_result));
    ok = false;
  } else {
    std::vector<unsigned char> readback(bytes, 0);
    CHECK(cudaMemcpy(readback.data(), device, bytes, cudaMemcpyDeviceToHost));
    if (std::memcmp(readback.data(), source.data(), bytes) != 0) {
      size_t index = 0;
      while (index < bytes && readback[index] == source[index]) {
        ++index;
      }
      std::fprintf(stderr,
                   "FAIL: %zu-byte async copy delivered wrong bytes: index "
                   "%zu is 0x%02x, expected 0x%02x\n",
                   bytes, index, readback[index], source[index]);
      ok = false;
    }
  }
  CHECK(cudaStreamDestroy(stream));
  CHECK(cudaFree(device));
  return ok;
}

// cudaGetDevice answers from the client's record of the lane, so a thread that
// asks before it copies has still made no call the server saw.
static bool copy_after_query(size_t bytes, unsigned char seed) {
  std::vector<unsigned char> source(bytes);
  for (size_t i = 0; i < bytes; ++i) {
    source[i] = static_cast<unsigned char>(seed + 7 * i);
  }
  void *device = nullptr;
  CHECK(cudaMalloc(&device, bytes));
  CHECK(cudaMemset(device, 0, bytes));
  CHECK(cudaDeviceSynchronize());

  cudaError_t copy_result = cudaErrorUnknown;
  std::thread writer([&]() {
    int queried = -1;
    if (cudaGetDevice(&queried) != cudaSuccess) {
      return;
    }
    copy_result =
        cudaMemcpy(device, source.data(), bytes, cudaMemcpyHostToDevice);
  });
  writer.join();

  bool ok = true;
  if (copy_result != cudaSuccess) {
    std::fprintf(stderr,
                 "FAIL: %zu-byte copy after cudaGetDevice returned %s (%d)\n",
                 bytes, error_name(copy_result),
                 static_cast<int>(copy_result));
    ok = false;
  } else {
    std::vector<unsigned char> readback(bytes, 0);
    CHECK(cudaMemcpy(readback.data(), device, bytes, cudaMemcpyDeviceToHost));
    if (std::memcmp(readback.data(), source.data(), bytes) != 0) {
      std::fprintf(stderr,
                   "FAIL: %zu-byte copy after cudaGetDevice delivered wrong "
                   "bytes\n",
                   bytes);
      ok = false;
    }
  }
  CHECK(cudaFree(device));
  return ok;
}

int main() {
  bool ok = true;
  ok &= copy_from_fresh_thread(1024, 0x11);
  ok &= copy_from_fresh_thread(1024 * 1024, 0x53);
  ok &= async_copy_from_fresh_thread(1024, 0x27);
  ok &= copy_after_query(1024, 0x6d);
  if (!ok) {
    return 1;
  }
  std::printf("PASS: a thread whose first CUDA call is a host-to-device copy "
              "binds its own lane\n");
  return 0;
}
