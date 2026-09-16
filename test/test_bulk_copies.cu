#include <cuda.h>

// HtoD and DtoH copies at and above the bulk-connection threshold: odd sizes,
// two threads striping at once, and copies queued right behind a memset on the
// same stream, which the striped path must still order after. This file is
// auto-discovered by test/run_custom_tests.sh.

#include <algorithm>
#include <atomic>
#include <cstdio>
#include <cstring>
#include <thread>
#include <vector>

namespace {

const char *error_name(CUresult result) {
  const char *name = nullptr;
  (void)cuGetErrorName(result, &name);
  return name == nullptr ? "unknown" : name;
}

bool check(CUresult result, const char *operation) {
  if (result == CUDA_SUCCESS) {
    return true;
  }
  std::fprintf(stderr, "%s failed: %s (%d)\n", operation, error_name(result),
               static_cast<int>(result));
  return false;
}

void fill(std::vector<unsigned char> &buffer, unsigned int seed) {
  unsigned int x = seed | 1u;
  for (size_t i = 0; i < buffer.size(); ++i) {
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    buffer[i] = static_cast<unsigned char>(x);
  }
}

bool round_trip(CUcontext context, size_t bytes, unsigned int seed,
                bool memset_first) {
  if (!check(cuCtxSetCurrent(context), "cuCtxSetCurrent")) {
    return false;
  }
  CUdeviceptr device = 0;
  if (!check(cuMemAlloc(&device, bytes), "cuMemAlloc")) {
    return false;
  }
  std::vector<unsigned char> source(bytes);
  std::vector<unsigned char> readback(bytes, 0);
  fill(source, seed);
  bool ok = true;
  if (memset_first) {
    ok = check(cuMemsetD8Async(device, 0xff, bytes, CU_STREAM_LEGACY),
               "cuMemsetD8Async");
  }
  ok = ok &&
       check(cuMemcpyHtoD(device, source.data(), bytes), "cuMemcpyHtoD") &&
       check(cuMemcpyDtoH(readback.data(), device, bytes), "cuMemcpyDtoH");
  if (ok && std::memcmp(source.data(), readback.data(), bytes) != 0) {
    std::fprintf(stderr, "readback mismatch for %zu bytes (seed %u)\n", bytes,
                 seed);
    ok = false;
  }
  (void)cuMemFree(device);
  return ok;
}

// A read queued behind a memset on a stream must see the memset, both through
// the synchronous copy and an async copy into pageable memory.
bool read_after_memset(CUcontext context, size_t bytes) {
  if (!check(cuCtxSetCurrent(context), "cuCtxSetCurrent")) {
    return false;
  }
  CUdeviceptr device = 0;
  CUstream stream = nullptr;
  if (!check(cuMemAlloc(&device, bytes), "cuMemAlloc")) {
    return false;
  }
  bool ok = check(cuStreamCreate(&stream, 0), "cuStreamCreate");
  std::vector<unsigned char> readback(bytes, 0);
  ok = ok &&
       check(cuMemsetD8Async(device, 0x5a, bytes, CU_STREAM_LEGACY),
             "cuMemsetD8Async") &&
       check(cuMemcpyDtoH(readback.data(), device, bytes), "cuMemcpyDtoH");
  ok = ok && std::all_of(readback.begin(), readback.end(),
                         [](unsigned char b) { return b == 0x5a; });
  ok = ok &&
       check(cuMemsetD8Async(device, 0xa5, bytes, stream), "cuMemsetD8Async") &&
       check(cuMemcpyDtoHAsync(readback.data(), device, bytes, stream),
             "cuMemcpyDtoHAsync");
  ok = ok && std::all_of(readback.begin(), readback.end(),
                         [](unsigned char b) { return b == 0xa5; });
  if (!ok) {
    std::fprintf(stderr, "read after memset failed for %zu bytes\n", bytes);
  }
  if (stream != nullptr) {
    (void)cuStreamDestroy(stream);
  }
  (void)cuMemFree(device);
  return ok;
}

} // namespace

int main() {
  if (!check(cuInit(0), "cuInit")) {
    return 1;
  }
  CUdevice device = 0;
  CUcontext context = nullptr;
  if (!check(cuDeviceGet(&device, 0), "cuDeviceGet") ||
      !check(cuDevicePrimaryCtxRetain(&context, device),
             "cuDevicePrimaryCtxRetain")) {
    return 1;
  }

  const size_t sizes[] = {8u << 20, (8u << 20) + 4097, (33u << 20) + 5};
  for (size_t bytes : sizes) {
    if (!round_trip(context, bytes, static_cast<unsigned int>(bytes), false) ||
        !round_trip(context, bytes, static_cast<unsigned int>(bytes) + 7,
                    true) ||
        !read_after_memset(context, bytes)) {
      return 1;
    }
  }

  std::atomic<bool> failed{false};
  std::vector<std::thread> workers;
  for (unsigned int t = 0; t < 2; ++t) {
    workers.emplace_back([&, t]() {
      for (int i = 0; i < 3 && !failed.load(); ++i) {
        if (!round_trip(context, (12u << 20) + t * 4096 + i, 1000 * t + i,
                        false)) {
          failed.store(true);
        }
      }
    });
  }
  for (auto &worker : workers) {
    worker.join();
  }
  if (failed.load()) {
    return 1;
  }
  std::printf("PASS: bulk copies round trip in order\n");
  return 0;
}
