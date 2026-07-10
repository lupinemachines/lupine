#include <cuda.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <vector>

static const char *result_name(CUresult result) {
  const char *name = nullptr;
  if (cuGetErrorName(result, &name) == CUDA_SUCCESS && name != nullptr) {
    return name;
  }
  return "UNKNOWN";
}

static void check(CUresult result, const char *expr, int line) {
  if (result != CUDA_SUCCESS) {
    std::fprintf(stderr, "%s failed at line %d: %s (%d)\n", expr, line,
                 result_name(result), static_cast<int>(result));
    std::exit(EXIT_FAILURE);
  }
}

#define CHECK(expr) check((expr), #expr, __LINE__)

static unsigned char pattern_byte(size_t index) {
  unsigned int value =
      static_cast<unsigned int>(index) * 1664525u + 1013904223u;
  value ^= value >> 13;
  value *= 2246822519u;
  return static_cast<unsigned char>(value >> 24);
}

int main() {
  CHECK(cuInit(0));

  CUdevice device = 0;
  CHECK(cuDeviceGet(&device, 0));

  CUcontext context = nullptr;
#if CUDA_VERSION >= 13000
  CHECK(cuCtxCreate(&context, nullptr, 0, device));
#else
  CHECK(cuCtxCreate(&context, 0, device));
#endif

  constexpr size_t kBlock = 4u * 1024u * 1024u;
  constexpr size_t kPipelineMin = 8u * kBlock;
  constexpr size_t kMaxBytes = kPipelineMin + 12345u;
  constexpr unsigned char kSentinel = 0xa5;
  const size_t sizes[] = {0,          1,          kBlock - 1,      kBlock,
                          kBlock + 1, 2 * kBlock, 2 * kBlock + 17, kPipelineMin,
                          kMaxBytes};

  std::vector<unsigned char> source(kMaxBytes);
  std::vector<unsigned char> received(kMaxBytes, kSentinel);
  for (size_t i = 0; i < source.size(); ++i) {
    source[i] = pattern_byte(i);
  }

  CUdeviceptr device_data = 0;
  CHECK(cuMemAlloc(&device_data, kMaxBytes));
  CHECK(cuMemcpyHtoD(device_data, source.data(), source.size()));

  for (size_t bytes : sizes) {
    std::fill(received.begin(), received.end(), kSentinel);
    CHECK(cuMemcpyDtoH(received.data(), device_data, bytes));
    if (!std::equal(source.begin(), source.begin() + bytes, received.begin())) {
      std::fprintf(stderr, "random DtoH mismatch at size %zu\n", bytes);
      return EXIT_FAILURE;
    }
    if (!std::all_of(received.begin() + bytes, received.end(),
                     [](unsigned char value) { return value == kSentinel; })) {
      std::fprintf(stderr, "DtoH overwrote the suffix at size %zu\n", bytes);
      return EXIT_FAILURE;
    }
  }

  // Exercise the compressed response path with a non-slot-aligned tail.
  constexpr size_t kCompressibleBytes = kPipelineMin + 123;
  CHECK(cuMemsetD8(device_data, 0, kCompressibleBytes));
  std::fill(received.begin(), received.end(), kSentinel);
  CHECK(cuMemcpyDtoH(received.data(), device_data, kCompressibleBytes));
  if (!std::all_of(received.begin(), received.begin() + kCompressibleBytes,
                   [](unsigned char value) { return value == 0; })) {
    std::fprintf(stderr, "compressible DtoH mismatch\n");
    return EXIT_FAILURE;
  }

  // A synchronous DtoH uses the legacy default stream and must retain its
  // implicit ordering with work submitted to a blocking stream.
  CUstream blocking_stream = nullptr;
  CHECK(cuStreamCreate(&blocking_stream, CU_STREAM_DEFAULT));
  CHECK(
      cuMemsetD8Async(device_data, 0x5a, kCompressibleBytes, blocking_stream));
  std::fill(received.begin(), received.end(), kSentinel);
  CHECK(cuMemcpyDtoH(received.data(), device_data, kCompressibleBytes));
  if (!std::all_of(received.begin(), received.begin() + kCompressibleBytes,
                   [](unsigned char value) { return value == 0x5a; })) {
    std::fprintf(stderr, "legacy-stream ordering mismatch\n");
    return EXIT_FAILURE;
  }

  CHECK(cuStreamDestroy(blocking_stream));
  CHECK(cuMemFree(device_data));
  CHECK(cuCtxDestroy(context));
  std::printf("PASS: synchronous DtoH pipeline boundaries and ordering\n");
  return EXIT_SUCCESS;
}
