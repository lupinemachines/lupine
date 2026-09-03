#include "ops/smemcpy.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <vector>

namespace {

constexpr unsigned char kUntouched = 0xcd;
constexpr size_t kSourceBytes = 8 * 1024 * 1024 + 4096;

bool check_cuda(cudaError_t status, const char *expression,
                const char *case_name) {
  if (status == cudaSuccess) {
    return true;
  }
  std::fprintf(stderr, "%s: %s failed: %s\n", case_name, expression,
               cudaGetErrorString(status));
  return false;
}

#define CHECK_CASE(name, call)                                                 \
  do {                                                                         \
    if (!check_cuda((call), #call, (name))) {                                  \
      return false;                                                            \
    }                                                                          \
  } while (0)

struct device_buffer {
  unsigned char *data = nullptr;

  ~device_buffer() {
    if (data != nullptr) {
      cudaFree(data);
    }
  }
};

template <typename NativeCopy>
bool compare_with_native(const char *name, size_t allocation_bytes,
                         size_t destination_offset, const unsigned char *host,
                         CUdeviceptr mapped, size_t source_offset,
                         lupine_smemcpy_params params, cudaStream_t stream,
                         NativeCopy native_copy) {
  if (allocation_bytes == 0 || source_offset > kSourceBytes ||
      params.bytes > kSourceBytes - source_offset) {
    std::fprintf(stderr, "%s: invalid test extent\n", name);
    return false;
  }

  device_buffer expected;
  device_buffer actual;
  CHECK_CASE(name, cudaMalloc(&expected.data, allocation_bytes));
  CHECK_CASE(name, cudaMalloc(&actual.data, allocation_bytes));
  CHECK_CASE(name, cudaMemsetAsync(expected.data, kUntouched, allocation_bytes,
                                   stream));
  CHECK_CASE(
      name, cudaMemsetAsync(actual.data, kUntouched, allocation_bytes, stream));
  CHECK_CASE(name, native_copy(expected.data));

  params.destination =
      reinterpret_cast<CUdeviceptr>(actual.data + destination_offset);
  params.source = mapped + source_offset;
  CHECK_CASE(name, lupine_smemcpy_async(&params, stream));
  CHECK_CASE(name, cudaStreamSynchronize(stream));

  std::vector<unsigned char> expected_host(allocation_bytes);
  std::vector<unsigned char> actual_host(allocation_bytes);
  CHECK_CASE(name, cudaMemcpy(expected_host.data(), expected.data,
                              allocation_bytes, cudaMemcpyDeviceToHost));
  CHECK_CASE(name, cudaMemcpy(actual_host.data(), actual.data, allocation_bytes,
                              cudaMemcpyDeviceToHost));
  if (expected_host == actual_host) {
    return true;
  }

  size_t mismatch = 0;
  while (mismatch < allocation_bytes &&
         expected_host[mismatch] == actual_host[mismatch]) {
    ++mismatch;
  }
  std::fprintf(stderr,
               "%s: mismatch at physical byte %zu: native=0x%02x "
               "scatter=0x%02x\n",
               name, mismatch, expected_host[mismatch], actual_host[mismatch]);
  return false;
}

struct linear_case {
  size_t logical_bytes;
  size_t logical_offset;
  size_t bytes;
  size_t source_offset;
  size_t destination_offset;
};

bool run_linear_case(const linear_case &test, const unsigned char *host,
                     CUdeviceptr mapped, cudaStream_t stream) {
  char name[192];
  std::snprintf(name, sizeof(name),
                "cudaMemcpyAsync 1D total=%zu offset=%zu bytes=%zu src=%zu "
                "dst=%zu",
                test.logical_bytes, test.logical_offset, test.bytes,
                test.source_offset, test.destination_offset);
  if (test.logical_offset > test.logical_bytes ||
      test.bytes > test.logical_bytes - test.logical_offset) {
    std::fprintf(stderr, "%s: invalid case\n", name);
    return false;
  }

  lupine_smemcpy_params params = {};
  params.logical_offset = test.logical_offset;
  params.bytes = test.bytes;
  params.width = test.logical_bytes;
  params.rows = 1;
  params.destination_row_stride = test.logical_bytes;
  params.destination_slice_stride = test.logical_bytes;
  size_t allocation_bytes = test.destination_offset + test.logical_bytes;
  return compare_with_native(
      name, allocation_bytes, test.destination_offset, host, mapped,
      test.source_offset, params, stream, [&](unsigned char *destination) {
        return cudaMemcpyAsync(destination + test.destination_offset +
                                   test.logical_offset,
                               host + test.source_offset, test.bytes,
                               cudaMemcpyHostToDevice, stream);
      });
}

struct plane_case {
  size_t width;
  size_t rows;
  size_t pitch;
  size_t source_offset;
  size_t destination_offset;
};

bool run_plane_case(const plane_case &test, const unsigned char *host,
                    CUdeviceptr mapped, cudaStream_t stream) {
  char name[192];
  std::snprintf(name, sizeof(name),
                "cudaMemcpy2DAsync width=%zu rows=%zu pitch=%zu src=%zu "
                "dst=%zu",
                test.width, test.rows, test.pitch, test.source_offset,
                test.destination_offset);
  if (test.width == 0 || test.rows == 0 || test.pitch < test.width) {
    std::fprintf(stderr, "%s: invalid case\n", name);
    return false;
  }

  size_t logical_bytes = test.width * test.rows;
  size_t allocation_bytes =
      test.destination_offset + (test.rows - 1) * test.pitch + test.width;
  lupine_smemcpy_params params = {};
  params.bytes = logical_bytes;
  params.width = test.width;
  params.rows = test.rows;
  params.destination_row_stride = test.pitch;
  params.destination_slice_stride = test.pitch * test.rows;
  return compare_with_native(
      name, allocation_bytes, test.destination_offset, host, mapped,
      test.source_offset, params, stream, [&](unsigned char *destination) {
        return cudaMemcpy2DAsync(destination + test.destination_offset,
                                 test.pitch, host + test.source_offset,
                                 test.width, test.width, test.rows,
                                 cudaMemcpyHostToDevice, stream);
      });
}

struct volume_case {
  size_t width;
  size_t rows;
  size_t slices;
  size_t pitch;
  size_t destination_slice_rows;
  size_t source_offset;
  size_t destination_offset;
};

bool run_volume_case(const volume_case &test, const unsigned char *host,
                     CUdeviceptr mapped, cudaStream_t stream) {
  char name[224];
  std::snprintf(name, sizeof(name),
                "cudaMemcpy3DAsync width=%zu rows=%zu slices=%zu pitch=%zu "
                "slice_rows=%zu src=%zu dst=%zu",
                test.width, test.rows, test.slices, test.pitch,
                test.destination_slice_rows, test.source_offset,
                test.destination_offset);
  if (test.width == 0 || test.rows == 0 || test.slices == 0 ||
      test.pitch < test.width || test.destination_slice_rows < test.rows) {
    std::fprintf(stderr, "%s: invalid case\n", name);
    return false;
  }

  size_t slice_stride = test.pitch * test.destination_slice_rows;
  size_t allocation_bytes = test.destination_offset +
                            (test.slices - 1) * slice_stride +
                            (test.rows - 1) * test.pitch + test.width;
  lupine_smemcpy_params params = {};
  params.bytes = test.width * test.rows * test.slices;
  params.width = test.width;
  params.rows = test.rows;
  params.destination_row_stride = test.pitch;
  params.destination_slice_stride = slice_stride;
  return compare_with_native(
      name, allocation_bytes, test.destination_offset, host, mapped,
      test.source_offset, params, stream, [&](unsigned char *destination) {
        cudaMemcpy3DParms copy = {};
        copy.srcPtr = make_cudaPitchedPtr(
            const_cast<unsigned char *>(host + test.source_offset), test.width,
            test.width, test.rows);
        copy.dstPtr = make_cudaPitchedPtr(destination + test.destination_offset,
                                          test.pitch, test.width,
                                          test.destination_slice_rows);
        copy.extent = make_cudaExtent(test.width, test.rows, test.slices);
        copy.kind = cudaMemcpyHostToDevice;
        return cudaMemcpy3DAsync(&copy, stream);
      });
}

struct fragment_case {
  size_t width;
  size_t rows;
  size_t slices;
  size_t row_stride;
  size_t slice_stride;
  size_t logical_offset;
  size_t bytes;
  size_t source_offset;
  size_t destination_offset;
};

bool run_fragment_case(const fragment_case &test, const unsigned char *host,
                       CUdeviceptr mapped, cudaStream_t stream) {
  char name[256];
  std::snprintf(name, sizeof(name),
                "fragment width=%zu rows=%zu slices=%zu row_stride=%zu "
                "slice_stride=%zu offset=%zu bytes=%zu src=%zu dst=%zu",
                test.width, test.rows, test.slices, test.row_stride,
                test.slice_stride, test.logical_offset, test.bytes,
                test.source_offset, test.destination_offset);
  size_t logical_bytes = test.width * test.rows * test.slices;
  if (test.width == 0 || test.rows == 0 || test.slices == 0 ||
      test.row_stride < test.width ||
      test.slice_stride < test.row_stride * test.rows ||
      test.logical_offset > logical_bytes ||
      test.bytes > logical_bytes - test.logical_offset) {
    std::fprintf(stderr, "%s: invalid case\n", name);
    return false;
  }

  size_t allocation_bytes = test.destination_offset +
                            (test.slices - 1) * test.slice_stride +
                            (test.rows - 1) * test.row_stride + test.width;
  lupine_smemcpy_params params = {};
  params.logical_offset = test.logical_offset;
  params.bytes = test.bytes;
  params.width = test.width;
  params.rows = test.rows;
  params.destination_row_stride = test.row_stride;
  params.destination_slice_stride = test.slice_stride;
  return compare_with_native(
      name, allocation_bytes, test.destination_offset, host, mapped,
      test.source_offset, params, stream, [&](unsigned char *destination) {
        size_t logical = test.logical_offset;
        size_t copied = 0;
        while (copied < test.bytes) {
          size_t row_index = logical / test.width;
          size_t x = logical - row_index * test.width;
          size_t slice = row_index / test.rows;
          size_t row = row_index - slice * test.rows;
          size_t chunk = std::min(test.bytes - copied, test.width - x);
          cudaError_t status = cudaMemcpyAsync(
              destination + test.destination_offset +
                  slice * test.slice_stride + row * test.row_stride + x,
              host + test.source_offset + copied, chunk, cudaMemcpyHostToDevice,
              stream);
          if (status != cudaSuccess) {
            return status;
          }
          logical += chunk;
          copied += chunk;
        }
        return cudaSuccess;
      });
}

bool expect_invalid(const char *name, const lupine_smemcpy_params &params) {
  lupine_smemcpy_launch launch = {};
  cudaError_t status = lupine_smemcpy_prepare_launch(&params, &launch);
  if (status == cudaErrorInvalidValue) {
    return true;
  }
  std::fprintf(stderr, "%s: expected cudaErrorInvalidValue, got %s\n", name,
               cudaGetErrorString(status));
  return false;
}

bool run_parameter_validation() {
  lupine_smemcpy_launch launch = {};
  lupine_smemcpy_params params = {};
  if (lupine_smemcpy_prepare_launch(nullptr, &launch) !=
          cudaErrorInvalidValue ||
      lupine_smemcpy_prepare_launch(&params, nullptr) !=
          cudaErrorInvalidValue) {
    std::fputs("null parameter validation failed\n", stderr);
    return false;
  }

  params.rows = 1;
  if (!expect_invalid("zero width", params)) {
    return false;
  }
  params.width = 1;
  params.rows = 0;
  if (!expect_invalid("zero rows", params)) {
    return false;
  }
  params.rows = 1;
  params.bytes = 1;
  if (!expect_invalid("null nonempty pointers", params)) {
    return false;
  }
  params.source = 1;
  params.destination = 1;
  params.width = std::numeric_limits<size_t>::max();
  params.rows = 2;
  if (!expect_invalid("width times rows overflow", params)) {
    return false;
  }
  params.width = 1;
  params.rows = 1;
  params.logical_offset = std::numeric_limits<size_t>::max();
  params.bytes = 1;
  if (!expect_invalid("logical extent overflow", params)) {
    return false;
  }

  params = {};
  params.width = 1;
  params.rows = 1;
  if (lupine_smemcpy_prepare_launch(&params, &launch) != cudaSuccess ||
      launch.blocks != 0 ||
      lupine_smemcpy_async(&params, nullptr) != cudaSuccess) {
    std::fputs("zero-byte copy validation failed\n", stderr);
    return false;
  }

  params = {};
  params.source = 1;
  params.destination = 1;
  params.width = 64;
  params.rows = 2;
  params.destination_row_stride = 64;
  params.destination_slice_stride = 128;
  params.logical_offset = 32;
  params.bytes = 160;
  if (lupine_smemcpy_prepare_launch(&params, &launch) != cudaSuccess ||
      !launch.use_cuda_memcpy || launch.kernel != nullptr) {
    std::fputs("contiguous dispatch validation failed\n", stderr);
    return false;
  }

  params.destination_slice_stride = 192;
  if (lupine_smemcpy_prepare_launch(&params, &launch) != cudaSuccess ||
      launch.use_cuda_memcpy || launch.kernel == nullptr) {
    std::fputs("scatter dispatch validation failed\n", stderr);
    return false;
  }
  return true;
}

bool unavailable_cuda(cudaError_t status) {
  return status == cudaErrorNoDevice || status == cudaErrorInsufficientDriver ||
         status == cudaErrorInitializationError;
}

} // namespace

int main() {
  cudaError_t status = cudaSetDeviceFlags(cudaDeviceMapHost);
  if (unavailable_cuda(status)) {
    std::printf("SKIP: CUDA unavailable: %s\n", cudaGetErrorString(status));
    return 77;
  }
  if (!check_cuda(status, "cudaSetDeviceFlags", "initialization")) {
    return 1;
  }

  int device_count = 0;
  status = cudaGetDeviceCount(&device_count);
  if (unavailable_cuda(status) ||
      (status == cudaSuccess && device_count == 0)) {
    std::printf("SKIP: no CUDA device available\n");
    return 77;
  }
  if (!check_cuda(status, "cudaGetDeviceCount", "initialization")) {
    return 1;
  }
  cudaDeviceProp properties = {};
  if (!check_cuda(cudaGetDeviceProperties(&properties, 0),
                  "cudaGetDeviceProperties", "initialization")) {
    return 1;
  }
  if (!properties.canMapHostMemory) {
    std::printf("SKIP: %s cannot map host memory\n", properties.name);
    return 77;
  }

  unsigned char *host = nullptr;
  if (!check_cuda(cudaHostAlloc(&host, kSourceBytes, cudaHostAllocMapped),
                  "cudaHostAlloc", "initialization")) {
    return 1;
  }
  for (size_t index = 0; index < kSourceBytes; ++index) {
    host[index] = static_cast<unsigned char>(
        (index * 131U + (index >> 7) * 29U + 17U) & 0xffU);
  }
  unsigned char *mapped_pointer = nullptr;
  if (!check_cuda(cudaHostGetDevicePointer(&mapped_pointer, host, 0),
                  "cudaHostGetDevicePointer", "initialization")) {
    cudaFreeHost(host);
    return 1;
  }
  CUdeviceptr mapped = reinterpret_cast<CUdeviceptr>(mapped_pointer);
  cudaStream_t stream = nullptr;
  if (!check_cuda(cudaStreamCreate(&stream), "cudaStreamCreate",
                  "initialization")) {
    cudaFreeHost(host);
    return 1;
  }

  const linear_case linear_cases[] = {
      {1, 0, 1, 0, 0},
      {2, 0, 2, 1, 0},
      {3, 1, 1, 3, 1},
      {7, 0, 7, 7, 3},
      {15, 2, 11, 1, 5},
      {16, 0, 16, 0, 0},
      {17, 0, 17, 1, 0},
      {31, 3, 27, 5, 9},
      {32, 0, 32, 16, 0},
      {63, 1, 61, 31, 7},
      {64, 0, 64, 0, 0},
      {127, 9, 111, 63, 13},
      {128, 0, 128, 128, 0},
      {129, 1, 127, 127, 1},
      {255, 17, 223, 255, 15},
      {256, 0, 256, 256, 0},
      {4095, 31, 4001, 4095, 17},
      {4096, 0, 4096, 0, 0},
      {4097, 1, 4095, 1, 31},
      {4 * 1024 * 1024, 0, 4 * 1024 * 1024, 0, 0},
      {4 * 1024 * 1024 + 257, 127, 4 * 1024 * 1024, 63, 1},
  };
  for (const linear_case &test : linear_cases) {
    if (!run_linear_case(test, host, mapped, stream)) {
      return 1;
    }
  }

  const plane_case plane_cases[] = {
      {1, 4097, 64, 0, 0},    {1, 4097, 128, 1, 1},     {1, 4097, 192, 31, 2},
      {1, 4097, 256, 127, 3}, {1, 4097, 512, 255, 0},   {2, 2053, 64, 0, 0},
      {2, 2053, 128, 1, 1},   {2, 2053, 256, 3, 2},     {2, 2053, 256, 5, 3},
      {2, 2053, 512, 31, 0},  {3, 1367, 64, 0, 0},      {3, 1367, 128, 1, 1},
      {3, 1367, 256, 2, 2},   {3, 1367, 256, 3, 1},     {3, 1367, 512, 31, 0},
      {4, 1025, 64, 0, 0},    {7, 587, 64, 1, 3},       {8, 513, 64, 8, 0},
      {15, 277, 64, 15, 1},   {16, 257, 64, 16, 0},     {31, 133, 64, 31, 7},
      {32, 129, 64, 32, 0},   {33, 127, 128, 1, 9},     {63, 67, 128, 63, 3},
      {64, 65, 128, 64, 0},   {127, 37, 256, 127, 11},  {128, 33, 256, 128, 0},
      {129, 31, 258, 1, 1},   {255, 19, 510, 255, 13},  {256, 17, 512, 256, 0},
      {257, 17, 514, 31, 7},  {4095, 3, 8190, 4095, 1}, {4096, 3, 8192, 0, 0},
      {4097, 3, 8194, 1, 15},
  };
  for (const plane_case &test : plane_cases) {
    if (!run_plane_case(test, host, mapped, stream)) {
      return 1;
    }
  }

  const volume_case volume_cases[] = {
      {1, 17, 5, 64, 17, 0, 0},      {1, 17, 5, 64, 19, 1, 1},
      {2, 129, 4, 256, 129, 3, 2},   {2, 129, 4, 256, 132, 5, 3},
      {3, 97, 5, 256, 97, 1, 1},     {3, 97, 5, 256, 99, 2, 2},
      {4, 19, 7, 64, 23, 7, 3},      {7, 13, 5, 64, 15, 31, 5},
      {16, 11, 4, 32, 13, 16, 0},    {31, 7, 3, 64, 9, 1, 7},
      {64, 64, 8, 256, 66, 64, 0},   {127, 32, 7, 256, 35, 127, 9},
      {128, 17, 6, 256, 19, 128, 0}, {129, 15, 5, 258, 18, 1, 1},
      {256, 8, 4, 512, 10, 256, 0},  {4096, 4, 3, 8192, 6, 0, 0},
  };
  for (const volume_case &test : volume_cases) {
    if (!run_volume_case(test, host, mapped, stream)) {
      return 1;
    }
  }

  const fragment_case fragment_cases[] = {
      {64, 5, 3, 80, 480, 3, 811, 3, 0},
      {24, 7, 4, 40, 320, 5, 523, 5, 0},
      {12, 9, 3, 20, 200, 7, 271, 3, 0},
      {6, 11, 4, 10, 120, 1, 241, 1, 0},
      {37, 11, 5, 61, 793, 398, 430, 17, 9},
      {5, 13, 4, 19, 267, 83, 31, 7, 3},
      {1, 257, 3, 17, 4385, 251, 281, 31, 7},
      {1, 257, 3, 256, 66304, 11, 173, 31, 0},
      {1, 257, 3, 256, 66304, 251, 281, 17, 1},
      {1, 257, 3, 256, 65792, 251, 281, 9, 3},
      {2, 129, 4, 18, 2340, 247, 293, 15, 5},
      {3, 97, 5, 19, 1901, 283, 337, 1, 11},
      {2, 129, 4, 256, 33024, 0, 1032, 0, 0},
      {2, 129, 4, 256, 33024, 0, 1032, 1, 1},
      {2, 129, 4, 256, 33024, 0, 1032, 3, 2},
      {2, 129, 4, 256, 33024, 0, 1032, 5, 3},
      {3, 97, 5, 256, 24832, 0, 1455, 0, 0},
      {3, 97, 5, 256, 24832, 0, 1455, 1, 1},
      {3, 97, 5, 256, 24832, 0, 1455, 2, 2},
      {3, 97, 5, 256, 24896, 0, 1455, 3, 0},
      {31, 7, 3, 31, 217, 19, 401, 9, 13},
  };
  for (const fragment_case &test : fragment_cases) {
    if (!run_fragment_case(test, host, mapped, stream)) {
      return 1;
    }
  }

  uint32_t random = 0x9e3779b9U;
  constexpr int random_cases = 64;
  for (int test_index = 0; test_index < random_cases; ++test_index) {
    auto next = [&] {
      random = random * 1664525U + 1013904223U;
      return random;
    };
    fragment_case test = {};
    test.width = 1 + next() % 257;
    test.rows = 1 + next() % 33;
    test.slices = 1 + next() % 7;
    test.row_stride = test.width + next() % 129;
    test.slice_stride = test.row_stride * test.rows + next() % 65;
    size_t logical_bytes = test.width * test.rows * test.slices;
    test.logical_offset = next() % logical_bytes;
    test.bytes = 1 + next() % (logical_bytes - test.logical_offset);
    test.source_offset = next() % 128;
    test.destination_offset = next() % 32;
    if (!run_fragment_case(test, host, mapped, stream)) {
      return 1;
    }
  }

  if (!run_parameter_validation()) {
    return 1;
  }
  if (!check_cuda(cudaStreamDestroy(stream), "cudaStreamDestroy", "cleanup") ||
      !check_cuda(cudaFreeHost(host), "cudaFreeHost", "cleanup")) {
    return 1;
  }

  std::printf("PASS: %s: %zu native 1D, %zu native 2D, %zu native 3D, "
              "%zu fragmented comparisons plus parameter validation\n",
              properties.name, sizeof(linear_cases) / sizeof(linear_cases[0]),
              sizeof(plane_cases) / sizeof(plane_cases[0]),
              sizeof(volume_cases) / sizeof(volume_cases[0]),
              sizeof(fragment_cases) / sizeof(fragment_cases[0]) +
                  random_cases);
  return 0;
}
