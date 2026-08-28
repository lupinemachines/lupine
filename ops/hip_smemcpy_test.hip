#include "ops/hip_smemcpy.h"

#include <hip/hip_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <vector>

namespace {

constexpr unsigned char kUntouched = 0xcd;
constexpr size_t kSourceBytes = 8 * 1024 * 1024 + 4096;

bool check_hip(hipError_t status, const char *expression,
               const char *case_name) {
  if (status == hipSuccess) {
    return true;
  }
  std::fprintf(stderr, "%s: %s failed: %s\n", case_name, expression,
               hipGetErrorString(status));
  return false;
}

#define CHECK_CASE(name, call)                                                 \
  do {                                                                         \
    if (!check_hip((call), #call, (name))) {                                   \
      return false;                                                            \
    }                                                                          \
  } while (0)

struct device_buffer {
  unsigned char *data = nullptr;

  ~device_buffer() {
    if (data != nullptr) {
      (void)hipFree(data);
    }
  }
};

template <typename NativeCopy>
bool compare_with_native(const char *name, size_t allocation_bytes,
                         size_t destination_offset, const unsigned char *host,
                         unsigned long long mapped, size_t source_offset,
                         lupine_hip_smemcpy_params params, hipStream_t stream,
                         NativeCopy native_copy) {
  if (allocation_bytes == 0 || source_offset > kSourceBytes ||
      params.bytes > kSourceBytes - source_offset) {
    std::fprintf(stderr, "%s: invalid test extent\n", name);
    return false;
  }

  device_buffer expected;
  device_buffer actual;
  CHECK_CASE(name, hipMalloc(&expected.data, allocation_bytes));
  CHECK_CASE(name, hipMalloc(&actual.data, allocation_bytes));
  CHECK_CASE(name, hipMemsetAsync(expected.data, kUntouched, allocation_bytes,
                                  stream));
  CHECK_CASE(name,
             hipMemsetAsync(actual.data, kUntouched, allocation_bytes, stream));
  CHECK_CASE(name, native_copy(expected.data));

  params.destination =
      reinterpret_cast<unsigned long long>(actual.data + destination_offset);
  params.source = mapped + source_offset;
  CHECK_CASE(name, lupine_hip_smemcpy_async(&params, stream));
  CHECK_CASE(name, hipStreamSynchronize(stream));

  std::vector<unsigned char> expected_host(allocation_bytes);
  std::vector<unsigned char> actual_host(allocation_bytes);
  CHECK_CASE(name, hipMemcpy(expected_host.data(), expected.data,
                             allocation_bytes, hipMemcpyDeviceToHost));
  CHECK_CASE(name, hipMemcpy(actual_host.data(), actual.data, allocation_bytes,
                             hipMemcpyDeviceToHost));
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
                     unsigned long long mapped, hipStream_t stream) {
  char name[192];
  std::snprintf(name, sizeof(name),
                "hipMemcpyAsync 1D total=%zu offset=%zu bytes=%zu src=%zu "
                "dst=%zu",
                test.logical_bytes, test.logical_offset, test.bytes,
                test.source_offset, test.destination_offset);
  if (test.logical_offset > test.logical_bytes ||
      test.bytes > test.logical_bytes - test.logical_offset) {
    std::fprintf(stderr, "%s: invalid case\n", name);
    return false;
  }

  lupine_hip_smemcpy_params params = {};
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
        return hipMemcpyAsync(destination + test.destination_offset +
                                  test.logical_offset,
                              host + test.source_offset, test.bytes,
                              hipMemcpyHostToDevice, stream);
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
                    unsigned long long mapped, hipStream_t stream) {
  char name[192];
  std::snprintf(name, sizeof(name),
                "hipMemcpy2DAsync width=%zu rows=%zu pitch=%zu src=%zu "
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
  lupine_hip_smemcpy_params params = {};
  params.bytes = logical_bytes;
  params.width = test.width;
  params.rows = test.rows;
  params.destination_row_stride = test.pitch;
  params.destination_slice_stride = test.pitch * test.rows;
  return compare_with_native(
      name, allocation_bytes, test.destination_offset, host, mapped,
      test.source_offset, params, stream, [&](unsigned char *destination) {
        return hipMemcpy2DAsync(destination + test.destination_offset,
                                test.pitch, host + test.source_offset,
                                test.width, test.width, test.rows,
                                hipMemcpyHostToDevice, stream);
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
                     unsigned long long mapped, hipStream_t stream) {
  char name[224];
  std::snprintf(name, sizeof(name),
                "hipMemcpy3DAsync width=%zu rows=%zu slices=%zu pitch=%zu "
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
  lupine_hip_smemcpy_params params = {};
  params.bytes = test.width * test.rows * test.slices;
  params.width = test.width;
  params.rows = test.rows;
  params.destination_row_stride = test.pitch;
  params.destination_slice_stride = slice_stride;
  return compare_with_native(
      name, allocation_bytes, test.destination_offset, host, mapped,
      test.source_offset, params, stream, [&](unsigned char *destination) {
        hipMemcpy3DParms copy = {};
        copy.srcPtr = make_hipPitchedPtr(
            const_cast<unsigned char *>(host + test.source_offset), test.width,
            test.width, test.rows);
        copy.dstPtr = make_hipPitchedPtr(destination + test.destination_offset,
                                         test.pitch, test.width,
                                         test.destination_slice_rows);
        copy.extent = make_hipExtent(test.width, test.rows, test.slices);
        copy.kind = hipMemcpyHostToDevice;
        return hipMemcpy3DAsync(&copy, stream);
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
                       unsigned long long mapped, hipStream_t stream) {
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
  lupine_hip_smemcpy_params params = {};
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
          hipError_t status = hipMemcpyAsync(
              destination + test.destination_offset +
                  slice * test.slice_stride + row * test.row_stride + x,
              host + test.source_offset + copied, chunk, hipMemcpyHostToDevice,
              stream);
          if (status != hipSuccess) {
            return status;
          }
          logical += chunk;
          copied += chunk;
        }
        return hipSuccess;
      });
}

bool expect_invalid(const char *name, const lupine_hip_smemcpy_params &params) {
  lupine_hip_smemcpy_launch launch = {};
  hipError_t status = lupine_hip_smemcpy_prepare_launch(&params, &launch);
  if (status == hipErrorInvalidValue) {
    return true;
  }
  std::fprintf(stderr, "%s: expected hipErrorInvalidValue, got %s\n", name,
               hipGetErrorString(status));
  return false;
}

bool run_parameter_validation() {
  lupine_hip_smemcpy_launch launch = {};
  lupine_hip_smemcpy_params params = {};
  if (lupine_hip_smemcpy_prepare_launch(nullptr, &launch) !=
          hipErrorInvalidValue ||
      lupine_hip_smemcpy_prepare_launch(&params, nullptr) !=
          hipErrorInvalidValue) {
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
  if (lupine_hip_smemcpy_prepare_launch(&params, &launch) != hipSuccess ||
      launch.blocks != 0 ||
      lupine_hip_smemcpy_async(&params, nullptr) != hipSuccess) {
    std::fputs("zero-byte copy validation failed\n", stderr);
    return false;
  }
  return true;
}

bool unavailable_hip(hipError_t status) {
  return status == hipErrorNoDevice || status == hipErrorInsufficientDriver ||
         status == hipErrorInitializationError;
}

} // namespace

int main() {
  hipError_t status = hipSetDeviceFlags(hipDeviceMapHost);
  if (unavailable_hip(status)) {
    std::printf("SKIP: HIP unavailable: %s\n", hipGetErrorString(status));
    return 77;
  }
  if (!check_hip(status, "hipSetDeviceFlags", "initialization")) {
    return 1;
  }

  int device_count = 0;
  status = hipGetDeviceCount(&device_count);
  if (unavailable_hip(status) || (status == hipSuccess && device_count == 0)) {
    std::printf("SKIP: no HIP device available\n");
    return 77;
  }
  if (!check_hip(status, "hipGetDeviceCount", "initialization")) {
    return 1;
  }
  hipDeviceProp_t properties = {};
  if (!check_hip(hipGetDeviceProperties(&properties, 0),
                 "hipGetDeviceProperties", "initialization")) {
    return 1;
  }
  if (!properties.canMapHostMemory) {
    std::printf("SKIP: %s cannot map host memory\n", properties.name);
    return 77;
  }

  unsigned char *host = nullptr;
  if (!check_hip(hipHostMalloc(&host, kSourceBytes, hipHostMallocMapped),
                 "hipHostMalloc", "initialization")) {
    return 1;
  }
  for (size_t index = 0; index < kSourceBytes; ++index) {
    host[index] = static_cast<unsigned char>(
        (index * 131U + (index >> 7) * 29U + 17U) & 0xffU);
  }
  unsigned char *mapped_pointer = nullptr;
  if (!check_hip(hipHostGetDevicePointer(
                     reinterpret_cast<void **>(&mapped_pointer), host, 0),
                 "hipHostGetDevicePointer", "initialization")) {
    (void)hipHostFree(host);
    return 1;
  }
  unsigned long long mapped =
      reinterpret_cast<unsigned long long>(mapped_pointer);
  hipStream_t stream = nullptr;
  if (!check_hip(hipStreamCreate(&stream), "hipStreamCreate",
                 "initialization")) {
    (void)hipHostFree(host);
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
  if (!check_hip(hipStreamDestroy(stream), "hipStreamDestroy", "cleanup") ||
      !check_hip(hipHostFree(host), "hipHostFree", "cleanup")) {
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
