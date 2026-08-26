#include <cuda.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace {

constexpr size_t kKiB = 1024;
constexpr size_t kMiB = 1024 * kKiB;
constexpr size_t kMaterializeStride = 64 * kKiB;
constexpr size_t kTargetBytesPerTrial = 16 * kMiB;
constexpr int kTrials = 5;

volatile uint64_t benchmark_sink = 0;

[[noreturn]] void fail_cuda(CUresult result, const char *operation) {
  const char *name = nullptr;
  const char *description = nullptr;
  cuGetErrorName(result, &name);
  cuGetErrorString(result, &description);
  std::fprintf(stderr, "%s failed: %s (%s)\n", operation,
               name == nullptr ? "unknown" : name,
               description == nullptr ? "no description" : description);
  std::exit(1);
}

void check_cuda(CUresult result, const char *operation) {
  if (result != CUDA_SUCCESS) {
    fail_cuda(result, operation);
  }
}

enum class destination_kind {
  pageable,
  pinned,
  managed_host,
  managed_unified,
};

const char *destination_name(destination_kind kind) {
  switch (kind) {
  case destination_kind::pageable:
    return "pageable";
  case destination_kind::pinned:
    return "pinned";
  case destination_kind::managed_host:
    return "managed-host";
  case destination_kind::managed_unified:
    return "managed-unified";
  }
  return "unknown";
}

struct shape {
  const char *name;
  size_t bytes;
  size_t width;
  size_t height;
  size_t depth;
};

struct host_buffer {
  void *data = nullptr;
  size_t bytes = 0;
  destination_kind kind = destination_kind::pageable;

  host_buffer(destination_kind allocation_kind, size_t allocation_bytes)
      : bytes(allocation_bytes), kind(allocation_kind) {
    switch (kind) {
    case destination_kind::pageable:
      if (posix_memalign(&data, 4096, bytes) != 0) {
        data = nullptr;
      }
      break;
    case destination_kind::pinned:
      check_cuda(cuMemAllocHost(&data, bytes), "cuMemAllocHost");
      break;
    case destination_kind::managed_host:
    case destination_kind::managed_unified: {
      CUdeviceptr managed = 0;
      check_cuda(cuMemAllocManaged(&managed, bytes, CU_MEM_ATTACH_GLOBAL),
                 "cuMemAllocManaged");
      data = reinterpret_cast<void *>(managed);
      break;
    }
    }
    if (data == nullptr) {
      std::fprintf(stderr, "host allocation of %zu bytes failed\n", bytes);
      std::exit(1);
    }
  }

  ~host_buffer() {
    switch (kind) {
    case destination_kind::pageable:
      std::free(data);
      break;
    case destination_kind::pinned:
      (void)cuMemFreeHost(data);
      break;
    case destination_kind::managed_host:
    case destination_kind::managed_unified:
      (void)cuMemFree(reinterpret_cast<CUdeviceptr>(data));
      break;
    }
  }

  host_buffer(const host_buffer &) = delete;
  host_buffer &operator=(const host_buffer &) = delete;
};

size_t destination_offset(const shape &copy_shape, size_t pitch,
                          size_t slice_pitch, size_t packed_offset) {
  size_t per_slice = copy_shape.width * copy_shape.height;
  size_t slice = packed_offset / per_slice;
  size_t within_slice = packed_offset % per_slice;
  size_t row = within_slice / copy_shape.width;
  size_t column = within_slice % copy_shape.width;
  return slice * slice_pitch + row * pitch + column;
}

CUresult copy_once(const shape &copy_shape, destination_kind kind,
                   void *destination, size_t pitch, size_t slice_pitch,
                   CUdeviceptr source) {
  if (copy_shape.height == 1 && copy_shape.depth == 1 &&
      kind != destination_kind::managed_unified) {
    return cuMemcpyDtoH(destination, source, copy_shape.bytes);
  }

  CUDA_MEMCPY3D copy = {};
  copy.srcMemoryType = CU_MEMORYTYPE_DEVICE;
  copy.srcDevice = source;
  copy.srcPitch = copy_shape.width;
  copy.srcHeight = copy_shape.height;
  copy.dstMemoryType = kind == destination_kind::managed_unified
                           ? CU_MEMORYTYPE_UNIFIED
                           : CU_MEMORYTYPE_HOST;
  if (copy.dstMemoryType == CU_MEMORYTYPE_HOST) {
    copy.dstHost = destination;
  } else {
    copy.dstDevice = reinterpret_cast<CUdeviceptr>(destination);
  }
  copy.dstPitch = pitch;
  copy.dstHeight = slice_pitch / pitch;
  copy.WidthInBytes = copy_shape.width;
  copy.Height = copy_shape.height;
  copy.Depth = copy_shape.depth;
  return cuMemcpy3D(&copy);
}

void materialize_managed(const shape &copy_shape, const void *destination,
                         size_t pitch, size_t slice_pitch) {
  auto *bytes = static_cast<const volatile unsigned char *>(destination);
  uint64_t sum = 0;
  for (size_t packed = 0; packed < copy_shape.bytes;
       packed += kMaterializeStride) {
    sum += bytes[destination_offset(copy_shape, pitch, slice_pitch, packed)];
  }
  sum += bytes[destination_offset(copy_shape, pitch, slice_pitch,
                                  copy_shape.bytes - 1)];
  benchmark_sink += sum;
}

void verify_copy(const shape &copy_shape, destination_kind kind,
                 const void *destination, size_t pitch, size_t slice_pitch,
                 const std::vector<unsigned char> &source) {
  const auto *bytes = static_cast<const unsigned char *>(destination);
  const size_t samples[] = {0, copy_shape.bytes / 3, (copy_shape.bytes * 2) / 3,
                            copy_shape.bytes - 1};
  for (size_t packed : samples) {
    size_t destination_index =
        destination_offset(copy_shape, pitch, slice_pitch, packed);
    if (bytes[destination_index] != source[packed]) {
      std::fprintf(stderr,
                   "verification failed for %s/%s at packed offset %zu: "
                   "got 0x%02x, expected 0x%02x\n",
                   destination_name(kind), copy_shape.name, packed,
                   bytes[destination_index], source[packed]);
      std::exit(1);
    }
  }
}

double median(std::vector<double> samples) {
  std::sort(samples.begin(), samples.end());
  return samples[samples.size() / 2];
}

void benchmark_shape(const shape &copy_shape, destination_kind kind,
                     CUdeviceptr source,
                     const std::vector<unsigned char> &source_bytes) {
  const size_t pitch = copy_shape.width + 128;
  const size_t destination_height = copy_shape.height + 3;
  const size_t slice_pitch = pitch * destination_height;
  const size_t allocation_bytes = (copy_shape.depth - 1) * slice_pitch +
                                  (copy_shape.height - 1) * pitch +
                                  copy_shape.width;
  host_buffer destination(kind, allocation_bytes);
  std::memset(destination.data, 0, allocation_bytes);

  auto run_copy = [&]() {
    CUresult result = copy_once(copy_shape, kind, destination.data, pitch,
                                slice_pitch, source);
    if (result != CUDA_SUCCESS) {
      fail_cuda(result, "DtoH benchmark copy");
    }
    if (kind == destination_kind::managed_host ||
        kind == destination_kind::managed_unified) {
      // A managed destination can stay resident at the server. Include the
      // cost of making the copied bytes observable on this client, otherwise
      // this is only timing a device-side copy rather than a DtoH transfer.
      materialize_managed(copy_shape, destination.data, pitch, slice_pitch);
    }
  };

  run_copy();
  run_copy();
  size_t iterations =
      std::max<size_t>(1, kTargetBytesPerTrial / copy_shape.bytes);
  iterations = std::min<size_t>(iterations, 200);
  std::vector<double> samples;
  samples.reserve(kTrials);
  for (int trial = 0; trial < kTrials; ++trial) {
    auto start = std::chrono::steady_clock::now();
    for (size_t iteration = 0; iteration < iterations; ++iteration) {
      run_copy();
    }
    auto end = std::chrono::steady_clock::now();
    double seconds = std::chrono::duration<double>(end - start).count();
    samples.push_back(seconds / static_cast<double>(iterations));
  }

  verify_copy(copy_shape, kind, destination.data, pitch, slice_pitch,
              source_bytes);
  double seconds = median(samples);
  double gib_per_second = static_cast<double>(copy_shape.bytes) / seconds /
                          (1024.0 * 1024.0 * 1024.0);
  std::printf("%s,%s,%zu,%zu,%zu,%zu,%zu,%.3f,%.6f\n", destination_name(kind),
              copy_shape.name, copy_shape.bytes, copy_shape.width,
              copy_shape.height, copy_shape.depth, iterations, seconds * 1e6,
              gib_per_second);
  std::fflush(stdout);
}

} // namespace

int main() {
  check_cuda(cuInit(0), "cuInit");
  CUdevice device = 0;
  check_cuda(cuDeviceGet(&device, 0), "cuDeviceGet");
  CUcontext context = nullptr;
  check_cuda(cuCtxCreate(&context, nullptr, 0, device), "cuCtxCreate");

  const shape shapes[] = {
      {"1d", 4 * kKiB, 4 * kKiB, 1, 1},   {"1d", 64 * kKiB, 64 * kKiB, 1, 1},
      {"1d", 1 * kMiB, 1 * kMiB, 1, 1},   {"1d", 4 * kMiB, 4 * kMiB, 1, 1},
      {"1d", 16 * kMiB, 16 * kMiB, 1, 1}, {"1d", 64 * kMiB, 64 * kMiB, 1, 1},
      {"2d", 4 * kKiB, 64, 64, 1},        {"2d", 64 * kKiB, 256, 256, 1},
      {"2d", 1 * kMiB, 1024, 1024, 1},    {"2d", 4 * kMiB, 2048, 2048, 1},
      {"2d", 16 * kMiB, 4096, 4096, 1},   {"2d", 64 * kMiB, 8192, 8192, 1},
      {"3d", 4 * kKiB, 16, 16, 16},       {"3d", 64 * kKiB, 64, 32, 32},
      {"3d", 1 * kMiB, 256, 64, 64},      {"3d", 4 * kMiB, 256, 128, 128},
      {"3d", 16 * kMiB, 1024, 128, 128},  {"3d", 64 * kMiB, 1024, 256, 256},
  };
  constexpr size_t max_bytes = 64 * kMiB;
  std::vector<unsigned char> source_bytes(max_bytes);
  uint32_t random = 0x12345678u;
  for (auto &byte : source_bytes) {
    random ^= random << 13;
    random ^= random >> 17;
    random ^= random << 5;
    byte = static_cast<unsigned char>(random);
  }

  CUdeviceptr source = 0;
  check_cuda(cuMemAlloc(&source, max_bytes), "cuMemAlloc");
  check_cuda(cuMemcpyHtoD(source, source_bytes.data(), source_bytes.size()),
             "benchmark source upload");

  std::puts(
      "memory,dimension,bytes,width,height,depth,iterations,usec,gib_per_sec");
  const destination_kind destinations[] = {
      destination_kind::pageable,
      destination_kind::pinned,
      destination_kind::managed_host,
      destination_kind::managed_unified,
  };
  for (destination_kind kind : destinations) {
    for (const auto &copy_shape : shapes) {
      benchmark_shape(copy_shape, kind, source, source_bytes);
    }
  }

  check_cuda(cuMemFree(source), "cuMemFree");
  check_cuda(cuCtxDestroy(context), "cuCtxDestroy");
  std::fprintf(stderr, "benchmark sink: %llu\n",
               static_cast<unsigned long long>(benchmark_sink));
  return 0;
}
