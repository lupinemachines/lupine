#include <cuda.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

struct host_buffer {
  const char *name = nullptr;
  unsigned char *data = nullptr;
  CUdeviceptr managed = 0;
  bool cuda_host = false;
  bool unified_descriptor = false;
  std::vector<unsigned char> pageable;
};

static CUresult allocate_host(host_buffer *buffer, int kind, size_t bytes) {
  if (kind == 0) {
    buffer->name = "paged";
    buffer->pageable.resize(bytes);
    buffer->data = buffer->pageable.data();
    return CUDA_SUCCESS;
  }
  if (kind == 1) {
    buffer->name = "pinned";
    buffer->cuda_host = true;
    return cuMemHostAlloc(reinterpret_cast<void **>(&buffer->data), bytes,
                          CU_MEMHOSTALLOC_PORTABLE);
  }
  if (kind == 2) {
    buffer->name = "managed";
    CUresult result =
        cuMemAllocManaged(&buffer->managed, bytes, CU_MEM_ATTACH_GLOBAL);
    buffer->data = reinterpret_cast<unsigned char *>(buffer->managed);
    buffer->unified_descriptor = true;
    return result;
  }
  buffer->name = "unified";
  buffer->cuda_host = true;
  buffer->unified_descriptor = true;
  return cuMemHostAlloc(reinterpret_cast<void **>(&buffer->data), bytes,
                        CU_MEMHOSTALLOC_PORTABLE | CU_MEMHOSTALLOC_DEVICEMAP);
}

static void free_host(host_buffer *buffer) {
  if (buffer->managed != 0) {
    cuMemFree(buffer->managed);
  } else if (buffer->cuda_host) {
    cuMemFreeHost(buffer->data);
  }
}

template <typename Copy>
static double measure(Copy copy, int iterations, size_t bytes) {
  copy();
  auto started = std::chrono::steady_clock::now();
  for (int iteration = 0; iteration < iterations; ++iteration) {
    CUresult result = copy();
    if (result != CUDA_SUCCESS) {
      return -static_cast<double>(result);
    }
  }
  double seconds =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - started)
          .count();
  return static_cast<double>(bytes) * iterations / seconds / 1.0e9;
}

int main() {
  if (cuInit(0) != CUDA_SUCCESS) {
    return 1;
  }
  CUdevice device = 0;
  CUcontext context = nullptr;
  if (cuDeviceGet(&device, 0) != CUDA_SUCCESS ||
#if CUDA_VERSION >= 13000
      cuCtxCreate(&context, nullptr, 0, device) != CUDA_SUCCESS) {
#else
      cuCtxCreate(&context, 0, device) != CUDA_SUCCESS) {
#endif
    return 1;
  }
  CUstream stream = nullptr;
  if (cuStreamCreate(&stream, CU_STREAM_NON_BLOCKING) != CUDA_SUCCESS) {
    return 1;
  }

  std::printf("mode,api,shape,bytes,iterations,gbps\n");
  const char *mode_filter = std::getenv("HTOD_MODE");
  const char *shape_filter = std::getenv("HTOD_SHAPE");
  const char *bytes_filter = std::getenv("HTOD_BYTES");
  const size_t sizes[] = {4 * 1024, 64 * 1024, 1024 * 1024, 8 * 1024 * 1024,
                          32 * 1024 * 1024};
  for (size_t bytes : sizes) {
    if (bytes_filter != nullptr && bytes != static_cast<size_t>(std::strtoull(
                                                bytes_filter, nullptr, 10))) {
      continue;
    }
    int iterations = static_cast<int>(
        std::min<size_t>(100, std::max<size_t>(5, 128 * 1024 * 1024 / bytes)));
    const size_t width_2d = std::min<size_t>(4096, bytes);
    const size_t height_2d = bytes / width_2d;
    const size_t pitch_2d = width_2d + 128;
    const size_t width_3d = std::min<size_t>(1024, bytes);
    const size_t height_3d = std::min<size_t>(64, bytes / width_3d);
    const size_t depth_3d = bytes / (width_3d * height_3d);
    const size_t pitch_3d = width_3d + 128;
    const size_t source_height_3d = height_3d + 1;
    const size_t source_slice_3d = pitch_3d * source_height_3d;
    const size_t host_bytes =
        std::max({bytes, pitch_2d * height_2d, source_slice_3d * depth_3d});

    CUdeviceptr destination = 0;
    if (cuMemAlloc(&destination, bytes) != CUDA_SUCCESS) {
      return 1;
    }
    for (int kind = 0; kind != 4; ++kind) {
      host_buffer source;
      if (allocate_host(&source, kind, host_bytes) != CUDA_SUCCESS) {
        continue;
      }
      if (mode_filter != nullptr &&
          std::strcmp(mode_filter, source.name) != 0) {
        free_host(&source);
        continue;
      }
      std::memset(source.data, 0x5a, host_bytes);

      auto print = [&](const char *api, const char *shape, double gbps) {
        std::printf("%s,%s,%s,%zu,%d,%.6f\n", source.name, api, shape, bytes,
                    iterations, gbps);
        std::fflush(stdout);
      };
      if (shape_filter == nullptr || std::strcmp(shape_filter, "1d") == 0) {
        print("sync", "1d",
              measure(
                  [&] { return cuMemcpyHtoD(destination, source.data, bytes); },
                  iterations, bytes));
        print("async", "1d",
              measure(
                  [&] {
                    CUresult result = cuMemcpyHtoDAsync(
                        destination, source.data, bytes, stream);
                    return result == CUDA_SUCCESS ? cuStreamSynchronize(stream)
                                                  : result;
                  },
                  iterations, bytes));
      }

      CUDA_MEMCPY2D copy_2d = {};
      copy_2d.srcMemoryType = source.unified_descriptor ? CU_MEMORYTYPE_UNIFIED
                                                        : CU_MEMORYTYPE_HOST;
      if (source.unified_descriptor) {
        copy_2d.srcDevice = reinterpret_cast<CUdeviceptr>(source.data);
      } else {
        copy_2d.srcHost = source.data;
      }
      copy_2d.srcPitch = pitch_2d;
      copy_2d.dstMemoryType = CU_MEMORYTYPE_DEVICE;
      copy_2d.dstDevice = destination;
      copy_2d.dstPitch = width_2d;
      copy_2d.WidthInBytes = width_2d;
      copy_2d.Height = height_2d;
      if (shape_filter == nullptr || std::strcmp(shape_filter, "2d") == 0) {
        print("sync", "2d",
              measure([&] { return cuMemcpy2D(&copy_2d); }, iterations, bytes));
        print("async", "2d",
              measure(
                  [&] {
                    CUresult result = cuMemcpy2DAsync(&copy_2d, stream);
                    return result == CUDA_SUCCESS ? cuStreamSynchronize(stream)
                                                  : result;
                  },
                  iterations, bytes));
      }

      CUDA_MEMCPY3D copy_3d = {};
      copy_3d.srcMemoryType = copy_2d.srcMemoryType;
      copy_3d.srcHost = copy_2d.srcHost;
      copy_3d.srcDevice = copy_2d.srcDevice;
      copy_3d.srcPitch = pitch_3d;
      copy_3d.srcHeight = source_height_3d;
      copy_3d.dstMemoryType = CU_MEMORYTYPE_DEVICE;
      copy_3d.dstDevice = destination;
      copy_3d.dstPitch = width_3d;
      copy_3d.dstHeight = height_3d;
      copy_3d.WidthInBytes = width_3d;
      copy_3d.Height = height_3d;
      copy_3d.Depth = depth_3d;
      if (shape_filter == nullptr || std::strcmp(shape_filter, "3d") == 0) {
        print("sync", "3d",
              measure([&] { return cuMemcpy3D(&copy_3d); }, iterations, bytes));
        print("async", "3d",
              measure(
                  [&] {
                    CUresult result = cuMemcpy3DAsync(&copy_3d, stream);
                    return result == CUDA_SUCCESS ? cuStreamSynchronize(stream)
                                                  : result;
                  },
                  iterations, bytes));
      }
      free_host(&source);
    }
    cuMemFree(destination);
  }
  cuStreamDestroy(stream);
  cuCtxDestroy(context);
  return 0;
}
