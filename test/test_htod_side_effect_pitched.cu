#include <cuda.h>

#include <algorithm>
#include <cstdio>
#include <vector>

#define CHECK(call)                                                            \
  do {                                                                         \
    CUresult check_result = (call);                                            \
    if (check_result != CUDA_SUCCESS) {                                        \
      const char *message = nullptr;                                           \
      cuGetErrorString(check_result, &message);                                \
      std::fprintf(stderr, "%s failed at line %d: %s (%d)\n", #call, __LINE__, \
                   message == nullptr ? "unknown" : message,                   \
                   static_cast<int>(check_result));                            \
      return 1;                                                                \
    }                                                                          \
  } while (0)

static void fill_rows(std::vector<unsigned char> &storage, size_t width,
                      size_t rows, size_t pitch, size_t slices,
                      size_t slice_pitch, unsigned char value) {
  for (size_t z = 0; z < slices; ++z) {
    for (size_t row = 0; row < rows; ++row) {
      std::fill_n(storage.data() + z * slice_pitch + row * pitch, width, value);
    }
  }
}

static bool check_rows(const std::vector<unsigned char> &storage, size_t width,
                       size_t rows, size_t pitch, size_t slices,
                       size_t slice_pitch, unsigned char value) {
  for (size_t z = 0; z < slices; ++z) {
    for (size_t row = 0; row < rows; ++row) {
      const unsigned char *begin =
          storage.data() + z * slice_pitch + row * pitch;
      if (!std::all_of(begin, begin + width,
                       [value](unsigned char byte) { return byte == value; })) {
        return false;
      }
    }
  }
  return true;
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
  CUstream stream = nullptr;
  CHECK(cuStreamCreate(&stream, CU_STREAM_NON_BLOCKING));

  constexpr size_t width_2d = 1024 * 1024 + 13;
  constexpr size_t height_2d = 40;
  constexpr size_t source_pitch_2d = width_2d + 257;
  const size_t destination_pitch_2d = (width_2d + 511) & ~size_t(511);
  std::vector<unsigned char> source_2d(source_pitch_2d * height_2d);
  std::vector<unsigned char> readback_2d(source_pitch_2d * height_2d);
  CUdeviceptr destination_2d = 0;
  CHECK(cuMemAlloc(&destination_2d, destination_pitch_2d * height_2d));

  CUDA_MEMCPY2D copy_2d = {};
  copy_2d.srcMemoryType = CU_MEMORYTYPE_HOST;
  copy_2d.srcHost = source_2d.data();
  copy_2d.srcPitch = source_pitch_2d;
  copy_2d.dstMemoryType = CU_MEMORYTYPE_DEVICE;
  copy_2d.dstDevice = destination_2d;
  copy_2d.dstPitch = destination_pitch_2d;
  copy_2d.WidthInBytes = width_2d;
  copy_2d.Height = height_2d;

  fill_rows(source_2d, width_2d, height_2d, source_pitch_2d, 1, 0, 0x21);
  CHECK(cuMemcpy2D(&copy_2d));

  CUDA_MEMCPY2D read_2d = copy_2d;
  read_2d.srcMemoryType = CU_MEMORYTYPE_DEVICE;
  read_2d.srcDevice = destination_2d;
  read_2d.srcPitch = destination_pitch_2d;
  read_2d.dstMemoryType = CU_MEMORYTYPE_HOST;
  read_2d.dstHost = readback_2d.data();
  read_2d.dstPitch = source_pitch_2d;
  CHECK(cuMemcpy2D(&read_2d));
  if (!check_rows(readback_2d, width_2d, height_2d, source_pitch_2d, 1, 0,
                  0x21)) {
    std::fprintf(stderr, "synchronous 2D HtoD mismatch\n");
    return 1;
  }

  CHECK(cuStreamBeginCapture(stream, CU_STREAM_CAPTURE_MODE_GLOBAL));
  CHECK(cuMemcpy2DAsync(&copy_2d, stream));
  CUgraph graph_2d = nullptr;
  CHECK(cuStreamEndCapture(stream, &graph_2d));
  CUgraphExec executable_2d = nullptr;
  CHECK(cuGraphInstantiateWithFlags(&executable_2d, graph_2d, 0));
  fill_rows(source_2d, width_2d, height_2d, source_pitch_2d, 1, 0, 0x32);
  CHECK(cuGraphLaunch(executable_2d, stream));
  CHECK(cuStreamSynchronize(stream));
  CHECK(cuMemcpy2D(&read_2d));
  if (!check_rows(readback_2d, width_2d, height_2d, source_pitch_2d, 1, 0,
                  0x32)) {
    std::fprintf(stderr, "captured 2D HtoD mismatch\n");
    return 1;
  }

  constexpr size_t width_3d = 256 * 1024 + 7;
  constexpr size_t height_3d = 17;
  constexpr size_t depth_3d = 10;
  constexpr size_t source_pitch_3d = width_3d + 127;
  constexpr size_t source_height_3d = height_3d + 2;
  constexpr size_t source_slice_3d = source_pitch_3d * source_height_3d;
  const size_t destination_pitch_3d = (width_3d + 511) & ~size_t(511);
  const size_t destination_slice_3d = destination_pitch_3d * height_3d;
  std::vector<unsigned char> source_3d(source_slice_3d * depth_3d);
  std::vector<unsigned char> readback_3d(source_slice_3d * depth_3d);
  CUdeviceptr destination_3d = 0;
  CHECK(cuMemAlloc(&destination_3d, destination_slice_3d * depth_3d));

  CUDA_MEMCPY3D copy_3d = {};
  copy_3d.srcMemoryType = CU_MEMORYTYPE_HOST;
  copy_3d.srcHost = source_3d.data();
  copy_3d.srcPitch = source_pitch_3d;
  copy_3d.srcHeight = source_height_3d;
  copy_3d.dstMemoryType = CU_MEMORYTYPE_DEVICE;
  copy_3d.dstDevice = destination_3d;
  copy_3d.dstPitch = destination_pitch_3d;
  copy_3d.dstHeight = height_3d;
  copy_3d.WidthInBytes = width_3d;
  copy_3d.Height = height_3d;
  copy_3d.Depth = depth_3d;

  fill_rows(source_3d, width_3d, height_3d, source_pitch_3d, depth_3d,
            source_slice_3d, 0x43);
  CHECK(cuMemcpy3D(&copy_3d));

  CUDA_MEMCPY3D read_3d = copy_3d;
  read_3d.srcMemoryType = CU_MEMORYTYPE_DEVICE;
  read_3d.srcDevice = destination_3d;
  read_3d.srcPitch = destination_pitch_3d;
  read_3d.srcHeight = height_3d;
  read_3d.dstMemoryType = CU_MEMORYTYPE_HOST;
  read_3d.dstHost = readback_3d.data();
  read_3d.dstPitch = source_pitch_3d;
  read_3d.dstHeight = source_height_3d;
  CHECK(cuMemcpy3D(&read_3d));
  if (!check_rows(readback_3d, width_3d, height_3d, source_pitch_3d, depth_3d,
                  source_slice_3d, 0x43)) {
    std::fprintf(stderr, "synchronous 3D HtoD mismatch\n");
    return 1;
  }

  CHECK(cuStreamBeginCapture(stream, CU_STREAM_CAPTURE_MODE_GLOBAL));
  CHECK(cuMemcpy3DAsync(&copy_3d, stream));
  CUgraph graph_3d = nullptr;
  CHECK(cuStreamEndCapture(stream, &graph_3d));
  CUgraphExec executable_3d = nullptr;
  CHECK(cuGraphInstantiateWithFlags(&executable_3d, graph_3d, 0));
  fill_rows(source_3d, width_3d, height_3d, source_pitch_3d, depth_3d,
            source_slice_3d, 0x54);
  CHECK(cuGraphLaunch(executable_3d, stream));
  CHECK(cuStreamSynchronize(stream));
  CHECK(cuMemcpy3D(&read_3d));
  if (!check_rows(readback_3d, width_3d, height_3d, source_pitch_3d, depth_3d,
                  source_slice_3d, 0x54)) {
    std::fprintf(stderr, "captured 3D HtoD mismatch\n");
    return 1;
  }

  CHECK(cuGraphExecDestroy(executable_3d));
  CHECK(cuGraphDestroy(graph_3d));
  CHECK(cuGraphExecDestroy(executable_2d));
  CHECK(cuGraphDestroy(graph_2d));
  CHECK(cuMemFree(destination_3d));
  CHECK(cuMemFree(destination_2d));
  CHECK(cuStreamDestroy(stream));
  CHECK(cuCtxDestroy(context));
  std::printf("PASS: pitched HtoD side effects preserve 2D/3D capture\n");
  return 0;
}
