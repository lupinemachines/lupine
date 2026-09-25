// Pitched copy descriptors at their edges: empty extents, slice strides and
// offsets on graph memcpy nodes, and captured pitched device-to-host copies.
// Every host byte outside the copied rows must be left untouched.
//
// Pure driver API, no kernels (so it is arch-independent).
#include <cstdio>
#include <cstring>
#include <cuda.h>
#include <vector>

static int g_failures = 0;
#define CHECK(cond, msg)                                                       \
  do {                                                                         \
    if (!(cond)) {                                                             \
      fprintf(stderr, "FAIL: %s (line %d)\n", msg, __LINE__);                  \
      g_failures++;                                                            \
    } else {                                                                   \
      fprintf(stderr, "ok:   %s\n", msg);                                      \
    }                                                                          \
  } while (0)

static const char *errstr(CUresult r) {
  const char *s = nullptr;
  cuGetErrorName(r, &s);
  return s ? s : "?";
}
#define DRV(call)                                                              \
  do {                                                                         \
    CUresult _r = (call);                                                      \
    if (_r != CUDA_SUCCESS) {                                                  \
      fprintf(stderr, "FATAL: %s -> %s (line %d)\n", #call, errstr(_r),        \
              __LINE__);                                                       \
      return 1;                                                                \
    }                                                                          \
  } while (0)

static const size_t kWidth = 16;
static const size_t kHeight = 3;
static const size_t kDepth = 2;
static const size_t kPitch = 32;
static const size_t kSliceHeight = 2 * kHeight;
static const size_t kX = 4, kY = 1, kZ = 1;
static const size_t kHostBytes = (kZ + kDepth) * kSliceHeight * kPitch;

static size_t host_index(size_t x, size_t y, size_t z) {
  return ((kZ + z) * kSliceHeight + kY + y) * kPitch + kX + x;
}

static bool in_copy(size_t i) {
  for (size_t z = 0; z < kDepth; ++z) {
    for (size_t y = 0; y < kHeight; ++y) {
      size_t row = host_index(0, y, z);
      if (i >= row && i < row + kWidth) {
        return true;
      }
    }
  }
  return false;
}

// Device side: a dense kWidth x kHeight x kDepth box.
static unsigned char device_value(size_t x, size_t y, size_t z) {
  return (unsigned char)(0x40 + (z * kHeight + y) * kWidth + x);
}

static CUDA_MEMCPY3D host_side_copy(CUdeviceptr device, unsigned char *host,
                                    bool to_device) {
  CUDA_MEMCPY3D copy = {};
  if (to_device) {
    copy.srcMemoryType = CU_MEMORYTYPE_HOST;
    copy.srcHost = host;
    copy.srcXInBytes = kX;
    copy.srcY = kY;
    copy.srcZ = kZ;
    copy.srcPitch = kPitch;
    copy.srcHeight = kSliceHeight;
    copy.dstMemoryType = CU_MEMORYTYPE_DEVICE;
    copy.dstDevice = device;
    copy.dstPitch = kWidth;
    copy.dstHeight = kHeight;
  } else {
    copy.srcMemoryType = CU_MEMORYTYPE_DEVICE;
    copy.srcDevice = device;
    copy.srcPitch = kWidth;
    copy.srcHeight = kHeight;
    copy.dstMemoryType = CU_MEMORYTYPE_HOST;
    copy.dstHost = host;
    copy.dstXInBytes = kX;
    copy.dstY = kY;
    copy.dstZ = kZ;
    copy.dstPitch = kPitch;
    copy.dstHeight = kSliceHeight;
  }
  copy.WidthInBytes = kWidth;
  copy.Height = kHeight;
  copy.Depth = kDepth;
  return copy;
}

static bool check_host_box(const unsigned char *host, unsigned char untouched) {
  for (size_t i = 0; i < kHostBytes; ++i) {
    if (!in_copy(i) && host[i] != untouched) {
      fprintf(stderr, "  byte %zu outside the copy is 0x%02x\n", i, host[i]);
      return false;
    }
  }
  for (size_t z = 0; z < kDepth; ++z) {
    for (size_t y = 0; y < kHeight; ++y) {
      for (size_t x = 0; x < kWidth; ++x) {
        if (host[host_index(x, y, z)] != device_value(x, y, z)) {
          fprintf(stderr, "  (%zu,%zu,%zu) is 0x%02x\n", x, y, z,
                  host[host_index(x, y, z)]);
          return false;
        }
      }
    }
  }
  return true;
}

static void fill_device_box(std::vector<unsigned char> &box) {
  box.resize(kWidth * kHeight * kDepth);
  for (size_t z = 0; z < kDepth; ++z) {
    for (size_t y = 0; y < kHeight; ++y) {
      for (size_t x = 0; x < kWidth; ++x) {
        box[(z * kHeight + y) * kWidth + x] = device_value(x, y, z);
      }
    }
  }
}

static int check_empty_extents(CUdeviceptr device, CUstream stream) {
  unsigned char host[kPitch * 4];
  memset(host, 0xa5, sizeof(host));

  CUDA_MEMCPY2D copy2d = {};
  copy2d.srcMemoryType = CU_MEMORYTYPE_DEVICE;
  copy2d.srcDevice = device;
  copy2d.srcPitch = kWidth;
  copy2d.dstMemoryType = CU_MEMORYTYPE_HOST;
  copy2d.dstHost = host;
  copy2d.dstXInBytes = kX;
  copy2d.dstPitch = kPitch;
  copy2d.WidthInBytes = kWidth;
  copy2d.Height = 0;
  CUresult sync2d = cuMemcpy2D(&copy2d);
  CUresult async2d = cuMemcpy2DAsync(&copy2d, stream);
  CUresult unaligned2d = cuMemcpy2DUnaligned(&copy2d);

  CUDA_MEMCPY3D copy3d = {};
  copy3d.srcMemoryType = CU_MEMORYTYPE_DEVICE;
  copy3d.srcDevice = device;
  copy3d.srcPitch = kWidth;
  copy3d.srcHeight = kHeight;
  copy3d.dstMemoryType = CU_MEMORYTYPE_HOST;
  copy3d.dstHost = host;
  copy3d.dstXInBytes = kX;
  copy3d.dstPitch = kPitch;
  copy3d.dstHeight = 2;
  copy3d.WidthInBytes = kWidth;
  copy3d.Height = kHeight;
  copy3d.Depth = 0;
  CUresult sync3d = cuMemcpy3D(&copy3d);
  CUresult async3d = cuMemcpy3DAsync(&copy3d, stream);
  DRV(cuStreamSynchronize(stream));

  fprintf(stderr,
          "  Height=0 2D: %s/%s/%s, Depth=0 3D: %s/%s\n", errstr(sync2d),
          errstr(async2d), errstr(unaligned2d), errstr(sync3d),
          errstr(async3d));
  CHECK(sync2d == CUDA_SUCCESS && async2d == CUDA_SUCCESS &&
            unaligned2d == CUDA_SUCCESS,
        "2D DtoH with Height 0 succeeds");
  CHECK(sync3d == CUDA_SUCCESS && async3d == CUDA_SUCCESS,
        "3D DtoH with Depth 0 succeeds");
  bool untouched = true;
  for (unsigned char byte : host) {
    untouched = untouched && byte == 0xa5;
  }
  CHECK(untouched, "empty DtoH copies leave the host buffer untouched");
  return 0;
}

static int check_graph_nodes(CUcontext ctx, CUdeviceptr device,
                             CUstream stream) {
  std::vector<unsigned char> box;
  fill_device_box(box);

  // HtoD node reading a strided sub-box of a host volume.
  std::vector<unsigned char> source(kHostBytes, 0xcc);
  for (size_t z = 0; z < kDepth; ++z) {
    for (size_t y = 0; y < kHeight; ++y) {
      for (size_t x = 0; x < kWidth; ++x) {
        source[host_index(x, y, z)] = device_value(x, y, z);
      }
    }
  }
  DRV(cuMemsetD8(device, 0, box.size()));
  CUgraph graph;
  DRV(cuGraphCreate(&graph, 0));
  CUDA_MEMCPY3D htod = host_side_copy(device, source.data(), true);
  CUgraphNode node;
  DRV(cuGraphAddMemcpyNode(&node, graph, nullptr, 0, &htod, ctx));
  CUgraphExec exec;
  DRV(cuGraphInstantiateWithFlags(&exec, graph, 0));
  DRV(cuGraphLaunch(exec, stream));
  DRV(cuStreamSynchronize(stream));
  std::vector<unsigned char> readback(box.size());
  DRV(cuMemcpyDtoH(readback.data(), device, readback.size()));
  CHECK(readback == box, "graph HtoD node honors srcHeight and offsets");
  DRV(cuGraphExecDestroy(exec));
  DRV(cuGraphDestroy(graph));

  // DtoH node writing a strided sub-box of a host volume.
  std::vector<unsigned char> destination(kHostBytes, 0xee);
  DRV(cuGraphCreate(&graph, 0));
  CUDA_MEMCPY3D dtoh = host_side_copy(device, destination.data(), false);
  DRV(cuGraphAddMemcpyNode(&node, graph, nullptr, 0, &dtoh, ctx));
  DRV(cuGraphInstantiateWithFlags(&exec, graph, 0));
  DRV(cuGraphLaunch(exec, stream));
  DRV(cuStreamSynchronize(stream));
  CHECK(check_host_box(destination.data(), 0xee),
        "graph DtoH node honors dstHeight and offsets");
  DRV(cuGraphExecDestroy(exec));
  DRV(cuGraphDestroy(graph));
  return 0;
}

static int check_captured_dtoh(CUdeviceptr device, CUstream stream,
                               bool three_d) {
  const char *name = three_d ? "3D" : "2D";
  char message[128];
  std::vector<unsigned char> box;
  fill_device_box(box);
  DRV(cuMemcpyHtoD(device, box.data(), box.size()));

  void *pinned = nullptr;
  DRV(cuMemAllocHost(&pinned, kHostBytes));
  auto *host = static_cast<unsigned char *>(pinned);
  memset(host, 0x5a, kHostBytes);

  CUDA_MEMCPY3D copy = host_side_copy(device, host, false);
  DRV(cuStreamBeginCapture(stream, CU_STREAM_CAPTURE_MODE_GLOBAL));
  if (three_d) {
    DRV(cuMemcpy3DAsync(&copy, stream));
  } else {
    // One slice at a time, each as a 2D copy.
    for (size_t z = 0; z < kDepth; ++z) {
      CUDA_MEMCPY2D slice = {};
      slice.srcMemoryType = CU_MEMORYTYPE_DEVICE;
      slice.srcDevice = device + z * kHeight * kWidth;
      slice.srcPitch = kWidth;
      slice.dstMemoryType = CU_MEMORYTYPE_HOST;
      slice.dstHost = host + (kZ + z) * kSliceHeight * kPitch;
      slice.dstXInBytes = kX;
      slice.dstY = kY;
      slice.dstPitch = kPitch;
      slice.WidthInBytes = kWidth;
      slice.Height = kHeight;
      DRV(cuMemcpy2DAsync(&slice, stream));
    }
  }
  CUgraph graph;
  DRV(cuStreamEndCapture(stream, &graph));
  bool untouched = true;
  for (size_t i = 0; i < kHostBytes; ++i) {
    untouched = untouched && host[i] == 0x5a;
  }
  snprintf(message, sizeof(message), "capturing a %s DtoH leaves the host",
           name);
  CHECK(untouched, message);
  CUgraphExec exec;
  DRV(cuGraphInstantiateWithFlags(&exec, graph, 0));
  DRV(cuGraphLaunch(exec, stream));
  DRV(cuStreamSynchronize(stream));
  snprintf(message, sizeof(message), "replaying a captured %s DtoH", name);
  CHECK(check_host_box(host, 0x5a), message);
  DRV(cuGraphExecDestroy(exec));
  DRV(cuGraphDestroy(graph));
  DRV(cuMemFreeHost(pinned));
  return 0;
}

int main() {
  DRV(cuInit(0));
  CUdevice dev;
  DRV(cuDeviceGet(&dev, 0));
  CUcontext ctx;
  DRV(cuDevicePrimaryCtxRetain(&ctx, dev));
  DRV(cuCtxSetCurrent(ctx));
  CUstream stream;
  DRV(cuStreamCreate(&stream, CU_STREAM_NON_BLOCKING));
  CUdeviceptr device;
  DRV(cuMemAlloc(&device, kWidth * kHeight * kDepth));

  if (check_empty_extents(device, stream) != 0 ||
      check_graph_nodes(ctx, device, stream) != 0 ||
      check_captured_dtoh(device, stream, false) != 0 ||
      check_captured_dtoh(device, stream, true) != 0) {
    g_failures++;
  }

  DRV(cuMemFree(device));
  DRV(cuStreamDestroy(stream));
  DRV(cuDevicePrimaryCtxRelease(dev));
  printf("%s\n", g_failures == 0 ? "PASSED" : "FAILED");
  return g_failures == 0 ? 0 : 1;
}
