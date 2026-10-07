// LUPINE_TEST_CONNECTIONS 2
#include <cuda.h>
#include <cuda_runtime_api.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>

extern "C" CUresult cuCtxCreate_v2(CUcontext *, unsigned int, CUdevice);

static void require(bool condition, const char *message) {
  if (!condition) {
    std::fprintf(stderr, "%s\n", message);
    std::exit(1);
  }
}

static void require_device(int expected, const char *where) {
  int runtime_device = -1;
  require(cudaGetDevice(&runtime_device) == cudaSuccess, "cudaGetDevice failed");
  CUdevice driver_device = -1;
  require(cuCtxGetDevice(&driver_device) == CUDA_SUCCESS,
          "cuCtxGetDevice failed");
  if (runtime_device != expected || static_cast<int>(driver_device) != expected) {
    std::fprintf(stderr,
                 "%s: expected device %d, cudaGetDevice said %d and "
                 "cuCtxGetDevice said %d\n",
                 where, expected, runtime_device,
                 static_cast<int>(driver_device));
    std::exit(1);
  }
}

// The answer is only worth anything if the work goes to the server that owns
// the device it names: a reported ordinal and a route that disagree are worse
// than a wrong ordinal alone. The driver's UUID for the ordinal and the
// runtime's are read through separate paths, and an allocation made through
// the runtime has to land in the context the driver says is current.
static void require_routes_agree(int ordinal, CUcontext expected_context) {
  CUdevice device = -1;
  require(cuDeviceGet(&device, ordinal) == CUDA_SUCCESS, "cuDeviceGet failed");
  CUuuid driver_uuid{};
  require(cuDeviceGetUuid(&driver_uuid, device) == CUDA_SUCCESS,
          "cuDeviceGetUuid failed");
  cudaDeviceProp properties{};
  require(cudaGetDeviceProperties(&properties, ordinal) == cudaSuccess,
          "cudaGetDeviceProperties failed");
  require(std::memcmp(driver_uuid.bytes, properties.uuid.bytes,
                      sizeof(driver_uuid.bytes)) == 0,
          "the runtime and the driver name different devices for one ordinal");

  void *buffer = nullptr;
  require(cudaMalloc(&buffer, 1024) == cudaSuccess, "cudaMalloc failed");
  CUcontext allocation_context = nullptr;
  require(cuPointerGetAttribute(&allocation_context,
                                CU_POINTER_ATTRIBUTE_CONTEXT,
                                reinterpret_cast<CUdeviceptr>(buffer)) ==
              CUDA_SUCCESS,
          "cuPointerGetAttribute failed");
  require(allocation_context == expected_context,
          "a runtime allocation landed outside the current context");
  require(cudaFree(buffer) == cudaSuccess, "cudaFree failed");
}

int main() {
  require(cuInit(0) == CUDA_SUCCESS, "cuInit failed");
  int count = 0;
  require(cuDeviceGetCount(&count) == CUDA_SUCCESS, "cuDeviceGetCount failed");
  require(count >= 2,
          "this test needs two servers with one device each; run it through "
          "test/run_custom_tests.sh");

  // Connection 0's lane has to be live and holding device 0, so that following
  // the driver context is the only thing that can produce the right answer.
  require(cudaSetDevice(0) == cudaSuccess, "cudaSetDevice(0) failed");
  void *warm = nullptr;
  require(cudaMalloc(&warm, 1024) == cudaSuccess, "cudaMalloc failed");
  require(cudaFree(warm) == cudaSuccess, "cudaFree failed");
  require_device(0, "after cudaSetDevice(0)");

  CUdevice second = -1;
  require(cuDeviceGet(&second, 1) == CUDA_SUCCESS, "cuDeviceGet(1) failed");
  CUcontext context = nullptr;
  require(cuCtxCreate_v2(&context, 0, second) == CUDA_SUCCESS,
          "cuCtxCreate failed");
  require_device(1, "with a driver context current on virtual device 1");
  CUcontext current = nullptr;
  require(cuCtxGetCurrent(&current) == CUDA_SUCCESS, "cuCtxGetCurrent failed");
  require(current == context,
          "the context created on virtual device 1 is no longer current");
  require_routes_agree(1, context);

  // The same question asked the other way: a context pushed and popped has to
  // carry the device binding both ways.
  require(cuCtxPopCurrent(&current) == CUDA_SUCCESS, "cuCtxPopCurrent failed");
  require_device(0, "after popping the virtual device 1 context");
  require(cuCtxPushCurrent(context) == CUDA_SUCCESS, "cuCtxPushCurrent failed");
  require_device(1, "after pushing the virtual device 1 context back");
  require(cuCtxPopCurrent(&current) == CUDA_SUCCESS, "cuCtxPopCurrent failed");

  require(cudaSetDevice(1) == cudaSuccess, "cudaSetDevice(1) failed");
  require_device(1, "after cudaSetDevice(1)");
  require(cudaSetDevice(0) == cudaSuccess, "cudaSetDevice(0) failed");
  require_device(0, "after cudaSetDevice(0)");
  require(cuCtxDestroy(context) == CUDA_SUCCESS, "cuCtxDestroy failed");

  std::printf("multi-connection context device test passed\n");
  return 0;
}
