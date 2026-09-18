// cuStreamGetDevice has to name the stream's device by the client's virtual
// ordinal, not by the ordinal its server knows it as.
// LUPINE_TEST_CONNECTIONS 2
#include <cuda.h>

#include <cstdio>
#include <cstdlib>
#include <vector>

static void require(bool condition, const char *message) {
  if (!condition) {
    std::fprintf(stderr, "%s\n", message);
    std::exit(1);
  }
}

static void expect_device(CUstream stream, int expected, const char *where) {
  CUdevice reported = -1;
  CUresult status = cuStreamGetDevice(stream, &reported);
  if (status != CUDA_SUCCESS) {
    const char *name = nullptr;
    cuGetErrorName(status, &name);
    std::fprintf(stderr, "cuStreamGetDevice (%s) failed: %d (%s)\n", where,
                 static_cast<int>(status), name == nullptr ? "unknown" : name);
    std::exit(1);
  }
  if (static_cast<int>(reported) != expected) {
    std::fprintf(stderr, "cuStreamGetDevice (%s) reported %d, want %d\n", where,
                 static_cast<int>(reported), expected);
    std::exit(1);
  }
}

int main() {
  require(cuInit(0) == CUDA_SUCCESS, "cuInit failed");
  int count = 0;
  require(cuDeviceGetCount(&count) == CUDA_SUCCESS, "cuDeviceGetCount failed");
  require(count >= 1, "no CUDA devices visible");
  if (count < 2) {
    std::printf("only one virtual device is visible: the ordinal a server "
                "answers with cannot differ from the client's here\n");
  }

  std::vector<CUcontext> contexts(count, nullptr);
  std::vector<CUstream> streams(count, nullptr);
  for (int ordinal = 0; ordinal < count; ++ordinal) {
    CUdevice device = -1;
    require(cuDeviceGet(&device, ordinal) == CUDA_SUCCESS,
            "cuDeviceGet failed");
    require(cuDevicePrimaryCtxRetain(&contexts[ordinal], device) ==
                CUDA_SUCCESS,
            "cuDevicePrimaryCtxRetain failed");
    require(cuCtxSetCurrent(contexts[ordinal]) == CUDA_SUCCESS,
            "cuCtxSetCurrent failed");
    require(cuStreamCreate(&streams[ordinal], CU_STREAM_NON_BLOCKING) ==
                CUDA_SUCCESS,
            "cuStreamCreate failed");
    expect_device(streams[ordinal], ordinal, "its own context current");
    expect_device(nullptr, ordinal, "the default stream");
    expect_device(CU_STREAM_LEGACY, ordinal, "the legacy stream");
    expect_device(CU_STREAM_PER_THREAD, ordinal, "the per-thread stream");
  }

  // The device belongs to the stream, not to the asking thread: with the first
  // context current every stream still has to name the ordinal it was made on.
  require(cuCtxSetCurrent(contexts[0]) == CUDA_SUCCESS, "cuCtxSetCurrent failed");
  for (int ordinal = 0; ordinal < count; ++ordinal) {
    expect_device(streams[ordinal], ordinal, "another context current");
  }

  require(cuStreamGetDevice(streams[0], nullptr) == CUDA_ERROR_INVALID_VALUE,
          "cuStreamGetDevice accepted a null out parameter");

  for (int ordinal = 0; ordinal < count; ++ordinal) {
    require(cuStreamDestroy(streams[ordinal]) == CUDA_SUCCESS,
            "cuStreamDestroy failed");
    CUdevice device = -1;
    require(cuDeviceGet(&device, ordinal) == CUDA_SUCCESS,
            "cuDeviceGet failed");
    require(cuDevicePrimaryCtxRelease(device) == CUDA_SUCCESS,
            "cuDevicePrimaryCtxRelease failed");
  }

  std::printf("cuStreamGetDevice named the client's ordinal for %d stream(s)\n",
              count);
  return 0;
}
