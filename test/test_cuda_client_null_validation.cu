// Regression test for client-side null-pointer and negative-length validation
// across generated and handwritten CUDA Driver API entry points.
#include <cuda.h>

#include <cstdio>
#include <cstdlib>

namespace {

bool expect_status(CUresult got, CUresult want, const char *what) {
  if (got == want) {
    return true;
  }
  std::fprintf(stderr, "FAIL: %s: got %d, want %d\n", what,
               static_cast<int>(got), static_cast<int>(want));
  return false;
}

} // namespace

int main() {
  if (cuInit(0) != CUDA_SUCCESS) {
    std::fprintf(stderr, "FAIL: cuInit\n");
    return EXIT_FAILURE;
  }
  int count = 0;
  if (cuDeviceGetCount(&count) != CUDA_SUCCESS || count == 0) {
    std::printf("SKIP: no CUDA devices found\n");
    return EXIT_SUCCESS;
  }

  CUdevice dev = 0;
  if (cuDeviceGet(&dev, 0) != CUDA_SUCCESS) {
    return EXIT_FAILURE;
  }

  int failures = 0;

  // Device & primary context entry points
  char name[256] = {};
  failures += !expect_status(cuDeviceGetName(nullptr, 64, dev),
                             CUDA_ERROR_INVALID_VALUE,
                             "cuDeviceGetName(nullptr)");

  failures += !expect_status(cuDevicePrimaryCtxRetain(nullptr, dev),
                             CUDA_ERROR_INVALID_VALUE,
                             "cuDevicePrimaryCtxRetain(nullptr)");
  unsigned int pflags = 0;
  int pactive = 0;
  failures += !expect_status(cuDevicePrimaryCtxGetState(dev, nullptr, nullptr),
                             CUDA_ERROR_INVALID_VALUE,
                             "cuDevicePrimaryCtxGetState(null, null)");
  failures += !expect_status(cuDevicePrimaryCtxGetState(dev, &pflags, nullptr),
                             CUDA_ERROR_INVALID_VALUE,
                             "cuDevicePrimaryCtxGetState(&flags, null)");
  failures += !expect_status(cuDevicePrimaryCtxGetState(dev, nullptr, &pactive),
                             CUDA_ERROR_INVALID_VALUE,
                             "cuDevicePrimaryCtxGetState(null, &active)");

  CUcontext ctx = nullptr;
  if (!expect_status(cuDevicePrimaryCtxRetain(&ctx, dev), CUDA_SUCCESS,
                     "cuDevicePrimaryCtxRetain(valid)") ||
      !expect_status(cuCtxSetCurrent(ctx), CUDA_SUCCESS, "cuCtxSetCurrent")) {
    return EXIT_FAILURE;
  }

  // Context, memory, stream, event, mempool, VMM, and graph null checks
  failures += !expect_status(cuCtxGetDevice(nullptr), CUDA_ERROR_INVALID_VALUE,
                             "cuCtxGetDevice(nullptr)");
  failures += !expect_status(cuCtxGetFlags(nullptr), CUDA_ERROR_INVALID_VALUE,
                             "cuCtxGetFlags(nullptr)");
  failures += !expect_status(cuCtxGetLimit(nullptr, CU_LIMIT_STACK_SIZE),
                             CUDA_ERROR_INVALID_VALUE,
                             "cuCtxGetLimit(nullptr)");
  failures += !expect_status(cuCtxGetCacheConfig(nullptr),
                             CUDA_ERROR_INVALID_VALUE,
                             "cuCtxGetCacheConfig(nullptr)");

  size_t pitch = 0;
  CUdeviceptr dptr = 0;
  failures += !expect_status(cuMemAlloc_v2(nullptr, 64),
                             CUDA_ERROR_INVALID_VALUE, "cuMemAlloc_v2(nullptr)");
  failures += !expect_status(cuMemAllocPitch_v2(nullptr, &pitch, 16, 4, 4),
                             CUDA_ERROR_INVALID_VALUE,
                             "cuMemAllocPitch_v2(nullptr, &pitch)");
  failures += !expect_status(cuMemAllocPitch_v2(&dptr, nullptr, 16, 4, 4),
                             CUDA_ERROR_INVALID_VALUE,
                             "cuMemAllocPitch_v2(&dptr, nullptr)");

  failures += !expect_status(cuStreamCreate(nullptr, 0),
                             CUDA_ERROR_INVALID_VALUE,
                             "cuStreamCreate(nullptr)");
  failures += !expect_status(cuStreamCreateWithPriority(nullptr, 0, 0),
                             CUDA_ERROR_INVALID_VALUE,
                             "cuStreamCreateWithPriority(nullptr)");

  CUstream stream = nullptr;
  if (expect_status(cuStreamCreate(&stream, CU_STREAM_NON_BLOCKING),
                    CUDA_SUCCESS, "cuStreamCreate(valid)")) {
    failures += !expect_status(cuStreamGetFlags(stream, nullptr),
                               CUDA_ERROR_INVALID_VALUE,
                               "cuStreamGetFlags(nullptr)");
    failures += !expect_status(cuStreamGetPriority(stream, nullptr),
                               CUDA_ERROR_INVALID_VALUE,
                               "cuStreamGetPriority(nullptr)");
    failures += !expect_status(cuStreamGetCtx(stream, nullptr),
                               CUDA_ERROR_INVALID_VALUE,
                               "cuStreamGetCtx(nullptr)");
    cuStreamDestroy_v2(stream);
  } else {
    ++failures;
  }

  failures += !expect_status(cuDeviceGetDefaultMemPool(nullptr, dev),
                             CUDA_ERROR_INVALID_VALUE,
                             "cuDeviceGetDefaultMemPool(nullptr)");
  CUmemPoolProps pool_props = {};
  pool_props.allocType = CU_MEM_ALLOCATION_TYPE_PINNED;
  pool_props.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
  pool_props.location.id = dev;
  failures += !expect_status(cuMemPoolCreate(nullptr, &pool_props),
                             CUDA_ERROR_INVALID_VALUE,
                             "cuMemPoolCreate(nullptr, &props)");
  CUmemoryPool pool = nullptr;
  failures += !expect_status(cuMemPoolCreate(&pool, nullptr),
                             CUDA_ERROR_INVALID_VALUE,
                             "cuMemPoolCreate(&pool, nullptr)");

  CUmemAllocationProp vmm_prop = {};
  vmm_prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
  vmm_prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
  vmm_prop.location.id = dev;
  failures += !expect_status(
      cuMemGetAllocationGranularity(nullptr, &vmm_prop,
                                    CU_MEM_ALLOC_GRANULARITY_MINIMUM),
      CUDA_ERROR_INVALID_VALUE, "cuMemGetAllocationGranularity(nullptr)");

  failures += !expect_status(cuGraphCreate(nullptr, 0),
                             CUDA_ERROR_INVALID_VALUE,
                             "cuGraphCreate(nullptr)");

  // A negative length must not drop the server connection: subsequent calls
  // must continue to succeed.
  (void)cuDeviceGetName(name, -1, dev);
  failures += !expect_status(cuDeviceGetName(name, sizeof(name), dev),
                             CUDA_SUCCESS, "cuDeviceGetName after negative len");

  cuDevicePrimaryCtxRelease(dev);
  if (failures != 0) {
    std::fprintf(stderr, "FAIL: %d CUDA null-validation checks failed\n",
                 failures);
    return EXIT_FAILURE;
  }
  std::printf("PASS: CUDA client null-pointer validation matches native\n");
  return EXIT_SUCCESS;
}
