// Driver APIs whose annotations once failed to parse, which left the shim
// without the symbol at all.
#include <cuda.h>
#include <cuda_runtime.h>

#include <cstdio>
#include <cstring>
#include <dlfcn.h>
#include <vector>

#pragma GCC diagnostic ignored "-Wdeprecated-declarations"

#define CHECK_CUDA(call)                                                       \
  do {                                                                         \
    cudaError_t status = (call);                                               \
    if (status != cudaSuccess) {                                               \
      fprintf(stderr, "%s failed: %s\n", #call, cudaGetErrorName(status));     \
      return 1;                                                                \
    }                                                                          \
  } while (0)

#define CHECK_DRIVER(call)                                                     \
  do {                                                                         \
    CUresult status = (call);                                                  \
    if (status != CUDA_SUCCESS) {                                              \
      const char *name = nullptr;                                              \
      cuGetErrorName(status, &name);                                           \
      fprintf(stderr, "%s failed: %s\n", #call, name ? name : "?");            \
      return 1;                                                                \
    }                                                                          \
  } while (0)

#ifdef cuGetProcAddress
#undef cuGetProcAddress
#endif
extern "C" CUresult cuGetProcAddress(const char *, void **, int, cuuint64_t);

__global__ void store(int *out, int value) { *out = value; }

static int resolve(void *library, const char *proc_name,
                   const char *symbol_name, void **function) {
  void *symbol = dlsym(library, symbol_name);
  *function = nullptr;
  CUresult result = cuGetProcAddress(proc_name, function,
                                   CUDA_VERSION < 12000 ? CUDA_VERSION : 12000, 0);
  if (symbol == nullptr || result != CUDA_SUCCESS || *function == nullptr) {
    fprintf(stderr, "%s: dlsym %p, cuGetProcAddress %d\n", proc_name, symbol,
            result);
    return 1;
  }
  return 0;
}

int main() {
  CHECK_CUDA(cudaSetDevice(0));
  CHECK_DRIVER(cuInit(0));
  void *library = dlopen("libcuda.so.1", RTLD_NOW | RTLD_NOLOAD);
  if (library == nullptr) {
    fprintf(stderr, "libcuda.so.1 is not loaded\n");
    return 1;
  }

  using AtoHAsync = CUresult (*)(void *, CUarray, size_t, size_t, CUstream);
  using HandleForRange = CUresult (*)(void *, CUdeviceptr, size_t,
                                      CUmemRangeHandleType, unsigned long long);
  using ParamSetv = CUresult (*)(CUfunction, int, void *, unsigned int);
  void *atoh_async = nullptr;
  void *handle_for_range = nullptr;
  void *param_setv = nullptr;
  if (resolve(library, "cuMemcpyAtoHAsync", "cuMemcpyAtoHAsync_v2",
              &atoh_async) ||
      resolve(library, "cuMemGetHandleForAddressRange",
              "cuMemGetHandleForAddressRange", &handle_for_range) ||
      resolve(library, "cuParamSetv", "cuParamSetv", &param_setv)) {
    return 1;
  }

  const size_t count = 4096;
  std::vector<unsigned char> expected(count);
  for (size_t i = 0; i < count; ++i) {
    expected[i] = static_cast<unsigned char>(i * 7 + 3);
  }
  CUdeviceptr staging = 0;
  CHECK_DRIVER(cuMemAlloc(&staging, 2 << 20));
  CHECK_DRIVER(cuMemcpyHtoD(staging, expected.data(), count));
  CUDA_ARRAY_DESCRIPTOR descriptor{};
  descriptor.Width = count;
  descriptor.Height = 0;
  descriptor.Format = CU_AD_FORMAT_UNSIGNED_INT8;
  descriptor.NumChannels = 1;
  CUarray array = nullptr;
  CHECK_DRIVER(cuArrayCreate(&array, &descriptor));
  CHECK_DRIVER(cuMemcpyDtoA(array, 0, staging, count));
  CUstream stream = nullptr;
  CHECK_DRIVER(cuStreamCreate(&stream, CU_STREAM_NON_BLOCKING));
  std::vector<unsigned char> actual(count, 0);
  CHECK_DRIVER(reinterpret_cast<AtoHAsync>(atoh_async)(actual.data(), array, 16,
                                                      count - 16, stream));
  CHECK_DRIVER(cuStreamSynchronize(stream));
  if (std::memcmp(actual.data(), expected.data() + 16, count - 16) != 0) {
    fprintf(stderr, "cuMemcpyAtoHAsync returned the wrong bytes\n");
    return 1;
  }

  int fd = -1;
  CUresult range_result = reinterpret_cast<HandleForRange>(handle_for_range)(
      &fd, staging, 2 << 20, CU_MEM_RANGE_HANDLE_TYPE_DMA_BUF_FD, 0);
  if (range_result != CUDA_SUCCESS && range_result != CUDA_ERROR_NOT_SUPPORTED &&
      range_result != CUDA_ERROR_INVALID_VALUE) {
    fprintf(stderr, "cuMemGetHandleForAddressRange returned %d\n",
            range_result);
    return 1;
  }
  printf("cuMemGetHandleForAddressRange: %d\n", range_result);

  CUfunction store_function = nullptr;
  CHECK_CUDA(cudaGetFuncBySymbol(reinterpret_cast<cudaFunction_t *>(
                                     &store_function),
                                 (const void *)store));
  int *out = reinterpret_cast<int *>(staging);
  CHECK_DRIVER(cuParamSetSize(store_function, sizeof(out) + sizeof(int)));
  CHECK_DRIVER(reinterpret_cast<ParamSetv>(param_setv)(store_function, 0, &out,
                                                      sizeof(out)));
  CHECK_DRIVER(cuParamSeti(store_function, sizeof(out), 1234));
  CHECK_DRIVER(cuFuncSetBlockShape(store_function, 1, 1, 1));
  CHECK_DRIVER(cuLaunchGrid(store_function, 1, 1));
  CHECK_DRIVER(cuCtxSynchronize());
  int stored = 0;
  CHECK_DRIVER(cuMemcpyDtoH(&stored, staging, sizeof(stored)));
  if (stored != 1234) {
    fprintf(stderr, "cuParamSetv launch stored %d\n", stored);
    return 1;
  }

#if CUDA_VERSION >= 12000
  using EncodeIm2col = CUresult (*)(
      CUtensorMap *, CUtensorMapDataType, cuuint32_t, void *,
      const cuuint64_t *, const cuuint64_t *, const int *, const int *,
      cuuint32_t, cuuint32_t, const cuuint32_t *, CUtensorMapInterleave,
      CUtensorMapSwizzle, CUtensorMapL2promotion, CUtensorMapFloatOOBfill);
  using ReplaceAddress = CUresult (*)(CUtensorMap *, void *);
  void *encode_im2col = nullptr;
  void *replace_address = nullptr;
  if (resolve(library, "cuTensorMapEncodeIm2col", "cuTensorMapEncodeIm2col",
              &encode_im2col) ||
      resolve(library, "cuTensorMapReplaceAddress",
              "cuTensorMapReplaceAddress", &replace_address)) {
    return 1;
  }

  // NHWC floats, innermost dimension first.
  const cuuint64_t dims[4] = {16, 8, 8, 2};
  const cuuint64_t strides[3] = {16 * 4, 16 * 4 * 8, 16 * 4 * 8 * 8};
  const int lower[2] = {-1, -1};
  const int upper[2] = {-1, -1};
  const cuuint32_t element_strides[4] = {1, 1, 1, 1};
  auto encode = [&](CUtensorMap *map, CUdeviceptr address) {
    return reinterpret_cast<EncodeIm2col>(encode_im2col)(
        map, CU_TENSOR_MAP_DATA_TYPE_FLOAT32, 4,
        reinterpret_cast<void *>(address), dims, strides, lower, upper, 16, 64,
        element_strides, CU_TENSOR_MAP_INTERLEAVE_NONE,
        CU_TENSOR_MAP_SWIZZLE_NONE, CU_TENSOR_MAP_L2_PROMOTION_NONE,
        CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  };
  CUtensorMap replaced{};
  CUtensorMap direct{};
  // Tensor maps need sm_90; older GPUs refuse the encode.
  CUresult encode_result = encode(&replaced, staging);
  printf("cuTensorMapEncodeIm2col: %d\n", encode_result);
  if (encode_result == CUDA_ERROR_NOT_SUPPORTED) {
    printf("cuTensorMapReplaceAddress: %d\n",
           reinterpret_cast<ReplaceAddress>(replace_address)(
               &replaced, reinterpret_cast<void *>(staging + 4096)));
  } else {
    CHECK_DRIVER(encode_result);
    CHECK_DRIVER(encode(&direct, staging + 4096));
    if (std::memcmp(&replaced, &direct, sizeof(direct)) == 0) {
      fprintf(stderr, "im2col maps ignore the global address\n");
      return 1;
    }
    CHECK_DRIVER(reinterpret_cast<ReplaceAddress>(replace_address)(
        &replaced, reinterpret_cast<void *>(staging + 4096)));
    if (std::memcmp(&replaced, &direct, sizeof(direct)) != 0) {
      fprintf(stderr, "replaced im2col map differs from a direct encode\n");
      return 1;
    }
  }
#endif

  CHECK_DRIVER(cuStreamDestroy(stream));
  CHECK_DRIVER(cuArrayDestroy(array));
  CHECK_DRIVER(cuMemFree(staging));
  printf("restored driver APIs passed\n");
  return 0;
}
