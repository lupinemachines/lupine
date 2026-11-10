#include <cuda.h>

#include <cstdio>
#include <cstring>
#include <dlfcn.h>
#include <vector>

#define CHECK_CU(call)                                                         \
  do {                                                                         \
    CUresult status = (call);                                                  \
    if (status != CUDA_SUCCESS) {                                              \
      const char *name = nullptr;                                              \
      cuGetErrorName(status, &name);                                           \
      fprintf(stderr, "FAIL %s:%d: %s returned %s (%d)\n", __FILE__, __LINE__, \
              #call, name ? name : "?", static_cast<int>(status));             \
      return 1;                                                                \
    }                                                                          \
  } while (0)

#ifdef cuGetProcAddress
#undef cuGetProcAddress
#endif
extern "C" CUresult cuGetProcAddress(const char *, void **, int, cuuint64_t);

static int verify_symbol(void *library, const char *proc_name,
                         const char *symbol_name) {
  void *sym = dlsym(library, symbol_name);
  void *proc = nullptr;
  CUresult rc = cuGetProcAddress(
      proc_name, &proc, CUDA_VERSION < 12000 ? CUDA_VERSION : 12000, 0);
  if (sym == nullptr || rc != CUDA_SUCCESS || proc == nullptr) {
    fprintf(stderr, "FAIL resolve(%s, %s): dlsym=%p cuGetProcAddress=%d (%p)\n",
            proc_name, symbol_name, sym, static_cast<int>(rc), proc);
    return 1;
  }
  return 0;
}

static int run_on_device(int ordinal) {
  CUdevice dev = 0;
  CHECK_CU(cuDeviceGet(&dev, ordinal));
  CUcontext ctx = nullptr;
  CHECK_CU(cuDevicePrimaryCtxRetain(&ctx, dev));
  CHECK_CU(cuCtxSetCurrent(ctx));

  constexpr size_t kWidth = 1024;
  CUDA_ARRAY_DESCRIPTOR desc = {};
  desc.Width = kWidth;
  desc.Height = 0;
  desc.Format = CU_AD_FORMAT_UNSIGNED_INT8;
  desc.NumChannels = 1;
  CUarray array = nullptr;
  CHECK_CU(cuArrayCreate(&array, &desc));

  std::vector<unsigned char> pageable(kWidth);
  std::vector<unsigned char> actual(kWidth, 0);
  for (size_t i = 0; i < kWidth; ++i) {
    pageable[i] = static_cast<unsigned char>((i * 13 + 5) & 0xff);
  }

  // 1. Synchronous cuMemcpyHtoA from pageable memory (including dstOffset > 0).
  CHECK_CU(cuMemcpyHtoA(array, 0, pageable.data(), kWidth));
  CHECK_CU(cuMemcpyAtoH(actual.data(), array, 0, kWidth));
  if (std::memcmp(actual.data(), pageable.data(), kWidth) != 0) {
    fprintf(stderr, "FAIL dev %d: cuMemcpyHtoA full copy mismatch\n", ordinal);
    return 1;
  }

  std::vector<unsigned char> patch(128);
  for (size_t i = 0; i < patch.size(); ++i) {
    patch[i] = static_cast<unsigned char>(0xa0 + i);
  }
  CHECK_CU(cuMemcpyHtoA(array, 17, patch.data(), patch.size()));
  std::fill(actual.begin(), actual.end(), 0);
  CHECK_CU(cuMemcpyAtoH(actual.data(), array, 17, patch.size()));
  if (std::memcmp(actual.data(), patch.data(), patch.size()) != 0) {
    fprintf(stderr, "FAIL dev %d: cuMemcpyHtoA offset 17 mismatch\n", ordinal);
    return 1;
  }

  // 2. Asynchronous cuMemcpyHtoAAsync from pageable, pinned, and managed host
  // memory.
  CUstream stream = nullptr;
  CHECK_CU(cuStreamCreate(&stream, CU_STREAM_NON_BLOCKING));
  for (size_t i = 0; i < patch.size(); ++i) {
    patch[i] = static_cast<unsigned char>(0x30 + i);
  }
  CHECK_CU(cuMemcpyHtoAAsync(array, 33, patch.data(), patch.size(), stream));
  std::fill(actual.begin(), actual.end(), 0);
  CHECK_CU(cuMemcpyAtoHAsync(actual.data(), array, 33, patch.size(), stream));
  CHECK_CU(cuStreamSynchronize(stream));
  if (std::memcmp(actual.data(), patch.data(), patch.size()) != 0) {
    fprintf(stderr, "FAIL dev %d: cuMemcpyHtoAAsync pageable mismatch\n",
            ordinal);
    return 1;
  }

  unsigned char *pinned = nullptr;
  CHECK_CU(cuMemAllocHost(reinterpret_cast<void **>(&pinned), 256));
  for (size_t i = 0; i < 256; ++i) {
    pinned[i] = static_cast<unsigned char>((i * 7 + 9) & 0xff);
  }
  CHECK_CU(cuMemcpyHtoAAsync(array, 64, pinned, 256, stream));
  std::fill(actual.begin(), actual.end(), 0);
  CHECK_CU(cuMemcpyAtoHAsync(actual.data(), array, 64, 256, stream));
  CHECK_CU(cuStreamSynchronize(stream));
  if (std::memcmp(actual.data(), pinned, 256) != 0) {
    fprintf(stderr, "FAIL dev %d: cuMemcpyHtoAAsync pinned mismatch\n",
            ordinal);
    return 1;
  }

  CUdeviceptr managed = 0;
  CHECK_CU(cuMemAllocManaged(&managed, 256, CU_MEM_ATTACH_GLOBAL));
  auto *managed_bytes = reinterpret_cast<unsigned char *>(managed);
  for (size_t i = 0; i < 256; ++i) {
    managed_bytes[i] = static_cast<unsigned char>((i * 11 + 3) & 0xff);
  }
  CHECK_CU(cuMemcpyHtoA(array, 128, managed_bytes, 256));
  std::fill(actual.begin(), actual.end(), 0);
  CHECK_CU(cuMemcpyAtoH(actual.data(), array, 128, 256));
  if (std::memcmp(actual.data(), managed_bytes, 256) != 0) {
    fprintf(stderr, "FAIL dev %d: cuMemcpyHtoA managed mismatch\n", ordinal);
    return 1;
  }

  // 3. Error cases.
  CUresult rc = cuMemcpyHtoA(nullptr, 0, pageable.data(), 16);
  if (rc != CUDA_ERROR_INVALID_VALUE) {
    fprintf(stderr, "FAIL dev %d: cuMemcpyHtoA(nullptr) returned %d\n", ordinal,
            static_cast<int>(rc));
    return 1;
  }
  rc = cuMemcpyHtoA(array, 0, nullptr, 16);
  if (rc != CUDA_ERROR_INVALID_VALUE) {
    fprintf(stderr, "FAIL dev %d: cuMemcpyHtoA(src=nullptr) returned %d\n",
            ordinal, static_cast<int>(rc));
    return 1;
  }
  rc = cuMemcpyHtoA(array, kWidth - 8, pageable.data(), 16);
  if (rc != CUDA_ERROR_INVALID_VALUE) {
    fprintf(stderr, "FAIL dev %d: cuMemcpyHtoA(oob) returned %d\n", ordinal,
            static_cast<int>(rc));
    return 1;
  }

  CHECK_CU(cuMemFree(managed));
  CHECK_CU(cuMemFreeHost(pinned));
  CHECK_CU(cuStreamDestroy(stream));
  CHECK_CU(cuArrayDestroy(array));
  CHECK_CU(cuDevicePrimaryCtxRelease(dev));
  return 0;
}

int main() {
  CHECK_CU(cuInit(0));
  void *library = dlopen("libcuda.so.1", RTLD_NOW | RTLD_NOLOAD);
  if (library == nullptr) {
    fprintf(stderr, "FAIL: libcuda.so.1 is not loaded\n");
    return 1;
  }
  if (verify_symbol(library, "cuMemcpyHtoA", "cuMemcpyHtoA") ||
      verify_symbol(library, "cuMemcpyHtoA", "cuMemcpyHtoA_v2") ||
      verify_symbol(library, "cuMemcpyHtoA", "cuMemcpyHtoA_v2_ptds") ||
      verify_symbol(library, "cuMemcpyHtoAAsync", "cuMemcpyHtoAAsync") ||
      verify_symbol(library, "cuMemcpyHtoAAsync", "cuMemcpyHtoAAsync_v2") ||
      verify_symbol(library, "cuMemcpyHtoAAsync",
                    "cuMemcpyHtoAAsync_v2_ptsz") ||
      verify_symbol(library, "cuMemcpyAtoH", "cuMemcpyAtoH") ||
      verify_symbol(library, "cuMemcpyAtoH", "cuMemcpyAtoH_v2") ||
      verify_symbol(library, "cuMemcpyAtoH", "cuMemcpyAtoH_v2_ptds") ||
      verify_symbol(library, "cuMemcpyAtoHAsync", "cuMemcpyAtoHAsync") ||
      verify_symbol(library, "cuMemcpyAtoHAsync", "cuMemcpyAtoHAsync_v2") ||
      verify_symbol(library, "cuMemcpyAtoHAsync",
                    "cuMemcpyAtoHAsync_v2_ptsz")) {
    return 1;
  }

  int count = 0;
  CHECK_CU(cuDeviceGetCount(&count));
  for (int i = 0; i < count; ++i) {
    if (run_on_device(i) != 0) {
      return 1;
    }
  }
  printf("PASS\n");
  return 0;
}
