// cuLibraryLoadData is context-free, so a library loaded before any context is
// current cannot answer cuKernelGetParamInfo during snapshot collection. The
// failed probe must not be cached as a zero-parameter layout, or a later
// launch of that kernel sends no arguments and crashes the server.
#include <cuda.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>

#if !defined(CUDA_VERSION) || CUDA_VERSION < 12000
int main() {
  std::printf("SKIP: CUlibrary requires CUDA 12.0 or newer\n");
  return 0;
}
#else

namespace {

struct Payload {
  uint32_t *dst;
  uint32_t value;
};

__global__ void write_payload(Payload payload) {
  *payload.dst = payload.value;
}

static const char kPtx[] =
    ".version 6.4\n"
    ".target sm_52\n"
    ".address_size 64\n"
    ".visible .entry store_pair(\n"
    "  .param .u64 param_ptr,\n"
    "  .param .u32 param_val\n"
    ")\n"
    "{\n"
    "  .reg .u64 %rd1;\n"
    "  .reg .u32 %r1;\n"
    "  ld.param.u64 %rd1, [param_ptr];\n"
    "  ld.param.u32 %r1, [param_val];\n"
    "  st.global.u32 [%rd1], %r1;\n"
    "  ret;\n"
    "}\n";

void check_cu(CUresult result, const char *what) {
  if (result == CUDA_SUCCESS) {
    return;
  }
  const char *name = nullptr;
  cuGetErrorName(result, &name);
  std::fprintf(stderr, "FAIL: %s: %d (%s)\n", what, static_cast<int>(result),
               name ? name : "unknown");
  std::exit(EXIT_FAILURE);
}

void check_rt(cudaError_t result, const char *what) {
  if (result == cudaSuccess) {
    return;
  }
  std::fprintf(stderr, "FAIL: %s: %s\n", what, cudaGetErrorString(result));
  std::exit(EXIT_FAILURE);
}

} // namespace

int main() {
  check_cu(cuInit(0), "cuInit");
  int count = 0;
  check_cu(cuDeviceGetCount(&count), "cuDeviceGetCount");
  if (count == 0) {
    std::printf("SKIP: no CUDA devices found\n");
    return EXIT_SUCCESS;
  }

  // 1. Load a CUlibrary and look up its kernel before any context exists.
  CUlibrary library = nullptr;
  CUkernel kernel = nullptr;
  check_cu(cuLibraryLoadData(&library, kPtx, nullptr, nullptr, 0, nullptr,
                             nullptr, 0),
           "cuLibraryLoadData before context");
  check_cu(cuLibraryGetKernel(&kernel, library, "store_pair"),
           "cuLibraryGetKernel");

  // 2. Also trigger cudart's lazy library registration after cuInit(0), then
  // launch a struct-by-value runtime kernel.
  cudaStream_t stream = nullptr;
  check_rt(cudaStreamCreate(&stream), "cudaStreamCreate");

  uint32_t *device_word = nullptr;
  check_rt(cudaMalloc(&device_word, 2 * sizeof(uint32_t)), "cudaMalloc");
  check_rt(cudaMemset(device_word, 0, 2 * sizeof(uint32_t)), "cudaMemset");

  Payload payload{device_word, 0x12345678U};
  write_payload<<<1, 1, 0, stream>>>(payload);
  check_rt(cudaGetLastError(), "write_payload launch");

  // 3. Now that the primary context is current, launch the pre-context CUkernel
  // directly through cuLaunchKernel and verify its parameter layout was queried.
  size_t offset0 = 99, size0 = 0, offset1 = 99, size1 = 0;
  check_cu(cuKernelGetParamInfo(kernel, 0, &offset0, &size0),
           "cuKernelGetParamInfo(0)");
  check_cu(cuKernelGetParamInfo(kernel, 1, &offset1, &size1),
           "cuKernelGetParamInfo(1)");
  if (size0 != sizeof(CUdeviceptr) || size1 != sizeof(uint32_t)) {
    std::fprintf(stderr, "FAIL: unexpected param sizes %zu, %zu\n", size0,
                 size1);
    return EXIT_FAILURE;
  }

  CUdeviceptr second_ptr =
      reinterpret_cast<CUdeviceptr>(device_word + 1);
  uint32_t second_val = 0x9abcdef0U;
  void *args[] = {&second_ptr, &second_val};
  check_cu(cuLaunchKernel(reinterpret_cast<CUfunction>(kernel), 1, 1, 1, 1, 1,
                          1, 0, stream, args, nullptr),
           "cuLaunchKernel(CUkernel)");
  check_rt(cudaStreamSynchronize(stream), "cudaStreamSynchronize");

  uint32_t host_words[2] = {};
  check_rt(cudaMemcpy(host_words, device_word, sizeof(host_words),
                      cudaMemcpyDeviceToHost),
           "cudaMemcpy");
  if (host_words[0] != 0x12345678U || host_words[1] != 0x9abcdef0U) {
    std::fprintf(stderr, "FAIL: got {0x%08x, 0x%08x}, want {0x12345678, 0x9abcdef0}\n",
                 host_words[0], host_words[1]);
    return EXIT_FAILURE;
  }

  check_rt(cudaFree(device_word), "cudaFree");
  check_rt(cudaStreamDestroy(stream), "cudaStreamDestroy");
  check_cu(cuLibraryUnload(library), "cuLibraryUnload");
  std::printf("PASS: library loaded before context preserves kernel parameters\n");
  return EXIT_SUCCESS;
}
#endif
