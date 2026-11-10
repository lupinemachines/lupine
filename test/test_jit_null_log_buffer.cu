// Null JIT option arrays or null CU_JIT_*_LOG_BUFFER pointers paired with a
// non-zero log buffer size must return CUDA_ERROR_INVALID_VALUE without
// dereferencing nullptr on either the client or the server, and leave the
// connection healthy for subsequent JIT loads.
#include <cuda.h>

#include <cstdio>
#include <cstdlib>
#include <unistd.h>

namespace {

const char kNopPtx[] = ".version 7.0\n"
                       ".target sm_52\n"
                       ".address_size 64\n"
                       ".visible .entry nop()\n"
                       "{\n"
                       "  ret;\n"
                       "}\n";

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
  int device_count = 0;
  if (cuDeviceGetCount(&device_count) != CUDA_SUCCESS || device_count == 0) {
    std::printf("SKIP: no CUDA devices found\n");
    return EXIT_SUCCESS;
  }

  CUdevice device = 0;
  CUcontext ctx = nullptr;
  if (cuDeviceGet(&device, 0) != CUDA_SUCCESS ||
      cuDevicePrimaryCtxRetain(&ctx, device) != CUDA_SUCCESS ||
      cuCtxSetCurrent(ctx) != CUDA_SUCCESS) {
    std::fprintf(stderr, "FAIL: context setup\n");
    return EXIT_FAILURE;
  }

  char ptx_path[] = "/tmp/lupine_jit_null_XXXXXX";
  int fd = mkstemp(ptx_path);
  if (fd < 0) {
    std::perror("mkstemp");
    return EXIT_FAILURE;
  }
  if (write(fd, kNopPtx, sizeof(kNopPtx) - 1) !=
      static_cast<ssize_t>(sizeof(kNopPtx) - 1)) {
    close(fd);
    unlink(ptx_path);
    return EXIT_FAILURE;
  }
  close(fd);

  int failures = 0;
  CUmodule mod = nullptr;
#if CUDA_VERSION >= 12000
  CUlibrary lib = nullptr;
#endif
  CUlinkState link = nullptr;
  CUjit_option single_opt = CU_JIT_WALL_TIME;
  void *single_val = nullptr;

  // 1. Null option/value arrays when numOptions != 0.
  failures += !expect_status(
      cuModuleLoadDataEx(&mod, kNopPtx, 1, nullptr, &single_val),
      CUDA_ERROR_INVALID_VALUE, "cuModuleLoadDataEx(options=nullptr)");
  failures += !expect_status(
      cuModuleLoadDataEx(&mod, kNopPtx, 1, &single_opt, nullptr),
      CUDA_ERROR_INVALID_VALUE, "cuModuleLoadDataEx(optionValues=nullptr)");
  failures += !expect_status(
      cuLinkCreate(1, nullptr, &single_val, &link), CUDA_ERROR_INVALID_VALUE,
      "cuLinkCreate(options=nullptr)");
  failures += !expect_status(
      cuLinkCreate(1, &single_opt, nullptr, &link), CUDA_ERROR_INVALID_VALUE,
      "cuLinkCreate(optionValues=nullptr)");

  // 2. Null CU_JIT_INFO_LOG_BUFFER / CU_JIT_ERROR_LOG_BUFFER with non-zero
  // buffer size across module, library, and linker entry points.
  const CUjit_option log_kinds[2][2] = {
      {CU_JIT_INFO_LOG_BUFFER, CU_JIT_INFO_LOG_BUFFER_SIZE_BYTES},
      {CU_JIT_ERROR_LOG_BUFFER, CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES},
  };
  for (int k = 0; k < 2; ++k) {
    CUjit_option opts[2] = {log_kinds[k][0], log_kinds[k][1]};
    void *vals[2] = {nullptr, reinterpret_cast<void *>(64)};
    failures += !expect_status(cuModuleLoadDataEx(&mod, kNopPtx, 2, opts, vals),
                               CUDA_ERROR_INVALID_VALUE,
                               "cuModuleLoadDataEx(null log buffer + size=64)");
#if CUDA_VERSION >= 12000
    vals[1] = reinterpret_cast<void *>(64);
    failures += !expect_status(
        cuLibraryLoadData(&lib, kNopPtx, opts, vals, 2, nullptr, nullptr, 0),
        CUDA_ERROR_INVALID_VALUE,
        "cuLibraryLoadData(null log buffer + size=64)");
#endif
    vals[1] = reinterpret_cast<void *>(64);
    failures += !expect_status(cuLinkCreate(2, opts, vals, &link),
                               CUDA_ERROR_INVALID_VALUE,
                               "cuLinkCreate(null log buffer + size=64)");
  }

  // 3. Linker addData / addFile validation and null log buffer handling.
  if (!expect_status(cuLinkCreate(0, nullptr, nullptr, &link), CUDA_SUCCESS,
                     "cuLinkCreate(valid)")) {
    unlink(ptx_path);
    return EXIT_FAILURE;
  }
  failures += !expect_status(
      cuLinkAddData(link, CU_JIT_INPUT_PTX, nullptr, 16, "nop.ptx", 0, nullptr,
                    nullptr),
      CUDA_ERROR_INVALID_VALUE, "cuLinkAddData(data=nullptr, size=16)");
  failures += !expect_status(
      cuLinkAddData(link, CU_JIT_INPUT_PTX, const_cast<char *>(kNopPtx),
                    sizeof(kNopPtx), "nop.ptx", 1, nullptr, &single_val),
      CUDA_ERROR_INVALID_VALUE, "cuLinkAddData(options=nullptr)");
  failures += !expect_status(
      cuLinkAddData(link, CU_JIT_INPUT_PTX, const_cast<char *>(kNopPtx),
                    sizeof(kNopPtx), "nop.ptx", 1, &single_opt, nullptr),
      CUDA_ERROR_INVALID_VALUE, "cuLinkAddData(optionValues=nullptr)");
  failures += !expect_status(
      cuLinkAddFile(link, CU_JIT_INPUT_PTX, nullptr, 0, nullptr, nullptr),
      CUDA_ERROR_INVALID_VALUE, "cuLinkAddFile(path=nullptr)");
  failures += !expect_status(
      cuLinkAddFile(link, CU_JIT_INPUT_PTX, ptx_path, 1, nullptr, &single_val),
      CUDA_ERROR_INVALID_VALUE, "cuLinkAddFile(options=nullptr)");
  failures += !expect_status(
      cuLinkAddFile(link, CU_JIT_INPUT_PTX, ptx_path, 1, &single_opt, nullptr),
      CUDA_ERROR_INVALID_VALUE, "cuLinkAddFile(optionValues=nullptr)");
  for (int k = 0; k < 2; ++k) {
    CUjit_option opts[2] = {log_kinds[k][0], log_kinds[k][1]};
    void *vals[2] = {nullptr, reinterpret_cast<void *>(64)};
    failures += !expect_status(
        cuLinkAddData(link, CU_JIT_INPUT_PTX, const_cast<char *>(kNopPtx),
                      sizeof(kNopPtx), "nop.ptx", 2, opts, vals),
        CUDA_ERROR_INVALID_VALUE,
        "cuLinkAddData(null log buffer + size=64)");
    vals[1] = reinterpret_cast<void *>(64);
    failures += !expect_status(
        cuLinkAddFile(link, CU_JIT_INPUT_PTX, ptx_path, 2, opts, vals),
        CUDA_ERROR_INVALID_VALUE,
        "cuLinkAddFile(null log buffer + size=64)");
  }
  failures += !expect_status(cuLinkDestroy(link), CUDA_SUCCESS, "cuLinkDestroy");

  // 4. Verify the connection and JIT pipeline remain healthy afterwards.
  char info_buf[128] = {};
  CUjit_option ok_opts[2] = {CU_JIT_INFO_LOG_BUFFER,
                             CU_JIT_INFO_LOG_BUFFER_SIZE_BYTES};
  void *ok_vals[2] = {info_buf, reinterpret_cast<void *>(sizeof(info_buf))};
  if (expect_status(cuModuleLoadDataEx(&mod, kNopPtx, 2, ok_opts, ok_vals),
                    CUDA_SUCCESS, "cuModuleLoadDataEx(follow-up valid)")) {
    CUfunction fn = nullptr;
    failures += !expect_status(cuModuleGetFunction(&fn, mod, "nop"),
                               CUDA_SUCCESS, "cuModuleGetFunction(nop)");
    failures += !expect_status(
        cuLaunchKernel(fn, 1, 1, 1, 1, 1, 1, 0, nullptr, nullptr, nullptr),
        CUDA_SUCCESS, "cuLaunchKernel(nop)");
    failures += !expect_status(cuCtxSynchronize(), CUDA_SUCCESS,
                               "cuCtxSynchronize(nop)");
    failures += !expect_status(cuModuleUnload(mod), CUDA_SUCCESS,
                               "cuModuleUnload");
  } else {
    ++failures;
  }

  unlink(ptx_path);
  cuDevicePrimaryCtxRelease(device);

  if (failures != 0) {
    std::fprintf(stderr, "FAIL: %d JIT null-buffer checks failed\n", failures);
    return EXIT_FAILURE;
  }
  std::printf("PASS: null JIT option arrays and log buffers rejected cleanly\n");
  return EXIT_SUCCESS;
}
