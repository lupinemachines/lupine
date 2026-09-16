// Exercises the nvJitLink shim end to end against the remote device: two NVRTC
// LTO IR units, one added as data and one as a file the client names, linked to
// PTX and to a CUDA binary, both loaded through the driver API and launched;
// the same link repeated from the file only; the logs of a missing file and of
// an input -ptx rejects; version; and, when two GPUs are present, a link for
// the second. The calls go through the header's inline wrappers, so the
// release-suffixed names are what the program imports.
#include <cuda.h>
#include <cuda_runtime.h>
#include <nvJitLink.h>
#include <nvrtc.h>

#include <unistd.h>

#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

#define CHECK_CUDA(call)                                                       \
  do {                                                                         \
    cudaError_t status = (call);                                               \
    if (status != cudaSuccess) {                                               \
      fprintf(stderr, "%s failed: %s\n", #call, cudaGetErrorName(status));     \
      return 1;                                                                \
    }                                                                          \
  } while (0)

#define CHECK_DRV(call)                                                        \
  do {                                                                         \
    CUresult status = (call);                                                  \
    if (status != CUDA_SUCCESS) {                                              \
      fprintf(stderr, "%s failed: %d\n", #call, static_cast<int>(status));     \
      return 1;                                                                \
    }                                                                          \
  } while (0)

#define CHECK_LINK(call)                                                       \
  do {                                                                         \
    nvJitLinkResult status = (call);                                           \
    if (status != NVJITLINK_SUCCESS) {                                         \
      fprintf(stderr, "%s failed: %d\n", #call, static_cast<int>(status));     \
      return 1;                                                                \
    }                                                                          \
  } while (0)

#define EXPECT(condition)                                                      \
  do {                                                                         \
    if (!(condition)) {                                                        \
      fprintf(stderr, "expectation failed: %s\n", #condition);                 \
      return 1;                                                                \
    }                                                                          \
  } while (0)

static const char *kKernel = R"(
extern "C" __device__ float twice(float x);
extern "C" __global__ void scale(float *x, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    x[i] = twice(x[i]) + 1.0f;
  }
}
)";
static const char *kHelper = R"(
extern "C" __device__ float twice(float x) { return 2.0f * x; }
)";
static const int kCount = 1000;

static int lto_ir(const char *source, const char *name, const std::string &arch,
                  bool lto, std::vector<char> *out) {
  nvrtcProgram prog = nullptr;
  if (nvrtcCreateProgram(&prog, source, name, 0, nullptr, nullptr) !=
      NVRTC_SUCCESS) {
    return 1;
  }
  const std::string gpu = "--gpu-architecture=sm_" + arch;
  const char *options[] = {gpu.c_str(), "-dlto"};
  const nvrtcResult compiled = nvrtcCompileProgram(prog, lto ? 2 : 1, options);
  if (compiled != NVRTC_SUCCESS) {
    size_t size = 0;
    nvrtcGetProgramLogSize(prog, &size);
    std::string log(size, '\0');
    nvrtcGetProgramLog(prog, &log[0]);
    fprintf(stderr, "%s: %s\n", name, log.c_str());
    return 1;
  }
  size_t size = 0;
  if (lto) {
    nvrtcGetLTOIRSize(prog, &size);
    out->resize(size);
    nvrtcGetLTOIR(prog, out->data());
  } else {
    nvrtcGetCUBINSize(prog, &size);
    out->resize(size);
    nvrtcGetCUBIN(prog, out->data());
  }
  nvrtcDestroyProgram(&prog);
  return size == 0;
}

static std::string log_of(nvJitLinkHandle link, bool error) {
  size_t size = 0;
  if ((error ? nvJitLinkGetErrorLogSize(link, &size)
             : nvJitLinkGetInfoLogSize(link, &size)) != NVJITLINK_SUCCESS) {
    return "<no log>";
  }
  // The library writes its terminator even for an empty log.
  std::string log(size + 1, 'Z');
  if (error) {
    nvJitLinkGetErrorLog(link, &log[0]);
  } else {
    nvJitLinkGetInfoLog(link, &log[0]);
  }
  log.resize(strlen(log.c_str()));
  return log;
}

// Loads an image on the device's primary context and launches scale over
// device memory, checking every element.
static int launch(int device, const std::vector<char> &image,
                  const char *what) {
  CHECK_CUDA(cudaSetDevice(device));
  CUdevice dev = 0;
  CUcontext ctx = nullptr;
  CHECK_DRV(cuDeviceGet(&dev, device));
  CHECK_DRV(cuDevicePrimaryCtxRetain(&ctx, dev));
  CHECK_DRV(cuCtxSetCurrent(ctx));
  CUmodule module = nullptr;
  const CUresult loaded = cuModuleLoadData(&module, image.data());
  if (loaded == CUDA_ERROR_UNSUPPORTED_PTX_VERSION) {
    printf("%s: image newer than the device's driver, load skipped\n", what);
    return 0;
  }
  CHECK_DRV(loaded);
  CUfunction function = nullptr;
  CHECK_DRV(cuModuleGetFunction(&function, module, "scale"));
  std::vector<float> host(kCount);
  for (int i = 0; i < kCount; ++i) {
    host[i] = static_cast<float>(i) * 0.25f;
  }
  float *data = nullptr;
  CHECK_CUDA(cudaMalloc(&data, kCount * sizeof(float)));
  CHECK_CUDA(cudaMemcpy(data, host.data(), kCount * sizeof(float),
                        cudaMemcpyHostToDevice));
  int n = kCount;
  void *params[] = {&data, &n};
  const unsigned block = 256;
  CHECK_DRV(cuLaunchKernel(function, (kCount + block - 1) / block, 1, 1, block,
                           1, 1, 0, nullptr, params, nullptr));
  CHECK_DRV(cuCtxSynchronize());
  std::vector<float> result(kCount);
  CHECK_CUDA(cudaMemcpy(result.data(), data, kCount * sizeof(float),
                        cudaMemcpyDeviceToHost));
  for (int i = 0; i < kCount; ++i) {
    EXPECT(result[i] == host[i] * 2.0f + 1.0f);
  }
  CHECK_CUDA(cudaFree(data));
  CHECK_DRV(cuModuleUnload(module));
  CHECK_DRV(cuDevicePrimaryCtxRelease(dev));
  printf("%s: passed on device %d\n", what, device);
  return 0;
}

static std::string arch_of(int device) {
  struct cudaDeviceProp props;
  if (cudaGetDeviceProperties(&props, device) != cudaSuccess) {
    return "";
  }
  return std::to_string(props.major) + std::to_string(props.minor);
}

static int write_file(const std::string &path, const std::vector<char> &bytes) {
  std::ofstream out(path, std::ios::binary);
  out.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
  return out ? 0 : 1;
}

// Links the kernel with the helper, taking the helper from the named client
// file when one is given, to PTX or to a CUDA binary.
static int link(const std::string &arch, const std::vector<char> &kernel,
                const std::vector<char> &helper, const char *helper_file,
                bool ptx, std::vector<char> *image) {
  const std::string arch_option = "-arch=sm_" + arch;
  const char *options[] = {arch_option.c_str(), "-lto", "-ptx"};
  nvJitLinkHandle link = nullptr;
  CHECK_LINK(nvJitLinkCreate(&link, ptx ? 3 : 2, options));
  CHECK_LINK(nvJitLinkAddData(link, NVJITLINK_INPUT_LTOIR, kernel.data(),
                              kernel.size(), "kernel"));
  if (helper_file != nullptr) {
    CHECK_LINK(nvJitLinkAddFile(link, NVJITLINK_INPUT_LTOIR, helper_file));
  } else {
    CHECK_LINK(nvJitLinkAddData(link, NVJITLINK_INPUT_LTOIR, helper.data(),
                                helper.size(), nullptr));
  }
  const nvJitLinkResult completed = nvJitLinkComplete(link);
  if (completed != NVJITLINK_SUCCESS) {
    fprintf(stderr, "link failed: %d\n%s\n", static_cast<int>(completed),
            log_of(link, true).c_str());
    return 1;
  }
  size_t size = 0;
  if (ptx) {
    CHECK_LINK(nvJitLinkGetLinkedPtxSize(link, &size));
    EXPECT(size > 0);
    // The library writes exactly the size it reported.
    image->assign(size + 4, 'Z');
    CHECK_LINK(nvJitLinkGetLinkedPtx(link, image->data()));
  } else {
    CHECK_LINK(nvJitLinkGetLinkedCubinSize(link, &size));
    EXPECT(size > 0);
    image->assign(size + 4, 'Z');
    CHECK_LINK(nvJitLinkGetLinkedCubin(link, image->data()));
  }
  EXPECT((*image)[size] == 'Z');
  image->resize(size);
  EXPECT(log_of(link, true).empty());
  log_of(link, false);
  CHECK_LINK(nvJitLinkDestroy(&link));
  EXPECT(link == nullptr);
  return 0;
}

static int test_links(int device) {
  const std::string arch = arch_of(device);
  EXPECT(!arch.empty());
  std::vector<char> kernel, helper, from_data, from_file, cubin;
  if (lto_ir(kKernel, "kernel.cu", arch, true, &kernel) ||
      lto_ir(kHelper, "helper.cu", arch, true, &helper)) {
    return 1;
  }
  char path[] = "/tmp/lupine-nvjitlink-XXXXXX";
  const int fd = mkstemp(path);
  EXPECT(fd >= 0);
  close(fd);
  if (write_file(path, helper) ||
      link(arch, kernel, helper, nullptr, true, &from_data) ||
      link(arch, kernel, helper, path, true, &from_file) ||
      link(arch, kernel, helper, path, false, &cubin)) {
    unlink(path);
    return 1;
  }
  unlink(path);
  // A file's bytes link the same as the data read from it.
  EXPECT(from_data == from_file);
  if (launch(device, from_data, "ptx") || launch(device, cubin, "cubin")) {
    return 1;
  }
  return 0;
}

static int test_errors(const std::string &arch, unsigned major,
                       unsigned minor) {
  const std::string arch_option = "-arch=sm_" + arch;
  const char *options[] = {arch_option.c_str(), "-lto", "-ptx"};
  nvJitLinkHandle link = nullptr;
  CHECK_LINK(nvJitLinkCreate(&link, 3, options));
  const char *missing = "/nonexistent/lupine-nvjitlink-missing.ltoir";
  EXPECT(nvJitLinkAddFile(link, NVJITLINK_INPUT_LTOIR, missing) ==
         NVJITLINK_ERROR_INVALID_INPUT);
  std::string log = log_of(link, true);
  // Releases before 12.8 log nothing for a missing file.
  if (major > 12 || (major == 12 && minor >= 8)) {
    EXPECT(log.find("lupine-nvjitlink-missing.ltoir") != std::string::npos);
  }
  CHECK_LINK(nvJitLinkDestroy(&link));

  // -ptx accepts only LTO IR, so a CUDA binary input fails the link. 12.4
  // already rejects it when it is added; later releases fail at Complete.
  std::vector<char> binary;
  if (lto_ir(kHelper, "helper.cu", arch, false, &binary)) {
    return 1;
  }
  CHECK_LINK(nvJitLinkCreate(&link, 3, options));
  nvJitLinkAddData(link, NVJITLINK_INPUT_CUBIN, binary.data(), binary.size(),
                   "helper");
  EXPECT(nvJitLinkComplete(link) != NVJITLINK_SUCCESS);
  log = log_of(link, true);
  EXPECT(!log.empty());
  size_t size = 1;
  EXPECT(nvJitLinkGetLinkedPtxSize(link, &size) != NVJITLINK_SUCCESS);
  CHECK_LINK(nvJitLinkDestroy(&link));
  printf("errors: passed (%s)\n", log.substr(0, log.find('\n')).c_str());
  return 0;
}

int main() {
  int devices = 0;
  CHECK_CUDA(cudaGetDeviceCount(&devices));
  EXPECT(devices > 0);
  CHECK_DRV(cuInit(0));
  CHECK_CUDA(cudaSetDevice(0));
  unsigned major = 0, minor = 0;
  CHECK_LINK(nvJitLinkVersion(&major, &minor));
  EXPECT(major >= 12);
  printf("nvJitLink %u.%u\n", major, minor);
  if (test_links(0) || test_errors(arch_of(0), major, minor)) {
    return 1;
  }
  if (devices > 1) {
    if (test_links(1)) {
      return 1;
    }
  } else {
    printf("multi-device: skipped, one device\n");
  }
  printf("nvjitlink shim: all checks passed\n");
  return 0;
}
