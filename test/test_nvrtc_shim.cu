// Exercises the NVRTC shim end to end against the remote device: version and
// architectures, error strings, a templated kernel with an in-memory header and
// name expressions compiled to PTX and to CUBIN, both loaded through the driver
// API and launched, headers included from files beside the program and on an
// include path, the log of a failing compile, LTO IR linked with nvJitLink
// when the client has it, the precompiled header and flow callback calls and,
// when two GPUs are present, a program compiled and launched on the second.
#include <cuda.h>
#include <cuda_runtime.h>
#include <nvrtc.h>

#include <dlfcn.h>
#include <stdlib.h>
#include <sys/stat.h>
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

#define CHECK_NVRTC(call)                                                      \
  do {                                                                         \
    nvrtcResult status = (call);                                               \
    if (status != NVRTC_SUCCESS) {                                             \
      fprintf(stderr, "%s failed: %s\n", #call, nvrtcGetErrorString(status));  \
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

static const char *kSource = R"(
#include "scale_factor.h"
template <typename T>
__global__ void scale(T *x, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    x[i] = x[i] * SCALE_FACTOR;
  }
}
)";
static const char *kHeader = "#define SCALE_FACTOR 3\n";
static const int kCount = 1000;

static std::string program_log(nvrtcProgram prog) {
  size_t size = 0;
  if (nvrtcGetProgramLogSize(prog, &size) != NVRTC_SUCCESS) {
    return "<no log>";
  }
  std::string log(size, '\0');
  nvrtcGetProgramLog(prog, &log[0]);
  return log;
}

// Compiles the kernel with its header and both name expressions.
static int compile(const std::vector<std::string> &options,
                   nvrtcProgram *prog) {
  const char *headers[] = {kHeader};
  const char *includes[] = {"scale_factor.h"};
  CHECK_NVRTC(nvrtcCreateProgram(prog, kSource, "scale.cu", 1, headers,
                                 includes));
  CHECK_NVRTC(nvrtcAddNameExpression(*prog, "scale<float>"));
  CHECK_NVRTC(nvrtcAddNameExpression(*prog, "scale<int>"));
  std::vector<const char *> pointers;
  for (const std::string &option : options) {
    pointers.push_back(option.c_str());
  }
  const nvrtcResult status = nvrtcCompileProgram(
      *prog, static_cast<int>(pointers.size()), pointers.data());
  if (status != NVRTC_SUCCESS) {
    fprintf(stderr, "compile failed: %s\n%s\n", nvrtcGetErrorString(status),
            program_log(*prog).c_str());
    return 1;
  }
  return 0;
}

// Loads an image on the device's primary context and launches scale<float>
// over device memory, checking every element.
static int launch(int device, const std::vector<char> &image,
                  const char *lowered, const char *what) {
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
  CHECK_DRV(cuModuleGetFunction(&function, module, lowered));

  std::vector<float> host(kCount);
  for (int i = 0; i < kCount; ++i) {
    host[i] = static_cast<float>(i) * 0.5f;
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
    EXPECT(result[i] == host[i] * 3.0f);
  }
  CHECK_CUDA(cudaFree(data));
  CHECK_DRV(cuModuleUnload(module));
  CHECK_DRV(cuDevicePrimaryCtxRelease(dev));
  printf("%s: passed on device %d\n", what, device);
  return 0;
}

static int test_version_and_archs() {
  int major = 0, minor = 0;
  CHECK_NVRTC(nvrtcVersion(&major, &minor));
  EXPECT(major >= 11);
  int count = 0;
  CHECK_NVRTC(nvrtcGetNumSupportedArchs(&count));
  EXPECT(count > 0);
  std::vector<int> archs(count, -1);
  CHECK_NVRTC(nvrtcGetSupportedArchs(archs.data()));
  for (int i = 1; i < count; ++i) {
    EXPECT(archs[i - 1] < archs[i]);
  }
  EXPECT(strcmp(nvrtcGetErrorString(NVRTC_ERROR_COMPILATION),
                "NVRTC_ERROR_COMPILATION") == 0);
  // The same status hands out the same string.
  EXPECT(nvrtcGetErrorString(NVRTC_SUCCESS) ==
         nvrtcGetErrorString(NVRTC_SUCCESS));
  printf("version %d.%d, %d architectures %d..%d: passed\n", major, minor,
         count, archs.front(), archs.back());
  return 0;
}

static int test_ptx_and_names(int device) {
  nvrtcProgram prog = nullptr;
  if (compile({"--std=c++17"}, &prog)) {
    return 1;
  }
  const char *float_name = nullptr, *int_name = nullptr, *again = nullptr;
  CHECK_NVRTC(nvrtcGetLoweredName(prog, "scale<float>", &float_name));
  CHECK_NVRTC(nvrtcGetLoweredName(prog, "scale<int>", &int_name));
  CHECK_NVRTC(nvrtcGetLoweredName(prog, "scale<float>", &again));
  EXPECT(strcmp(float_name, int_name) != 0);
  EXPECT(again == float_name);
  EXPECT(nvrtcGetLoweredName(prog, "scale<double>", &again) ==
         NVRTC_ERROR_NAME_EXPRESSION_NOT_VALID);

  size_t size = 0;
  CHECK_NVRTC(nvrtcGetPTXSize(prog, &size));
  EXPECT(size > 1);
  // The library writes exactly the size it reported.
  std::vector<char> ptx(size + 4, 'Z');
  CHECK_NVRTC(nvrtcGetPTX(prog, ptx.data()));
  EXPECT(ptx[size] == 'Z' && ptx[size - 1] == '\0');
  ptx.resize(size);
  const std::string lowered = float_name;
  if (launch(device, ptx, lowered.c_str(), "ptx")) {
    return 1;
  }
  CHECK_NVRTC(nvrtcDestroyProgram(&prog));
  EXPECT(prog == nullptr);
  return 0;
}

static int test_cubin(int device) {
  struct cudaDeviceProp props;
  CHECK_CUDA(cudaGetDeviceProperties(&props, device));
  const std::string arch =
      "-arch=sm_" + std::to_string(props.major) + std::to_string(props.minor);
  nvrtcProgram prog = nullptr;
  if (compile({"--std=c++17", arch}, &prog)) {
    return 1;
  }
  size_t size = 0;
  CHECK_NVRTC(nvrtcGetCUBINSize(prog, &size));
  EXPECT(size > 0);
  std::vector<char> cubin(size);
  CHECK_NVRTC(nvrtcGetCUBIN(prog, cubin.data()));
  const char *lowered = nullptr;
  CHECK_NVRTC(nvrtcGetLoweredName(prog, "scale<float>", &lowered));
  const std::string name = lowered;
  if (launch(device, cubin, name.c_str(), "cubin")) {
    return 1;
  }
  CHECK_NVRTC(nvrtcDestroyProgram(&prog));
  return 0;
}

// The kernel includes a header beside the program, which includes one found on
// an include path; both exist only in this process's filesystem.
static int test_included_files(int device) {
  char root[] = "/tmp/lupine-nvrtc-XXXXXX";
  EXPECT(mkdtemp(root) != nullptr);
  const std::string program_dir = std::string(root) + "/src";
  const std::string include_dir = std::string(root) + "/include";
  EXPECT(mkdir(program_dir.c_str(), 0700) == 0);
  EXPECT(mkdir(include_dir.c_str(), 0700) == 0);
  std::ofstream(program_dir + "/scale_kernel.h") << "#include <factor.h>\n"
      "template <typename T>\n"
      "__global__ void scale(T *x, int n) {\n"
      "  int i = blockIdx.x * blockDim.x + threadIdx.x;\n"
      "  if (i < n) {\n"
      "    x[i] = x[i] * SCALE_FACTOR;\n"
      "  }\n"
      "}\n";
  std::ofstream(include_dir + "/factor.h") << "#define SCALE_FACTOR 3\n";
  const std::string name = program_dir + "/scale.cu";
  const std::string option = "--include-path=" + include_dir;
  const char *options[] = {option.c_str()};

  nvrtcProgram prog = nullptr;
  CHECK_NVRTC(nvrtcCreateProgram(&prog, "#include \"scale_kernel.h\"\n",
                                 name.c_str(), 0, nullptr, nullptr));
  CHECK_NVRTC(nvrtcAddNameExpression(prog, "scale<float>"));
  const nvrtcResult status = nvrtcCompileProgram(prog, 1, options);
  if (status != NVRTC_SUCCESS) {
    fprintf(stderr, "included files compile failed: %s\n%s\n",
            nvrtcGetErrorString(status), program_log(prog).c_str());
    return 1;
  }
  const char *lowered = nullptr;
  CHECK_NVRTC(nvrtcGetLoweredName(prog, "scale<float>", &lowered));
  const std::string function = lowered;
  size_t size = 0;
  CHECK_NVRTC(nvrtcGetPTXSize(prog, &size));
  std::vector<char> ptx(size);
  CHECK_NVRTC(nvrtcGetPTX(prog, ptx.data()));
  CHECK_NVRTC(nvrtcDestroyProgram(&prog));
  unlink((program_dir + "/scale_kernel.h").c_str());
  unlink((include_dir + "/factor.h").c_str());
  rmdir(program_dir.c_str());
  rmdir(include_dir.c_str());
  rmdir(root);
  return launch(device, ptx, function.c_str(), "included files");
}

static int test_compile_error() {
  nvrtcProgram prog = nullptr;
  CHECK_NVRTC(nvrtcCreateProgram(&prog, "__global__ void broken() { oops; }",
                                 nullptr, 0, nullptr, nullptr));
  EXPECT(nvrtcCompileProgram(prog, 0, nullptr) == NVRTC_ERROR_COMPILATION);
  const std::string log = program_log(prog);
  EXPECT(log.find("oops") != std::string::npos);
  size_t size = 0;
  EXPECT(nvrtcGetPTXSize(prog, &size) != NVRTC_SUCCESS || size <= 1);
  CHECK_NVRTC(nvrtcDestroyProgram(&prog));
  printf("compile error log: passed\n");
  return 0;
}

#if CUDA_VERSION >= 12000
using link_create_t = int (*)(void **, unsigned, const char **);
using link_add_t = int (*)(void *, int, const char *, size_t, const char *);
using link_complete_t = int (*)(void *);
using link_size_t = int (*)(void *, size_t *);
using link_get_t = int (*)(void *, char *);
using link_destroy_t = int (*)(void **);

// nvJitLink exports each call under its release name, so the client's library
// is resolved at runtime instead of linked.
static void *link_symbol(void *library, const char *call) {
  const int major = CUDA_VERSION / 1000;
  for (int minor = 9; minor >= 0; --minor) {
    const std::string name = "__nvJitLink" + std::string(call) + "_" +
                             std::to_string(major) + "_" +
                             std::to_string(minor);
    if (void *symbol = dlsym(library, name.c_str())) {
      return symbol;
    }
  }
  return nullptr;
}

static int test_lto(int device) {
  nvrtcProgram prog = nullptr;
  if (compile({"--std=c++17", "-dlto"}, &prog)) {
    return 1;
  }
  size_t size = 0;
  CHECK_NVRTC(nvrtcGetLTOIRSize(prog, &size));
  EXPECT(size > 0);
  std::vector<char> ir(size);
  CHECK_NVRTC(nvrtcGetLTOIR(prog, ir.data()));
  const char *lowered = nullptr;
  CHECK_NVRTC(nvrtcGetLoweredName(prog, "scale<float>", &lowered));
  const std::string name = lowered;
  CHECK_NVRTC(nvrtcDestroyProgram(&prog));

  const std::string soname =
      "libnvJitLink.so." + std::to_string(CUDA_VERSION / 1000);
  void *library = dlopen(soname.c_str(), RTLD_NOW | RTLD_LOCAL);
  const auto create =
      reinterpret_cast<link_create_t>(link_symbol(library, "Create"));
  const auto add = reinterpret_cast<link_add_t>(link_symbol(library, "AddData"));
  const auto complete =
      reinterpret_cast<link_complete_t>(link_symbol(library, "Complete"));
  const auto get_size =
      reinterpret_cast<link_size_t>(link_symbol(library, "GetLinkedPtxSize"));
  const auto get = reinterpret_cast<link_get_t>(link_symbol(library, "GetLinkedPtx"));
  const auto destroy =
      reinterpret_cast<link_destroy_t>(link_symbol(library, "Destroy"));
  if (library == nullptr || !create || !add || !complete || !get_size ||
      !get || !destroy) {
    printf("lto: %zu bytes of IR, nvJitLink unavailable on the client, link "
           "skipped\n",
           size);
    return 0;
  }
  struct cudaDeviceProp props;
  CHECK_CUDA(cudaGetDeviceProperties(&props, device));
  const std::string arch =
      "-arch=sm_" + std::to_string(props.major) + std::to_string(props.minor);
  void *link = nullptr;
  const char *options[] = {arch.c_str(), "-lto", "-ptx"};
  EXPECT(create(&link, 3, options) == 0);
  EXPECT(add(link, 3 /* NVJITLINK_INPUT_LTOIR */, ir.data(), ir.size(),
             "scale") == 0);
  EXPECT(complete(link) == 0);
  size_t linked_size = 0;
  EXPECT(get_size(link, &linked_size) == 0 && linked_size > 0);
  std::vector<char> linked(linked_size);
  EXPECT(get(link, linked.data()) == 0);
  destroy(&link);
  return launch(device, linked, name.c_str(), "lto");
}
#endif

#if CUDA_VERSION >= 12080
static int flow_callback(void *, void *) { return 0; }

static int test_pch_and_flow() {
  size_t heap = 0;
  CHECK_NVRTC(nvrtcGetPCHHeapSize(&heap));
  CHECK_NVRTC(nvrtcSetPCHHeapSize(heap));
  nvrtcProgram prog = nullptr;
  CHECK_NVRTC(nvrtcCreateProgram(&prog, "__global__ void k() {}", "k.cu", 0,
                                 nullptr, nullptr));
  // The native library records the callback; a remote compiler cannot call a
  // client function, so the shim refuses it.
  const nvrtcResult callback = nvrtcSetFlowCallback(prog, flow_callback, &heap);
  EXPECT(callback == NVRTC_SUCCESS || callback == NVRTC_ERROR_INVALID_INPUT);
  CHECK_NVRTC(nvrtcCompileProgram(prog, 0, nullptr));
  const nvrtcResult pch = nvrtcGetPCHCreateStatus(prog);
  EXPECT(pch == NVRTC_SUCCESS || pch == NVRTC_ERROR_NO_PCH_CREATE_ATTEMPTED);
  CHECK_NVRTC(nvrtcDestroyProgram(&prog));
  printf("pch heap %zu, flow callback %s: passed\n", heap,
         nvrtcGetErrorString(callback));
  return 0;
}
#endif

int main() {
  int devices = 0;
  CHECK_CUDA(cudaGetDeviceCount(&devices));
  EXPECT(devices > 0);
  CHECK_DRV(cuInit(0));
  if (test_version_and_archs() || test_ptx_and_names(0) || test_cubin(0) ||
      test_included_files(0) || test_compile_error()) {
    return 1;
  }
#if CUDA_VERSION >= 12000
  if (test_lto(0)) {
    return 1;
  }
#endif
#if CUDA_VERSION >= 12080
  if (test_pch_and_flow()) {
    return 1;
  }
#endif
  if (devices > 1) {
    CHECK_CUDA(cudaSetDevice(1));
    if (test_ptx_and_names(1) || test_cubin(1)) {
      return 1;
    }
  } else {
    printf("multi-device: skipped, one device\n");
  }
  printf("nvrtc shim: all checks passed\n");
  return 0;
}
