// lupine_cudart.cpp — native CUDA runtime API (libcudart) on the LUPINE
// client shim, for the `lupine` Python package. Implements the surface
// mapped from CUDA-13 torch (93 symbols: 9 __cuda* registration ABI +
// 84 cuda*), verified against torch's undefined symbols and the shim's
// exports. Mechanical forwarders go through the lcudart_call* templates;
// everything with real logic (registration tables, lazy module loading,
// memcpy kind dispatch, pointer classification, launch translation) stays
// explicit.
//
// Build (see python/cudart/CMakeLists.txt; clients must already be built):
//   Linux:   libcudart.so.13   (version node matching torch's references)
//   macOS:   libcudart.dylib
//   Windows: cudart64_13.dll
//
// Runtime: set LUPINE_SERVER=host:14833 and load this library (or let the
// lupine Python package preload it) before torch initializes CUDA. Links
// directly against the LUPINE client shim (no interposition needed).

#include <cuda.h>
#include <cuda_runtime_api.h>

#include <cstdint>
#include <cstddef>
#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

// ---------------------------------------------------------------------------
// Fatbin container layout shared with the shim (lupine_fatbin.h).
// ---------------------------------------------------------------------------
struct lcudart_fatbin_wrapper {
  uint32_t magic; // 0x466243b1
  uint32_t version;
  const void *data;
  const void *filename_or_fatbins;
};
#define LCUDART_FATBINC_MAGIC 0x466243b1u

// ---------------------------------------------------------------------------
// Tracing
// ---------------------------------------------------------------------------
static int lcudart_trace_enabled() {
  static int enabled = [] {
    const char *v = getenv("LUPINE_CUDART_TRACE");
    return v && v[0] && strcmp(v, "0") != 0 ? 1 : 0;
  }();
  return enabled;
}
#define LCUDART_TRACE(msg)                                                     \
  do {                                                                         \
    if (lcudart_trace_enabled())                                               \
      fprintf(stderr, "[lupine_cudart] %s\n", msg);                            \
  } while (0)

#define LCUDART_TRACEF(...)                                                    \
  do {                                                                         \
    if (lcudart_trace_enabled()) {                                             \
      fprintf(stderr, "[lupine_cudart] ");                                    \
      fprintf(stderr, __VA_ARGS__);                                            \
      fprintf(stderr, "\n");                                                  \
    }                                                                          \
  } while (0)

// ---------------------------------------------------------------------------
// Error model
// ---------------------------------------------------------------------------

// CUresult -> cudaError_t. Numerically identical for the common codes
// (verified against the CUDA 13 headers); the exceptions:
static inline cudaError_t lcudart_map_error(CUresult r) {
  switch (r) {
  case CUDA_SUCCESS:
    return cudaSuccess;
  case CUDA_ERROR_DEINITIALIZED: // 4 -> 201
    return cudaErrorDeviceUninitialized;
  case CUDA_ERROR_INVALID_HANDLE: // 400 (identical number)
    return cudaErrorInvalidResourceHandle;
  default:
    return static_cast<cudaError_t>(r);
  }
}

static cudaError_t &lcudart_last_error() {
  static thread_local cudaError_t err = cudaSuccess;
  return err;
}

// Record a failure into the sticky error (matches libcudart: every
// non-success return, including cudaErrorNotReady from queries, is sticky).
static inline cudaError_t lcudart_fail(cudaError_t e) {
  if (e != cudaSuccess) {
    lcudart_last_error() = e;
  }
  return e;
}
static inline cudaError_t lcudart_from_cu(CUresult r) {
  return lcudart_fail(lcudart_map_error(r));
}

extern "C" const char *cudaGetErrorName(cudaError_t error) {
  switch (error) {
  case cudaSuccess: return "cudaSuccess";
  case cudaErrorInvalidValue: return "cudaErrorInvalidValue";
  case cudaErrorMemoryAllocation: return "cudaErrorMemoryAllocation";
  case cudaErrorInitializationError: return "cudaErrorInitializationError";
  case cudaErrorCudartUnloading: return "cudaErrorCudartUnloading";
  case cudaErrorDeviceUninitialized: return "cudaErrorDeviceUninitialized";
  case cudaErrorInvalidSymbol: return "cudaErrorInvalidSymbol";
  case cudaErrorNoDevice: return "cudaErrorNoDevice";
  case cudaErrorInvalidDevice: return "cudaErrorInvalidDevice";
  case cudaErrorInvalidPitchValue: return "cudaErrorInvalidPitchValue";
  case cudaErrorInvalidDevicePointer: return "cudaErrorInvalidDevicePointer";
  case cudaErrorInvalidHostPointer: return "cudaErrorInvalidHostPointer";
  case cudaErrorLaunchFailure: return "cudaErrorLaunchFailure";
  case cudaErrorLaunchOutOfResources: return "cudaErrorLaunchOutOfResources";
  case cudaErrorInvalidConfiguration: return "cudaErrorInvalidConfiguration";
  case cudaErrorPeerAccessAlreadyEnabled: return "cudaErrorPeerAccessAlreadyEnabled";
  case cudaErrorPeerAccessNotEnabled: return "cudaErrorPeerAccessNotEnabled";
  case cudaErrorHostMemoryAlreadyRegistered: return "cudaErrorHostMemoryAlreadyRegistered";
  case cudaErrorHostMemoryNotRegistered: return "cudaErrorHostMemoryNotRegistered";
  case cudaErrorNotSupported: return "cudaErrorNotSupported";
  case cudaErrorNotReady: return "cudaErrorNotReady";
  case cudaErrorUnknown: return "cudaErrorUnknown";
  case cudaErrorInvalidResourceHandle: return "cudaErrorInvalidResourceHandle";
  default: return "cudaError<unrecognized>";
  }
}

extern "C" const char *cudaGetErrorString(cudaError_t error) {
  switch (error) {
  case cudaSuccess: return "no error";
  case cudaErrorInvalidValue: return "invalid argument";
  case cudaErrorMemoryAllocation: return "out of memory";
  case cudaErrorInitializationError: return "initialization error";
  case cudaErrorDeviceUninitialized: return "device is not initialized";
  case cudaErrorNoDevice: return "no CUDA-capable device is detected";
  case cudaErrorInvalidDevice: return "invalid device ordinal";
  case cudaErrorLaunchFailure: return "unspecified launch failure";
  case cudaErrorLaunchOutOfResources: return "too many resources requested for launch";
  case cudaErrorNotSupported: return "operation not supported";
  case cudaErrorNotReady: return "cudaErrorNotReady";
  case cudaErrorUnknown: return "unknown error";
  default: {
    static thread_local char buf[64];
    snprintf(buf, sizeof(buf), "CUDA error %d", (int)error);
    return buf;
  }
  }
}

extern "C" cudaError_t cudaGetLastError(void) {
  cudaError_t e = lcudart_last_error();
  lcudart_last_error() = cudaSuccess;
  return e;
}

extern "C" cudaError_t cudaPeekAtLastError(void) {
  return lcudart_last_error();
}

// ---------------------------------------------------------------------------
// Client state
// ---------------------------------------------------------------------------

struct lcudart_state {
  std::mutex mutex;
  bool initialized = false;
  int device_count = 0;
  std::vector<CUdevice> devices; // ordinal -> CUdevice
  // Visible-device map honoring CUDA_VISIBLE_DEVICES (read once at init,
  // like the real runtime reads it at cuInit time): visible ordinal ->
  // physical ordinal. Empty => no devices visible.
  std::vector<int> visible_devices;

  // Registration tables.
  std::unordered_map<const void *, const void *> fatbins; // handle -> image
  struct FunctionRec {
    const void *fatbin;
    std::string device_name; // mangled
  };
  std::unordered_map<const void *, FunctionRec> functions; // host stub -> rec
  struct VarRec {
    const void *fatbin;
    std::string device_name;
    size_t size;
    int constant;
    int global;
  };
  std::unordered_map<const void *, VarRec> vars;

  // Lazy per-device module/function caches.
  std::unordered_map<const void *, std::unordered_map<int, CUmodule>> modules;
  std::unordered_map<const void *, std::unordered_map<int, CUfunction>>
      stub_functions;
  std::unordered_map<const void *, std::unordered_map<int, CUdeviceptr>>
      var_addresses;

  // Minted kernel objects (cudaKernel_t / __cudaGetKernel / cudaLibraryGetKernel).
  enum class KernelMode { Module, Library };
  struct KernelRec {
    KernelMode mode;
    CUkernel kernel;    // Library mode
    const void *stub;   // Module mode: host stub
    CUfunction function; // Module mode: pre-resolved on current device
  };
  std::unordered_map<const void *, KernelRec> kernel_objects;

  // Allocation bookkeeping (pointer classification for
  // cudaMemcpyDefault / cudaPointerGetAttributes).
  struct DeviceAlloc {
    size_t size;
    int device;
  };
  std::unordered_map<const void *, DeviceAlloc> device_allocs;
  struct HostAlloc {
    size_t size;
    unsigned flags;
    void *device_ptr; // non-null when mapped
  };
  std::unordered_map<const void *, HostAlloc> host_allocs;
  struct RegisteredRange {
    void *base;
    size_t size;
  };
  std::vector<RegisteredRange> registered_ranges;

  std::vector<CUmodule> modules_for_fatbin(const void *fatbin) const {
    std::vector<CUmodule> out;
    auto it = modules.find(fatbin);
    if (it != modules.end()) {
      for (auto &kv : it->second) {
        out.push_back(kv.second);
      }
    }
    return out;
  }
};

static lcudart_state &lcudart() {
  static lcudart_state *s = new lcudart_state(); // leaky: destroyed after exit
  return *s;
}

static int &lcudart_current_device() {
  static thread_local int dev = 0;
  return dev;
}

static int &lcudart_ctx_device() {
  static thread_local int dev = -1; // device whose primary ctx is current
  return dev;
}

// ---------------------------------------------------------------------------
// CUDA_VISIBLE_DEVICES
// ---------------------------------------------------------------------------

static std::string lcudart_uuid_hex(const CUuuid &uuid) {
  static const char *hex = "0123456789abcdef";
  std::string out;
  out.reserve(32);
  for (int i = 0; i < 16; ++i) {
    out.push_back(hex[(uuid.bytes[i] >> 4) & 0xf]);
    out.push_back(hex[uuid.bytes[i] & 0xf]);
  }
  return out;
}

// Build visible_devices from CUDA_VISIBLE_DEVICES:
//   unset          -> all devices (identity)
//   "" or "-1"     -> no devices
//   "0,1"          -> subset / reordering by physical ordinal
//   "GPU-<uuid>"   -> subset by UUID (hyphenated or bare, case-insensitive)
// Invalid entries are dropped, matching driver behavior. MIG- entries never
// match (no MIG devices are exposed).
// Called with s.mutex held; only issues RPCs for UUID lookups.
static void lcudart_build_visible_devices() {
  lcudart_state &s = lcudart();
  s.visible_devices.clear();
  const char *cvd = getenv("CUDA_VISIBLE_DEVICES");
  if (cvd == nullptr) {
    for (int i = 0; i < s.device_count; ++i) {
      s.visible_devices.push_back(i);
    }
    return;
  }
  std::string spec(cvd);
  if (spec.empty() || spec == "-1") {
    return; // explicitly no devices
  }

  std::vector<std::string> uuid_hex; // physical ordinal -> lowercase hex
  bool uuids_queried = false;
  auto query_uuids = [&]() {
    if (uuids_queried) {
      return;
    }
    uuids_queried = true;
    uuid_hex.assign(s.device_count, std::string());
    CUuuid uuid = {};
    for (int i = 0; i < s.device_count; ++i) {
      if (cuDeviceGetUuid_v2(&uuid, s.devices[i]) == CUDA_SUCCESS) {
        uuid_hex[i] = lcudart_uuid_hex(uuid);
      }
    }
  };
  auto normalize_token = [](std::string tok) -> std::string {
    const size_t b = tok.find_first_not_of(" \t");
    if (b == std::string::npos) {
      return std::string();
    }
    const size_t e = tok.find_last_not_of(" \t");
    tok = tok.substr(b, e - b + 1);
    if (tok.rfind("GPU-", 0) == 0) {
      tok = tok.substr(4);
    }
    std::string out;
    out.reserve(tok.size());
    for (char c : tok) {
      if (c == '-') {
        continue;
      }
      out.push_back(static_cast<char>(tolower(static_cast<unsigned char>(c))));
    }
    return out;
  };

  size_t pos = 0;
  while (pos <= spec.size()) {
    size_t comma = spec.find(',', pos);
    std::string raw =
        spec.substr(pos, comma == std::string::npos ? std::string::npos
                                                     : comma - pos);
    std::string norm = normalize_token(raw);
    if (!norm.empty()) {
      const bool numeric =
          norm.find_first_not_of("0123456789") == std::string::npos;
      if (numeric) {
        long v = strtol(norm.c_str(), nullptr, 10);
        if (v >= 0 && v < s.device_count) {
          s.visible_devices.push_back(static_cast<int>(v));
        }
      } else if (norm.size() >= 16) {
        query_uuids();
        for (int i = 0; i < s.device_count; ++i) {
          if (!uuid_hex[i].empty() && uuid_hex[i] == norm) {
            s.visible_devices.push_back(i);
            break;
          }
        }
      }
    }
    if (comma == std::string::npos) {
      break;
    }
    pos = comma + 1;
  }
}

struct lcudart_call_config {
  dim3 grid;
  dim3 block;
  size_t shared;
  cudaStream_t stream;
};

static constexpr size_t kLcudartMaxConfigDepth = 64;
static thread_local lcudart_call_config lcudart_config_stack[kLcudartMaxConfigDepth];
static thread_local size_t lcudart_config_depth = 0;

// ---------------------------------------------------------------------------
// Initialization & context bootstrap
// ---------------------------------------------------------------------------

static cudaError_t lcudart_ensure_init() {
  lcudart_state &s = lcudart();
  std::lock_guard<std::mutex> lock(s.mutex);
  if (s.initialized) {
    return cudaSuccess;
  }
  CUresult r = cuInit(0);
  if (r != CUDA_SUCCESS) {
    return lcudart_map_error(r);
  }
  int count = 0;
  r = cuDeviceGetCount(&count);
  if (r != CUDA_SUCCESS) {
    return lcudart_map_error(r);
  }
  s.devices.resize(count);
  for (int i = 0; i < count; ++i) {
    r = cuDeviceGet(&s.devices[i], i);
    if (r != CUDA_SUCCESS) {
      return lcudart_map_error(r);
    }
  }
  s.device_count = count;
  lcudart_build_visible_devices();
  s.initialized = true;
  return cudaSuccess;
}

static CUdevice lcudart_cu_device(int ordinal, cudaError_t *err) {
  lcudart_state &s = lcudart();
  if (ordinal < 0 || ordinal >= static_cast<int>(s.visible_devices.size())) {
    *err = cudaErrorInvalidDevice;
    return 0;
  }
  *err = cudaSuccess;
  return s.devices[s.visible_devices[ordinal]];
}

// Ensure the current thread's device primary context is current server-side.
static cudaError_t lcudart_ensure_context() {
  int dev = lcudart_current_device();
  if (lcudart_ctx_device() == dev) {
    return cudaSuccess;
  }
  cudaError_t err = cudaSuccess;
  CUdevice cu = lcudart_cu_device(dev, &err);
  if (err != cudaSuccess) {
    return err;
  }
  CUcontext ctx = nullptr;
  CUresult r = cuDevicePrimaryCtxRetain(&ctx, cu);
  if (r != CUDA_SUCCESS) {
    return lcudart_map_error(r);
  }
  r = cuCtxSetCurrent(ctx);
  if (r != CUDA_SUCCESS) {
    return lcudart_map_error(r);
  }
  lcudart_ctx_device() = dev;
  return cudaSuccess;
}

// Standard preamble: init + context. Most entry points call this.
static cudaError_t lcudart_ready() {
  cudaError_t e = lcudart_ensure_init();
  if (e != cudaSuccess) {
    return lcudart_fail(e);
  }
  e = lcudart_ensure_context();
  return e == cudaSuccess ? cudaSuccess : lcudart_fail(e);
}

// ---------------------------------------------------------------------------
// Mechanical-forwarding templates
//
// The overwhelming majority of runtime entry points are a fixed preamble
// (initialization, context, device-ordinal resolution) followed by one
// driver call whose CUresult maps through lcudart_from_cu. These templates
// express that; a function body reduces to its null checks plus one call.
// ---------------------------------------------------------------------------

// Preamble (init + current context), then body() -> CUresult.
template <typename Body> static cudaError_t lcudart_call(Body &&body) {
  cudaError_t e = lcudart_ready();
  if (e != cudaSuccess) {
    return e;
  }
  return lcudart_from_cu(body());
}

// Initialization only (no context switch), then body() -> CUresult.
template <typename Body> static cudaError_t lcudart_call_init(Body &&body) {
  cudaError_t e = lcudart_ensure_init();
  if (e != cudaSuccess) {
    return lcudart_fail(e);
  }
  return lcudart_from_cu(body());
}

// Initialization plus one ordinal resolved to a CUdevice, then body(dev).
template <typename Body>
static cudaError_t lcudart_call_device(int ordinal, Body &&body) {
  cudaError_t e = lcudart_ensure_init();
  if (e != cudaSuccess) {
    return lcudart_fail(e);
  }
  CUdevice dev = lcudart_cu_device(ordinal, &e);
  if (e != cudaSuccess) {
    return lcudart_fail(e);
  }
  return lcudart_from_cu(body(dev));
}

// Initialization plus two ordinals (peer queries), then body(a, b).
template <typename Body>
static cudaError_t lcudart_call_devices(int ordinal_a, int ordinal_b,
                                        Body &&body) {
  cudaError_t e = lcudart_ensure_init();
  if (e != cudaSuccess) {
    return lcudart_fail(e);
  }
  CUdevice a = lcudart_cu_device(ordinal_a, &e);
  if (e == cudaSuccess) {
    CUdevice b = lcudart_cu_device(ordinal_b, &e);
    if (e == cudaSuccess) {
      return lcudart_from_cu(body(a, b));
    }
  }
  return lcudart_fail(e);
}

// ---------------------------------------------------------------------------
// Lazy module loading & resolution
// ---------------------------------------------------------------------------

static CUmodule lcudart_module_for(const void *fatbin, int device,
                                   cudaError_t *err) {
  lcudart_state &s = lcudart();
  std::lock_guard<std::mutex> lock(s.mutex);
  auto it = s.fatbins.find(fatbin);
  if (it == s.fatbins.end()) {
    *err = cudaErrorInvalidDeviceFunction;
    return nullptr;
  }
  auto &per_device = s.modules[fatbin];
  auto found = per_device.find(device);
  if (found != per_device.end()) {
    *err = cudaSuccess;
    return found->second;
  }
  CUmodule module = nullptr;
  CUresult r = cuModuleLoadData(&module, it->second);
  LCUDART_TRACEF("lazy module load fatbin=%p dev=%d -> %d module=%p", fatbin,
                 device, (int)r, (void *)module);
  if (r != CUDA_SUCCESS) {
    *err = lcudart_map_error(r);
    return nullptr;
  }
  per_device[device] = module;
  *err = cudaSuccess;
  return module;
}

static CUfunction lcudart_function_for(const void *stub, int device,
                                       cudaError_t *err) {
  lcudart_state &s = lcudart();
  const void *fatbin;
  std::string name;
  {
    std::lock_guard<std::mutex> lock(s.mutex);
    auto it = s.functions.find(stub);
    if (it == s.functions.end()) {
      *err = cudaErrorInvalidDeviceFunction;
      return nullptr;
    }
    fatbin = it->second.fatbin;
    name = it->second.device_name;
    auto &cache = s.stub_functions[stub];
    auto found = cache.find(device);
    if (found != cache.end()) {
      *err = cudaSuccess;
      return found->second;
    }
  }
  CUmodule module = lcudart_module_for(fatbin, device, err);
  if (module == nullptr) {
    return nullptr;
  }
  CUfunction fn = nullptr;
  CUresult r = cuModuleGetFunction(&fn, module, name.c_str());
  LCUDART_TRACEF("cuModuleGetFunction name=%s -> %d fn=%p", name.c_str(),
                 (int)r, (void *)fn);
  if (r != CUDA_SUCCESS) {
    *err = lcudart_map_error(r);
    return nullptr;
  }
  {
    std::lock_guard<std::mutex> lock(s.mutex);
    s.stub_functions[stub][device] = fn;
  }
  *err = cudaSuccess;
  return fn;
}

static CUdeviceptr lcudart_var_for(const void *host_var, int device,
                                   size_t *size_out, cudaError_t *err) {
  lcudart_state &s = lcudart();
  const void *fatbin;
  std::string name;
  size_t size = 0;
  {
    std::lock_guard<std::mutex> lock(s.mutex);
    auto it = s.vars.find(host_var);
    if (it == s.vars.end()) {
      *err = cudaErrorInvalidSymbol;
      return 0;
    }
    fatbin = it->second.fatbin;
    name = it->second.device_name;
    size = it->second.size;
    auto &cache = s.var_addresses[host_var];
    auto found = cache.find(device);
    if (found != cache.end()) {
      if (size_out) {
        *size_out = size;
      }
      *err = cudaSuccess;
      return found->second;
    }
  }
  CUmodule module = lcudart_module_for(fatbin, device, err);
  if (module == nullptr) {
    return 0;
  }
  CUdeviceptr dptr = 0;
  size_t bytes = 0;
  CUresult r = cuModuleGetGlobal_v2(&dptr, &bytes, module, name.c_str());
  if (r != CUDA_SUCCESS) {
    *err = lcudart_map_error(r);
    return 0;
  }
  {
    std::lock_guard<std::mutex> lock(s.mutex);
    s.var_addresses[host_var][device] = dptr;
  }
  if (size_out) {
    *size_out = bytes != 0 ? bytes : size;
  }
  *err = cudaSuccess;
  return dptr;
}

// Resolve a `const void *func` given to cudaLaunchKernel / ExC /
// attribute queries: either a minted kernel object or a registered host
// stub. Returns the CUfunction for `device`. Never called with s.mutex held.
static CUfunction lcudart_resolve(const void *func, int device,
                                  cudaError_t *err) {
  lcudart_state &s = lcudart();
  const void *stub = func;
  {
    std::lock_guard<std::mutex> lock(s.mutex);
    auto it = s.kernel_objects.find(func);
    if (it != s.kernel_objects.end()) {
      if (it->second.mode == lcudart_state::KernelMode::Library) {
        // The shim's cuLaunchKernel accepts CUkernel handles natively;
        // attribute paths use the CUkernel variant directly.
        *err = cudaSuccess;
        return reinterpret_cast<CUfunction>(it->second.kernel);
      }
      stub = it->second.stub;
    }
  }
  return lcudart_function_for(stub, device, err);
}

// ---------------------------------------------------------------------------
// Registration ABI (nvcc static-initializer protocol)
// ---------------------------------------------------------------------------

extern "C" void **__cudaRegisterFatBinary(void *fatBinCWrapper) {
  LCUDART_TRACE("__cudaRegisterFatBinary");
  lcudart_state &s = lcudart();
  auto *handle = new void *(fatBinCWrapper);
  std::lock_guard<std::mutex> lock(s.mutex);
  s.fatbins[handle] = fatBinCWrapper;
  return handle;
}

extern "C" void __cudaRegisterFatBinaryEnd(void **fatBinCWrapper) {
  (void)fatBinCWrapper; // end-of-TU marker; nothing to do
}

// CUDA 13 lazy-module-init hook emitted by nvcc 13 TUs. Module loading in
// this runtime is already lazy (first use), so this is a no-op success.
extern "C" cudaError_t __cudaInitModule(void) { return cudaSuccess; }

extern "C" void __cudaRegisterFunction(
    void **fatBinaryHandle, const char *hostFun, char *deviceFun,
    const char *deviceName, int thread_limit, uint3 *tid, uint3 *bid,
    dim3 *bDim, dim3 *gDim, int *wSize) {
  (void)thread_limit;
  (void)tid;
  (void)bid;
  (void)bDim;
  (void)gDim;
  (void)wSize;
  (void)deviceName;
  lcudart_state &s = lcudart();
  std::lock_guard<std::mutex> lock(s.mutex);
  s.functions[hostFun] = {fatBinaryHandle, deviceFun ? deviceFun : ""};
}

extern "C" void __cudaRegisterVar(void **fatBinaryHandle, char *hostVar,
                                  char *deviceAddress,
                                  const char *deviceName, int ext, size_t size,
                                  int constant, int global) {
  (void)deviceAddress;
  (void)ext;
  lcudart_state &s = lcudart();
  std::lock_guard<std::mutex> lock(s.mutex);
  s.vars[hostVar] = {fatBinaryHandle, deviceName ? deviceName : "", size,
                     constant, global};
}

extern "C" void __cudaUnregisterFatBinary(void **fatBinaryHandle) {
  LCUDART_TRACE("__cudaUnregisterFatBinary");
  lcudart_state &s = lcudart();
  std::vector<CUmodule> to_unload;
  {
    std::lock_guard<std::mutex> lock(s.mutex);
    to_unload = s.modules_for_fatbin(fatBinaryHandle);
    s.modules.erase(fatBinaryHandle);
    for (auto it = s.functions.begin(); it != s.functions.end();) {
      if (it->second.fatbin == fatBinaryHandle) {
        s.stub_functions.erase(it->first);
        it = s.functions.erase(it);
      } else {
        ++it;
      }
    }
    for (auto it = s.vars.begin(); it != s.vars.end();) {
      if (it->second.fatbin == fatBinaryHandle) {
        s.var_addresses.erase(it->first);
        it = s.vars.erase(it);
      } else {
        ++it;
      }
    }
    s.fatbins.erase(fatBinaryHandle);
    delete fatBinaryHandle;
  }
  for (CUmodule m : to_unload) {
    (void)cuModuleUnload(m);
  }
}

extern "C" cudaError_t __cudaPushCallConfiguration(dim3 gridDim,
                                                   dim3 blockDim,
                                                   size_t sharedMem,
                                                   cudaStream_t stream) {
  if (lcudart_config_depth >= kLcudartMaxConfigDepth) {
    return lcudart_fail(cudaErrorInvalidConfiguration);
  }
  lcudart_config_stack[lcudart_config_depth++] = {gridDim, blockDim, sharedMem,
                                                  stream};
  return cudaSuccess;
}

extern "C" cudaError_t __cudaPopCallConfiguration(dim3 *gridDim,
                                                  dim3 *blockDim,
                                                  size_t *sharedMem,
                                                  cudaStream_t *stream) {
  if (lcudart_config_depth == 0) {
    return lcudart_fail(cudaErrorInvalidConfiguration);
  }
  const lcudart_call_config &cfg =
      lcudart_config_stack[--lcudart_config_depth];
  *gridDim = cfg.grid;
  *blockDim = cfg.block;
  *sharedMem = cfg.shared;
  *stream = cfg.stream;
  return cudaSuccess;
}

extern "C" cudaError_t __cudaGetKernel(cudaKernel_t *kernelPtr,
                                       const void *hostFun) {
  if (kernelPtr == nullptr) {
    return lcudart_fail(cudaErrorInvalidValue);
  }
  cudaError_t e = lcudart_ready();
  if (e != cudaSuccess) {
    return e;
  }
  lcudart_state &s = lcudart();
  {
    std::lock_guard<std::mutex> lock(s.mutex);
    if (s.functions.find(hostFun) == s.functions.end()) {
      return lcudart_fail(cudaErrorInvalidDeviceFunction);
    }
  }
  auto *stable = new lcudart_state::KernelRec{
      lcudart_state::KernelMode::Module, nullptr, hostFun, nullptr};
  {
    std::lock_guard<std::mutex> lock(s.mutex);
    s.kernel_objects[stable] = *stable;
  }
  *kernelPtr = reinterpret_cast<cudaKernel_t>(stable);
  return cudaSuccess;
}

// Public variant of the same thing.
extern "C" cudaError_t cudaGetKernel(cudaKernel_t *kernelPtr,
                                     const void *func) {
  return __cudaGetKernel(kernelPtr, func);
}

// ---------------------------------------------------------------------------
// Device management
// ---------------------------------------------------------------------------

extern "C" cudaError_t cudaGetDevice(int *device) {
  if (device == nullptr) {
    return lcudart_fail(cudaErrorInvalidValue);
  }
  *device = lcudart_current_device();
  return cudaSuccess;
}

extern "C" cudaError_t cudaSetDevice(int device) {
  cudaError_t e = lcudart_ensure_init();
  if (e != cudaSuccess) {
    return lcudart_fail(e);
  }
  if (device < 0 || device >= static_cast<int>(lcudart().visible_devices.size())) {
    return lcudart_fail(cudaErrorInvalidDevice);
  }
  lcudart_current_device() = device;
  e = lcudart_ensure_context();
  return e == cudaSuccess ? cudaSuccess : lcudart_fail(e);
}

extern "C" cudaError_t cudaGetDeviceCount(int *count) {
  if (count == nullptr) {
    return lcudart_fail(cudaErrorInvalidValue);
  }
  cudaError_t e = lcudart_ensure_init();
  if (e != cudaSuccess) {
    return lcudart_fail(e);
  }
  *count = static_cast<int>(lcudart().visible_devices.size());
  return cudaSuccess;
}

extern "C" cudaError_t cudaDeviceGetAttribute(int *value,
                                              enum cudaDeviceAttr attr,
                                              int device) {
  if (value == nullptr) {
    return lcudart_fail(cudaErrorInvalidValue);
  }
  // cudaDeviceAttr values are numerically identical to CUdevice_attribute
  // (verified value-by-value for all 137 runtime attributes).
  return lcudart_call_device(device, [&](CUdevice cu) {
    return cuDeviceGetAttribute(value, static_cast<CUdevice_attribute>(attr),
                                cu);
  });
}

static int lcudart_devattr(int device, enum cudaDeviceAttr a,
                           int fallback = 0) {
  int v = fallback;
  (void)cudaDeviceGetAttribute(&v, a, device);
  return v;
}

extern "C" cudaError_t cudaGetDeviceProperties(struct cudaDeviceProp *prop,
                                               int device) {
  if (prop == nullptr) {
    return lcudart_fail(cudaErrorInvalidValue);
  }
  cudaError_t e = lcudart_ensure_init();
  if (e != cudaSuccess) {
    return lcudart_fail(e);
  }
  CUdevice cu = lcudart_cu_device(device, &e);
  if (e != cudaSuccess) {
    return lcudart_fail(e);
  }
  memset(prop, 0, sizeof(*prop));

  CUresult r = cuDeviceGetName(prop->name, sizeof(prop->name), cu);
  if (r != CUDA_SUCCESS) {
    return lcudart_from_cu(r);
  }
  CUuuid uuid;
  r = cuDeviceGetUuid_v2(&uuid, cu);
  if (r == CUDA_SUCCESS) {
    memcpy(prop->uuid.bytes, uuid.bytes, 16);
  }
  unsigned node_mask = 0;
  r = cuDeviceGetLuid(prop->luid, &node_mask, cu);
  if (r == CUDA_SUCCESS) {
    prop->luidDeviceNodeMask = node_mask;
  }
  size_t total = 0;
  r = cuDeviceTotalMem_v2(&total, cu);
  if (r != CUDA_SUCCESS) {
    return lcudart_from_cu(r);
  }
  prop->totalGlobalMem = total;

  prop->sharedMemPerBlock =
      (size_t)lcudart_devattr(device, cudaDevAttrMaxSharedMemoryPerBlock);
  prop->regsPerBlock = lcudart_devattr(device, cudaDevAttrMaxRegistersPerBlock);
  prop->warpSize = lcudart_devattr(device, cudaDevAttrWarpSize, 32);
  prop->memPitch = (size_t)lcudart_devattr(device, cudaDevAttrMaxPitch);
  prop->maxThreadsPerBlock =
      lcudart_devattr(device, cudaDevAttrMaxThreadsPerBlock);
  prop->maxThreadsDim[0] = lcudart_devattr(device, cudaDevAttrMaxBlockDimX);
  prop->maxThreadsDim[1] = lcudart_devattr(device, cudaDevAttrMaxBlockDimY);
  prop->maxThreadsDim[2] = lcudart_devattr(device, cudaDevAttrMaxBlockDimZ);
  prop->maxGridSize[0] = lcudart_devattr(device, cudaDevAttrMaxGridDimX);
  prop->maxGridSize[1] = lcudart_devattr(device, cudaDevAttrMaxGridDimY);
  prop->maxGridSize[2] = lcudart_devattr(device, cudaDevAttrMaxGridDimZ);
  prop->totalConstMem =
      (size_t)lcudart_devattr(device, cudaDevAttrTotalConstantMemory);
  prop->major = lcudart_devattr(device, cudaDevAttrComputeCapabilityMajor, 7);
  prop->minor = lcudart_devattr(device, cudaDevAttrComputeCapabilityMinor, 5);
  prop->textureAlignment =
      (size_t)lcudart_devattr(device, cudaDevAttrTextureAlignment);
  prop->texturePitchAlignment = (size_t)lcudart_devattr(
      device, cudaDevAttrTexturePitchAlignment);
  prop->multiProcessorCount =
      lcudart_devattr(device, cudaDevAttrMultiProcessorCount);
  prop->integrated = lcudart_devattr(device, cudaDevAttrIntegrated);
  prop->canMapHostMemory =
      lcudart_devattr(device, cudaDevAttrCanMapHostMemory);
  prop->maxTexture1D = lcudart_devattr(device, cudaDevAttrMaxTexture1DWidth);
  prop->maxTexture1DMipmap =
      lcudart_devattr(device, cudaDevAttrMaxTexture1DMipmappedWidth);
  prop->maxTexture2D[0] = lcudart_devattr(device, cudaDevAttrMaxTexture2DWidth);
  prop->maxTexture2D[1] =
      lcudart_devattr(device, cudaDevAttrMaxTexture2DHeight);
  prop->maxTexture2DMipmap[0] =
      lcudart_devattr(device, cudaDevAttrMaxTexture2DMipmappedWidth);
  prop->maxTexture2DMipmap[1] =
      lcudart_devattr(device, cudaDevAttrMaxTexture2DMipmappedHeight);
  prop->maxTexture2DLinear[0] =
      lcudart_devattr(device, cudaDevAttrMaxTexture2DLinearWidth);
  prop->maxTexture2DLinear[1] =
      lcudart_devattr(device, cudaDevAttrMaxTexture2DLinearHeight);
  prop->maxTexture2DLinear[2] =
      lcudart_devattr(device, cudaDevAttrMaxTexture2DLinearPitch);
  prop->maxTexture2DGather[0] =
      lcudart_devattr(device, cudaDevAttrMaxTexture2DGatherWidth);
  prop->maxTexture2DGather[1] =
      lcudart_devattr(device, cudaDevAttrMaxTexture2DGatherHeight);
  prop->maxTexture3D[0] = lcudart_devattr(device, cudaDevAttrMaxTexture3DWidth);
  prop->maxTexture3D[1] =
      lcudart_devattr(device, cudaDevAttrMaxTexture3DHeight);
  prop->maxTexture3D[2] =
      lcudart_devattr(device, cudaDevAttrMaxTexture3DDepth);
  prop->maxTexture3DAlt[0] =
      lcudart_devattr(device, cudaDevAttrMaxTexture3DWidthAlt);
  prop->maxTexture3DAlt[1] =
      lcudart_devattr(device, cudaDevAttrMaxTexture3DHeightAlt);
  prop->maxTexture3DAlt[2] =
      lcudart_devattr(device, cudaDevAttrMaxTexture3DDepthAlt);
  prop->maxTextureCubemap =
      lcudart_devattr(device, cudaDevAttrMaxTextureCubemapWidth);
  prop->maxTexture1DLayered[0] =
      lcudart_devattr(device, cudaDevAttrMaxTexture1DLayeredWidth);
  prop->maxTexture1DLayered[1] =
      lcudart_devattr(device, cudaDevAttrMaxTexture1DLayeredLayers);
  prop->maxTexture2DLayered[0] =
      lcudart_devattr(device, cudaDevAttrMaxTexture2DLayeredWidth);
  prop->maxTexture2DLayered[1] =
      lcudart_devattr(device, cudaDevAttrMaxTexture2DLayeredHeight);
  prop->maxTexture2DLayered[2] =
      lcudart_devattr(device, cudaDevAttrMaxTexture2DLayeredLayers);
  prop->maxTextureCubemapLayered[0] =
      lcudart_devattr(device, cudaDevAttrMaxTextureCubemapLayeredWidth);
  prop->maxTextureCubemapLayered[1] =
      lcudart_devattr(device, cudaDevAttrMaxTextureCubemapLayeredLayers);
  prop->maxSurface1D = lcudart_devattr(device, cudaDevAttrMaxSurface1DWidth);
  prop->maxSurface2D[0] = lcudart_devattr(device, cudaDevAttrMaxSurface2DWidth);
  prop->maxSurface2D[1] =
      lcudart_devattr(device, cudaDevAttrMaxSurface2DHeight);
  prop->maxSurface3D[0] = lcudart_devattr(device, cudaDevAttrMaxSurface3DWidth);
  prop->maxSurface3D[1] =
      lcudart_devattr(device, cudaDevAttrMaxSurface3DHeight);
  prop->maxSurface3D[2] =
      lcudart_devattr(device, cudaDevAttrMaxSurface3DDepth);
  prop->maxSurface1DLayered[0] =
      lcudart_devattr(device, cudaDevAttrMaxSurface1DLayeredWidth);
  prop->maxSurface1DLayered[1] =
      lcudart_devattr(device, cudaDevAttrMaxSurface1DLayeredLayers);
  prop->maxSurface2DLayered[0] =
      lcudart_devattr(device, cudaDevAttrMaxSurface2DLayeredWidth);
  prop->maxSurface2DLayered[1] =
      lcudart_devattr(device, cudaDevAttrMaxSurface2DLayeredHeight);
  prop->maxSurface2DLayered[2] =
      lcudart_devattr(device, cudaDevAttrMaxSurface2DLayeredLayers);
  prop->maxSurfaceCubemap =
      lcudart_devattr(device, cudaDevAttrMaxSurfaceCubemapWidth);
  prop->maxSurfaceCubemapLayered[0] =
      lcudart_devattr(device, cudaDevAttrMaxSurfaceCubemapLayeredWidth);
  prop->maxSurfaceCubemapLayered[1] =
      lcudart_devattr(device, cudaDevAttrMaxSurfaceCubemapLayeredLayers);
  prop->surfaceAlignment =
      (size_t)lcudart_devattr(device, cudaDevAttrSurfaceAlignment);
  prop->concurrentKernels =
      lcudart_devattr(device, cudaDevAttrConcurrentKernels, 1);
  prop->ECCEnabled = lcudart_devattr(device, cudaDevAttrEccEnabled);
  prop->pciBusID = lcudart_devattr(device, cudaDevAttrPciBusId);
  prop->pciDeviceID = lcudart_devattr(device, cudaDevAttrPciDeviceId);
  prop->pciDomainID = lcudart_devattr(device, cudaDevAttrPciDomainId);
  prop->tccDriver = lcudart_devattr(device, cudaDevAttrTccDriver);
  prop->asyncEngineCount =
      lcudart_devattr(device, cudaDevAttrAsyncEngineCount, 1);
  prop->unifiedAddressing =
      lcudart_devattr(device, cudaDevAttrUnifiedAddressing, 1);
  prop->memoryBusWidth = lcudart_devattr(device, cudaDevAttrGlobalMemoryBusWidth);
  prop->l2CacheSize = lcudart_devattr(device, cudaDevAttrL2CacheSize);
  prop->persistingL2CacheMaxSize =
      lcudart_devattr(device, cudaDevAttrMaxPersistingL2CacheSize);
  prop->maxThreadsPerMultiProcessor =
      lcudart_devattr(device, cudaDevAttrMaxThreadsPerMultiProcessor);
  prop->streamPrioritiesSupported =
      lcudart_devattr(device, cudaDevAttrStreamPrioritiesSupported);
  prop->globalL1CacheSupported =
      lcudart_devattr(device, cudaDevAttrGlobalL1CacheSupported);
  prop->localL1CacheSupported =
      lcudart_devattr(device, cudaDevAttrLocalL1CacheSupported);
  prop->sharedMemPerMultiprocessor = (size_t)lcudart_devattr(
      device, cudaDevAttrMaxSharedMemoryPerMultiprocessor);
  prop->regsPerMultiprocessor =
      lcudart_devattr(device, cudaDevAttrMaxRegistersPerMultiprocessor);
  prop->managedMemory = lcudart_devattr(device, cudaDevAttrManagedMemory);
  prop->isMultiGpuBoard = lcudart_devattr(device, cudaDevAttrIsMultiGpuBoard);
  prop->multiGpuBoardGroupID =
      lcudart_devattr(device, cudaDevAttrMultiGpuBoardGroupID);
  prop->hostNativeAtomicSupported =
      lcudart_devattr(device, cudaDevAttrHostNativeAtomicSupported);
  prop->pageableMemoryAccess =
      lcudart_devattr(device, cudaDevAttrPageableMemoryAccess);
  prop->concurrentManagedAccess =
      lcudart_devattr(device, cudaDevAttrConcurrentManagedAccess);
  prop->computePreemptionSupported =
      lcudart_devattr(device, cudaDevAttrComputePreemptionSupported);
  prop->canUseHostPointerForRegisteredMem =
      lcudart_devattr(device, cudaDevAttrCanUseHostPointerForRegisteredMem);
  prop->cooperativeLaunch =
      lcudart_devattr(device, cudaDevAttrCooperativeLaunch);
  prop->sharedMemPerBlockOptin =
      (size_t)lcudart_devattr(device, cudaDevAttrMaxSharedMemoryPerBlockOptin);
  prop->pageableMemoryAccessUsesHostPageTables = lcudart_devattr(
      device, cudaDevAttrPageableMemoryAccessUsesHostPageTables);
  prop->directManagedMemAccessFromHost =
      lcudart_devattr(device, cudaDevAttrDirectManagedMemAccessFromHost);
  prop->maxBlocksPerMultiProcessor =
      lcudart_devattr(device, cudaDevAttrMaxBlocksPerMultiprocessor);
  prop->accessPolicyMaxWindowSize =
      lcudart_devattr(device, cudaDevAttrMaxAccessPolicyWindowSize);
  prop->reservedSharedMemPerBlock = (size_t)lcudart_devattr(
      device, cudaDevAttrReservedSharedMemoryPerBlock);
  prop->hostRegisterSupported =
      lcudart_devattr(device, cudaDevAttrHostRegisterSupported, 1);
  prop->sparseCudaArraySupported =
      lcudart_devattr(device, cudaDevAttrSparseCudaArraySupported);
  prop->hostRegisterReadOnlySupported =
      lcudart_devattr(device, cudaDevAttrHostRegisterReadOnlySupported);
  prop->timelineSemaphoreInteropSupported =
      lcudart_devattr(device, cudaDevAttrTimelineSemaphoreInteropSupported);
  prop->memoryPoolsSupported =
      lcudart_devattr(device, cudaDevAttrMemoryPoolsSupported, 1);
  prop->gpuDirectRDMASupported =
      lcudart_devattr(device, cudaDevAttrGPUDirectRDMASupported);
  prop->gpuDirectRDMAFlushWritesOptions = (unsigned)lcudart_devattr(
      device, cudaDevAttrGPUDirectRDMAFlushWritesOptions);
  prop->gpuDirectRDMAWritesOrdering =
      lcudart_devattr(device, cudaDevAttrGPUDirectRDMAWritesOrdering);
  prop->memoryPoolSupportedHandleTypes = (unsigned)lcudart_devattr(
      device, cudaDevAttrMemoryPoolSupportedHandleTypes);
  prop->deferredMappingCudaArraySupported =
      lcudart_devattr(device, cudaDevAttrDeferredMappingCudaArraySupported);
  prop->ipcEventSupported = lcudart_devattr(device, cudaDevAttrIpcEventSupport);
  prop->clusterLaunch = lcudart_devattr(device, cudaDevAttrClusterLaunch);
  prop->unifiedFunctionPointers = 1; // CU_DEVICE_ATTRIBUTE_UNIFIED_FUNCTION_POINTERS=129
  prop->deviceNumaConfig = lcudart_devattr(device, cudaDevAttrNumaConfig);
  prop->deviceNumaId = lcudart_devattr(device, cudaDevAttrNumaId);
  prop->mpsEnabled = lcudart_devattr(device, cudaDevAttrMpsEnabled);
  prop->hostNumaId = lcudart_devattr(device, cudaDevAttrHostNumaId);
  prop->gpuPciDeviceID =
      (unsigned)lcudart_devattr(device, cudaDevAttrGpuPciDeviceId);
  prop->gpuPciSubsystemID =
      (unsigned)lcudart_devattr(device, cudaDevAttrGpuPciSubsystemId);
  prop->hostNumaMultinodeIpcSupported =
      lcudart_devattr(device, cudaDevAttrHostNumaMultinodeIpcSupported);
  return cudaSuccess;
}

extern "C" cudaError_t cudaDeviceSynchronize(void) {
  return lcudart_call([] { return cuCtxSynchronize(); });
}

extern "C" cudaError_t cudaDeviceCanAccessPeer(int *canAccessPeer, int device,
                                               int peerDevice) {
  if (canAccessPeer == nullptr) {
    return lcudart_fail(cudaErrorInvalidValue);
  }
  return lcudart_call_devices(device, peerDevice,
                              [&](CUdevice a, CUdevice b) {
                                return cuDeviceCanAccessPeer(canAccessPeer, a, b);
                              });
}

extern "C" cudaError_t cudaDeviceEnablePeerAccess(int peerDevice,
                                                  unsigned int flags) {
  cudaError_t e = lcudart_ready();
  if (e != cudaSuccess) {
    return e;
  }
  CUdevice peer = lcudart_cu_device(peerDevice, &e);
  if (e != cudaSuccess) {
    return lcudart_fail(e);
  }
  CUcontext peer_ctx = nullptr;
  CUresult r = cuDevicePrimaryCtxRetain(&peer_ctx, peer);
  if (r != CUDA_SUCCESS) {
    return lcudart_from_cu(r);
  }
  return lcudart_from_cu(cuCtxEnablePeerAccess(peer_ctx, flags));
}

extern "C" cudaError_t cudaDeviceGetPCIBusId(char *pciBusId, int len,
                                             int device) {
  if (pciBusId == nullptr || len < 13) {
    return lcudart_fail(cudaErrorInvalidValue);
  }
  return lcudart_call_device(
      device, [&](CUdevice cu) { return cuDeviceGetPCIBusId(pciBusId, len, cu); });
}

extern "C" cudaError_t cudaDeviceGetStreamPriorityRange(int *leastPriority,
                                                        int *greatestPriority) {
  return lcudart_call(
      [&] { return cuCtxGetStreamPriorityRange(leastPriority, greatestPriority); });
}

extern "C" cudaError_t cudaDeviceGetDefaultMemPool(cudaMemPool_t *memPool,
                                                   int device) {
  if (memPool == nullptr) {
    return lcudart_fail(cudaErrorInvalidValue);
  }
  return lcudart_call_device(
      device, [&](CUdevice cu) { return cuDeviceGetDefaultMemPool(memPool, cu); });
}

extern "C" cudaError_t cudaDriverGetVersion(int *driverVersion) {
  if (driverVersion == nullptr) {
    return lcudart_fail(cudaErrorInvalidValue);
  }
  return lcudart_call_init([&] { return cuDriverGetVersion(driverVersion); });
}

extern "C" cudaError_t cudaRuntimeGetVersion(int *runtimeVersion) {
  if (runtimeVersion == nullptr) {
    return lcudart_fail(cudaErrorInvalidValue);
  }
  *runtimeVersion = CUDA_VERSION;
  return cudaSuccess;
}

extern "C" cudaError_t cudaGetDriverEntryPointByVersion(
    const char *symbol, void **funcPtr, unsigned int cudaVersion,
    unsigned long long flags,
    enum cudaDriverEntryPointQueryResult *driverStatus) {
  if (symbol == nullptr || funcPtr == nullptr) {
    return lcudart_fail(cudaErrorInvalidValue);
  }
  CUresult r = cuGetProcAddress_v2(
      symbol, funcPtr, static_cast<int>(cudaVersion),
      static_cast<cuuint64_t>(flags),
      reinterpret_cast<CUdriverProcAddressQueryResult *>(driverStatus));
  return lcudart_from_cu(r);
}

// ---------------------------------------------------------------------------
// Memory
// ---------------------------------------------------------------------------

extern "C" cudaError_t cudaMalloc(void **devPtr, size_t size) {
  if (devPtr == nullptr) {
    return lcudart_fail(cudaErrorInvalidValue);
  }
  return lcudart_call([&]() -> CUresult {
    CUdeviceptr p = 0;
    CUresult r = cuMemAlloc_v2(&p, size);
    if (r != CUDA_SUCCESS) {
      return r;
    }
    *devPtr = reinterpret_cast<void *>(p);
    lcudart_state &s = lcudart();
    std::lock_guard<std::mutex> lock(s.mutex);
    s.device_allocs[*devPtr] = {size, lcudart_current_device()};
    return CUDA_SUCCESS;
  });
}

extern "C" cudaError_t cudaFree(void *devPtr) {
  return lcudart_call([&]() -> CUresult {
    CUresult r = cuMemFree_v2(reinterpret_cast<CUdeviceptr>(devPtr));
    if (r == CUDA_SUCCESS) {
      lcudart_state &s = lcudart();
      std::lock_guard<std::mutex> lock(s.mutex);
      s.device_allocs.erase(devPtr);
    }
    return r;
  });
}

extern "C" cudaError_t cudaMallocAsync(void **devPtr, size_t size,
                                       cudaStream_t hStream) {
  if (devPtr == nullptr) {
    return lcudart_fail(cudaErrorInvalidValue);
  }
  return lcudart_call([&]() -> CUresult {
    CUdeviceptr p = 0;
    CUresult r = cuMemAllocAsync(&p, size, hStream);
    if (r != CUDA_SUCCESS) {
      return r;
    }
    *devPtr = reinterpret_cast<void *>(p);
    lcudart_state &s = lcudart();
    std::lock_guard<std::mutex> lock(s.mutex);
    s.device_allocs[*devPtr] = {size, lcudart_current_device()};
    return CUDA_SUCCESS;
  });
}

extern "C" cudaError_t cudaFreeAsync(void *devPtr, cudaStream_t hStream) {
  return lcudart_call([&]() -> CUresult {
    CUresult r = cuMemFreeAsync(reinterpret_cast<CUdeviceptr>(devPtr), hStream);
    if (r == CUDA_SUCCESS) {
      lcudart_state &s = lcudart();
      std::lock_guard<std::mutex> lock(s.mutex);
      s.device_allocs.erase(devPtr);
    }
    return r;
  });
}

extern "C" cudaError_t cudaMemGetInfo(size_t *free, size_t *total) {
  if (free == nullptr || total == nullptr) {
    return lcudart_fail(cudaErrorInvalidValue);
  }
  return lcudart_call([&] { return cuMemGetInfo_v2(free, total); });
}

extern "C" cudaError_t cudaHostAlloc(void **pHost, size_t size,
                                     unsigned int flags) {
  if (pHost == nullptr) {
    return lcudart_fail(cudaErrorInvalidValue);
  }
  return lcudart_call([&]() -> CUresult {
    void *host = nullptr;
    CUresult r = cuMemAllocHost_v2(&host, size);
    if (r != CUDA_SUCCESS) {
      return r;
    }
    void *device_ptr = nullptr;
    if (flags & cudaHostAllocMapped) {
      CUdeviceptr dptr = 0;
      if (cuMemHostGetDevicePointer(&dptr, host, 0) == CUDA_SUCCESS) {
        device_ptr = reinterpret_cast<void *>(dptr);
      }
    }
    *pHost = host;
    lcudart_state &s = lcudart();
    std::lock_guard<std::mutex> lock(s.mutex);
    s.host_allocs[host] = {size, flags, device_ptr};
    return CUDA_SUCCESS;
  });
}

extern "C" cudaError_t cudaFreeHost(void *ptr) {
  return lcudart_call([&] {
    CUresult r = cuMemFreeHost(ptr);
    if (r == CUDA_SUCCESS) {
      lcudart_state &s = lcudart();
      std::lock_guard<std::mutex> lock(s.mutex);
      s.host_allocs.erase(ptr);
    }
    return r;
  });
}

extern "C" cudaError_t cudaHostRegister(void *ptr, size_t size,
                                        unsigned int flags) {
  return lcudart_call([&] {
    CUresult r = cuMemHostRegister(ptr, size, flags);
    if (r == CUDA_SUCCESS) {
      lcudart_state &s = lcudart();
      std::lock_guard<std::mutex> lock(s.mutex);
      s.registered_ranges.push_back({ptr, size});
    }
    return r;
  });
}

extern "C" cudaError_t cudaHostUnregister(void *ptr) {
  return lcudart_call([&] {
    CUresult r = cuMemHostUnregister(ptr);
    if (r == CUDA_SUCCESS) {
      lcudart_state &s = lcudart();
      std::lock_guard<std::mutex> lock(s.mutex);
      for (auto it = s.registered_ranges.begin();
           it != s.registered_ranges.end();) {
        if (it->base == ptr) {
          it = s.registered_ranges.erase(it);
        } else {
          ++it;
        }
      }
    }
    return r;
  });
}

// Pointer classification for cudaMemcpyDefault / cudaPointerGetAttributes.
enum lcudart_ptr_kind { LCUDART_PTR_UNKNOWN, LCUDART_PTR_DEVICE, LCUDART_PTR_HOST };

static lcudart_ptr_kind lcudart_classify(const void *ptr) {
  lcudart_state &s = lcudart();
  std::lock_guard<std::mutex> lock(s.mutex);
  if (s.device_allocs.find(ptr) != s.device_allocs.end()) {
    return LCUDART_PTR_DEVICE;
  }
  if (s.host_allocs.find(ptr) != s.host_allocs.end()) {
    return LCUDART_PTR_HOST;
  }
  for (const auto &range : s.registered_ranges) {
    if (ptr >= range.base &&
        ptr < static_cast<char *>(range.base) + range.size) {
      return LCUDART_PTR_HOST;
    }
  }
  return LCUDART_PTR_UNKNOWN;
}

extern "C" cudaError_t cudaMemcpy(void *dst, const void *src, size_t count,
                                  enum cudaMemcpyKind kind) {
  cudaError_t e = lcudart_ready();
  if (e != cudaSuccess) {
    return e;
  }
  if (kind == cudaMemcpyDefault) {
    lcudart_ptr_kind d = lcudart_classify(dst);
    lcudart_ptr_kind s = lcudart_classify(src);
    if (d == LCUDART_PTR_DEVICE && s == LCUDART_PTR_HOST) {
      kind = cudaMemcpyHostToDevice;
    } else if (d == LCUDART_PTR_HOST && s == LCUDART_PTR_DEVICE) {
      kind = cudaMemcpyDeviceToHost;
    } else if (d == LCUDART_PTR_DEVICE && s == LCUDART_PTR_DEVICE) {
      kind = cudaMemcpyDeviceToDevice;
    } else if (d == LCUDART_PTR_HOST && s == LCUDART_PTR_HOST) {
      kind = cudaMemcpyHostToHost;
    } else {
      return lcudart_fail(cudaErrorInvalidValue);
    }
  }
  switch (kind) {
  case cudaMemcpyHostToHost:
    if (count != 0 && (dst == nullptr || src == nullptr)) {
      return lcudart_fail(cudaErrorInvalidValue);
    }
    memmove(dst, src, count);
    return cudaSuccess;
  case cudaMemcpyHostToDevice:
    return lcudart_from_cu(cuMemcpyHtoD_v2(
        reinterpret_cast<CUdeviceptr>(dst), src, count));
  case cudaMemcpyDeviceToHost:
    return lcudart_from_cu(
        cuMemcpyDtoH_v2(dst, reinterpret_cast<CUdeviceptr>(src), count));
  case cudaMemcpyDeviceToDevice:
    return lcudart_from_cu(cuMemcpyDtoD_v2(
        reinterpret_cast<CUdeviceptr>(dst),
        reinterpret_cast<CUdeviceptr>(src), count));
  default:
    return lcudart_fail(cudaErrorInvalidValue);
  }
}

extern "C" cudaError_t cudaMemcpyAsync(void *dst, const void *src,
                                       size_t count, enum cudaMemcpyKind kind,
                                       cudaStream_t stream) {
  cudaError_t e = lcudart_ready();
  if (e != cudaSuccess) {
    return e;
  }
  if (kind == cudaMemcpyDefault) {
    lcudart_ptr_kind d = lcudart_classify(dst);
    lcudart_ptr_kind s = lcudart_classify(src);
    if (d == LCUDART_PTR_DEVICE && s == LCUDART_PTR_HOST) {
      kind = cudaMemcpyHostToDevice;
    } else if (d == LCUDART_PTR_HOST && s == LCUDART_PTR_DEVICE) {
      kind = cudaMemcpyDeviceToHost;
    } else if (d == LCUDART_PTR_DEVICE && s == LCUDART_PTR_DEVICE) {
      kind = cudaMemcpyDeviceToDevice;
    } else if (d == LCUDART_PTR_HOST && s == LCUDART_PTR_HOST) {
      kind = cudaMemcpyHostToHost;
    } else {
      return lcudart_fail(cudaErrorInvalidValue);
    }
  }
  switch (kind) {
  case cudaMemcpyHostToHost:
    // libcudart implements HtoH async as a synchronous copy.
    if (count != 0 && (dst == nullptr || src == nullptr)) {
      return lcudart_fail(cudaErrorInvalidValue);
    }
    memmove(dst, src, count);
    return cudaSuccess;
  case cudaMemcpyHostToDevice:
    return lcudart_from_cu(cuMemcpyHtoDAsync_v2(
        reinterpret_cast<CUdeviceptr>(dst), src, count, stream));
  case cudaMemcpyDeviceToHost:
    return lcudart_from_cu(cuMemcpyDtoHAsync_v2(
        dst, reinterpret_cast<CUdeviceptr>(src), count, stream));
  case cudaMemcpyDeviceToDevice:
    return lcudart_from_cu(cuMemcpyDtoDAsync_v2(
        reinterpret_cast<CUdeviceptr>(dst),
        reinterpret_cast<CUdeviceptr>(src), count, stream));
  default:
    return lcudart_fail(cudaErrorInvalidValue);
  }
}

extern "C" cudaError_t cudaMemcpyPeerAsync(void *dst, int dstDevice,
                                           const void *src, int srcDevice,
                                           size_t count, cudaStream_t stream) {
  cudaError_t e = lcudart_ready();
  if (e != cudaSuccess) {
    return e;
  }
  CUdevice cu_dst = lcudart_cu_device(dstDevice, &e);
  CUdevice cu_src = lcudart_cu_device(srcDevice, &e);
  if (e != cudaSuccess) {
    return lcudart_fail(e);
  }
  // The driver form takes contexts, not devices.
  CUcontext dst_ctx = nullptr, src_ctx = nullptr;
  CUresult r = cuDevicePrimaryCtxRetain(&dst_ctx, cu_dst);
  if (r != CUDA_SUCCESS) {
    return lcudart_from_cu(r);
  }
  r = cuDevicePrimaryCtxRetain(&src_ctx, cu_src);
  if (r != CUDA_SUCCESS) {
    return lcudart_from_cu(r);
  }
  return lcudart_from_cu(cuMemcpyPeerAsync(reinterpret_cast<CUdeviceptr>(dst),
                                           dst_ctx,
                                           reinterpret_cast<CUdeviceptr>(src),
                                           src_ctx, count, stream));
}

extern "C" cudaError_t cudaMemcpyToSymbol(const void *symbol, const void *src,
                                          size_t count, size_t offset,
                                          enum cudaMemcpyKind kind) {
  cudaError_t e = lcudart_ready();
  if (e != cudaSuccess) {
    return e;
  }
  size_t sym_size = 0;
  CUdeviceptr dptr = lcudart_var_for(symbol, lcudart_current_device(),
                                     &sym_size, &e);
  if (e != cudaSuccess) {
    return lcudart_fail(e);
  }
  if (offset + count > sym_size) {
    return lcudart_fail(cudaErrorInvalidValue);
  }
  if (kind == cudaMemcpyDeviceToHost) {
    return lcudart_from_cu(cuMemcpyDtoH_v2(
        const_cast<void *>(src), dptr + offset, count));
  }
  return lcudart_from_cu(
      cuMemcpyHtoD_v2(dptr + offset, src, count));
}

extern "C" cudaError_t cudaMemcpyFromSymbol(void *dst, const void *symbol,
                                            size_t count, size_t offset,
                                            enum cudaMemcpyKind kind) {
  cudaError_t e = lcudart_ready();
  if (e != cudaSuccess) {
    return e;
  }
  size_t sym_size = 0;
  CUdeviceptr dptr = lcudart_var_for(symbol, lcudart_current_device(),
                                     &sym_size, &e);
  if (e != cudaSuccess) {
    return lcudart_fail(e);
  }
  if (offset + count > sym_size) {
    return lcudart_fail(cudaErrorInvalidValue);
  }
  if (kind == cudaMemcpyHostToDevice) {
    return lcudart_from_cu(
        cuMemcpyHtoD_v2(dptr + offset, dst, count));
  }
  return lcudart_from_cu(cuMemcpyDtoH_v2(dst, dptr + offset, count));
}

extern "C" cudaError_t cudaMemset(void *devPtr, int value, size_t count) {
  return lcudart_call([&] {
    return cuMemsetD8_v2(reinterpret_cast<CUdeviceptr>(devPtr),
                         static_cast<unsigned char>(value), count);
  });
}

extern "C" cudaError_t cudaMemsetAsync(void *devPtr, int value, size_t count,
                                       cudaStream_t stream) {
  return lcudart_call([&] {
    return cuMemsetD8Async(reinterpret_cast<CUdeviceptr>(devPtr),
                           static_cast<unsigned char>(value), count, stream);
  });
}

extern "C" cudaError_t cudaGetSymbolAddress(void **devPtr,
                                            const void *symbol) {
  if (devPtr == nullptr) {
    return lcudart_fail(cudaErrorInvalidValue);
  }
  cudaError_t e = lcudart_ready();
  if (e != cudaSuccess) {
    return e;
  }
  size_t size = 0;
  CUdeviceptr dptr = lcudart_var_for(symbol, lcudart_current_device(), &size,
                                     &e);
  if (e != cudaSuccess) {
    return lcudart_fail(e);
  }
  *devPtr = reinterpret_cast<void *>(dptr);
  return cudaSuccess;
}

extern "C" cudaError_t cudaPointerGetAttributes(
    struct cudaPointerAttributes *attributes, const void *ptr) {
  if (attributes == nullptr) {
    return lcudart_fail(cudaErrorInvalidValue);
  }
  memset(attributes, 0, sizeof(*attributes));
  lcudart_state &s = lcudart();
  std::lock_guard<std::mutex> lock(s.mutex);
  auto da = s.device_allocs.find(ptr);
  if (da != s.device_allocs.end()) {
    attributes->type = cudaMemoryTypeDevice;
    attributes->device = da->second.device;
    attributes->devicePointer = const_cast<void *>(ptr);
    attributes->hostPointer = nullptr;
    return cudaSuccess;
  }
  auto ha = s.host_allocs.find(ptr);
  if (ha != s.host_allocs.end()) {
    attributes->type = cudaMemoryTypeHost;
    attributes->device = lcudart_current_device();
    attributes->devicePointer = ha->second.device_ptr;
    attributes->hostPointer = const_cast<void *>(ptr);
    return cudaSuccess;
  }
  for (const auto &range : s.registered_ranges) {
    if (ptr >= range.base &&
        ptr < static_cast<char *>(range.base) + range.size) {
      attributes->type = cudaMemoryTypeHost;
      attributes->device = lcudart_current_device();
      attributes->devicePointer = const_cast<void *>(ptr);
      attributes->hostPointer = const_cast<void *>(ptr);
      return cudaSuccess;
    }
  }
  // Modern libcudart reports unregistered pointers as success.
  attributes->type = cudaMemoryTypeUnregistered;
  attributes->device = -1;
  attributes->devicePointer = nullptr;
  attributes->hostPointer = const_cast<void *>(ptr);
  return cudaSuccess;
}

extern "C" cudaError_t cudaMemPoolSetAccess(
    cudaMemPool_t memPool, const struct cudaMemAccessDesc *descList,
    size_t count) {
  return lcudart_call([&] {
    return cuMemPoolSetAccess(
        memPool, reinterpret_cast<const CUmemAccessDesc *>(descList), count);
  });
}

extern "C" cudaError_t cudaMemPoolSetAttribute(cudaMemPool_t memPool,
                                               enum cudaMemPoolAttr attr,
                                               void *value) {
  return lcudart_call([&] {
    return cuMemPoolSetAttribute(memPool,
                                 static_cast<CUmemPool_attribute>(attr), value);
  });
}

extern "C" cudaError_t cudaMemPoolGetAttribute(cudaMemPool_t memPool,
                                               enum cudaMemPoolAttr attr,
                                               void *value) {
  return lcudart_call([&] {
    return cuMemPoolGetAttribute(memPool,
                                 static_cast<CUmemPool_attribute>(attr), value);
  });
}

extern "C" cudaError_t cudaMemPoolTrimTo(cudaMemPool_t memPool,
                                         size_t minBytesToKeep) {
  return lcudart_call([&] { return cuMemPoolTrimTo(memPool, minBytesToKeep); });
}

// ---------------------------------------------------------------------------
// Streams & events
// ---------------------------------------------------------------------------

extern "C" cudaError_t cudaStreamCreateWithFlags(cudaStream_t *pStream,
                                                 unsigned int flags) {
  return lcudart_call([&] { return cuStreamCreate(pStream, flags); });
}

extern "C" cudaError_t cudaStreamCreateWithPriority(cudaStream_t *pStream,
                                                    unsigned int flags,
                                                    int priority) {
  return lcudart_call(
      [&] { return cuStreamCreateWithPriority(pStream, flags, priority); });
}

extern "C" cudaError_t cudaStreamDestroy(cudaStream_t stream) {
  return lcudart_call([&] { return cuStreamDestroy(stream); });
}

extern "C" cudaError_t cudaStreamSynchronize(cudaStream_t stream) {
  return lcudart_call([&] { return cuStreamSynchronize(stream); });
}

extern "C" cudaError_t cudaStreamQuery(cudaStream_t stream) {
  // NOT_READY stays sticky, matching libcudart.
  return lcudart_call([&] { return cuStreamQuery(stream); });
}

extern "C" cudaError_t cudaStreamWaitEvent(cudaStream_t stream,
                                           cudaEvent_t event,
                                           unsigned int flags) {
  return lcudart_call([&] { return cuStreamWaitEvent(stream, event, flags); });
}

extern "C" cudaError_t cudaStreamAddCallback(cudaStream_t stream,
                                             cudaStreamCallback_t callback,
                                             void *userData,
                                             unsigned int flags) {
  return lcudart_call([&] {
    return cuStreamAddCallback(
        stream, reinterpret_cast<CUstreamCallback>(callback), userData, flags);
  });
}

extern "C" cudaError_t cudaStreamBeginCapture(cudaStream_t stream,
                                              enum cudaStreamCaptureMode mode) {
  return lcudart_call([&] {
    return cuStreamBeginCapture(stream, static_cast<CUstreamCaptureMode>(mode));
  });
}

extern "C" cudaError_t cudaStreamBeginCaptureToGraph(
    cudaStream_t stream, cudaGraph_t graph, const cudaGraphNode_t *dependencies,
    const cudaGraphEdgeData *dependencyData, size_t numDependencies,
    enum cudaStreamCaptureMode mode) {
  return lcudart_call([&] {
    return cuStreamBeginCaptureToGraph(
        stream, graph, reinterpret_cast<const CUgraphNode *>(dependencies),
        reinterpret_cast<const CUgraphEdgeData *>(dependencyData),
        numDependencies, static_cast<CUstreamCaptureMode>(mode));
  });
}

extern "C" cudaError_t cudaStreamEndCapture(cudaStream_t stream,
                                            cudaGraph_t *pGraph) {
  return lcudart_call([&] { return cuStreamEndCapture(stream, pGraph); });
}

extern "C" cudaError_t cudaStreamGetCaptureInfo(
    cudaStream_t stream, enum cudaStreamCaptureStatus *captureStatus_out,
    unsigned long long *id_out, cudaGraph_t *graph_out,
    const cudaGraphNode_t **dependencies_out,
    const cudaGraphEdgeData **edgeData_out, size_t *numDependencies_out) {
  return lcudart_call([&] {
    return cuStreamGetCaptureInfo_v3(
        stream, reinterpret_cast<CUstreamCaptureStatus *>(captureStatus_out),
        reinterpret_cast<cuuint64_t *>(id_out), graph_out,
        reinterpret_cast<const CUgraphNode **>(dependencies_out),
        reinterpret_cast<const CUgraphEdgeData **>(edgeData_out),
        numDependencies_out);
  });
}

extern "C" cudaError_t cudaStreamIsCapturing(
    cudaStream_t stream, enum cudaStreamCaptureStatus *pCaptureStatus) {
  return lcudart_call([&] {
    return cuStreamIsCapturing(
        stream, reinterpret_cast<CUstreamCaptureStatus *>(pCaptureStatus));
  });
}

extern "C" cudaError_t cudaStreamUpdateCaptureDependencies(
    cudaStream_t stream, cudaGraphNode_t *dependencies,
    const cudaGraphEdgeData *dependencyData, size_t numDependencies,
    unsigned int flags) {
  return lcudart_call([&] {
    return cuStreamUpdateCaptureDependencies_v2(
        stream, reinterpret_cast<CUgraphNode *>(dependencies),
        reinterpret_cast<const CUgraphEdgeData *>(dependencyData),
        numDependencies, flags);
  });
}

extern "C" cudaError_t cudaThreadExchangeStreamCaptureMode(
    enum cudaStreamCaptureMode *mode) {
  return lcudart_call([&] {
    return cuThreadExchangeStreamCaptureMode(
        reinterpret_cast<CUstreamCaptureMode *>(mode));
  });
}

extern "C" cudaError_t cudaEventCreate(cudaEvent_t *event) {
  return lcudart_call([&] { return cuEventCreate(event, CU_EVENT_DEFAULT); });
}

extern "C" cudaError_t cudaEventCreateWithFlags(cudaEvent_t *event,
                                                unsigned int flags) {
  return lcudart_call([&] { return cuEventCreate(event, flags); });
}

extern "C" cudaError_t cudaEventDestroy(cudaEvent_t event) {
  return lcudart_call([&] { return cuEventDestroy(event); });
}

extern "C" cudaError_t cudaEventRecord(cudaEvent_t event, cudaStream_t stream) {
  return lcudart_call([&] { return cuEventRecord(event, stream); });
}

extern "C" cudaError_t cudaEventRecordWithFlags(cudaEvent_t event,
                                                cudaStream_t stream,
                                                unsigned int flags) {
  return lcudart_call(
      [&] { return cuEventRecordWithFlags(event, stream, flags); });
}

extern "C" cudaError_t cudaEventQuery(cudaEvent_t event) {
  return lcudart_call([&] { return cuEventQuery(event); });
}

extern "C" cudaError_t cudaEventSynchronize(cudaEvent_t event) {
  return lcudart_call([&] { return cuEventSynchronize(event); });
}

extern "C" cudaError_t cudaEventElapsedTime(float *ms, cudaEvent_t start,
                                            cudaEvent_t end) {
  return lcudart_call([&] { return cuEventElapsedTime_v2(ms, start, end); });
}

// ---------------------------------------------------------------------------
// Launch
// ---------------------------------------------------------------------------

static cudaError_t lcudart_launch(const void *func, dim3 gridDim,
                                  dim3 blockDim, void **args,
                                  size_t sharedMem, cudaStream_t stream) {
  cudaError_t e = lcudart_ready();
  if (e != cudaSuccess) {
    return e;
  }
  if (lcudart_trace_enabled()) {
    fprintf(stderr, "[lupine_cudart] launch func=%p grid=%u,%u,%u block=%u,%u,%u "
                    "shmem=%zu stream=%p\n",
            func, gridDim.x, gridDim.y, gridDim.z, blockDim.x, blockDim.y,
            blockDim.z, sharedMem, (void *)stream);
  }
  // Minted kernel objects in Library mode carry a CUkernel, which the
  // shim's cuLaunchKernel accepts natively. Module mode resolves per
  // current device.
  CUfunction f = nullptr;
  {
    lcudart_state &s = lcudart();
    std::lock_guard<std::mutex> lock(s.mutex);
    auto it = s.kernel_objects.find(func);
    if (it != s.kernel_objects.end() &&
        it->second.mode == lcudart_state::KernelMode::Library) {
      f = reinterpret_cast<CUfunction>(it->second.kernel);
    }
  }
  if (f == nullptr) {
    f = lcudart_resolve(func, lcudart_current_device(), &e);
    if (f == nullptr) {
      if (lcudart_trace_enabled()) {
        fprintf(stderr, "[lupine_cudart] launch resolve failed: %d\n", (int)e);
      }
      return lcudart_fail(e);
    }
  }
  CUresult launch_r = cuLaunchKernel(
      f, gridDim.x, gridDim.y, gridDim.z, blockDim.x, blockDim.y, blockDim.z,
      static_cast<unsigned int>(sharedMem), stream, args, nullptr);
  if (lcudart_trace_enabled()) {
    fprintf(stderr, "[lupine_cudart] cuLaunchKernel -> %d\n", (int)launch_r);
  }
  return lcudart_from_cu(launch_r);
}

extern "C" cudaError_t cudaLaunchKernel(const void *func, dim3 gridDim,
                                        dim3 blockDim, void **args,
                                        size_t sharedMem,
                                        cudaStream_t stream) {
  return lcudart_launch(func, gridDim, blockDim, args, sharedMem, stream);
}

// CUDA 12.x/13.x launch protocol: the device stub pops its own call
// configuration and passes the kernel handle (from __cudaGetKernel) plus the
// full launch geometry here. (crt/device_functions.h)
extern "C" cudaError_t __cudaLaunchKernel(cudaKernel_t kernel, dim3 gridDim,
                                          dim3 blockDim, void **args,
                                          size_t sharedMem,
                                          cudaStream_t stream) {
  return lcudart_launch(reinterpret_cast<const void *>(kernel), gridDim,
                        blockDim, args, sharedMem, stream);
}

// Per-thread-default-stream variant; same signature.
extern "C" cudaError_t __cudaLaunchKernel_ptsz(cudaKernel_t kernel,
                                             dim3 gridDim, dim3 blockDim,
                                             void **args, size_t sharedMem,
                                             cudaStream_t stream) {
  return lcudart_launch(reinterpret_cast<const void *>(kernel), gridDim,
                        blockDim, args, sharedMem, stream);
}

extern "C" cudaError_t cudaLaunchKernelExC(const cudaLaunchConfig_t *config,
                                           const void *func, void **args) {
  if (config == nullptr) {
    return lcudart_fail(cudaErrorInvalidValue);
  }
  cudaError_t e = lcudart_ready();
  if (e != cudaSuccess) {
    return e;
  }
  CUfunction f = nullptr;
  {
    lcudart_state &s = lcudart();
    std::lock_guard<std::mutex> lock(s.mutex);
    auto it = s.kernel_objects.find(func);
    if (it != s.kernel_objects.end() &&
        it->second.mode == lcudart_state::KernelMode::Library) {
      f = reinterpret_cast<CUfunction>(it->second.kernel);
    }
  }
  if (f == nullptr) {
    f = lcudart_resolve(func, lcudart_current_device(), &e);
    if (f == nullptr) {
      return lcudart_fail(e);
    }
  }
  CUlaunchConfig cfg = {};
  cfg.gridDimX = config->gridDim.x;
  cfg.gridDimY = config->gridDim.y;
  cfg.gridDimZ = config->gridDim.z;
  cfg.blockDimX = config->blockDim.x;
  cfg.blockDimY = config->blockDim.y;
  cfg.blockDimZ = config->blockDim.z;
  cfg.sharedMemBytes = static_cast<unsigned int>(config->dynamicSmemBytes);
  cfg.hStream = config->stream;
  cfg.attrs = reinterpret_cast<CUlaunchAttribute *>(config->attrs);
  cfg.numAttrs = config->numAttrs;
  return lcudart_from_cu(cuLaunchKernelEx(&cfg, f, args, nullptr));
}

extern "C" cudaError_t cudaFuncGetAttributes(struct cudaFuncAttributes *attr,
                                             const void *func) {
  if (attr == nullptr) {
    return lcudart_fail(cudaErrorInvalidValue);
  }
  cudaError_t e = lcudart_ready();
  if (e != cudaSuccess) {
    return e;
  }
  CUfunction f = lcudart_resolve(func, lcudart_current_device(), &e);
  if (f == nullptr) {
    return lcudart_fail(e);
  }
  memset(attr, 0, sizeof(*attr));
  int v = 0;
  CUresult r;
  if ((r = cuFuncGetAttribute(&v, CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES, f)) ==
      CUDA_SUCCESS) {
    attr->sharedSizeBytes = v;
  }
  if ((r = cuFuncGetAttribute(&v, CU_FUNC_ATTRIBUTE_CONST_SIZE_BYTES, f)) ==
      CUDA_SUCCESS) {
    attr->constSizeBytes = v;
  }
  if ((r = cuFuncGetAttribute(&v, CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES, f)) ==
      CUDA_SUCCESS) {
    attr->localSizeBytes = v;
  }
  if ((r = cuFuncGetAttribute(&v, CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK,
                              f)) == CUDA_SUCCESS) {
    attr->maxThreadsPerBlock = v;
  }
  if ((r = cuFuncGetAttribute(&v, CU_FUNC_ATTRIBUTE_NUM_REGS, f)) ==
      CUDA_SUCCESS) {
    attr->numRegs = v;
  }
  if ((r = cuFuncGetAttribute(&v, CU_FUNC_ATTRIBUTE_PTX_VERSION, f)) ==
      CUDA_SUCCESS) {
    attr->ptxVersion = v;
  }
  if ((r = cuFuncGetAttribute(&v, CU_FUNC_ATTRIBUTE_BINARY_VERSION, f)) ==
      CUDA_SUCCESS) {
    attr->binaryVersion = v;
  }
  if ((r = cuFuncGetAttribute(&v, CU_FUNC_ATTRIBUTE_CACHE_MODE_CA, f)) ==
      CUDA_SUCCESS) {
    attr->cacheModeCA = v;
  }
  if ((r = cuFuncGetAttribute(&v, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                              f)) == CUDA_SUCCESS) {
    attr->maxDynamicSharedSizeBytes = v;
  }
  if ((r = cuFuncGetAttribute(
           &v, CU_FUNC_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT, f)) ==
      CUDA_SUCCESS) {
    attr->preferredShmemCarveout = v;
  }
  return cudaSuccess;
}

extern "C" cudaError_t cudaFuncSetAttribute(const void *func,
                                            enum cudaFuncAttribute attr,
                                            int value) {
  cudaError_t e = lcudart_ready();
  if (e != cudaSuccess) {
    return e;
  }
  CUfunction f = lcudart_resolve(func, lcudart_current_device(), &e);
  if (f == nullptr) {
    return lcudart_fail(e);
  }
  // Settable runtime attrs (8-15) are numerically identical to the driver's.
  return lcudart_from_cu(
      cuFuncSetAttribute(f, static_cast<CUfunction_attribute>(attr), value));
}

extern "C" cudaError_t cudaKernelSetAttributeForDevice(cudaKernel_t kernel,
                                                       enum cudaFuncAttribute attr,
                                                       int value, int device) {
  cudaError_t e = lcudart_ready();
  if (e != cudaSuccess) {
    return e;
  }
  lcudart_state &s = lcudart();
  std::lock_guard<std::mutex> lock(s.mutex);
  auto it = s.kernel_objects.find(kernel);
  if (it == s.kernel_objects.end()) {
    return lcudart_fail(cudaErrorInvalidDeviceFunction);
  }
  if (it->second.mode == lcudart_state::KernelMode::Library) {
    // cuKernelSetAttribute(attrib, val, kernel, dev)
    return lcudart_from_cu(cuKernelSetAttribute(
        static_cast<CUfunction_attribute>(attr), value, it->second.kernel,
        lcudart().devices[device]));
  }
  CUfunction f = lcudart_function_for(it->second.stub, device, &e);
  if (f == nullptr) {
    return lcudart_fail(e);
  }
  return lcudart_from_cu(
      cuFuncSetAttribute(f, static_cast<CUfunction_attribute>(attr), value));
}

extern "C" cudaError_t cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(
    int *numBlocks, const void *func, int blockSize, size_t dynamicSMemSize,
    unsigned int flags) {
  if (numBlocks == nullptr) {
    return lcudart_fail(cudaErrorInvalidValue);
  }
  cudaError_t e = lcudart_ready();
  if (e != cudaSuccess) {
    return e;
  }
  CUfunction f = lcudart_resolve(func, lcudart_current_device(), &e);
  if (f == nullptr) {
    return lcudart_fail(e);
  }
  return lcudart_from_cu(cuOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(
      numBlocks, f, blockSize, dynamicSMemSize, flags));
}

// ---------------------------------------------------------------------------
// Libraries (runtime PTX / cubin loading — server-side JIT)
// ---------------------------------------------------------------------------

extern "C" cudaError_t cudaLibraryLoadData(
    cudaLibrary_t *library, const void *code, enum cudaJitOption *jitOptions,
    void **jitOptionsValues, unsigned int numJitOptions,
    enum cudaLibraryOption *libraryOptions, void **libraryOptionValues,
    unsigned int numLibraryOptions) {
  if (library == nullptr) {
    return lcudart_fail(cudaErrorInvalidValue);
  }
  cudaError_t e = lcudart_ready();
  if (e != cudaSuccess) {
    return e;
  }
  return lcudart_from_cu(cuLibraryLoadData(
      library, code, reinterpret_cast<CUjit_option *>(jitOptions),
      jitOptionsValues, numJitOptions,
      reinterpret_cast<CUlibraryOption *>(libraryOptions),
      libraryOptionValues, numLibraryOptions));
}

extern "C" cudaError_t cudaLibraryGetKernel(cudaKernel_t *pKernel,
                                            cudaLibrary_t library,
                                            const char *name) {
  if (pKernel == nullptr) {
    return lcudart_fail(cudaErrorInvalidValue);
  }
  cudaError_t e = lcudart_ready();
  if (e != cudaSuccess) {
    return e;
  }
  CUkernel kernel = nullptr;
  CUresult r = cuLibraryGetKernel(&kernel, library, name);
  if (r != CUDA_SUCCESS) {
    return lcudart_from_cu(r);
  }
  // Wrap the driver kernel handle so launch/attribute paths can recognize it.
  auto *rec = new lcudart_state::KernelRec{
      lcudart_state::KernelMode::Library, kernel, nullptr, nullptr};
  lcudart_state &s = lcudart();
  std::lock_guard<std::mutex> lock(s.mutex);
  s.kernel_objects[rec] = *rec;
  *pKernel = reinterpret_cast<cudaKernel_t>(rec);
  return cudaSuccess;
}

extern "C" cudaError_t cudaLibraryUnload(cudaLibrary_t library) {
  return lcudart_call([&] { return cuLibraryUnload(library); });
}

// ---------------------------------------------------------------------------
// Graphs
// ---------------------------------------------------------------------------

extern "C" cudaError_t cudaGraphAddNode(
    cudaGraphNode_t *pGraphNode, cudaGraph_t graph,
    const cudaGraphNode_t *pDependencies, const cudaGraphEdgeData *dependencyData,
    size_t numDependencies, struct cudaGraphNodeParams *nodeParams) {
  cudaError_t e = lcudart_ready();
  if (e != cudaSuccess) {
    return e;
  }
  if (nodeParams == nullptr) {
    return lcudart_fail(cudaErrorInvalidValue);
  }
  // cudaGraphNodeParams and CUgraphNodeParams are layout-identical unions
  // (same header-generated members); zero + copy the runtime struct, then
  // patch the kernel member, whose function handle needs resolution.
  CUgraphNodeParams driver_params;
  memset(&driver_params, 0, sizeof(driver_params));
  memcpy(&driver_params, nodeParams,
         sizeof(driver_params) < sizeof(*nodeParams) ? sizeof(driver_params)
                                                    : sizeof(*nodeParams));
  if (nodeParams->type == cudaGraphNodeTypeKernel) {
    const struct cudaKernelNodeParamsV2 *kp = &nodeParams->kernel;
    CUDA_KERNEL_NODE_PARAMS_v3 &out = driver_params.kernel;
    out.func = nullptr;
    out.kern = nullptr;
    out.ctx = nullptr;
    const void *handle = kp->func; // host stub or minted kernel object
    bool resolved = false;
#if CUDA_VERSION >= 13030
    if (kp->functionType == cudaKernelFunctionTypeFunction) {
      out.func = reinterpret_cast<CUfunction>(const_cast<void *>(handle));
      resolved = true;
    }
#endif
    if (!resolved) {
      lcudart_state &s = lcudart();
      std::lock_guard<std::mutex> lock(s.mutex);
      auto it = s.kernel_objects.find(handle);
      if (it != s.kernel_objects.end()) {
        if (it->second.mode == lcudart_state::KernelMode::Library) {
          out.kern = it->second.kernel;
          resolved = true;
        } else {
          handle = it->second.stub;
          // resolve below without holding the lock
        }
      }
    }
    if (!resolved) {
      CUfunction f = lcudart_resolve(handle, lcudart_current_device(), &e);
      if (f == nullptr) {
        return lcudart_fail(e);
      }
      out.func = f;
    }
    out.gridDimX = kp->gridDim.x;
    out.gridDimY = kp->gridDim.y;
    out.gridDimZ = kp->gridDim.z;
    out.blockDimX = kp->blockDim.x;
    out.blockDimY = kp->blockDim.y;
    out.blockDimZ = kp->blockDim.z;
    out.sharedMemBytes = kp->sharedMemBytes;
    out.kernelParams = kp->kernelParams;
    out.extra = kp->extra;
  }
  return lcudart_from_cu(cuGraphAddNode(
      pGraphNode, graph, reinterpret_cast<const CUgraphNode *>(pDependencies),
      reinterpret_cast<const CUgraphEdgeData *>(dependencyData),
      numDependencies, &driver_params));
}

extern "C" cudaError_t cudaGraphConditionalHandleCreate(
    cudaGraphConditionalHandle *pHandle_out, cudaGraph_t graph,
    unsigned int defaultLaunchValue, unsigned int flags) {
  cudaError_t e = lcudart_ready();
  if (e != cudaSuccess) {
    return e;
  }
  CUcontext ctx = nullptr;
  (void)cuCtxGetCurrent(&ctx);
  return lcudart_from_cu(cuGraphConditionalHandleCreate(
      reinterpret_cast<CUgraphConditionalHandle *>(pHandle_out), graph, ctx,
      defaultLaunchValue, flags));
}

extern "C" cudaError_t cudaGraphDebugDotPrint(cudaGraph_t graph,
                                              const char *path,
                                              unsigned int flags) {
  return lcudart_call([&] { return cuGraphDebugDotPrint(graph, path, flags); });
}

extern "C" cudaError_t cudaGraphDestroy(cudaGraph_t graph) {
  return lcudart_call([&] { return cuGraphDestroy(graph); });
}

extern "C" cudaError_t cudaGraphExecDestroy(cudaGraphExec_t graphExec) {
  return lcudart_call([&] { return cuGraphExecDestroy(graphExec); });
}

extern "C" cudaError_t cudaGraphGetNodes(cudaGraph_t graph,
                                         cudaGraphNode_t *nodes,
                                         size_t *numNodes) {
  return lcudart_call([&] {
    return cuGraphGetNodes(graph, reinterpret_cast<CUgraphNode *>(nodes),
                           numNodes);
  });
}

extern "C" cudaError_t cudaGraphInstantiateWithFlags(cudaGraphExec_t *pGraphExec,
                                                     cudaGraph_t graph,
                                                     unsigned long long flags) {
  return lcudart_call(
      [&] { return cuGraphInstantiateWithFlags(pGraphExec, graph, flags); });
}

extern "C" cudaError_t cudaGraphLaunch(cudaGraphExec_t graphExec,
                                       cudaStream_t stream) {
  return lcudart_call([&] { return cuGraphLaunch(graphExec, stream); });
}

extern "C" cudaError_t cudaGraphNodeGetDependencies(
    cudaGraphNode_t node, cudaGraphNode_t *pDependencies,
    cudaGraphEdgeData *edgeData, size_t *pNumDependencies) {
  return lcudart_call([&] {
    return cuGraphNodeGetDependencies(
        node, reinterpret_cast<CUgraphNode *>(pDependencies),
        reinterpret_cast<CUgraphEdgeData *>(edgeData), pNumDependencies);
  });
}

// ---------------------------------------------------------------------------
// IPC (v1 stubs)
// ---------------------------------------------------------------------------

extern "C" cudaError_t cudaIpcGetMemHandle(cudaIpcMemHandle_t *handle,
                                           void *devPtr) {
  (void)handle;
  (void)devPtr;
  return lcudart_fail(cudaErrorNotSupported);
}

extern "C" cudaError_t cudaIpcCloseMemHandle(void *devPtr) {
  (void)devPtr;
  return lcudart_fail(cudaErrorNotSupported);
}

extern "C" cudaError_t cudaIpcOpenMemHandle(void **devPtr,
                                            cudaIpcMemHandle_t handle,
                                            unsigned int flags) {
  (void)devPtr;
  (void)handle;
  (void)flags;
  return lcudart_fail(cudaErrorNotSupported);
}

extern "C" cudaError_t cudaIpcGetEventHandle(cudaIpcEventHandle_t *handle,
                                             cudaEvent_t event) {
  (void)handle;
  (void)event;
  return lcudart_fail(cudaErrorNotSupported);
}

extern "C" cudaError_t cudaIpcOpenEventHandle(cudaEvent_t *event,
                                              cudaIpcEventHandle_t handle) {
  (void)event;
  (void)handle;
  return lcudart_fail(cudaErrorNotSupported);
}
