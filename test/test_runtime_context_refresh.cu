// A runtime call can leave the server lane holding a driver context the client
// never asked for, so the shim re-reads the lane's current context before the
// next driver-side decision. Three things ride on that read: whether CUDA
// counts as initialized, what the lane-context cache holds, and which server a
// library shim's call goes to. Nothing else pins them, so this does.
#include <cublas_v2.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cudaProfiler.h>

#include <cstdio>
#include <cstdlib>
#include <thread>
#include <vector>

extern "C" CUresult cuCtxCreate_v2(CUcontext *, unsigned int, CUdevice);

static int failures = 0;

static void fail(const char *message) {
  std::fprintf(stderr, "FAIL: %s\n", message);
  ++failures;
}

static void require(bool condition, const char *message) {
  if (!condition) {
    fail(message);
  }
}

static void require_fatal(bool condition, const char *message) {
  if (!condition) {
    std::fprintf(stderr, "FAIL: %s\n", message);
    std::exit(1);
  }
}

// Runtime traffic with no driver call in it, so the only thing that can have
// moved the lane's context is the runtime itself.
static void runtime_traffic() {
  void *buffer = nullptr;
  require_fatal(cudaMalloc(&buffer, 256) == cudaSuccess, "cudaMalloc failed");
  require_fatal(cudaMemset(buffer, 0, 256) == cudaSuccess, "cudaMemset failed");
  require_fatal(cudaFree(buffer) == cudaSuccess, "cudaFree failed");
}

static CUcontext current_context(const char *what) {
  CUcontext context = reinterpret_cast<CUcontext>(0x1);
  CUresult result = cuCtxGetCurrent(&context);
  if (result != CUDA_SUCCESS) {
    fail(what);
    return nullptr;
  }
  return context;
}

// --------------------------------------------------------------------------
// Initialization state
// --------------------------------------------------------------------------

// Nothing here calls cuInit. The runtime's own initialization is what makes
// the client-answered driver entry points stop reporting NOT_INITIALIZED, and
// the only way the client learns of it is the context read.
static void check_initialized_without_cuinit() {
  require_fatal(cudaFree(nullptr) == cudaSuccess, "cudaFree(0) failed");
  CUcontext context = nullptr;
  require(cuCtxGetCurrent(&context) == CUDA_SUCCESS,
          "cuCtxGetCurrent reported NOT_INITIALIZED after runtime init");
  require(context != nullptr,
          "the runtime left no context current on the lane");
  CUdevice device = -1;
  require(cuCtxGetDevice(&device) == CUDA_SUCCESS,
          "cuCtxGetDevice reported NOT_INITIALIZED after runtime init");
  require(cuProfilerStart() == CUDA_SUCCESS,
          "cuProfilerStart reported NOT_INITIALIZED after runtime init");
  require(cuProfilerStop() == CUDA_SUCCESS,
          "cuProfilerStop reported NOT_INITIALIZED after runtime init");
  require(cuCtxPushCurrent(context) == CUDA_SUCCESS,
          "cuCtxPushCurrent reported NOT_INITIALIZED after runtime init");
  CUcontext popped = nullptr;
  require(cuCtxPopCurrent(&popped) == CUDA_SUCCESS && popped == context,
          "cuCtxPopCurrent did not return the pushed context");
}

// Initialization is a property of the process, not of the thread that did it:
// a thread that has never made a runtime call still gets past the gate, and
// reports the empty context its lane actually holds.
static void check_initialized_on_a_fresh_thread() {
  std::thread worker([] {
    CUcontext context = reinterpret_cast<CUcontext>(0x1);
    require(cuCtxGetCurrent(&context) == CUDA_SUCCESS,
            "a fresh thread reported NOT_INITIALIZED");
    runtime_traffic();
    require(current_context("fresh thread cuCtxGetCurrent failed") != nullptr,
            "a fresh thread's runtime traffic bound no context");
  });
  worker.join();
}

// --------------------------------------------------------------------------
// The lane-context cache
// --------------------------------------------------------------------------

// Runtime traffic must not move the context the lane holds, and the client
// must not decide it did.
static void check_context_survives_runtime_traffic() {
  CUcontext before = current_context("cuCtxGetCurrent failed");
  require_fatal(before != nullptr, "no context to hold steady");
  for (int i = 0; i < 20; ++i) {
    runtime_traffic();
    require(current_context("cuCtxGetCurrent failed under traffic") == before,
            "runtime traffic changed the context the client reports");
  }
  CUdevice device = -1;
  require(cuCtxGetDevice(&device) == CUDA_SUCCESS,
          "cuCtxGetDevice failed under traffic");
}

// The driver's own rebinders must still be seen through interleaved runtime
// traffic: each one leaves a different context current, and the client reports
// the one that is really there.
static void check_driver_rebinders_under_traffic(CUdevice driver_device) {
  CUcontext primary = current_context("cuCtxGetCurrent failed");
  require_fatal(primary != nullptr, "no primary context");

  for (int i = 0; i < 3; ++i) {
    CUcontext created = nullptr;
    require_fatal(cuCtxCreate_v2(&created, 0, driver_device) == CUDA_SUCCESS,
                  "cuCtxCreate failed");
    require(current_context("cuCtxGetCurrent failed") == created,
            "cuCtxCreate did not make its context current");
    runtime_traffic();
    require(current_context("cuCtxGetCurrent failed") == created,
            "runtime traffic displaced a created context");

    require_fatal(cuCtxPushCurrent(primary) == CUDA_SUCCESS,
                  "cuCtxPushCurrent failed");
    runtime_traffic();
    require(current_context("cuCtxGetCurrent failed") == primary,
            "runtime traffic displaced a pushed context");
    CUcontext popped = nullptr;
    require_fatal(cuCtxPopCurrent(&popped) == CUDA_SUCCESS,
                  "cuCtxPopCurrent failed");
    runtime_traffic();
    require(current_context("cuCtxGetCurrent failed") == created,
            "cuCtxPopCurrent did not restore the created context");

    require_fatal(cuCtxSetCurrent(primary) == CUDA_SUCCESS,
                  "cuCtxSetCurrent failed");
    require_fatal(cuCtxDestroy(created) == CUDA_SUCCESS, "cuCtxDestroy failed");
    runtime_traffic();
    require(current_context("cuCtxGetCurrent failed") == primary,
            "destroying a context disturbed the one left current");
  }
}

// --------------------------------------------------------------------------
// The three runtime rebinders
// --------------------------------------------------------------------------

static void check_set_device(int count) {
  for (int i = 0; i < 2 * count; ++i) {
    const int device = i % count;
    require_fatal(cudaSetDevice(device) == cudaSuccess, "cudaSetDevice failed");
    runtime_traffic();
    CUcontext context = current_context("cuCtxGetCurrent failed");
    require(context != nullptr, "cudaSetDevice bound no context");
    CUdevice reported = -1;
    require(cuCtxGetDevice(&reported) == CUDA_SUCCESS &&
                static_cast<int>(reported) == device,
            "the context current after cudaSetDevice belongs to another device");
  }
}

static void check_set_valid_devices(int count) {
  for (int i = 0; i < 2 * count; ++i) {
    int preference[2] = {i % count, (i + 1) % count};
    cudaError_t result = cudaSetValidDevices(preference, count > 1 ? 2 : 1);
    if (result != cudaSuccess) {
      std::printf("note: cudaSetValidDevices rejected here: %s\n",
                  cudaGetErrorName(result));
      return;
    }
    runtime_traffic();
    CUcontext context = current_context("cuCtxGetCurrent failed");
    require(context != nullptr, "no context current after cudaSetValidDevices");
    CUdevice reported = -1;
    require(cuCtxGetDevice(&reported) == CUDA_SUCCESS,
            "cuCtxGetDevice failed after cudaSetValidDevices");
  }
}

// A reset frees the primary context handle. The server is free to hand the
// same address back, so the test is that the client goes and asks rather than
// that the answer differs.
static void check_device_reset() {
  for (int i = 0; i < 4; ++i) {
    require_fatal(cudaSetDevice(0) == cudaSuccess, "cudaSetDevice(0) failed");
    runtime_traffic();
    require_fatal(cudaDeviceReset() == cudaSuccess, "cudaDeviceReset failed");
    runtime_traffic();
    CUcontext context = current_context("cuCtxGetCurrent failed after reset");
    require(context != nullptr, "no context current after a reset");
    CUdevice reported = -1;
    require(cuCtxGetDevice(&reported) == CUDA_SUCCESS &&
                static_cast<int>(reported) == 0,
            "the context after a reset belongs to another device");
    void *buffer = nullptr;
    require(cudaMalloc(&buffer, 256) == cudaSuccess,
            "the context after a reset is not usable");
    require(cudaFree(buffer) == cudaSuccess, "cudaFree failed after a reset");
  }
}

// --------------------------------------------------------------------------
// Route selection for a library shim
// --------------------------------------------------------------------------

static void check_library_routing(int count) {
  for (int device = 0; device < count; ++device) {
    require_fatal(cudaSetDevice(device) == cudaSuccess,
                  "cudaSetDevice failed");
    cublasHandle_t handle = nullptr;
    require_fatal(cublasCreate(&handle) == CUBLAS_STATUS_SUCCESS,
                  "cublasCreate failed");

    // Interleave the traffic that leaves a context read pending with the
    // library calls whose route depends on what that read returns.
    for (int i = 0; i < 5; ++i) {
      runtime_traffic();
      const int n = 4;
      float host[n] = {1.0f, 2.0f, 3.0f, 4.0f};
      float *values = nullptr;
      require_fatal(cudaMalloc(&values, sizeof(host)) == cudaSuccess,
                    "cudaMalloc failed");
      require_fatal(cudaMemcpy(values, host, sizeof(host),
                               cudaMemcpyHostToDevice) == cudaSuccess,
                    "cudaMemcpy failed");
      float alpha = 2.0f;
      require(cublasSscal(handle, n, &alpha, values, 1) ==
                  CUBLAS_STATUS_SUCCESS,
              "cublasSscal failed");
      require_fatal(cudaMemcpy(host, values, sizeof(host),
                               cudaMemcpyDeviceToHost) == cudaSuccess,
                    "cudaMemcpy back failed");
      require(host[0] == 2.0f && host[3] == 8.0f,
              "cublasSscal produced the wrong result");
      require_fatal(cudaFree(values) == cudaSuccess, "cudaFree failed");

      CUcontext context = current_context("cuCtxGetCurrent failed");
      require(context != nullptr, "library traffic lost the lane's context");
      CUdevice reported = -1;
      require(cuCtxGetDevice(&reported) == CUDA_SUCCESS &&
                  static_cast<int>(reported) == device,
              "library traffic moved the lane to another device");
    }
    require_fatal(cublasDestroy(handle) == CUBLAS_STATUS_SUCCESS,
                  "cublasDestroy failed");
  }
}

// --------------------------------------------------------------------------
// Graph memcpy nodes
// --------------------------------------------------------------------------

// A memcpy node is built against the context current on the lane, so adding
// one is a caller of that read like any other. Nothing else covers this entry
// point.
static void check_graph_memcpy_nodes(int count) {
  const int n = 16;
  for (int device = 0; device < count; ++device) {
    require_fatal(cudaSetDevice(device) == cudaSuccess, "cudaSetDevice failed");
    runtime_traffic();

    int *host = nullptr;
    int *result = nullptr;
    int *values = nullptr;
    require_fatal(cudaMallocHost(&host, n * sizeof(int)) == cudaSuccess,
                  "cudaMallocHost failed");
    require_fatal(cudaMallocHost(&result, n * sizeof(int)) == cudaSuccess,
                  "cudaMallocHost failed");
    require_fatal(cudaMalloc(&values, n * sizeof(int)) == cudaSuccess,
                  "cudaMalloc failed");
    for (int i = 0; i < n; ++i) {
      host[i] = device * 1000 + i;
      result[i] = -1;
    }

    cudaGraph_t graph = nullptr;
    require_fatal(cudaGraphCreate(&graph, 0) == cudaSuccess,
                  "cudaGraphCreate failed");
    cudaGraphNode_t upload = nullptr;
    require(cudaGraphAddMemcpyNode1D(&upload, graph, nullptr, 0, values, host,
                                     n * sizeof(int),
                                     cudaMemcpyHostToDevice) == cudaSuccess,
            "cudaGraphAddMemcpyNode1D failed for the upload");

    // The three-dimensional form, so the parameter path is covered too.
    cudaMemcpy3DParms params{};
    params.srcPtr = make_cudaPitchedPtr(values, n * sizeof(int),
                                        n * sizeof(int), 1);
    params.dstPtr = make_cudaPitchedPtr(result, n * sizeof(int),
                                        n * sizeof(int), 1);
    params.extent = make_cudaExtent(n * sizeof(int), 1, 1);
    params.kind = cudaMemcpyDeviceToHost;
    cudaGraphNode_t download = nullptr;
    require(cudaGraphAddMemcpyNode(&download, graph, &upload, 1, &params) ==
                cudaSuccess,
            "cudaGraphAddMemcpyNode failed for the download");

    cudaGraphExec_t exec = nullptr;
    require_fatal(cudaGraphInstantiate(&exec, graph, nullptr, nullptr, 0) ==
                      cudaSuccess,
                  "cudaGraphInstantiate failed");
    cudaStream_t stream = nullptr;
    require_fatal(cudaStreamCreate(&stream) == cudaSuccess,
                  "cudaStreamCreate failed");
    require_fatal(cudaGraphLaunch(exec, stream) == cudaSuccess,
                  "cudaGraphLaunch failed");
    require_fatal(cudaStreamSynchronize(stream) == cudaSuccess,
                  "cudaStreamSynchronize failed");
    for (int i = 0; i < n; ++i) {
      if (result[i] != host[i]) {
        fail("a graph memcpy node did not move the bytes");
        break;
      }
    }

    CUdevice reported = -1;
    require(cuCtxGetDevice(&reported) == CUDA_SUCCESS &&
                static_cast<int>(reported) == device,
            "building a graph moved the lane to another device");

    require_fatal(cudaStreamDestroy(stream) == cudaSuccess,
                  "cudaStreamDestroy failed");
    require_fatal(cudaGraphExecDestroy(exec) == cudaSuccess,
                  "cudaGraphExecDestroy failed");
    require_fatal(cudaGraphDestroy(graph) == cudaSuccess,
                  "cudaGraphDestroy failed");
    require_fatal(cudaFree(values) == cudaSuccess, "cudaFree failed");
    require_fatal(cudaFreeHost(host) == cudaSuccess, "cudaFreeHost failed");
    require_fatal(cudaFreeHost(result) == cudaSuccess, "cudaFreeHost failed");
  }
}

int main() {
  int count = 0;
  require_fatal(cudaGetDeviceCount(&count) == cudaSuccess && count > 0,
                "cudaGetDeviceCount failed");
  std::printf("devices=%d\n", count);

  check_initialized_without_cuinit();
  check_initialized_on_a_fresh_thread();

  require_fatal(cudaSetDevice(0) == cudaSuccess, "cudaSetDevice(0) failed");
  runtime_traffic();
  check_context_survives_runtime_traffic();

  CUdevice driver_device = 0;
  require_fatal(cuDeviceGet(&driver_device, 0) == CUDA_SUCCESS,
                "cuDeviceGet failed");
  check_driver_rebinders_under_traffic(driver_device);

  check_set_device(count);
  check_set_valid_devices(count);
  check_library_routing(count);
  check_graph_memcpy_nodes(count);
  check_device_reset();

  std::printf(failures == 0 ? "PASSED\n" : "FAILED\n");
  return failures == 0 ? 0 : 1;
}
