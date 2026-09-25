// Stream-zero variant of the capture/replay pattern in NVIDIA simpleCudaGraphs.
// Driver-only lookups also exercise both dlsym and cuGetProcAddress contracts.
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cuda.h>
#include <dlfcn.h>
#include <mutex>
#include <thread>

#if CUDA_VERSION < 12000
int main() {
  std::puts("SKIP: PTDS lookup and launch descriptor checks require CUDA 12.0");
}
#else

#define CHECK(call)                                                            \
  do {                                                                         \
    CUresult result = (call);                                                  \
    if (result != CUDA_SUCCESS) {                                              \
      std::fprintf(stderr, "%s:%d: %s -> %d\n", __FILE__, __LINE__, #call,     \
                   result);                                                    \
      std::abort();                                                            \
    }                                                                          \
  } while (0)
#define REQUIRE(value)                                                         \
  do {                                                                         \
    if (!(value)) {                                                            \
      std::fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, #value);         \
      std::abort();                                                            \
    }                                                                          \
  } while (0)

template <typename Fn>
Fn resolve(
    const char *name, int version,
    unsigned long long flags = CU_GET_PROC_ADDRESS_PER_THREAD_DEFAULT_STREAM) {
  void *pointer = nullptr;
  CUdriverProcAddressQueryResult status = CU_GET_PROC_ADDRESS_SYMBOL_NOT_FOUND;
  CHECK(cuGetProcAddress(name, &pointer, version, flags, &status));
  if (status != CU_GET_PROC_ADDRESS_SUCCESS || pointer == nullptr) {
    std::fprintf(stderr, "resolve %s version=%d flags=%llu status=%d\n", name,
                 version, flags, status);
    std::abort();
  }
  return reinterpret_cast<Fn>(pointer);
}
template <typename Fn> Fn symbol(const char *name) {
  void *pointer = dlsym(RTLD_DEFAULT, name);
  REQUIRE(pointer != nullptr);
  return reinterpret_cast<Fn>(pointer);
}
using Begin = CUresult (*)(CUstream, CUstreamCaptureMode);
using End = CUresult (*)(CUstream, CUgraph *);
using Sync = CUresult (*)(CUstream);
using Launch = CUresult (*)(CUgraphExec, CUstream);
using HtoD = CUresult (*)(CUdeviceptr, const void *, size_t, CUstream);
using DtoH = CUresult (*)(void *, CUdeviceptr, size_t, CUstream);
using Set = CUresult (*)(CUdeviceptr, unsigned int, size_t);
using Get = CUresult (*)(void *, CUdeviceptr, size_t);
using Id = CUresult (*)(CUstream, unsigned long long *);
using CaptureInfo = CUresult (*)(CUstream, CUstreamCaptureStatus *,
                                 cuuint64_t *);
using Host = CUresult (*)(CUstream, CUhostFn, void *);

struct Barrier {
  std::mutex mutex;
  std::condition_variable condition;
  unsigned generation = 0, count = 0;
  void wait() {
    std::unique_lock<std::mutex> lock(mutex);
    unsigned current = generation;
    if (++count == 2) {
      count = 0;
      ++generation;
      condition.notify_all();
    } else
      condition.wait(lock, [&] { return generation != current; });
  }
};

static void CUDA_CB increment(void *data) {
  static_cast<std::atomic<int> *>(data)->fetch_add(1);
}

static void CUDA_CB stream_callback(CUstream stream, CUresult status,
                                    void *data) {
  REQUIRE(stream == nullptr);
  REQUIRE(status == CUDA_SUCCESS);
  increment(data);
}

struct Gate {
  std::atomic<bool> entered{false}, release{false}, timed_out{false};
};
static void CUDA_CB block_stream(void *opaque) {
  auto *gate = static_cast<Gate *>(opaque);
  auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
  gate->entered = true;
  while (!gate->release) {
    if (std::chrono::steady_clock::now() >= deadline) {
      gate->timed_out = true;
      return;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }
}

int main() {
  CHECK(cuInit(0));
  CUcontext context = nullptr;
  CHECK(cuDevicePrimaryCtxRetain(&context, 0));
  CHECK(cuCtxSetCurrent(context));
  auto begin = resolve<Begin>("cuStreamBeginCapture", 10010);
  auto end = resolve<End>("cuStreamEndCapture", 10000);
  auto sync = resolve<Sync>("cuStreamSynchronize", 7000);
  auto legacy_begin = resolve<Begin>("cuStreamBeginCapture", 10010,
                                     CU_GET_PROC_ADDRESS_LEGACY_STREAM);
  REQUIRE(legacy_begin(nullptr, CU_STREAM_CAPTURE_MODE_THREAD_LOCAL) ==
          CUDA_ERROR_STREAM_CAPTURE_UNSUPPORTED);
  REQUIRE(begin(CU_STREAM_LEGACY, CU_STREAM_CAPTURE_MODE_THREAD_LOCAL) ==
          CUDA_ERROR_STREAM_CAPTURE_UNSUPPORTED);
  void *unused = nullptr;
  REQUIRE(cuGetProcAddress("cuStreamSynchronize", &unused, 7000, 8, nullptr) ==
          CUDA_ERROR_INVALID_VALUE);
  REQUIRE(resolve<Sync>("cuStreamSynchronize", 7000,
                        CU_GET_PROC_ADDRESS_LEGACY_STREAM |
                            CU_GET_PROC_ADDRESS_PER_THREAD_DEFAULT_STREAM) ==
          sync);

  CUdriverProcAddressQueryResult unavailable{};
  CHECK(cuGetProcAddress("cuMemsetD32", &unused, 3020,
                         CU_GET_PROC_ADDRESS_PER_THREAD_DEFAULT_STREAM,
                         &unavailable));
  REQUIRE(unused == nullptr &&
          unavailable == CU_GET_PROC_ADDRESS_VERSION_NOT_SUFFICIENT);
  using ProcV1 = CUresult (*)(const char *, void **, int, unsigned long long);
  REQUIRE(symbol<ProcV1>("cuGetProcAddress")(
              "cuMemsetD32", &unused, 3020,
              CU_GET_PROC_ADDRESS_PER_THREAD_DEFAULT_STREAM) ==
          CUDA_ERROR_NOT_FOUND);

  // Synchronous PTDS APIs preserve ordering with stream-zero async operations.
  CUdeviceptr device = 0;
  CHECK(cuMemAlloc(&device, sizeof(unsigned int)));
  auto set = resolve<Set>("cuMemsetD32", 7000);
  auto get = symbol<Get>("cuMemcpyDtoH_v2_ptds");
  CHECK(set(device, 37, 1));
  unsigned int value = 0;
  CHECK(get(&value, device, sizeof(value)));
  REQUIRE(value == 37);
  CHECK(symbol<Set>("cuMemsetD32_v2_ptds")(device, 91, 1));
  CHECK(resolve<Get>("cuMemcpyDtoH", 7000)(&value, device, sizeof(value)));
  REQUIRE(value == 91);

  // PTDS synchronous operations must complete while an unrelated blocking
  // stream is occupied. A legacy default-stream call waits for the gate.
  CUstream blocked = nullptr;
  CHECK(cuStreamCreate(&blocked, CU_STREAM_DEFAULT));
  Gate gate;
  CHECK(cuLaunchHostFunc(blocked, block_stream, &gate));
  while (!gate.entered)
    std::this_thread::yield();
  CHECK(set(device, 71, 1));
  CHECK(get(&value, device, sizeof(value)));
  gate.release = true;
  CHECK(cuStreamSynchronize(blocked));
  REQUIRE(!gate.timed_out && value == 71);
  CHECK(cuStreamDestroy(blocked));

  // Legacy literal begin-capture exports have the one-argument CUDA 10.0 ABI.
  auto old_begin =
      resolve<CUresult (*)(CUstream)>("cuStreamBeginCapture", 10000);
  CHECK(old_begin(nullptr));
  CUgraph empty = nullptr;
  CHECK(end(nullptr, &empty));
  CHECK(cuGraphDestroy(empty));
  CUstream explicit_stream = nullptr;
  CHECK(cuStreamCreate(&explicit_stream, CU_STREAM_NON_BLOCKING));
  CHECK(
      symbol<CUresult (*)(CUstream)>("cuStreamBeginCapture")(explicit_stream));
  CHECK(cuStreamEndCapture(explicit_stream, &empty));
  CHECK(cuGraphDestroy(empty));
  CHECK(cuStreamDestroy(explicit_stream));

  CHECK(symbol<Begin>("cuStreamBeginCapture_v2_ptsz")(
      nullptr, CU_STREAM_CAPTURE_MODE_THREAD_LOCAL));
  CUstreamCaptureStatus status{};
  cuuint64_t capture_id = 0;
  CHECK(resolve<CaptureInfo>("cuStreamGetCaptureInfo", 10010)(nullptr, &status,
                                                              &capture_id));
  REQUIRE(status == CU_STREAM_CAPTURE_STATUS_ACTIVE && capture_id != 0);
  CHECK(symbol<End>("cuStreamEndCapture_ptsz")(nullptr, &empty));
  CHECK(cuGraphDestroy(empty));

  // The upload stream lives inside an in/out descriptor. Preserve the caller's
  // stream field while copying the instantiation result back out.
  CHECK(begin(nullptr, CU_STREAM_CAPTURE_MODE_THREAD_LOCAL));
  CHECK(end(nullptr, &empty));
  CUDA_GRAPH_INSTANTIATE_PARAMS params{};
  params.flags = CUDA_GRAPH_INSTANTIATE_FLAG_UPLOAD;
  params.result_out = CUDA_GRAPH_INSTANTIATE_ERROR;
  CUgraphExec uploaded = nullptr;
  using Instantiate =
      CUresult (*)(CUgraphExec *, CUgraph, CUDA_GRAPH_INSTANTIATE_PARAMS *);
  CHECK(resolve<Instantiate>("cuGraphInstantiateWithParams",
                             12000)(&uploaded, empty, &params));
  REQUIRE(params.result_out == CUDA_GRAPH_INSTANTIATE_SUCCESS);
  REQUIRE(params.hUploadStream == nullptr);
  CHECK(sync(nullptr));
  CHECK(cuGraphExecDestroy(uploaded));
  CHECK(cuGraphDestroy(empty));

  // A null stream nested in CUlaunchConfig must capture into this graph.
  const char *ptx = R"(.version 7.0
.target sm_52
.address_size 64
.visible .entry store_value(.param .u64 output) {
.reg .u64 address;
.reg .u32 value;
ld.param.u64 address, [output];
mov.u32 value, 123;
st.global.u32 [address], value;
ret;
})";
  CUmodule module = nullptr;
  CUfunction kernel = nullptr;
  CHECK(cuModuleLoadData(&module, ptx));
  CHECK(cuModuleGetFunction(&kernel, module, "store_value"));
  CUlaunchConfig config{};
  config.gridDimX = config.gridDimY = config.gridDimZ = 1;
  config.blockDimX = config.blockDimY = config.blockDimZ = 1;
  void *arguments[] = {&device};
  using KernelEx =
      CUresult (*)(const CUlaunchConfig *, CUfunction, void **, void **);
  CHECK(begin(nullptr, CU_STREAM_CAPTURE_MODE_THREAD_LOCAL));
  CHECK(resolve<KernelEx>("cuLaunchKernelEx", 12000)(&config, kernel, arguments,
                                                     nullptr));
  CHECK(end(nullptr, &empty));
  CHECK(cuGraphInstantiateWithFlags(&uploaded, empty, 0));
  CHECK(resolve<Launch>("cuGraphLaunch", 10000)(uploaded, nullptr));
  CHECK(sync(nullptr));
  CHECK(get(&value, device, sizeof(value)));
  REQUIRE(value == 123);
  REQUIRE(config.hStream == nullptr);
  CHECK(cuGraphExecDestroy(uploaded));
  CHECK(cuGraphDestroy(empty));
  CHECK(cuModuleUnload(module));

  std::atomic<int> callbacks{0};
  CHECK(symbol<Host>("cuLaunchHostFunc_ptsz")(nullptr, increment, &callbacks));
  CHECK(sync(nullptr));
  REQUIRE(callbacks == 1);
  using AddCallback =
      CUresult (*)(CUstream, CUstreamCallback, void *, unsigned int);
  CHECK(symbol<AddCallback>("cuStreamAddCallback_ptsz")(
      nullptr, stream_callback, &callbacks, 0));
  CHECK(sync(nullptr));
  REQUIRE(callbacks == 2);

  // Captures overlap in two host threads sharing one CUDA context. Their
  // resource maps, capture caches, pinned copy staging and RPC lanes must be
  // independent even though both streams have the same sentinel handle.
  unsigned int *host = nullptr;
  CHECK(cuMemAllocHost(reinterpret_cast<void **>(&host), 4 * sizeof(*host)));
  host[0] = 101;
  host[1] = 202;
  host[2] = host[3] = 0;
  CUdeviceptr devices[2]{};
  CHECK(cuMemAlloc(&devices[0], sizeof(*host)));
  CHECK(cuMemAlloc(&devices[1], sizeof(*host)));
  Barrier barrier;
  unsigned long long stream_ids[2]{};
  cuuint64_t capture_ids[2]{};
  auto worker = [&](int index) {
    CHECK(cuCtxSetCurrent(context));
    CHECK(resolve<Id>("cuStreamGetId", 12000)(nullptr, &stream_ids[index]));
    CHECK(begin(nullptr, CU_STREAM_CAPTURE_MODE_THREAD_LOCAL));
    barrier.wait();
    CUstreamCaptureStatus thread_status{};
    CHECK(resolve<CaptureInfo>("cuStreamGetCaptureInfo", 10010)(
        nullptr, &thread_status, &capture_ids[index]));
    REQUIRE(thread_status == CU_STREAM_CAPTURE_STATUS_ACTIVE);
    CHECK(resolve<HtoD>("cuMemcpyHtoDAsync", 7000)(devices[index], host + index,
                                                   sizeof(*host), nullptr));
    CHECK(symbol<DtoH>("cuMemcpyDtoHAsync_v2_ptsz")(
        host + index + 2, devices[index], sizeof(*host), nullptr));
    barrier.wait();
    CUgraph graph = nullptr;
    CHECK(end(nullptr, &graph));
    barrier.wait();
    CUgraphExec exec = nullptr;
    CHECK(cuGraphInstantiateWithFlags(&exec, graph, 0));
    CHECK(resolve<Launch>("cuGraphLaunch", 10000)(exec, nullptr));
    CHECK(sync(nullptr));
    REQUIRE(host[index + 2] == host[index]);
    CHECK(cuGraphExecDestroy(exec));
    CHECK(cuGraphDestroy(graph));
  };
  std::thread first(worker, 0), second(worker, 1);
  first.join();
  second.join();
  REQUIRE(stream_ids[0] != stream_ids[1]);
  REQUIRE(capture_ids[0] != capture_ids[1]);

  // The capture's staging must outlive its originating thread, while each
  // retired lane must release its cache once the graph is destroyed.
  for (unsigned iteration = 0; iteration < 16; ++iteration) {
    CUgraph graph = nullptr;
    host[0] = 300 + iteration;
    host[2] = 0;
    std::thread capture_thread([&] {
      CHECK(cuCtxSetCurrent(context));
      CHECK(begin(nullptr, CU_STREAM_CAPTURE_MODE_THREAD_LOCAL));
      CHECK(resolve<HtoD>("cuMemcpyHtoDAsync", 7000)(devices[0], host,
                                                     sizeof(*host), nullptr));
      CHECK(symbol<DtoH>("cuMemcpyDtoHAsync_v2_ptsz")(host + 2, devices[0],
                                                      sizeof(*host), nullptr));
      CHECK(end(nullptr, &graph));
    });
    capture_thread.join();
    CUgraphExec exec = nullptr;
    CHECK(cuGraphInstantiateWithFlags(&exec, graph, 0));
    CHECK(cuGraphDestroy(graph));
    CHECK(resolve<Launch>("cuGraphLaunch", 10000)(exec, nullptr));
    CHECK(sync(nullptr));
    REQUIRE(host[2] == host[0]);
    CHECK(cuGraphExecDestroy(exec));
  }
  CHECK(cuMemFree(devices[0]));
  CHECK(cuMemFree(devices[1]));
  CHECK(cuMemFreeHost(host));
  CHECK(cuMemFree(device));
  CHECK(cuDevicePrimaryCtxRelease(0));
  std::puts("PASS: PTDS lookup, direct aliases, synchronous copies, callbacks, "
            "and concurrent stream-zero graphs");
}

#endif
