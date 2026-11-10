// Host functions (cuLaunchHostFunc / cudaLaunchHostFunc, cuGraphAddHostNode)
// and stream callbacks (cuStreamAddCallback / cudaStreamAddCallback) execute
// after all preceding work on the stream or graph branch has completed, so
// mapped and managed host memory written by earlier kernels must be refreshed
// before the callback runs—without preventing subsequent synchronization from
// refreshing writes made by kernels queued after the callback.
#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>

namespace {

__global__ void write_add(unsigned int *ptr, unsigned int addend) {
  *ptr += addend;
}

struct CallbackState {
  unsigned int *mapped = nullptr;
  unsigned int *registered = nullptr;
  unsigned int *managed = nullptr;
  unsigned int observed_mapped = 0;
  unsigned int observed_registered = 0;
  unsigned int observed_managed = 0;
  unsigned int add_in_callback = 0;
};

void CUDART_CB host_fn(void *user_data) {
  auto *st = static_cast<CallbackState *>(user_data);
  if (st->mapped != nullptr) {
    st->observed_mapped = *st->mapped;
    if (st->add_in_callback != 0) {
      *st->mapped += st->add_in_callback;
    }
  }
  if (st->registered != nullptr) {
    st->observed_registered = *st->registered;
  }
  if (st->managed != nullptr) {
    st->observed_managed = *st->managed;
  }
}

void CUDART_CB stream_cb(cudaStream_t, cudaError_t, void *user_data) {
  host_fn(user_data);
}

void check(cudaError_t err, const char *what) {
  if (err == cudaSuccess) {
    return;
  }
  std::fprintf(stderr, "FAIL: %s: %s\n", what, cudaGetErrorString(err));
  std::exit(EXIT_FAILURE);
}

} // namespace

int main() {
  int device_count = 0;
  check(cudaGetDeviceCount(&device_count), "cudaGetDeviceCount");
  if (device_count == 0) {
    std::printf("SKIP: no CUDA devices found\n");
    return EXIT_SUCCESS;
  }

  check(cudaSetDeviceFlags(cudaDeviceMapHost), "cudaSetDeviceFlags");

  unsigned int *mapped_a = nullptr;
  unsigned int *mapped_a_dev = nullptr;
  unsigned int *mapped_b = nullptr;
  unsigned int *mapped_b_dev = nullptr;
  unsigned int *reg_host =
      static_cast<unsigned int *>(std::malloc(64 * sizeof(unsigned int)));
  unsigned int *reg_dev = nullptr;
  unsigned int *managed = nullptr;

  check(cudaHostAlloc(&mapped_a, sizeof(unsigned int), cudaHostAllocMapped),
        "cudaHostAlloc(a)");
  check(cudaHostGetDevicePointer(&mapped_a_dev, mapped_a, 0),
        "cudaHostGetDevicePointer(a)");
  check(cudaHostAlloc(&mapped_b, sizeof(unsigned int), cudaHostAllocMapped),
        "cudaHostAlloc(b)");
  check(cudaHostGetDevicePointer(&mapped_b_dev, mapped_b, 0),
        "cudaHostGetDevicePointer(b)");
  check(cudaHostRegister(reg_host, 64 * sizeof(unsigned int),
                         cudaHostRegisterMapped),
        "cudaHostRegister");
  check(cudaHostGetDevicePointer(&reg_dev, reg_host, 0),
        "cudaHostGetDevicePointer(reg)");
  check(cudaMallocManaged(&managed, sizeof(unsigned int)), "cudaMallocManaged");

  cudaStream_t blocking_stream = nullptr;
  cudaStream_t nonblocking_stream = nullptr;
  check(cudaStreamCreate(&blocking_stream), "cudaStreamCreate");
  check(cudaStreamCreateWithFlags(&nonblocking_stream, cudaStreamNonBlocking),
        "cudaStreamCreateWithFlags");

  int failures = 0;

  // 1. Eager cudaLaunchHostFunc on a blocking stream (tests both mapped and
  // cudaHostRegister buffers without deadlocking CU_STREAM_LEGACY) followed by
  // another kernel writing mapped_b.
  *mapped_a = 10;
  *mapped_b = 50;
  reg_host[0] = 30;
  check(cudaDeviceSynchronize(), "sync init 1");
  // Prime client read views so pages start readable before GPU writes.
  volatile unsigned int prime = *mapped_a + *mapped_b + reg_host[0];
  (void)prime;

  CallbackState st1{};
  st1.mapped = mapped_a;
  st1.registered = reg_host;
  st1.add_in_callback = 5;

  write_add<<<1, 1, 0, blocking_stream>>>(mapped_a_dev, 20);
  write_add<<<1, 1, 0, blocking_stream>>>(reg_dev, 40);
  check(cudaLaunchHostFunc(blocking_stream, host_fn, &st1),
        "cudaLaunchHostFunc");
  write_add<<<1, 1, 0, blocking_stream>>>(mapped_a_dev, 100);
  write_add<<<1, 1, 0, blocking_stream>>>(mapped_b_dev, 7);
  check(cudaStreamSynchronize(blocking_stream), "sync blocking_stream");

  if (st1.observed_mapped != 30 || st1.observed_registered != 70 ||
      *mapped_a != 135 || *mapped_b != 57) {
    std::fprintf(stderr,
                 "FAIL: eager host_fn: obs_mapped=%u(30) obs_reg=%u(70) "
                 "final_a=%u(135) final_b=%u(57)\n",
                 st1.observed_mapped, st1.observed_registered, *mapped_a,
                 *mapped_b);
    ++failures;
  }

  // 2. cudaStreamAddCallback reading mapped memory written by preceding kernel.
  *mapped_a = 3;
  check(cudaDeviceSynchronize(), "sync init 2");
  prime = *mapped_a;
  CallbackState st2{};
  st2.mapped = mapped_a;
  write_add<<<1, 1, 0, nonblocking_stream>>>(mapped_a_dev, 19);
  check(cudaStreamAddCallback(nonblocking_stream, stream_cb, &st2, 0),
        "cudaStreamAddCallback");
  check(cudaStreamSynchronize(nonblocking_stream), "sync nonblocking_stream");
  if (st2.observed_mapped != 22) {
    std::fprintf(stderr, "FAIL: stream_cb: obs_mapped=%u, want 22\n",
                 st2.observed_mapped);
    ++failures;
  }

  // 3. Captured CUDA graph: kernel updates mapped + managed -> host node reads
  // both and increments mapped -> trailing kernel adds 1000 to mapped.
  CallbackState st3{};
  st3.mapped = mapped_a;
  st3.managed = managed;
  st3.add_in_callback = 9;

  cudaGraph_t graph = nullptr;
  cudaGraphExec_t exec = nullptr;
  check(cudaStreamBeginCapture(nonblocking_stream,
                               cudaStreamCaptureModeThreadLocal),
        "cudaStreamBeginCapture");
  write_add<<<1, 1, 0, nonblocking_stream>>>(mapped_a_dev, 15);
  write_add<<<1, 1, 0, nonblocking_stream>>>(managed, 25);
  check(cudaLaunchHostFunc(nonblocking_stream, host_fn, &st3),
        "captured cudaLaunchHostFunc");
  write_add<<<1, 1, 0, nonblocking_stream>>>(mapped_a_dev, 1000);
  check(cudaStreamEndCapture(nonblocking_stream, &graph),
        "cudaStreamEndCapture");
  check(cudaGraphInstantiate(&exec, graph, nullptr, nullptr, 0),
        "cudaGraphInstantiate");

  for (unsigned int replay = 0; replay < 3; ++replay) {
    unsigned int base = (replay + 1) * 100;
    *mapped_a = base;
    *managed = base * 2;
    check(cudaStreamSynchronize(nonblocking_stream), "sync pre-replay");
    prime = *mapped_a + *managed;

    st3.observed_mapped = 0;
    st3.observed_managed = 0;
    check(cudaGraphLaunch(exec, nonblocking_stream), "cudaGraphLaunch");
    check(cudaStreamSynchronize(nonblocking_stream), "sync replay");

    if (st3.observed_mapped != base + 15 ||
        st3.observed_managed != base * 2 + 25 ||
        *mapped_a != base + 15 + 9 + 1000) {
      std::fprintf(stderr,
                   "FAIL: graph replay %u: obs_mapped=%u(%u) "
                   "obs_managed=%u(%u) final_a=%u(%u)\n",
                   replay, st3.observed_mapped, base + 15, st3.observed_managed,
                   base * 2 + 25, *mapped_a, base + 1024);
      ++failures;
    }
  }

  check(cudaGraphExecDestroy(exec), "cudaGraphExecDestroy");
  check(cudaGraphDestroy(graph), "cudaGraphDestroy");
  check(cudaStreamDestroy(nonblocking_stream), "cudaStreamDestroy(nb)");
  check(cudaStreamDestroy(blocking_stream), "cudaStreamDestroy(b)");
  check(cudaFree(managed), "cudaFree(managed)");
  check(cudaHostUnregister(reg_host), "cudaHostUnregister");
  std::free(reg_host);
  check(cudaFreeHost(mapped_b), "cudaFreeHost(b)");
  check(cudaFreeHost(mapped_a), "cudaFreeHost(a)");

  if (failures != 0) {
    std::fprintf(stderr, "FAIL: %d host callback refresh checks failed\n",
                 failures);
    return EXIT_FAILURE;
  }
  std::printf("PASS: host callbacks observe fresh mapped and managed writes\n");
  return EXIT_SUCCESS;
}
