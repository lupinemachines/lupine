// Host callbacks must publish mapped-memory writes before their stream resumes.
// Queue the entire pipeline before releasing the writer, then wait on a CPU
// flag so no application CUDA call can incidentally flush the callback's
// writes.
#include <cuda_runtime.h>

#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <thread>

static constexpr int kCount = 1 << 16;

static void check(cudaError_t result, const char *expr, int line) {
  if (result != cudaSuccess) {
    std::fprintf(stderr, "%s failed at line %d: %s\n", expr, line,
                 cudaGetErrorString(result));
    std::exit(EXIT_FAILURE);
  }
}

#define CHECK(expr) check((expr), #expr, __LINE__)

struct CallbackState {
  int *host = nullptr;
  int value = 0;
  std::atomic<bool> ready{false};
  std::atomic<bool> done{false};
  bool timed_out = false;
};

static bool wait_for(const std::atomic<bool> &flag) {
  const auto deadline =
      std::chrono::steady_clock::now() + std::chrono::seconds(30);
  while (!flag.load(std::memory_order_acquire)) {
    if (std::chrono::steady_clock::now() >= deadline) {
      return false;
    }
    std::this_thread::yield();
  }
  return true;
}

static void CUDART_CB fill(void *opaque) {
  auto *state = static_cast<CallbackState *>(opaque);
  if (!wait_for(state->ready)) {
    state->timed_out = true;
    return;
  }
  for (int i = 0; i < kCount; ++i) {
    state->host[i] = state->value;
  }
}

static void CUDART_CB fill_stream(cudaStream_t, cudaError_t status,
                                  void *opaque) {
  if (status == cudaSuccess) {
    fill(opaque);
  }
}

__global__ void copy_values(const int *src, int *dst) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < kCount) {
    dst[i] = src[i];
  }
}

static void CUDART_CB complete(void *opaque) {
  auto *state = static_cast<CallbackState *>(opaque);
  state->done.store(true, std::memory_order_release);
}

int main() {
  CallbackState state;
  int *mapped = nullptr;
  int *device = nullptr;
  cudaStream_t stream = nullptr;
  constexpr size_t bytes = kCount * sizeof(int);
  int readback[kCount];
  CHECK(cudaHostAlloc(&state.host, bytes, cudaHostAllocMapped));
  CHECK(cudaHostGetDevicePointer(&mapped, state.host, 0));
  CHECK(cudaMalloc(&device, bytes));
  CHECK(cudaStreamCreate(&stream));
  for (int i = 0; i < kCount; ++i) {
    state.host[i] = 0;
  }
  // Load the kernel before a gated callback can block lazy module loading.
  copy_values<<<kCount / 256, 256, 0, stream>>>(mapped, device);
  CHECK(cudaGetLastError());
  CHECK(cudaStreamSynchronize(stream));

  const char *names[] = {"host function", "stream callback", "graph host node"};
  int failures = 0;
  for (int mode = 0; mode < 3; ++mode) {
    for (int round = 0; round < 3; ++round) {
      ++state.value;
      state.ready.store(false, std::memory_order_relaxed);
      state.done.store(false, std::memory_order_relaxed);
      state.timed_out = false;
      cudaGraph_t graph = nullptr;
      cudaGraphExec_t exec = nullptr;
      if (mode == 2) {
        CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
      }
      if (mode == 1) {
        CHECK(cudaStreamAddCallback(stream, fill_stream, &state, 0));
      } else {
        CHECK(cudaLaunchHostFunc(stream, fill, &state));
      }
      copy_values<<<kCount / 256, 256, 0, stream>>>(mapped, device);
      CHECK(cudaGetLastError());
      CHECK(cudaLaunchHostFunc(stream, complete, &state));
      if (mode == 2) {
        CHECK(cudaStreamEndCapture(stream, &graph));
        CHECK(cudaGraphInstantiate(&exec, graph, nullptr, nullptr, 0));
        CHECK(cudaGraphLaunch(exec, stream));
      }

      // Every CUDA call above has returned before the callback writes. Do not
      // make another CUDA call until the trailing callback proves the kernel
      // has finished: a later flush cannot change the values it already read.
      state.ready.store(true, std::memory_order_release);
      if (!wait_for(state.done)) {
        std::fprintf(stderr, "FAIL: %s timed out\n", names[mode]);
        std::exit(EXIT_FAILURE);
      }
      CHECK(cudaStreamSynchronize(stream));
      // Read the completed output directly so this test does not also depend
      // on deferred DtoH delivery across callbacks.
      CHECK(cudaMemcpy(readback, device, bytes, cudaMemcpyDeviceToHost));
      if (state.timed_out) {
        std::fprintf(stderr, "FAIL: %s round %d writer gate timed out\n",
                     names[mode], round);
        ++failures;
      }
      for (int i = 0; i < kCount; ++i) {
        if (readback[i] != state.value) {
          std::fprintf(stderr, "FAIL: %s round %d read %d at %d, expected %d\n",
                       names[mode], round, readback[i], i, state.value);
          ++failures;
          break;
        }
      }
      if (exec != nullptr) {
        CHECK(cudaGraphExecDestroy(exec));
        CHECK(cudaGraphDestroy(graph));
      }
    }
  }

  CHECK(cudaStreamDestroy(stream));
  CHECK(cudaFree(device));
  CHECK(cudaFreeHost(state.host));
  if (failures != 0) {
    return EXIT_FAILURE;
  }
  std::printf("PASS: kernels see mapped writes from host functions, stream "
              "callbacks, and graph host nodes\n");
  return EXIT_SUCCESS;
}
