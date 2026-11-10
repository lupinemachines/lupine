// A device-to-host copy into page-locked memory followed, on the same stream,
// by a host-to-device copy from that memory: the second copy must send what
// the first one wrote. The stream is kept busy so both copies are still queued
// when the second is issued, and the pair is also replayed from a graph.
// This file is auto-discovered by test/run_custom_tests.sh.
#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#define CHECK(call)                                                            \
  do {                                                                         \
    cudaError_t check_result_ = (call);                                        \
    if (check_result_ != cudaSuccess) {                                        \
      std::fprintf(stderr, "%s failed: %s\n", #call,                           \
                   cudaGetErrorString(check_result_));                         \
      return 2;                                                                \
    }                                                                          \
  } while (0)

__global__ void spin(long long cycles) {
  long long start = clock64();
  while (clock64() - start < cycles) {
  }
}

// One round trip of `count` words through `host`, which sits `offset` words
// into a page-locked allocation.
static int bounce(cudaStream_t stream, unsigned int *host, size_t count,
                  unsigned int seed, long long spin_cycles) {
  size_t bytes = count * sizeof(unsigned int);
  std::vector<unsigned int> expected(count), got(count, 0);
  for (size_t i = 0; i < count; ++i) {
    expected[i] = seed * 0x9e3779b1u + static_cast<unsigned int>(i);
  }
  unsigned int *a = nullptr, *b = nullptr;
  CHECK(cudaMalloc(&a, bytes));
  CHECK(cudaMalloc(&b, bytes));
  CHECK(cudaMemcpy(a, expected.data(), bytes, cudaMemcpyHostToDevice));
  CHECK(cudaMemsetAsync(b, 0, bytes, stream));
  std::memset(host, 0, bytes);

  spin<<<1, 1, 0, stream>>>(spin_cycles);
  CHECK(cudaGetLastError());
  CHECK(cudaMemcpyAsync(host, a, bytes, cudaMemcpyDeviceToHost, stream));
  CHECK(cudaMemcpyAsync(b, host, bytes, cudaMemcpyHostToDevice, stream));
  CHECK(cudaMemcpyAsync(got.data(), b, bytes, cudaMemcpyDeviceToHost, stream));
  CHECK(cudaStreamSynchronize(stream));
  CHECK(cudaFree(a));
  CHECK(cudaFree(b));

  for (size_t i = 0; i < count; ++i) {
    if (got[i] != expected[i]) {
      std::fprintf(stderr,
                   "%zu words, round %u: word %zu is 0x%x, expected 0x%x\n",
                   count, seed, i, got[i], expected[i]);
      return 1;
    }
  }
  return 0;
}

// The same round trip captured once into a graph and replayed.
static int bounce_graph(cudaStream_t stream, unsigned int *host, size_t count,
                        long long spin_cycles) {
  size_t bytes = count * sizeof(unsigned int);
  std::vector<unsigned int> expected(count), got(count, 0);
  unsigned int *a = nullptr, *b = nullptr;
  CHECK(cudaMalloc(&a, bytes));
  CHECK(cudaMalloc(&b, bytes));
  cudaGraph_t graph;
  cudaGraphExec_t exec;
  CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
  spin<<<1, 1, 0, stream>>>(spin_cycles);
  CHECK(cudaMemcpyAsync(host, a, bytes, cudaMemcpyDeviceToHost, stream));
  CHECK(cudaMemcpyAsync(b, host, bytes, cudaMemcpyHostToDevice, stream));
  CHECK(cudaStreamEndCapture(stream, &graph));
  CHECK(cudaGraphInstantiate(&exec, graph, nullptr, nullptr, 0));

  for (unsigned int round = 1; round <= 10; ++round) {
    for (size_t i = 0; i < count; ++i) {
      expected[i] = round * 0x9e3779b1u + static_cast<unsigned int>(i);
    }
    CHECK(cudaMemcpy(a, expected.data(), bytes, cudaMemcpyHostToDevice));
    CHECK(cudaMemsetAsync(b, 0, bytes, stream));
    std::memset(host, 0, bytes);
    CHECK(cudaGraphLaunch(exec, stream));
    CHECK(cudaStreamSynchronize(stream));
    CHECK(cudaMemcpy(got.data(), b, bytes, cudaMemcpyDeviceToHost));
    for (size_t i = 0; i < count; ++i) {
      if (got[i] != expected[i]) {
        std::fprintf(stderr,
                     "graph, %zu words, round %u: word %zu is 0x%x, expected "
                     "0x%x\n",
                     count, round, i, got[i], expected[i]);
        return 1;
      }
    }
  }
  CHECK(cudaGraphExecDestroy(exec));
  CHECK(cudaGraphDestroy(graph));
  CHECK(cudaFree(a));
  CHECK(cudaFree(b));
  return 0;
}

int main() {
  CHECK(cudaSetDevice(0));
  cudaStream_t stream;
  CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
  int khz = 0;
  CHECK(cudaDeviceGetAttribute(&khz, cudaDevAttrClockRate, 0));
  long long spin_cycles = static_cast<long long>(khz) / 5; // ~200 us

  constexpr size_t kWords = (3u << 20) / sizeof(unsigned int);
  unsigned int *pinned = nullptr;
  CHECK(cudaMallocHost(&pinned, (kWords + 16) * sizeof(unsigned int)));

  for (unsigned int round = 1; round <= 20; ++round) {
    // A single word, then a span past the first page of the allocation and
    // large enough to cross several HtoD staging fragments.
    if (bounce(stream, pinned, 1, round, spin_cycles) != 0 ||
        bounce(stream, pinned + 16, kWords, round, spin_cycles) != 0) {
      std::printf("FAIL: HtoD from page-locked memory missed the DtoH queued "
                  "ahead of it\n");
      return 1;
    }
  }

  if (bounce_graph(stream, pinned, 1, spin_cycles) != 0 ||
      bounce_graph(stream, pinned + 16, kWords, spin_cycles) != 0) {
    std::printf("FAIL: a graph's HtoD from page-locked memory missed the DtoH "
                "captured ahead of it\n");
    return 1;
  }

  CHECK(cudaFreeHost(pinned));
  CHECK(cudaStreamDestroy(stream));
  std::printf("PASS: HtoD from page-locked memory sees the DtoH queued ahead "
              "of it\n");
  return 0;
}
