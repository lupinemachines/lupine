// An explicit graph: copy a device word to host memory, then an empty node,
// then a host node that reads the word. The host node follows the copy through
// the empty node, so every launch must see that launch's bytes.
#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>

namespace {

constexpr int kLaunches = 50;

unsigned int host_word = 0;
unsigned int expected = 0;
int failures = 0;

void check(cudaError_t result, const char *operation) {
  if (result != cudaSuccess) {
    std::fprintf(stderr, "FAIL: %s: %s\n", operation,
                 cudaGetErrorString(result));
    std::exit(EXIT_FAILURE);
  }
}

__global__ void store(unsigned int *destination, unsigned int value) {
  *destination = value;
}

void CUDART_CB read_word(void *) {
  unsigned int got = *static_cast<volatile unsigned int *>(&host_word);
  if (got != expected) {
    std::fprintf(stderr, "FAIL: host node read 0x%x want 0x%x\n", got,
                 expected);
    ++failures;
  }
}

} // namespace

int main() {
  unsigned int *device = nullptr;
  cudaStream_t stream = nullptr;
  check(cudaMalloc(&device, sizeof(*device)), "cudaMalloc");
  check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "stream");

  cudaGraph_t graph = nullptr;
  check(cudaGraphCreate(&graph, 0), "cudaGraphCreate");
  cudaGraphNode_t copy = nullptr;
  check(cudaGraphAddMemcpyNode1D(&copy, graph, nullptr, 0, &host_word, device,
                                 sizeof(host_word), cudaMemcpyDeviceToHost),
        "cudaGraphAddMemcpyNode1D");
  cudaGraphNode_t empty = nullptr;
  check(cudaGraphAddEmptyNode(&empty, graph, &copy, 1),
        "cudaGraphAddEmptyNode");
  cudaHostNodeParams host_params = {};
  host_params.fn = read_word;
  cudaGraphNode_t host = nullptr;
  check(cudaGraphAddHostNode(&host, graph, &empty, 1, &host_params),
        "cudaGraphAddHostNode");
  cudaGraphExec_t exec = nullptr;
  check(cudaGraphInstantiateWithFlags(&exec, graph, 0),
        "cudaGraphInstantiateWithFlags");

  for (int launch = 0; launch < kLaunches; ++launch) {
    expected = 0x9000 + launch;
    // Written on the launch stream: a synchronous copy from pageable memory
    // may still be in flight when it returns, unordered with this stream.
    store<<<1, 1, 0, stream>>>(device, expected);
    check(cudaGetLastError(), "store launch");
    check(cudaGraphLaunch(exec, stream), "cudaGraphLaunch");
    check(cudaStreamSynchronize(stream), "cudaStreamSynchronize");
  }
  check(cudaGraphExecDestroy(exec), "cudaGraphExecDestroy");
  check(cudaGraphDestroy(graph), "cudaGraphDestroy");
  check(cudaStreamDestroy(stream), "cudaStreamDestroy");
  check(cudaFree(device), "cudaFree");
  if (failures != 0) {
    std::fprintf(stderr, "FAIL: %d stale host-node reads\n", failures);
    return EXIT_FAILURE;
  }
  std::printf("PASS: host nodes after an empty node see graph copies\n");
  return EXIT_SUCCESS;
}
