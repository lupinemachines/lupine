// Kernel arguments passed as structs by value (spans, iterators, descriptors)
// and kernel nodes added explicitly to CUDA graphs can carry mapped, portable,
// or managed pointers without those pointers ever appearing as bare 8-byte
// launch arguments or passing through cudaHostGetDevicePointer.
#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>

namespace {

struct SpanArg {
  uint32_t *ptr;
  uint32_t value;
  uint32_t count;
};

__global__ void write_span(SpanArg arg) {
  for (uint32_t i = 0; i < arg.count; ++i) {
    arg.ptr[i] = arg.value + i;
  }
}

__global__ void copy_word(uint32_t *dst, const uint32_t *src) { *dst = *src; }

void check(cudaError_t result, const char *what) {
  if (result == cudaSuccess) {
    return;
  }
  std::fprintf(stderr, "FAIL: %s: %s\n", what, cudaGetErrorString(result));
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

  int failures = 0;
  cudaStream_t stream = nullptr;
  check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking),
        "cudaStreamCreateWithFlags");

  // 1. Unexposed mapped host pointer embedded in a struct-by-value launch arg.
  uint32_t *mapped = nullptr;
  check(cudaHostAlloc(&mapped, sizeof(uint32_t), cudaHostAllocMapped),
        "cudaHostAlloc(mapped)");
  *mapped = 11;
  write_span<<<1, 1, 0, stream>>>(SpanArg{mapped, 77, 1});
  check(cudaGetLastError(), "write_span(mapped)");
  check(cudaStreamSynchronize(stream), "cudaStreamSynchronize");
  uint32_t got_mapped = *static_cast<volatile uint32_t *>(mapped);
  if (got_mapped != 77) {
    std::fprintf(stderr,
                 "FAIL: struct-by-value mapped write: got %u, want 77\n",
                 got_mapped);
    ++failures;
  }
  check(cudaFreeHost(mapped), "cudaFreeHost(mapped)");

  // 2. Managed pointer embedded in a struct-by-value launch arg: touching a
  // disjoint word on the same page before synchronizing must fetch behind the
  // launch rather than flushing pre-launch bytes over the kernel's write.
  uint32_t *managed = nullptr;
  uint32_t *device_out = nullptr;
  check(cudaMallocManaged(&managed, 2 * sizeof(uint32_t)), "cudaMallocManaged");
  check(cudaMalloc(&device_out, sizeof(uint32_t)), "cudaMalloc");
  managed[0] = 11;
  managed[1] = 21;
  write_span<<<1, 1, 0, stream>>>(SpanArg{&managed[0], 88, 1});
  check(cudaGetLastError(), "write_span(managed)");
  managed[1] = 99;
  copy_word<<<1, 1, 0, stream>>>(device_out, &managed[0]);
  check(cudaGetLastError(), "copy_word");
  check(cudaStreamSynchronize(stream), "cudaStreamSynchronize");
  uint32_t got_managed_dev = 0;
  check(cudaMemcpy(&got_managed_dev, device_out, sizeof(uint32_t),
                   cudaMemcpyDeviceToHost),
        "cudaMemcpy(device_out)");
  uint32_t got_managed_host = static_cast<volatile uint32_t *>(managed)[0];
  uint32_t got_managed_sib = static_cast<volatile uint32_t *>(managed)[1];
  if (got_managed_dev != 88 || got_managed_host != 88 ||
      got_managed_sib != 99) {
    std::fprintf(stderr,
                 "FAIL: struct-by-value managed write: dev=%u host=%u sib=%u, "
                 "want 88/88/99\n",
                 got_managed_dev, got_managed_host, got_managed_sib);
    ++failures;
  }
  check(cudaFree(device_out), "cudaFree");
  check(cudaFree(managed), "cudaFree(managed)");

  // 3. Explicit graph kernel nodes (AddKernelNode and ExecKernelNodeSetParams)
  // given fresh unexposed mapped host allocations inside a struct argument.
  uint32_t *graph_a = nullptr;
  uint32_t *graph_b = nullptr;
  check(cudaHostAlloc(&graph_a, sizeof(uint32_t), cudaHostAllocMapped),
        "cudaHostAlloc(graph_a)");
  check(cudaHostAlloc(&graph_b, sizeof(uint32_t), cudaHostAllocMapped),
        "cudaHostAlloc(graph_b)");
  *graph_a = 1;
  *graph_b = 2;

  cudaGraph_t graph = nullptr;
  cudaGraphNode_t node = nullptr;
  cudaGraphExec_t exec = nullptr;
  check(cudaGraphCreate(&graph, 0), "cudaGraphCreate");

  SpanArg span_a{graph_a, 55, 1};
  void *node_args_a[] = {&span_a};
  cudaKernelNodeParams node_params{};
  node_params.func = reinterpret_cast<void *>(write_span);
  node_params.gridDim = dim3(1, 1, 1);
  node_params.blockDim = dim3(1, 1, 1);
  node_params.kernelParams = node_args_a;
  check(cudaGraphAddKernelNode(&node, graph, nullptr, 0, &node_params),
        "cudaGraphAddKernelNode");
  check(cudaGraphInstantiate(&exec, graph, nullptr, nullptr, 0),
        "cudaGraphInstantiate");
  check(cudaGraphLaunch(exec, stream), "cudaGraphLaunch(a)");
  check(cudaStreamSynchronize(stream), "cudaStreamSynchronize(graph_a)");
  uint32_t got_graph_a = *static_cast<volatile uint32_t *>(graph_a);
  if (got_graph_a != 55) {
    std::fprintf(stderr, "FAIL: graph AddKernelNode mapped write: got %u, want 55\n",
                 got_graph_a);
    ++failures;
  }

  SpanArg span_b{graph_b, 66, 1};
  void *node_args_b[] = {&span_b};
  node_params.kernelParams = node_args_b;
  check(cudaGraphExecKernelNodeSetParams(exec, node, &node_params),
        "cudaGraphExecKernelNodeSetParams");
  check(cudaGraphLaunch(exec, stream), "cudaGraphLaunch(b)");
  check(cudaStreamSynchronize(stream), "cudaStreamSynchronize(graph_b)");
  uint32_t got_graph_b = *static_cast<volatile uint32_t *>(graph_b);
  if (got_graph_b != 66) {
    std::fprintf(stderr,
                 "FAIL: graph ExecKernelNodeSetParams mapped write: got %u, want 66\n",
                 got_graph_b);
    ++failures;
  }

  check(cudaGraphExecDestroy(exec), "cudaGraphExecDestroy");
  check(cudaGraphDestroy(graph), "cudaGraphDestroy");
  check(cudaFreeHost(graph_b), "cudaFreeHost(graph_b)");
  check(cudaFreeHost(graph_a), "cudaFreeHost(graph_a)");
  check(cudaStreamDestroy(stream), "cudaStreamDestroy");

  if (failures != 0) {
    std::fprintf(stderr, "FAIL: %d sub-cases failed\n", failures);
    return EXIT_FAILURE;
  }
  std::printf("PASS: struct-by-value and graph kernel node pointers prepared\n");
  return EXIT_SUCCESS;
}
