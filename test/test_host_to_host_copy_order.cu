// A copy between two host pointers, where one is pinned, mapped or managed, is
// stream-ordered like any other copy: it reads its source after the work ahead
// of it on the stream wrote it, and later work and the host see what it wrote.
#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>

namespace {

constexpr long long kSpinCycles = 100000000;

__global__ void slow_store(unsigned int *destination, unsigned int value) {
  long long start = clock64();
  while (clock64() - start < kSpinCycles) {
  }
  *destination = value;
}

struct host_store {
  unsigned int *destination;
  unsigned int value;
};

void CUDART_CB store_from_host(void *opaque) {
  auto *store = static_cast<host_store *>(opaque);
  *store->destination = store->value;
}

void check(cudaError_t result, const char *operation) {
  if (result == cudaSuccess) {
    return;
  }
  std::fprintf(stderr, "FAIL: %s: %s\n", operation, cudaGetErrorString(result));
  std::exit(EXIT_FAILURE);
}

int expect(const char *name, unsigned int got, unsigned int want) {
  if (got == want) {
    return 0;
  }
  std::fprintf(stderr, "FAIL: %s got 0x%x want 0x%x\n", name, got, want);
  return 1;
}

unsigned int host_read(const unsigned int *buffer) {
  return *static_cast<const volatile unsigned int *>(buffer);
}

unsigned int copy_read(const unsigned int *buffer) {
  unsigned int value = 0;
  check(cudaMemcpy(&value, buffer, sizeof(value), cudaMemcpyDefault),
        "cudaMemcpy read back");
  return value;
}

int run(const char *name, unsigned int *buffer, cudaStream_t stream) {
  int failures = 0;
  unsigned int *pinned = nullptr;
  check(cudaMallocHost(&pinned, sizeof(*pinned)), "cudaMallocHost");
  char label[128];

  // A host function fills the pinned source the copy behind it reads.
  *buffer = 0;
  *pinned = 0;
  host_store store{pinned, 0x1001};
  check(cudaLaunchHostFunc(stream, store_from_host, &store),
        "cudaLaunchHostFunc");
  check(cudaMemcpyAsync(buffer, pinned, sizeof(*buffer), cudaMemcpyDefault,
                        stream),
        "cudaMemcpyAsync pinned source");
  check(cudaStreamSynchronize(stream), "cudaStreamSynchronize");
  std::snprintf(label, sizeof(label), "%s: host function source, host read",
                name);
  failures += expect(label, host_read(buffer), 0x1001);
  std::snprintf(label, sizeof(label), "%s: host function source, copy read",
                name);
  failures += expect(label, copy_read(buffer), 0x1001);

  // The same through the 2D API.
  store.value = 0x1002;
  check(cudaLaunchHostFunc(stream, store_from_host, &store),
        "cudaLaunchHostFunc");
  check(cudaMemcpy2DAsync(buffer, sizeof(*buffer), pinned, sizeof(*pinned),
                          sizeof(*buffer), 1, cudaMemcpyDefault, stream),
        "cudaMemcpy2DAsync pinned source");
  check(cudaStreamSynchronize(stream), "cudaStreamSynchronize");
  std::snprintf(label, sizeof(label), "%s: host function 2D source", name);
  failures += expect(label, host_read(buffer), 0x1002);

  // A kernel's store lands before a later copy overwrites it.
  unsigned int pageable = 0x1003;
  slow_store<<<1, 1, 0, stream>>>(buffer, 0x2003);
  check(cudaGetLastError(), "slow_store launch");
  check(cudaMemcpyAsync(buffer, &pageable, sizeof(*buffer), cudaMemcpyDefault,
                        stream),
        "cudaMemcpyAsync pageable source");
  check(cudaStreamSynchronize(stream), "cudaStreamSynchronize");
  std::snprintf(label, sizeof(label), "%s: copy after kernel", name);
  failures += expect(label, host_read(buffer), 0x1003);

  // A copy out of the buffer reads the kernel's store.
  unsigned int out = 0;
  slow_store<<<1, 1, 0, stream>>>(buffer, 0x2004);
  check(cudaGetLastError(), "slow_store launch");
  check(cudaMemcpyAsync(&out, buffer, sizeof(out), cudaMemcpyDefault, stream),
        "cudaMemcpyAsync to pageable");
  check(cudaStreamSynchronize(stream), "cudaStreamSynchronize");
  std::snprintf(label, sizeof(label), "%s: async read after kernel", name);
  failures += expect(label, out, 0x2004);

  // A synchronous copy waits for the legacy stream.
  slow_store<<<1, 1>>>(buffer, 0x2005);
  check(cudaGetLastError(), "slow_store launch");
  check(cudaMemcpy(&out, buffer, sizeof(out), cudaMemcpyDefault),
        "cudaMemcpy to pageable");
  std::snprintf(label, sizeof(label), "%s: sync read after kernel", name);
  failures += expect(label, out, 0x2005);
  pageable = 0x1006;
  slow_store<<<1, 1>>>(buffer, 0x2006);
  check(cudaGetLastError(), "slow_store launch");
  check(cudaMemcpy(buffer, &pageable, sizeof(pageable), cudaMemcpyDefault),
        "cudaMemcpy from pageable");
  check(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
  std::snprintf(label, sizeof(label), "%s: sync write after kernel", name);
  failures += expect(label, host_read(buffer), 0x1006);

  check(cudaFreeHost(pinned), "cudaFreeHost");
  return failures;
}

} // namespace

int main() {
  int device_count = 0;
  check(cudaGetDeviceCount(&device_count), "cudaGetDeviceCount");
  if (device_count == 0) {
    std::printf("SKIP: no CUDA devices found\n");
    return EXIT_SUCCESS;
  }

  unsigned int *mapped = nullptr;
  unsigned int *managed = nullptr;
  cudaStream_t stream = nullptr;
  check(cudaHostAlloc(&mapped, sizeof(*mapped), cudaHostAllocMapped),
        "cudaHostAlloc");
  check(cudaMallocManaged(&managed, sizeof(*managed)), "cudaMallocManaged");
  check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking),
        "cudaStreamCreateWithFlags");

  int failures = run("mapped", mapped, stream) +
                 run("managed", managed, stream) +
                 run("mapped per-thread", mapped, cudaStreamPerThread);

  check(cudaStreamDestroy(stream), "cudaStreamDestroy");
  check(cudaFree(managed), "cudaFree");
  check(cudaFreeHost(mapped), "cudaFreeHost");
  if (failures != 0) {
    std::fprintf(stderr, "FAIL: %d host-to-host copies ran out of order\n",
                 failures);
    return EXIT_FAILURE;
  }
  std::printf("PASS: host-to-host copies are stream-ordered\n");
  return EXIT_SUCCESS;
}
