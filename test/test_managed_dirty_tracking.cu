#include <cuda_runtime.h>

#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <sys/mman.h>
#include <thread>

static void check(cudaError_t result, const char *what) {
  if (result != cudaSuccess) {
    std::fprintf(stderr, "%s failed: %s\n", what, cudaGetErrorString(result));
    std::exit(1);
  }
}

__global__ void check_sparse_pages(const unsigned char *data, size_t pages,
                                   size_t page_size, unsigned int *errors) {
  size_t page = blockIdx.x * blockDim.x + threadIdx.x;
  if (page < pages && data[page * page_size] != (page & 0xff)) {
    atomicAdd(errors, 1);
  }
}

int main() {
  constexpr size_t page_size = 4096;
  constexpr size_t managed_bytes = 260ULL * 1024 * 1024;
  constexpr size_t managed_pages = managed_bytes / page_size;

  unsigned char *managed = nullptr;
  check(cudaMallocManaged(&managed, managed_bytes), "cudaMallocManaged");
  for (size_t page = 0; page < managed_pages; ++page) {
    managed[page * page_size] = static_cast<unsigned char>(page);
  }
  std::puts("completed >256 MiB sparse host write");

  unsigned int *errors = nullptr;
  check(cudaMalloc(&errors, sizeof(*errors)), "cudaMalloc errors");
  check(cudaMemset(errors, 0, sizeof(*errors)), "cudaMemset errors");
  check_sparse_pages<<<(managed_pages + 255) / 256, 256>>>(
      managed, managed_pages, page_size, errors);
  check(cudaDeviceSynchronize(), "check sparse pages");
  unsigned int host_errors = 0;
  check(cudaMemcpy(&host_errors, errors, sizeof(host_errors),
                   cudaMemcpyDeviceToHost),
        "copy sparse errors");
  if (host_errors != 0) {
    std::fprintf(stderr, "%u sparse pages were not synchronized\n",
                 host_errors);
    return 1;
  }
  check(cudaFree(errors), "cudaFree errors");
  check(cudaFree(managed), "cudaFree managed");

  constexpr size_t registered_bytes = 257ULL * 1024 * 1024;
  void *registered = mmap(nullptr, registered_bytes, PROT_READ | PROT_WRITE,
                          MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
  if (registered == MAP_FAILED) {
    std::perror("mmap registered memory");
    return 1;
  }
  check(cudaHostRegister(registered, registered_bytes, cudaHostRegisterMapped),
        "cudaHostRegister >256 MiB");
  check(cudaHostUnregister(registered), "cudaHostUnregister >256 MiB");
  munmap(registered, registered_bytes);

  // Keep one allocation faulting while another thread adds and removes fault
  // table entries. Both memory lifetimes are valid; this specifically stresses
  // concurrent handler traversal and metadata retirement.
  constexpr size_t stress_bytes = 2ULL * 1024 * 1024;
  unsigned char *stable = nullptr;
  check(cudaMallocManaged(&stable, stress_bytes), "stress stable allocation");
  std::atomic<bool> stop{false};
  std::atomic<bool> failed{false};
  std::thread writer([&] {
    while (!stop.load(std::memory_order_acquire)) {
      for (size_t offset = 0; offset < stress_bytes; offset += page_size) {
        stable[offset]++;
      }
      if (cudaDeviceSynchronize() != cudaSuccess) {
        failed.store(true, std::memory_order_release);
        return;
      }
    }
  });
  for (int iteration = 0; iteration < 100; ++iteration) {
    unsigned char *temporary = nullptr;
    check(cudaMallocManaged(&temporary, stress_bytes),
          "stress temporary allocation");
    temporary[iteration * page_size] = static_cast<unsigned char>(iteration);
    check(cudaFree(temporary), "stress temporary free");
  }
  stop.store(true, std::memory_order_release);
  writer.join();
  if (failed.load(std::memory_order_acquire)) {
    std::fprintf(stderr, "concurrent dirty tracking stress failed\n");
    return 1;
  }
  check(cudaFree(stable), "stress stable free");
  std::puts("managed dirty tracking regression passed");
  return 0;
}
