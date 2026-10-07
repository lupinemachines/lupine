#include <cuda_runtime_api.h>

#include <cstdio>
#include <cstdlib>
#include <vector>

static void require(cudaError_t status, const char *stage) {
  if (status != cudaSuccess) {
    std::fprintf(stderr, "%s: %s\n", stage, cudaGetErrorString(status));
    std::exit(1);
  }
}

// The server stages a pushed host-to-device copy through a per-context ring
// with its own transfer stream. cudaDeviceReset destroys the primary context
// and every stream in it, so a reset that leaves the ring behind leaves the
// next copy synchronizing a freed stream: the server child dies on the
// garbage handle and the client sees cudaErrorDeviceUninitialized from here
// on. Pageable and page-locked sources take different routes into the ring,
// so each round copies both.
static void copy_round(const char *stage) {
  const size_t count = 64 * 1024;
  std::vector<int> pageable(count);
  for (size_t i = 0; i < count; ++i) {
    pageable[i] = static_cast<int>(i);
  }

  int *device = nullptr;
  require(cudaMalloc(&device, count * sizeof(int)), stage);
  require(cudaMemcpy(device, pageable.data(), count * sizeof(int),
                     cudaMemcpyHostToDevice),
          stage);

  int *pinned = nullptr;
  require(cudaMallocHost(&pinned, count * sizeof(int)), stage);
  for (size_t i = 0; i < count; ++i) {
    pinned[i] = static_cast<int>(i) + 1;
  }
  require(cudaMemcpy(device, pinned, count * sizeof(int),
                     cudaMemcpyHostToDevice),
          stage);

  std::vector<int> host(count, 0);
  require(cudaMemcpy(host.data(), device, count * sizeof(int),
                     cudaMemcpyDeviceToHost),
          stage);
  for (size_t i = 0; i < count; ++i) {
    if (host[i] != static_cast<int>(i) + 1) {
      std::fprintf(stderr, "%s: element %zu read back %d\n", stage, i, host[i]);
      std::exit(1);
    }
  }

  require(cudaFreeHost(pinned), stage);
  require(cudaFree(device), stage);
}

int main() {
  copy_round("first round");
  for (int round = 0; round < 3; ++round) {
    require(cudaDeviceReset(), "cudaDeviceReset");
    copy_round("after reset");
  }

  int count = 0;
  require(cudaGetDeviceCount(&count), "cudaGetDeviceCount");
  if (count > 1) {
    require(cudaSetDevice(1), "cudaSetDevice(1)");
    copy_round("second device");
    require(cudaDeviceReset(), "cudaDeviceReset on the second device");
    copy_round("second device after reset");
    require(cudaSetDevice(0), "cudaSetDevice(0)");
    copy_round("first device after resetting the second");
  }

  std::puts("host-to-device copies survive cudaDeviceReset");
  return 0;
}
