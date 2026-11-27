// Repeated allocation sizes may be served from earlier batched allocations,
// which must still behave like separate cudaMalloc results: no two live blocks
// overlap, each holds its own data, an allocation that cannot succeed reports
// out of memory without breaking later ones, and a reset device never hands
// out memory of the context it destroyed.
// This file is auto-discovered by test/run_custom_tests.sh.
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <utility>
#include <vector>

#define CHECK(call)                                                            \
  do {                                                                         \
    cudaError_t check_result_ = (call);                                        \
    if (check_result_ != cudaSuccess) {                                        \
      std::fprintf(stderr, "%s failed: %s\n", #call,                           \
                   cudaGetErrorString(check_result_));                         \
      std::exit(2);                                                            \
    }                                                                          \
  } while (0)

namespace {

struct block {
  unsigned char *ptr;
  size_t bytes;
};

__global__ void stamp(unsigned *data, size_t words, unsigned tag) {
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < words;
       i += gridDim.x * blockDim.x) {
    data[i] = tag ^ static_cast<unsigned>(i);
  }
}

void allocate(std::vector<block> &blocks, size_t bytes, int count) {
  for (int i = 0; i < count; ++i) {
    void *ptr = nullptr;
    CHECK(cudaMalloc(&ptr, bytes));
    blocks.push_back({static_cast<unsigned char *>(ptr), bytes});
  }
}

void require_disjoint(std::vector<block> blocks, const char *label) {
  std::sort(blocks.begin(), blocks.end(),
            [](const block &a, const block &b) { return a.ptr < b.ptr; });
  for (size_t i = 1; i < blocks.size(); ++i) {
    if (blocks[i - 1].ptr + blocks[i - 1].bytes > blocks[i].ptr) {
      std::fprintf(stderr, "FAIL: %s: blocks %p and %p overlap\n", label,
                   static_cast<void *>(blocks[i - 1].ptr),
                   static_cast<void *>(blocks[i].ptr));
      std::exit(1);
    }
  }
}

// Every block gets its own pattern, then every block is read back, so two
// blocks sharing memory would show the later block's pattern in the earlier.
void require_own_data(const std::vector<block> &blocks, const char *label) {
  for (size_t b = 0; b < blocks.size(); ++b) {
    stamp<<<64, 256>>>(reinterpret_cast<unsigned *>(blocks[b].ptr),
                       blocks[b].bytes / sizeof(unsigned),
                       0x9e3779b9u * static_cast<unsigned>(b + 1));
    CHECK(cudaGetLastError());
  }
  std::vector<unsigned> back;
  for (size_t b = 0; b < blocks.size(); ++b) {
    size_t words = blocks[b].bytes / sizeof(unsigned);
    back.resize(words);
    CHECK(cudaMemcpy(back.data(), blocks[b].ptr, words * sizeof(unsigned),
                     cudaMemcpyDeviceToHost));
    unsigned tag = 0x9e3779b9u * static_cast<unsigned>(b + 1);
    for (size_t i = 0; i < words; i += 4093) {
      if (back[i] != (tag ^ static_cast<unsigned>(i))) {
        std::fprintf(stderr, "FAIL: %s: block %zu word %zu holds 0x%08x\n",
                     label, b, i, back[i]);
        std::exit(1);
      }
    }
  }
}

void free_all(std::vector<block> &blocks) {
  for (const block &b : blocks) {
    CHECK(cudaFree(b.ptr));
  }
  blocks.clear();
}

} // namespace

int main() {
  CHECK(cudaSetDevice(0));
  std::vector<block> blocks;
  allocate(blocks, 2u << 20, 24);
  allocate(blocks, 20u << 20, 12);
  allocate(blocks, 4096, 16);
  require_disjoint(blocks, "repeated sizes");
  require_own_data(blocks, "repeated sizes");

  // Free every other block, then allocate the same sizes again while the rest
  // stay live.
  std::vector<block> kept;
  for (size_t i = 0; i < blocks.size(); ++i) {
    if (i % 2 == 0) {
      CHECK(cudaFree(blocks[i].ptr));
    } else {
      kept.push_back(blocks[i]);
    }
  }
  blocks = std::move(kept);
  allocate(blocks, 2u << 20, 12);
  allocate(blocks, 20u << 20, 6);
  require_disjoint(blocks, "after partial free");
  require_own_data(blocks, "after partial free");

  void *impossible = nullptr;
  cudaError_t result = cudaMalloc(&impossible, size_t{1} << 50);
  if (result != cudaErrorMemoryAllocation) {
    std::fprintf(stderr, "FAIL: impossible allocation returned %s\n",
                 cudaGetErrorName(result));
    return 1;
  }
  (void)cudaGetLastError();
  allocate(blocks, 2u << 20, 4);
  require_disjoint(blocks, "after a failed allocation");
  require_own_data(blocks, "after a failed allocation");
  free_all(blocks);

  // Allocations made before a reset die with that context; none may come back.
  allocate(blocks, 2u << 20, 6);
  blocks.clear();
  CHECK(cudaDeviceReset());
  CHECK(cudaSetDevice(0));
  allocate(blocks, 2u << 20, 12);
  require_disjoint(blocks, "after a device reset");
  require_own_data(blocks, "after a device reset");
  free_all(blocks);

  std::printf("repeated allocation sizes behave as separate allocations\n");
  return 0;
}
