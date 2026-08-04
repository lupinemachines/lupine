// Integration test for demand-fetched mapped/managed device-to-host sync.
//
// Synchronization points invalidate the client mirror (PROT_NONE) instead of
// copying the whole allocation back; the first host touch after that fetches
// the server backing through the fault handler. This exercises:
//   - device writes at a page-sized granularity visible after sync (mapped
//     and managed, base and offset pointers, default and created streams)
//   - host writes flushed to the device after a demand fetch
//   - cudaMemcpy with a stale mirror as the host source (the pre-touch that
//     keeps the fault out of the connection-locked payload write)
//   - repeated post-sync polling with no host access (no transfer expected)
#include <chrono>
#include <cuda_runtime.h>
#include <stdio.h>
#include <string.h>

static const size_t kBytes = 64ull << 20;
static const size_t kPage = 4096;
static const size_t kSparseOffset = 33ull << 20;

static int failures = 0;

#define REQUIRE(err, what)                                                     \
  do {                                                                         \
    cudaError_t e_ = (err);                                                    \
    if (e_ != cudaSuccess) {                                                   \
      printf("RESULT: ERROR %s %s\n", what, cudaGetErrorString(e_));           \
      return 2;                                                                \
    }                                                                          \
  } while (0)

#define EXPECT(cond, what)                                                     \
  do {                                                                         \
    if (!(cond)) {                                                             \
      printf("FAIL: %s\n", what);                                              \
      failures++;                                                              \
    }                                                                          \
  } while (0)

__global__ void write_bytes(unsigned char *dst, unsigned char value,
                            size_t count) {
  size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < count) {
    dst[idx] = value;
  }
}

__global__ void check_bytes(const unsigned char *src, unsigned char expect,
                            size_t count, int *ok) {
  size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < count && src[idx] != expect) {
    atomicExch(ok, 0);
  }
}

static int run_case(const char *label, unsigned char *host,
                    unsigned char *device, cudaStream_t stream, int *dev_ok) {
  const unsigned char kHostA = 0x11, kHostB = 0x22, kDevC = 0x33,
                      kDevD = 0x44, kHostE = 0x55;
  char what[128];

  // Host writes two pages, device writes one page at an offset pointer, all
  // visible after one synchronization.
  memset(host, kHostA, kPage);
  memset(host + (kBytes / 2), kHostB, kPage);
  write_bytes<<<(unsigned)(kPage / 256), 256, 0, stream>>>(
      device + kSparseOffset, kDevC, kPage);
  REQUIRE(cudaStreamSynchronize(stream), label);

  snprintf(what, sizeof(what), "%s: device write visible after sync", label);
  EXPECT(host[kSparseOffset] == kDevC &&
             host[kSparseOffset + kPage - 1] == kDevC,
         what);
  snprintf(what, sizeof(what), "%s: host writes survive sync", label);
  EXPECT(host[0] == kHostA && host[kBytes / 2] == kHostB, what);

  // The device must observe the pre-launch host writes (flush still works).
  int one = 1;
  REQUIRE(cudaMemcpy(dev_ok, &one, sizeof(one), cudaMemcpyHostToDevice),
          label);
  check_bytes<<<(unsigned)(kPage / 256), 256>>>(device, kHostA, kPage, dev_ok);
  REQUIRE(cudaDeviceSynchronize(), label);
  int ok = 0;
  REQUIRE(cudaMemcpy(&ok, dev_ok, sizeof(ok), cudaMemcpyDeviceToHost), label);
  snprintf(what, sizeof(what), "%s: device sees host writes", label);
  EXPECT(ok == 1, what);

  // Invalidate again, then write a different page first: the write fault must
  // fetch before tracking so the flush cannot push stale bytes, and the
  // device-written page must still be intact afterwards.
  write_bytes<<<(unsigned)(kPage / 256), 256>>>(device + kSparseOffset, kDevD,
                                                kPage);
  REQUIRE(cudaDeviceSynchronize(), label);
  memset(host + kPage, kHostE, kPage);
  snprintf(what, sizeof(what), "%s: fetch-before-write keeps device bytes",
           label);
  EXPECT(host[kSparseOffset] == kDevD, what);

  check_bytes<<<(unsigned)(kPage / 256), 256>>>(device + kPage, kHostE, kPage,
                                                dev_ok);
  REQUIRE(cudaDeviceSynchronize(), label);
  REQUIRE(cudaMemcpy(&ok, dev_ok, sizeof(ok), cudaMemcpyDeviceToHost), label);
  snprintf(what, sizeof(what), "%s: post-fetch host write reaches device",
           label);
  EXPECT(ok == 1, what);
  return 0;
}

int main() {
  int device_count = 0;
  REQUIRE(cudaGetDeviceCount(&device_count), "device count");
  if (device_count == 0) {
    printf("RESULT: ERROR no devices\n");
    return 2;
  }

  int *dev_ok = nullptr;
  REQUIRE(cudaMalloc((void **)&dev_ok, sizeof(int)), "cudaMalloc ok flag");

  // Mapped host allocation.
  unsigned char *mapped_host = nullptr;
  unsigned char *mapped_dev = nullptr;
  REQUIRE(cudaHostAlloc((void **)&mapped_host, kBytes, cudaHostAllocMapped),
          "cudaHostAlloc");
  REQUIRE(cudaHostGetDevicePointer((void **)&mapped_dev, mapped_host, 0),
          "cudaHostGetDevicePointer");
  cudaStream_t stream;
  REQUIRE(cudaStreamCreate(&stream), "stream create");
  if (run_case("mapped", mapped_host, mapped_dev, stream, dev_ok) != 0) {
    return 2;
  }

  // Managed allocation, same expectations through the default stream.
  unsigned char *managed = nullptr;
  REQUIRE(cudaMallocManaged((void **)&managed, kBytes), "cudaMallocManaged");
  if (run_case("managed", managed, managed, (cudaStream_t)0, dev_ok) != 0) {
    return 2;
  }

  // Stale mirror as a cudaMemcpy host source: the copy has to refresh the
  // mirror before it enters the connection-locked payload write, and the
  // destination must receive the device-written bytes.
  const unsigned char kDevF = 0x66;
  write_bytes<<<(unsigned)(kPage / 256), 256>>>(mapped_dev + kSparseOffset,
                                                kDevF, kPage);
  REQUIRE(cudaDeviceSynchronize(), "htod source sync");
  unsigned char *scratch = nullptr;
  REQUIRE(cudaMalloc((void **)&scratch, kPage), "scratch alloc");
  REQUIRE(cudaMemcpy(scratch, mapped_host + kSparseOffset, kPage,
                     cudaMemcpyHostToDevice),
          "htod from stale mirror");
  int one = 1;
  REQUIRE(cudaMemcpy(dev_ok, &one, sizeof(one), cudaMemcpyHostToDevice),
          "ok reset");
  check_bytes<<<(unsigned)(kPage / 256), 256>>>(scratch, kDevF, kPage, dev_ok);
  REQUIRE(cudaDeviceSynchronize(), "scratch check sync");
  int ok = 0;
  REQUIRE(cudaMemcpy(&ok, dev_ok, sizeof(ok), cudaMemcpyDeviceToHost),
          "ok readback");
  EXPECT(ok == 1, "htod from stale mirror carries device bytes");

  // Post-sync polling with no host access should not move the allocation.
  write_bytes<<<(unsigned)(kPage / 256), 256>>>(mapped_dev, 0x77, kPage);
  REQUIRE(cudaDeviceSynchronize(), "poll warmup sync");
  auto poll_start = std::chrono::steady_clock::now();
  for (int i = 0; i < 200; ++i) {
    cudaStreamQuery(0);
  }
  auto poll_micros = std::chrono::duration_cast<std::chrono::microseconds>(
                         std::chrono::steady_clock::now() - poll_start)
                         .count();
  printf("INFO: 200 post-sync polls with a %zu MiB stale mapping: %lld us\n",
         kBytes >> 20, (long long)poll_micros);
  EXPECT(mapped_host[0] == 0x77, "demand fetch after polling");

  cudaFree(scratch);
  cudaFree(dev_ok);
  cudaFreeHost(mapped_host);
  cudaFree(managed);
  cudaStreamDestroy(stream);

  if (failures == 0) {
    printf("RESULT: PASS\n");
    return 0;
  }
  printf("RESULT: FAIL (%d)\n", failures);
  return 1;
}
