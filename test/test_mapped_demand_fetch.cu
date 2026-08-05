// Integration test for demand-fetched mapped/managed device-to-host sync.
// Synchronization points invalidate the client mirror instead of copying it
// back; the first host touch fetches the affected chunks. Covers device
// writes at page granularity (mapped and managed, base and offset pointers,
// default and created streams), host writes flushed after a fetch, cudaMemcpy
// with a stale mirror as the host source, scattered and sequential reads
// through the chunked fetch path, and post-sync polling with no host access.
#include <chrono>
#include <cuda_runtime.h>
#include <stdio.h>
#include <string.h>

static const size_t kBytes = 64ull << 20;
static const size_t kPage = 4096;
static const size_t kSparseOffset = 33ull << 20;

static int failures = 0;

static int fatal(cudaError_t err, const char *what) {
  if (err == cudaSuccess) {
    return 0;
  }
  printf("RESULT: ERROR %s %s\n", what, cudaGetErrorString(err));
  return 1;
}

static void expect(bool ok, const char *what) {
  if (!ok) {
    printf("FAIL: %s\n", what);
    failures++;
  }
}

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

static int check_device_bytes(const unsigned char *device, unsigned char value,
                              int *dev_ok, const char *what) {
  int one = 1;
  if (fatal(cudaMemcpy(dev_ok, &one, sizeof(one), cudaMemcpyHostToDevice),
            what)) {
    return -1;
  }
  check_bytes<<<(unsigned)(kPage / 256), 256>>>(device, value, kPage, dev_ok);
  if (fatal(cudaDeviceSynchronize(), what)) {
    return -1;
  }
  int ok = 0;
  if (fatal(cudaMemcpy(&ok, dev_ok, sizeof(ok), cudaMemcpyDeviceToHost),
            what)) {
    return -1;
  }
  return ok;
}

static int run_case(const char *label, unsigned char *host,
                    unsigned char *device, cudaStream_t stream, int *dev_ok) {
  const unsigned char kHostA = 0x11, kHostB = 0x22, kDevC = 0x33,
                      kDevD = 0x44, kHostE = 0x55;
  char what[128];

  memset(host, kHostA, kPage);
  memset(host + (kBytes / 2), kHostB, kPage);
  write_bytes<<<(unsigned)(kPage / 256), 256, 0, stream>>>(
      device + kSparseOffset, kDevC, kPage);
  if (fatal(cudaStreamSynchronize(stream), label)) {
    return 1;
  }

  snprintf(what, sizeof(what), "%s: device write visible after sync", label);
  expect(host[kSparseOffset] == kDevC &&
             host[kSparseOffset + kPage - 1] == kDevC,
         what);
  snprintf(what, sizeof(what), "%s: host writes survive sync", label);
  expect(host[0] == kHostA && host[kBytes / 2] == kHostB, what);

  int ok = check_device_bytes(device, kHostA, dev_ok, label);
  if (ok < 0) {
    return 1;
  }
  snprintf(what, sizeof(what), "%s: device sees host writes", label);
  expect(ok == 1, what);

  write_bytes<<<(unsigned)(kPage / 256), 256>>>(device + kSparseOffset, kDevD,
                                                kPage);
  if (fatal(cudaDeviceSynchronize(), label)) {
    return 1;
  }
  memset(host + kPage, kHostE, kPage);
  snprintf(what, sizeof(what), "%s: fetch-before-write keeps device bytes",
           label);
  expect(host[kSparseOffset] == kDevD, what);

  ok = check_device_bytes(device + kPage, kHostE, dev_ok, label);
  if (ok < 0) {
    return 1;
  }
  snprintf(what, sizeof(what), "%s: post-fetch host write reaches device",
           label);
  expect(ok == 1, what);
  return 0;
}

int main() {
  int device_count = 0;
  if (fatal(cudaGetDeviceCount(&device_count), "device count")) {
    return 2;
  }
  if (device_count == 0) {
    printf("RESULT: ERROR no devices\n");
    return 2;
  }

  int *dev_ok = nullptr;
  if (fatal(cudaMalloc((void **)&dev_ok, sizeof(int)), "cudaMalloc ok flag")) {
    return 2;
  }

  unsigned char *mapped_host = nullptr;
  unsigned char *mapped_dev = nullptr;
  if (fatal(cudaHostAlloc((void **)&mapped_host, kBytes, cudaHostAllocMapped),
            "cudaHostAlloc") ||
      fatal(cudaHostGetDevicePointer((void **)&mapped_dev, mapped_host, 0),
            "cudaHostGetDevicePointer")) {
    return 2;
  }
  cudaStream_t stream;
  if (fatal(cudaStreamCreate(&stream), "stream create")) {
    return 2;
  }
  if (run_case("mapped", mapped_host, mapped_dev, stream, dev_ok) != 0) {
    return 2;
  }

  unsigned char *managed = nullptr;
  if (fatal(cudaMallocManaged((void **)&managed, kBytes),
            "cudaMallocManaged")) {
    return 2;
  }
  if (run_case("managed", managed, managed, (cudaStream_t)0, dev_ok) != 0) {
    return 2;
  }

  // Stale mirror as a cudaMemcpy host source.
  const unsigned char kDevF = 0x66;
  write_bytes<<<(unsigned)(kPage / 256), 256>>>(mapped_dev + kSparseOffset,
                                                kDevF, kPage);
  if (fatal(cudaDeviceSynchronize(), "htod source sync")) {
    return 2;
  }
  unsigned char *scratch = nullptr;
  if (fatal(cudaMalloc((void **)&scratch, kPage), "scratch alloc") ||
      fatal(cudaMemcpy(scratch, mapped_host + kSparseOffset, kPage,
                       cudaMemcpyHostToDevice),
            "htod from stale mirror")) {
    return 2;
  }
  int ok = check_device_bytes(scratch, kDevF, dev_ok, "scratch check");
  if (ok < 0) {
    return 2;
  }
  expect(ok == 1, "htod from stale mirror carries device bytes");

  // Scattered single-byte device writes read back in random order, then a
  // long sequential scan through the readahead escalation.
  const size_t kSpots[] = {0, (1u << 20) + 123, (32u << 20) + 4096,
                           kBytes - 1};
  for (size_t spot : kSpots) {
    write_bytes<<<1, 1>>>(mapped_dev + spot, 0x99, 1);
  }
  if (fatal(cudaDeviceSynchronize(), "scatter sync")) {
    return 2;
  }
  expect(mapped_host[kBytes - 1] == 0x99, "scatter: last byte");
  expect(mapped_host[0] == 0x99, "scatter: first byte");
  expect(mapped_host[(32u << 20) + 4096] == 0x99, "scatter: middle byte");
  size_t nonzero = 0;
  for (size_t i = 0; i < (16u << 20); ++i) {
    nonzero += mapped_host[i] != 0;
  }
  expect(mapped_host[(1u << 20) + 123] == 0x99 && nonzero >= 2,
         "scatter: sequential scan sees device bytes");

  // Host write into a freshly fetched chunk while most of the mapping is
  // still stale must reach the device.
  mapped_host[(1u << 20) + 200] = 0xAB;
  int one = 1;
  if (fatal(cudaMemcpy(dev_ok, &one, sizeof(one), cudaMemcpyHostToDevice),
            "ok reset 2")) {
    return 2;
  }
  check_bytes<<<1, 1>>>(mapped_dev + (1u << 20) + 200, 0xAB, 1, dev_ok);
  if (fatal(cudaDeviceSynchronize(), "mixed-state flush sync") ||
      fatal(cudaMemcpy(&ok, dev_ok, sizeof(ok), cudaMemcpyDeviceToHost),
            "ok readback 2")) {
    return 2;
  }
  expect(ok == 1, "host write in fresh chunk reaches device");

  // Post-sync polling with no host access should not move the allocation.
  write_bytes<<<(unsigned)(kPage / 256), 256>>>(mapped_dev, 0x77, kPage);
  if (fatal(cudaDeviceSynchronize(), "poll warmup sync")) {
    return 2;
  }
  auto poll_start = std::chrono::steady_clock::now();
  for (int i = 0; i < 200; ++i) {
    cudaStreamQuery(0);
  }
  auto poll_micros = std::chrono::duration_cast<std::chrono::microseconds>(
                         std::chrono::steady_clock::now() - poll_start)
                         .count();
  printf("INFO: 200 post-sync polls with a %zu MiB stale mapping: %lld us\n",
         kBytes >> 20, (long long)poll_micros);
  expect(mapped_host[0] == 0x77, "demand fetch after polling");

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
