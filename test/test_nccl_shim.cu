// Exercises the NCCL shim end to end against the remote devices: version and
// error strings, unique ids, a one-rank communicator (collectives in and out
// of groups, pre-multiplied sums with host and device scalars, send to self,
// ncclMemAlloc buffers, a configured communicator, split, finalize and
// abort) and, with two GPUs, a clique made by ncclCommInitAll and one made by
// a group of ncclCommInitRank calls, driven from one thread with groups and
// from a thread per rank without them. Results are checked against CPU
// references.
#include <cuda_runtime.h>
#include <nccl.h>

#include <cstdio>
#include <cstring>
#include <thread>
#include <vector>

#define CHECK_CUDA(call)                                                       \
  do {                                                                         \
    cudaError_t status = (call);                                               \
    if (status != cudaSuccess) {                                               \
      fprintf(stderr, "%s failed: %s\n", #call, cudaGetErrorName(status));     \
      return 1;                                                                \
    }                                                                          \
  } while (0)

#define CHECK_NCCL(call)                                                       \
  do {                                                                         \
    ncclResult_t status = (call);                                              \
    if (status != ncclSuccess) {                                               \
      fprintf(stderr, "%s failed: %d %s\n", #call, status,                     \
              ncclGetErrorString(status));                                     \
      return 1;                                                                \
    }                                                                          \
  } while (0)

#define EXPECT(condition)                                                      \
  do {                                                                         \
    if (!(condition)) {                                                        \
      fprintf(stderr, "expectation failed: %s\n", #condition);                 \
      return 1;                                                                \
    }                                                                          \
  } while (0)

static int upload(const std::vector<float> &host, float **device) {
  CHECK_CUDA(cudaMalloc(device, host.size() * sizeof(float)));
  CHECK_CUDA(cudaMemcpy(*device, host.data(), host.size() * sizeof(float),
                        cudaMemcpyHostToDevice));
  return 0;
}

static int download(const float *device, std::vector<float> &host) {
  CHECK_CUDA(cudaMemcpy(host.data(), device, host.size() * sizeof(float),
                        cudaMemcpyDeviceToHost));
  return 0;
}

static int compare(const std::vector<float> &got,
                   const std::vector<float> &want, const char *what) {
  for (size_t i = 0; i < want.size(); ++i) {
    if (got[i] != want[i]) {
      fprintf(stderr, "%s[%zu] = %g, expected %g\n", what, i, got[i], want[i]);
      return 1;
    }
  }
  return 0;
}

static std::vector<float> ramp(size_t count, float base) {
  std::vector<float> values(count);
  for (size_t i = 0; i < count; ++i) {
    values[i] = base + static_cast<float>(i);
  }
  return values;
}

static int test_strings() {
  int version = 0;
  CHECK_NCCL(ncclGetVersion(&version));
  EXPECT(version >= 21304);
  printf("NCCL %d, built against %d\n", version, NCCL_VERSION_CODE);
  EXPECT(strstr(ncclGetErrorString(ncclInvalidArgument), "invalid") !=
         nullptr);
  EXPECT(ncclGetErrorString(ncclInvalidArgument) ==
         ncclGetErrorString(ncclInvalidArgument));
  EXPECT(ncclGroupEnd() == ncclInvalidUsage);

  ncclUniqueId first, second;
  CHECK_NCCL(ncclGetUniqueId(&first));
  CHECK_NCCL(ncclGetUniqueId(&second));
  EXPECT(memcmp(&first, &second, sizeof(first)) != 0);
  printf("strings and unique ids: passed\n");
  return 0;
}

// One rank on device 0: every collective is its own identity, which checks
// the marshalling of each call and of the group brackets around them.
static int test_single_rank() {
  CHECK_CUDA(cudaSetDevice(0));
  ncclUniqueId id;
  CHECK_NCCL(ncclGetUniqueId(&id));
  ncclComm_t comm = nullptr;
  CHECK_NCCL(ncclCommInitRank(&comm, 1, id, 0));
  int count = 0, rank = -1, device = -1;
  CHECK_NCCL(ncclCommCount(comm, &count));
  CHECK_NCCL(ncclCommUserRank(comm, &rank));
  CHECK_NCCL(ncclCommCuDevice(comm, &device));
  EXPECT(count == 1 && rank == 0 && device == 0);

  cudaStream_t stream = nullptr;
  CHECK_CUDA(cudaStreamCreate(&stream));
  constexpr size_t kCount = 64;
  const std::vector<float> input = ramp(kCount, 1.0f);
  std::vector<float> output(kCount);
  float *send = nullptr, *recv = nullptr;
  if (upload(input, &send) || upload(std::vector<float>(kCount, 0), &recv)) {
    return 1;
  }

  // Outside a group each call is a request with the library's own result.
  CHECK_NCCL(ncclAllReduce(send, recv, kCount, ncclFloat, ncclSum, comm,
                           stream));
  CHECK_CUDA(cudaStreamSynchronize(stream));
  if (download(recv, output) || compare(output, input, "AllReduce")) {
    return 1;
  }
  EXPECT(ncclAllReduce(send, recv, kCount, ncclNumTypes, ncclSum, comm,
                       stream) == ncclInvalidArgument);

  // Inside a group the calls are queued; the group's result covers them.
  CHECK_NCCL(ncclGroupStart());
  CHECK_NCCL(ncclBroadcast(send, recv, kCount, ncclFloat, 0, comm, stream));
  CHECK_NCCL(ncclGroupStart());
  CHECK_NCCL(ncclAllGather(send, recv, kCount, ncclFloat, comm, stream));
  CHECK_NCCL(ncclGroupEnd());
  CHECK_NCCL(ncclReduceScatter(send, recv, kCount, ncclFloat, ncclSum, comm,
                               stream));
  CHECK_NCCL(ncclGroupEnd());
  CHECK_CUDA(cudaStreamSynchronize(stream));
  if (download(recv, output) || compare(output, input, "group")) {
    return 1;
  }

  // Send to self needs both halves in one group.
  CHECK_NCCL(ncclGroupStart());
  CHECK_NCCL(ncclSend(send, kCount, ncclFloat, 0, comm, stream));
  CHECK_NCCL(ncclRecv(recv, kCount, ncclFloat, 0, comm, stream));
  CHECK_NCCL(ncclGroupEnd());
  CHECK_CUDA(cudaStreamSynchronize(stream));
  if (download(recv, output) || compare(output, input, "Send/Recv")) {
    return 1;
  }

  // A pre-multiplied sum scales by a host scalar read at creation, or by a
  // device scalar read while the collective runs.
  std::vector<float> scaled(kCount);
  for (size_t i = 0; i < kCount; ++i) {
    scaled[i] = 3.0f * input[i];
  }
  float host_scale = 3.0f;
  ncclRedOp_t host_op;
  CHECK_NCCL(ncclRedOpCreatePreMulSum(&host_op, &host_scale, ncclFloat,
                                      ncclScalarHostImmediate, comm));
  host_scale = 100.0f;
  CHECK_NCCL(ncclAllReduce(send, recv, kCount, ncclFloat, host_op, comm,
                           stream));
  CHECK_CUDA(cudaStreamSynchronize(stream));
  if (download(recv, output) || compare(output, scaled, "host PreMulSum")) {
    return 1;
  }
  CHECK_NCCL(ncclRedOpDestroy(host_op, comm));
  float *device_scale = nullptr;
  if (upload(std::vector<float>{3.0f}, &device_scale)) {
    return 1;
  }
  ncclRedOp_t device_op;
  CHECK_NCCL(ncclRedOpCreatePreMulSum(&device_op, device_scale, ncclFloat,
                                      ncclScalarDevice, comm));
  CHECK_NCCL(ncclGroupStart());
  CHECK_NCCL(ncclAllReduce(send, recv, kCount, ncclFloat, device_op, comm,
                           stream));
  CHECK_NCCL(ncclGroupEnd());
  CHECK_CUDA(cudaStreamSynchronize(stream));
  if (download(recv, output) || compare(output, scaled, "device PreMulSum")) {
    return 1;
  }
  CHECK_NCCL(ncclRedOpDestroy(device_op, comm));
  CHECK_CUDA(cudaFree(device_scale));

#if NCCL_VERSION_CODE >= 21903
  void *pool = nullptr;
  CHECK_NCCL(ncclMemAlloc(&pool, kCount * sizeof(float)));
  CHECK_CUDA(cudaMemcpy(pool, input.data(), kCount * sizeof(float),
                        cudaMemcpyHostToDevice));
  CHECK_NCCL(ncclAllReduce(pool, recv, kCount, ncclFloat, ncclSum, comm,
                           stream));
  CHECK_CUDA(cudaStreamSynchronize(stream));
  if (download(recv, output) || compare(output, input, "ncclMemAlloc")) {
    return 1;
  }
  CHECK_NCCL(ncclMemFree(pool));
#endif

  ncclResult_t async_error = ncclInternalError;
  CHECK_NCCL(ncclCommGetAsyncError(comm, &async_error));
  EXPECT(async_error == ncclSuccess);
  (void)ncclGetLastError(comm);

#if NCCL_VERSION_CODE >= 21801
  ncclComm_t split = nullptr;
  CHECK_NCCL(ncclCommSplit(comm, 0, 0, &split, nullptr));
  CHECK_NCCL(ncclCommCount(split, &count));
  EXPECT(split != nullptr && count == 1);
  CHECK_NCCL(ncclCommDestroy(split));
#endif
  CHECK_NCCL(ncclCommFinalize(comm));
  CHECK_NCCL(ncclCommDestroy(comm));

#if NCCL_VERSION_CODE >= 22703
  // The configuration's strings travel with it.
  ncclConfig_t config = NCCL_CONFIG_INITIALIZER;
  config.commName = "lupine-test";
  CHECK_NCCL(ncclGetUniqueId(&id));
  CHECK_NCCL(ncclCommInitRankConfig(&comm, 1, id, 0, &config));
  CHECK_NCCL(ncclCommCount(comm, &count));
  EXPECT(count == 1);
  CHECK_NCCL(ncclCommAbort(comm));
#endif

#if NCCL_VERSION_CODE >= 23007
  const char **keys = nullptr;
  int key_count = 0;
  CHECK_NCCL(ncclParamGetAllParameterKeys(&keys, &key_count));
  EXPECT(key_count > 0 && keys[0] != nullptr);
  const char *value = nullptr;
  int value_length = -1;
  CHECK_NCCL(ncclParamGetParameter(keys[0], &value, &value_length));
  EXPECT(value != nullptr && value_length >= 0);
#endif

  CHECK_CUDA(cudaFree(send));
  CHECK_CUDA(cudaFree(recv));
  CHECK_CUDA(cudaStreamDestroy(stream));
  printf("single rank: passed\n");
  return 0;
}

struct rank_buffers {
  cudaStream_t stream = nullptr;
  float *send = nullptr;
  float *recv = nullptr;
};

constexpr size_t kRankCount = 32;

static int make_buffers(int device, rank_buffers &buffers) {
  CHECK_CUDA(cudaSetDevice(device));
  CHECK_CUDA(cudaStreamCreate(&buffers.stream));
  if (upload(ramp(kRankCount, static_cast<float>(device + 1)),
             &buffers.send) ||
      upload(std::vector<float>(2 * kRankCount, 0), &buffers.recv)) {
    return 1;
  }
  return 0;
}

// Two ranks, one per device, checked after AllReduce, Broadcast, AllGather
// and ReduceScatter issued in groups from one thread, and a ring of sends.
static int two_rank_collectives(ncclComm_t comms[2], const char *what) {
  rank_buffers ranks[2];
  for (int r = 0; r < 2; ++r) {
    if (make_buffers(r, ranks[r])) {
      return 1;
    }
  }
  const std::vector<float> a = ramp(kRankCount, 1.0f);
  const std::vector<float> b = ramp(kRankCount, 2.0f);
  std::vector<float> sum(kRankCount), gathered(2 * kRankCount), output;
  for (size_t i = 0; i < kRankCount; ++i) {
    sum[i] = a[i] + b[i];
    gathered[i] = a[i];
    gathered[kRankCount + i] = b[i];
  }

  CHECK_NCCL(ncclGroupStart());
  for (int r = 0; r < 2; ++r) {
    CHECK_NCCL(ncclAllReduce(ranks[r].send, ranks[r].recv, kRankCount,
                             ncclFloat, ncclSum, comms[r], ranks[r].stream));
  }
  CHECK_NCCL(ncclGroupEnd());
  for (int r = 0; r < 2; ++r) {
    CHECK_CUDA(cudaSetDevice(r));
    CHECK_CUDA(cudaStreamSynchronize(ranks[r].stream));
    output.assign(kRankCount, 0);
    if (download(ranks[r].recv, output) || compare(output, sum, what)) {
      return 1;
    }
  }

  CHECK_NCCL(ncclGroupStart());
  for (int r = 0; r < 2; ++r) {
    CHECK_NCCL(ncclBroadcast(ranks[r].send, ranks[r].recv, kRankCount,
                             ncclFloat, 1, comms[r], ranks[r].stream));
  }
  CHECK_NCCL(ncclGroupEnd());
  for (int r = 0; r < 2; ++r) {
    CHECK_CUDA(cudaSetDevice(r));
    CHECK_CUDA(cudaStreamSynchronize(ranks[r].stream));
    output.assign(kRankCount, 0);
    if (download(ranks[r].recv, output) || compare(output, b, "Broadcast")) {
      return 1;
    }
  }

  CHECK_NCCL(ncclGroupStart());
  for (int r = 0; r < 2; ++r) {
    CHECK_NCCL(ncclAllGather(ranks[r].send, ranks[r].recv, kRankCount,
                             ncclFloat, comms[r], ranks[r].stream));
  }
  CHECK_NCCL(ncclGroupEnd());
  for (int r = 0; r < 2; ++r) {
    CHECK_CUDA(cudaSetDevice(r));
    CHECK_CUDA(cudaStreamSynchronize(ranks[r].stream));
    output.assign(2 * kRankCount, 0);
    if (download(ranks[r].recv, output) ||
        compare(output, gathered, "AllGather")) {
      return 1;
    }
  }

  // The gathered buffer holds both ranks' halves; scattering its sum gives
  // each rank twice its own half.
  CHECK_NCCL(ncclGroupStart());
  for (int r = 0; r < 2; ++r) {
    CHECK_NCCL(ncclReduceScatter(ranks[r].recv, ranks[r].send, kRankCount,
                                 ncclFloat, ncclSum, comms[r],
                                 ranks[r].stream));
  }
  CHECK_NCCL(ncclGroupEnd());
  for (int r = 0; r < 2; ++r) {
    CHECK_CUDA(cudaSetDevice(r));
    CHECK_CUDA(cudaStreamSynchronize(ranks[r].stream));
    output.assign(kRankCount, 0);
    std::vector<float> want(kRankCount);
    for (size_t i = 0; i < kRankCount; ++i) {
      want[i] = 2.0f * gathered[r * kRankCount + i];
    }
    if (download(ranks[r].send, output) ||
        compare(output, want, "ReduceScatter")) {
      return 1;
    }
  }

  // Rank r sends its buffer to the other rank.
  CHECK_NCCL(ncclGroupStart());
  for (int r = 0; r < 2; ++r) {
    CHECK_NCCL(ncclSend(ranks[r].send, kRankCount, ncclFloat, 1 - r, comms[r],
                        ranks[r].stream));
    CHECK_NCCL(ncclRecv(ranks[r].recv, kRankCount, ncclFloat, 1 - r, comms[r],
                        ranks[r].stream));
  }
  CHECK_NCCL(ncclGroupEnd());
  for (int r = 0; r < 2; ++r) {
    CHECK_CUDA(cudaSetDevice(r));
    CHECK_CUDA(cudaStreamSynchronize(ranks[r].stream));
    output.assign(kRankCount, 0);
    std::vector<float> want(kRankCount);
    for (size_t i = 0; i < kRankCount; ++i) {
      want[i] = 2.0f * gathered[(1 - r) * kRankCount + i];
    }
    if (download(ranks[r].recv, output) || compare(output, want, "Send/Recv")) {
      return 1;
    }
  }

  for (int r = 0; r < 2; ++r) {
    CHECK_CUDA(cudaSetDevice(r));
    CHECK_CUDA(cudaFree(ranks[r].send));
    CHECK_CUDA(cudaFree(ranks[r].recv));
    CHECK_CUDA(cudaStreamDestroy(ranks[r].stream));
  }
  return 0;
}

static int test_init_all() {
  ncclComm_t comms[2] = {};
  const int devices[2] = {0, 1};
  CHECK_NCCL(ncclCommInitAll(comms, 2, devices));
  int device = -1;
  CHECK_NCCL(ncclCommCuDevice(comms[1], &device));
  EXPECT(device == 1);
  if (two_rank_collectives(comms, "InitAll AllReduce")) {
    return 1;
  }
  for (ncclComm_t comm : comms) {
    CHECK_NCCL(ncclCommDestroy(comm));
  }
  printf("ncclCommInitAll clique: passed\n");
  return 0;
}

static int test_group_init() {
  ncclUniqueId id;
  CHECK_NCCL(ncclGetUniqueId(&id));
  ncclComm_t comms[2] = {};
  CHECK_NCCL(ncclGroupStart());
  for (int r = 0; r < 2; ++r) {
    CHECK_CUDA(cudaSetDevice(r));
    CHECK_NCCL(ncclCommInitRank(&comms[r], 2, id, r));
  }
  CHECK_NCCL(ncclGroupEnd());
  if (two_rank_collectives(comms, "grouped InitRank AllReduce")) {
    return 1;
  }
  for (ncclComm_t comm : comms) {
    CHECK_NCCL(ncclCommDestroy(comm));
  }
  printf("grouped ncclCommInitRank clique: passed\n");
  return 0;
}

// A thread per rank and no groups: each call waits for its peer natively,
// here on the server thread that serves the calling client thread.
static int test_thread_per_rank() {
  ncclUniqueId id;
  CHECK_NCCL(ncclGetUniqueId(&id));
  int failures[2] = {};
  std::vector<float> results[2];
  std::thread threads[2];
  for (int r = 0; r < 2; ++r) {
    threads[r] = std::thread([&, r]() {
      failures[r] = [&]() -> int {
        CHECK_CUDA(cudaSetDevice(r));
        ncclComm_t comm = nullptr;
        CHECK_NCCL(ncclCommInitRank(&comm, 2, id, r));
        rank_buffers buffers;
        if (make_buffers(r, buffers)) {
          return 1;
        }
        for (int round = 0; round < 3; ++round) {
          CHECK_NCCL(ncclAllReduce(buffers.send, buffers.recv, kRankCount,
                                   ncclFloat, ncclSum, comm, buffers.stream));
        }
        CHECK_CUDA(cudaStreamSynchronize(buffers.stream));
        results[r].assign(kRankCount, 0);
        if (download(buffers.recv, results[r])) {
          return 1;
        }
        CHECK_CUDA(cudaFree(buffers.send));
        CHECK_CUDA(cudaFree(buffers.recv));
        CHECK_CUDA(cudaStreamDestroy(buffers.stream));
        CHECK_NCCL(ncclCommDestroy(comm));
        return 0;
      }();
    });
  }
  for (std::thread &thread : threads) {
    thread.join();
  }
  EXPECT(failures[0] == 0 && failures[1] == 0);
  const std::vector<float> a = ramp(kRankCount, 1.0f);
  const std::vector<float> b = ramp(kRankCount, 2.0f);
  std::vector<float> sum(kRankCount);
  for (size_t i = 0; i < kRankCount; ++i) {
    sum[i] = a[i] + b[i];
  }
  if (compare(results[0], sum, "thread rank 0") ||
      compare(results[1], sum, "thread rank 1")) {
    return 1;
  }
  printf("thread per rank: passed\n");
  return 0;
}

int main() {
  if (test_strings() || test_single_rank()) {
    return 1;
  }
  int devices = 0;
  CHECK_CUDA(cudaGetDeviceCount(&devices));
  if (devices < 2) {
    printf("two ranks: skipped, one device\n");
  } else if (test_init_all() || test_group_init() || test_thread_per_rank()) {
    return 1;
  }
  printf("nccl shim: all checks passed\n");
  return 0;
}
