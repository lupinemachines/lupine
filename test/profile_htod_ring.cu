#include <cuda.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <new>
#include <thread>
#include <vector>

// Standalone comparison of pageable, registered, and CUDA-allocated pinned
// staging. The overlapped pageable graph case is opt-in because it is expected
// to deadlock; use HTOD_RING_TRY_OVERLAPPED_PAGEABLE=1 under a timeout to
// reproduce it. HTOD_RING_GRAPH_ONLY, HTOD_RING_GRAPH_MODE, and
// HTOD_RING_GRAPH_RATE select smaller runs.

using clock_type = std::chrono::steady_clock;

static constexpr size_t kSlotCount = 2;
static constexpr size_t kSlotBytes = 8 * 1024 * 1024;
static constexpr size_t kRingBytes = kSlotCount * kSlotBytes;

static const char *cuda_name(CUresult result) {
  const char *name = nullptr;
  return cuGetErrorName(result, &name) == CUDA_SUCCESS && name != nullptr
             ? name
             : "CUDA_ERROR_UNKNOWN";
}

#define CUDA_TRY(call)                                                         \
  do {                                                                         \
    CUresult cuda_try_result = (call);                                         \
    if (cuda_try_result != CUDA_SUCCESS) {                                     \
      std::fprintf(stderr, "%s failed: %s\n", #call,                           \
                   cuda_name(cuda_try_result));                                \
      std::exit(1);                                                            \
    }                                                                          \
  } while (0)

enum class host_mode { pageable, registered, pinned };

static const char *mode_name(host_mode mode) {
  switch (mode) {
  case host_mode::pageable:
    return "pageable";
  case host_mode::registered:
    return "registered";
  case host_mode::pinned:
    return "pinned";
  }
  return "unknown";
}

struct host_ring {
  host_mode mode = host_mode::pageable;
  unsigned char *data = nullptr;

  CUresult allocate(host_mode requested_mode) {
    mode = requested_mode;
    if (mode == host_mode::pinned) {
      return cuMemHostAlloc(reinterpret_cast<void **>(&data), kRingBytes,
                            CU_MEMHOSTALLOC_PORTABLE);
    }
    if (posix_memalign(reinterpret_cast<void **>(&data), 4096, kRingBytes) !=
        0) {
      data = nullptr;
      return CUDA_ERROR_OUT_OF_MEMORY;
    }
    if (mode == host_mode::registered) {
      CUresult result =
          cuMemHostRegister(data, kRingBytes, CU_MEMHOSTREGISTER_PORTABLE);
      if (result != CUDA_SUCCESS) {
        std::free(data);
        data = nullptr;
        return result;
      }
    }
    return CUDA_SUCCESS;
  }

  void release() {
    if (data == nullptr) {
      return;
    }
    if (mode == host_mode::pinned) {
      CUDA_TRY(cuMemFreeHost(data));
    } else {
      if (mode == host_mode::registered) {
        CUDA_TRY(cuMemHostUnregister(data));
      }
      std::free(data);
    }
    data = nullptr;
  }

  unsigned char *slot(size_t index) {
    return data + (index % kSlotCount) * kSlotBytes;
  }
};

struct sample_summary {
  double median_ms = 0;
  double p95_ms = 0;
};

static sample_summary summarize(std::vector<double> samples) {
  std::sort(samples.begin(), samples.end());
  const size_t median = samples.size() / 2;
  const size_t p95 = static_cast<size_t>(
      std::ceil(static_cast<double>(samples.size()) * 0.95) - 1.0);
  return {samples[median], samples[std::min(p95, samples.size() - 1)]};
}

static void profile_initialization() {
  constexpr int kIterations = 25;
  std::printf("\ninitialization_mode,allocation_median_ms,allocation_p95_ms,"
              "touch_median_ms,touch_p95_ms,release_median_ms,"
              "release_p95_ms\n");
  for (host_mode mode :
       {host_mode::pageable, host_mode::registered, host_mode::pinned}) {
    std::vector<double> allocation;
    std::vector<double> touch;
    std::vector<double> release;
    for (int iteration = 0; iteration < kIterations; ++iteration) {
      host_ring ring;
      auto started = clock_type::now();
      CUDA_TRY(ring.allocate(mode));
      auto allocated = clock_type::now();
      std::memset(ring.data, iteration, kRingBytes);
      // Make the page touch observable to the optimizer.
      volatile unsigned char observed = ring.data[iteration % kRingBytes];
      (void)observed;
      auto touched = clock_type::now();
      ring.release();
      auto released = clock_type::now();
      allocation.push_back(
          std::chrono::duration<double, std::milli>(allocated - started)
              .count());
      touch.push_back(
          std::chrono::duration<double, std::milli>(touched - allocated)
              .count());
      release.push_back(
          std::chrono::duration<double, std::milli>(released - touched)
              .count());
    }
    sample_summary allocation_summary = summarize(std::move(allocation));
    sample_summary touch_summary = summarize(std::move(touch));
    sample_summary release_summary = summarize(std::move(release));
    std::printf("%s,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f\n", mode_name(mode),
                allocation_summary.median_ms, allocation_summary.p95_ms,
                touch_summary.median_ms, touch_summary.p95_ms,
                release_summary.median_ms, release_summary.p95_ms);
  }
}

struct raw_copy_result {
  double call_us = 0;
  double completion_us = 0;
  double gbps = 0;
};

static raw_copy_result profile_raw_copy(host_ring &ring, CUdeviceptr device,
                                        CUstream stream) {
  constexpr int kIterations = 100;
  CUDA_TRY(cuMemcpyHtoD(device, ring.data, kSlotBytes));
  double call_seconds = 0;
  auto started = clock_type::now();
  for (int iteration = 0; iteration < kIterations; ++iteration) {
    auto call_started = clock_type::now();
    CUDA_TRY(
        cuMemcpyHtoDAsync(device, ring.slot(iteration), kSlotBytes, stream));
    call_seconds +=
        std::chrono::duration<double>(clock_type::now() - call_started).count();
  }
  CUDA_TRY(cuStreamSynchronize(stream));
  double total_seconds =
      std::chrono::duration<double>(clock_type::now() - started).count();
  return {
      call_seconds * 1.0e6 / kIterations, total_seconds * 1.0e6 / kIterations,
      static_cast<double>(kSlotBytes) * kIterations / total_seconds / 1.0e9};
}

static bool profile_graph_replay(host_ring &ring, CUdeviceptr device,
                                 CUstream stream, double *call_us,
                                 double *completion_us,
                                 CUresult *capture_result) {
  *capture_result =
      cuStreamBeginCapture(stream, CU_STREAM_CAPTURE_MODE_RELAXED);
  if (*capture_result != CUDA_SUCCESS) {
    return false;
  }
  *capture_result = cuMemcpyHtoDAsync(device, ring.data, kSlotBytes, stream);
  CUgraph graph = nullptr;
  CUresult end_result = cuStreamEndCapture(stream, &graph);
  if (*capture_result != CUDA_SUCCESS || end_result != CUDA_SUCCESS ||
      graph == nullptr) {
    if (*capture_result == CUDA_SUCCESS) {
      *capture_result = end_result;
    }
    if (graph != nullptr) {
      CUDA_TRY(cuGraphDestroy(graph));
    }
    return false;
  }

  CUgraphExec graph_exec = nullptr;
  *capture_result = cuGraphInstantiate(&graph_exec, graph, 0);
  if (*capture_result != CUDA_SUCCESS) {
    CUDA_TRY(cuGraphDestroy(graph));
    return false;
  }

  std::memset(ring.data, 0x31, kSlotBytes);
  CUDA_TRY(cuGraphLaunch(graph_exec, stream));
  CUDA_TRY(cuStreamSynchronize(stream));
  std::memset(ring.data, 0x72, kSlotBytes);
  CUDA_TRY(cuGraphLaunch(graph_exec, stream));
  CUDA_TRY(cuStreamSynchronize(stream));
  unsigned char observed[2] = {};
  CUDA_TRY(cuMemcpyDtoH(observed, device, 1));
  CUDA_TRY(cuMemcpyDtoH(observed + 1, device + kSlotBytes - 1, 1));
  bool correct = observed[0] == 0x72 && observed[1] == 0x72;

  constexpr int kIterations = 100;
  double call_seconds = 0;
  auto started = clock_type::now();
  for (int iteration = 0; iteration < kIterations; ++iteration) {
    auto call_started = clock_type::now();
    CUDA_TRY(cuGraphLaunch(graph_exec, stream));
    call_seconds +=
        std::chrono::duration<double>(clock_type::now() - call_started).count();
    CUDA_TRY(cuStreamSynchronize(stream));
  }
  *call_us = call_seconds * 1.0e6 / kIterations;
  *completion_us =
      std::chrono::duration<double>(clock_type::now() - started).count() *
      1.0e6 / kIterations;

  CUDA_TRY(cuGraphExecDestroy(graph_exec));
  CUDA_TRY(cuGraphDestroy(graph));
  return correct;
}

struct pipeline_result {
  size_t chunks = 0;
  double seconds = 0;
  double effective_gbps = 0;
  double copy_call_ms = 0;
  double slot_wait_ms = 0;
  bool correct = false;
};

static pipeline_result profile_throttled_pipeline(host_ring &ring,
                                                  CUdeviceptr device,
                                                  CUstream stream,
                                                  double network_gbps) {
  const double network_bytes_per_second = network_gbps * 1.0e9 / 8.0;
  size_t chunks = static_cast<size_t>(
      std::ceil(network_bytes_per_second / static_cast<double>(kSlotBytes)));
  chunks = std::max<size_t>(2, chunks);

  CUevent complete[kSlotCount] = {};
  for (CUevent &event : complete) {
    CUDA_TRY(cuEventCreate(&event, CU_EVENT_DISABLE_TIMING));
  }

  double copy_call_seconds = 0;
  double slot_wait_seconds = 0;
  auto started = clock_type::now();
  for (size_t chunk = 0; chunk < chunks; ++chunk) {
    const size_t slot = chunk % kSlotCount;
    if (chunk >= kSlotCount) {
      auto wait_started = clock_type::now();
      CUDA_TRY(cuEventSynchronize(complete[slot]));
      slot_wait_seconds +=
          std::chrono::duration<double>(clock_type::now() - wait_started)
              .count();
    }

    // Model bytes becoming available from the network independently of CUDA.
    // If a pageable CUDA call stalls this thread, subsequent data is already
    // waiting in the socket and the deadline will be in the past.
    auto delivery =
        started + std::chrono::duration_cast<clock_type::duration>(
                      std::chrono::duration<double>((chunk + 1) * kSlotBytes /
                                                    network_bytes_per_second));
    std::this_thread::sleep_until(delivery);
    unsigned char pattern = static_cast<unsigned char>((chunk % 251) + 1);
    std::memset(ring.slot(slot), pattern, kSlotBytes);

    auto call_started = clock_type::now();
    CUDA_TRY(cuMemcpyHtoDAsync(device + slot * kSlotBytes, ring.slot(slot),
                               kSlotBytes, stream));
    copy_call_seconds +=
        std::chrono::duration<double>(clock_type::now() - call_started).count();
    CUDA_TRY(cuEventRecord(complete[slot], stream));
  }
  CUDA_TRY(cuStreamSynchronize(stream));
  double total_seconds =
      std::chrono::duration<double>(clock_type::now() - started).count();

  bool correct = true;
  for (size_t slot = 0; slot < kSlotCount; ++slot) {
    size_t last_chunk = chunks - 1;
    if (last_chunk % kSlotCount != slot) {
      --last_chunk;
    }
    unsigned char expected = static_cast<unsigned char>((last_chunk % 251) + 1);
    unsigned char observed[2] = {};
    CUDA_TRY(cuMemcpyDtoH(observed, device + slot * kSlotBytes, 1));
    CUDA_TRY(
        cuMemcpyDtoH(observed + 1, device + (slot + 1) * kSlotBytes - 1, 1));
    correct = correct && observed[0] == expected && observed[1] == expected;
  }
  for (CUevent event : complete) {
    CUDA_TRY(cuEventDestroy(event));
  }

  const double logical_bytes = static_cast<double>(chunks) * kSlotBytes;
  return {chunks,
          total_seconds,
          logical_bytes * 8.0 / total_seconds / 1.0e9,
          copy_call_seconds * 1.0e3,
          slot_wait_seconds * 1.0e3,
          correct};
}

struct alignas(64) graph_ring_signal {
  std::atomic<unsigned int> value{0};
};

static_assert(std::atomic<unsigned int>::is_always_lock_free);

struct graph_pipeline_callback_data {
  host_ring *ring = nullptr;
  graph_ring_signal *signals = nullptr;
  size_t chunks = 0;
  double network_bytes_per_second = 0;
};

static void CUDA_CB pump_graph_ring(void *opaque) {
  auto *data = static_cast<graph_pipeline_callback_data *>(opaque);
  auto started = clock_type::now();
  for (size_t chunk = 0; chunk < data->chunks; ++chunk) {
    size_t slot = chunk % kSlotCount;
    while (data->signals[slot].value.load(std::memory_order_acquire) != 0) {
      std::this_thread::yield();
    }
    auto delivery =
        started +
        std::chrono::duration_cast<clock_type::duration>(
            std::chrono::duration<double>((chunk + 1) * kSlotBytes /
                                          data->network_bytes_per_second));
    std::this_thread::sleep_until(delivery);
    unsigned char pattern = static_cast<unsigned char>((chunk % 251) + 1);
    std::memset(data->ring->slot(slot), pattern, kSlotBytes);
    data->signals[slot].value.store(1, std::memory_order_release);
  }
  for (size_t slot = 0; slot < kSlotCount; ++slot) {
    while (data->signals[slot].value.load(std::memory_order_acquire) != 0) {
      std::this_thread::yield();
    }
  }
}

static pipeline_result
profile_graph_pipeline(host_ring &ring, CUdeviceptr device, double network_gbps,
                       double *instantiate_ms, CUresult *capture_result) {
  const double network_bytes_per_second = network_gbps * 1.0e9 / 8.0;
  size_t chunks = static_cast<size_t>(
      std::ceil(network_bytes_per_second / static_cast<double>(kSlotBytes)));
  chunks = std::max<size_t>(2, std::min<size_t>(150, chunks));

  graph_ring_signal *signals = nullptr;
  CUDA_TRY(
      cuMemHostAlloc(reinterpret_cast<void **>(&signals),
                     kSlotCount * sizeof(graph_ring_signal),
                     CU_MEMHOSTALLOC_PORTABLE | CU_MEMHOSTALLOC_DEVICEMAP));
  for (size_t slot = 0; slot < kSlotCount; ++slot) {
    new (&signals[slot]) graph_ring_signal();
  }
  CUdeviceptr device_signals = 0;
  CUDA_TRY(cuMemHostGetDevicePointer(&device_signals, signals, 0));

  CUstream callback_stream = nullptr;
  CUstream transfer_stream = nullptr;
  CUevent fork = nullptr;
  CUevent join = nullptr;
  CUDA_TRY(cuStreamCreate(&callback_stream, CU_STREAM_NON_BLOCKING));
  CUDA_TRY(cuStreamCreate(&transfer_stream, CU_STREAM_NON_BLOCKING));
  CUDA_TRY(cuEventCreate(&fork, CU_EVENT_DISABLE_TIMING));
  CUDA_TRY(cuEventCreate(&join, CU_EVENT_DISABLE_TIMING));

  graph_pipeline_callback_data callback = {&ring, signals, chunks,
                                           network_bytes_per_second};
  *capture_result =
      cuStreamBeginCapture(callback_stream, CU_STREAM_CAPTURE_MODE_RELAXED);
  if (*capture_result == CUDA_SUCCESS) {
    *capture_result = cuEventRecord(fork, callback_stream);
  }
  if (*capture_result == CUDA_SUCCESS) {
    *capture_result = cuStreamWaitEvent(transfer_stream, fork, 0);
  }
  for (size_t chunk = 0; *capture_result == CUDA_SUCCESS && chunk < chunks;
       ++chunk) {
    size_t slot = chunk % kSlotCount;
    *capture_result = cuStreamWaitValue32(
        transfer_stream, device_signals + slot * sizeof(graph_ring_signal), 1,
        CU_STREAM_WAIT_VALUE_EQ);
    if (*capture_result == CUDA_SUCCESS) {
      *capture_result =
          cuMemcpyHtoDAsync(device + slot * kSlotBytes, ring.slot(slot),
                            kSlotBytes, transfer_stream);
    }
    if (*capture_result == CUDA_SUCCESS) {
      *capture_result = cuStreamWriteValue32(
          transfer_stream, device_signals + slot * sizeof(graph_ring_signal), 0,
          0);
    }
  }
  if (*capture_result == CUDA_SUCCESS) {
    *capture_result = cuEventRecord(join, transfer_stream);
  }
  if (*capture_result == CUDA_SUCCESS) {
    *capture_result =
        cuLaunchHostFunc(callback_stream, pump_graph_ring, &callback);
  }
  if (*capture_result == CUDA_SUCCESS) {
    *capture_result = cuStreamWaitEvent(callback_stream, join, 0);
  }

  CUgraph graph = nullptr;
  CUresult end_result = cuStreamEndCapture(callback_stream, &graph);
  if (*capture_result == CUDA_SUCCESS) {
    *capture_result = end_result;
  }
  CUgraphExec graph_exec = nullptr;
  auto instantiate_started = clock_type::now();
  if (*capture_result == CUDA_SUCCESS) {
    *capture_result = cuGraphInstantiate(&graph_exec, graph, 0);
  }
  *instantiate_ms = std::chrono::duration<double, std::milli>(
                        clock_type::now() - instantiate_started)
                        .count();

  pipeline_result result;
  result.chunks = chunks;
  if (*capture_result == CUDA_SUCCESS) {
    std::memset(ring.data, 0, kRingBytes);
    auto started = clock_type::now();
    CUDA_TRY(cuGraphLaunch(graph_exec, callback_stream));
    CUDA_TRY(cuStreamSynchronize(callback_stream));
    result.seconds =
        std::chrono::duration<double>(clock_type::now() - started).count();
    result.effective_gbps =
        static_cast<double>(chunks) * kSlotBytes * 8.0 / result.seconds / 1.0e9;
    result.correct = true;
    for (size_t slot = 0; slot < kSlotCount; ++slot) {
      size_t last_chunk = chunks - 1;
      if (last_chunk % kSlotCount != slot) {
        --last_chunk;
      }
      unsigned char expected =
          static_cast<unsigned char>((last_chunk % 251) + 1);
      unsigned char observed[2] = {};
      CUDA_TRY(cuMemcpyDtoH(observed, device + slot * kSlotBytes, 1));
      CUDA_TRY(
          cuMemcpyDtoH(observed + 1, device + (slot + 1) * kSlotBytes - 1, 1));
      result.correct =
          result.correct && observed[0] == expected && observed[1] == expected;
    }
  }

  if (graph_exec != nullptr) {
    CUDA_TRY(cuGraphExecDestroy(graph_exec));
  }
  if (graph != nullptr) {
    CUDA_TRY(cuGraphDestroy(graph));
  }
  CUDA_TRY(cuEventDestroy(join));
  CUDA_TRY(cuEventDestroy(fork));
  CUDA_TRY(cuStreamDestroy(transfer_stream));
  CUDA_TRY(cuStreamDestroy(callback_stream));
  CUDA_TRY(cuMemFreeHost(signals));
  return result;
}

struct serial_graph_state {
  host_ring *ring = nullptr;
  clock_type::time_point started;
  double network_bytes_per_second = 0;
};

struct serial_graph_callback_data {
  serial_graph_state *state = nullptr;
  size_t chunk = 0;
};

static void CUDA_CB fill_serial_graph_slot(void *opaque) {
  auto *data = static_cast<serial_graph_callback_data *>(opaque);
  if (data->chunk == 0) {
    data->state->started = clock_type::now();
  }
  auto delivery =
      data->state->started +
      std::chrono::duration_cast<clock_type::duration>(
          std::chrono::duration<double>((data->chunk + 1) * kSlotBytes /
                                        data->state->network_bytes_per_second));
  std::this_thread::sleep_until(delivery);
  unsigned char pattern = static_cast<unsigned char>((data->chunk % 251) + 1);
  std::memset(data->state->ring->slot(0), pattern, kSlotBytes);
}

static pipeline_result profile_serial_graph_pipeline(
    host_ring &ring, CUdeviceptr device, double network_gbps,
    double *instantiate_ms, double *launch_call_ms, CUresult *capture_result) {
  const double network_bytes_per_second = network_gbps * 1.0e9 / 8.0;
  size_t chunks = static_cast<size_t>(
      std::ceil(network_bytes_per_second / static_cast<double>(kSlotBytes)));
  chunks = std::max<size_t>(2, std::min<size_t>(150, chunks));

  CUstream stream = nullptr;
  CUDA_TRY(cuStreamCreate(&stream, CU_STREAM_NON_BLOCKING));
  serial_graph_state state = {&ring, {}, network_bytes_per_second};
  std::vector<serial_graph_callback_data> callbacks(chunks);

  *capture_result =
      cuStreamBeginCapture(stream, CU_STREAM_CAPTURE_MODE_RELAXED);
  for (size_t chunk = 0; *capture_result == CUDA_SUCCESS && chunk < chunks;
       ++chunk) {
    callbacks[chunk] = {&state, chunk};
    *capture_result =
        cuLaunchHostFunc(stream, fill_serial_graph_slot, &callbacks[chunk]);
    if (*capture_result == CUDA_SUCCESS) {
      *capture_result =
          cuMemcpyHtoDAsync(device, ring.slot(0), kSlotBytes, stream);
    }
  }
  CUgraph graph = nullptr;
  CUresult end_result = cuStreamEndCapture(stream, &graph);
  if (*capture_result == CUDA_SUCCESS) {
    *capture_result = end_result;
  }
  CUgraphExec graph_exec = nullptr;
  auto instantiate_started = clock_type::now();
  if (*capture_result == CUDA_SUCCESS) {
    *capture_result = cuGraphInstantiate(&graph_exec, graph, 0);
  }
  *instantiate_ms = std::chrono::duration<double, std::milli>(
                        clock_type::now() - instantiate_started)
                        .count();

  pipeline_result result;
  result.chunks = chunks;
  *launch_call_ms = 0;
  if (*capture_result == CUDA_SUCCESS) {
    std::memset(ring.data, 0, kRingBytes);
    auto started = clock_type::now();
    auto launch_started = clock_type::now();
    CUDA_TRY(cuGraphLaunch(graph_exec, stream));
    *launch_call_ms = std::chrono::duration<double, std::milli>(
                          clock_type::now() - launch_started)
                          .count();
    CUDA_TRY(cuStreamSynchronize(stream));
    result.seconds =
        std::chrono::duration<double>(clock_type::now() - started).count();
    result.effective_gbps =
        static_cast<double>(chunks) * kSlotBytes * 8.0 / result.seconds / 1.0e9;
    unsigned char expected =
        static_cast<unsigned char>(((chunks - 1) % 251) + 1);
    unsigned char observed[2] = {};
    CUDA_TRY(cuMemcpyDtoH(observed, device, 1));
    CUDA_TRY(cuMemcpyDtoH(observed + 1, device + kSlotBytes - 1, 1));
    result.correct = observed[0] == expected && observed[1] == expected;
  }

  if (graph_exec != nullptr) {
    CUDA_TRY(cuGraphExecDestroy(graph_exec));
  }
  if (graph != nullptr) {
    CUDA_TRY(cuGraphDestroy(graph));
  }
  CUDA_TRY(cuStreamDestroy(stream));
  return result;
}

int main() {
  std::setvbuf(stdout, nullptr, _IOLBF, 0);
  CUDA_TRY(cuInit(0));
  const char *device_filter = std::getenv("HTOD_RING_DEVICE");
  int device_ordinal = device_filter == nullptr ? 0 : std::atoi(device_filter);
  CUdevice device = 0;
  CUDA_TRY(cuDeviceGet(&device, device_ordinal));
  char device_name[256] = {};
  CUDA_TRY(cuDeviceGetName(device_name, sizeof(device_name), device));
  CUcontext context = nullptr;
#if CUDA_VERSION >= 13000
  CUDA_TRY(cuCtxCreate(&context, nullptr, 0, device));
#else
  CUDA_TRY(cuCtxCreate(&context, 0, device));
#endif
  CUstream stream = nullptr;
  CUDA_TRY(cuStreamCreate(&stream, CU_STREAM_NON_BLOCKING));
  CUdeviceptr device_ring = 0;
  CUDA_TRY(cuMemAlloc(&device_ring, kRingBytes));

  std::printf("device,%d,%s\n", device_ordinal, device_name);
  std::printf("ring_slots,%zu\nslot_bytes,%zu\nring_bytes,%zu\n", kSlotCount,
              kSlotBytes, kRingBytes);
  const bool graph_only = std::getenv("HTOD_RING_GRAPH_ONLY") != nullptr;
  const double network_rates[] = {0.1, 1.0, 10.0, 100.0};
  if (!graph_only) {
    profile_initialization();

    std::printf("\ncopy_mode,async_call_us,completion_us,effective_GBps,"
                "graph_capture,graph_launch_call_us,graph_completion_us,"
                "graph_replay_correct\n");
    for (host_mode mode :
         {host_mode::pageable, host_mode::registered, host_mode::pinned}) {
      host_ring ring;
      CUDA_TRY(ring.allocate(mode));
      std::memset(ring.data, 0x5a, kRingBytes);
      raw_copy_result raw = profile_raw_copy(ring, device_ring, stream);
      double graph_call_us = 0;
      double graph_completion_us = 0;
      CUresult graph_result = CUDA_SUCCESS;
      bool graph_correct =
          profile_graph_replay(ring, device_ring, stream, &graph_call_us,
                               &graph_completion_us, &graph_result);
      std::printf("%s,%.3f,%.3f,%.3f,%s,%.3f,%.3f,%s\n", mode_name(mode),
                  raw.call_us, raw.completion_us, raw.gbps,
                  cuda_name(graph_result), graph_call_us, graph_completion_us,
                  graph_correct ? "yes" : "no");
      ring.release();
    }

    std::printf("\nnetwork_Gbps,ring_mode,chunks,seconds,effective_Gbps,"
                "copy_call_ms,slot_wait_ms,correct\n");
    for (double rate : network_rates) {
      for (host_mode mode :
           {host_mode::pageable, host_mode::registered, host_mode::pinned}) {
        host_ring ring;
        CUDA_TRY(ring.allocate(mode));
        pipeline_result result =
            profile_throttled_pipeline(ring, device_ring, stream, rate);
        std::printf("%.1f,%s,%zu,%.6f,%.4f,%.3f,%.3f,%s\n", rate,
                    mode_name(mode), result.chunks, result.seconds,
                    result.effective_gbps, result.copy_call_ms,
                    result.slot_wait_ms, result.correct ? "yes" : "no");
        ring.release();
      }
    }
  }

  std::printf("\ngraph_network_Gbps,ring_mode,chunks,instantiate_ms,seconds,"
              "effective_Gbps,capture,correct\n");
  const char *graph_mode_filter = std::getenv("HTOD_RING_GRAPH_MODE");
  const char *graph_rate_filter = std::getenv("HTOD_RING_GRAPH_RATE");
  for (double rate : network_rates) {
    if (graph_rate_filter != nullptr &&
        rate != std::strtod(graph_rate_filter, nullptr)) {
      continue;
    }
    for (host_mode mode :
         {host_mode::pinned, host_mode::registered, host_mode::pageable}) {
      if (graph_mode_filter != nullptr &&
          std::strcmp(graph_mode_filter, mode_name(mode)) != 0) {
        continue;
      }
      if (mode == host_mode::pageable &&
          std::getenv("HTOD_RING_TRY_OVERLAPPED_PAGEABLE") == nullptr) {
        continue;
      }
      host_ring ring;
      CUDA_TRY(ring.allocate(mode));
      double instantiate_ms = 0;
      CUresult graph_result = CUDA_SUCCESS;
      pipeline_result result = profile_graph_pipeline(
          ring, device_ring, rate, &instantiate_ms, &graph_result);
      std::printf("%.1f,%s,%zu,%.3f,%.6f,%.4f,%s,%s\n", rate, mode_name(mode),
                  result.chunks, instantiate_ms, result.seconds,
                  result.effective_gbps, cuda_name(graph_result),
                  result.correct ? "yes" : "no");
      ring.release();
    }
  }

  std::printf("\nserial_graph_network_Gbps,ring_mode,chunks,instantiate_ms,"
              "launch_call_ms,seconds,effective_Gbps,capture,correct\n");
  for (double rate : network_rates) {
    if (graph_rate_filter != nullptr &&
        rate != std::strtod(graph_rate_filter, nullptr)) {
      continue;
    }
    for (host_mode mode :
         {host_mode::pageable, host_mode::registered, host_mode::pinned}) {
      if (graph_mode_filter != nullptr &&
          std::strcmp(graph_mode_filter, mode_name(mode)) != 0) {
        continue;
      }
      host_ring ring;
      CUDA_TRY(ring.allocate(mode));
      double instantiate_ms = 0;
      double launch_call_ms = 0;
      CUresult graph_result = CUDA_SUCCESS;
      pipeline_result result = profile_serial_graph_pipeline(
          ring, device_ring, rate, &instantiate_ms, &launch_call_ms,
          &graph_result);
      std::printf("%.1f,%s,%zu,%.3f,%.3f,%.6f,%.4f,%s,%s\n", rate,
                  mode_name(mode), result.chunks, instantiate_ms,
                  launch_call_ms, result.seconds, result.effective_gbps,
                  cuda_name(graph_result), result.correct ? "yes" : "no");
      ring.release();
    }
  }

  CUDA_TRY(cuMemFree(device_ring));
  CUDA_TRY(cuStreamDestroy(stream));
  CUDA_TRY(cuCtxDestroy(context));
  return 0;
}
