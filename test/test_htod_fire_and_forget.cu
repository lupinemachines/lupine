// A small synchronous host-to-device copy may reach the device after its call
// returns, but what the call promises must still hold: the source is free to
// reuse, and anything that reads the destination sees the bytes. Each case
// writes with cudaMemcpy and then reads them back through a path that never
// issued the write -- a kernel, a device-to-host copy, a device-to-device copy,
// a second host thread, and a replayed graph -- or overwrites the source the
// instant the call returns.
// This file is auto-discovered by test/run_custom_tests.sh.
#include <cuda_runtime.h>

#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <thread>
#include <vector>

namespace {

constexpr size_t kChunkBytes = 1024;
constexpr size_t kChunks = 4096; // more than the transport buffers at once

__global__ void accumulate(const unsigned char *data, size_t bytes,
                           unsigned long long *sum) {
  unsigned long long total = 0;
  for (size_t i = 0; i < bytes; ++i) {
    total += data[i];
  }
  *sum = total;
}

void check(cudaError_t result, const char *operation) {
  if (result == cudaSuccess) {
    return;
  }
  std::fprintf(stderr, "FAIL: %s: %s\n", operation, cudaGetErrorString(result));
  std::exit(EXIT_FAILURE);
}

void fill(std::vector<unsigned char> &buffer, unsigned char seed) {
  for (size_t i = 0; i < buffer.size(); ++i) {
    buffer[i] = static_cast<unsigned char>(seed + i * 31 + (i >> 9));
  }
}

void require_equal(const char *label, const std::vector<unsigned char> &actual,
                   const std::vector<unsigned char> &expected) {
  if (actual.size() != expected.size()) {
    std::fprintf(stderr, "FAIL: %s: %zu bytes, expected %zu\n", label,
                 actual.size(), expected.size());
    std::exit(EXIT_FAILURE);
  }
  for (size_t i = 0; i < actual.size(); ++i) {
    if (actual[i] != expected[i]) {
      std::fprintf(stderr, "FAIL: %s: byte %zu is 0x%02x, expected 0x%02x\n",
                   label, i, actual[i], expected[i]);
      std::exit(EXIT_FAILURE);
    }
  }
  std::printf("%s: ok\n", label);
}

unsigned long long sum_of(const std::vector<unsigned char> &buffer) {
  unsigned long long total = 0;
  for (unsigned char byte : buffer) {
    total += byte;
  }
  return total;
}

// The contract the call actually makes about the source: once it returns, the
// caller owns those bytes again. Overwriting them immediately must not reach
// the device.
void test_source_is_free_on_return(unsigned char *device) {
  std::vector<unsigned char> pattern(kChunkBytes);
  std::vector<unsigned char> source(kChunkBytes);
  std::vector<unsigned char> back(kChunkBytes);
  fill(pattern, 5);
  source = pattern;
  check(cudaMemset(device, 0, kChunkBytes), "cudaMemset");
  check(cudaMemcpy(device, source.data(), kChunkBytes, cudaMemcpyHostToDevice),
        "cudaMemcpy small HtoD");
  std::memset(source.data(), 0x11, source.size());
  check(cudaMemcpy(back.data(), device, kChunkBytes, cudaMemcpyDeviceToHost),
        "cudaMemcpy DtoH");
  require_equal("source overwritten right after the call returns", back,
                pattern);
}

// A kernel reading the destination must see the copy even though no other CUDA
// call came between the two.
void test_kernel_reads_the_write(unsigned char *device,
                                 unsigned long long *device_sum) {
  std::vector<unsigned char> pattern(kChunkBytes);
  fill(pattern, 3);
  check(cudaMemset(device, 0, kChunkBytes), "cudaMemset");
  check(cudaMemcpy(device, pattern.data(), kChunkBytes,
                   cudaMemcpyHostToDevice),
        "cudaMemcpy small HtoD");
  accumulate<<<1, 1>>>(device, kChunkBytes, device_sum);
  unsigned long long observed = 0;
  check(cudaMemcpy(&observed, device_sum, sizeof(observed),
                   cudaMemcpyDeviceToHost),
        "cudaMemcpy sum DtoH");
  unsigned long long expected = sum_of(pattern);
  if (observed != expected) {
    std::fprintf(stderr, "FAIL: kernel read %llu, expected %llu\n", observed,
                 expected);
    std::exit(EXIT_FAILURE);
  }
  std::printf("kernel reads the write: ok\n");
}

// The copy read back over the wire, and again after a device-to-device copy
// that never touches the host.
void test_copies_read_the_write(unsigned char *device, unsigned char *other) {
  std::vector<unsigned char> pattern(kChunkBytes);
  std::vector<unsigned char> back(kChunkBytes);
  fill(pattern, 11);
  check(cudaMemset(device, 0, kChunkBytes), "cudaMemset");
  check(cudaMemcpy(device, pattern.data(), kChunkBytes,
                   cudaMemcpyHostToDevice),
        "cudaMemcpy small HtoD");
  check(cudaMemcpy(back.data(), device, kChunkBytes, cudaMemcpyDeviceToHost),
        "cudaMemcpy DtoH");
  require_equal("device-to-host reads the write", back, pattern);

  fill(pattern, 23);
  check(cudaMemset(other, 0, kChunkBytes), "cudaMemset other");
  check(cudaMemcpy(device, pattern.data(), kChunkBytes,
                   cudaMemcpyHostToDevice),
        "cudaMemcpy small HtoD");
  check(cudaMemcpy(other, device, kChunkBytes, cudaMemcpyDeviceToDevice),
        "cudaMemcpy DtoD");
  std::memset(back.data(), 0, back.size());
  check(cudaMemcpy(back.data(), other, kChunkBytes, cudaMemcpyDeviceToHost),
        "cudaMemcpy DtoH after DtoD");
  require_equal("device-to-device reads the write", back, pattern);
}

// The writer returns, hands the destination to another thread through plain
// host synchronization, and that thread reads it with no CUDA call of its own
// in between. Nothing orders an unwaited copy against another thread's
// synchronous read, so a second thread issuing CUDA work is what takes this
// copy off the unwaited path.
void test_other_thread_reads_the_write(unsigned char *device) {
  std::vector<unsigned char> pattern(kChunkBytes);
  std::vector<unsigned char> back(kChunkBytes);
  fill(pattern, 37);
  check(cudaMemset(device, 0, kChunkBytes), "cudaMemset");

  std::atomic<int> phase{0};
  std::thread writer([&] {
    // Bind this thread to the device before the copy. A thread whose first
    // CUDA call is a copy fails with cudaErrorDeviceUninitialized on main
    // today, which is a separate bug from what this case is about.
    check(cudaSetDevice(0), "cudaSetDevice on writer thread");
    check(cudaMemcpy(device, pattern.data(), kChunkBytes,
                     cudaMemcpyHostToDevice),
          "cudaMemcpy small HtoD on writer thread");
    phase.store(1, std::memory_order_release);
  });
  while (phase.load(std::memory_order_acquire) != 1) {
    std::this_thread::yield();
  }
  check(cudaMemcpy(back.data(), device, kChunkBytes, cudaMemcpyDeviceToHost),
        "cudaMemcpy DtoH on reader thread");
  writer.join();
  require_equal("another thread reads the write", back, pattern);
}

// A replayed graph reads its input at launch. The write that fills that input
// is a small copy the caller did not wait for, so the launch has to be ordered
// behind it.
void test_captured_graph_reads_the_write(unsigned char *device,
                                         unsigned long long *device_sum) {
  cudaStream_t stream = nullptr;
  check(cudaStreamCreate(&stream), "cudaStreamCreate");
  cudaGraph_t graph = nullptr;
  cudaGraphExec_t executable = nullptr;
  check(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal),
        "cudaStreamBeginCapture");
  accumulate<<<1, 1, 0, stream>>>(device, kChunkBytes, device_sum);
  check(cudaStreamEndCapture(stream, &graph), "cudaStreamEndCapture");
  check(cudaGraphInstantiate(&executable, graph, nullptr, nullptr, 0),
        "cudaGraphInstantiate");

  std::vector<unsigned char> pattern(kChunkBytes);
  const unsigned char seeds[] = {41, 97};
  for (unsigned char seed : seeds) {
    fill(pattern, seed);
    check(cudaMemcpy(device, pattern.data(), kChunkBytes,
                     cudaMemcpyHostToDevice),
          "cudaMemcpy small HtoD before replay");
    check(cudaGraphLaunch(executable, stream), "cudaGraphLaunch");
    check(cudaStreamSynchronize(stream), "cudaStreamSynchronize");
    unsigned long long observed = 0;
    check(cudaMemcpy(&observed, device_sum, sizeof(observed),
                     cudaMemcpyDeviceToHost),
          "cudaMemcpy sum DtoH");
    unsigned long long expected = sum_of(pattern);
    if (observed != expected) {
      std::fprintf(stderr, "FAIL: replay read %llu, expected %llu\n", observed,
                   expected);
      std::exit(EXIT_FAILURE);
    }
  }
  check(cudaGraphExecDestroy(executable), "cudaGraphExecDestroy");
  check(cudaGraphDestroy(graph), "cudaGraphDestroy");
  check(cudaStreamDestroy(stream), "cudaStreamDestroy");
  std::printf("replayed graph reads the write: ok\n");
}

// The shape issue #832 is about, at enough volume to outrun what the transport
// buffers: every byte still has to land where it was sent.
void test_upload_loop(unsigned char *first, unsigned char *second) {
  std::vector<unsigned char> first_expected(kChunks * kChunkBytes);
  std::vector<unsigned char> second_expected(kChunks * kChunkBytes);
  std::vector<unsigned char> chunk(kChunkBytes);
  check(cudaMemset(first, 0, first_expected.size()), "cudaMemset first");
  check(cudaMemset(second, 0, second_expected.size()), "cudaMemset second");
  for (size_t i = 0; i < kChunks; ++i) {
    fill(chunk, static_cast<unsigned char>(i));
    std::memcpy(first_expected.data() + i * kChunkBytes, chunk.data(),
                kChunkBytes);
    check(cudaMemcpy(first + i * kChunkBytes, chunk.data(), kChunkBytes,
                     cudaMemcpyHostToDevice),
          "cudaMemcpy first run");
    fill(chunk, static_cast<unsigned char>(i + 128));
    std::memcpy(second_expected.data() + i * kChunkBytes, chunk.data(),
                kChunkBytes);
    check(cudaMemcpy(second + i * kChunkBytes, chunk.data(), kChunkBytes,
                     cudaMemcpyHostToDevice),
          "cudaMemcpy second run");
  }
  std::vector<unsigned char> back(first_expected.size());
  check(cudaMemcpy(back.data(), first, back.size(), cudaMemcpyDeviceToHost),
        "cudaMemcpy first run DtoH");
  require_equal("upload loop, first buffer", back, first_expected);
  check(cudaMemcpy(back.data(), second, back.size(), cudaMemcpyDeviceToHost),
        "cudaMemcpy second run DtoH");
  require_equal("upload loop, second buffer", back, second_expected);
}

// Two writes to the same bytes with nothing between them: the later one wins,
// whatever the client did with the earlier one.
void test_overlapping_writes(unsigned char *device) {
  std::vector<unsigned char> first(2 * kChunkBytes);
  std::vector<unsigned char> second(kChunkBytes);
  std::vector<unsigned char> back(2 * kChunkBytes);
  fill(first, 53);
  fill(second, 97);
  check(cudaMemset(device, 0, first.size()), "cudaMemset");
  check(cudaMemcpy(device, first.data(), first.size(), cudaMemcpyHostToDevice),
        "cudaMemcpy overlapping first");
  check(cudaMemcpy(device + kChunkBytes / 2, second.data(), second.size(),
                   cudaMemcpyHostToDevice),
        "cudaMemcpy overlapping second");
  check(cudaMemcpy(back.data(), device, back.size(), cudaMemcpyDeviceToHost),
        "cudaMemcpy overlapping DtoH");
  std::vector<unsigned char> expected = first;
  std::memcpy(expected.data() + kChunkBytes / 2, second.data(), second.size());
  require_equal("later write to the same bytes wins", back, expected);
}

} // namespace

int main() {
  unsigned char *first = nullptr;
  unsigned char *second = nullptr;
  unsigned long long *device_sum = nullptr;
  check(cudaMalloc(&first, kChunks * kChunkBytes), "cudaMalloc first");
  check(cudaMalloc(&second, kChunks * kChunkBytes), "cudaMalloc second");
  check(cudaMalloc(&device_sum, sizeof(*device_sum)), "cudaMalloc sum");

  test_source_is_free_on_return(first);
  test_kernel_reads_the_write(first, device_sum);
  test_copies_read_the_write(first, second);
  test_other_thread_reads_the_write(first);
  test_captured_graph_reads_the_write(first, device_sum);
  test_upload_loop(first, second);
  test_overlapping_writes(first);

  check(cudaFree(first), "cudaFree first");
  check(cudaFree(second), "cudaFree second");
  check(cudaFree(device_sum), "cudaFree sum");
  std::printf("PASS: unwaited host-to-device copies keep every promise the "
              "call makes\n");
  return 0;
}
