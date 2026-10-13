// Exercises the generated RPC fragments over an in-memory transport. The mock
// vendor deliberately writes before failing to test caller-buffer preservation.
#include <nvml.h>
#undef nvmlDeviceGetComputeRunningProcesses
#undef nvmlDeviceGetGraphicsRunningProcesses
#undef nvmlDeviceGetMPSComputeRunningProcesses
#include "codegen/gen_rpc_ids.h"
#include <cassert>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <type_traits>
#include <vector>

struct conn_t {
  std::vector<unsigned char> request, response;
  size_t cursor = 0;
  bool server = false, closed = false;
  int op = 0;
} transport;
static nvmlReturn_t native_status = NVML_SUCCESS;
static unsigned int native_count = 2;
static bool missing_symbol = false;
static int corrupt_response = 0;
static int calls = 0;
static size_t element_size = 0;

int rpc_write(conn_t *conn, const void *data, size_t size) {
  if (!size)
    return 0;
  auto &bytes = conn->server ? conn->response : conn->request;
  auto *begin = static_cast<const unsigned char *>(data);
  bytes.insert(bytes.end(), begin, begin + size);
  return 0;
}
int rpc_read(conn_t *conn, void *data, size_t size) {
  const auto &bytes = conn->server ? conn->request : conn->response;
  if (size > bytes.size() - conn->cursor) {
    conn->closed = true;
    return -1;
  }
  if (size)
    std::memcpy(data, bytes.data() + conn->cursor, size);
  conn->cursor += size;
  return 0;
}
int rpc_read_end(conn_t *conn) {
  const auto &bytes = conn->server ? conn->request : conn->response;
  if (conn->cursor != bytes.size()) {
    conn->closed = true;
    return -1;
  }
  return 0;
}
int rpc_write_start_request(conn_t *conn, int op) {
  *conn = {};
  conn->op = op;
  ++calls;
  return 0;
}
int rpc_write_start_response(conn_t *conn, int) {
  assert(conn->cursor == conn->request.size());
  return 0;
}
int rpc_write_end(conn_t *) { return 0; }
void rpc_close_transport_socket(conn_t *conn) { conn->closed = true; }
nvmlReturn_t rpc_error() { return NVML_ERROR_UNKNOWN; }
nvmlReturn_t function_not_found() { return NVML_ERROR_FUNCTION_NOT_FOUND; }
conn_t *connection_for_device(nvmlDevice_t *) { return &transport; }
int rpc_wait_for_response(conn_t *);

template <typename Sample>
nvmlReturn_t mock_processes(nvmlDevice_t, unsigned int *count,
                            Sample *samples) {
  unsigned int capacity = *count;
  if (samples && capacity) {
    // Fill only allocated entries, even when reporting an inconsistent count.
    for (unsigned int i = 0; i < capacity && i < native_count; ++i) {
      std::memset(&samples[i], 0x42, sizeof(Sample));
      samples[i].pid = 123 + i;
    }
  }
  *count = native_count;
  return native_status;
}
nvmlReturn_t mock_utilization(nvmlDevice_t device,
                              nvmlProcessUtilizationSample_t *samples,
                              unsigned int *count,
                              unsigned long long timestamp) {
  assert(timestamp == 456);
  return mock_processes(device, count, samples);
}
template <typename Fn> Fn nvml_symbol(const char *name) {
  if (missing_symbol)
    return nullptr;
  if constexpr (std::is_same_v<Fn, decltype(&mock_utilization)>) {
    assert(std::strcmp(name, "nvmlDeviceGetProcessUtilization") == 0);
    return mock_utilization;
  } else if constexpr (std::is_same_v<Fn, decltype(&mock_processes<
                                                   nvmlProcessInfo_v2_t>)>) {
    assert(std::strstr(name, "_v2"));
    return mock_processes<nvmlProcessInfo_v2_t>;
  } else {
    assert(!std::strstr(name, "_v2"));
    return mock_processes<nvmlProcessInfo_v1_t>;
  }
}
#include "client.inc"
#include "server.inc"

int rpc_wait_for_response(conn_t *conn) {
  conn->server = true;
  conn->cursor = 0;
  int result = -1;
#define DISPATCH(name)                                                         \
  case RPC_##name:                                                             \
    result = handle_##name(conn);                                              \
    break
  switch (conn->op) {
    DISPATCH(nvmlDeviceGetComputeRunningProcesses);
    DISPATCH(nvmlDeviceGetComputeRunningProcesses_v2);
    DISPATCH(nvmlDeviceGetGraphicsRunningProcesses);
    DISPATCH(nvmlDeviceGetGraphicsRunningProcesses_v2);
    DISPATCH(nvmlDeviceGetMPSComputeRunningProcesses);
    DISPATCH(nvmlDeviceGetMPSComputeRunningProcesses_v2);
    DISPATCH(nvmlDeviceGetProcessUtilization);
  }
#undef DISPATCH
  conn->server = false;
  conn->cursor = 0;
  // Responses are count, presence, optional samples, then status.
  if (corrupt_response == 1)
    conn->response[sizeof(unsigned int)] = 2;
  if (corrupt_response == 2) {
    unsigned int count = 100;
    std::memcpy(conn->response.data(), &count, sizeof(count));
  }
  if (corrupt_response == 3)
    conn->response.resize(sizeof(unsigned int) + 1 + element_size - 1);
  if (corrupt_response == 4)
    conn->response[sizeof(unsigned int)] = 1;
  return result;
}

template <typename Sample, typename Call> void exercise(Call call) {
  element_size = sizeof(Sample);
  std::vector<Sample> samples(3);
  auto reset = [&] {
    std::memset(samples.data(), 0xa5, samples.size() * sizeof(Sample));
    native_status = NVML_SUCCESS;
    native_count = 2;
    missing_symbol = false;
    corrupt_response = 0;
  };
  auto unchanged = [&] {
    for (auto byte : std::vector<unsigned char>(
             reinterpret_cast<unsigned char *>(samples.data()),
             reinterpret_cast<unsigned char *>(samples.data() +
                                               samples.size())))
      assert(byte == 0xa5);
  };
  reset();
  int previous_calls = calls;
  assert(call(nullptr, samples.data()) == NVML_ERROR_INVALID_ARGUMENT);
  assert(calls == previous_calls);
  unsigned int count = 3;
  assert(call(&count, samples.data()) == NVML_SUCCESS);
  assert(count == 2 && samples[0].pid == 123 && samples[1].pid == 124);
  for (size_t i = 2 * sizeof(Sample); i < 3 * sizeof(Sample); ++i)
    assert(reinterpret_cast<unsigned char *>(samples.data())[i] == 0xa5);
  assert(transport.response.size() ==
         sizeof(count) + 1 + 2 * sizeof(Sample) + sizeof(nvmlReturn_t));
  for (auto error : {NVML_ERROR_INSUFFICIENT_SIZE, NVML_ERROR_NOT_FOUND,
                     NVML_ERROR_NO_PERMISSION, NVML_ERROR_UNKNOWN}) {
    reset();
    native_status = error;
    native_count = 5;
    count = 1;
    assert(call(&count, samples.data()) == error);
    assert(count == 5);
    unchanged();
    assert(transport.response.size() ==
           sizeof(count) + 1 + sizeof(nvmlReturn_t));
  }
  reset();
  native_status = NVML_ERROR_INSUFFICIENT_SIZE;
  count = 99;
  assert(call(&count, nullptr) == NVML_ERROR_INSUFFICIENT_SIZE && count == 2);
  reset();
  native_count = 0;
  count = 99;
  assert(call(&count, nullptr) == NVML_SUCCESS && count == 0);
  reset();
  native_count = 0;
  count = 3;
  assert(call(&count, samples.data()) == NVML_SUCCESS && count == 0);
  unchanged();
  reset();
  count = 0;
  assert(call(&count, samples.data()) == NVML_ERROR_UNKNOWN && count == 2);
  unchanged();
  reset();
  native_count = 5;
  count = 1;
  assert(call(&count, samples.data()) == NVML_ERROR_UNKNOWN && count == 5);
  unchanged();
  reset();
  missing_symbol = true;
  count = 3;
  assert(call(&count, samples.data()) == NVML_ERROR_FUNCTION_NOT_FOUND);
  unchanged();
  for (int corruption : {1, 2, 3}) {
    reset();
    corrupt_response = corruption;
    count = 3;
    assert(call(&count, samples.data()) == NVML_ERROR_UNKNOWN);
    assert(transport.closed);
    unchanged();
  }
  reset();
  corrupt_response = 4;
  native_status = NVML_ERROR_NOT_FOUND;
  count = 0;
  assert(call(&count, nullptr) == NVML_ERROR_UNKNOWN && transport.closed);
  reset();
  count = 3;
  assert(call(&count, samples.data()) == NVML_SUCCESS);
}
int main() {
#define TEST(name, type)                                                       \
  exercise<type>([](unsigned int *count, type *samples) {                      \
    return name(reinterpret_cast<nvmlDevice_t>(uintptr_t(1)), count, samples); \
  })
  TEST(nvmlDeviceGetComputeRunningProcesses, nvmlProcessInfo_v1_t);
  TEST(nvmlDeviceGetComputeRunningProcesses_v2, nvmlProcessInfo_v2_t);
  TEST(nvmlDeviceGetGraphicsRunningProcesses, nvmlProcessInfo_v1_t);
  TEST(nvmlDeviceGetGraphicsRunningProcesses_v2, nvmlProcessInfo_v2_t);
  TEST(nvmlDeviceGetMPSComputeRunningProcesses, nvmlProcessInfo_v1_t);
  TEST(nvmlDeviceGetMPSComputeRunningProcesses_v2, nvmlProcessInfo_v2_t);
#undef TEST
  exercise<nvmlProcessUtilizationSample_t>(
      [](unsigned int *count, nvmlProcessUtilizationSample_t *samples) {
        return nvmlDeviceGetProcessUtilization(
            reinterpret_cast<nvmlDevice_t>(uintptr_t(1)), samples, count, 456);
      });
  std::cout << "7 generated NVML APIs: success, error, query, ABI and "
               "malformed-response cases passed\n";
}
