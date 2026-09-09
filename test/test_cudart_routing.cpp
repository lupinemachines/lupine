// Exercise the shared routing table without a GPU. Only the vendor runtime's
// device-count RPC may initialize it; driver-only discovery must not do so.
#undef NDEBUG
#include <cassert>
#include <cstring>
#include <vector>

#include "client_routing.h"
#include "codegen/gen_rpc_ids.h"

static conn_t servers[2];
static bool initialized[2];
static bool fail_second_server = true;
static int requests, operation, ordinal;
static std::vector<unsigned char> response;
static size_t read_offset;

int rpc_open() { return 0; }
int rpc_size() { return 2; }
conn_t *rpc_client_get_connection(unsigned int index) {
  return index < 2 ? &servers[index] : nullptr;
}
bool lupine_local_cuda_available() { return false; }
extern "C" void *lupine_real_cuda_symbol(const char *) {
  assert(false && "remote routing called a local driver");
  return nullptr;
}
extern "C" int lupine_prepare_rpc(conn_t *) { return 0; }

int rpc_write_start_request(conn_t *, int op) {
  assert(op == RPC_cuDeviceGetCount || op == RPC_cudaGetDeviceCount ||
         op == RPC_cuDeviceGet); // In particular, never inject cuInit.
  operation = op;
  ++requests;
  response.clear();
  read_offset = 0;
  return 0;
}
int rpc_write(conn_t *, const void *data, size_t size) {
  assert(operation == RPC_cuDeviceGet && size == sizeof(ordinal));
  std::memcpy(&ordinal, data, size);
  return 0;
}
int rpc_wait_for_response(conn_t *conn) {
  int server = conn->logical_index;
  CUresult result = CUDA_SUCCESS;
  int value = 0;
  if (operation == RPC_cudaGetDeviceCount) {
    if (server == 1 && fail_second_server)
      result = CUDA_ERROR_DEVICE_UNAVAILABLE;
    else
      initialized[server] = true;
  }
  if (result == CUDA_SUCCESS && !initialized[server])
    result = CUDA_ERROR_NOT_INITIALIZED;
  if (result == CUDA_SUCCESS)
    value = operation == RPC_cuDeviceGet ? ordinal : (server == 0 ? 2 : 1);
  response.resize(sizeof(value) + sizeof(result));
  std::memcpy(response.data(), &value, sizeof(value));
  std::memcpy(response.data() + sizeof(value), &result, sizeof(result));
  return 0;
}
int rpc_read(conn_t *, void *data, size_t size) {
  assert(read_offset + size <= response.size());
  std::memcpy(data, response.data() + read_offset, size);
  read_offset += size;
  return 0;
}
int rpc_read_end(conn_t *) {
  assert(read_offset == response.size());
  return 0;
}

int main() {
  servers[0].logical_index = 0;
  servers[1].logical_index = 1;
  int count = -1;
  assert(lupine_virtual_device_count(nullptr, true) ==
         CUDA_ERROR_INVALID_VALUE);
  assert(requests == 0);
  assert(lupine_virtual_device_count(&count) == CUDA_ERROR_NOT_INITIALIZED);
  assert(!initialized[0] && !initialized[1]);

  // A partial discovery must not publish a truncated table or poison a retry.
  assert(lupine_virtual_device_count(&count, true) ==
         CUDA_ERROR_DEVICE_UNAVAILABLE);
  assert(count == -1 && initialized[0] && !initialized[1]);
  fail_second_server = false;
  assert(lupine_virtual_device_count(&count, true) == CUDA_SUCCESS);
  assert(count == 3 && initialized[0] && initialized[1]);

  int discovery_requests = requests;
  assert(lupine_virtual_device_count(&count) == CUDA_SUCCESS && count == 3);
  for (int device = 0; device < count; ++device) {
    CUdevice remote_device = device;
    assert(lupine_rpc_conn_for_device(&remote_device) == &servers[device == 2]);
    assert(remote_device == (device == 2 ? 0 : device));
  }
  CUdevice invalid_device = count;
  assert(lupine_rpc_conn_for_device(&invalid_device) == nullptr);
  assert(requests == discovery_requests); // Both APIs share the cached table.
}
