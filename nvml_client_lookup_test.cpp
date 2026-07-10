#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>

// Compile the production NVML client in this test translation unit so the
// mock RPC transport below can exercise its exported API without a GPU or a
// running Lupine server.
#ifndef LUPINE_NVML_CLIENT_SOURCE
#define LUPINE_NVML_CLIENT_SOURCE "nvml_client.cpp"
#endif
#include LUPINE_NVML_CLIENT_SOURCE

namespace {

struct fake_device {
  nvmlDevice_t remote;
  std::string uuid;
  std::string pci_bus_id;
};

struct fake_server {
  std::vector<fake_device> devices;
};

struct fake_call {
  int op = 0;
  std::vector<unsigned char> request;
  std::vector<unsigned char> response;
  size_t response_offset = 0;
};

std::vector<fake_server> fake_servers;
fake_call fake_calls[16];
std::vector<unsigned int> lookup_connections;
unsigned int last_device_connection = UINT32_MAX;
nvmlDevice_t last_remote_device = nullptr;

[[noreturn]] void fail(const char *message) {
  std::cerr << "nvml_client_lookup_test: " << message << std::endl;
  std::exit(1);
}

void require(bool condition, const char *message) {
  if (!condition) {
    fail(message);
  }
}

unsigned int fake_connection_index(conn_t *conn) {
  if (conn < &conns[0] || conn >= &conns[16]) {
    fail("unexpected connection pointer");
  }
  return static_cast<unsigned int>(conn - &conns[0]);
}

template <typename T> void append_response(fake_call *call, const T &value) {
  const auto *bytes = reinterpret_cast<const unsigned char *>(&value);
  call->response.insert(call->response.end(), bytes, bytes + sizeof(value));
}

template <typename T> T read_request(const fake_call &call, size_t *offset) {
  if (*offset + sizeof(T) > call.request.size()) {
    fail("truncated mock RPC request");
  }
  T value;
  std::memcpy(&value, call.request.data() + *offset, sizeof(value));
  *offset += sizeof(value);
  return value;
}

std::string read_request_string(const fake_call &call, size_t *offset) {
  unsigned int length = read_request<unsigned int>(call, offset);
  if (*offset + length > call.request.size()) {
    fail("truncated mock RPC string");
  }
  std::string value;
  if (length != 0) {
    const char *data =
        reinterpret_cast<const char *>(call.request.data() + *offset);
    value.assign(data, strnlen(data, length));
  }
  *offset += length;
  return value;
}

void append_string_response(fake_call *call, const std::string &value,
                            unsigned int length) {
  size_t start = call->response.size();
  call->response.resize(start + length, 0);
  if (length > 0) {
    size_t copied = std::min(value.size(), static_cast<size_t>(length - 1));
    std::memcpy(call->response.data() + start, value.data(), copied);
  }
}

void handle_fake_call(conn_t *conn) {
  unsigned int conn_index = fake_connection_index(conn);
  require(conn_index < fake_servers.size(), "request used missing server");
  fake_call *call = &fake_calls[conn_index];
  fake_server &server = fake_servers[conn_index];
  call->response.clear();
  call->response_offset = 0;

  size_t offset = 0;
  if (call->op == RPC_nvmlDeviceGetCount_v2) {
    unsigned int count = static_cast<unsigned int>(server.devices.size());
    append_response(call, count);
    append_response(call, NVML_SUCCESS);
    return;
  }

  if (call->op == RPC_nvmlDeviceGetHandleByIndex_v2) {
    unsigned int index = read_request<unsigned int>(*call, &offset);
    nvmlDevice_t remote = nullptr;
    nvmlReturn_t result = NVML_ERROR_INVALID_ARGUMENT;
    if (index < server.devices.size()) {
      remote = server.devices[index].remote;
      result = NVML_SUCCESS;
    }
    append_response(call, remote);
    append_response(call, result);
    return;
  }

  if (call->op == RPC_nvmlDeviceGetHandleByUUID ||
      call->op == RPC_nvmlDeviceGetHandleByPciBusId_v2) {
    lookup_connections.push_back(conn_index);
    std::string value = read_request_string(*call, &offset);
    auto found = std::find_if(server.devices.begin(), server.devices.end(),
                              [&](const auto &device) {
                                return call->op == RPC_nvmlDeviceGetHandleByUUID
                                           ? device.uuid == value
                                           : device.pci_bus_id == value;
                              });
    nvmlDevice_t remote =
        found == server.devices.end() ? nullptr : found->remote;
    nvmlReturn_t result =
        found == server.devices.end() ? NVML_ERROR_NOT_FOUND : NVML_SUCCESS;
    append_response(call, remote);
    append_response(call, result);
    return;
  }

  if (call->op == RPC_nvmlDeviceGetUUID) {
    nvmlDevice_t remote = read_request<nvmlDevice_t>(*call, &offset);
    unsigned int length = read_request<unsigned int>(*call, &offset);
    last_device_connection = conn_index;
    last_remote_device = remote;
    auto found = std::find_if(
        server.devices.begin(), server.devices.end(),
        [&](const auto &device) { return device.remote == remote; });
    nvmlReturn_t result = found == server.devices.end()
                              ? NVML_ERROR_INVALID_ARGUMENT
                              : NVML_SUCCESS;
    append_string_response(
        call, found == server.devices.end() ? "" : found->uuid, length);
    append_response(call, result);
    return;
  }

  fail("unexpected mock RPC operation");
}

void reset_client(std::vector<fake_server> servers) {
  fake_servers = std::move(servers);
  lookup_connections.clear();
  last_device_connection = UINT32_MAX;
  last_remote_device = nullptr;
  devices.clear();
  devices_ready = false;
  conn_labels.clear();
  connected = true;
  nconns = static_cast<int>(fake_servers.size());
  require(nconns <= 16, "too many mock servers");
  for (int i = 0; i < 16; ++i) {
    conns[i] = {};
    fake_calls[i] = {};
  }
}

void test_uuid_lookup_returns_virtual_indexable_handle() {
  nvmlDevice_t remote = reinterpret_cast<nvmlDevice_t>(0x1234);
  fake_server server;
  server.devices.push_back({remote, "GPU-single", "0000:01:00.0"});
  reset_client({std::move(server)});

  nvmlDevice_t device = nullptr;
  require(nvmlDeviceGetHandleByUUID("GPU-single", &device) == NVML_SUCCESS,
          "UUID lookup failed");
  require(device != nullptr && device != remote,
          "UUID lookup leaked the raw server handle");

  unsigned int index = UINT32_MAX;
  require(nvmlDeviceGetIndex(device, &index) == NVML_SUCCESS,
          "UUID handle was rejected by nvmlDeviceGetIndex");
  require(index == 0, "UUID handle resolved to the wrong global index");

  char uuid[32] = {};
  require(nvmlDeviceGetUUID(device, uuid, sizeof(uuid)) == NVML_SUCCESS,
          "UUID handle was rejected by a routed device call");
  require(std::string(uuid) == "GPU-single", "routed UUID response mismatch");
  require(last_device_connection == 0 && last_remote_device == remote,
          "single-server device call did not restore the remote handle");
}

void test_string_lookups_search_and_route_across_servers() {
  // Deliberately use the same opaque value on both servers. Remote handles are
  // process-local, so connection identity must be part of the mapping.
  nvmlDevice_t shared_remote = reinterpret_cast<nvmlDevice_t>(0x5678);
  fake_server first;
  first.devices.push_back({shared_remote, "GPU-first", "0000:01:00.0"});
  fake_server second;
  second.devices.push_back({shared_remote, "GPU-second", "0000:02:00.0"});
  reset_client({std::move(first), std::move(second)});

  nvmlDevice_t by_uuid = nullptr;
  require(nvmlDeviceGetHandleByUUID("GPU-second", &by_uuid) == NVML_SUCCESS,
          "multi-server UUID lookup failed");
  require(lookup_connections == std::vector<unsigned int>({0, 1}),
          "UUID lookup did not search servers in order");

  unsigned int index = UINT32_MAX;
  require(nvmlDeviceGetIndex(by_uuid, &index) == NVML_SUCCESS && index == 1,
          "multi-server UUID handle mapped to the wrong global index");

  char uuid[32] = {};
  require(nvmlDeviceGetUUID(by_uuid, uuid, sizeof(uuid)) == NVML_SUCCESS,
          "multi-server UUID handle could not route a device call");
  require(last_device_connection == 1 && last_remote_device == shared_remote,
          "multi-server UUID handle routed to the wrong server");

  lookup_connections.clear();
  nvmlDevice_t by_pci = nullptr;
  require(nvmlDeviceGetHandleByPciBusId_v2("0000:02:00.0", &by_pci) ==
              NVML_SUCCESS,
          "multi-server PCI lookup failed");
  require(lookup_connections == std::vector<unsigned int>({0, 1}),
          "PCI lookup did not search servers in order");
  require(by_pci == by_uuid,
          "UUID and PCI lookup returned different virtual handles");
}

} // namespace

int rpc_dispatch(conn_t *, int) { return -1; }
int rpc_read_end(conn_t *) { return 1; }
int rpc_http2_client_init(conn_t *) { return 0; }
void rpc_conn_destroy(conn_t *) {}

int rpc_write_start_request(conn_t *conn, const int op) {
  fake_call &call = fake_calls[fake_connection_index(conn)];
  call.op = op;
  call.request.clear();
  call.response.clear();
  call.response_offset = 0;
  return 0;
}

int rpc_write(conn_t *conn, const void *data, const size_t size) {
  fake_call &call = fake_calls[fake_connection_index(conn)];
  const auto *bytes = static_cast<const unsigned char *>(data);
  call.request.insert(call.request.end(), bytes, bytes + size);
  return 0;
}

int rpc_wait_for_response(conn_t *conn) {
  handle_fake_call(conn);
  return 0;
}

int rpc_read(conn_t *conn, void *data, size_t size) {
  fake_call &call = fake_calls[fake_connection_index(conn)];
  if (call.response_offset + size > call.response.size()) {
    return -1;
  }
  std::memcpy(data, call.response.data() + call.response_offset, size);
  call.response_offset += size;
  return static_cast<int>(size);
}

int main() {
  test_uuid_lookup_returns_virtual_indexable_handle();
  test_string_lookups_search_and_route_across_servers();
  connected = false;
  nconns = 0;
  std::cout << "nvml_client_lookup_test: PASS" << std::endl;
  return 0;
}
