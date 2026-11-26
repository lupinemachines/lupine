#ifdef NDEBUG
#undef NDEBUG
#endif
// CPU-only monitoring tests: the CUDA calls are never used by these cases.
#include "monitor_server.cpp"
#include <cassert>
#include <iostream>

CUresult CUDAAPI cuDevicePrimaryCtxGetState(CUdevice, unsigned int *, int *) {
  return CUDA_ERROR_NOT_SUPPORTED;
}
CUresult CUDAAPI cuDeviceGetPCIBusId(char *, int, CUdevice) {
  return CUDA_ERROR_NOT_SUPPORTED;
}

int main() {
  child_snapshot child;
  strcpy(child.client_address, "10.0.0.1");
  strcpy(child.metadata.client_hostname, "worker");
  child.metadata.client_pid = 7;
  child.metadata.client_pid_namespace_inode = 42;
  strcpy(child.workload_id, R"(job:"worker\name)");
  device_snapshot device;
  device.uuid = "GPU-test";
  assert(client_id_for(child) == R"(job:"worker\name)");
  auto labels = client_labels(client_key_for(child, device));
  assert(labels.find(R"(client_id="job:\"worker\\name")") != std::string::npos);
  assert(labels.find("workload_id=") == std::string::npos);
  child.workload_id[0] = '\0';
  assert(client_id_for(child) == "7");
  registry = new monitor_registry{};
  pthread_mutex_init(&registry->registry_mutex, nullptr);
  for (int i = 0; i < 2; ++i) {
    auto &slot = registry->slots[i];
    slot.state = kSlotActive;
    slot.server_pid = 100 + i;
    slot.metadata = child.metadata;
    strcpy(slot.client_address, child.client_address);
    strcpy(slot.workload_id, "job:one");
  }
  auto metrics = lupine_monitoring_render_metrics();
  assert(metrics.find("client_id=\"job:one\"") != std::string::npos);
  assert(metrics.find("workload_id=") == std::string::npos);
  strcpy(registry->slots[1].workload_id, "job:two");
  metrics = lupine_monitoring_render_metrics();
  assert(metrics.find("client_id=\"job:one\"") == std::string::npos);
  assert(metrics.find("client_id=\"job:two\"") == std::string::npos);
  assert(metrics.find("client_id=\"7\"") != std::string::npos);
  registry->slots[0].state = registry->slots[1].state = kSlotFree;
  metrics = lupine_monitoring_render_metrics();
  assert(metrics.find("lupine_server_connection_info{") == std::string::npos);
  pthread_mutex_destroy(&registry->registry_mutex);
  delete registry;
  registry = nullptr;

  // An IPv4 client on the dual-stack listener keeps its IPv4 address label.
  sockaddr_in6 any = {};
  any.sin6_family = AF_INET6;
  socklen_t length = sizeof(any);
  const int v6only = 0;
  int listener = socket(AF_INET6, SOCK_STREAM, 0);
  if (listener >= 0 &&
      setsockopt(listener, IPPROTO_IPV6, IPV6_V6ONLY, &v6only,
                 sizeof(v6only)) == 0 &&
      bind(listener, reinterpret_cast<sockaddr *>(&any), length) == 0 &&
      listen(listener, 1) == 0) {
    getsockname(listener, reinterpret_cast<sockaddr *>(&any), &length);
    sockaddr_in loopback = {};
    loopback.sin_family = AF_INET;
    loopback.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    loopback.sin_port = any.sin6_port;
    int client = socket(AF_INET, SOCK_STREAM, 0);
    assert(connect(client, reinterpret_cast<sockaddr *>(&loopback),
                   sizeof(loopback)) == 0);
    int accepted = accept(listener, nullptr, nullptr);
    assert(peer_address(accepted) == "127.0.0.1");
    close(accepted);
    close(client);
  }
  if (listener >= 0) {
    close(listener);
  }
  std::cout << "monitor_server_test: PASS\n";
}
