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
  assert(client_id_for(child) == "10.0.0.1:worker:42:7");
  const auto labels = client_labels(client_key_for(child, device));
  assert(labels.find(R"(workload_id="job:\"worker\\name")") !=
         std::string::npos);
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
  assert(metrics.find("workload_id=\"job:one\"") != std::string::npos);
  strcpy(registry->slots[1].workload_id, "job:two");
  metrics = lupine_monitoring_render_metrics();
  assert(metrics.find("workload_id=\"job:one\"") == std::string::npos);
  assert(metrics.find("workload_id=\"job:two\"") == std::string::npos);
  assert(metrics.find("workload_id=\"\"") != std::string::npos);
  registry->slots[0].state = registry->slots[1].state = kSlotFree;
  metrics = lupine_monitoring_render_metrics();
  assert(metrics.find("lupine_server_connection_info{") == std::string::npos);
  pthread_mutex_destroy(&registry->registry_mutex);
  delete registry;
  registry = nullptr;
  std::cout << "monitor_server_test: PASS\n";
}
