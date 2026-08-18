#include "monitoring.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <map>
#include <sstream>
#include <string>
#include <tuple>
#include <unordered_map>
#include <vector>

#include "lupine_log.h"

#include <arpa/inet.h>
#include <cuda.h>
#include <dlfcn.h>
#include <errno.h>
#include <fcntl.h>
#include <memory>
#include <netinet/in.h>
#include <nvml.h>
#include <poll.h>
#include <pthread.h>
#include <signal.h>
#include <set>
#include <sys/mman.h>
#include <sys/socket.h>
#include <sys/time.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

namespace {

constexpr uint32_t kRegistryMagic = 0x4c504d52;
constexpr uint32_t kRegistryVersion = 4;
constexpr size_t kMaxServerChildren = 4096;
constexpr uint32_t kSlotFree = 0;
constexpr uint32_t kSlotActive = 1;

struct alignas(64) monitor_slot {
  uint32_t state;
  uint32_t sequence;
  int32_t server_pid;
  int32_t host_pid;
  char client_address[64];
  lupine_client_metadata_v2 metadata;
};

struct monitor_registry {
  uint32_t magic;
  uint32_t version;
  uint64_t dropped_connections;
  uint64_t host_pid_discovery_failures;
  uint64_t host_pid_discovery_ambiguities;
  pthread_mutex_t host_pid_discovery_mutex;
  monitor_slot slots[kMaxServerChildren];
};

monitor_registry *registry = nullptr;
pid_t monitor_pid = -1;

uint32_t atomic_load(const uint32_t *value) {
  return __atomic_load_n(value, __ATOMIC_ACQUIRE);
}

void atomic_store(uint32_t *value, uint32_t next) {
  __atomic_store_n(value, next, __ATOMIC_RELEASE);
}

uint64_t atomic_load(const uint64_t *value) {
  return __atomic_load_n(value, __ATOMIC_RELAXED);
}

bool atomic_compare_exchange(uint32_t *value, uint32_t expected,
                             uint32_t desired) {
  return __atomic_compare_exchange_n(value, &expected, desired, false,
                                     __ATOMIC_ACQ_REL, __ATOMIC_ACQUIRE);
}

void atomic_increment(uint64_t *value) {
  __atomic_fetch_add(value, 1, __ATOMIC_RELAXED);
}

void terminate_strings(lupine_client_metadata_v2 *metadata) {
  metadata->connection_kind[sizeof(metadata->connection_kind) - 1] = '\0';
  metadata->client_process_name[sizeof(metadata->client_process_name) - 1] =
      '\0';
  metadata->client_boot_id[sizeof(metadata->client_boot_id) - 1] = '\0';
  metadata->vdevice_index[sizeof(metadata->vdevice_index) - 1] = '\0';
  metadata->client_hostname[sizeof(metadata->client_hostname) - 1] = '\0';
  metadata->workload_id[sizeof(metadata->workload_id) - 1] = '\0';
}

bool store_metadata(int slot_index,
                    const lupine_client_metadata_v2 &metadata,
                    const char *client_address) {
  if (registry == nullptr || slot_index < 0 ||
      slot_index >= static_cast<int>(kMaxServerChildren)) {
    return false;
  }
  monitor_slot *slot = &registry->slots[slot_index];
  if (atomic_load(&slot->state) != kSlotActive) {
    return false;
  }
  __atomic_fetch_add(&slot->sequence, 1, __ATOMIC_ACQ_REL);
  slot->metadata = metadata;
  if (client_address != nullptr) {
    size_t length = strnlen(client_address, sizeof(slot->client_address) - 1);
    memcpy(slot->client_address, client_address, length);
    slot->client_address[length] = '\0';
  }
  __atomic_fetch_add(&slot->sequence, 1, __ATOMIC_RELEASE);
  return true;
}

bool store_host_pid(int slot_index, int32_t host_pid) {
  if (registry == nullptr || slot_index < 0 || host_pid <= 0 ||
      slot_index >= static_cast<int>(kMaxServerChildren)) {
    return false;
  }
  monitor_slot *slot = &registry->slots[slot_index];
  if (atomic_load(&slot->state) != kSlotActive) {
    return false;
  }
  __atomic_fetch_add(&slot->sequence, 1, __ATOMIC_ACQ_REL);
  slot->host_pid = host_pid;
  __atomic_fetch_add(&slot->sequence, 1, __ATOMIC_RELEASE);
  return true;
}

struct child_snapshot {
  int32_t server_pid = 0;
  int32_t host_pid = 0;
  char client_address[64] = {};
  lupine_client_metadata_v2 metadata = {};
};

std::vector<child_snapshot> snapshot_children() {
  std::vector<child_snapshot> result;
  if (registry == nullptr) {
    return result;
  }
  for (size_t i = 0; i < kMaxServerChildren; ++i) {
    monitor_slot *slot = &registry->slots[i];
    if (atomic_load(&slot->state) != kSlotActive) {
      continue;
    }
    child_snapshot snapshot;
    bool copied = false;
    for (int attempt = 0; attempt < 4; ++attempt) {
      uint32_t before = atomic_load(&slot->sequence);
      if ((before & 1U) != 0) {
        continue;
      }
      snapshot.server_pid = slot->server_pid;
      snapshot.host_pid = slot->host_pid;
      memcpy(snapshot.client_address, slot->client_address,
             sizeof(snapshot.client_address));
      snapshot.metadata = slot->metadata;
      __atomic_thread_fence(__ATOMIC_ACQUIRE);
      uint32_t after = atomic_load(&slot->sequence);
      if (before == after && (after & 1U) == 0 &&
          atomic_load(&slot->state) == kSlotActive) {
        copied = true;
        break;
      }
    }
    if (copied && snapshot.server_pid > 0) {
      if (kill(snapshot.server_pid, 0) != 0 && errno == ESRCH) {
        atomic_compare_exchange(&slot->state, kSlotActive, kSlotFree);
        continue;
      }
      terminate_strings(&snapshot.metadata);
      snapshot.client_address[sizeof(snapshot.client_address) - 1] = '\0';
      result.push_back(snapshot);
    }
  }
  return result;
}

std::string prometheus_escape(const char *value) {
  std::string result;
  if (value == nullptr) {
    return result;
  }
  for (const unsigned char *p =
           reinterpret_cast<const unsigned char *>(value);
       *p != '\0'; ++p) {
    switch (*p) {
    case '\\':
      result += "\\\\";
      break;
    case '"':
      result += "\\\"";
      break;
    case '\n':
      result += "\\n";
      break;
    case '\r':
      result += "\\r";
      break;
    default:
      if (*p >= 0x20 && *p != 0x7f) {
        result.push_back(static_cast<char>(*p));
      }
      break;
    }
  }
  return result;
}

std::string label(const char *name, const std::string &value) {
  return std::string(name) + "=\"" + prometheus_escape(value.c_str()) + "\"";
}

std::string peer_address(lupine_socket_t socket) {
  sockaddr_storage address = {};
  socklen_t length = sizeof(address);
  if (getpeername(socket, reinterpret_cast<sockaddr *>(&address), &length) !=
      0) {
    return "unknown";
  }
  char text[INET6_ADDRSTRLEN] = {};
  const void *source = nullptr;
  if (address.ss_family == AF_INET) {
    source = &reinterpret_cast<const sockaddr_in *>(&address)->sin_addr;
  } else if (address.ss_family == AF_INET6) {
    source = &reinterpret_cast<const sockaddr_in6 *>(&address)->sin6_addr;
  }
  if (source == nullptr ||
      inet_ntop(address.ss_family, source, text, sizeof(text)) == nullptr) {
    return "unknown";
  }
  return text;
}

struct nvml_api {
  void *library = nullptr;
  nvmlReturn_t (*init)() = nullptr;
  nvmlReturn_t (*shutdown)() = nullptr;
  nvmlReturn_t (*device_count)(unsigned int *) = nullptr;
  nvmlReturn_t (*device_handle)(unsigned int, nvmlDevice_t *) = nullptr;
  nvmlReturn_t (*device_handle_by_pci_bus_id)(const char *, nvmlDevice_t *) =
      nullptr;
  nvmlReturn_t (*device_uuid)(nvmlDevice_t, char *, unsigned int) = nullptr;
  nvmlReturn_t (*device_name)(nvmlDevice_t, char *, unsigned int) = nullptr;
  nvmlReturn_t (*memory_info)(nvmlDevice_t, nvmlMemory_t *) = nullptr;
  nvmlReturn_t (*utilization)(nvmlDevice_t, nvmlUtilization_t *) = nullptr;
  nvmlReturn_t (*compute_processes)(nvmlDevice_t, unsigned int *,
                                    nvmlProcessInfo_t *) = nullptr;
  nvmlReturn_t (*process_utilization)(nvmlDevice_t,
                                      nvmlProcessUtilizationSample_t *,
                                      unsigned int *, unsigned long long) =
      nullptr;
  bool initialized = false;

  template <typename Fn> Fn symbol(const char *name) {
    return reinterpret_cast<Fn>(dlsym(library, name));
  }

  bool open() {
    library = dlopen("libnvidia-ml.so.1", RTLD_NOW | RTLD_LOCAL);
    if (library == nullptr) {
      return false;
    }
    init = symbol<decltype(init)>("nvmlInit_v2");
    if (init == nullptr) {
      init = symbol<decltype(init)>("nvmlInit");
    }
    shutdown = symbol<decltype(shutdown)>("nvmlShutdown");
    device_count =
        symbol<decltype(device_count)>("nvmlDeviceGetCount_v2");
    if (device_count == nullptr) {
      device_count = symbol<decltype(device_count)>("nvmlDeviceGetCount");
    }
    device_handle =
        symbol<decltype(device_handle)>("nvmlDeviceGetHandleByIndex_v2");
    if (device_handle == nullptr) {
      device_handle =
          symbol<decltype(device_handle)>("nvmlDeviceGetHandleByIndex");
    }
    device_handle_by_pci_bus_id =
        symbol<decltype(device_handle_by_pci_bus_id)>(
            "nvmlDeviceGetHandleByPciBusId_v2");
    if (device_handle_by_pci_bus_id == nullptr) {
      device_handle_by_pci_bus_id =
          symbol<decltype(device_handle_by_pci_bus_id)>(
              "nvmlDeviceGetHandleByPciBusId");
    }
    device_uuid = symbol<decltype(device_uuid)>("nvmlDeviceGetUUID");
    device_name = symbol<decltype(device_name)>("nvmlDeviceGetName");
    memory_info = symbol<decltype(memory_info)>("nvmlDeviceGetMemoryInfo");
    utilization =
        symbol<decltype(utilization)>("nvmlDeviceGetUtilizationRates");
    compute_processes = symbol<decltype(compute_processes)>(
        "nvmlDeviceGetComputeRunningProcesses_v2");
    if (compute_processes == nullptr) {
      compute_processes = symbol<decltype(compute_processes)>(
          "nvmlDeviceGetComputeRunningProcesses");
    }
    process_utilization = symbol<decltype(process_utilization)>(
        "nvmlDeviceGetProcessUtilization");
    if (init == nullptr || device_count == nullptr ||
        device_handle == nullptr || device_handle_by_pci_bus_id == nullptr ||
        device_uuid == nullptr ||
        device_name == nullptr || memory_info == nullptr ||
        utilization == nullptr || compute_processes == nullptr ||
        init() != NVML_SUCCESS) {
      return false;
    }
    initialized = true;
    return true;
  }

  ~nvml_api() {
    if (initialized && shutdown != nullptr) {
      shutdown();
    }
    if (library != nullptr) {
      dlclose(library);
    }
  }
};

std::vector<nvmlProcessInfo_t> running_processes(nvml_api &api,
                                                 nvmlDevice_t device,
                                                 bool *query_ok = nullptr) {
  if (query_ok != nullptr) {
    *query_ok = false;
  }
  for (int attempt = 0; attempt < 4; ++attempt) {
    unsigned int count = 0;
    nvmlReturn_t status = api.compute_processes(device, &count, nullptr);
    if (status == NVML_SUCCESS && count == 0) {
      if (query_ok != nullptr) {
        *query_ok = true;
      }
      return {};
    }
    if (status != NVML_ERROR_INSUFFICIENT_SIZE || count == 0) {
      return {};
    }
    count = std::min<unsigned int>(count + 16, 65536);
    std::vector<nvmlProcessInfo_t> processes(count);
    unsigned int returned = count;
    status = api.compute_processes(device, &returned, processes.data());
    if (status == NVML_SUCCESS) {
      processes.resize(std::min<unsigned int>(returned, processes.size()));
      if (query_ok != nullptr) {
        *query_ok = true;
      }
      return processes;
    }
    if (status != NVML_ERROR_INSUFFICIENT_SIZE) {
      return {};
    }
  }
  return {};
}

bool lock_host_pid_discovery() {
  if (registry == nullptr) {
    return false;
  }
  int status = pthread_mutex_lock(&registry->host_pid_discovery_mutex);
  if (status == EOWNERDEAD) {
    pthread_mutex_consistent(&registry->host_pid_discovery_mutex);
    return true;
  }
  return status == 0;
}

bool slot_has_host_pid(int slot_index) {
  return registry != nullptr && slot_index >= 0 &&
         slot_index < static_cast<int>(kMaxServerChildren) &&
         atomic_load(&registry->slots[slot_index].state) == kSlotActive &&
         __atomic_load_n(&registry->slots[slot_index].host_pid,
                         __ATOMIC_ACQUIRE) > 0;
}

bool host_pid_owned_by_other_slot(int32_t host_pid, int slot_index) {
  if (registry == nullptr || host_pid <= 0) {
    return false;
  }
  for (size_t i = 0; i < kMaxServerChildren; ++i) {
    if (static_cast<int>(i) == slot_index ||
        atomic_load(&registry->slots[i].state) != kSlotActive) {
      continue;
    }
    if (__atomic_load_n(&registry->slots[i].host_pid, __ATOMIC_ACQUIRE) ==
        host_pid) {
      return true;
    }
  }
  return false;
}

struct host_pid_probe {
  int slot_index = -1;
  nvml_api api;
  nvmlDevice_t device = nullptr;
  std::set<unsigned int> before_pids;
  bool lock_held = false;
};

thread_local std::unique_ptr<host_pid_probe> active_host_pid_probe;

void note_discovery_failure() {
  if (registry != nullptr) {
    atomic_increment(&registry->host_pid_discovery_failures);
  }
}

void begin_context_probe(conn_t *conn, int cuda_device) {
  if (conn == nullptr || registry == nullptr || conn->monitor_slot < 0 ||
      slot_has_host_pid(conn->monitor_slot) || active_host_pid_probe != nullptr) {
    return;
  }

  std::unique_ptr<host_pid_probe> probe(new (std::nothrow) host_pid_probe());
  if (probe == nullptr || !probe->api.open()) {
    note_discovery_failure();
    return;
  }
  char pci_bus_id[64] = {};
  if (cuDeviceGetPCIBusId(pci_bus_id, sizeof(pci_bus_id), cuda_device) !=
          CUDA_SUCCESS ||
      probe->api.device_handle_by_pci_bus_id(pci_bus_id, &probe->device) !=
          NVML_SUCCESS ||
      probe->device == nullptr) {
    note_discovery_failure();
    return;
  }
  if (!lock_host_pid_discovery()) {
    note_discovery_failure();
    return;
  }
  probe->lock_held = true;

  if (slot_has_host_pid(conn->monitor_slot)) {
    pthread_mutex_unlock(&registry->host_pid_discovery_mutex);
    return;
  }

  bool query_ok = false;
  std::vector<nvmlProcessInfo_t> before =
      running_processes(probe->api, probe->device, &query_ok);
  if (!query_ok) {
    pthread_mutex_unlock(&registry->host_pid_discovery_mutex);
    note_discovery_failure();
    return;
  }
  for (const auto &process : before) {
    probe->before_pids.insert(process.pid);
  }
  probe->slot_index = conn->monitor_slot;
  active_host_pid_probe = std::move(probe);
}

void end_context_probe(conn_t *conn, bool context_created) {
  (void)conn;
  std::unique_ptr<host_pid_probe> probe = std::move(active_host_pid_probe);
  if (probe == nullptr) {
    return;
  }

  bool resolved = false;
  bool ambiguous = false;
  if (context_created) {
    for (int attempt = 0; attempt < 50; ++attempt) {
      bool query_ok = false;
      std::vector<nvmlProcessInfo_t> after =
          running_processes(probe->api, probe->device, &query_ok);
      if (!query_ok) {
        break;
      }
      std::vector<unsigned int> candidates;
      for (const auto &process : after) {
        if (probe->before_pids.count(process.pid) == 0) {
          candidates.push_back(process.pid);
        }
      }
      if (candidates.size() == 1) {
        int32_t candidate = static_cast<int32_t>(candidates.front());
        if (candidate > 0 &&
            !host_pid_owned_by_other_slot(candidate, probe->slot_index)) {
          resolved = store_host_pid(probe->slot_index, candidate);
        } else {
          ambiguous = true;
        }
        break;
      }
      if (candidates.size() > 1) {
        ambiguous = true;
        break;
      }
      usleep(20 * 1000);
    }
  }

  if (probe->lock_held && registry != nullptr) {
    pthread_mutex_unlock(&registry->host_pid_discovery_mutex);
  }
  if (context_created && !resolved) {
    if (ambiguous && registry != nullptr) {
      atomic_increment(&registry->host_pid_discovery_ambiguities);
    }
    note_discovery_failure();
  }
}

using process_util_map = std::unordered_map<unsigned int, unsigned int>;

process_util_map process_utilization(nvml_api &api, nvmlDevice_t device,
                                     unsigned long long *last_timestamp) {
  process_util_map result;
  if (api.process_utilization == nullptr) {
    return result;
  }
  unsigned int count = 0;
  nvmlReturn_t status = api.process_utilization(
      device, nullptr, &count, *last_timestamp);
  if ((status != NVML_ERROR_INSUFFICIENT_SIZE && status != NVML_SUCCESS) ||
      count == 0) {
    return result;
  }
  count = std::min<unsigned int>(count + 16, 65536);
  std::vector<nvmlProcessUtilizationSample_t> samples(count);
  status = api.process_utilization(device, samples.data(), &count,
                                   *last_timestamp);
  if (status != NVML_SUCCESS) {
    return result;
  }
  samples.resize(std::min<unsigned int>(count, samples.size()));
  std::unordered_map<unsigned int,
                     std::pair<unsigned long long, unsigned int>>
      latest;
  for (const auto &sample : samples) {
    auto &entry = latest[sample.pid];
    if (sample.timeStamp >= entry.first) {
      entry = {sample.timeStamp, sample.smUtil};
    }
    *last_timestamp = std::max(*last_timestamp, sample.timeStamp);
  }
  for (const auto &entry : latest) {
    result.emplace(entry.first, entry.second.second);
  }
  return result;
}

struct device_snapshot {
  unsigned int index = 0;
  std::string uuid;
  std::string name;
  unsigned long long memory_used = 0;
  unsigned int utilization = 0;
  std::vector<nvmlProcessInfo_t> processes;
  process_util_map process_utils;
};

struct client_key {
  std::string client_id;
  std::string client_address;
  std::string client_hostname;
  std::string client_process_name;
  std::string workload_id;
  std::string vdevice_index;
  unsigned int device_index = 0;
  std::string device_uuid;

  bool operator<(const client_key &other) const {
    return std::tie(client_id, client_address, client_hostname,
                    client_process_name, workload_id, vdevice_index,
                    device_index, device_uuid) <
           std::tie(other.client_id, other.client_address,
                    other.client_hostname, other.client_process_name,
                    other.workload_id, other.vdevice_index,
                    other.device_index, other.device_uuid);
  }
};

struct workload_key {
  std::string workload_id;
  unsigned int device_index = 0;
  std::string device_uuid;

  bool operator<(const workload_key &other) const {
    return std::tie(workload_id, device_index, device_uuid) <
           std::tie(other.workload_id, other.device_index, other.device_uuid);
  }
};

struct process_sample {
  unsigned long long memory_used = 0;
  unsigned int sm_percent = 0;
};

std::string client_id_for(const child_snapshot &child) {
  const auto &metadata = child.metadata;
  if (metadata.client_pid > 0 && metadata.client_start_time_ticks > 0 &&
      metadata.client_pid_namespace_inode > 0) {
    std::ostringstream out;
    if (metadata.client_boot_id[0] != '\0') {
      out << metadata.client_boot_id;
    } else if (metadata.client_hostname[0] != '\0') {
      out << metadata.client_hostname;
    } else {
      out << child.client_address;
    }
    out << ':' << metadata.client_pid_namespace_inode << ':'
        << metadata.client_pid << ':' << metadata.client_start_time_ticks;
    return out.str();
  }
  return std::string("connection:") + child.client_address + ':' +
         std::to_string(child.server_pid);
}

client_key client_key_for(const child_snapshot &child,
                          const device_snapshot &device) {
  const auto &metadata = child.metadata;
  client_key key;
  key.client_id = client_id_for(child);
  key.client_address = child.client_address[0] != '\0'
                           ? child.client_address
                           : "unknown";
  key.client_hostname = metadata.client_hostname[0] != '\0'
                            ? metadata.client_hostname
                            : "unknown";
  key.client_process_name = metadata.client_process_name[0] != '\0'
                                ? metadata.client_process_name
                                : "unknown";
  key.workload_id = metadata.workload_id;
  key.vdevice_index = metadata.vdevice_index[0] != '\0'
                          ? metadata.vdevice_index
                          : "unknown";
  key.device_index = device.index;
  key.device_uuid = device.uuid;
  return key;
}

std::string client_labels(const client_key &key) {
  std::ostringstream out;
  out << label("client_id", key.client_id) << ','
      << label("client_address", key.client_address) << ','
      << label("client_hostname", key.client_hostname) << ','
      << label("client_name", key.client_process_name) << ','
      << label("device_uuid", key.device_uuid) << ','
      << "device_index=\"" << key.device_index << "\","
      << label("vdevice_index", key.vdevice_index) << ','
      << label("workload_id", key.workload_id);
  return out.str();
}

std::string workload_labels(const workload_key &key) {
  std::ostringstream out;
  out << label("workload_id", key.workload_id) << ','
      << label("device_uuid", key.device_uuid) << ','
      << "device_index=\"" << key.device_index << "\"";
  return out.str();
}

std::string render_metrics(nvml_api *api,
                           std::vector<unsigned long long> *last_timestamps) {
  std::ostringstream out;
  out << "# HELP lupine_monitor_nvml_up Whether the monitor can query NVML.\n"
      << "# TYPE lupine_monitor_nvml_up gauge\n"
      << "lupine_monitor_nvml_up " << (api != nullptr ? 1 : 0) << "\n"
      << "# HELP lupine_monitor_process_utilization_supported Whether per-process "
         "SM utilization is available from NVML.\n"
      << "# TYPE lupine_monitor_process_utilization_supported gauge\n"
      << "lupine_monitor_process_utilization_supported "
      << (api != nullptr && api->process_utilization != nullptr ? 1 : 0)
      << "\n"
      << "# HELP lupine_monitor_dropped_connections_total Server connections "
         "not tracked because the shared registry was full.\n"
      << "# TYPE lupine_monitor_dropped_connections_total counter\n"
      << "lupine_monitor_dropped_connections_total "
      << (registry != nullptr
              ? atomic_load(&registry->dropped_connections)
              : 0)
      << "\n"
      << "# HELP lupine_monitor_host_pid_discovery_failures_total Host-PID "
         "discovery attempts that did not resolve a server host PID.\n"
      << "# TYPE lupine_monitor_host_pid_discovery_failures_total counter\n"
      << "lupine_monitor_host_pid_discovery_failures_total "
      << (registry != nullptr
              ? atomic_load(&registry->host_pid_discovery_failures)
              : 0)
      << "\n"
      << "# HELP lupine_monitor_host_pid_discovery_ambiguities_total Host-PID "
         "discoveries that did not find one unique unused NVML process.\n"
      << "# TYPE lupine_monitor_host_pid_discovery_ambiguities_total counter\n"
      << "lupine_monitor_host_pid_discovery_ambiguities_total "
      << (registry != nullptr
              ? atomic_load(&registry->host_pid_discovery_ambiguities)
              : 0)
      << "\n";

  std::vector<child_snapshot> children = snapshot_children();
  out << "# HELP lupine_server_connection_info Active server child process and "
         "client identity mapping.\n"
      << "# TYPE lupine_server_connection_info gauge\n";
  for (const auto &child : children) {
    const auto &metadata = child.metadata;
    out << "lupine_server_connection_info{"
        << label("client_id", client_id_for(child)) << ','
        << label("client_address", child.client_address) << ','
        << label("client_hostname", metadata.client_hostname) << ','
        << label("client_name", metadata.client_process_name) << ','
        << "client_pid=\"" << metadata.client_pid << "\","
        << "client_pid_namespace_inode=\""
        << metadata.client_pid_namespace_inode << "\","
        << "client_start_time_ticks=\"" << metadata.client_start_time_ticks
        << "\","
        << label("connection_kind", metadata.connection_kind) << ','
        << "connection_index=\"" << metadata.connection_index << "\","
        << "host_pid=\"" << child.host_pid << "\","
        << "server_pid=\"" << child.server_pid << "\","
        << label("vdevice_index", metadata.vdevice_index) << ','
        << label("workload_id", metadata.workload_id) << "} 1\n";
  }

  if (api == nullptr) {
    return out.str();
  }

  unsigned int device_count = 0;
  if (api->device_count(&device_count) != NVML_SUCCESS) {
    return out.str();
  }
  if (last_timestamps->size() < device_count) {
    last_timestamps->resize(device_count, 0);
  }

  std::unordered_map<int32_t, const child_snapshot *> by_pid;
  for (const auto &child : children) {
    if (child.host_pid > 0) {
      by_pid[child.host_pid] = &child;
    }
  }
  std::vector<device_snapshot> devices;
  for (unsigned int i = 0; i < device_count; ++i) {
    nvmlDevice_t handle = nullptr;
    if (api->device_handle(i, &handle) != NVML_SUCCESS) {
      continue;
    }
    char uuid[NVML_DEVICE_UUID_BUFFER_SIZE] = {};
    char name[NVML_DEVICE_NAME_BUFFER_SIZE] = {};
    nvmlMemory_t memory = {};
    nvmlUtilization_t utilization = {};
    if (api->device_uuid(handle, uuid, sizeof(uuid)) != NVML_SUCCESS) {
      snprintf(uuid, sizeof(uuid), "device-%u", i);
    }
    if (api->device_name(handle, name, sizeof(name)) != NVML_SUCCESS) {
      snprintf(name, sizeof(name), "unknown");
    }
    device_snapshot device;
    device.index = i;
    device.uuid = uuid;
    device.name = name;
    if (api->memory_info(handle, &memory) == NVML_SUCCESS) {
      device.memory_used = memory.used;
    }
    if (api->utilization(handle, &utilization) == NVML_SUCCESS) {
      device.utilization = utilization.gpu;
    }
    device.processes = running_processes(*api, handle);
    device.process_utils =
        process_utilization(*api, handle, &(*last_timestamps)[i]);
    devices.push_back(std::move(device));
  }

  out << "# HELP lupine_host_gpu_memory_used_bytes GPU device memory usage in "
         "bytes.\n"
      << "# TYPE lupine_host_gpu_memory_used_bytes gauge\n"
      << "# HELP lupine_host_gpu_utilization_percent GPU core utilization "
         "percentage from 0 to 100.\n"
      << "# TYPE lupine_host_gpu_utilization_percent gauge\n";
  out << std::setprecision(12);
  for (const auto &device : devices) {
    std::string labels = "device_index=\"" +
                         std::to_string(device.index) + "\"," +
                         label("device_type", device.name) + "," +
                         label("device_uuid", device.uuid);
    out << "lupine_host_gpu_memory_used_bytes{" << labels << "} "
        << device.memory_used << "\n"
        << "lupine_host_gpu_utilization_percent{" << labels << "} "
        << std::min(device.utilization, 100U) << "\n";
  }

  std::map<client_key, process_sample> clients;
  std::map<workload_key, process_sample> workloads;
  for (const auto &device : devices) {
    for (const auto &process : device.processes) {
      auto child = by_pid.find(static_cast<int32_t>(process.pid));
      if (child == by_pid.end()) {
        continue;
      }
      const child_snapshot &snapshot = *child->second;
      auto &sample = clients[client_key_for(snapshot, device)];
      if (process.usedGpuMemory != NVML_VALUE_NOT_AVAILABLE) {
        sample.memory_used += process.usedGpuMemory;
        if (snapshot.metadata.workload_id[0] != '\0') {
          workload_key key = {snapshot.metadata.workload_id, device.index,
                              device.uuid};
          workloads[key].memory_used += process.usedGpuMemory;
        }
      }
    }
    for (const auto &util : device.process_utils) {
      auto child = by_pid.find(static_cast<int32_t>(util.first));
      if (child == by_pid.end()) {
        continue;
      }
      const child_snapshot &snapshot = *child->second;
      auto &sample = clients[client_key_for(snapshot, device)];
      sample.sm_percent =
          std::min(100U, sample.sm_percent + std::min(100U, util.second));
      if (snapshot.metadata.workload_id[0] != '\0') {
        workload_key key = {snapshot.metadata.workload_id, device.index,
                            device.uuid};
        auto &workload = workloads[key];
        workload.sm_percent = std::min(
            100U, workload.sm_percent + std::min(100U, util.second));
      }
    }
  }

  out << "# HELP lupine_client_device_memory_used_bytes GPU memory used by "
         "Lupine server processes serving one client process.\n"
      << "# TYPE lupine_client_device_memory_used_bytes gauge\n"
      << "# HELP lupine_client_device_utilization_percent SM utilization "
         "percentage of Lupine server processes serving one client process, "
         "from 0 to 100.\n"
      << "# TYPE lupine_client_device_utilization_percent gauge\n";
  for (const auto &entry : clients) {
    std::string labels = client_labels(entry.first);
    out << "lupine_client_device_memory_used_bytes{" << labels << "} "
        << entry.second.memory_used << "\n"
        << "lupine_client_device_utilization_percent{" << labels << "} "
        << entry.second.sm_percent << "\n";
  }

  out << "# HELP lupine_workload_device_memory_used_bytes GPU memory used by "
         "all Lupine clients sharing an explicit workload ID.\n"
      << "# TYPE lupine_workload_device_memory_used_bytes gauge\n"
      << "# HELP lupine_workload_device_utilization_percent SM utilization "
         "percentage of all Lupine clients sharing an explicit workload ID, "
         "from 0 to 100.\n"
      << "# TYPE lupine_workload_device_utilization_percent gauge\n";
  for (const auto &entry : workloads) {
    std::string labels = workload_labels(entry.first);
    out << "lupine_workload_device_memory_used_bytes{" << labels << "} "
        << entry.second.memory_used << "\n"
        << "lupine_workload_device_utilization_percent{" << labels << "} "
        << entry.second.sm_percent << "\n";
  }
  return out.str();
}

bool parse_uint16_env(const char *name, uint16_t *value, bool required) {
  const char *text = getenv(name);
  if (text == nullptr || text[0] == '\0') {
    return !required;
  }
  char *end = nullptr;
  errno = 0;
  unsigned long parsed = strtoul(text, &end, 10);
  if (errno != 0 || end == text || *end != '\0' || parsed == 0 ||
      parsed > 65535) {
    return false;
  }
  *value = static_cast<uint16_t>(parsed);
  return true;
}

unsigned int metrics_interval_ms() {
  const char *text = getenv("LUPINE_METRICS_INTERVAL_MS");
  if (text == nullptr || text[0] == '\0') {
    return 1000;
  }
  char *end = nullptr;
  unsigned long parsed = strtoul(text, &end, 10);
  if (end == text || *end != '\0' || parsed < 100 || parsed > 60000) {
    return 1000;
  }
  return static_cast<unsigned int>(parsed);
}

int create_metrics_listener(uint16_t port) {
  int fd = socket(AF_INET, SOCK_STREAM, 0);
  if (fd < 0) {
    return -1;
  }
  int one = 1;
  setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &one, sizeof(one));
  sockaddr_in address = {};
  address.sin_family = AF_INET;
  address.sin_port = htons(port);
  const char *bind_address = getenv("LUPINE_METRICS_BIND");
  if (bind_address == nullptr || bind_address[0] == '\0') {
    bind_address = "0.0.0.0";
  }
  if (inet_pton(AF_INET, bind_address, &address.sin_addr) != 1 ||
      bind(fd, reinterpret_cast<sockaddr *>(&address), sizeof(address)) != 0 ||
      listen(fd, 64) != 0) {
    close(fd);
    return -1;
  }
  return fd;
}

bool send_all(int fd, const std::string &data) {
  size_t offset = 0;
  while (offset < data.size()) {
    ssize_t sent = send(fd, data.data() + offset, data.size() - offset,
                        MSG_NOSIGNAL);
    if (sent <= 0) {
      return false;
    }
    offset += static_cast<size_t>(sent);
  }
  return true;
}

void serve_request(int listener, const std::string &metrics) {
  int client = accept(listener, nullptr, nullptr);
  if (client < 0) {
    return;
  }
  timeval timeout = {};
  timeout.tv_sec = 2;
  setsockopt(client, SOL_SOCKET, SO_RCVTIMEO, &timeout, sizeof(timeout));
  char request[4096] = {};
  ssize_t length = recv(client, request, sizeof(request) - 1, 0);
  bool get_metrics =
      length > 0 &&
      (strncmp(request, "GET /metrics ", 13) == 0 ||
       strncmp(request, "GET /metrics?", 13) == 0);
  const std::string body = get_metrics ? metrics : std::string("not found\n");
  std::ostringstream response;
  response << (get_metrics ? "HTTP/1.1 200 OK\r\n"
                           : "HTTP/1.1 404 Not Found\r\n")
           << "Content-Type: text/plain; version=0.0.4; charset=utf-8\r\n"
           << "Content-Length: " << body.size() << "\r\n"
           << "Connection: close\r\n\r\n"
           << body;
  send_all(client, response.str());
  close(client);
}

[[noreturn]] void monitor_main(int listener) {
  signal(SIGPIPE, SIG_IGN);
  nvml_api api;
  nvml_api *active_api = api.open() ? &api : nullptr;
  std::vector<unsigned long long> last_timestamps;
  unsigned int interval = metrics_interval_ms();
  std::string metrics = render_metrics(active_api, &last_timestamps);
  auto next_sample = std::chrono::steady_clock::now() +
                     std::chrono::milliseconds(interval);
  for (;;) {
    auto now = std::chrono::steady_clock::now();
    if (now >= next_sample) {
      metrics = render_metrics(active_api, &last_timestamps);
      next_sample = now + std::chrono::milliseconds(interval);
    }
    auto remaining = std::chrono::duration_cast<std::chrono::milliseconds>(
        next_sample - std::chrono::steady_clock::now());
    int timeout = static_cast<int>(std::max<int64_t>(0, remaining.count()));
    pollfd descriptor = {listener, POLLIN, 0};
    int result = poll(&descriptor, 1, timeout);
    if (result > 0 && (descriptor.revents & POLLIN) != 0) {
      serve_request(listener, metrics);
    }
  }
}

} // namespace

int lupine_monitoring_start(lupine_socket_t rpc_listener) {
  const char *port_text = getenv("LUPINE_METRICS_PORT");
  if (port_text == nullptr || port_text[0] == '\0' ||
      strcmp(port_text, "0") == 0) {
    return 0;
  }
  uint16_t port = 0;
  if (!parse_uint16_env("LUPINE_METRICS_PORT", &port, true)) {
    LUPINE_LOG_ERROR("Invalid LUPINE_METRICS_PORT '" << port_text
                                                     << "'; expected 1-65535");
    return -1;
  }
  void *memory = mmap(nullptr, sizeof(monitor_registry), PROT_READ | PROT_WRITE,
                      MAP_SHARED | MAP_ANONYMOUS, -1, 0);
  if (memory == MAP_FAILED) {
    LUPINE_LOG_ERROR("Unable to allocate the Lupine monitor registry: "
                     << strerror(errno));
    return -1;
  }
  registry = static_cast<monitor_registry *>(memory);
  memset(registry, 0, sizeof(*registry));
  registry->magic = kRegistryMagic;
  registry->version = kRegistryVersion;

  pthread_mutexattr_t mutex_attributes;
  int mutex_status = pthread_mutexattr_init(&mutex_attributes);
  bool mutex_attributes_initialized = mutex_status == 0;
  if (mutex_status == 0) {
    mutex_status = pthread_mutexattr_setpshared(
        &mutex_attributes, PTHREAD_PROCESS_SHARED);
  }
  if (mutex_status == 0) {
    mutex_status = pthread_mutexattr_setrobust(&mutex_attributes,
                                               PTHREAD_MUTEX_ROBUST);
  }
  if (mutex_status == 0) {
    mutex_status = pthread_mutex_init(&registry->host_pid_discovery_mutex,
                                      &mutex_attributes);
  }
  if (mutex_attributes_initialized) {
    pthread_mutexattr_destroy(&mutex_attributes);
  }
  if (mutex_status != 0) {
    LUPINE_LOG_ERROR("Unable to initialize the host-PID discovery lock: "
                     << strerror(mutex_status));
    munmap(registry, sizeof(*registry));
    registry = nullptr;
    return -1;
  }

  int listener = create_metrics_listener(port);
  if (listener < 0) {
    LUPINE_LOG_ERROR("Unable to bind the Lupine metrics endpoint: "
                     << strerror(errno));
    pthread_mutex_destroy(&registry->host_pid_discovery_mutex);
    munmap(registry, sizeof(*registry));
    registry = nullptr;
    return -1;
  }
  pid_t pid = fork();
  if (pid < 0) {
    LUPINE_LOG_ERROR("Unable to start the Lupine monitor process: "
                     << strerror(errno));
    close(listener);
    pthread_mutex_destroy(&registry->host_pid_discovery_mutex);
    munmap(registry, sizeof(*registry));
    registry = nullptr;
    return -1;
  }
  if (pid == 0) {
    close(rpc_listener);
    monitor_main(listener);
  }
  monitor_pid = pid;
  close(listener);
  LUPINE_LOG_DEBUG("Metrics endpoint listening on port " << port
                                                           << " (pid " << pid
                                                           << ")");
  return 0;
}

void lupine_monitoring_stop() {
  if (monitor_pid > 0) {
    (void)kill(monitor_pid, SIGTERM);
    int status = 0;
    while (waitpid(monitor_pid, &status, 0) < 0 && errno == EINTR) {
    }
    monitor_pid = -1;
  }
  if (registry != nullptr) {
    pthread_mutex_destroy(&registry->host_pid_discovery_mutex);
    munmap(registry, sizeof(*registry));
    registry = nullptr;
  }
}

void lupine_monitoring_note_process_exit(int64_t pid) {
  if (pid > 0 && pid == static_cast<int64_t>(monitor_pid)) {
    monitor_pid = -1;
  }
}

int lupine_monitoring_register_child() {
  if (registry == nullptr || registry->magic != kRegistryMagic ||
      registry->version != kRegistryVersion) {
    return -1;
  }
  for (size_t i = 0; i < kMaxServerChildren; ++i) {
    monitor_slot *slot = &registry->slots[i];
    if (!atomic_compare_exchange(&slot->state, kSlotFree, kSlotActive)) {
      continue;
    }
    __atomic_fetch_add(&slot->sequence, 1, __ATOMIC_ACQ_REL);
    slot->server_pid = static_cast<int32_t>(getpid());
    slot->host_pid = 0;
    memset(slot->client_address, 0, sizeof(slot->client_address));
    slot->metadata = {};
    __atomic_fetch_add(&slot->sequence, 1, __ATOMIC_RELEASE);
    return static_cast<int>(i);
  }
  atomic_increment(&registry->dropped_connections);
  return -1;
}

void lupine_monitoring_unregister_child(int slot_index) {
  if (registry == nullptr || slot_index < 0 ||
      slot_index >= static_cast<int>(kMaxServerChildren)) {
    return;
  }
  monitor_slot *slot = &registry->slots[slot_index];
  atomic_store(&slot->state, kSlotFree);
}

int handle_lupine_client_metadata(conn_t *conn) {
  lupine_client_metadata_header header = {};
  if (conn == nullptr ||
      rpc_read(conn, &header, sizeof(header)) != sizeof(header)) {
    return -1;
  }
  int status = 0;
  lupine_client_metadata_v2 metadata = {};
  if (header.payload_size > LUPINE_CLIENT_METADATA_MAX_PAYLOAD) {
    status = 2;
    if (rpc_drain(conn, header.payload_size) < 0) {
      return -1;
    }
  } else {
    std::vector<unsigned char> payload(header.payload_size);
    if (header.payload_size != 0 &&
        rpc_read(conn, payload.data(), payload.size()) !=
            static_cast<int>(payload.size())) {
      return -1;
    }
    if (header.version != LUPINE_CLIENT_METADATA_VERSION ||
        payload.size() < sizeof(metadata)) {
      status = 1;
    } else {
      memcpy(&metadata, payload.data(), sizeof(metadata));
      terminate_strings(&metadata);
      std::string address = peer_address(conn->connfd);
      store_metadata(conn->monitor_slot, metadata, address.c_str());
    }
  }
  int request_id = rpc_read_end(conn);
  if (request_id < 0 || rpc_write_start_response(conn, request_id) < 0 ||
      rpc_write(conn, &status, sizeof(status)) < 0 || rpc_write_end(conn) < 0) {
    return -1;
  }
  return 0;
}

void lupine_monitoring_begin_context_create(conn_t *conn, int cuda_device) {
  begin_context_probe(conn, cuda_device);
}

void lupine_monitoring_end_context_create(conn_t *conn,
                                          bool context_created) {
  end_context_probe(conn, context_created);
}
