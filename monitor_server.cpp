#include "monitoring.h"

#ifdef LUPINE_MONITORING_ENABLED

#include <algorithm>
#include <array>
#include <cerrno>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <map>
#include <memory>
#include <new>
#include <set>
#include <sstream>
#include <string>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>

#include <arpa/inet.h>
#include <cuda.h>
#include <netinet/in.h>
#include <nvml.h>
#include <pthread.h>
#include <sys/mman.h>
#include <sys/socket.h>
#include <unistd.h>

#include "lupine_log.h"
#include "nvml_runtime.h"

#define LUPINE_STRINGIFY_INNER(value) #value
#define LUPINE_STRINGIFY(value) LUPINE_STRINGIFY_INNER(value)
#define LUPINE_NVML_SYMBOL(function)                                           \
  lupine_nvml_symbol<decltype(&function)>(LUPINE_STRINGIFY(function))

namespace {

constexpr uint32_t kRegistryMagic = 0x4c504d52;
constexpr uint32_t kRegistryVersion = 1;
constexpr size_t kMaxServerChildren = 4096;
constexpr uint64_t kProcessUtilizationWindowUs = 2 * 1000 * 1000;
constexpr uint32_t kSlotFree = 0;
constexpr uint32_t kSlotActive = 1;

struct alignas(64) monitor_slot {
  uint32_t state;
  int32_t server_pid;
  int32_t host_pid;
  char client_address[64];
  lupine_client_metadata metadata;
};

struct monitor_registry {
  uint32_t magic;
  uint32_t version;
  uint64_t dropped_connections;
  uint64_t host_pid_discovery_failures;
  uint64_t host_pid_discovery_ambiguities;
  pthread_mutex_t registry_mutex;
  pthread_mutex_t host_pid_discovery_mutex;
  monitor_slot slots[kMaxServerChildren];
};

struct child_snapshot {
  int32_t server_pid = 0;
  int32_t host_pid = 0;
  char client_address[64] = {};
  lupine_client_metadata metadata = {};
};

monitor_registry *registry = nullptr;
int child_slot = -1;

bool lock_shared_mutex(pthread_mutex_t *mutex) {
  int status = pthread_mutex_lock(mutex);
  if (status == EOWNERDEAD) {
    if (pthread_mutex_consistent(mutex) == 0) {
      return true;
    }
    pthread_mutex_unlock(mutex);
    return false;
  }
  return status == 0;
}

uint64_t current_time_us() {
  auto now = std::chrono::system_clock::now().time_since_epoch();
  return static_cast<uint64_t>(
      std::chrono::duration_cast<std::chrono::microseconds>(now).count());
}

void terminate_strings(lupine_client_metadata *metadata) {
  metadata->connection_kind[sizeof(metadata->connection_kind) - 1] = '\0';
  metadata->client_process_name[sizeof(metadata->client_process_name) - 1] =
      '\0';
  metadata->client_hostname[sizeof(metadata->client_hostname) - 1] = '\0';
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

bool store_metadata(int slot_index, const lupine_client_metadata &metadata,
                    const char *client_address) {
  if (registry == nullptr || slot_index < 0 ||
      slot_index >= static_cast<int>(kMaxServerChildren)) {
    return false;
  }
  if (!lock_shared_mutex(&registry->registry_mutex)) {
    return false;
  }
  monitor_slot *slot = &registry->slots[slot_index];
  bool stored = slot->state == kSlotActive;
  if (stored) {
    slot->metadata = metadata;
    memset(slot->client_address, 0, sizeof(slot->client_address));
    if (client_address != nullptr) {
      size_t length = strnlen(client_address, sizeof(slot->client_address) - 1);
      memcpy(slot->client_address, client_address, length);
    }
  }
  pthread_mutex_unlock(&registry->registry_mutex);
  return stored;
}

bool store_host_pid(int slot_index, int32_t host_pid) {
  if (registry == nullptr || slot_index < 0 || host_pid <= 0 ||
      slot_index >= static_cast<int>(kMaxServerChildren)) {
    return false;
  }
  if (!lock_shared_mutex(&registry->registry_mutex)) {
    return false;
  }
  monitor_slot *slot = &registry->slots[slot_index];
  bool stored = slot->state == kSlotActive;
  if (stored) {
    slot->host_pid = host_pid;
  }
  pthread_mutex_unlock(&registry->registry_mutex);
  return stored;
}

struct registry_snapshot {
  std::vector<child_snapshot> children;
  uint64_t dropped_connections = 0;
  uint64_t host_pid_discovery_failures = 0;
  uint64_t host_pid_discovery_ambiguities = 0;
};

registry_snapshot snapshot_registry() {
  registry_snapshot result;
  if (registry == nullptr) {
    return result;
  }
  if (!lock_shared_mutex(&registry->registry_mutex)) {
    return result;
  }
  result.dropped_connections = registry->dropped_connections;
  result.host_pid_discovery_failures = registry->host_pid_discovery_failures;
  result.host_pid_discovery_ambiguities =
      registry->host_pid_discovery_ambiguities;
  for (size_t index = 0; index < kMaxServerChildren; ++index) {
    const monitor_slot &slot = registry->slots[index];
    if (slot.state != kSlotActive || slot.server_pid <= 0) {
      continue;
    }
    child_snapshot snapshot;
    snapshot.server_pid = slot.server_pid;
    snapshot.host_pid = slot.host_pid;
    memcpy(snapshot.client_address, slot.client_address,
           sizeof(snapshot.client_address));
    snapshot.metadata = slot.metadata;
    terminate_strings(&snapshot.metadata);
    snapshot.client_address[sizeof(snapshot.client_address) - 1] = '\0';
    result.children.push_back(snapshot);
  }
  pthread_mutex_unlock(&registry->registry_mutex);
  return result;
}

std::string prometheus_escape(const char *value) {
  std::string result;
  if (value == nullptr) {
    return result;
  }
  for (const unsigned char *cursor =
           reinterpret_cast<const unsigned char *>(value);
       *cursor != '\0'; ++cursor) {
    if (*cursor == '\\') {
      result += "\\\\";
    } else if (*cursor == '"') {
      result += "\\\"";
    } else if (*cursor == '\n') {
      result += "\\n";
    } else if (*cursor >= 0x20 && *cursor != 0x7f) {
      result.push_back(static_cast<char>(*cursor));
    }
  }
  return result;
}

std::string label(const char *name, const std::string &value) {
  return std::string(name) + "=\"" + prometheus_escape(value.c_str()) + "\"";
}

class nvml_session {
public:
  nvml_session() {
    auto init = LUPINE_NVML_SYMBOL(nvmlInit_v2);
    active_ = init != nullptr && init() == NVML_SUCCESS;
  }
  ~nvml_session() {
    if (active_) {
      auto shutdown = LUPINE_NVML_SYMBOL(nvmlShutdown);
      if (shutdown != nullptr) {
        (void)shutdown();
      }
    }
  }
  bool active() const { return active_; }

private:
  bool active_;
};

std::vector<nvmlProcessInfo_t> running_processes(nvmlDevice_t device,
                                                 bool *query_ok = nullptr) {
  if (query_ok != nullptr) {
    *query_ok = false;
  }
  auto get_processes = LUPINE_NVML_SYMBOL(nvmlDeviceGetComputeRunningProcesses);
  if (get_processes == nullptr) {
    return {};
  }
  for (int attempt = 0; attempt < 4; ++attempt) {
    unsigned int count = 0;
    nvmlReturn_t status = get_processes(device, &count, nullptr);
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
    status = get_processes(device, &returned, processes.data());
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
  return lock_shared_mutex(&registry->host_pid_discovery_mutex);
}

bool slot_has_host_pid(int slot_index) {
  if (registry == nullptr || slot_index < 0 ||
      slot_index >= static_cast<int>(kMaxServerChildren) ||
      !lock_shared_mutex(&registry->registry_mutex)) {
    return false;
  }
  const monitor_slot &slot = registry->slots[slot_index];
  bool found = slot.state == kSlotActive && slot.host_pid > 0;
  pthread_mutex_unlock(&registry->registry_mutex);
  return found;
}

bool host_pid_owned_by_other_slot(int32_t host_pid, int slot_index) {
  if (registry == nullptr || host_pid <= 0 ||
      !lock_shared_mutex(&registry->registry_mutex)) {
    return false;
  }
  bool found = false;
  for (size_t index = 0; index < kMaxServerChildren; ++index) {
    const monitor_slot &slot = registry->slots[index];
    if (static_cast<int>(index) == slot_index || slot.state != kSlotActive) {
      continue;
    }
    if (slot.host_pid == host_pid) {
      found = true;
      break;
    }
  }
  pthread_mutex_unlock(&registry->registry_mutex);
  return found;
}

struct host_pid_probe {
  int slot_index = -1;
  nvml_session session;
  nvmlDevice_t device = nullptr;
  std::set<unsigned int> before_pids;
  bool lock_held = false;
};

thread_local std::unique_ptr<host_pid_probe> active_host_pid_probe;

void note_discovery_failure() {
  if (registry != nullptr && lock_shared_mutex(&registry->registry_mutex)) {
    ++registry->host_pid_discovery_failures;
    pthread_mutex_unlock(&registry->registry_mutex);
  }
}

void begin_context_probe(int cuda_device) {
  if (registry == nullptr || child_slot < 0 || slot_has_host_pid(child_slot) ||
      active_host_pid_probe != nullptr) {
    return;
  }
  std::unique_ptr<host_pid_probe> probe(new (std::nothrow) host_pid_probe());
  if (probe == nullptr || !probe->session.active()) {
    note_discovery_failure();
    return;
  }
  char pci_bus_id[64] = {};
  auto get_device = LUPINE_NVML_SYMBOL(nvmlDeviceGetHandleByPciBusId);
  if (cuDeviceGetPCIBusId(pci_bus_id, sizeof(pci_bus_id), cuda_device) !=
          CUDA_SUCCESS ||
      get_device == nullptr ||
      get_device(pci_bus_id, &probe->device) != NVML_SUCCESS ||
      probe->device == nullptr) {
    note_discovery_failure();
    return;
  }
  if (!lock_host_pid_discovery()) {
    note_discovery_failure();
    return;
  }
  probe->lock_held = true;
  if (slot_has_host_pid(child_slot)) {
    pthread_mutex_unlock(&registry->host_pid_discovery_mutex);
    return;
  }
  bool query_ok = false;
  std::vector<nvmlProcessInfo_t> before =
      running_processes(probe->device, &query_ok);
  if (!query_ok) {
    pthread_mutex_unlock(&registry->host_pid_discovery_mutex);
    note_discovery_failure();
    return;
  }
  for (const auto &process : before) {
    probe->before_pids.insert(process.pid);
  }
  probe->slot_index = child_slot;
  active_host_pid_probe = std::move(probe);
}

void end_context_probe(bool context_created) {
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
          running_processes(probe->device, &query_ok);
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
    if (ambiguous && registry != nullptr &&
        lock_shared_mutex(&registry->registry_mutex)) {
      ++registry->host_pid_discovery_ambiguities;
      pthread_mutex_unlock(&registry->registry_mutex);
    }
    note_discovery_failure();
  }
}

using process_util_map = std::unordered_map<unsigned int, unsigned int>;

struct process_utilization_result {
  process_util_map samples;
  bool supported = false;
  bool valid = false;
};

process_utilization_result process_utilization(nvmlDevice_t device) {
  process_utilization_result result;
  auto get_utilization = LUPINE_NVML_SYMBOL(nvmlDeviceGetProcessUtilization);
  if (get_utilization == nullptr) {
    return result;
  }
  uint64_t now = current_time_us();
  uint64_t last_seen =
      now > kProcessUtilizationWindowUs ? now - kProcessUtilizationWindowUs : 0;
  std::vector<nvmlProcessUtilizationSample_t> samples;
  for (int attempt = 0; attempt < 4; ++attempt) {
    unsigned int count = 0;
    nvmlReturn_t status = get_utilization(device, nullptr, &count, last_seen);
    if (status == NVML_ERROR_NOT_SUPPORTED ||
        status == NVML_ERROR_FUNCTION_NOT_FOUND) {
      return result;
    }
    result.supported = true;
    if (status == NVML_ERROR_NOT_FOUND ||
        (status == NVML_SUCCESS && count == 0)) {
      result.valid = true;
      return result;
    }
    if (status != NVML_ERROR_INSUFFICIENT_SIZE || count == 0) {
      return result;
    }
    count = std::min<unsigned int>(count + 16, 65536);
    samples.resize(count);
    unsigned int returned = count;
    status = get_utilization(device, samples.data(), &returned, last_seen);
    if (status == NVML_SUCCESS) {
      samples.resize(std::min<unsigned int>(returned, samples.size()));
      result.valid = true;
      break;
    }
    if (status == NVML_ERROR_NOT_FOUND) {
      result.valid = true;
      samples.clear();
      return result;
    }
    if (status != NVML_ERROR_INSUFFICIENT_SIZE) {
      return result;
    }
  }
  if (!result.valid) {
    return result;
  }
  std::unordered_map<unsigned int, std::pair<unsigned long long, unsigned int>>
      latest;
  for (const auto &sample : samples) {
    auto &entry = latest[sample.pid];
    if (sample.timeStamp >= entry.first) {
      entry = {sample.timeStamp, sample.smUtil};
    }
  }
  for (const auto &entry : latest) {
    result.samples.emplace(entry.first, entry.second.second);
  }
  return result;
}

struct device_snapshot {
  unsigned int index = 0;
  std::string uuid;
  std::string name;
  unsigned long long memory_total = 0;
  unsigned long long memory_used = 0;
  unsigned int utilization = 0;
  bool memory_valid = false;
  bool utilization_valid = false;
  bool processes_valid = false;
  bool process_utilization_supported = false;
  bool process_utilization_valid = false;
  std::vector<nvmlProcessInfo_t> processes;
  process_util_map process_utils;
};

struct client_key {
  std::string client_id;
  std::string client_address;
  std::string client_hostname;
  std::string client_process_name;
  unsigned int device_index = 0;
  std::string device_uuid;

  bool operator<(const client_key &other) const {
    return std::tie(client_id, client_address, client_hostname,
                    client_process_name, device_index, device_uuid) <
           std::tie(other.client_id, other.client_address,
                    other.client_hostname, other.client_process_name,
                    other.device_index, other.device_uuid);
  }
};

struct process_sample {
  unsigned long long memory_used = 0;
  unsigned int sm_percent = 0;
  bool memory_valid = false;
  bool utilization_valid = false;
};

std::string client_id_for(const child_snapshot &child) {
  const auto &metadata = child.metadata;
  std::ostringstream out;
  if (metadata.client_pid > 0) {
    out << child.client_address << ':'
        << (metadata.client_hostname[0] == '\0' ? "unknown"
                                                : metadata.client_hostname);
    if (metadata.client_pid_namespace_inode > 0) {
      out << ':' << metadata.client_pid_namespace_inode;
    }
    out << ':' << metadata.client_pid;
    return out.str();
  }
  return std::string("connection:") + child.client_address + ':' +
         std::to_string(child.server_pid);
}

client_key client_key_for(const child_snapshot &child,
                          const device_snapshot &device) {
  client_key key;
  key.client_id = client_id_for(child);
  key.client_address =
      child.client_address[0] == '\0' ? "unknown" : child.client_address;
  key.client_hostname = child.metadata.client_hostname[0] == '\0'
                            ? "unknown"
                            : child.metadata.client_hostname;
  key.client_process_name = child.metadata.client_process_name[0] == '\0'
                                ? "unknown"
                                : child.metadata.client_process_name;
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
      << label("device_uuid", key.device_uuid) << ',' << "device_index=\""
      << key.device_index << "\"";
  return out.str();
}

} // namespace

bool lupine_monitoring_initialize() {
  if (registry != nullptr) {
    return true;
  }
  void *memory = mmap(nullptr, sizeof(monitor_registry), PROT_READ | PROT_WRITE,
                      MAP_SHARED | MAP_ANONYMOUS, -1, 0);
  if (memory == MAP_FAILED) {
    LUPINE_LOG_ERROR(
        "Unable to allocate monitoring registry: " << strerror(errno));
    return false;
  }
  registry = static_cast<monitor_registry *>(memory);
  memset(registry, 0, sizeof(*registry));
  registry->magic = kRegistryMagic;
  registry->version = kRegistryVersion;

  pthread_mutexattr_t attributes;
  int status = pthread_mutexattr_init(&attributes);
  bool initialized = status == 0;
  if (status == 0) {
    status = pthread_mutexattr_setpshared(&attributes, PTHREAD_PROCESS_SHARED);
  }
  if (status == 0) {
    status = pthread_mutexattr_setrobust(&attributes, PTHREAD_MUTEX_ROBUST);
  }
  bool registry_mutex_initialized = false;
  if (status == 0) {
    status = pthread_mutex_init(&registry->registry_mutex, &attributes);
    registry_mutex_initialized = status == 0;
  }
  if (status == 0) {
    status =
        pthread_mutex_init(&registry->host_pid_discovery_mutex, &attributes);
  }
  if (initialized) {
    pthread_mutexattr_destroy(&attributes);
  }
  if (status != 0) {
    LUPINE_LOG_ERROR(
        "Unable to initialize monitoring mutex: " << strerror(status));
    if (registry_mutex_initialized) {
      pthread_mutex_destroy(&registry->registry_mutex);
    }
    munmap(registry, sizeof(*registry));
    registry = nullptr;
    return false;
  }
  return true;
}

void lupine_monitoring_shutdown() {
  if (registry == nullptr) {
    return;
  }
  pthread_mutex_destroy(&registry->registry_mutex);
  pthread_mutex_destroy(&registry->host_pid_discovery_mutex);
  munmap(registry, sizeof(*registry));
  registry = nullptr;
}

void lupine_monitoring_register_child() {
  child_slot = -1;
  if (registry == nullptr || registry->magic != kRegistryMagic ||
      registry->version != kRegistryVersion) {
    return;
  }
  if (!lock_shared_mutex(&registry->registry_mutex)) {
    return;
  }
  for (size_t index = 0; index < kMaxServerChildren; ++index) {
    monitor_slot *slot = &registry->slots[index];
    if (slot->state != kSlotFree) {
      continue;
    }
    memset(slot, 0, sizeof(*slot));
    slot->server_pid = static_cast<int32_t>(getpid());
    slot->state = kSlotActive;
    child_slot = static_cast<int>(index);
    pthread_mutex_unlock(&registry->registry_mutex);
    return;
  }
  ++registry->dropped_connections;
  pthread_mutex_unlock(&registry->registry_mutex);
}

void lupine_monitoring_unregister_pid(int64_t server_pid) {
  if (registry == nullptr || server_pid <= 0) {
    return;
  }
  if (!lock_shared_mutex(&registry->registry_mutex)) {
    return;
  }
  for (size_t index = 0; index < kMaxServerChildren; ++index) {
    monitor_slot *slot = &registry->slots[index];
    if (slot->state == kSlotActive && slot->server_pid == server_pid) {
      memset(slot, 0, sizeof(*slot));
      break;
    }
  }
  pthread_mutex_unlock(&registry->registry_mutex);
}

int handle_lupine_client_metadata(conn_t *conn) {
  lupine_client_metadata_header header = {};
  if (conn == nullptr ||
      rpc_read(conn, &header, sizeof(header)) != sizeof(header)) {
    return -1;
  }
  int status = 0;
  lupine_client_metadata metadata = {};
  if (header.payload_size > LUPINE_CLIENT_METADATA_MAX_PAYLOAD) {
    status = 2;
    if (rpc_drain(conn, header.payload_size) < 0) {
      return -1;
    }
  } else {
    std::array<unsigned char, LUPINE_CLIENT_METADATA_MAX_PAYLOAD> payload = {};
    if (header.payload_size != 0 &&
        rpc_read(conn, payload.data(), header.payload_size) !=
            static_cast<int>(header.payload_size)) {
      return -1;
    }
    if (header.version != LUPINE_CLIENT_METADATA_VERSION ||
        header.payload_size < sizeof(metadata)) {
      status = 1;
    } else {
      memcpy(&metadata, payload.data(), sizeof(metadata));
      terminate_strings(&metadata);
      std::string address = peer_address(conn->connfd);
      if (!store_metadata(child_slot, metadata, address.c_str())) {
        status = 3;
      }
    }
  }
  int request_id = rpc_read_end(conn);
  if (request_id < 0 || rpc_write_start_response(conn, request_id) < 0 ||
      rpc_write(conn, &status, sizeof(status)) < 0 || rpc_write_end(conn) < 0) {
    return -1;
  }
  return 0;
}

void lupine_monitoring_begin_context_create(int cuda_device) {
  begin_context_probe(cuda_device);
}

void lupine_monitoring_end_context_create(bool context_created) {
  end_context_probe(context_created);
}

std::string lupine_monitoring_render_metrics() {
  registry_snapshot registry_values = snapshot_registry();
  const std::vector<child_snapshot> &children = registry_values.children;
  nvml_session session;
  std::vector<device_snapshot> devices;

  if (session.active()) {
    auto get_count = LUPINE_NVML_SYMBOL(nvmlDeviceGetCount);
    auto get_handle = LUPINE_NVML_SYMBOL(nvmlDeviceGetHandleByIndex);
    auto get_uuid = LUPINE_NVML_SYMBOL(nvmlDeviceGetUUID);
    auto get_name = LUPINE_NVML_SYMBOL(nvmlDeviceGetName);
    auto get_memory = LUPINE_NVML_SYMBOL(nvmlDeviceGetMemoryInfo);
    auto get_utilization = LUPINE_NVML_SYMBOL(nvmlDeviceGetUtilizationRates);
    unsigned int device_count = 0;
    if (get_count != nullptr && get_handle != nullptr &&
        get_count(&device_count) == NVML_SUCCESS) {
      for (unsigned int index = 0; index < device_count; ++index) {
        nvmlDevice_t handle = nullptr;
        if (get_handle(index, &handle) != NVML_SUCCESS) {
          continue;
        }
        char uuid[NVML_DEVICE_UUID_BUFFER_SIZE] = {};
        char name[NVML_DEVICE_NAME_BUFFER_SIZE] = {};
        nvmlMemory_t memory = {};
        nvmlUtilization_t utilization = {};
        if (get_uuid == nullptr ||
            get_uuid(handle, uuid, sizeof(uuid)) != NVML_SUCCESS) {
          snprintf(uuid, sizeof(uuid), "device-%u", index);
        }
        if (get_name == nullptr ||
            get_name(handle, name, sizeof(name)) != NVML_SUCCESS) {
          snprintf(name, sizeof(name), "unknown");
        }
        device_snapshot device;
        device.index = index;
        device.uuid = uuid;
        device.name = name;
        if (get_memory != nullptr &&
            get_memory(handle, &memory) == NVML_SUCCESS) {
          device.memory_total = memory.total;
          device.memory_used = memory.used;
          device.memory_valid = true;
        }
        if (get_utilization != nullptr &&
            get_utilization(handle, &utilization) == NVML_SUCCESS) {
          device.utilization = utilization.gpu;
          device.utilization_valid = true;
        }
        device.processes = running_processes(handle, &device.processes_valid);
        process_utilization_result process_result = process_utilization(handle);
        device.process_utils = std::move(process_result.samples);
        device.process_utilization_supported = process_result.supported;
        device.process_utilization_valid = process_result.valid;
        devices.push_back(std::move(device));
      }
    }
  }

  std::ostringstream out;
  out << "# HELP lupine_monitor_nvml_up Whether NVML is available.\n"
      << "# TYPE lupine_monitor_nvml_up gauge\n"
      << "lupine_monitor_nvml_up " << (session.active() ? 1 : 0) << "\n"
      << "# HELP lupine_monitor_process_utilization_supported Whether NVML "
         "provides per-process SM utilization for a GPU.\n"
      << "# TYPE lupine_monitor_process_utilization_supported gauge\n"
      << "# HELP lupine_monitor_dropped_connections_total Connections not "
         "tracked because the registry was full.\n"
      << "# TYPE lupine_monitor_dropped_connections_total counter\n"
      << "lupine_monitor_dropped_connections_total "
      << registry_values.dropped_connections << "\n"
      << "# HELP lupine_monitor_host_pid_discovery_failures_total Host PID "
         "discovery attempts that did not resolve a process.\n"
      << "# TYPE lupine_monitor_host_pid_discovery_failures_total counter\n"
      << "lupine_monitor_host_pid_discovery_failures_total "
      << registry_values.host_pid_discovery_failures << "\n"
      << "# HELP lupine_monitor_host_pid_discovery_ambiguities_total Host PID "
         "discoveries with multiple candidates.\n"
      << "# TYPE lupine_monitor_host_pid_discovery_ambiguities_total counter\n"
      << "lupine_monitor_host_pid_discovery_ambiguities_total "
      << registry_values.host_pid_discovery_ambiguities << "\n";

  for (const auto &device : devices) {
    out << "lupine_monitor_process_utilization_supported{"
        << "device_index=\"" << device.index << "\","
        << label("device_uuid", device.uuid) << "} "
        << (device.process_utilization_supported ? 1 : 0) << "\n";
  }

  out << "# HELP lupine_server_connection_info Active server process and "
         "client mapping.\n"
      << "# TYPE lupine_server_connection_info gauge\n";
  for (const auto &child : children) {
    const auto &metadata = child.metadata;
    out << "lupine_server_connection_info{"
        << label("client_id", client_id_for(child)) << ','
        << label("client_address", child.client_address) << ','
        << label("client_hostname", metadata.client_hostname) << ','
        << label("client_name", metadata.client_process_name) << ','
        << "client_pid=\"" << metadata.client_pid << "\","
        << label("connection_kind", metadata.connection_kind) << ','
        << "host_pid=\"" << child.host_pid << "\","
        << "server_pid=\"" << child.server_pid << "\"} 1\n";
  }

  out << "# HELP lupine_host_gpu_memory_total_bytes Total GPU memory in "
         "bytes.\n"
      << "# TYPE lupine_host_gpu_memory_total_bytes gauge\n"
      << "# HELP lupine_host_gpu_memory_used_bytes Used GPU memory in bytes.\n"
      << "# TYPE lupine_host_gpu_memory_used_bytes gauge\n"
      << "# HELP lupine_host_gpu_utilization_percent GPU utilization from 0 "
         "to 100.\n"
      << "# TYPE lupine_host_gpu_utilization_percent gauge\n";
  for (const auto &device : devices) {
    std::string labels = "device_index=\"" + std::to_string(device.index) +
                         "\"," + label("device_type", device.name) + "," +
                         label("device_uuid", device.uuid);
    if (device.memory_valid) {
      out << "lupine_host_gpu_memory_total_bytes{" << labels << "} "
          << device.memory_total << "\n"
          << "lupine_host_gpu_memory_used_bytes{" << labels << "} "
          << device.memory_used << "\n";
    }
    if (device.utilization_valid) {
      out << "lupine_host_gpu_utilization_percent{" << labels << "} "
          << std::min(device.utilization, 100U) << "\n";
    }
  }

  std::unordered_map<int32_t, const child_snapshot *> by_pid;
  for (const auto &child : children) {
    if (child.host_pid > 0) {
      by_pid[child.host_pid] = &child;
    }
  }
  std::map<client_key, process_sample> clients;
  for (const auto &device : devices) {
    if (device.processes_valid) {
      for (const auto &process : device.processes) {
        auto child = by_pid.find(static_cast<int32_t>(process.pid));
        if (child == by_pid.end()) {
          continue;
        }
        auto &sample = clients[client_key_for(*child->second, device)];
        if (process.usedGpuMemory !=
            static_cast<unsigned long long>(NVML_VALUE_NOT_AVAILABLE)) {
          sample.memory_used += process.usedGpuMemory;
          sample.memory_valid = true;
        }
        if (device.process_utilization_valid) {
          sample.utilization_valid = true;
        }
      }
    }
    if (device.process_utilization_valid) {
      for (const auto &utilization : device.process_utils) {
        auto child = by_pid.find(static_cast<int32_t>(utilization.first));
        if (child == by_pid.end()) {
          continue;
        }
        auto &sample = clients[client_key_for(*child->second, device)];
        sample.utilization_valid = true;
        sample.sm_percent = std::min(
            100U, sample.sm_percent + std::min(100U, utilization.second));
      }
    }
  }

  out << "# HELP lupine_client_device_memory_used_bytes GPU memory used by "
         "one client process.\n"
      << "# TYPE lupine_client_device_memory_used_bytes gauge\n"
      << "# HELP lupine_client_device_utilization_percent Client GPU "
         "utilization from 0 to 100.\n"
      << "# TYPE lupine_client_device_utilization_percent gauge\n";
  for (const auto &entry : clients) {
    std::string labels = client_labels(entry.first);
    if (entry.second.memory_valid) {
      out << "lupine_client_device_memory_used_bytes{" << labels << "} "
          << entry.second.memory_used << "\n";
    }
    if (entry.second.utilization_valid) {
      out << "lupine_client_device_utilization_percent{" << labels << "} "
          << entry.second.sm_percent << "\n";
    }
  }
  return out.str();
}

#endif
