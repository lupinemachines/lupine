#include <atomic>
#include <cstdint>
#include <cstring>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <utility>
#include <vector>

#include <hip/hip_runtime_api.h>

#include "codegen/gen_rpc_ids.h"
#include "device_stdout.h"
#include "hip_fatbin.h"
#include "lupine_log.h"
#include "rpc.h"
#include "transport.h"

namespace {

struct lupine_hip_remote_device {
  unsigned int conn_index = 0;
  int remote_ordinal = 0;
};

std::mutex devices_mutex;
std::vector<lupine_hip_remote_device> devices;
bool devices_ready = false;
uint64_t devices_generation = 0;
std::atomic<uint64_t> connection_generation{1};

hipError_t rpc_error() { return hipErrorUnknown; }

void *hip_transport_dispatch(void *argument) {
  auto *connection = static_cast<conn_t *>(argument);
  while (!connection->closed) {
    int op = rpc_dispatch(connection, 1);
    if (op < 0 || connection->closed || rpc_read_end(connection) < 0) {
      break;
    }
  }
  return nullptr;
}

// Teardown can hold the transport mutex while invoking this callback. An
// atomic generation invalidates the cache without taking devices_mutex in the
// reverse order; the next device-scoped call rebuilds the ordinal table.
void hip_connection_closed(conn_t *) {
  connection_generation.fetch_add(1, std::memory_order_acq_rel);
}

struct hip_transport_shutdown {
  ~hip_transport_shutdown() { lupine_client_transport_close(); }
};

const lupine_client_transport_config &hip_transport_config() {
  static hip_transport_shutdown shutdown;
  static const auto config = [] {
    lupine_client_transport_config config;
    config.dial_policy = lupine_client_dial_policy::single_attempt;
    config.strict_endpoints = true;
    config.log_missing_server = true;
    config.dispatch = hip_transport_dispatch;
    config.connection_closed = hip_connection_closed;
    return config;
  }();
  (void)shutdown;
  return config;
}

void hip_retire_thread_lane(uint64_t lane_id) {
  lupine_client_transport_retire_lane(lane_id);
}

pthread_once_t hip_rpc_lifecycle_once = PTHREAD_ONCE_INIT;

void install_rpc_lifecycle_hooks() {
  const rpc_lifecycle_hooks hooks = {hip_connection_closed,
                                     hip_retire_thread_lane, nullptr};
  if (rpc_set_lifecycle_hooks(&hooks) < 0) {
    LUPINE_LOG_ERROR("Failed to install HIP RPC lifecycle hooks");
  }
}

int open_connections() {
  if (pthread_once(&hip_rpc_lifecycle_once, install_rpc_lifecycle_hooks) != 0) {
    return -1;
  }
  return lupine_client_transport_open(hip_transport_config());
}

hipError_t call_int_out_on(conn_t *conn, int op, int *value) {
  hipError_t result = rpc_error();
  int remote_value = 0;
  if (conn == nullptr || rpc_write_start_request(conn, op) < 0 ||
      rpc_wait_for_response(conn) < 0 ||
      rpc_read(conn, &remote_value, sizeof(remote_value)) < 0 ||
      rpc_read(conn, &result, sizeof(result)) < 0 || rpc_read_end(conn) < 0) {
    return rpc_error();
  }
  if (value != nullptr) {
    *value = remote_value;
  }
  return result;
}

hipError_t call_device_get_on(conn_t *conn, int ordinal, int *device) {
  hipError_t result = rpc_error();
  int remote_device = 0;
  if (conn == nullptr || rpc_write_start_request(conn, RPC_hipDeviceGet) < 0 ||
      rpc_write(conn, &ordinal, sizeof(ordinal)) < 0 ||
      rpc_wait_for_response(conn) < 0 ||
      rpc_read(conn, &remote_device, sizeof(remote_device)) < 0 ||
      rpc_read(conn, &result, sizeof(result)) < 0 || rpc_read_end(conn) < 0) {
    return rpc_error();
  }
  if (device != nullptr) {
    *device = remote_device;
  }
  return result;
}

// devices_mutex must be held by the caller. Keeping discovery under this lock
// makes the virtual ordinal table immutable while another thread is routing a
// device-scoped request.
hipError_t ensure_devices_locked() {
  if (open_connections() < 0) {
    return rpc_error();
  }
  uint64_t generation = connection_generation.load(std::memory_order_acquire);
  if (devices_ready && devices_generation == generation) {
    return hipSuccess;
  }

  std::vector<lupine_hip_remote_device> discovered;
  unsigned int connection_count = lupine_client_transport_size();
  for (unsigned int i = 0; i < connection_count; ++i) {
    conn_t *conn = lupine_client_transport_connection(i);
    int count = 0;
    hipError_t result = call_int_out_on(conn, RPC_hipGetDeviceCount, &count);
    if (result != hipSuccess) {
      return result;
    }
    if (count < 0) {
      return rpc_error();
    }
    for (int ordinal = 0; ordinal < count; ++ordinal) {
      int remote_device = 0;
      result = call_device_get_on(conn, ordinal, &remote_device);
      if (result != hipSuccess) {
        return result;
      }
      discovered.push_back({i, remote_device});
    }
  }
  if (connection_generation.load(std::memory_order_acquire) != generation) {
    return rpc_error();
  }
  devices = std::move(discovered);
  devices_generation = generation;
  devices_ready = true;
  return hipSuccess;
}

// The virtual ordinal hipSetDevice last selected on this thread; every
// generated call without its own routing key goes to that device's server.
thread_local int current_device = 0;

conn_t *connection_for_device(int *device) {
  if (device == nullptr) {
    return nullptr;
  }

  std::lock_guard<std::mutex> lock(devices_mutex);
  if (ensure_devices_locked() != hipSuccess || devices.empty()) {
    return nullptr;
  }
  int virtual_ordinal = *device;
  if (virtual_ordinal < 0 ||
      virtual_ordinal >= static_cast<int>(devices.size())) {
    return nullptr;
  }
  lupine_hip_remote_device mapped = devices[virtual_ordinal];
  *device = mapped.remote_ordinal;
  return lupine_client_transport_connection(mapped.conn_index);
}

conn_t *connection() {
  int device = current_device;
  return connection_for_device(&device);
}

} // namespace

#include "codegen/gen_hip_client.inc"

extern "C" hipError_t hipInit(unsigned int flags) {
  std::lock_guard<std::mutex> lock(devices_mutex);
  if (open_connections() < 0) {
    return rpc_error();
  }

  hipError_t first_error = hipSuccess;
  unsigned int connection_count = lupine_client_transport_size();
  for (unsigned int i = 0; i < connection_count; ++i) {
    conn_t *conn = lupine_client_transport_connection(i);
    hipError_t result = rpc_error();
    if (rpc_write_start_request(conn, RPC_hipInit) < 0 ||
        rpc_write(conn, &flags, sizeof(flags)) < 0 ||
        rpc_wait_for_response(conn) < 0 ||
        rpc_read(conn, &result, sizeof(result)) < 0 || rpc_read_end(conn) < 0) {
      result = rpc_error();
    }
    if (result != hipSuccess && first_error == hipSuccess) {
      first_error = result;
    }
  }

  devices_ready = false;
  devices.clear();
  return first_error;
}

extern "C" hipError_t hipGetDeviceCount(int *count) {
  if (count == nullptr) {
    return hipErrorInvalidValue;
  }
  std::lock_guard<std::mutex> lock(devices_mutex);
  hipError_t result = ensure_devices_locked();
  if (result != hipSuccess) {
    return result;
  }
  *count = static_cast<int>(devices.size());
  return hipSuccess;
}

extern "C" hipError_t hipDeviceGet(int *device, int ordinal) {
  if (device == nullptr) {
    return hipErrorInvalidValue;
  }
  std::lock_guard<std::mutex> lock(devices_mutex);
  hipError_t result = ensure_devices_locked();
  if (result != hipSuccess) {
    return result;
  }
  if (ordinal < 0 || ordinal >= static_cast<int>(devices.size())) {
    return hipErrorInvalidDevice;
  }
  *device = ordinal;
  return hipSuccess;
}

extern "C" hipError_t hipSetDevice(int deviceId) {
  int remote_device = deviceId;
  conn_t *conn = connection_for_device(&remote_device);
  if (conn == nullptr) {
    return hipErrorInvalidDevice;
  }
  hipError_t result = rpc_error();
  if (rpc_write_start_request(conn, RPC_hipSetDevice) < 0 ||
      rpc_write(conn, &remote_device, sizeof(remote_device)) < 0 ||
      rpc_wait_for_response(conn) < 0 ||
      rpc_read(conn, &result, sizeof(result)) < 0 || rpc_read_end(conn) < 0) {
    return rpc_error();
  }
  if (result == hipSuccess) {
    current_device = deviceId;
  }
  return result;
}

extern "C" hipError_t hipGetDevice(int *deviceId) {
  if (deviceId == nullptr) {
    return hipErrorInvalidValue;
  }
  *deviceId = current_device;
  return hipSuccess;
}

extern "C" hipError_t hipDeviceReset() {
  conn_t *conn = connection();
  hipError_t result = rpc_error();
  if (rpc_write_start_request(conn, RPC_hipDeviceReset) < 0 ||
      rpc_wait_for_response(conn) < 0 ||
      rpc_read(conn, &result, sizeof(result)) < 0 || rpc_read_end(conn) < 0) {
    return rpc_error();
  }
  if (result == hipSuccess) {
    connection_generation.fetch_add(1, std::memory_order_acq_rel);
  }
  return result;
}

namespace {

struct hip_launch_configuration {
  dim3 grid;
  dim3 block;
  size_t shared_memory;
  hipStream_t stream;
};

thread_local std::vector<hip_launch_configuration> launch_configurations;

struct hip_remote_module {
  uint64_t generation = 0;
  hipModule_t module = nullptr;
  std::map<std::string, hipFunction_t> functions;
};

struct hip_registration {
  const void *image;
  std::map<int, hip_remote_module> modules;
};

struct hip_registered_kernel {
  hip_registration *registration;
  std::string name;
};

struct hip_registrations {
  std::mutex mutex;
  std::map<void **, std::unique_ptr<hip_registration>> images;
  std::map<const void *, hip_registered_kernel> kernels;
};

hip_registrations &registrations() {
  // Compiler constructors/destructors can run before/after this DSO's own
  // globals. No transport or native HIP library is needed for registration.
  static auto *state = new hip_registrations;
  return *state;
}

const char *error_text(hipError_t error, bool name) {
  static std::mutex mutex;
  static std::map<std::pair<int, bool>, std::string> cache;
  std::lock_guard<std::mutex> lock(mutex);
  auto key = std::make_pair(static_cast<int>(error), name);
  auto found = cache.find(key);
  if (found != cache.end()) {
    return found->second.c_str();
  }
  // Error descriptions don't require a GPU or a selected device.
  if (open_connections() == 0) {
    conn_t *conn = lupine_client_transport_connection(0);
    uint64_t size = 0;
    int op = name ? RPC_hipGetErrorName : RPC_hipGetErrorString;
    if (rpc_write_start_request(conn, op) >= 0 &&
        rpc_write(conn, &error, sizeof(error)) >= 0 &&
        rpc_wait_for_response(conn) >= 0 &&
        rpc_read(conn, &size, sizeof(size)) >= 0 && size > 0 && size <= 4096) {
      std::string text(static_cast<size_t>(size), '\0');
      if (rpc_read(conn, text.data(), text.size()) >= 0 &&
          rpc_read_end(conn) >= 0 && text.back() == '\0') {
        return cache.emplace(key, std::move(text)).first->second.c_str();
      }
    }
  }
  // In particular, reporting a transport failure must always yield a valid
  // string, even when there is no server to supply the native description.
  if (error == hipSuccess) {
    return name ? "hipSuccess" : "no error";
  }
  return name ? "hipErrorUnknown" : "unknown error";
}

} // namespace

extern "C" const char *hipGetErrorString(hipError_t error) {
  return error_text(error, false);
}

extern "C" const char *hipGetErrorName(hipError_t error) {
  return error_text(error, true);
}

extern "C" void **__hipRegisterFatBinary(const void *data) {
  struct wrapper {
    uint32_t magic;
    uint32_t version;
    const void *image;
    void *reserved;
  };
  const auto *fatbin = static_cast<const wrapper *>(data);
  if (fatbin == nullptr || fatbin->magic != 0x48495046 ||
      fatbin->version != 1 || fatbin->image == nullptr) {
    return nullptr;
  }
  auto registration = std::make_unique<hip_registration>();
  registration->image = fatbin->image;
  auto handle = reinterpret_cast<void **>(registration.get());
  auto &state = registrations();
  std::lock_guard<std::mutex> lock(state.mutex);
  state.images.emplace(handle, std::move(registration));
  return handle;
}

extern "C" void __hipRegisterFunction(void **handle, const void *host_function,
                                      char *, const char *device_name,
                                      unsigned int, uint3 *, uint3 *, dim3 *,
                                      dim3 *, int *) {
  if (host_function == nullptr || device_name == nullptr) {
    return;
  }
  auto &state = registrations();
  std::lock_guard<std::mutex> lock(state.mutex);
  auto image = state.images.find(handle);
  if (image != state.images.end()) {
    state.kernels.insert_or_assign(
        host_function, hip_registered_kernel{image->second.get(), device_name});
  }
}

extern "C" void __hipUnregisterFatBinary(void **handle) {
  auto &state = registrations();
  std::lock_guard<std::mutex> lock(state.mutex);
  auto image = state.images.find(handle);
  if (image == state.images.end()) {
    return;
  }
  for (auto kernel = state.kernels.begin(); kernel != state.kernels.end();) {
    if (kernel->second.registration == image->second.get()) {
      kernel = state.kernels.erase(kernel);
    } else {
      ++kernel;
    }
  }
  // The connection's server process owns the uploaded modules until session
  // teardown. An ELF destructor must not reconnect to unload a dead session.
  state.images.erase(image);
}

extern "C" hipError_t __hipPushCallConfiguration(dim3 grid, dim3 block,
                                                 size_t shared_memory,
                                                 hipStream_t stream) {
  launch_configurations.push_back({grid, block, shared_memory, stream});
  return hipSuccess;
}

extern "C" hipError_t __hipPopCallConfiguration(dim3 *grid, dim3 *block,
                                                size_t *shared_memory,
                                                hipStream_t *stream) {
  if (grid == nullptr || block == nullptr || shared_memory == nullptr ||
      stream == nullptr || launch_configurations.empty()) {
    return hipErrorInvalidValue;
  }
  auto config = launch_configurations.back();
  launch_configurations.pop_back();
  *grid = config.grid;
  *block = config.block;
  *shared_memory = config.shared_memory;
  *stream = config.stream;
  return hipSuccess;
}

extern "C" hipError_t hipModuleLoadData(hipModule_t *module,
                                        const void *image) {
  if (module == nullptr || image == nullptr) {
    return hipErrorInvalidValue;
  }
  uint64_t size = lupine_hip_bundle_size(image);
  if (size == 0) {
    return hipErrorInvalidImage;
  }
  conn_t *conn = connection();
  hipModule_t remote_module = nullptr;
  hipError_t result = rpc_error();
  if (rpc_write_start_request(conn, RPC_hipModuleLoadData) < 0 ||
      rpc_write(conn, &size, sizeof(size)) < 0 ||
      rpc_write(conn, image, static_cast<size_t>(size)) < 0 ||
      rpc_wait_for_response(conn) < 0 ||
      rpc_read(conn, &remote_module, sizeof(remote_module)) < 0 ||
      rpc_read(conn, &result, sizeof(result)) < 0 || rpc_read_end(conn) < 0) {
    return rpc_error();
  }
  if (result == hipSuccess) {
    *module = remote_module;
  }
  return result;
}

extern "C" hipError_t
hipModuleLaunchKernel(hipFunction_t function, unsigned int grid_x,
                      unsigned int grid_y, unsigned int grid_z,
                      unsigned int block_x, unsigned int block_y,
                      unsigned int block_z, unsigned int shared_memory,
                      hipStream_t stream, void **kernel_params, void **extra) {
  if (kernel_params != nullptr || extra != nullptr) {
    return hipErrorNotSupported;
  }
  const unsigned int dimensions[] = {grid_x,  grid_y,  grid_z,
                                     block_x, block_y, block_z};
  conn_t *conn = connection();
  hipError_t result = rpc_error();
  if (rpc_write_start_request(conn, RPC_hipModuleLaunchKernel) < 0 ||
      rpc_write(conn, &function, sizeof(function)) < 0 ||
      rpc_write(conn, dimensions, sizeof(dimensions)) < 0 ||
      rpc_write(conn, &shared_memory, sizeof(shared_memory)) < 0 ||
      rpc_write(conn, &stream, sizeof(stream)) < 0 ||
      rpc_wait_for_response(conn) < 0 ||
      rpc_read(conn, &result, sizeof(result)) < 0 ||
      lupine_read_captured_stdout(conn) < 0 || rpc_read_end(conn) < 0) {
    return rpc_error();
  }
  return result;
}

extern "C" hipError_t hipLaunchKernel(const void *host_function, dim3 grid,
                                      dim3 block, void **args,
                                      size_t shared_memory,
                                      hipStream_t stream) {
  if (shared_memory > std::numeric_limits<unsigned int>::max()) {
    return hipErrorInvalidValue;
  }
  // Clang passes a non-null args array even for a zero-argument kernel. The
  // native module launch validates its signature with null kernelParams and
  // rejects kernels requiring arguments; client pointers never cross the wire.
  (void)args;
  auto &state = registrations();
  std::lock_guard<std::mutex> lock(state.mutex);
  auto kernel = state.kernels.find(host_function);
  if (kernel == state.kernels.end()) {
    return hipErrorInvalidDeviceFunction;
  }
  if (connection() == nullptr) {
    return rpc_error();
  }
  uint64_t generation = connection_generation.load(std::memory_order_acquire);
  auto &remote = kernel->second.registration->modules[current_device];
  if (remote.generation != generation) {
    remote = {};
    remote.generation = generation;
  }
  if (remote.module == nullptr) {
    hipError_t result =
        hipModuleLoadData(&remote.module, kernel->second.registration->image);
    if (result != hipSuccess) {
      return result;
    }
  }
  auto &function = remote.functions[kernel->second.name];
  if (function == nullptr) {
    hipError_t result = hipModuleGetFunction(&function, remote.module,
                                             kernel->second.name.c_str());
    if (result != hipSuccess) {
      return result;
    }
  }
  return hipModuleLaunchKernel(function, grid.x, grid.y, grid.z, block.x,
                               block.y, block.z, shared_memory, stream, nullptr,
                               nullptr);
}

// Allows the server to reject an accidental attempt to load this shim as its
// native backend (for example, an inherited client LD_LIBRARY_PATH).
extern "C" int lupine_hip_client_marker() { return 1; }
