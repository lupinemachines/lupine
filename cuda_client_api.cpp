#include "cuda_client_api.h"

#include <cuda.h>

#include <cstring>
#include <mutex>
#include <new>
#include <unordered_map>
#include <vector>

#include "client_routing.h"
#include "rpc.h"

struct lupine_cuda_call {
  int operation = -1;
  int route_id = -2;
  CUcontext context = nullptr;
  conn_t *connection = nullptr;
  std::vector<unsigned char> request;
  bool invocation_started = false;
  bool response_ready = false;
  bool finished = false;
};

namespace {

struct owner_key {
  lupine_cuda_owner_kind kind;
  std::uintptr_t value;

  bool operator==(const owner_key &other) const {
    return kind == other.kind && value == other.value;
  }
};

struct owner_key_hash {
  std::size_t operator()(const owner_key &key) const {
    const auto kind = static_cast<std::size_t>(key.kind);
    const auto value = static_cast<std::size_t>(key.value);
    return value ^ (kind + 0x9e3779b9U + (value << 6U) + (value >> 2U));
  }
};

struct owner_record {
  int route_id = -2;
  CUcontext context = nullptr;
  bool constrained = false;
};

std::mutex &owner_mutex() {
  static auto *mutex = new std::mutex();
  return *mutex;
}

std::unordered_map<owner_key, owner_record, owner_key_hash> &owners() {
  static auto *records =
      new std::unordered_map<owner_key, owner_record, owner_key_hash>();
  return *records;
}

owner_record current_owner() {
  const auto route = lupine_route_for_default();
  return {lupine_route_identity(route), lupine_current_context_hint(), true};
}

int resolve_owner(const lupine_cuda_owner &owner, owner_record *record) {
  if (record == nullptr) {
    return LUPINE_CUDA_CALL_INVALID_OWNER;
  }

  switch (owner.kind) {
  case LUPINE_CUDA_OWNER_CURRENT_CONTEXT:
    *record = current_owner();
    return LUPINE_CUDA_CALL_SUCCESS;
  case LUPINE_CUDA_OWNER_CUBLAS_HANDLE: {
    std::lock_guard<std::mutex> lock(owner_mutex());
    const auto found = owners().find({owner.kind, owner.value});
    if (found == owners().end()) {
      return LUPINE_CUDA_CALL_INVALID_OWNER;
    }
    *record = found->second;
    return LUPINE_CUDA_CALL_SUCCESS;
  }
  case LUPINE_CUDA_OWNER_STREAM: {
    if (owner.value == 0) {
      *record = {};
      return LUPINE_CUDA_CALL_SUCCESS;
    }
    const auto stream = reinterpret_cast<CUstream>(owner.value);
    const auto route = lupine_route_for_known_stream(stream);
    if (route.kind == LUPINE_ROUTE_INVALID) {
      return LUPINE_CUDA_CALL_INVALID_OWNER;
    }
    *record = {lupine_route_identity(route), lupine_context_for_stream(stream),
               true};
    return LUPINE_CUDA_CALL_SUCCESS;
  }
  case LUPINE_CUDA_OWNER_DEVICE_POINTER: {
    if (owner.value == 0) {
      *record = {};
      return LUPINE_CUDA_CALL_SUCCESS;
    }
    const auto pointer = static_cast<CUdeviceptr>(owner.value);
    const auto route = lupine_route_for_deviceptr(pointer);
    *record = {lupine_route_identity(route),
               lupine_context_for_deviceptr(pointer), true};
    return LUPINE_CUDA_CALL_SUCCESS;
  }
  }
  return LUPINE_CUDA_CALL_INVALID_OWNER;
}

int begin_call(int operation, const lupine_cuda_owner *call_owners,
               std::size_t owner_count, lupine_cuda_call **call_out) {
  if (call_out == nullptr || (owner_count != 0 && call_owners == nullptr)) {
    return LUPINE_CUDA_CALL_INVALID_OWNER;
  }
  *call_out = nullptr;

  owner_record selected;
  for (std::size_t index = 0; index < owner_count; ++index) {
    owner_record candidate;
    const int result = resolve_owner(call_owners[index], &candidate);
    if (result != LUPINE_CUDA_CALL_SUCCESS) {
      return result;
    }
    if (!candidate.constrained) {
      continue;
    }
    if (!selected.constrained) {
      selected = candidate;
      continue;
    }
    if (selected.route_id != candidate.route_id ||
        (selected.context != nullptr && candidate.context != nullptr &&
         selected.context != candidate.context)) {
      return LUPINE_CUDA_CALL_OWNER_MISMATCH;
    }
    if (selected.context == nullptr) {
      selected.context = candidate.context;
    }
  }
  if (!selected.constrained) {
    selected = current_owner();
  }

  const auto route = lupine_route_from_identity(selected.route_id);
  conn_t *connection = lupine_route_remote_conn(route);
  if (connection == nullptr) {
    return LUPINE_CUDA_CALL_UNAVAILABLE;
  }

  auto *call = new (std::nothrow) lupine_cuda_call();
  if (call == nullptr) {
    return LUPINE_CUDA_CALL_UNAVAILABLE;
  }
  call->operation = operation;
  call->route_id = selected.route_id;
  call->context = selected.context;
  call->connection = connection;
  *call_out = call;
  return LUPINE_CUDA_CALL_SUCCESS;
}

int write_call(lupine_cuda_call *call, const void *data, std::size_t size) {
  if (call == nullptr || call->invocation_started ||
      (data == nullptr && size != 0)) {
    return -1;
  }
  if (size == 0) {
    return 0;
  }
  const std::size_t offset = call->request.size();
  if (size > call->request.max_size() - offset) {
    return -1;
  }
  try {
    call->request.resize(offset + size);
  } catch (const std::bad_alloc &) {
    return -1;
  }
  std::memcpy(call->request.data() + offset, data, size);
  return 0;
}

int invoke_call(lupine_cuda_call *call) {
  if (call == nullptr || call->invocation_started || call->finished) {
    return -1;
  }
  call->invocation_started = true;
  const auto route = lupine_route_from_identity(call->route_id);
  if ((call->context != nullptr && lupine_set_current_context_on_route(
                                       route, call->context) != CUDA_SUCCESS) ||
      lupine_prepare_rpc(call->connection) < 0 ||
      rpc_write_start_request(call->connection, call->operation) < 0) {
    return -1;
  }
  if (rpc_write(call->connection, call->request.data(), call->request.size()) <
      0) {
    (void)rpc_write_end(call->connection);
    return -1;
  }
  const int result = rpc_wait_for_response(call->connection);
  call->request.clear();
  call->response_ready = result == 0;
  return result;
}

int read_call(lupine_cuda_call *call, void *data, std::size_t size) {
  if (call == nullptr || !call->response_ready || call->finished) {
    return -1;
  }
  return rpc_read(call->connection, data, size);
}

int finish_call(lupine_cuda_call *call) {
  if (call == nullptr || !call->response_ready || call->finished) {
    return -1;
  }
  call->finished = true;
  return rpc_read_end(call->connection) < 0 ? -1 : 0;
}

int record_owner(lupine_cuda_call *call, lupine_cuda_owner_kind kind,
                 std::uintptr_t value) {
  if (call == nullptr || !call->finished || value == 0 ||
      kind == LUPINE_CUDA_OWNER_CURRENT_CONTEXT) {
    return -1;
  }
  try {
    std::lock_guard<std::mutex> lock(owner_mutex());
    owners()[{kind, value}] = {call->route_id, call->context, true};
  } catch (const std::bad_alloc &) {
    return -1;
  }
  return 0;
}

void forget_owner(lupine_cuda_owner_kind kind, std::uintptr_t value) {
  std::lock_guard<std::mutex> lock(owner_mutex());
  owners().erase({kind, value});
}

void destroy_call(lupine_cuda_call *call) {
  if (call != nullptr && call->response_ready && !call->finished) {
    (void)rpc_read_end(call->connection);
  }
  delete call;
}

const lupine_cuda_client_api api = {
    LUPINE_CUDA_CLIENT_API_VERSION,
    begin_call,
    write_call,
    invoke_call,
    read_call,
    finish_call,
    record_owner,
    forget_owner,
    destroy_call,
};

} // namespace

extern "C" LUPINE_CUDA_CLIENT_API_EXPORT const lupine_cuda_client_api *
lupine_cuda_client_api_v1() {
  return &api;
}
