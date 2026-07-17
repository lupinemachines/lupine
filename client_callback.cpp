#include "client_callback.h"

#include <cstdlib>
#include <memory>
#include <mutex>
#include <new>
#include <unordered_map>

namespace {

enum class callback_kind { host, stream };

struct callback_registration {
  conn_t *conn = nullptr;
  callback_kind kind = callback_kind::host;
  bool persistent = false;
  CUhostFn host_callback = nullptr;
  CUstreamCallback stream_callback = nullptr;
  void *user_data = nullptr;
};

using callback_map =
    std::unordered_map<void *, std::unique_ptr<callback_registration>>;

callback_map &registrations() {
  // The CUDA shim can be called during process teardown. Keep registry
  // infrastructure alive until process exit and clear per-connection records
  // explicitly from rpc_close().
  static auto *map = new callback_map();
  return *map;
}

std::mutex &registration_mutex() {
  static auto *mutex = new std::mutex();
  return *mutex;
}

bool insert_registration(std::unique_ptr<callback_registration> registration,
                         lupine_wire_callback *wire) {
  if (!registration || wire == nullptr) {
    return false;
  }
  void *token = registration.get();
  try {
    std::lock_guard<std::mutex> lock(registration_mutex());
    registrations().emplace(token, std::move(registration));
  } catch (const std::bad_alloc &) {
    return false;
  }
  wire->function_token = token;
  wire->user_data_token = token;
  return true;
}

bool valid_wire_registration(callback_map::iterator it, conn_t *conn,
                             const lupine_wire_callback &wire,
                             callback_kind kind) {
  return it != registrations().end() && it->second->conn == conn &&
         it->second->kind == kind && wire.function_token != nullptr &&
         wire.function_token == wire.user_data_token;
}

void *default_allocate(size_t size) { return std::malloc(size); }
void default_deallocate(void *ptr) { std::free(ptr); }

constexpr int kMaxCallbackTransfers = 64 * 1024;

int poison(conn_t *conn) {
  rpc_poison_connection(conn);
  return LUPINE_CALLBACK_DISPATCH_FATAL;
}

} // namespace

bool lupine_register_host_callback(conn_t *conn, CUhostFn callback,
                                   void *user_data, bool persistent,
                                   lupine_wire_callback *wire) {
  if (conn == nullptr || callback == nullptr || wire == nullptr) {
    return false;
  }
  auto registration = std::unique_ptr<callback_registration>(
      new (std::nothrow) callback_registration());
  if (!registration) {
    return false;
  }
  registration->conn = conn;
  registration->kind = callback_kind::host;
  registration->persistent = persistent;
  registration->host_callback = callback;
  registration->user_data = user_data;
  return insert_registration(std::move(registration), wire);
}

bool lupine_register_stream_callback(conn_t *conn, CUstreamCallback callback,
                                     void *user_data,
                                     lupine_wire_callback *wire) {
  if (conn == nullptr || callback == nullptr || wire == nullptr) {
    return false;
  }
  auto registration = std::unique_ptr<callback_registration>(
      new (std::nothrow) callback_registration());
  if (!registration) {
    return false;
  }
  registration->conn = conn;
  registration->kind = callback_kind::stream;
  registration->stream_callback = callback;
  registration->user_data = user_data;
  return insert_registration(std::move(registration), wire);
}

void lupine_revoke_callback(conn_t *conn, void *function_token) {
  std::lock_guard<std::mutex> lock(registration_mutex());
  auto it = registrations().find(function_token);
  if (it != registrations().end() && it->second->conn == conn) {
    registrations().erase(it);
  }
}

void lupine_clear_callbacks(conn_t *conn) {
  std::lock_guard<std::mutex> lock(registration_mutex());
  for (auto it = registrations().begin(); it != registrations().end();) {
    if (it->second->conn == conn) {
      it = registrations().erase(it);
    } else {
      ++it;
    }
  }
}

bool lupine_resolve_host_callback(conn_t *conn,
                                  const lupine_wire_callback &wire,
                                  bool consume, CUhostFn *callback,
                                  void **user_data) {
  if (callback == nullptr || user_data == nullptr) {
    return false;
  }
  std::lock_guard<std::mutex> lock(registration_mutex());
  auto it = registrations().find(wire.function_token);
  if (!valid_wire_registration(it, conn, wire, callback_kind::host)) {
    return false;
  }
  *callback = it->second->host_callback;
  *user_data = it->second->user_data;
  if (consume && !it->second->persistent) {
    registrations().erase(it);
  }
  return true;
}

bool lupine_resolve_stream_callback(conn_t *conn,
                                    const lupine_wire_callback &wire,
                                    CUstreamCallback *callback,
                                    void **user_data) {
  if (callback == nullptr || user_data == nullptr) {
    return false;
  }
  std::lock_guard<std::mutex> lock(registration_mutex());
  auto it = registrations().find(wire.function_token);
  if (!valid_wire_registration(it, conn, wire, callback_kind::stream)) {
    return false;
  }
  *callback = it->second->stream_callback;
  *user_data = it->second->user_data;
  registrations().erase(it);
  return true;
}

int lupine_dispatch_host_callback(
    conn_t *conn, const lupine_host_callback_dispatch_options *options,
    int *request_id) {
  if (conn == nullptr || request_id == nullptr) {
    return poison(conn);
  }
  *request_id = -1;

  lupine_host_callback_dispatch_options configured;
  if (options != nullptr) {
    configured = *options;
  }
  if (configured.allocate == nullptr) {
    configured.allocate = default_allocate;
  }
  if (configured.deallocate == nullptr) {
    configured.deallocate = default_deallocate;
  }

  int transfer_count = 0;
  if (rpc_read(conn, &transfer_count, sizeof(transfer_count)) < 0 ||
      transfer_count < 0 || transfer_count > kMaxCallbackTransfers) {
    return poison(conn);
  }

  bool aborted = false;
  size_t drain_budget = configured.max_drain_bytes;
  for (int i = 0; i < transfer_count; ++i) {
    void *dst = nullptr;
    size_t count = 0;
    if (rpc_read(conn, &dst, sizeof(dst)) < 0 ||
        rpc_read(conn, &count, sizeof(count)) < 0) {
      return poison(conn);
    }

    if (aborted) {
      if (count > drain_budget ||
          rpc_drain_payload(conn, lupine_payload_framed(conn, count), count) <
              0) {
        return poison(conn);
      }
      drain_budget -= count;
      continue;
    }

    void *host_data = count == 0 ? nullptr : configured.allocate(count);
    if (count != 0 && host_data == nullptr) {
      aborted = true;
      if (count > drain_budget ||
          rpc_drain_payload(conn, lupine_payload_framed(conn, count), count) <
              0) {
        return poison(conn);
      }
      drain_budget -= count;
      continue;
    }

    if (count != 0 && rpc_read_payload(conn, host_data, count) < 0) {
      configured.deallocate(host_data);
      return poison(conn);
    }
    if (count != 0 && configured.commit_copy != nullptr) {
      configured.commit_copy(dst, host_data, count);
    }
    configured.deallocate(host_data);
  }

  lupine_wire_callback wire;
  if (rpc_read(conn, &wire.function_token, sizeof(wire.function_token)) < 0 ||
      rpc_read(conn, &wire.user_data_token, sizeof(wire.user_data_token)) < 0) {
    return poison(conn);
  }
  *request_id = rpc_read_end(conn);
  if (*request_id < 0) {
    return poison(conn);
  }

  if (aborted) {
    lupine_revoke_callback(conn, wire.function_token);
    return LUPINE_CALLBACK_DISPATCH_ABORTED;
  }

  CUhostFn callback = nullptr;
  void *user_data = nullptr;
  if (!lupine_resolve_host_callback(conn, wire, true, &callback, &user_data) ||
      callback == nullptr) {
    return LUPINE_CALLBACK_DISPATCH_ABORTED;
  }
  callback(user_data);
  return LUPINE_CALLBACK_DISPATCH_COMPLETE;
}
