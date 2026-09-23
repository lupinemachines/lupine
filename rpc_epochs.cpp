#include "rpc_epochs.h"
#include "rpc.h"

#include <algorithm>
#include <atomic>
#include <cassert>
#include <condition_variable>
#include <map>
#include <memory>
#include <mutex>
#include <new>
#include <set>
#include <tuple>
#include <unordered_map>

struct rpc_ordering_domain {
  uint64_t id;
  std::atomic<uint64_t> published{0};
};

struct rpc_epoch_state {
  struct progress {
    uint64_t completed = 0;
    std::set<uint64_t> out_of_order;
  };
  std::mutex mutex;
  std::condition_variable changed;
  bool closed = false;
  std::map<std::tuple<uint64_t, uint64_t, uint64_t, uint64_t>,
           std::unique_ptr<rpc_ordering_domain>>
      domains;
  std::unordered_map<uint64_t, progress> epochs;
};

static thread_local rpc_epoch_call *current_call = nullptr;

rpc_epoch_state *rpc_epoch_create() {
  return new (std::nothrow) rpc_epoch_state;
}

void rpc_epoch_destroy(rpc_epoch_state *state) { delete state; }

void rpc_epoch_shutdown(conn_t *conn) {
  if (conn->epochs == nullptr) {
    return;
  }
  auto &state = *conn->epochs;
  std::lock_guard<std::mutex> lock(state.mutex);
  state.closed = true;
  state.changed.notify_all();
}

rpc_ordering_domain *rpc_epoch_domain(conn_t *conn, uint64_t kind,
                                      uint64_t object, uint64_t context,
                                      uint64_t lane) {
  if (conn == nullptr) {
    return nullptr;
  }
  auto &state = *conn->epochs;
  std::lock_guard<std::mutex> lock(state.mutex);
  auto key = std::make_tuple(kind, object, context, lane);
  auto found = state.domains.find(key);
  if (found != state.domains.end()) {
    return found->second.get();
  }
  auto domain = std::make_unique<rpc_ordering_domain>();
  domain->id = state.domains.size() + 1;
  return state.domains.emplace(key, std::move(domain)).first->second.get();
}

rpc_epoch_call::rpc_epoch_call(
    conn_t *conn, const std::vector<rpc_ordering_domain *> &required,
    std::vector<rpc_ordering_domain *> published)
    : conn_(conn), previous_(current_call), published_(std::move(published)) {
  required_.reserve(required.size());
  for (auto *domain : required) {
    if (domain == nullptr) {
      continue;
    }
    uint64_t value = domain->published.load(std::memory_order_acquire);
    if (value != 0 &&
        std::none_of(required_.begin(), required_.end(),
                     [=](rpc_epoch e) { return e.domain == domain->id; })) {
      required_.push_back({domain->id, value});
    }
  }
  published_.erase(std::remove(published_.begin(), published_.end(), nullptr),
                   published_.end());
  std::sort(published_.begin(), published_.end(),
            [](auto *a, auto *b) { return a->id < b->id; });
  published_.erase(std::unique(published_.begin(), published_.end()),
                   published_.end());
  current_call = this;
}

rpc_epoch_call::~rpc_epoch_call() { current_call = previous_; }

bool rpc_epoch_scoped(conn_t *conn) {
  return current_call != nullptr && current_call->conn_ == conn;
}

rpc_request_epochs rpc_epoch_prepare(conn_t *conn, bool async) {
  if (!rpc_epoch_scoped(conn)) {
    return {};
  }
  rpc_request_epochs result;
  result.required = current_call->required_;
  if (async) {
    result.published.reserve(current_call->published_.size());
    for (auto *domain : current_call->published_) {
      result.published.push_back(
          {domain->id, domain->published.load(std::memory_order_relaxed) + 1});
    }
  }
  return result;
}

void rpc_epoch_publish(conn_t *conn, const rpc_request_epochs &epochs) {
  if (epochs.published.empty()) {
    return;
  }
  // prepare/publish run inside the same call scope and existing request
  // builder lock. Entry snapshots remain lock-free while encoding proceeds.
  assert(rpc_epoch_scoped(conn));
  assert(epochs.published.size() == current_call->published_.size());
  for (size_t i = 0; i < epochs.published.size(); ++i) {
    auto *domain = current_call->published_[i];
    assert(domain->id == epochs.published[i].domain);
    domain->published.store(epochs.published[i].value,
                            std::memory_order_release);
  }
}

int rpc_epoch_wait(conn_t *conn, const std::vector<rpc_epoch> &required) {
  if (required.empty()) {
    return 0;
  }
  auto &state = *conn->epochs;
  std::unique_lock<std::mutex> lock(state.mutex);
  state.changed.wait(lock, [&] {
    return state.closed ||
           std::all_of(required.begin(), required.end(), [&](rpc_epoch epoch) {
             return state.epochs[epoch.domain].completed >= epoch.value;
           });
  });
  return state.closed ? -1 : 0;
}

void rpc_epoch_complete(conn_t *conn, const std::vector<rpc_epoch> &published) {
  if (published.empty()) {
    return;
  }
  auto &state = *conn->epochs;
  std::lock_guard<std::mutex> lock(state.mutex);
  for (rpc_epoch epoch : published) {
    auto &progress = state.epochs[epoch.domain];
    if (epoch.value == progress.completed + 1) {
      ++progress.completed;
      while (progress.out_of_order.erase(progress.completed + 1) != 0) {
        ++progress.completed;
      }
    } else if (epoch.value > progress.completed) {
      progress.out_of_order.insert(epoch.value);
    }
  }
  state.changed.notify_all();
}
