#include "rpc_dependencies.h"
#include "rpc.h"

#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <map>
#include <mutex>
#include <new>
#include <tuple>

// Most resources have one outstanding producer. Multiple producers use a
// vector and the metadata mutex. Publication already holds call_mutex.
static constexpr uint64_t multiple_producers = UINT64_MAX;

static uint64_t pack(rpc_submission submission) {
  return (uint64_t(submission.lane) << 32) | uint32_t(submission.request);
}

static rpc_submission unpack(uint64_t value) {
  return {int32_t(value >> 32), int(uint32_t(value))};
}

struct rpc_ordering_domain {
  std::atomic<uint64_t> published{0};
  std::vector<rpc_submission> concurrent;
};

struct rpc_dependency_state {
  uint64_t id;
  std::mutex mutex;
  std::condition_variable changed;
  bool closed = false;
  std::map<std::tuple<uint64_t, uint64_t, uint64_t, uint64_t>,
           rpc_ordering_domain>
      domains;
  std::map<int32_t, int> completed;
};

static thread_local rpc_dependency_call *current_call = nullptr;

// A thread owns its sending lane. Remember its fences locally, with no shared
// lane tracker. Switching connections or lanes merely loses this optimization.
static thread_local bool fences_destroyed = false;

static std::map<int32_t, int> *sent_fences(conn_t *conn, int32_t lane) {
  // CUDA's process-exit destructors can issue RPCs after TLS destructors ran.
  // Those calls still send dependencies, but must not reuse the dead cache.
  if (fences_destroyed) {
    return nullptr;
  }
  struct cache {
    uint64_t connection_id = 0;
    int32_t sending_lane = 0;
    std::map<int32_t, int> fences;
    ~cache() { fences_destroyed = true; }
  };
  static thread_local cache sent;
  if (sent.connection_id != conn->dependencies->id ||
      sent.sending_lane != lane) {
    sent.fences.clear();
    sent.connection_id = conn->dependencies->id;
    sent.sending_lane = lane;
  }
  return &sent.fences;
}

bool rpc_dependency_call::covers(rpc_submission submission) const {
  return std::any_of(required_.begin(), required_.end(), [&](auto r) {
    return r.lane == submission.lane && r.request >= submission.request;
  });
}

void rpc_dependency_call::require(rpc_submission submission) {
  auto found = std::find_if(required_.begin(), required_.end(),
                            [&](auto r) { return r.lane == submission.lane; });
  if (found == required_.end()) {
    required_.push_back(submission);
  } else {
    found->request = std::max(found->request, submission.request);
  }
}

rpc_dependency_state *rpc_dependency_create() {
  static std::atomic<uint64_t> next_id{1};
  auto *state = new (std::nothrow) rpc_dependency_state;
  if (state != nullptr) {
    state->id = next_id.fetch_add(1, std::memory_order_relaxed);
  }
  return state;
}

void rpc_dependency_destroy(rpc_dependency_state *state) { delete state; }

void rpc_dependency_shutdown(conn_t *conn) {
  if (conn->dependencies == nullptr) {
    return;
  }
  auto &state = *conn->dependencies;
  std::lock_guard<std::mutex> lock(state.mutex);
  state.closed = true;
  state.changed.notify_all();
}

rpc_ordering_domain *rpc_dependency_domain(conn_t *conn, uint64_t kind,
                                           uint64_t object, uint64_t context,
                                           uint64_t thread) {
  if (conn == nullptr) {
    return nullptr;
  }
  auto &state = *conn->dependencies;
  std::lock_guard<std::mutex> lock(state.mutex);
  return &state.domains[{kind, object, context, thread}];
}

rpc_dependency_call::rpc_dependency_call(
    conn_t *conn, std::initializer_list<rpc_ordering_domain *> required,
    std::initializer_list<rpc_ordering_domain *> published)
    : rpc_dependency_call(conn, required.begin(), required.size(), published) {}

rpc_dependency_call::rpc_dependency_call(
    conn_t *conn, const std::vector<rpc_ordering_domain *> &required)
    : rpc_dependency_call(conn, required.data(), required.size(), {}) {}

rpc_dependency_call::rpc_dependency_call(
    conn_t *conn, rpc_ordering_domain *const *required, size_t count,
    std::initializer_list<rpc_ordering_domain *> published)
    : conn_(conn), previous_(current_call) {
  for (auto *domain : published) {
    if (domain != nullptr) {
      published_.push_back(domain);
    }
  }
  for (size_t i = 0; i < count; ++i) {
    auto *domain = required[i];
    if (domain == nullptr) {
      continue;
    }
    uint64_t value = domain->published.load(std::memory_order_acquire);
    if (value == multiple_producers) {
      std::lock_guard<std::mutex> lock(conn->dependencies->mutex);
      // A publication may have compacted the vector while we took the lock.
      value = domain->published.load(std::memory_order_relaxed);
      if (value == multiple_producers) {
        for (auto submission : domain->concurrent) {
          require(submission);
        }
        continue;
      }
    }
    if (value != 0) {
      require(unpack(value));
    }
  }
  current_call = this;
}

rpc_dependency_call::~rpc_dependency_call() { current_call = previous_; }

bool rpc_dependency_scoped(conn_t *conn) {
  return current_call != nullptr && current_call->conn_ == conn;
}

std::vector<rpc_submission> rpc_dependency_prepare(conn_t *conn, int32_t lane) {
  std::vector<rpc_submission> required;
  if (rpc_dependency_scoped(conn)) {
    for (auto submission : current_call->required_) {
      if (submission.lane == lane) {
        continue;
      }
      auto *fences = sent_fences(conn, lane);
      if (fences == nullptr ||
          (*fences)[submission.lane] < submission.request) {
        required.push_back(submission);
      }
    }
  }
  return required;
}

void rpc_dependency_publish(conn_t *conn, rpc_submission submission,
                            bool async) {
  if (!rpc_dependency_scoped(conn)) {
    return;
  }
  for (auto required : current_call->required_) {
    if (required.lane != submission.lane) {
      if (auto *fences = sent_fences(conn, submission.lane)) {
        int &sent = (*fences)[required.lane];
        sent = std::max(sent, required.request);
      }
    }
  }
  if (!async) {
    return;
  }
  std::unique_lock<std::mutex> lock(conn->dependencies->mutex, std::defer_lock);
  for (auto *domain : current_call->published_) {
    uint64_t previous = domain->published.load(std::memory_order_relaxed);
    if (previous != multiple_producers &&
        (previous == 0 || unpack(previous).lane == submission.lane ||
         current_call->covers(unpack(previous)))) {
      domain->published.store(pack(submission), std::memory_order_release);
      continue;
    }
    if (!lock.owns_lock()) {
      lock.lock();
    }
    auto &frontier = domain->concurrent;
    if (previous != multiple_producers) {
      frontier = {unpack(previous)};
    }
    // Keep publications that happened after this call's entry snapshot.
    frontier.erase(std::remove_if(frontier.begin(), frontier.end(),
                                  [&](auto s) {
                                    return s.lane == submission.lane ||
                                           current_call->covers(s);
                                  }),
                   frontier.end());
    frontier.push_back(submission);
    domain->published.store(frontier.size() == 1 ? pack(submission)
                                                 : multiple_producers,
                            std::memory_order_release);
  }
}

int rpc_dependency_wait(conn_t *conn, rpc_submission required) {
  auto &state = *conn->dependencies;
  std::unique_lock<std::mutex> lock(state.mutex);
  state.changed.wait(lock, [&] {
    return state.closed || state.completed[required.lane] >= required.request;
  });
  return state.closed ? -1 : 0;
}

void rpc_dependency_complete(conn_t *conn, rpc_submission submission) {
  auto &state = *conn->dependencies;
  std::lock_guard<std::mutex> lock(state.mutex);
  state.completed[submission.lane] = submission.request;
  state.changed.notify_all();
}
