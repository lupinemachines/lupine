#include "rpc_dependencies.h"
#include "rpc.h"

#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <map>
#include <mutex>
#include <new>
#include <tuple>

struct rpc_ordering_domain {
  std::vector<rpc_submission> published;
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

static bool covers(const std::vector<rpc_submission> &required,
                   rpc_submission s) {
  return std::any_of(required.begin(), required.end(), [&](auto r) {
    return r.lane == s.lane && r.request >= s.request;
  });
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
    conn_t *conn, const std::vector<rpc_ordering_domain *> &required,
    std::vector<rpc_ordering_domain *> published)
    : conn_(conn), previous_(current_call), published_(std::move(published)) {
  if (conn != nullptr) {
    // Snapshot before taking the request builder lock. Overlapping calls do
    // not acquire dependencies on one another's later publications.
    std::lock_guard<std::mutex> lock(conn->dependencies->mutex);
    for (auto *domain : required) {
      if (domain == nullptr) {
        continue;
      }
      for (auto submission : domain->published) {
        auto found =
            std::find_if(required_.begin(), required_.end(),
                         [&](auto s) { return s.lane == submission.lane; });
        if (found == required_.end()) {
          required_.push_back(submission);
        } else {
          found->request = std::max(found->request, submission.request);
        }
      }
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
    auto *fences = sent_fences(conn, lane);
    for (auto submission : current_call->required_) {
      if (submission.lane != lane &&
          (fences == nullptr ||
           (*fences)[submission.lane] < submission.request)) {
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
  if (auto *fences = sent_fences(conn, submission.lane)) {
    for (auto required : current_call->required_) {
      int &sent = (*fences)[required.lane];
      sent = std::max(sent, required.request);
    }
  }
  if (!async) {
    return;
  }
  std::lock_guard<std::mutex> lock(conn->dependencies->mutex);
  for (auto *domain : current_call->published_) {
    if (domain == nullptr) {
      continue;
    }
    auto &frontier = domain->published;
    // Replace only prerequisites actually observed at entry (or covered by
    // this lane's FIFO). Keep publications from overlapping calls on other
    // lanes, including producers of a shared context aggregate.
    frontier.erase(std::remove_if(frontier.begin(), frontier.end(),
                                  [&](auto s) {
                                    return s.lane == submission.lane ||
                                           covers(current_call->required_, s);
                                  }),
                   frontier.end());
    frontier.push_back(submission);
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
