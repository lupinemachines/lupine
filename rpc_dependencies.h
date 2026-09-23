#ifndef LUPINE_RPC_DEPENDENCIES_H
#define LUPINE_RPC_DEPENDENCIES_H

#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <vector>

struct conn_t;
struct rpc_dependency_state;
struct rpc_ordering_domain;

struct rpc_submission {
  int32_t lane;
  int request;
};

// Only the client interprets resource identities. The server sees lane/request
// prerequisites and relies on the existing FIFO execution within each lane.
rpc_ordering_domain *rpc_dependency_domain(conn_t *conn, uint64_t kind,
                                           uint64_t object,
                                           uint64_t context = 0,
                                           uint64_t thread = 0);

// The usual scope has one prerequisite and at most five publications. Keep
// those on the stack, with an unbounded fallback for concurrent producers.
template <typename T, size_t N> class rpc_dependency_list {
public:
  T *begin() { return overflow_.empty() ? inline_ : overflow_.data(); }
  const T *begin() const {
    return overflow_.empty() ? inline_ : overflow_.data();
  }
  T *end() { return begin() + size_; }
  const T *end() const { return begin() + size_; }
  void push_back(T value) {
    if (size_ < N) {
      inline_[size_] = value;
    } else {
      if (size_ == N) {
        overflow_.reserve(N * 2);
        overflow_.assign(inline_, inline_ + N);
      }
      overflow_.push_back(value);
    }
    ++size_;
  }

private:
  T inline_[N];
  size_t size_ = 0;
  std::vector<T> overflow_;
};

class rpc_dependency_call {
public:
  rpc_dependency_call(
      conn_t *conn, std::initializer_list<rpc_ordering_domain *> required,
      std::initializer_list<rpc_ordering_domain *> published = {});
  rpc_dependency_call(conn_t *conn,
                      const std::vector<rpc_ordering_domain *> &required);
  ~rpc_dependency_call();
  rpc_dependency_call(const rpc_dependency_call &) = delete;
  rpc_dependency_call &operator=(const rpc_dependency_call &) = delete;

private:
  rpc_dependency_call(conn_t *conn, rpc_ordering_domain *const *required,
                      size_t count,
                      std::initializer_list<rpc_ordering_domain *> published);
  void require(rpc_submission submission);
  bool covers(rpc_submission submission) const;
  friend bool rpc_dependency_scoped(conn_t *);
  friend std::vector<rpc_submission> rpc_dependency_prepare(conn_t *, int32_t);
  friend void rpc_dependency_publish(conn_t *, rpc_submission, bool);
  conn_t *conn_;
  rpc_dependency_call *previous_;
  rpc_dependency_list<rpc_submission, 2> required_;
  rpc_dependency_list<rpc_ordering_domain *, 5> published_;
};

rpc_dependency_state *rpc_dependency_create();
void rpc_dependency_destroy(rpc_dependency_state *state);
void rpc_dependency_shutdown(conn_t *conn);
bool rpc_dependency_scoped(conn_t *conn);
// The existing request builder lock serializes prepare/publish. Entry snapshots
// may run concurrently; they never hold that lock while reading prerequisites.
std::vector<rpc_submission> rpc_dependency_prepare(conn_t *conn, int32_t lane);
void rpc_dependency_publish(conn_t *conn, rpc_submission submission,
                            bool async);
int rpc_dependency_wait(conn_t *conn, rpc_submission required);
void rpc_dependency_complete(conn_t *conn, rpc_submission submission);

#endif
