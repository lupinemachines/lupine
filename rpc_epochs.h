#ifndef LUPINE_RPC_EPOCHS_H
#define LUPINE_RPC_EPOCHS_H

#include <cstddef>
#include <cstdint>
#include <vector>

struct conn_t;
struct rpc_epoch_state;
struct rpc_ordering_domain;

constexpr size_t RPC_MAX_REQUEST_EPOCHS = 32;

struct rpc_epoch {
  uint64_t domain;
  uint64_t value;
};

struct rpc_request_epochs {
  std::vector<rpc_epoch> required;
  std::vector<rpc_epoch> published;
};

// The backend names ordering domains; the transport never interprets handles.
rpc_ordering_domain *rpc_epoch_domain(conn_t *conn, uint64_t kind,
                                      uint64_t object, uint64_t context = 0,
                                      uint64_t lane = 0);

// Snapshot at call entry, before taking the request builder lock. Only
// fire-and-forget requests publish epochs. Nested calls restore their caller's
// snapshot, including when a helper uses a different connection.
class rpc_epoch_call {
public:
  rpc_epoch_call(conn_t *conn,
                 const std::vector<rpc_ordering_domain *> &required,
                 std::vector<rpc_ordering_domain *> published = {});
  ~rpc_epoch_call();
  rpc_epoch_call(const rpc_epoch_call &) = delete;
  rpc_epoch_call &operator=(const rpc_epoch_call &) = delete;

private:
  friend bool rpc_epoch_scoped(conn_t *);
  friend rpc_request_epochs rpc_epoch_prepare(conn_t *, bool);
  friend void rpc_epoch_publish(conn_t *, const rpc_request_epochs &);
  conn_t *conn_;
  rpc_epoch_call *previous_;
  std::vector<rpc_epoch> required_;
  std::vector<rpc_ordering_domain *> published_;
};

// RPC implementation hooks. Publication is serialized by the existing request
// builder lock, but snapshots and completion never hold it or a native-call
// lock. Values label publication order, not execution order.
rpc_epoch_state *rpc_epoch_create();
void rpc_epoch_destroy(rpc_epoch_state *state);
void rpc_epoch_shutdown(conn_t *conn);
bool rpc_epoch_scoped(conn_t *conn);
rpc_request_epochs rpc_epoch_prepare(conn_t *conn, bool async);
void rpc_epoch_publish(conn_t *conn, const rpc_request_epochs &epochs);
int rpc_epoch_wait(conn_t *conn, const std::vector<rpc_epoch> &required);
void rpc_epoch_complete(conn_t *conn, const std::vector<rpc_epoch> &published);

#endif
