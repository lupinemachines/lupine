#ifndef LUPINE_RPC_DEPENDENCIES_H
#define LUPINE_RPC_DEPENDENCIES_H

#include <cstdint>
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

class rpc_dependency_call {
public:
  rpc_dependency_call(conn_t *conn,
                      const std::vector<rpc_ordering_domain *> &required,
                      std::vector<rpc_ordering_domain *> published = {});
  ~rpc_dependency_call();
  rpc_dependency_call(const rpc_dependency_call &) = delete;
  rpc_dependency_call &operator=(const rpc_dependency_call &) = delete;

private:
  friend bool rpc_dependency_scoped(conn_t *);
  friend std::vector<rpc_submission> rpc_dependency_prepare(conn_t *, int32_t);
  friend void rpc_dependency_publish(conn_t *, rpc_submission, bool);
  conn_t *conn_;
  rpc_dependency_call *previous_;
  std::vector<rpc_submission> required_;
  std::vector<rpc_ordering_domain *> published_;
};

rpc_dependency_state *rpc_dependency_create();
void rpc_dependency_destroy(rpc_dependency_state *state);
void rpc_dependency_shutdown(conn_t *conn);
bool rpc_dependency_scoped(conn_t *conn);
std::vector<rpc_submission> rpc_dependency_prepare(conn_t *conn, int32_t lane);
void rpc_dependency_publish(conn_t *conn, rpc_submission submission,
                            bool async);
int rpc_dependency_wait(conn_t *conn, rpc_submission required);
void rpc_dependency_complete(conn_t *conn, rpc_submission submission);

#endif
