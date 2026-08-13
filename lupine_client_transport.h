#ifndef LUPINE_CLIENT_TRANSPORT_H
#define LUPINE_CLIENT_TRANSPORT_H

#include "rpc.h"

// Shared client transport, linked by every lupine client shim (libcuda,
// libnvidia-ml, libamdhip64). Each shim is separately dlopened, so each links
// its own copy of lupine_client_transport.cpp and gets an independent
// connection pool -- the definitions below live in the TU, and shims reach
// them through these extern declarations.
//
// The pool state names are intentionally bare (conn_mutex, conns, nconns) so
// the CUDA client's rpc_destroy_thread_lane, which predates this TU, keeps
// referencing the pool by the names it always has.

extern pthread_mutex_t conn_mutex;
extern conn_t conns[16];
extern int nconns;

// Hooks a shim passes to the common transport so it can inject API-specific
// behavior without the transport knowing the API. Pass zero-initialised
// fields (nullptr) for behaviour you do not need.
struct lupine_transport_hooks {
  // Required. The per-connection read dispatch thread, handed to
  // pthread_create for each new connection. nvml and hip pass
  // lupine_transport_dispatch_thread; the CUDA client passes its own richer
  // dispatch thread (op==1 CUhostFn transfer, op==2 CUstream callback).
  void *(*dispatch_thread)(void *);

  // Optional. Called under conn_mutex after a successful connect, BEFORE the
  // pool slot's index is published (++nconns), so any per-conn record the hook
  // keeps lines up with the connection's index. host/port/tls are the parsed
  // endpoint. The NVML client records the server label here so
  // nvmlDeviceGetName can append "(via lupine <label>)"; the CUDA client
  // invalidates its per-context cache here.
  void (*on_connect)(conn_t *conn, const char *host, const char *port, int tls);

  // Optional. Called under conn_mutex for each connection during shutdown
  // instead of the plain manual close. The CUDA client passes rpc_close (which
  // also invalidates its per-context cache); nvml and hip pass nullptr to get
  // the plain shutdown + lupine_socket_close + read_cond broadcast.
  void (*on_close_conn)(conn_t *conn);
};

// Parses LUPINE_SERVER (comma-separated host[:port] entries, optional
// "https://" scheme for TLS, default port 14833) and connects one pool slot
// per reachable endpoint, spawning dispatch_thread per connection. Idempotent:
// returns 0 if the pool is already up. Returns -1 if LUPINE_SERVER is unset or
// no endpoint could be reached.
//
// Connections use lupine_tcp_connect, which retries with backoff when a server
// is not reachable yet. The CUDA client already behaved this way; nvml and
// hip previously did a single blocking connect, so they now gain the same
// startup robustness.
int lupine_transport_open(const lupine_transport_hooks *hooks);

// The minimal read dispatch thread used by shims that do not need rich
// dispatch (nvml, hip). Shims with richer dispatch pass their own thread fn to
// lupine_transport_open instead.
void *lupine_transport_dispatch_thread(void *conn);

// Shuts down every connection in the pool: for each connection calls
// on_close_conn if set, otherwise a plain shutdown + lupine_socket_close +
// read_cond broadcast; then joins all read/rpc threads, frees any TLS
// session, and destroys the connection. Resets nconns to 0. A shim clears its
// own API-specific state (device tables, labels, etc.) after this returns.
void lupine_transport_close_all(void (*on_close_conn)(conn_t *));

#endif