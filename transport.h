#ifndef LUPINE_TRANSPORT_H
#define LUPINE_TRANSPORT_H

#include "rpc.h"

#include <string>

#if defined(__GNUC__) && !defined(_WIN32)
#define LUPINE_TRANSPORT_INTERNAL __attribute__((visibility("hidden")))
#else
#define LUPINE_TRANSPORT_INTERNAL
#endif

enum class lupine_client_dial_policy {
  single_attempt,
  bounded_retry,
};

struct lupine_client_endpoint {
  std::string host;
  std::string port;
  std::string label;
  bool tls = false;
};

// Extra connections to one server for striping large transfers. Each carries
// one stream that the caller writes raw RPC frames to while holding mutex.
constexpr unsigned int LUPINE_BULK_CONNECTIONS_MAX = 8;
struct lupine_bulk_lanes {
  pthread_mutex_t mutex = PTHREAD_MUTEX_INITIALIZER;
  unsigned int count = 0;
  bool failed = false;
  conn_t *conn[LUPINE_BULK_CONNECTIONS_MAX] = {};
  int32_t stream[LUPINE_BULK_CONNECTIONS_MAX] = {};
};

struct lupine_client_transport_config {
  lupine_client_dial_policy dial_policy =
      lupine_client_dial_policy::bounded_retry;
  bool strict_endpoints = false;
  bool log_missing_server = false;
  void *(*dispatch)(void *conn) = nullptr;
  void (*connection_opened)(conn_t *conn) = nullptr;
  void (*connection_closed)(conn_t *conn) = nullptr;
  const char *connection_kind = nullptr;
  intptr_t w_offset = 0;
};

// Each API shim links its own hidden copy of this transport, so one pool per
// linked image keeps CUDA, NVML, and HIP state isolated without instance
// plumbing at every call site.
LUPINE_TRANSPORT_INTERNAL int
lupine_client_transport_open(const lupine_client_transport_config &config,
                             const char *servers = nullptr);
LUPINE_TRANSPORT_INTERNAL void lupine_client_transport_close();
LUPINE_TRANSPORT_INTERNAL void
lupine_client_transport_close_connection(conn_t *conn);
LUPINE_TRANSPORT_INTERNAL unsigned int lupine_client_transport_size();
LUPINE_TRANSPORT_INTERNAL conn_t *
lupine_client_transport_connection(unsigned int index);
LUPINE_TRANSPORT_INTERNAL const lupine_client_endpoint *
lupine_client_transport_endpoint(unsigned int index);
LUPINE_TRANSPORT_INTERNAL void
lupine_client_transport_retire_lane(uint64_t lane_id);
// The bulk lanes for a connection's server, dialed on first use. Null when the
// server does not route bulk connections, LUPINE_BULK_CONNECTIONS is 0, the
// endpoint uses TLS, or a lane has failed.
LUPINE_TRANSPORT_INTERNAL lupine_bulk_lanes *
lupine_client_transport_bulk_lanes(conn_t *conn);

#undef LUPINE_TRANSPORT_INTERNAL

#endif
