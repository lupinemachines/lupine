#ifndef LUPINE_CLIENT_TRANSPORT_H
#define LUPINE_CLIENT_TRANSPORT_H

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

struct lupine_client_transport_config {
  lupine_client_dial_policy dial_policy =
      lupine_client_dial_policy::bounded_retry;
  bool strict_endpoints = false;
  bool log_missing_server = false;
  void *(*dispatch)(void *conn) = nullptr;
  void (*connection_opened)(conn_t *conn) = nullptr;
  void (*connection_closed)(conn_t *conn) = nullptr;
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

#undef LUPINE_TRANSPORT_INTERNAL

#endif
