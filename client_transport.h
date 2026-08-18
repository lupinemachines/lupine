#ifndef LUPINE_CLIENT_TRANSPORT_H
#define LUPINE_CLIENT_TRANSPORT_H

#include "rpc.h"

#include <string>
#include <vector>

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
  const char *default_port = "14833";
  unsigned int max_connections = 16;
  lupine_client_dial_policy dial_policy =
      lupine_client_dial_policy::bounded_retry;
  bool strict_endpoints = false;
  bool log_missing_server = false;
  void *user_data = nullptr;
  lupine_socket_t (*dial)(const lupine_client_endpoint *endpoint,
                          lupine_client_dial_policy policy,
                          void *user_data) = nullptr;
  void (*dispatch)(conn_t *conn, void *user_data) = nullptr;
  void (*connection_opened)(conn_t *conn,
                            const lupine_client_endpoint *endpoint,
                            void *user_data) = nullptr;
  void (*connection_closed)(conn_t *conn, void *user_data) = nullptr;
};

struct lupine_client_transport;

LUPINE_TRANSPORT_INTERNAL int
lupine_client_parse_endpoints(const char *servers,
                              const lupine_client_transport_config *config,
                              std::vector<lupine_client_endpoint> *endpoints);
LUPINE_TRANSPORT_INTERNAL lupine_client_transport *
lupine_client_transport_create(const lupine_client_transport_config *config);
LUPINE_TRANSPORT_INTERNAL void
lupine_client_transport_destroy(lupine_client_transport *transport);
LUPINE_TRANSPORT_INTERNAL int
lupine_client_transport_open(lupine_client_transport *transport,
                             const char *servers = nullptr);
LUPINE_TRANSPORT_INTERNAL void
lupine_client_transport_close(lupine_client_transport *transport);
LUPINE_TRANSPORT_INTERNAL void
lupine_client_transport_close_connection(lupine_client_transport *transport,
                                         conn_t *conn);
LUPINE_TRANSPORT_INTERNAL unsigned int
lupine_client_transport_size(const lupine_client_transport *transport);
LUPINE_TRANSPORT_INTERNAL bool
lupine_client_transport_is_open(const lupine_client_transport *transport);
LUPINE_TRANSPORT_INTERNAL conn_t *
lupine_client_transport_connection(lupine_client_transport *transport,
                                   unsigned int index);
LUPINE_TRANSPORT_INTERNAL const lupine_client_endpoint *
lupine_client_transport_endpoint(const lupine_client_transport *transport,
                                 unsigned int index);
LUPINE_TRANSPORT_INTERNAL void
lupine_client_transport_retire_lane(lupine_client_transport *transport,
                                    uint64_t lane_id);

#undef LUPINE_TRANSPORT_INTERNAL

#endif
