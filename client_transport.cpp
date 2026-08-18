#include "client_transport.h"

#include "lupine_log.h"

#include <algorithm>
#include <array>
#include <cstdlib>
#include <cstring>
#include <netdb.h>
#include <sys/socket.h>

#ifdef LUPINE_TLS_OPENSSL
#include <openssl/ssl.h>
#endif

namespace {

constexpr unsigned int kTransportCapacity = 16;

struct lupine_client_connection_slot {
  conn_t conn = {};
  struct lupine_client_transport *transport = nullptr;
};

} // namespace

struct lupine_client_transport {
  explicit lupine_client_transport(
      const lupine_client_transport_config &transport_config)
      : config(transport_config) {}

  lupine_client_transport_config config;
  pthread_mutex_t mutex = PTHREAD_MUTEX_INITIALIZER;
  std::array<lupine_client_connection_slot, kTransportCapacity> slots = {};
  std::array<lupine_client_endpoint, kTransportCapacity> endpoints;
  unsigned int count = 0;
  bool shutting_down = false;
};

namespace {

bool endpoint_has_invalid_scheme(const std::string &endpoint) {
  return endpoint.find("://") != std::string::npos ||
         endpoint.compare(0, 5, "http:") == 0 ||
         endpoint.compare(0, 6, "https:") == 0;
}

lupine_socket_t dial_once(const lupine_client_endpoint &endpoint) {
  addrinfo hints = {};
  hints.ai_family = AF_INET;
  hints.ai_socktype = SOCK_STREAM;
  addrinfo *addresses = nullptr;
  if (getaddrinfo(endpoint.host.c_str(), endpoint.port.c_str(), &hints,
                  &addresses) != 0 ||
      addresses == nullptr) {
    return LUPINE_INVALID_SOCKET;
  }

  lupine_socket_t socket_fd = socket(
      addresses->ai_family, addresses->ai_socktype, addresses->ai_protocol);
  if (socket_fd != LUPINE_INVALID_SOCKET) {
    lupine_socket_apply_transport_options(socket_fd);
    if (connect(socket_fd, addresses->ai_addr,
                static_cast<socklen_t>(addresses->ai_addrlen)) != 0) {
      lupine_socket_close(socket_fd);
      socket_fd = LUPINE_INVALID_SOCKET;
    }
  }
  freeaddrinfo(addresses);
  return socket_fd;
}

lupine_socket_t dial_endpoint(lupine_client_transport *transport,
                              const lupine_client_endpoint &endpoint) {
  if (transport->config.dial != nullptr) {
    return transport->config.dial(&endpoint, transport->config.dial_policy,
                                  transport->config.user_data);
  }
  if (transport->config.dial_policy ==
      lupine_client_dial_policy::bounded_retry) {
    return lupine_tcp_connect(endpoint.host.c_str(), endpoint.port.c_str());
  }
  return dial_once(endpoint);
}

#ifdef LUPINE_TLS_OPENSSL
SSL_CTX *client_tls_context() {
  static SSL_CTX *context = [] {
    SSL_CTX *created = SSL_CTX_new(TLS_client_method());
    if (created != nullptr) {
      SSL_CTX_set_min_proto_version(created, TLS1_2_VERSION);
      SSL_CTX_set_default_verify_paths(created);
      SSL_CTX_set_verify(created, SSL_VERIFY_PEER, nullptr);
    }
    return created;
  }();
  return context;
}
#endif

int initialize_tls(conn_t *conn, const lupine_client_endpoint &endpoint) {
  if (!endpoint.tls) {
    return 0;
  }
#ifdef LUPINE_TLS_OPENSSL
  SSL_CTX *context = client_tls_context();
  SSL *session = context != nullptr ? SSL_new(context) : nullptr;
  if (session != nullptr &&
      SSL_set_tlsext_host_name(session, endpoint.host.c_str()) == 1 &&
      SSL_set1_host(session, endpoint.host.c_str()) == 1 &&
      SSL_set_fd(session, static_cast<int>(conn->connfd)) == 1 &&
      SSL_connect(session) == 1) {
    conn->tls_session = session;
    return 0;
  }
  if (session != nullptr) {
    SSL_free(session);
  }
  LUPINE_LOG_ERROR("TLS handshake with " << endpoint.host << " failed");
#else
  LUPINE_LOG_ERROR("LUPINE_SERVER entry "
                   << endpoint.host << ":" << endpoint.port
                   << " uses https:// but this client was built without TLS "
                      "support");
#endif
  return -1;
}

void free_tls(conn_t *conn) {
#ifdef LUPINE_TLS_OPENSSL
  if (conn->tls_session != nullptr) {
    SSL_free(static_cast<SSL *>(conn->tls_session));
    conn->tls_session = nullptr;
  }
#else
  (void)conn;
#endif
}

void notify_connection_closed(lupine_client_transport *transport,
                              conn_t *conn) {
  if (transport->config.connection_closed != nullptr) {
    transport->config.connection_closed(conn, transport->config.user_data);
  }
}

void close_connection(lupine_client_transport *transport, conn_t *conn) {
  if (conn == nullptr) {
    return;
  }
  notify_connection_closed(transport, conn);
  rpc_close_transport_socket(conn);
  pthread_mutex_lock(&conn->read_mutex);
  pthread_cond_broadcast(&conn->read_cond);
  pthread_mutex_unlock(&conn->read_mutex);
}

void *dispatch_connection(void *argument) {
  auto *slot = static_cast<lupine_client_connection_slot *>(argument);
  lupine_client_transport *transport = slot->transport;
  if (transport->config.dispatch != nullptr) {
    transport->config.dispatch(&slot->conn, transport->config.user_data);
  }
  close_connection(transport, &slot->conn);
  return nullptr;
}

void reset_failed_connection(conn_t *conn, bool read_mutex_initialized,
                             bool write_mutex_initialized,
                             bool call_mutex_initialized,
                             bool read_cond_initialized) {
  rpc_http2_destroy(conn);
  free_tls(conn);
  rpc_close_transport_socket(conn);
  rpc_write_queue_free(conn);
  if (read_cond_initialized) {
    pthread_cond_destroy(&conn->read_cond);
  }
  if (call_mutex_initialized) {
    pthread_mutex_destroy(&conn->call_mutex);
  }
  if (write_mutex_initialized) {
    pthread_mutex_destroy(&conn->write_mutex);
  }
  if (read_mutex_initialized) {
    pthread_mutex_destroy(&conn->read_mutex);
  }
  *conn = {};
  conn->connfd = LUPINE_INVALID_SOCKET;
}

int connect_endpoint(lupine_client_transport *transport,
                     const lupine_client_endpoint &endpoint,
                     unsigned int index) {
  lupine_client_connection_slot &slot = transport->slots[index];
  conn_t *conn = &slot.conn;
  *conn = {};
  conn->connfd = dial_endpoint(transport, endpoint);
  if (conn->connfd == LUPINE_INVALID_SOCKET) {
    LUPINE_LOG_ERROR("Connecting to " << endpoint.host << " port "
                                      << endpoint.port << " failed");
    return -1;
  }
  conn->local_request_parity = conn->request_id & 1;
  conn->logical_index = static_cast<int>(index);

  bool read_mutex_initialized = false;
  bool write_mutex_initialized = false;
  bool call_mutex_initialized = false;
  bool read_cond_initialized = false;
  if (initialize_tls(conn, endpoint) < 0 ||
      !(read_mutex_initialized =
            pthread_mutex_init(&conn->read_mutex, nullptr) == 0) ||
      !(write_mutex_initialized =
            pthread_mutex_init(&conn->write_mutex, nullptr) == 0) ||
      !(call_mutex_initialized =
            pthread_mutex_init(&conn->call_mutex, nullptr) == 0) ||
      !(read_cond_initialized =
            pthread_cond_init(&conn->read_cond, nullptr) == 0) ||
      rpc_http2_client_init(conn) < 0) {
    reset_failed_connection(conn, read_mutex_initialized,
                            write_mutex_initialized, call_mutex_initialized,
                            read_cond_initialized);
    return -1;
  }

  slot.transport = transport;
  transport->endpoints[index] = endpoint;
  if (transport->config.connection_opened != nullptr) {
    transport->config.connection_opened(conn, &transport->endpoints[index],
                                        transport->config.user_data);
  }
  if (pthread_create(&conn->read_thread, nullptr, dispatch_connection, &slot) !=
      0) {
    reset_failed_connection(conn, true, true, true, true);
    return -1;
  }
  rpc_http2_client_start_heartbeat(conn);
  return 0;
}

} // namespace

int lupine_client_parse_endpoints(
    const char *servers, const lupine_client_transport_config *config,
    std::vector<lupine_client_endpoint> *endpoints) {
  if (servers == nullptr || config == nullptr || endpoints == nullptr ||
      config->default_port == nullptr || config->default_port[0] == '\0') {
    return -1;
  }
  endpoints->clear();
  const std::string value(servers);
  size_t begin = 0;
  while (begin <= value.size()) {
    size_t comma = value.find(',', begin);
    std::string token = value.substr(begin, comma - begin);
    begin = comma == std::string::npos ? value.size() + 1 : comma + 1;
    if (config->strict_endpoints && token.empty()) {
      continue;
    }

    lupine_client_endpoint endpoint;
    endpoint.port = config->default_port;
    if (token.compare(0, 8, "https://") == 0) {
      endpoint.tls = true;
      endpoint.port = "443";
      token.erase(0, 8);
    } else if (token.compare(0, 7, "http://") == 0) {
      token.erase(0, 7);
    } else if (config->strict_endpoints && endpoint_has_invalid_scheme(token)) {
      LUPINE_LOG_ERROR("Invalid LUPINE_SERVER URL scheme: " << token);
      continue;
    }

    size_t colon = token.find(':');
    endpoint.host = token.substr(0, colon);
    if (colon != std::string::npos) {
      endpoint.port = token.substr(colon + 1);
    }
    if (config->strict_endpoints &&
        (endpoint.host.empty() || endpoint.port.empty())) {
      LUPINE_LOG_ERROR("Invalid LUPINE_SERVER endpoint");
      continue;
    }
    endpoint.label = endpoint.host;
    if (endpoint.port != config->default_port) {
      endpoint.label += ":" + endpoint.port;
    }
    endpoints->push_back(std::move(endpoint));
  }
  return endpoints->empty() ? -1 : 0;
}

lupine_client_transport *
lupine_client_transport_create(const lupine_client_transport_config *config) {
  if (config == nullptr || config->max_connections == 0 ||
      config->default_port == nullptr) {
    return nullptr;
  }
  return new lupine_client_transport(*config);
}

void lupine_client_transport_destroy(lupine_client_transport *transport) {
  if (transport == nullptr) {
    return;
  }
  lupine_client_transport_close(transport);
  pthread_mutex_destroy(&transport->mutex);
  delete transport;
}

int lupine_client_transport_open(lupine_client_transport *transport,
                                 const char *servers) {
  if (transport == nullptr || pthread_mutex_lock(&transport->mutex) != 0) {
    return -1;
  }
  if (transport->count != 0) {
    pthread_mutex_unlock(&transport->mutex);
    return 0;
  }
  if (servers == nullptr) {
    servers = getenv("LUPINE_SERVER");
  }
  if (servers == nullptr) {
    if (transport->config.log_missing_server) {
      LUPINE_LOG_ERROR("LUPINE_SERVER environment variable not set");
    }
    pthread_mutex_unlock(&transport->mutex);
    return -1;
  }

  std::vector<lupine_client_endpoint> endpoints;
  if (lupine_client_parse_endpoints(servers, &transport->config, &endpoints) <
      0) {
    pthread_mutex_unlock(&transport->mutex);
    return -1;
  }
  unsigned int limit =
      std::min(transport->config.max_connections, kTransportCapacity);
  for (const auto &endpoint : endpoints) {
    if (transport->count >= limit) {
      LUPINE_LOG_ERROR("Too many LUPINE_SERVER entries; ignoring the rest");
      break;
    }
    if (connect_endpoint(transport, endpoint, transport->count) == 0) {
      ++transport->count;
    }
  }
  int result = transport->count == 0 ? -1 : 0;
  pthread_mutex_unlock(&transport->mutex);
  return result;
}

void lupine_client_transport_close_connection(
    lupine_client_transport *transport, conn_t *conn) {
  if (transport != nullptr && conn != nullptr) {
    close_connection(transport, conn);
  }
}

void lupine_client_transport_close(lupine_client_transport *transport) {
  if (transport == nullptr || pthread_mutex_lock(&transport->mutex) != 0) {
    return;
  }
  if (transport->shutting_down) {
    pthread_mutex_unlock(&transport->mutex);
    return;
  }
  transport->shutting_down = true;
  unsigned int count = transport->count;
  for (unsigned int i = 0; i < count; ++i) {
    close_connection(transport, &transport->slots[i].conn);
  }
  pthread_mutex_unlock(&transport->mutex);

  for (unsigned int i = 0; i < count; ++i) {
    conn_t *conn = &transport->slots[i].conn;
    if (conn->read_thread != 0) {
      pthread_join(conn->read_thread, nullptr);
      conn->read_thread = 0;
    }
    if (conn->rpc_thread != 0) {
      pthread_join(conn->rpc_thread, nullptr);
      conn->rpc_thread = 0;
    }
    free_tls(conn);
    rpc_conn_destroy(conn);
    *conn = {};
    conn->connfd = LUPINE_INVALID_SOCKET;
    transport->slots[i].transport = nullptr;
    transport->endpoints[i] = {};
  }

  if (pthread_mutex_lock(&transport->mutex) == 0) {
    transport->count = 0;
    transport->shutting_down = false;
    pthread_mutex_unlock(&transport->mutex);
  }
}

unsigned int
lupine_client_transport_size(const lupine_client_transport *transport) {
  return transport == nullptr ? 0 : transport->count;
}

bool lupine_client_transport_is_open(const lupine_client_transport *transport) {
  return lupine_client_transport_size(transport) != 0;
}

conn_t *lupine_client_transport_connection(lupine_client_transport *transport,
                                           unsigned int index) {
  if (transport == nullptr || index >= transport->count) {
    return nullptr;
  }
  return &transport->slots[index].conn;
}

const lupine_client_endpoint *
lupine_client_transport_endpoint(const lupine_client_transport *transport,
                                 unsigned int index) {
  if (transport == nullptr || index >= transport->count) {
    return nullptr;
  }
  return &transport->endpoints[index];
}

void lupine_client_transport_retire_lane(lupine_client_transport *transport,
                                         uint64_t lane_id) {
  if (transport == nullptr || pthread_mutex_lock(&transport->mutex) != 0) {
    return;
  }
  std::array<conn_t *, kTransportCapacity> active = {};
  unsigned int count = 0;
  if (!transport->shutting_down) {
    for (unsigned int i = 0; i < transport->count; ++i) {
      conn_t *conn = &transport->slots[i].conn;
      if (!conn->closed) {
        active[count++] = conn;
      }
    }
  }
  pthread_mutex_unlock(&transport->mutex);

  for (unsigned int i = 0; i < count; ++i) {
    rpc_write_lane_termination(active[i], lane_id);
  }
}
