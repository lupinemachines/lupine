#include "transport.h"

#include "address_space.h"
#include "lupine_log.h"

#include <algorithm>
#include <array>
#include <cstdlib>
#include <vector>

#ifdef LUPINE_TLS_OPENSSL
#include <openssl/ssl.h>
#endif

namespace {

constexpr unsigned int kTransportCapacity = 16;
constexpr unsigned int kBoundedRetryCount = 5;
constexpr const char *kDefaultPort = "14833";

struct client_transport_state {
  lupine_client_transport_config config;
  pthread_mutex_t mutex = PTHREAD_MUTEX_INITIALIZER;
  std::array<conn_t, kTransportCapacity> connections = {};
  std::array<lupine_client_endpoint, kTransportCapacity> endpoints;
  unsigned int count = 0;
  bool shutting_down = false;
};

client_transport_state &transport() {
  static auto *state = new client_transport_state;
  return *state;
}

bool endpoint_has_invalid_scheme(const std::string &endpoint) {
  return endpoint.find("://") != std::string::npos ||
         endpoint.compare(0, 5, "http:") == 0 ||
         endpoint.compare(0, 6, "https:") == 0;
}

int parse_endpoints(const char *servers, bool strict,
                    std::vector<lupine_client_endpoint> *endpoints) {
  endpoints->clear();
  const std::string value(servers);
  size_t begin = 0;
  while (begin <= value.size()) {
    size_t comma = value.find(',', begin);
    std::string token = value.substr(begin, comma - begin);
    begin = comma == std::string::npos ? value.size() + 1 : comma + 1;
    if (strict && token.empty()) {
      continue;
    }

    lupine_client_endpoint endpoint;
    endpoint.port = kDefaultPort;
    if (token.compare(0, 8, "https://") == 0) {
      endpoint.tls = true;
      endpoint.port = "443";
      token.erase(0, 8);
    } else if (token.compare(0, 7, "http://") == 0) {
      token.erase(0, 7);
    } else if (strict && endpoint_has_invalid_scheme(token)) {
      LUPINE_LOG_ERROR("Invalid LUPINE_SERVER URL scheme: " << token);
      continue;
    }

    size_t colon = token.find(':');
    endpoint.host = token.substr(0, colon);
    if (colon != std::string::npos) {
      endpoint.port = token.substr(colon + 1);
    }
    if (strict && (endpoint.host.empty() || endpoint.port.empty())) {
      LUPINE_LOG_ERROR("Invalid LUPINE_SERVER endpoint");
      continue;
    }
    endpoint.label = endpoint.host;
    if (endpoint.port != kDefaultPort) {
      endpoint.label += ":" + endpoint.port;
    }
    endpoints->push_back(std::move(endpoint));
  }
  return endpoints->empty() ? -1 : 0;
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

void reset_connection(conn_t *conn) {
  rpc_conn_destroy(conn);
  free_tls(conn);
  *conn = {};
  conn->connfd = LUPINE_INVALID_SOCKET;
}

void *dispatch_connection(void *argument) {
  auto *conn = static_cast<conn_t *>(argument);
  auto &state = transport();
  if (state.config.dispatch != nullptr) {
    state.config.dispatch(conn);
  }
  lupine_client_transport_close_connection(conn);
  return nullptr;
}

int connect_endpoint(client_transport_state &state,
                     const lupine_client_endpoint &endpoint,
                     unsigned int index) {
  conn_t *conn = &state.connections[index];
  unsigned int min_slot = 0;
  for (;;) {
    *conn = {};
    unsigned int retries =
        state.config.dial_policy == lupine_client_dial_policy::bounded_retry
            ? kBoundedRetryCount
            : 0;
    lupine_socket_t connfd = lupine_tcp_connect(endpoint.host.c_str(),
                                                endpoint.port.c_str(), retries);
    if (connfd == LUPINE_INVALID_SOCKET) {
      LUPINE_LOG_ERROR("Connecting to " << endpoint.host << " port "
                                        << endpoint.port << " failed");
      return -1;
    }
    if (rpc_conn_init(conn, connfd, 0) < 0) {
      return -1;
    }
    conn->logical_index = static_cast<int>(index);
    conn->w_offset = state.config.w_offset;
    unsigned int slot = min_slot;
    if (conn->w_offset != 0) {
      int reserve_result = lupine_va_reserve_client(conn, min_slot, &slot);
      if (reserve_result < 0) {
        reset_connection(conn);
        return -1;
      }
    }
    if (initialize_tls(conn, endpoint) < 0) {
      reset_connection(conn);
      return -1;
    }
    int http2_result = rpc_http2_client_init(conn);
    if (http2_result == LUPINE_RPC_HTTP2_VA_CONFLICT && conn->va_size != 0 &&
        slot + 1 < LUPINE_VA_ARENA_COUNT) {
      min_slot = slot + 1;
      reset_connection(conn);
      continue;
    }
    if (http2_result < 0) {
      reset_connection(conn);
      return -1;
    }
    break;
  }

  state.endpoints[index] = endpoint;
  if (state.config.connection_opened != nullptr) {
    state.config.connection_opened(conn);
  }
  if (pthread_create(&conn->read_thread, nullptr, dispatch_connection, conn) !=
      0) {
    reset_connection(conn);
    return -1;
  }
  rpc_http2_client_start_heartbeat(conn);
  return 0;
}

} // namespace

int lupine_client_transport_open(const lupine_client_transport_config &config,
                                 const char *servers) {
  auto &state = transport();
  if (pthread_mutex_lock(&state.mutex) != 0) {
    return -1;
  }
  if (state.count != 0) {
    pthread_mutex_unlock(&state.mutex);
    return 0;
  }
  if (servers == nullptr) {
    servers = getenv("LUPINE_SERVER");
  }
  if (servers == nullptr) {
    if (config.log_missing_server) {
      LUPINE_LOG_ERROR("LUPINE_SERVER environment variable not set");
    }
    pthread_mutex_unlock(&state.mutex);
    return -1;
  }

  std::vector<lupine_client_endpoint> endpoints;
  if (parse_endpoints(servers, config.strict_endpoints, &endpoints) < 0) {
    pthread_mutex_unlock(&state.mutex);
    return -1;
  }
  state.config = config;
  for (const auto &endpoint : endpoints) {
    if (state.count == kTransportCapacity) {
      LUPINE_LOG_ERROR("Too many LUPINE_SERVER entries; ignoring the rest");
      break;
    }
    if (connect_endpoint(state, endpoint, state.count) == 0) {
      ++state.count;
    }
  }
  int result = state.count == 0 ? -1 : 0;
  pthread_mutex_unlock(&state.mutex);
  return result;
}

void lupine_client_transport_close_connection(conn_t *conn) {
  if (conn == nullptr) {
    return;
  }
  auto &state = transport();
  if (state.config.connection_closed != nullptr) {
    state.config.connection_closed(conn);
  }
  rpc_shutdown_transport_socket(conn);
}

void lupine_client_transport_close() {
  auto &state = transport();
  if (pthread_mutex_lock(&state.mutex) != 0) {
    return;
  }
  if (state.shutting_down) {
    pthread_mutex_unlock(&state.mutex);
    return;
  }
  state.shutting_down = true;
  unsigned int count = state.count;
  for (unsigned int i = 0; i < count; ++i) {
    lupine_client_transport_close_connection(&state.connections[i]);
  }
  pthread_mutex_unlock(&state.mutex);

  for (unsigned int i = 0; i < count; ++i) {
    conn_t *conn = &state.connections[i];
    if (conn->read_thread != 0) {
      pthread_join(conn->read_thread, nullptr);
      conn->read_thread = 0;
    }
    reset_connection(conn);
    state.endpoints[i] = {};
  }

  if (pthread_mutex_lock(&state.mutex) == 0) {
    state.count = 0;
    state.shutting_down = false;
    pthread_mutex_unlock(&state.mutex);
  }
}

unsigned int lupine_client_transport_size() { return transport().count; }

conn_t *lupine_client_transport_connection(unsigned int index) {
  auto &state = transport();
  return index < state.count ? &state.connections[index] : nullptr;
}

const lupine_client_endpoint *
lupine_client_transport_endpoint(unsigned int index) {
  auto &state = transport();
  return index < state.count ? &state.endpoints[index] : nullptr;
}

void lupine_client_transport_retire_lane(uint64_t lane_id) {
  auto &state = transport();
  if (pthread_mutex_lock(&state.mutex) != 0) {
    return;
  }
  std::array<conn_t *, kTransportCapacity> active = {};
  unsigned int count = 0;
  if (!state.shutting_down) {
    for (unsigned int i = 0; i < state.count; ++i) {
      if (!state.connections[i].closed) {
        active[count++] = &state.connections[i];
      }
    }
  }
  pthread_mutex_unlock(&state.mutex);

  for (unsigned int i = 0; i < count; ++i) {
    rpc_write_lane_termination(active[i], lane_id);
  }
}
