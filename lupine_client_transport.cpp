#include "lupine_client_transport.h"

#include <cstring>
#include <cstdlib>

#ifdef LUPINE_TLS_OPENSSL
#include <openssl/ssl.h>
#endif

#include "lupine_log.h"
#include "rpc.h"

// Pool state. Each client .so links its own copy of this TU, so each shim gets
// an independent pool. PTHREAD_MUTEX_INITIALIZER avoids any runtime init-order
// hazard with the CUDA client's __attribute__((destructor)) teardown, which
// calls lupine_transport_close_all during process exit.
pthread_mutex_t conn_mutex = PTHREAD_MUTEX_INITIALIZER;
conn_t conns[16] = {};
int nconns = 0;

namespace {
constexpr const char *DEFAULT_PORT = "14833";
} // namespace

void *lupine_transport_dispatch_thread(void *p) {
  conn_t *connection = static_cast<conn_t *>(p);
  while (!connection->closed) {
    int op = rpc_dispatch(connection, 1);
    if (op < 0 || connection->closed) {
      break;
    }
    if (rpc_read_end(connection) < 0) {
      break;
    }
  }
  return nullptr;
}

int lupine_transport_open(const lupine_transport_hooks *hooks) {
  if (pthread_mutex_lock(&conn_mutex) < 0) {
    return -1;
  }
  if (nconns > 0) {
    pthread_mutex_unlock(&conn_mutex);
    return 0;
  }

  char *servers_env = getenv("LUPINE_SERVER");
  if (servers_env == nullptr) {
    LUPINE_LOG_ERROR("LUPINE_SERVER environment variable not set");
    pthread_mutex_unlock(&conn_mutex);
    return -1;
  }

  char *servers = strdup(servers_env);
  if (servers == nullptr) {
    pthread_mutex_unlock(&conn_mutex);
    return -1;
  }

  char *cursor = servers;
  char *token = nullptr;
  while ((token = strsep(&cursor, ",")) != nullptr) {
    if (token[0] == '\0') {
      continue;
    }
    if (nconns >= static_cast<int>(sizeof(conns) / sizeof(conns[0]))) {
      LUPINE_LOG_ERROR("Too many LUPINE_SERVER entries; ignoring the rest");
      break;
    }

    bool tls = false;
    if (strncmp(token, "https://", 8) == 0) {
      tls = true;
      token += 8;
    } else if (strncmp(token, "http://", 7) == 0) {
      token += 7;
    } else if (strstr(token, "://") != nullptr ||
               strncmp(token, "http:",  5) == 0 ||
               strncmp(token, "https:", 6) == 0) {
      LUPINE_LOG_ERROR("Invalid LUPINE_SERVER URL scheme: " << token);
      continue;
    }

    char *host = token;
    char *port = const_cast<char *>(tls ? "443" : DEFAULT_PORT);
    char *colon = strchr(token, ':');
    if (colon != nullptr) {
      *colon = '\0';
      port = colon + 1;
    }
    if (host[0] == '\0' || port[0] == '\0') {
      LUPINE_LOG_ERROR("Invalid LUPINE_SERVER endpoint");
      continue;
    }

    lupine_socket_t sockfd = lupine_tcp_connect(host, port);
    if (sockfd == LUPINE_INVALID_SOCKET) {
      LUPINE_LOG_ERROR("Connecting to " << host << " port " << port << " failed");
      continue;
    }

    conn_t *c = &conns[nconns];
    rpc_write_queue_free(c);
    *c = {};
    c->connfd = sockfd;
    c->request_id = 0;
    c->local_request_parity = c->request_id & 1;
    c->logical_index = nconns;
    if (tls) {
#ifdef LUPINE_TLS_OPENSSL
      static SSL_CTX *tls_ctx = []() {
        SSL_CTX *ctx = SSL_CTX_new(TLS_client_method());
        if (ctx != nullptr) {
          SSL_CTX_set_min_proto_version(ctx, TLS1_2_VERSION);
          SSL_CTX_set_default_verify_paths(ctx);
          SSL_CTX_set_verify(ctx, SSL_VERIFY_PEER, nullptr);
        }
        return ctx;
      }();
      SSL *ssl = tls_ctx != nullptr ? SSL_new(tls_ctx) : nullptr;
      if (ssl == nullptr || SSL_set_tlsext_host_name(ssl, host) != 1 ||
          SSL_set1_host(ssl, host) != 1 || SSL_set_fd(ssl, sockfd) != 1 ||
          SSL_connect(ssl) != 1) {
        if (ssl != nullptr) {
          SSL_free(ssl);
        }
        LUPINE_LOG_ERROR("TLS handshake with " << host << " failed");
        lupine_socket_close(sockfd);
        continue;
      }
      c->tls_session = ssl;
#else
      LUPINE_LOG_ERROR("LUPINE_SERVER entry "
                       << host << ":" << port
                       << " uses https:// but this client was built "
                          "without TLS support");
      lupine_socket_close(sockfd);
      continue;
#endif
    }
    if (pthread_mutex_init(&c->read_mutex, nullptr) < 0 ||
        pthread_mutex_init(&c->write_mutex, nullptr) < 0 ||
        pthread_mutex_init(&c->call_mutex, nullptr) < 0 ||
        pthread_cond_init(&c->read_cond, nullptr) < 0 ||
        rpc_http2_client_init(c) < 0 ||
        pthread_create(&c->read_thread, nullptr, hooks->dispatch_thread, c) < 0) {
#ifdef LUPINE_TLS_OPENSSL
      if (c->tls_session != nullptr) {
        SSL_free(static_cast<SSL *>(c->tls_session));
        c->tls_session = nullptr;
      }
#endif
      lupine_socket_close(sockfd);
      continue;
    }

    if (hooks->on_connect != nullptr) {
      hooks->on_connect(c, host, port, tls ? 1 : 0);
    }
    ++nconns;
  }
  free(servers);

  if (nconns == 0) {
    pthread_mutex_unlock(&conn_mutex);
    return -1;
  }

  pthread_mutex_unlock(&conn_mutex);
  return 0;
}

void lupine_transport_close_all(void (*on_close_conn)(conn_t *)) {
  if (pthread_mutex_lock(&conn_mutex) != 0) {
    return;
  }
  int count = nconns;
  for (int i = 0; i < count; ++i) {
    conn_t *c = &conns[i];
    if (on_close_conn != nullptr) {
      on_close_conn(c);
    } else if (!c->closed) {
      c->closed = 1;
      shutdown(c->connfd, SHUT_RDWR);
      lupine_socket_close(c->connfd);
      pthread_mutex_lock(&c->read_mutex);
      pthread_cond_broadcast(&c->read_cond);
      pthread_mutex_unlock(&c->read_mutex);
    } else {
      // Already closed: still wake any reader parked on the cond so its
      // dispatch thread observes the closed flag and exits for the join below.
      pthread_mutex_lock(&c->read_mutex);
      pthread_cond_broadcast(&c->read_cond);
      pthread_mutex_unlock(&c->read_mutex);
    }
  }
  pthread_mutex_unlock(&conn_mutex);

  for (int i = 0; i < count; ++i) {
    conn_t *c = &conns[i];
    if (c->read_thread != 0) {
      pthread_join(c->read_thread, nullptr);
      c->read_thread = 0;
    }
    if (c->rpc_thread != 0) {
      pthread_join(c->rpc_thread, nullptr);
      c->rpc_thread = 0;
    }
#ifdef LUPINE_TLS_OPENSSL
    if (c->tls_session != nullptr) {
      SSL_free(static_cast<SSL *>(c->tls_session));
      c->tls_session = nullptr;
    }
#endif
    rpc_conn_destroy(c);
  }

  if (pthread_mutex_lock(&conn_mutex) == 0) {
    nconns = 0;
    pthread_mutex_unlock(&conn_mutex);
  }
}