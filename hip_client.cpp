#include <arpa/inet.h>
#include <netdb.h>
#include <netinet/tcp.h>
#ifdef LUPINE_TLS_OPENSSL
#include <openssl/ssl.h>
#endif
#include <pthread.h>
#include <sys/socket.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "hip_compat.h"

#include "codegen/gen_api.h"
#include "lupine_log.h"
#include "rpc.h"

namespace {

constexpr const char *DEFAULT_PORT = "14833";

pthread_mutex_t conn_mutex = PTHREAD_MUTEX_INITIALIZER;
conn_t conns[16] = {};
int nconns = 0;
bool connected = false;

// HIP exposes devices as plain int ordinals (unlike NVML's opaque
// nvmlDevice_t handles), so the virtual device table just remembers which
// connection a virtual ordinal lives on and what the remote ordinal is.
struct lupine_hip_remote_device {
  unsigned int conn_index = 0;
  int remote_ordinal = 0;
};

std::vector<lupine_hip_remote_device> devices;
bool devices_ready = false;

// hipInit is optional in the HIP runtime (the runtime auto-initializes on
// first use), so unlike the NVML shim we do not gate calls on an init
// reference count. We still forward hipInit to the server and keep a count so
// a future hipDeviceReset-style teardown has somewhere to hook.
std::atomic<int> init_refcount{0};

hipError_t rpc_error() { return hipErrorUnknown; }

void *rpc_client_dispatch_thread(void *p) {
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

int open_connection() {
  if (pthread_mutex_lock(&conn_mutex) < 0) {
    return -1;
  }
  if (connected) {
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

    bool tls = false;
    if (strncmp(token, "https://", 8) == 0) {
      tls = true;
      token += 8;
    } else if (strncmp(token, "http://", 7) == 0) {
      token += 7;
    } else if (strstr(token, "://") != nullptr ||
               strncmp(token, "http:", 5) == 0 ||
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

    addrinfo hints = {};
    hints.ai_family = AF_INET;
    hints.ai_socktype = SOCK_STREAM;
    addrinfo *res = nullptr;
    if (getaddrinfo(host, port, &hints, &res) != 0) {
      continue;
    }

    int sockfd = socket(res->ai_family, res->ai_socktype, res->ai_protocol);
    if (sockfd >= 0) {
      lupine_socket_apply_transport_options(sockfd);
      if (connect(sockfd, res->ai_addr, res->ai_addrlen) == 0) {
        if (nconns >= static_cast<int>(sizeof(conns) / sizeof(conns[0]))) {
          close(sockfd);
          freeaddrinfo(res);
          break;
        }

        conn_t *c = &conns[nconns];
        *c = {};
        c->connfd = sockfd;
        c->request_id = 0;
        c->local_request_parity = c->request_id & 1;
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
            close(sockfd);
            freeaddrinfo(res);
            continue;
          }
          c->tls_session = ssl;
#else
          LUPINE_LOG_ERROR("LUPINE_SERVER entry "
                           << host << ":" << port
                           << " uses https:// but this client was built "
                              "without TLS support");
          close(sockfd);
          freeaddrinfo(res);
          continue;
#endif
        }
        if (pthread_mutex_init(&c->read_mutex, nullptr) < 0 ||
            pthread_mutex_init(&c->write_mutex, nullptr) < 0 ||
            pthread_mutex_init(&c->call_mutex, nullptr) < 0 ||
            pthread_cond_init(&c->read_cond, nullptr) < 0 ||
            rpc_http2_client_init(c) < 0 ||
            pthread_create(&c->read_thread, nullptr, rpc_client_dispatch_thread,
                           c) < 0) {
#ifdef LUPINE_TLS_OPENSSL
          if (c->tls_session != nullptr) {
            SSL_free(static_cast<SSL *>(c->tls_session));
            c->tls_session = nullptr;
          }
#endif
          close(sockfd);
          freeaddrinfo(res);
          continue;
        }
        ++nconns;
        freeaddrinfo(res);
        continue;
      }
      close(sockfd);
    }
    freeaddrinfo(res);
  }
  free(servers);

  if (nconns == 0) {
    pthread_mutex_unlock(&conn_mutex);
    return -1;
  }

  connected = true;
  pthread_mutex_unlock(&conn_mutex);
  return 0;
}

conn_t *connection(unsigned int index = 0) {
  if (open_connection() < 0) {
    return nullptr;
  }
  if (index >= static_cast<unsigned int>(nconns)) {
    return nullptr;
  }
  return &conns[index];
}

void close_connections() {
  if (pthread_mutex_lock(&conn_mutex) != 0) {
    return;
  }
  int count = nconns;
  for (int i = 0; i < count; ++i) {
    conn_t *c = &conns[i];
    if (!c->closed) {
      c->closed = 1;
      shutdown(c->connfd, SHUT_RDWR);
      close(c->connfd);
    }
    pthread_mutex_lock(&c->read_mutex);
    pthread_cond_broadcast(&c->read_cond);
    pthread_mutex_unlock(&c->read_mutex);
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
    connected = false;
    devices_ready = false;
    devices.clear();
    pthread_mutex_unlock(&conn_mutex);
  }
}

// hipGetDeviceCount/hipDeviceGet are @disabled client (the client fabricates
// the virtual device table itself), so the codegen emits no lupine_rpc_
// helpers for them. These round-trip the server directly to enumerate the
// remote device table.
hipError_t call_int_out_on(conn_t *c, int op, int *value) {
  hipError_t result = rpc_error();
  int temp = 0;
  if (c == nullptr || rpc_write_start_request(c, op) < 0 ||
      rpc_wait_for_response(c) < 0 || rpc_read(c, &temp, sizeof(temp)) < 0 ||
      rpc_read(c, &result, sizeof(result)) < 0 || rpc_read_end(c) < 0) {
    return rpc_error();
  }
  if (value != nullptr) {
    *value = temp;
  }
  return result;
}

hipError_t call_device_get_on(conn_t *c, int op, int ordinal, int *device) {
  hipError_t result = rpc_error();
  int temp = 0;
  if (c == nullptr || rpc_write_start_request(c, op) < 0 ||
      rpc_write(c, &ordinal, sizeof(ordinal)) < 0 ||
      rpc_wait_for_response(c) < 0 || rpc_read(c, &temp, sizeof(temp)) < 0 ||
      rpc_read(c, &result, sizeof(result)) < 0 || rpc_read_end(c) < 0) {
    return rpc_error();
  }
  if (device != nullptr) {
    *device = temp;
  }
  return result;
}

hipError_t ensure_devices() {
  if (open_connection() < 0) {
    return rpc_error();
  }
  if (devices_ready) {
    return hipSuccess;
  }

  devices.clear();
  for (int i = 0; i < nconns; ++i) {
    int count = 0;
    hipError_t result =
        call_int_out_on(&conns[i], RPC_hipGetDeviceCount, &count);
    if (result != hipSuccess) {
      devices.clear();
      return result;
    }
    for (int ordinal = 0; ordinal < count; ++ordinal) {
      int remote = 0;
      result =
          call_device_get_on(&conns[i], RPC_hipDeviceGet, ordinal, &remote);
      if (result != hipSuccess) {
        devices.clear();
        return result;
      }
      devices.push_back(
          lupine_hip_remote_device{static_cast<unsigned int>(i), remote});
    }
  }
  devices_ready = true;
  return hipSuccess;
}

// Rewrites a virtual device ordinal (index into the device table) into the
// remote ordinal the chosen connection understands, and returns that
// connection. Mirrors nvml_client's connection_for_device, but with int
// ordinals instead of opaque nvmlDevice_t handles.
conn_t *connection_for_device(int *device) {
  if (device == nullptr || ensure_devices() != hipSuccess) {
    return nullptr;
  }
  if (devices.empty()) {
    return nullptr;
  }
  int virt = *device;
  if (virt < 0 || virt >= static_cast<int>(devices.size())) {
    return nullptr;
  }
  const auto &mapped = devices[virt];
  *device = mapped.remote_ordinal;
  return connection(mapped.conn_index);
}

} // namespace

#include "codegen/gen_hip_client.inc"

extern "C" hipError_t hipInit(unsigned int flags) {
  init_refcount.fetch_add(1, std::memory_order_acq_rel);
  if (open_connection() < 0) {
    init_refcount.fetch_sub(1, std::memory_order_acq_rel);
    return hipErrorUnknown;
  }
  hipError_t first_error = hipSuccess;
  for (int i = 0; i < nconns; ++i) {
    conn_t *c = &conns[i];
    hipError_t result = rpc_error();
    if (rpc_write_start_request(c, RPC_hipInit) < 0 ||
        rpc_write(c, &flags, sizeof(flags)) < 0 ||
        rpc_wait_for_response(c) < 0 ||
        rpc_read(c, &result, sizeof(result)) < 0 || rpc_read_end(c) < 0) {
      result = rpc_error();
    }
    if (result != hipSuccess && first_error == hipSuccess) {
      first_error = result;
    }
  }
  devices_ready = false;
  devices.clear();
  return first_error;
}

extern "C" hipError_t hipGetDeviceCount(int *count) {
  hipError_t result = ensure_devices();
  if (result != hipSuccess) {
    return result;
  }
  if (count == nullptr) {
    return hipErrorInvalidValue;
  }
  *count = static_cast<int>(devices.size());
  return hipSuccess;
}

extern "C" hipError_t hipDeviceGet(int *device, int ordinal) {
  hipError_t result = ensure_devices();
  if (result != hipSuccess) {
    return result;
  }
  if (device == nullptr) {
    return hipErrorInvalidValue;
  }
  if (ordinal < 0 || ordinal >= static_cast<int>(devices.size())) {
    return hipErrorInvalidDevice;
  }
  // The virtual ordinal is the table index; hipDeviceGet just validates it.
  *device = ordinal;
  return hipSuccess;
}