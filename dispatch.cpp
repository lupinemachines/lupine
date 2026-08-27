// Protocol dispatch for the server's single listening port. Lupine clients
// speak prior-knowledge HTTP/2 and always open with the fixed 24-byte client
// preface, so the first bytes of a connection identify the protocol without
// consuming anything. Everything else on the port is treated as one plain
// HTTP/1.x request so that curl, load balancer health checks, and future
// HTTP/1.1-only consumers such as Prometheus can share the RPC port. The
// RPC/non-RPC split is decided by the request route, not the HTTP version:
// this file only detects the framing.

#include "dispatch.h"

#include <chrono>
#include <cstring>
#include <string>
#include <thread>

#include "lupine_log.h"

namespace {

constexpr char kH2Preface[] = "PRI * HTTP/2.0\r\n\r\nSM\r\n\r\n";
constexpr size_t kH2PrefaceLength = sizeof(kH2Preface) - 1;
// Bounds how long a connection may sit unidentified. Lupine clients send the
// preface immediately after connecting, so only a silent or dribbling peer
// ever waits this long.
constexpr int kDispatchTimeoutMs = 5000;
constexpr size_t kMaxHttp1RequestBytes = 8192;

int remaining_ms(std::chrono::steady_clock::time_point deadline) {
  auto remaining = std::chrono::duration_cast<std::chrono::milliseconds>(
      deadline - std::chrono::steady_clock::now());
  return static_cast<int>(remaining.count());
}

int wait_readable(lupine_socket_t fd, int timeout_ms) {
#ifdef _WIN32
  fd_set readable;
  FD_ZERO(&readable);
  FD_SET(fd, &readable);
  timeval timeout = {timeout_ms / 1000, (timeout_ms % 1000) * 1000};
  return select(0, &readable, nullptr, nullptr, &timeout);
#else
  struct pollfd descriptor = {fd, POLLIN, 0};
  return poll(&descriptor, 1, timeout_ms);
#endif
}

ssize_t peek_bytes(lupine_socket_t fd, unsigned char *data, size_t size) {
#ifdef _WIN32
  return recv(fd, reinterpret_cast<char *>(data), static_cast<int>(size),
              MSG_PEEK);
#else
  return recv(fd, data, size, MSG_PEEK);
#endif
}

bool send_all(lupine_socket_t fd, const std::string &data) {
  size_t offset = 0;
  while (offset < data.size()) {
    struct iovec buffer = {const_cast<char *>(data.data()) + offset,
                           data.size() - offset};
    ssize_t sent = lupine_socket_sendv(fd, &buffer, 1);
    if (sent <= 0) {
      return false;
    }
    offset += static_cast<size_t>(sent);
  }
  return true;
}

std::string http1_response(int status, const char *reason,
                           const std::string &body, bool include_body,
                           const rpc_http2_server_metadata *metadata) {
  std::string response = "HTTP/1.1 " + std::to_string(status) + " " + reason +
                         "\r\n";
  if (metadata != nullptr && metadata->backend_version != nullptr &&
      metadata->backend_version[0] != '\0') {
    // Same header the HTTP/2 version probe answers with, so an HTTP/1.1
    // client learns the backend version the same way.
    response += "x-lupine-cuda-version: ";
    response += metadata->backend_version;
    response += "\r\n";
  }
  response += "content-type: text/plain\r\n";
  response += "content-length: " + std::to_string(body.size()) + "\r\n";
  response += "connection: close\r\n\r\n";
  if (include_body) {
    response += body;
  }
  return response;
}

int serve_http1(lupine_socket_t fd, const rpc_http2_server_metadata *metadata,
                std::chrono::steady_clock::time_point deadline) {
  std::string request;
  while (request.find("\r\n\r\n") == std::string::npos) {
    if (request.size() >= kMaxHttp1RequestBytes) {
      return -1;
    }
    int timeout_ms = remaining_ms(deadline);
    if (timeout_ms <= 0 || wait_readable(fd, timeout_ms) <= 0) {
      return -1;
    }
    char chunk[1024];
    ssize_t received = lupine_socket_recv(fd, chunk, sizeof(chunk));
    if (received < 0 && lupine_socket_error_is_intr()) {
      continue;
    }
    if (received <= 0) {
      return -1;
    }
    request.append(chunk, static_cast<size_t>(received));
  }

  std::string line = request.substr(0, request.find("\r\n"));
  size_t method_end = line.find(' ');
  size_t path_end =
      method_end == std::string::npos ? std::string::npos
                                      : line.find(' ', method_end + 1);
  if (path_end == std::string::npos) {
    send_all(fd, http1_response(400, "Bad Request", "bad request\n", true,
                                metadata));
    return 1;
  }
  std::string method = line.substr(0, method_end);
  std::string path = line.substr(method_end + 1, path_end - method_end - 1);
  LUPINE_LOG_DEBUG("HTTP/1.1 " << method << " " << path);

  bool head = method == "HEAD";
  if (path == "/" && (head || method == "GET")) {
    send_all(fd, http1_response(200, "OK", "lupine\n", !head, metadata));
  } else {
    send_all(fd,
             http1_response(404, "Not Found", "not found\n", !head, metadata));
  }
  return 1;
}

} // namespace

int lupine_h2_preface_check(const unsigned char *data, size_t len) {
  size_t compare = len < kH2PrefaceLength ? len : kH2PrefaceLength;
  if (memcmp(data, kH2Preface, compare) != 0) {
    return -1;
  }
  return len >= kH2PrefaceLength ? 1 : 0;
}

int lupine_connection_dispatch(lupine_socket_t connfd,
                               const rpc_http2_server_metadata *metadata) {
  auto deadline = std::chrono::steady_clock::now() +
                  std::chrono::milliseconds(kDispatchTimeoutMs);
  bool waited_for_first_byte = false;
  for (;;) {
    int timeout_ms = remaining_ms(deadline);
    if (timeout_ms <= 0) {
      return -1;
    }
    if (!waited_for_first_byte) {
      if (wait_readable(connfd, timeout_ms) <= 0) {
        return -1;
      }
      waited_for_first_byte = true;
    }
    unsigned char data[kH2PrefaceLength];
    ssize_t received = peek_bytes(connfd, data, sizeof(data));
    if (received < 0 && lupine_socket_error_is_intr()) {
      continue;
    }
    if (received <= 0) {
      return -1;
    }
    int verdict =
        lupine_h2_preface_check(data, static_cast<size_t>(received));
    if (verdict > 0) {
      return 0;
    }
    if (verdict < 0) {
      return serve_http1(connfd, metadata, deadline);
    }
    // A preface prefix shorter than 24 bytes: the bytes stay queued because
    // of MSG_PEEK, so polling reports readable immediately. Sleep instead of
    // spinning while the remainder is in flight.
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }
}
