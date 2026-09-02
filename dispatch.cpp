// Protocol dispatch for the server's single listening port. Lupine clients
// speak prior-knowledge HTTP/2 and always open with the fixed 24-byte client
// preface, so the first bytes of a connection identify the protocol without
// consuming anything. Everything else on the port is treated as one plain
// HTTP/1.x request so that curl, load balancer health checks, and future
// HTTP/1.1-only consumers such as Prometheus can share the RPC port. The
// RPC/non-RPC split is decided by the request route, not the HTTP version:
// this file only detects the framing.
//
// Reads block without an application deadline. Dispatch runs inside the
// per-connection child, so a stalled peer only occupies its own child until
// the keepalive options from lupine_socket_apply_transport_options declare
// the peer dead and recv fails.

#include "dispatch.h"

#include <algorithm>
#include <chrono>
#include <cstring>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "client_bundle.h"
#include "lupine_log.h"

namespace {

constexpr char kH2Preface[] = "PRI * HTTP/2.0\r\n\r\nSM\r\n\r\n";
constexpr size_t kH2PrefaceLength = sizeof(kH2Preface) - 1;
constexpr size_t kMaxHttp1RequestBytes = 8192;

ssize_t peek_bytes(lupine_socket_t fd, unsigned char *data, size_t size) {
#ifdef _WIN32
  return recv(fd, reinterpret_cast<char *>(data), static_cast<int>(size),
              MSG_PEEK);
#else
  return recv(fd, data, size, MSG_PEEK);
#endif
}

bool send_all(lupine_socket_t fd, const char *data, size_t size) {
  size_t offset = 0;
  while (offset < size) {
    struct iovec buffer = {const_cast<char *>(data) + offset, size - offset};
    ssize_t sent = lupine_socket_sendv(fd, &buffer, 1);
    if (sent <= 0) {
      return false;
    }
    offset += static_cast<size_t>(sent);
  }
  return true;
}

bool send_all(lupine_socket_t fd, const std::string &data) {
  return send_all(fd, data.data(), data.size());
}

using http_header = std::pair<std::string, std::string>;

std::string http1_headers(int status, const char *reason,
                          uint64_t content_length, const char *content_type,
                          const std::vector<http_header> &headers,
                          const rpc_http2_server_metadata *metadata) {
  std::string response =
      "HTTP/1.1 " + std::to_string(status) + " " + reason + "\r\n";
  if (metadata != nullptr && metadata->backend_version != nullptr &&
      metadata->backend_version[0] != '\0') {
    // Same header the HTTP/2 version probe answers with, so an HTTP/1.1
    // client learns the backend version the same way.
    response += "x-lupine-cuda-version: ";
    response += metadata->backend_version;
    response += "\r\n";
  }
  if (content_type != nullptr) {
    response += "content-type: ";
    response += content_type;
    response += "\r\n";
  }
  for (const auto &header : headers) {
    response += header.first;
    response += ": ";
    response += header.second;
    response += "\r\n";
  }
  response += "content-length: " + std::to_string(content_length) + "\r\n";
  response += "connection: close\r\n\r\n";
  return response;
}

std::string http1_response(int status, const char *reason,
                           const std::string &body, bool include_body,
                           const rpc_http2_server_metadata *metadata,
                           const std::vector<http_header> &headers = {}) {
  std::string response = http1_headers(status, reason, body.size(),
                                       "text/plain", headers, metadata);
  if (include_body) {
    response += body;
  }
  return response;
}

bool send_bundle(lupine_socket_t fd,
                 const lupine_client_bundle_payload &bundle) {
  size_t sent = 0;
  for (size_t i = 0; i < bundle.chunk_count; ++i) {
    const lupine_client_bundle_chunk &chunk = bundle.chunks[i];
    if (chunk.data == nullptr || !send_all(fd, chunk.data, chunk.size)) {
      return false;
    }
    sent += chunk.size;
  }
  return sent == bundle.size;
}

std::string ascii_lower(std::string value) {
  std::transform(value.begin(), value.end(), value.begin(), [](char c) {
    return c >= 'A' && c <= 'Z' ? static_cast<char>(c - 'A' + 'a') : c;
  });
  return value;
}

std::string request_header(const std::string &request, const char *wanted) {
  size_t line_start = request.find("\r\n");
  if (line_start == std::string::npos) {
    return "";
  }
  line_start += 2;
  while (line_start < request.size()) {
    size_t line_end = request.find("\r\n", line_start);
    if (line_end == std::string::npos || line_end == line_start) {
      break;
    }
    size_t colon = request.find(':', line_start);
    if (colon != std::string::npos && colon < line_end &&
        ascii_lower(request.substr(line_start, colon - line_start)) == wanted) {
      size_t value_start = colon + 1;
      while (value_start < line_end &&
             (request[value_start] == ' ' || request[value_start] == '\t')) {
        ++value_start;
      }
      size_t value_end = line_end;
      while (value_end > value_start && (request[value_end - 1] == ' ' ||
                                         request[value_end - 1] == '\t')) {
        --value_end;
      }
      return request.substr(value_start, value_end - value_start);
    }
    line_start = line_end + 2;
  }
  return "";
}

int serve_client_bundle(lupine_socket_t fd, const std::string &method,
                        const std::string &path, const std::string &request,
                        const rpc_http2_server_metadata *metadata) {
  bool head = method == "HEAD";
  if (!head && method != "GET") {
    send_all(fd,
             http1_response(405, "Method Not Allowed", "method not allowed\n",
                            true, metadata, {{"allow", "GET, HEAD"}}));
    return 1;
  }

  std::string platform;
  if (!lupine_client_bundle_request_platform(path, &platform)) {
    return 0;
  }
  const lupine_client_bundle_registry *registry =
      metadata == nullptr ? nullptr : metadata->client_bundles;
  const lupine_client_bundle_payload *bundle =
      lupine_client_bundle_lookup(registry, platform);
  if (bundle == nullptr) {
    send_all(fd,
             http1_response(503, "Service Unavailable",
                            "client bundle unavailable\n", !head, metadata));
    return 1;
  }

  std::vector<http_header> headers = {
      {"etag", bundle->etag},
      {"content-digest", bundle->content_digest},
      {"cache-control", "no-cache"},
  };
  if (request_header(request, "if-none-match") == bundle->etag) {
    return send_all(fd, http1_headers(304, "Not Modified", 0, nullptr, headers,
                                      metadata))
               ? 1
               : -1;
  }

  if (!send_all(fd, http1_headers(200, "OK", bundle->size,
                                  "application/vnd.lupine.client-bundle.v1+zip",
                                  headers, metadata))) {
    return -1;
  }
  return head || send_bundle(fd, *bundle) ? 1 : -1;
}

int serve_http1(lupine_socket_t fd, const rpc_http2_server_metadata *metadata) {
  std::string request;
  while (request.find("\r\n\r\n") == std::string::npos) {
    if (request.size() >= kMaxHttp1RequestBytes) {
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
  size_t path_end = method_end == std::string::npos
                        ? std::string::npos
                        : line.find(' ', method_end + 1);
  if (path_end == std::string::npos) {
    send_all(fd, http1_response(400, "Bad Request", "bad request\n", true,
                                metadata));
    return 1;
  }
  std::string method = line.substr(0, method_end);
  std::string path = line.substr(method_end + 1, path_end - method_end - 1);
  LUPINE_LOG_DEBUG("HTTP/1.1 " << method << " " << path);

  if (path.compare(0, strlen("/.well-known/lupine/client/v1/"),
                   "/.well-known/lupine/client/v1/") == 0) {
    int result = serve_client_bundle(fd, method, path, request, metadata);
    if (result != 0) {
      return result;
    }
  }

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
  for (;;) {
    unsigned char data[kH2PrefaceLength];
    ssize_t received = peek_bytes(connfd, data, sizeof(data));
    if (received < 0 && lupine_socket_error_is_intr()) {
      continue;
    }
    if (received <= 0) {
      return -1;
    }
    int verdict = lupine_h2_preface_check(data, static_cast<size_t>(received));
    if (verdict > 0) {
      return 0;
    }
    if (verdict < 0) {
      return serve_http1(connfd, metadata);
    }
    // A preface prefix shorter than 24 bytes: the bytes stay queued because
    // of MSG_PEEK, so another blocking peek returns immediately. Sleep
    // instead of spinning while the remainder is in flight.
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }
}
