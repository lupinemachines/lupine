// Covers the single-port protocol dispatch: an HTTP/2 preface must reach the
// RPC path with nothing consumed, and anything else must be answered as one
// HTTP/1.x request. Uses socketpair(), so this test is Unix-only like h2_test.

#include "client_bundle.h"
#include "dispatch.h"

#include <cstring>
#include <stdio.h>
#include <string>
#include <sys/socket.h>
#include <thread>
#include <unistd.h>

namespace {

int failures = 0;

void check(bool ok, const char *what) {
  printf("%-5s %s\n", ok ? "ok" : "FAIL", what);
  if (!ok) {
    ++failures;
  }
}

constexpr char kPreface[] = "PRI * HTTP/2.0\r\n\r\nSM\r\n\r\n";
constexpr size_t kPrefaceLength = sizeof(kPreface) - 1;
const rpc_http2_server_metadata kMetadata = {"13.0-test", nullptr};

bool make_pair(int fds[2]) {
  return socketpair(AF_UNIX, SOCK_STREAM, 0, fds) == 0;
}

std::string read_response(int fd) {
  std::string response;
  char chunk[1024];
  for (;;) {
    ssize_t received = read(fd, chunk, sizeof(chunk));
    if (received <= 0) {
      break;
    }
    response.append(chunk, static_cast<size_t>(received));
  }
  return response;
}

void test_preface_check() {
  const unsigned char *preface =
      reinterpret_cast<const unsigned char *>(kPreface);
  check(lupine_h2_preface_check(preface, kPrefaceLength) == 1,
        "complete preface matches");
  check(lupine_h2_preface_check(preface, 10) == 0,
        "preface prefix is undecided");
  check(lupine_h2_preface_check(
            reinterpret_cast<const unsigned char *>("GET / HTTP/1.1"), 14) ==
            -1,
        "http/1.1 request line is rejected");
  check(lupine_h2_preface_check(
            reinterpret_cast<const unsigned char *>("POST / HTTP/1.1"), 15) ==
            -1,
        "POST diverges from PRI before the preface completes");
}

void test_preface_reaches_rpc_with_nothing_consumed() {
  int fds[2];
  if (!make_pair(fds)) {
    check(false, "socketpair");
    return;
  }
  std::string sent = std::string(kPreface) + "SETTINGS-bytes";
  check(write(fds[1], sent.data(), sent.size()) ==
            static_cast<ssize_t>(sent.size()),
        "client writes preface");
  check(lupine_connection_dispatch(fds[0], &kMetadata) == 0,
        "preface dispatches to rpc");
  char data[64] = {};
  ssize_t received = read(fds[0], data, sent.size());
  check(received == static_cast<ssize_t>(sent.size()) &&
            memcmp(data, sent.data(), sent.size()) == 0,
        "dispatch consumed nothing");
  close(fds[0]);
  close(fds[1]);
}

void test_split_preface_still_dispatches_to_rpc() {
  int fds[2];
  if (!make_pair(fds)) {
    check(false, "socketpair");
    return;
  }
  std::thread client([&]() {
    write(fds[1], kPreface, 9);
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    write(fds[1], kPreface + 9, kPrefaceLength - 9);
  });
  check(lupine_connection_dispatch(fds[0], &kMetadata) == 0,
        "split preface dispatches to rpc");
  client.join();
  close(fds[0]);
  close(fds[1]);
}

void test_http1_get_root() {
  int fds[2];
  if (!make_pair(fds)) {
    check(false, "socketpair");
    return;
  }
  const char request[] = "GET / HTTP/1.1\r\nhost: lupine\r\n\r\n";
  write(fds[1], request, sizeof(request) - 1);
  check(lupine_connection_dispatch(fds[0], &kMetadata) == 1,
        "http/1.1 request is served here");
  close(fds[0]);
  std::string response = read_response(fds[1]);
  check(response.rfind("HTTP/1.1 200 OK\r\n", 0) == 0, "GET / returns 200");
  check(response.find("x-lupine-cuda-version: 13.0-test\r\n") !=
            std::string::npos,
        "GET / reports the backend version");
  check(response.find("\r\n\r\nlupine\n") != std::string::npos,
        "GET / has a body");
  close(fds[1]);
}

void test_http1_head_root_has_no_body() {
  int fds[2];
  if (!make_pair(fds)) {
    check(false, "socketpair");
    return;
  }
  const char request[] = "HEAD / HTTP/1.1\r\n\r\n";
  write(fds[1], request, sizeof(request) - 1);
  check(lupine_connection_dispatch(fds[0], &kMetadata) == 1,
        "HEAD / is served here");
  close(fds[0]);
  std::string response = read_response(fds[1]);
  check(response.rfind("HTTP/1.1 200 OK\r\n", 0) == 0, "HEAD / returns 200");
  check(response.find("\r\n\r\n") == response.size() - 4, "HEAD / has no body");
  close(fds[1]);
}

void test_http1_unknown_path_is_404() {
  int fds[2];
  if (!make_pair(fds)) {
    check(false, "socketpair");
    return;
  }
  const char request[] = "GET /metrics HTTP/1.1\r\n\r\n";
  write(fds[1], request, sizeof(request) - 1);
  check(lupine_connection_dispatch(fds[0], &kMetadata) == 1,
        "unknown path is served here");
  close(fds[0]);
  std::string response = read_response(fds[1]);
  check(response.rfind("HTTP/1.1 404 Not Found\r\n", 0) == 0,
        "unknown path returns 404");
  close(fds[1]);
}

constexpr char kBundleEtag[] =
    "\"sha256:"
    "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb\"";
const char kBundleFirst[] = "embedded-";
const char kBundleSecond[] = "zip";
const lupine_client_bundle_chunk kBundleChunks[] = {
    {kBundleFirst, sizeof(kBundleFirst) - 1},
    {kBundleSecond, sizeof(kBundleSecond) - 1},
};
const lupine_client_bundle_payload kBundle = {
    kBundleEtag, "sha-256=:ZW1iZWRkZWQtemlw:", kBundleChunks, 2, 12};
const lupine_client_bundle_entry kBundleEntries[] = {
    {"linux/amd64", &kBundle},
};
const lupine_client_bundle_registry kBundles = {kBundleEntries, 1};
const rpc_http2_server_metadata kBundleMetadata = {"13.0-test", &kBundles};

std::string bundle_request(const rpc_http2_server_metadata *metadata,
                           const std::string &request) {
  int fds[2];
  if (!make_pair(fds)) {
    check(false, "socketpair");
    return "";
  }
  write(fds[1], request.data(), request.size());
  check(lupine_connection_dispatch(fds[0], metadata) == 1,
        "bundle request is served here");
  close(fds[0]);
  std::string response = read_response(fds[1]);
  close(fds[1]);
  return response;
}

void test_http1_client_bundle() {
  const std::string path = "/.well-known/lupine/client/v1/linux/amd64";

  std::string response = bundle_request(
      &kBundleMetadata, "GET " + path + " HTTP/1.1\r\nhost: lupine\r\n\r\n");
  check(response.rfind("HTTP/1.1 200 OK\r\n", 0) == 0,
        "client bundle GET returns 200");
  check(response.find(
            "content-type: application/vnd.lupine.client-bundle.v1+zip\r\n") !=
            std::string::npos,
        "client bundle reports its media type");
  check(response.find(std::string("etag: ") + kBundleEtag + "\r\n") !=
            std::string::npos,
        "client bundle reports its ETag");
  check(response.find("\r\n\r\nembedded-zip") != std::string::npos,
        "client bundle GET streams every compiled chunk");

  response =
      bundle_request(&kBundleMetadata, "HEAD " + path + " HTTP/1.1\r\n\r\n");
  check(response.find("content-length: 12\r\n") != std::string::npos,
        "client bundle HEAD reports object size");
  check(response.find("\r\n\r\n") == response.size() - 4,
        "client bundle HEAD has no body");

  response = bundle_request(&kBundleMetadata,
                            "GET " + path + " HTTP/1.1\r\nIf-None-Match: " +
                                kBundleEtag + "\r\n\r\n");
  check(response.rfind("HTTP/1.1 304 Not Modified\r\n", 0) == 0,
        "matching client bundle ETag returns 304");
}

void test_closed_peer_fails_dispatch() {
  int fds[2];
  if (!make_pair(fds)) {
    check(false, "socketpair");
    return;
  }
  close(fds[1]);
  check(lupine_connection_dispatch(fds[0], &kMetadata) == -1,
        "closed peer fails dispatch");
  close(fds[0]);
}

} // namespace

int main() {
  test_preface_check();
  test_preface_reaches_rpc_with_nothing_consumed();
  test_split_preface_still_dispatches_to_rpc();
  test_http1_get_root();
  test_http1_head_root_has_no_body();
  test_http1_unknown_path_is_404();
  test_http1_client_bundle();
  test_closed_peer_fails_dispatch();
  printf("\ndispatch_test: %s\n", failures == 0 ? "PASS" : "FAIL");
  return failures == 0 ? 0 : 1;
}
