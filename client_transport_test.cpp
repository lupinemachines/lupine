#include "client_transport.h"

#include "lupine_log.h"

#include <cstdlib>
#include <iostream>
#include <sys/socket.h>
#include <unistd.h>
#include <vector>

namespace {

void require(bool condition, const char *message) {
  if (!condition) {
    LUPINE_LOG_ERROR(message);
    std::exit(1);
  }
}

void test_endpoint_parsing() {
  lupine_client_transport_config config;
  std::vector<lupine_client_endpoint> endpoints;
  require(lupine_client_parse_endpoints(
              "server-a,https://server-b:9443,http://server-c", &config,
              &endpoints) == 0,
          "endpoint parsing failed");
  require(endpoints.size() == 3, "endpoint count changed");
  require(endpoints[0].host == "server-a" && endpoints[0].port == "14833" &&
              endpoints[0].label == "server-a" && !endpoints[0].tls,
          "default endpoint changed");
  require(endpoints[1].host == "server-b" && endpoints[1].port == "9443" &&
              endpoints[1].label == "server-b:9443" && endpoints[1].tls,
          "TLS endpoint changed");
  require(endpoints[2].host == "server-c" && endpoints[2].port == "14833" &&
              endpoints[2].label == "server-c" && !endpoints[2].tls,
          "HTTP endpoint changed");
}

void test_strict_endpoint_parsing() {
  lupine_client_transport_config config;
  config.strict_endpoints = true;
  std::vector<lupine_client_endpoint> endpoints;
  require(lupine_client_parse_endpoints(
              ",ftp://invalid,missing-port:,https://server", &config,
              &endpoints) == 0,
          "strict endpoint parsing failed");
  require(endpoints.size() == 1 && endpoints[0].host == "server" &&
              endpoints[0].port == "443" &&
              endpoints[0].label == "server:443" && endpoints[0].tls,
          "strict endpoint filtering changed");
}

struct transport_observer {
  lupine_client_dial_policy policy = lupine_client_dial_policy::bounded_retry;
  int peer_fd = -1;
  int opened = 0;
  int closed = 0;
};

lupine_socket_t test_dial(const lupine_client_endpoint *,
                          lupine_client_dial_policy policy, void *user_data) {
  auto *observer = static_cast<transport_observer *>(user_data);
  int sockets[2] = {-1, -1};
  require(socketpair(AF_UNIX, SOCK_STREAM, 0, sockets) == 0,
          "test socketpair failed");
  observer->policy = policy;
  observer->peer_fd = sockets[1];
  return sockets[0];
}

void test_dispatch(conn_t *, void *) {}

void test_connection_opened(conn_t *, const lupine_client_endpoint *,
                            void *user_data) {
  ++static_cast<transport_observer *>(user_data)->opened;
}

void test_connection_closed(conn_t *, void *user_data) {
  ++static_cast<transport_observer *>(user_data)->closed;
}

lupine_client_transport_config test_config(lupine_client_dial_policy policy,
                                           transport_observer *observer) {
  lupine_client_transport_config config;
  config.dial_policy = policy;
  config.strict_endpoints = true;
  config.user_data = observer;
  config.dial = test_dial;
  config.dispatch = test_dispatch;
  config.connection_opened = test_connection_opened;
  config.connection_closed = test_connection_closed;
  return config;
}

void test_transport_policy_lifecycle_and_isolation() {
  transport_observer first_observer;
  transport_observer second_observer;
  auto first_config =
      test_config(lupine_client_dial_policy::bounded_retry, &first_observer);
  auto second_config =
      test_config(lupine_client_dial_policy::single_attempt, &second_observer);
  auto *first = lupine_client_transport_create(&first_config);
  auto *second = lupine_client_transport_create(&second_config);
  require(first != nullptr && second != nullptr && first != second,
          "transport instances alias");
  require(lupine_client_transport_open(first, "first") == 0 &&
              lupine_client_transport_open(second, "second") == 0,
          "injected transports did not open");
  require(first_observer.policy == lupine_client_dial_policy::bounded_retry &&
              second_observer.policy ==
                  lupine_client_dial_policy::single_attempt,
          "dial policy changed");
  require(first_observer.opened == 1 && second_observer.opened == 1,
          "connection-open hook count changed");
  require(lupine_client_transport_size(first) == 1 &&
              lupine_client_transport_size(second) == 1,
          "transport instances did not retain independent pools");
  lupine_client_transport_close(first);
  require(lupine_client_transport_size(first) == 0 &&
              lupine_client_transport_size(second) == 1 &&
              first_observer.closed > 0,
          "closing one transport changed the other");
  lupine_client_transport_destroy(first);
  lupine_client_transport_destroy(second);
  require(second_observer.closed > 0, "connection-close hook did not run");
  close(first_observer.peer_fd);
  close(second_observer.peer_fd);
}

} // namespace

int main() {
  test_endpoint_parsing();
  test_strict_endpoint_parsing();
  test_transport_policy_lifecycle_and_isolation();
  std::cout << "client_transport_test: PASS" << std::endl;
  return 0;
}
