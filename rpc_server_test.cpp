#include "rpc_server.h"

#include "lupine_log.h"

#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

namespace {

std::vector<std::string> events;

void require(bool condition, const char *message) {
  if (!condition) {
    LUPINE_LOG_ERROR(message);
    std::exit(1);
  }
}

int handle_first(conn_t *) {
  events.push_back("handle:first");
  return 0;
}

int handle_second(conn_t *) {
  events.push_back("handle:second");
  return 0;
}

rpc_handler lookup_first(int op) {
  return op == 10 ? rpc_handler{handle_first, "first"} : rpc_handler{};
}

rpc_handler lookup_second(int op) {
  return op == 20 ? rpc_handler{handle_second, "second"} : rpc_handler{};
}

const int *first_operations(size_t *count) {
  static const int operations[] = {10};
  *count = 1;
  return operations;
}

const int *second_operations(size_t *count) {
  static const int operations[] = {20};
  *count = 1;
  return operations;
}

const int *duplicate_operations(size_t *count) {
  static const int operations[] = {10};
  *count = 1;
  return operations;
}

bool start_first(lupine_socket_t) {
  events.push_back("start:first");
  return true;
}

bool start_second(lupine_socket_t) {
  events.push_back("start:second");
  return true;
}

int finish_first() {
  events.push_back("finish:first");
  return 0;
}

int finish_second() {
  events.push_back("finish:second");
  return 0;
}

bool open_first(conn_t *) {
  events.push_back("open:first");
  return true;
}

bool open_second(conn_t *) {
  events.push_back("open:second");
  return true;
}

bool ready_first(conn_t *, const char *) {
  events.push_back("ready:first");
  return true;
}

bool ready_second(conn_t *, const char *) {
  events.push_back("ready:second");
  return true;
}

int dispatch_first(const rpc_handler &handler, conn_t *conn) {
  events.push_back("dispatch:first");
  return handler.handler(conn);
}

int dispatch_second(const rpc_handler &handler, conn_t *conn) {
  events.push_back("dispatch:second");
  return handler.handler(conn);
}

void close_first(conn_t *) { events.push_back("close:first"); }
void close_second(conn_t *) { events.push_back("close:second"); }

const rpc_backend first_backend = {
    "first",    lookup_first, first_operations, start_first, finish_first,
    open_first, ready_first,  dispatch_first,   close_first,
};

const rpc_backend second_backend = {
    "second",    lookup_second, second_operations, start_second, finish_second,
    open_second, ready_second,  dispatch_second,   close_second,
};

const rpc_backend duplicate_backend = {
    "duplicate", lookup_second, duplicate_operations,
    nullptr,     nullptr,       nullptr,
    nullptr,     nullptr,       nullptr,
};

void test_lookup_and_duplicates() {
  const rpc_backend *valid[] = {&first_backend, &second_backend};
  const rpc_backend *duplicate[] = {&first_backend, &duplicate_backend};
  require(rpc_server_validate_backends(valid, 2),
          "valid backend registry rejected");
  require(!rpc_server_validate_backends(duplicate, 2),
          "duplicate operation accepted");
  conn_t conn = {};
  events.clear();
  require(rpc_server_dispatch(valid, 2, &conn, 20) == 0,
          "second backend lookup failed");
  require(events ==
              std::vector<std::string>({"dispatch:second", "handle:second"}),
          "dispatch ordering changed");
  require(rpc_server_dispatch(valid, 2, &conn, 30) < 0,
          "unknown operation accepted");
}

void test_lifecycle_ordering() {
  const rpc_backend *backends[] = {&first_backend, &second_backend};
  conn_t conn = {};
  events.clear();
  require(rpc_server_child_start(backends, 2, 0), "child start failed");
  require(rpc_server_connection_open(backends, 2, &conn),
          "connection open failed");
  require(rpc_server_connection_ready(backends, 2, &conn, "session"),
          "connection ready failed");
  rpc_server_connection_close(backends, 2, &conn);
  require(rpc_server_child_finish(backends, 2) == 0, "child finish failed");
  require(events ==
              std::vector<std::string>(
                  {"start:first", "start:second", "open:first", "open:second",
                   "ready:first", "ready:second", "close:second", "close:first",
                   "finish:second", "finish:first"}),
          "backend lifecycle ordering changed");
}

} // namespace

int main() {
  test_lookup_and_duplicates();
  test_lifecycle_ordering();
  std::cout << "rpc_server_test: PASS" << std::endl;
  return 0;
}
