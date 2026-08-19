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
    "first",     start_first,    finish_first, open_first,
    ready_first, dispatch_first, close_first,
};

const rpc_backend second_backend = {
    "second",     start_second,    finish_second, open_second,
    ready_second, dispatch_second, close_second,
};

const rpc_backend unknown_backend = {
    "unknown", nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
};

const rpc_handler_registry handlers = {
    {10, {handle_first, &first_backend}},
    {20, {handle_second, &second_backend}},
};

void test_registry_and_dispatch() {
  const rpc_backend *backends[] = {&first_backend, &second_backend};
  const rpc_handler_registry invalid_handlers = {
      {30, {handle_first, &unknown_backend}},
  };
  require(rpc_server_validate(handlers, backends, 2),
          "valid handler registry rejected");
  require(!rpc_server_validate(invalid_handlers, backends, 2),
          "unknown backend accepted");
  conn_t conn = {};
  events.clear();
  require(rpc_server_dispatch(handlers, &conn, 20) == 0,
          "second backend lookup failed");
  require(events ==
              std::vector<std::string>({"dispatch:second", "handle:second"}),
          "dispatch ordering changed");
  require(rpc_server_dispatch(handlers, &conn, 30) < 0,
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
  test_registry_and_dispatch();
  test_lifecycle_ordering();
  std::cout << "rpc_server_test: PASS" << std::endl;
  return 0;
}
