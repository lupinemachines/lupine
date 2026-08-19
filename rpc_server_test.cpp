#include "rpc_server.h"

#include "lupine_log.h"

#include <cstdlib>
#include <iostream>

namespace {

int first_calls = 0;
int second_calls = 0;

void require(bool condition, const char *message) {
  if (!condition) {
    LUPINE_LOG_ERROR(message);
    std::exit(1);
  }
}

int handle_first(conn_t *) {
  ++first_calls;
  return 0;
}

int handle_second(conn_t *) {
  ++second_calls;
  return 0;
}

const rpc_handler_registry handlers = {
    {10, {handle_first, rpc_backend::cuda}},
    {20, {handle_second, rpc_backend::nvml}},
};

void test_registry_and_dispatch() {
  const rpc_handler_registry null_handler = {
      {30, {nullptr, rpc_backend::cuda}},
  };
  const rpc_handler_registry invalid_backend = {
      {30, {handle_first, static_cast<rpc_backend>(2)}},
  };
  require(rpc_server_validate(handlers), "valid handler registry rejected");
  require(!rpc_server_validate(null_handler), "null handler accepted");
  require(!rpc_server_validate(invalid_backend), "unknown backend accepted");

  conn_t conn = {};
  first_calls = 0;
  second_calls = 0;
  require(rpc_server_dispatch(handlers, &conn, 10) == 0,
          "CUDA backend lookup failed");
  require(rpc_server_dispatch(handlers, &conn, 20) == 0,
          "NVML backend lookup failed");
  require(first_calls == 1 && second_calls == 1,
          "registered handler was not called");
  require(rpc_server_dispatch(handlers, &conn, 30) < 0,
          "unknown operation accepted");
}

} // namespace

int main() {
  test_registry_and_dispatch();
  std::cout << "rpc_server_test: PASS" << std::endl;
  return 0;
}
