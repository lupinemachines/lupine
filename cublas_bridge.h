#pragma once

#include <cstddef>
#include <cstdint>

struct conn_t;

#if defined(_WIN32)
#if defined(LUPINE_CUBLAS_BRIDGE_EXPORTS)
#define LUPINE_CUBLAS_BRIDGE_API __declspec(dllexport)
#else
#define LUPINE_CUBLAS_BRIDGE_API __declspec(dllimport)
#endif
#else
#define LUPINE_CUBLAS_BRIDGE_API __attribute__((visibility("default")))
#endif

static constexpr std::uint32_t LUPINE_CUBLAS_RPC_API_VERSION = 1;

struct lupine_cublas_rpc_api {
  std::uint32_t version;
  int (*default_route)();
  int (*stream_route)(void *stream);
  conn_t *(*connection)(int route_id);
  int (*prepare)(conn_t *conn);
  int (*write_start_request)(conn_t *conn, int operation);
  int (*write)(conn_t *conn, const void *data, std::size_t size);
  int (*wait_for_response)(conn_t *conn);
  int (*read)(conn_t *conn, void *data, std::size_t size);
  int (*read_end)(conn_t *conn);
};

extern "C" LUPINE_CUBLAS_BRIDGE_API const lupine_cublas_rpc_api *
lupine_cublas_rpc_api_v1();
