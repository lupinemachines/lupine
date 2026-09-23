#include "cuda_client_rpc.h"
#include "rpc.h"

extern int rpc_open();
extern int rpc_size();
extern conn_t *rpc_client_get_connection(unsigned int index);

extern "C" int lupine_rpc_open() { return rpc_open(); }

extern "C" int lupine_rpc_size() { return rpc_size(); }

extern "C" conn_t *lupine_rpc_client_get_connection(unsigned int index) {
  return rpc_client_get_connection(index);
}

extern "C" int lupine_rpc_write_start_request(conn_t *conn, int op) {
  return rpc_write_start_request(conn, op);
}

extern "C" int lupine_rpc_write_start_async_request(conn_t *conn, int op,
                                                    uint64_t *sequence) {
  return rpc_write_start_async_request(conn, op, sequence);
}

extern "C" int lupine_rpc_write(conn_t *conn, const void *data, size_t size) {
  return rpc_write(conn, data, size);
}

extern "C" int lupine_rpc_write_end(conn_t *conn) {
  return rpc_write_end(conn);
}

extern "C" int lupine_rpc_wait_for_response(conn_t *conn) {
  return rpc_wait_for_response(conn);
}

extern "C" int lupine_rpc_read(conn_t *conn, void *data, size_t size) {
  return rpc_read(conn, data, size);
}

extern "C" int lupine_rpc_read_end(conn_t *conn) { return rpc_read_end(conn); }
