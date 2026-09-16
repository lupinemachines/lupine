// NPP core (libnppc) server handlers. nppGetLibVersion and nppGetGpuName return
// memory the library owns, so the response carries its bytes.

#include "npp_server.h"

int handle_nppGetLibVersion(conn_t *conn) {
  int request_id = rpc_read_end(conn);
  if (request_id < 0) {
    return -1;
  }
  using fn_t = const NppLibraryVersion *(*)();
  fn_t fn = npp_symbol<fn_t>("nppGetLibVersion");
  const NppLibraryVersion *library = fn == nullptr ? nullptr : fn();
  NppLibraryVersion version =
      library == nullptr ? NppLibraryVersion{} : *library;
  if (rpc_write_start_response(conn, request_id) < 0 ||
      rpc_write(conn, &version, sizeof(version)) < 0 ||
      rpc_write_end(conn) < 0) {
    return -1;
  }
  return 0;
}

int handle_nppGetGpuName(conn_t *conn) {
  int request_id = rpc_read_end(conn);
  if (request_id < 0) {
    return -1;
  }
  using fn_t = const char *(*)();
  fn_t fn = npp_symbol<fn_t>("nppGetGpuName");
  const char *name = fn == nullptr ? nullptr : fn();
  uint32_t length = name == nullptr ? 0 : static_cast<uint32_t>(strlen(name));
  if (rpc_write_start_response(conn, request_id) < 0 ||
      rpc_write(conn, &length, sizeof(length)) < 0 ||
      rpc_write(conn, name, length) < 0 || rpc_write_end(conn) < 0) {
    return -1;
  }
  return 0;
}

#include "codegen/gen_nppc_server.inc"
