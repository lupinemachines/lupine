#ifndef LUPINE_MONITORING_H
#define LUPINE_MONITORING_H

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <string>

#include "rpc.h"

constexpr int LUPINE_RPC_CLIENT_METADATA = 0x4c504d01;
constexpr uint32_t LUPINE_CLIENT_METADATA_VERSION = 1;
constexpr uint32_t LUPINE_CLIENT_METADATA_MAX_PAYLOAD = 1024;
constexpr size_t LUPINE_WORKLOAD_ID_MAX_BYTES = 256;

struct lupine_client_metadata_header {
  uint32_t version;
  uint32_t payload_size;
};

struct lupine_client_metadata {
  uint64_t client_pid;
  uint64_t client_pid_namespace_inode;
  char connection_kind[16];
  char client_process_name[128];
  char client_hostname[128];
};

static_assert(sizeof(lupine_client_metadata) <=
                  LUPINE_CLIENT_METADATA_MAX_PAYLOAD,
              "client metadata payload exceeds protocol limit");

// Optional trailing extension; the original metadata prefix stays unchanged.
struct lupine_client_metadata_extended {
  lupine_client_metadata client;
  uint32_t workload_id_size;
  char workload_id[LUPINE_WORKLOAD_ID_MAX_BYTES];
};

// Reject malformed UTF-8 and embedded NULs; never truncate identity strings.
inline bool lupine_valid_workload_id(const char *data, size_t size) {
  if (size > LUPINE_WORKLOAD_ID_MAX_BYTES)
    return false;
  for (size_t i = 0; i < size;) {
    unsigned char c = static_cast<unsigned char>(data[i++]);
    if (c == 0)
      return false;
    if (c < 0x80)
      continue;
    unsigned remaining;
    uint32_t cp, minimum;
    if (c >= 0xc2 && c <= 0xdf) {
      remaining = 1;
      cp = c & 0x1f;
      minimum = 0x80;
    } else if (c >= 0xe0 && c <= 0xef) {
      remaining = 2;
      cp = c & 0xf;
      minimum = 0x800;
    } else if (c >= 0xf0 && c <= 0xf4) {
      remaining = 3;
      cp = c & 7;
      minimum = 0x10000;
    } else
      return false;
    if (remaining > size - i)
      return false;
    while (remaining--) {
      c = static_cast<unsigned char>(data[i++]);
      if ((c & 0xc0) != 0x80)
        return false;
      cp = (cp << 6) | (c & 0x3f);
    }
    if (cp < minimum || cp > 0x10ffff || (cp >= 0xd800 && cp <= 0xdfff))
      return false;
  }
  return true;
}

// Decode old and new peers without relying on payload alignment. Invalid
// identity extensions are ignored while preserving the process metadata.
inline bool lupine_decode_client_metadata(
    const lupine_client_metadata_header &header, const void *payload,
    lupine_client_metadata *client,
    char (&workload_id)[LUPINE_WORKLOAD_ID_MAX_BYTES + 1]) {
  *client = {};
  memset(workload_id, 0, sizeof(workload_id));
  if (header.payload_size < sizeof(*client) ||
      header.payload_size > LUPINE_CLIENT_METADATA_MAX_PAYLOAD ||
      header.version != LUPINE_CLIENT_METADATA_VERSION)
    return false;
  memcpy(client, payload, sizeof(*client));
  const size_t offset = offsetof(lupine_client_metadata_extended, workload_id);
  if (header.payload_size >= offset) {
    uint32_t size = 0;
    const char *bytes = static_cast<const char *>(payload);
    memcpy(&size, bytes + offsetof(lupine_client_metadata_extended, workload_id_size),
           sizeof(size));
    if (size <= LUPINE_WORKLOAD_ID_MAX_BYTES &&
        header.payload_size == offset + size &&
        lupine_valid_workload_id(bytes + offset, size))
      memcpy(workload_id, bytes + offset, size);
  }
  return true;
}

static_assert(sizeof(lupine_client_metadata_extended) <=
                  LUPINE_CLIENT_METADATA_MAX_PAYLOAD,
              "extended client metadata exceeds protocol limit");

int lupine_report_client_metadata(conn_t *conn, const char *connection_kind);

#ifdef LUPINE_MONITORING_ENABLED

bool lupine_monitoring_initialize();
void lupine_monitoring_shutdown();
void lupine_monitoring_register_child();
void lupine_monitoring_unregister_pid(int64_t server_pid);
int handle_lupine_client_metadata(conn_t *conn);
void lupine_monitoring_begin_context_create(int cuda_device);
void lupine_monitoring_end_context_create(bool context_created);
std::string lupine_monitoring_render_metrics();

#else

inline bool lupine_monitoring_initialize() { return true; }
inline void lupine_monitoring_shutdown() {}
inline void lupine_monitoring_register_child() {}
inline void lupine_monitoring_unregister_pid(int64_t) {}
inline int handle_lupine_client_metadata(conn_t *conn) {
  lupine_client_metadata_header header = {};
  if (rpc_read(conn, &header, sizeof(header)) != 0 ||
      rpc_drain(conn, header.payload_size) < 0) {
    return -1;
  }
  return rpc_read_end(conn) < 0 ? -1 : 0;
}
inline void lupine_monitoring_begin_context_create(int) {}
inline void lupine_monitoring_end_context_create(bool) {}

#endif

#endif
