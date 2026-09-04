#ifndef LUPINE_MONITORING_H
#define LUPINE_MONITORING_H

#include <cstdint>
#include <string>

#include "rpc.h"

constexpr int LUPINE_RPC_CLIENT_METADATA = 0x4c504d01;
constexpr uint32_t LUPINE_CLIENT_METADATA_VERSION = 1;
constexpr uint32_t LUPINE_CLIENT_METADATA_MAX_PAYLOAD = 1024;

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
inline void lupine_monitoring_begin_context_create(int) {}
inline void lupine_monitoring_end_context_create(bool) {}

#endif

#endif
