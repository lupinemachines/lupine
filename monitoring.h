#ifndef LUPINE_MONITORING_H
#define LUPINE_MONITORING_H

#include <cstddef>
#include <cstdint>

#include "rpc.h"

// Private RPC used to send client identity.
constexpr int LUPINE_RPC_CLIENT_METADATA = 0x4c504d01;
constexpr uint32_t LUPINE_CLIENT_METADATA_VERSION = 2;
constexpr uint32_t LUPINE_CLIENT_METADATA_MAX_PAYLOAD = 4096;

struct lupine_client_metadata_header {
  uint32_t version;
  uint32_t payload_size;
};

// Identifies one process across its CUDA and NVML connections.
struct lupine_client_metadata_v2 {
  uint64_t client_pid;
  uint64_t client_start_time_ticks;
  uint64_t client_pid_namespace_inode;
  int32_t connection_index;
  int32_t reserved;
  char connection_kind[16];
  char client_process_name[128];
  char client_boot_id[40];
  char vdevice_index[32];
  char client_hostname[256];
  // Groups related clients in workload metrics.
  char workload_id[128];
};

static_assert(sizeof(lupine_client_metadata_v2) <=
                  LUPINE_CLIENT_METADATA_MAX_PAYLOAD,
              "client metadata payload exceeds protocol limit");

#ifndef _WIN32

int lupine_report_client_metadata(conn_t *conn, const char *connection_kind,
                                  int connection_index);

// Tracks server child processes in shared memory.
int lupine_monitoring_start(lupine_socket_t rpc_listener);
void lupine_monitoring_stop();
void lupine_monitoring_note_process_exit(int64_t pid);
int lupine_monitoring_register_child();
void lupine_monitoring_unregister_child(int slot);
int handle_lupine_client_metadata(conn_t *conn);

// Finds the host PID after the first context is created.
void lupine_monitoring_begin_context_create(conn_t *conn, int cuda_device);
void lupine_monitoring_end_context_create(conn_t *conn,
                                          bool context_created);

#else

inline int lupine_monitoring_start(lupine_socket_t) { return 0; }
inline void lupine_monitoring_stop() {}
inline void lupine_monitoring_note_process_exit(int64_t) {}
inline int lupine_monitoring_register_child() { return -1; }
inline void lupine_monitoring_unregister_child(int) {}
inline void lupine_monitoring_begin_context_create(conn_t *, int) {}
inline void lupine_monitoring_end_context_create(conn_t *, bool) {}

#endif

#endif
