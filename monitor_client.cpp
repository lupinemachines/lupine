#include "monitoring.h"

#include <cstring>

#ifdef _WIN32
#include <process.h>
#include <windows.h>
#else
#include <unistd.h>
#ifdef __linux__
#include <sys/stat.h>
#endif
#endif

#include "lupine_log.h"

namespace {

template <size_t N> void copy_value(char (&destination)[N], const char *value) {
  if (value == nullptr) {
    destination[0] = '\0';
    return;
  }
  size_t length = strnlen(value, N - 1);
  memcpy(destination, value, length);
  destination[length] = '\0';
}

uint64_t process_id() {
#ifdef _WIN32
  return static_cast<uint64_t>(_getpid());
#else
  return static_cast<uint64_t>(getpid());
#endif
}

void fill_hostname(char *destination, size_t size) {
  if (size == 0) {
    return;
  }
  destination[0] = '\0';
  if (gethostname(destination, static_cast<int>(size)) != 0) {
    destination[0] = '\0';
  }
  destination[size - 1] = '\0';
}

#if defined(__linux__)

uint64_t pid_namespace_inode() {
  struct stat info = {};
  return stat("/proc/self/ns/pid", &info) == 0
             ? static_cast<uint64_t>(info.st_ino)
             : 0;
}

template <size_t N> void fill_process_name(char (&destination)[N]) {
  char executable[4096] = {};
  ssize_t length =
      readlink("/proc/self/exe", executable, sizeof(executable) - 1);
  if (length <= 0) {
    destination[0] = '\0';
    return;
  }
  executable[length] = '\0';
  const char *base = strrchr(executable, '/');
  copy_value(destination, base == nullptr ? executable : base + 1);
}

#elif defined(_WIN32)

template <size_t N> void fill_process_name(char (&destination)[N]) {
  char executable[MAX_PATH] = {};
  DWORD length = GetModuleFileNameA(nullptr, executable, MAX_PATH);
  if (length == 0 || length >= MAX_PATH) {
    destination[0] = '\0';
    return;
  }
  const char *slash = strrchr(executable, '\\');
  copy_value(destination, slash == nullptr ? executable : slash + 1);
}

#endif

#if !defined(__linux__) && !defined(_WIN32)

template <size_t N> void fill_process_name(char (&destination)[N]) {
  destination[0] = '\0';
}

#endif

} // namespace

int lupine_report_client_metadata(conn_t *conn, const char *connection_kind) {
  if (!rpc_http2_peer_supports(conn,
                               LUPINE_SERVER_CAPABILITY_CLIENT_METADATA)) {
    return 0;
  }

  lupine_client_metadata_header header = {
      LUPINE_CLIENT_METADATA_VERSION,
      static_cast<uint32_t>(sizeof(lupine_client_metadata)),
  };
  lupine_client_metadata metadata = {};
  metadata.client_pid = process_id();
#ifdef __linux__
  metadata.client_pid_namespace_inode = pid_namespace_inode();
#endif
  copy_value(metadata.connection_kind, connection_kind);
  fill_process_name(metadata.client_process_name);
  fill_hostname(metadata.client_hostname, sizeof(metadata.client_hostname));

  int status = -1;
  if (rpc_write_start_request(conn, LUPINE_RPC_CLIENT_METADATA) < 0 ||
      rpc_write(conn, &header, sizeof(header)) < 0 ||
      rpc_write(conn, &metadata, sizeof(metadata)) < 0 ||
      rpc_wait_for_response(conn) < 0 ||
      rpc_read(conn, &status, sizeof(status)) < 0 || rpc_read_end(conn) < 0) {
    LUPINE_LOG_ERROR("Failed to send client metadata");
    return -1;
  }
  if (status != 0) {
    LUPINE_LOG_ERROR("LUPINE server rejected client metadata with status "
                     << status);
  }
  return 0;
}
