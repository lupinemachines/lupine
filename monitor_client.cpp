#include "monitoring.h"

#include <algorithm>
#include <cerrno>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>

#include <sys/stat.h>
#include <strings.h>
#include <unistd.h>

#include "lupine_log.h"

namespace {

bool env_truthy(const char *value) {
  return value != nullptr && value[0] != '\0' && strcmp(value, "0") != 0 &&
         strcasecmp(value, "false") != 0 && strcasecmp(value, "no") != 0 &&
         strcasecmp(value, "off") != 0;
}

bool metadata_enabled() {
  const char *explicit_value = getenv("LUPINE_CLIENT_METADATA");
  if (explicit_value != nullptr) {
    return env_truthy(explicit_value);
  }
  return true;
}

template <size_t N> void copy_env(char (&dst)[N], const char *name) {
  const char *value = getenv(name);
  if (value == nullptr) {
    dst[0] = '\0';
    return;
  }
  size_t length = strnlen(value, N - 1);
  memcpy(dst, value, length);
  dst[length] = '\0';
}

template <size_t N> void copy_value(char (&dst)[N], const char *value) {
  if (value == nullptr) {
    dst[0] = '\0';
    return;
  }
  size_t length = strnlen(value, N - 1);
  memcpy(dst, value, length);
  dst[length] = '\0';
}

uint64_t process_id() {
  return static_cast<uint64_t>(getpid());
}

void fill_hostname(char *dst, size_t size) {
  if (size == 0) {
    return;
  }
  dst[0] = '\0';
  if (gethostname(dst, static_cast<int>(size)) != 0) {
    dst[0] = '\0';
  }
  dst[size - 1] = '\0';
}

uint64_t process_start_time_ticks() {
  std::ifstream input("/proc/self/stat");
  std::string line;
  if (!std::getline(input, line)) {
    return 0;
  }
  size_t name_end = line.rfind(')');
  if (name_end == std::string::npos || name_end + 2 >= line.size()) {
    return 0;
  }
  std::istringstream fields(line.substr(name_end + 2));
  std::string value;
  for (int field = 3; field <= 22; ++field) {
    if (!(fields >> value)) {
      return 0;
    }
    if (field == 22) {
      char *end = nullptr;
      errno = 0;
      unsigned long long parsed = strtoull(value.c_str(), &end, 10);
      return errno == 0 && end != value.c_str() && *end == '\0'
                 ? static_cast<uint64_t>(parsed)
                 : 0;
    }
  }
  return 0;
}

uint64_t pid_namespace_inode() {
  struct stat info = {};
  return stat("/proc/self/ns/pid", &info) == 0
             ? static_cast<uint64_t>(info.st_ino)
             : 0;
}

template <size_t N> void copy_first_line(char (&dst)[N], const char *path) {
  std::ifstream input(path);
  std::string value;
  if (!std::getline(input, value)) {
    dst[0] = '\0';
    return;
  }
  copy_value(dst, value.c_str());
}

template <size_t N> void fill_process_name(char (&dst)[N]) {
  const char *override_name = getenv("LUPINE_CLIENT_NAME");
  if (override_name != nullptr && override_name[0] != '\0') {
    copy_value(dst, override_name);
    return;
  }
  char executable[4096] = {};
  ssize_t length = readlink("/proc/self/exe", executable,
                            sizeof(executable) - 1);
  if (length > 0) {
    executable[length] = '\0';
    const char *base = strrchr(executable, '/');
    copy_value(dst, base != nullptr ? base + 1 : executable);
    return;
  }
  copy_first_line(dst, "/proc/self/comm");
}

} // namespace

int lupine_report_client_metadata(conn_t *conn, const char *connection_kind,
                                  int connection_index) {
  if (!metadata_enabled()) {
    return 0;
  }
  if (conn == nullptr) {
    return -1;
  }

  lupine_client_metadata_header header = {
      LUPINE_CLIENT_METADATA_VERSION,
      static_cast<uint32_t>(sizeof(lupine_client_metadata_v2)),
  };
  lupine_client_metadata_v2 metadata = {};
  metadata.client_pid = process_id();
  metadata.client_start_time_ticks = process_start_time_ticks();
  metadata.client_pid_namespace_inode = pid_namespace_inode();
  metadata.connection_index = connection_index;
  copy_value(metadata.connection_kind, connection_kind);
  fill_process_name(metadata.client_process_name);
  copy_first_line(metadata.client_boot_id,
                  "/proc/sys/kernel/random/boot_id");
  copy_env(metadata.vdevice_index, "LUPINE_VDEVICE_INDEX");
  fill_hostname(metadata.client_hostname, sizeof(metadata.client_hostname));
  copy_env(metadata.workload_id, "LUPINE_WORKLOAD_ID");

  int status = -1;
  if (rpc_write_start_request(conn, LUPINE_RPC_CLIENT_METADATA) < 0 ||
      rpc_write(conn, &header, sizeof(header)) < 0 ||
      rpc_write(conn, &metadata, sizeof(metadata)) < 0 ||
      rpc_wait_for_response(conn) < 0 ||
      rpc_read(conn, &status, sizeof(status)) < 0 || rpc_read_end(conn) < 0) {
    LUPINE_LOG_ERROR("Unable to report client metadata to the Lupine server");
    return -1;
  }
  if (status != 0) {
    LUPINE_LOG_ERROR("Lupine server rejected client metadata with status "
                     << status);
    return -1;
  }
  return 0;
}
