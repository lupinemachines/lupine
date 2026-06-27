#include "snapshot.h"

#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <string>
#include <vector>

#ifndef _WIN32
#include <fcntl.h>
#include <ftw.h>
#include <signal.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>
#endif

#include "lupine_log.h"

// Snapshots are built on CRIU, which is Linux-only. The full implementation is
// POSIX; on Windows we provide stubs at the bottom so the server still builds.
#ifndef _WIN32

extern void client_handler(lupine_socket_t connfd);

namespace {

struct snapshot_result {
  CUresult cuda_result = CUDA_ERROR_UNKNOWN;
};

static const char kBootstrapMagic[] = "LUPSNAP1";

const char *snapshot_root() {
  const char *root = getenv("LUPINE_SNAPSHOT_DIR");
  return root != nullptr && root[0] != '\0' ? root : nullptr;
}

std::string join_path(const std::string &a, const std::string &b) {
  if (a.empty()) {
    return b;
  }
  if (a.back() == '/') {
    return a + b;
  }
  return a + "/" + b;
}

bool mkdir_p(const std::string &path) {
  if (path.empty()) {
    return false;
  }
  std::string cur;
  size_t start = path[0] == '/' ? 1 : 0;
  for (size_t i = start; i <= path.size(); ++i) {
    if (i != path.size() && path[i] != '/') {
      continue;
    }
    cur = i == 0 ? "." : path.substr(0, i);
    if (cur.empty() || cur == "/") {
      continue;
    }
    if (mkdir(cur.c_str(), 0700) != 0 && errno != EEXIST) {
      return false;
    }
  }
  return true;
}

// Map an arbitrary snapshot id to a stable, path-safe directory name. The id
// may be any non-empty string; hashing it (FNV-1a 64-bit) means the on-disk
// name is always a safe single component and is identical across processes
// (save and restore must resolve to the same directory).
std::string snapshot_id_dir(const char *id) {
  uint64_t h = 1469598103934665603ULL;  // FNV-1a offset basis
  for (const unsigned char *p = reinterpret_cast<const unsigned char *>(id); *p;
       ++p) {
    h ^= *p;
    h *= 1099511628211ULL;  // FNV-1a prime
  }
  char buf[17];
  snprintf(buf, sizeof(buf), "%016llx", static_cast<unsigned long long>(h));
  return std::string(buf);
}

std::string fd_target(int fd) {
#ifdef _WIN32
  (void)fd;
  return "";
#else
  char path[64];
  snprintf(path, sizeof(path), "/proc/self/fd/%d", fd);
  char buf[512];
  ssize_t n = readlink(path, buf, sizeof(buf) - 1);
  if (n < 0) {
    return "";
  }
  buf[n] = '\0';
  return std::string(buf);
#endif
}

std::string criu_inherit_target(const std::string &target) {
  if (!target.empty() && target[0] == '/') {
    return target.substr(1);
  }
  return target;
}

void redirect_stdio_to_devnull() {
#ifndef _WIN32
  const char *log_path = getenv("LUPINE_SNAPSHOT_STDIO_LOG");
  int fd = -1;
  if (log_path != nullptr && log_path[0] != '\0') {
    fd = open(log_path, O_WRONLY | O_CREAT | O_APPEND | O_CLOEXEC, 0600);
  }
  if (fd < 0) {
    fd = open("/dev/null", O_RDWR | O_CLOEXEC);
  }
  if (fd < 0) {
    return;
  }
  dup2(fd, STDOUT_FILENO);
  dup2(fd, STDERR_FILENO);
  close(fd);
#endif
}

static thread_local uint64_t g_tree_size = 0;

int tree_size_cb(const char *, const struct stat *st, int type, struct FTW *) {
  if (type == FTW_F) {
    g_tree_size += static_cast<uint64_t>(st->st_size);
  }
  return 0;
}

uint64_t tree_size(const std::string &path) {
  g_tree_size = 0;
  nftw(path.c_str(), tree_size_cb, 32, FTW_PHYS);
  return g_tree_size;
}

int remove_tree_cb(const char *path, const struct stat *, int, struct FTW *) {
  return remove(path);
}

bool remove_tree(const std::string &path) {
  struct stat st = {};
  if (lstat(path.c_str(), &st) != 0) {
    return errno == ENOENT;
  }
  return nftw(path.c_str(), remove_tree_cb, 32, FTW_DEPTH | FTW_PHYS) == 0;
}

bool write_text_file(const std::string &path, const std::string &text) {
#ifdef _WIN32
  (void)path;
  (void)text;
  return false;
#else
  int fd = open(path.c_str(), O_WRONLY | O_CREAT | O_TRUNC | O_CLOEXEC, 0600);
  if (fd < 0) {
    return false;
  }
  bool ok = true;
  size_t off = 0;
  while (off < text.size()) {
    ssize_t wrote = write(fd, text.data() + off, text.size() - off);
    if (wrote < 0 && errno == EINTR) {
      continue;
    }
    if (wrote <= 0) {
      ok = false;
      break;
    }
    off += static_cast<size_t>(wrote);
  }
  if (ok && (getenv("LUPINE_SNAPSHOT_FSYNC") == nullptr ||
             strcmp(getenv("LUPINE_SNAPSHOT_FSYNC"), "0") != 0)) {
    ok = fsync(fd) == 0;
  }
  close(fd);
  return ok;
#endif
}

bool manifest_exists(const std::string &artifact) {
  struct stat st = {};
  return stat(join_path(artifact, "manifest.json").c_str(), &st) == 0 &&
         S_ISREG(st.st_mode);
}

bool publish_artifact(const std::string &staging, const std::string &final_dir) {
  remove_tree(final_dir);
  return rename(staging.c_str(), final_dir.c_str()) == 0;
}

bool write_status(int fd, int result) {
#ifdef _WIN32
  (void)fd;
  (void)result;
  return false;
#else
  const char *cursor = reinterpret_cast<const char *>(&result);
  size_t written = 0;
  while (written < sizeof(result)) {
    ssize_t n = write(fd, cursor + written, sizeof(result) - written);
    if (n < 0 && errno == EINTR) {
      continue;
    }
    if (n <= 0) {
      return false;
    }
    written += static_cast<size_t>(n);
  }
  return true;
#endif
}

// CRIU log verbosity: warnings + errors only. The verbose log is written while
// the target (and its GPU) are frozen, so keep it minimal.
static const char *kCriuVerbosity = "-v1";

int run_criu_dump(pid_t pid, const std::string &images_dir,
                  const std::string &log_file) {
#ifdef _WIN32
  (void)pid;
  (void)images_dir;
  (void)log_file;
  return -1;
#else
  int status_pipe[2];
  if (pipe(status_pipe) != 0) {
    return -1;
  }
  pid_t launcher = fork();
  if (launcher < 0) {
    close(status_pipe[0]);
    close(status_pipe[1]);
    return -1;
  }
  if (launcher == 0) {
    close(status_pipe[0]);
    pid_t helper = fork();
    if (helper < 0) {
      int result = -1;
      write_status(status_pipe[1], result);
      _exit(1);
    }
    if (helper > 0) {
      _exit(0);
    }

    setsid();
    signal(SIGCHLD, SIG_DFL);
    pid_t criu_child = fork();
    if (criu_child < 0) {
      int result = -1;
      write_status(status_pipe[1], result);
      _exit(1);
    }
    if (criu_child == 0) {
      close(status_pipe[1]);
      signal(SIGCHLD, SIG_DFL);
      std::string pid_arg = std::to_string(static_cast<long long>(pid));
      execlp("criu", "criu", "dump", "--unprivileged", kCriuVerbosity, "--tree",
             pid_arg.c_str(), "--images-dir", images_dir.c_str(),
             "--leave-running", "--shell-job", "--tcp-close",
             "--file-locks", "--log-file", log_file.c_str(),
             static_cast<char *>(nullptr));
      _exit(127);
    }

    int child_status = 0;
    int result = -1;
    while (waitpid(criu_child, &child_status, 0) < 0) {
      if (errno == EINTR) {
        continue;
      }
      write_status(status_pipe[1], result);
      _exit(1);
    }
    if (WIFEXITED(child_status)) {
      result = WEXITSTATUS(child_status);
    }
    write_status(status_pipe[1], result);
    _exit(0);
  }

  close(status_pipe[1]);
  int result = -1;
  size_t got = 0;
  while (got < sizeof(result)) {
    ssize_t n = read(status_pipe[0], reinterpret_cast<char *>(&result) + got,
                     sizeof(result) - got);
    if (n < 0 && errno == EINTR) {
      continue;
    }
    if (n <= 0) {
      break;
    }
    got += static_cast<size_t>(n);
  }
  close(status_pipe[0]);
  return got == sizeof(result) ? result : -1;
#endif
}

int run_criu_restore(const std::string &images_dir, const std::string &log_file,
                     lupine_socket_t connfd,
                     const std::string &inherited_target) {
#ifdef _WIN32
  (void)images_dir;
  (void)log_file;
  (void)connfd;
  (void)inherited_target;
  return -1;
#else
  if (inherited_target.empty()) {
    return -1;
  }
  signal(SIGCHLD, SIG_DFL);
  int fd_flags = fcntl(connfd, F_GETFD);
  if (fd_flags >= 0) {
    fcntl(connfd, F_SETFD, fd_flags & ~FD_CLOEXEC);
  }
  std::string inherit_arg =
      "fd[" + std::to_string(connfd) + "]:" + inherited_target;
  pid_t child = fork();
  if (child < 0) {
    return -1;
  }
  if (child == 0) {
    signal(SIGCHLD, SIG_DFL);
    execlp("criu", "criu", "restore", "--unprivileged", kCriuVerbosity,
           "--images-dir", images_dir.c_str(), "--restore-detached",
           "--shell-job", "--tcp-close", "--file-locks", "--inherit-fd",
           inherit_arg.c_str(), "--log-file", log_file.c_str(),
           static_cast<char *>(nullptr));
    _exit(127);
  }
  int status = 0;
  while (waitpid(child, &status, 0) < 0) {
    if (errno == EINTR) {
      continue;
    }
    return -1;
  }
  if (WIFEXITED(status)) {
    return WEXITSTATUS(status);
  }
  return -1;
#endif
}

CUresult cuda_checkpoint_current_process(const std::string &images_dir,
                                         const std::string &log_file,
                                         int restore_fd,
                                         const std::string &restore_fd_target) {
  int criu_result = run_criu_dump(getpid(), images_dir, log_file);

  if (restore_fd >= 0 && !restore_fd_target.empty() &&
      fd_target(restore_fd) != restore_fd_target) {
    client_handler(restore_fd);
    _exit(0);
  }

  if (criu_result != 0) {
    LUPINE_LOG_ERROR("criu dump failed with exit code " << criu_result
                                                       << "; log: "
                                                       << log_file);
    return CUDA_ERROR_OPERATING_SYSTEM;
  }
  return CUDA_SUCCESS;
}

int prepare_restore_socket_placeholder(const std::string &path) {
#ifdef _WIN32
  (void)path;
  return -1;
#else
  int source = open(path.c_str(), O_RDWR | O_CREAT | O_TRUNC | O_CLOEXEC, 0600);
  if (source < 0) {
    LUPINE_LOG_ERROR("Failed to create snapshot client socket placeholder "
                     << path << ": errno=" << errno << " (" << strerror(errno)
                     << ")");
    return -1;
  }
  int restore_fd = fcntl(source, F_DUPFD, 64);
  int saved_errno = errno;
  close(source);
  errno = saved_errno;
  if (restore_fd < 0) {
    LUPINE_LOG_ERROR("Failed to allocate snapshot restore fd for "
                     << path << ": errno=" << errno << " (" << strerror(errno)
                     << ")");
  }
  return restore_fd;
#endif
}

std::string staging_root_for(const std::string &root) {
  const char *configured = getenv("LUPINE_SNAPSHOT_STAGING_DIR");
  if (configured != nullptr && configured[0] != '\0') {
    return configured;
  }
  return join_path(root, ".staging");
}

std::string manifest_json(const std::string &id, uint64_t bytes,
                          const std::string &client_fd_target) {
  char buf[1024];
  snprintf(buf, sizeof(buf),
           "{\n"
           "  \"id\": \"%s\",\n"
           "  \"state\": \"READY\",\n"
           "  \"bytes\": %llu,\n"
           "  \"created_unix_seconds\": %lld,\n"
           "  \"pid\": %lld,\n"
           "  \"client_fd_target\": \"%s\"\n"
           "}\n",
           id.c_str(), static_cast<unsigned long long>(bytes),
           static_cast<long long>(time(nullptr)), static_cast<long long>(getpid()),
           client_fd_target.c_str());
  return std::string(buf);
}

snapshot_result save_snapshot_artifact(const char *id, int client_fd) {
  const char *root_env = snapshot_root();
  if (root_env == nullptr) {
    return {CUDA_ERROR_NOT_SUPPORTED};
  }
  if (!lupine_snapshot_id_valid(id)) {
    return {CUDA_ERROR_INVALID_VALUE};
  }

  std::string root(root_env);
  std::string objects = join_path(root, "objects");
  std::string staging_root = staging_root_for(root);
  if (!mkdir_p(objects) || !mkdir_p(staging_root)) {
    return {CUDA_ERROR_OPERATING_SYSTEM};
  }

  std::string snapshot_id = snapshot_id_dir(id);
  std::string staging = join_path(staging_root, snapshot_id + ".partial");
  std::string criu_dir = join_path(staging, "criu");
  std::string logs_dir = join_path(staging, "logs");
  std::string final_dir = join_path(objects, snapshot_id);

  remove_tree(staging);
  if (!mkdir_p(criu_dir) || !mkdir_p(logs_dir)) {
    remove_tree(staging);
    return {CUDA_ERROR_OPERATING_SYSTEM};
  }

  int restore_fd = -1;
  std::string client_target;
  std::string restore_fd_target;
  if (client_fd >= 0) {
    restore_fd = prepare_restore_socket_placeholder(
        join_path(staging, "client-socket-placeholder"));
    if (restore_fd < 0) {
      remove_tree(staging);
      return {CUDA_ERROR_OPERATING_SYSTEM};
    }
    restore_fd_target = fd_target(restore_fd);
    if (restore_fd_target.empty()) {
      close(restore_fd);
      remove_tree(staging);
      return {CUDA_ERROR_OPERATING_SYSTEM};
    }
    client_target = criu_inherit_target(restore_fd_target);
  }
  if (client_fd >= 0) {
    redirect_stdio_to_devnull();
  }
  CUresult result = cuda_checkpoint_current_process(
      criu_dir, join_path(logs_dir, "dump.log"), restore_fd, restore_fd_target);
  if (restore_fd >= 0) {
    close(restore_fd);
  }
  if (result != CUDA_SUCCESS) {
    LUPINE_LOG_ERROR("Snapshot checkpoint failed for " << snapshot_id
                                                       << " with CUDA result "
                                                       << result);
    if (getenv("LUPINE_SNAPSHOT_KEEP_FAILED") == nullptr) {
      remove_tree(staging);
    } else {
      LUPINE_LOG_ERROR("Keeping failed snapshot staging directory " << staging);
    }
    return {result};
  }

  uint64_t bytes = tree_size(staging);
  if (!publish_artifact(staging, final_dir)) {
    remove_tree(staging);
    return {CUDA_ERROR_OPERATING_SYSTEM};
  }
  if (!write_text_file(join_path(final_dir, "manifest.json"),
                       manifest_json(snapshot_id, bytes, client_target))) {
    remove_tree(final_dir);
    return {CUDA_ERROR_OPERATING_SYSTEM};
  }
  return {CUDA_SUCCESS};
}

int recv_exact(lupine_socket_t connfd, void *data, size_t size) {
  char *cursor = static_cast<char *>(data);
  size_t read_bytes = 0;
  while (read_bytes < size) {
    ssize_t n = lupine_socket_recv(connfd, cursor + read_bytes, size - read_bytes);
    if (n < 0 && lupine_socket_error_is_intr()) {
      continue;
    }
    if (n <= 0) {
      return -1;
    }
    read_bytes += static_cast<size_t>(n);
  }
  return 0;
}

int peek_bootstrap_magic(lupine_socket_t connfd) {
#ifdef _WIN32
  (void)connfd;
  return 0;
#else
  char magic[sizeof(kBootstrapMagic) - 1] = {};
  size_t want = sizeof(magic);
  for (;;) {
    ssize_t n = recv(connfd, magic, want, MSG_PEEK);
    if (n < 0 && errno == EINTR) {
      continue;
    }
    if (n <= 0) {
      return -1;
    }
    if (magic[0] != kBootstrapMagic[0]) {
      return 0;
    }
    if (static_cast<size_t>(n) < want) {
      continue;
    }
    return memcmp(magic, kBootstrapMagic, want) == 0 ? 1 : 0;
  }
#endif
}

std::string read_manifest_client_fd_target(const std::string &manifest) {
#ifdef _WIN32
  (void)manifest;
  return "";
#else
  int fd = open(manifest.c_str(), O_RDONLY | O_CLOEXEC);
  if (fd < 0) {
    return "";
  }
  std::string text;
  char buf[4096];
  for (;;) {
    ssize_t n = read(fd, buf, sizeof(buf));
    if (n < 0 && errno == EINTR) {
      continue;
    }
    if (n <= 0) {
      break;
    }
    text.append(buf, static_cast<size_t>(n));
  }
  close(fd);
  const std::string key = "\"client_fd_target\": \"";
  size_t pos = text.find(key);
  if (pos == std::string::npos) {
    return "";
  }
  pos += key.size();
  size_t end = text.find('"', pos);
  if (end == std::string::npos) {
    return "";
  }
  return text.substr(pos, end - pos);
#endif
}

// The pid recorded in the manifest is the worker that produced the snapshot.
// CRIU restores a process at its original pid, so that pid must be free before
// restore. Returns 0 if the manifest has no pid.
pid_t read_manifest_pid(const std::string &manifest) {
#ifdef _WIN32
  (void)manifest;
  return 0;
#else
  int fd = open(manifest.c_str(), O_RDONLY | O_CLOEXEC);
  if (fd < 0) {
    return 0;
  }
  std::string text;
  char buf[4096];
  for (;;) {
    ssize_t n = read(fd, buf, sizeof(buf));
    if (n < 0 && errno == EINTR) {
      continue;
    }
    if (n <= 0) {
      break;
    }
    text.append(buf, static_cast<size_t>(n));
  }
  close(fd);
  const std::string key = "\"pid\": ";
  size_t pos = text.find(key);
  if (pos == std::string::npos) {
    return 0;
  }
  return static_cast<pid_t>(strtoll(text.c_str() + pos + key.size(), nullptr, 10));
#endif
}

// Wait for `pid` to exit. The snapshot is taken with CRIU `--leave-running`, so
// the worker that saved keeps running until its connection closes; a quick
// reconnect can race its exit and CRIU restore then fails with "Can't fork:
// File exists". Poll until the pid is gone (or a timeout elapses).
void wait_for_pid_exit(pid_t pid, int timeout_ms) {
#ifndef _WIN32
  if (pid <= 0) {
    return;
  }
  int waited = 0;
  while (waited < timeout_ms) {
    if (kill(pid, 0) != 0 && errno == ESRCH) {
      return;
    }
    struct timespec ts = {0, 5 * 1000 * 1000};  // 5ms
    nanosleep(&ts, nullptr);
    waited += 5;
  }
#else
  (void)pid;
  (void)timeout_ms;
#endif
}

int read_snapshot_id(conn_t *conn, std::string *id) {
  uint32_t len = 0;
  if (rpc_read(conn, &len, sizeof(len)) < 0) {
    LUPINE_LOG_ERROR("Snapshot request failed while reading id length.");
    return -1;
  }
  if (len == 0 || len > LUPINE_SNAPSHOT_ID_MAX_BYTES) {
    LUPINE_LOG_ERROR("Snapshot request had out-of-range id length " << len);
    return -1;
  }
  std::vector<char> buf(len + 1, '\0');
  if (rpc_read(conn, buf.data(), len) < 0) {
    LUPINE_LOG_ERROR("Snapshot request failed while reading id bytes.");
    return -1;
  }
  *id = std::string(buf.data(), len);
  if (!lupine_snapshot_id_valid(id->c_str())) {
    LUPINE_LOG_ERROR("Snapshot request had invalid id bytes: " << *id);
    return -1;
  }
  return 0;
}

} // namespace

int handle_lupine_snapshot_save_and_exit(conn_t *conn) {
  std::string id;
  if (read_snapshot_id(conn, &id) < 0) {
    LUPINE_LOG_ERROR("Snapshot save request had an invalid id.");
    return -1;
  }
  int request_id = rpc_read_end(conn);

  snapshot_result saved =
      save_snapshot_artifact(id.c_str(), static_cast<int>(conn->connfd));
  CUresult result = saved.cuda_result;
  LUPINE_LOG_DEBUG("Snapshot save for " << id << " completed with CUDA result "
                                        << result);
  if (rpc_write_start_response(conn, request_id) < 0) {
    LUPINE_LOG_ERROR("Snapshot save failed to start RPC response.");
    return -1;
  }
  if (rpc_write(conn, &result, sizeof(result)) < 0) {
    LUPINE_LOG_ERROR("Snapshot save failed to write RPC response.");
    return -1;
  }
  if (rpc_write_end(conn) < 0) {
    LUPINE_LOG_ERROR("Snapshot save failed to finish RPC response.");
    return -1;
  }

  conn->closed = 1;
#ifndef _WIN32
  // The snapshot has been taken with CRIU `--leave-running`, so this worker is
  // still alive. It has served its purpose now; exit immediately (rather than
  // unwinding the dispatch loop lazily) so its pid is freed for the restore,
  // which CRIU recreates at this same pid. A FIN flushes the response first.
  shutdown(conn->connfd, SHUT_WR);
  _exit(0);
#endif
  return 0;
}

int lupine_snapshot_read_bootstrap(lupine_socket_t connfd,
                                   char id[LUPINE_SNAPSHOT_ID_BUFFER_BYTES]) {
  int has_bootstrap = peek_bootstrap_magic(connfd);
  if (has_bootstrap <= 0) {
    return has_bootstrap;
  }

  char magic[sizeof(kBootstrapMagic) - 1] = {};
  uint32_t len = 0;
  if (recv_exact(connfd, magic, sizeof(magic)) < 0 ||
      memcmp(magic, kBootstrapMagic, sizeof(magic)) != 0 ||
      recv_exact(connfd, &len, sizeof(len)) < 0 || len == 0 ||
      len > LUPINE_SNAPSHOT_ID_MAX_BYTES || recv_exact(connfd, id, len) < 0) {
    return -1;
  }
  id[len] = '\0';
  return lupine_snapshot_id_valid(id) ? 1 : -1;
}

int lupine_snapshot_restore_for_connection(const char *id,
                                           lupine_socket_t connfd) {
  const char *root_env = snapshot_root();
  if (root_env == nullptr || !lupine_snapshot_id_valid(id)) {
    return -1;
  }

  std::string artifact =
      join_path(join_path(root_env, "objects"), snapshot_id_dir(id));
  std::string manifest = join_path(artifact, "manifest.json");
  if (!manifest_exists(artifact)) {
    return 1;
  }
  std::string inherited_target = read_manifest_client_fd_target(manifest);
  if (inherited_target.empty()) {
    return -1;
  }

  // CRIU restores the worker at its original pid; wait for the worker that
  // produced this snapshot to exit so its pid is free (avoids a "Can't fork:
  // File exists" race on a quick reconnect).
  wait_for_pid_exit(read_manifest_pid(manifest), 5000);

  std::string logs_dir = join_path(artifact, "logs");
  mkdir_p(logs_dir);
  int result = run_criu_restore(join_path(artifact, "criu"),
                                join_path(logs_dir, "restore.log"), connfd,
                                inherited_target);
  return result == 0 ? 0 : -1;
}

#else  // _WIN32: CRIU is Linux-only, so snapshots are unavailable here.

int handle_lupine_snapshot_save_and_exit(conn_t *conn) {
  (void)conn;
  return -1;
}

int lupine_snapshot_read_bootstrap(lupine_socket_t connfd,
                                   char id[LUPINE_SNAPSHOT_ID_BUFFER_BYTES]) {
  (void)connfd;
  (void)id;
  return 0;  // no bootstrap: every connection is handled as a fresh client
}

int lupine_snapshot_restore_for_connection(const char *id,
                                           lupine_socket_t connfd) {
  (void)id;
  (void)connfd;
  return 1;  // nothing restored
}

#endif  // _WIN32
