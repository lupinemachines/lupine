#include "snapshot.h"

#include <algorithm>
#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <signal.h>
#include <string>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>
#include <vector>

#ifndef _WIN32
#include <dirent.h>
#include <fcntl.h>
#endif

#include "lupine_log.h"

extern void client_handler(lupine_socket_t connfd);

namespace {

struct snapshot_result {
  CUresult cuda_result = CUDA_ERROR_UNKNOWN;
  std::string id;
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

bool mkdir_one(const std::string &path) {
  if (path.empty()) {
    return false;
  }
  if (mkdir(path.c_str(), 0700) == 0) {
    return true;
  }
  return errno == EEXIST;
}

bool mkdir_p(const std::string &path) {
  if (path.empty()) {
    return false;
  }
  std::string cur;
  size_t start = path[0] == '/' ? 1 : 0;
  if (start == 1) {
    cur = "/";
  }
  for (size_t i = start; i <= path.size(); ++i) {
    if (i != path.size() && path[i] != '/') {
      continue;
    }
    if (i == start) {
      continue;
    }
    cur = path.substr(0, i);
    if (!mkdir_one(cur)) {
      return false;
    }
  }
  return true;
}

bool valid_snapshot_id(const char *id) {
  if (id == nullptr || strlen(id) != LUPINE_SNAPSHOT_ID_HEX_BYTES) {
    return false;
  }
  for (size_t i = 0; i < LUPINE_SNAPSHOT_ID_HEX_BYTES; ++i) {
    char c = id[i];
    if (!((c >= '0' && c <= '9') || (c >= 'a' && c <= 'f'))) {
      return false;
    }
  }
  return true;
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

bool random_snapshot_id(char out[LUPINE_SNAPSHOT_ID_BUFFER_BYTES]) {
  unsigned char bytes[16] = {};
  bool ok = false;
#ifndef _WIN32
  int fd = open("/dev/urandom", O_RDONLY | O_CLOEXEC);
  if (fd >= 0) {
    size_t off = 0;
    while (off < sizeof(bytes)) {
      ssize_t got = read(fd, bytes + off, sizeof(bytes) - off);
      if (got < 0 && errno == EINTR) {
        continue;
      }
      if (got <= 0) {
        break;
      }
      off += static_cast<size_t>(got);
    }
    close(fd);
    ok = off == sizeof(bytes);
  }
#endif
  if (!ok) {
    uint64_t seed = static_cast<uint64_t>(time(nullptr)) ^
                    (static_cast<uint64_t>(getpid()) << 32);
    for (size_t i = 0; i < sizeof(bytes); ++i) {
      seed = seed * 6364136223846793005ULL + 1;
      bytes[i] = static_cast<unsigned char>(seed >> 56);
    }
  }
  static const char hex[] = "0123456789abcdef";
  for (size_t i = 0; i < sizeof(bytes); ++i) {
    out[i * 2] = hex[bytes[i] >> 4];
    out[i * 2 + 1] = hex[bytes[i] & 0xf];
  }
  out[LUPINE_SNAPSHOT_ID_HEX_BYTES] = '\0';
  return true;
}

uint64_t tree_size(const std::string &path) {
#ifdef _WIN32
  (void)path;
  return 0;
#else
  struct stat st = {};
  if (lstat(path.c_str(), &st) != 0) {
    return 0;
  }
  if (!S_ISDIR(st.st_mode)) {
    return static_cast<uint64_t>(std::max<off_t>(0, st.st_size));
  }
  uint64_t total = 0;
  DIR *dir = opendir(path.c_str());
  if (dir == nullptr) {
    return 0;
  }
  while (dirent *ent = readdir(dir)) {
    if (strcmp(ent->d_name, ".") == 0 || strcmp(ent->d_name, "..") == 0) {
      continue;
    }
    total += tree_size(join_path(path, ent->d_name));
  }
  closedir(dir);
  return total;
#endif
}

bool remove_tree(const std::string &path) {
#ifdef _WIN32
  (void)path;
  return false;
#else
  struct stat st = {};
  if (lstat(path.c_str(), &st) != 0) {
    return errno == ENOENT;
  }
  if (!S_ISDIR(st.st_mode)) {
    return unlink(path.c_str()) == 0;
  }
  DIR *dir = opendir(path.c_str());
  if (dir == nullptr) {
    return false;
  }
  bool ok = true;
  while (dirent *ent = readdir(dir)) {
    if (strcmp(ent->d_name, ".") == 0 || strcmp(ent->d_name, "..") == 0) {
      continue;
    }
    ok = remove_tree(join_path(path, ent->d_name)) && ok;
  }
  closedir(dir);
  return rmdir(path.c_str()) == 0 && ok;
#endif
}

bool copy_file_chunked(const std::string &src, const std::string &dst) {
#ifdef _WIN32
  (void)src;
  (void)dst;
  return false;
#else
  int in = open(src.c_str(), O_RDONLY | O_CLOEXEC);
  if (in < 0) {
    return false;
  }
  int out = open(dst.c_str(), O_WRONLY | O_CREAT | O_TRUNC | O_CLOEXEC, 0600);
  if (out < 0) {
    close(in);
    return false;
  }
  std::vector<char> buf(1024 * 1024);
  bool ok = true;
  for (;;) {
    ssize_t got = read(in, buf.data(), buf.size());
    if (got < 0 && errno == EINTR) {
      continue;
    }
    if (got < 0) {
      ok = false;
      break;
    }
    if (got == 0) {
      break;
    }
    ssize_t off = 0;
    while (off < got) {
      ssize_t wrote = write(out, buf.data() + off, static_cast<size_t>(got - off));
      if (wrote < 0 && errno == EINTR) {
        continue;
      }
      if (wrote <= 0) {
        ok = false;
        break;
      }
      off += wrote;
    }
    if (!ok) {
      break;
    }
  }
  if (getenv("LUPINE_SNAPSHOT_FSYNC") == nullptr ||
      strcmp(getenv("LUPINE_SNAPSHOT_FSYNC"), "0") != 0) {
    fsync(out);
  }
  close(out);
  close(in);
  return ok;
#endif
}

bool copy_tree(const std::string &src, const std::string &dst) {
#ifdef _WIN32
  (void)src;
  (void)dst;
  return false;
#else
  struct stat st = {};
  if (lstat(src.c_str(), &st) != 0) {
    return false;
  }
  if (!S_ISDIR(st.st_mode)) {
    return copy_file_chunked(src, dst);
  }
  if (!mkdir_one(dst)) {
    return false;
  }
  DIR *dir = opendir(src.c_str());
  if (dir == nullptr) {
    return false;
  }
  bool ok = true;
  while (dirent *ent = readdir(dir)) {
    if (strcmp(ent->d_name, ".") == 0 || strcmp(ent->d_name, "..") == 0) {
      continue;
    }
    ok = copy_tree(join_path(src, ent->d_name), join_path(dst, ent->d_name)) && ok;
  }
  closedir(dir);
  return ok;
#endif
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
  if (rename(staging.c_str(), final_dir.c_str()) == 0) {
    return true;
  }
  if (errno != EXDEV) {
    return false;
  }
  if (!copy_tree(staging, final_dir)) {
    remove_tree(final_dir);
    return false;
  }
  remove_tree(staging);
  return true;
}

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
      write(status_pipe[1], &result, sizeof(result));
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
      write(status_pipe[1], &result, sizeof(result));
      _exit(1);
    }
    if (criu_child == 0) {
      close(status_pipe[1]);
      signal(SIGCHLD, SIG_DFL);
      std::string pid_arg = std::to_string(static_cast<long long>(pid));
      execlp("criu", "criu", "dump", "--unprivileged", "-v4", "--tree",
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
      write(status_pipe[1], &result, sizeof(result));
      _exit(1);
    }
    if (WIFEXITED(child_status)) {
      result = WEXITSTATUS(child_status);
    }
    write(status_pipe[1], &result, sizeof(result));
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
    execlp("criu", "criu", "restore", "--unprivileged", "-v4", "--images-dir",
           images_dir.c_str(), "--restore-detached", "--shell-job",
           "--tcp-close", "--file-locks", "--inherit-fd",
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
                                         int client_fd,
                                         const std::string &client_fd_target) {
  int criu_result = run_criu_dump(getpid(), images_dir, log_file);

  if (client_fd >= 0 && !client_fd_target.empty() &&
      fd_target(client_fd) != client_fd_target) {
    client_handler(client_fd);
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

snapshot_result create_snapshot_artifact(unsigned int flags, const char *requested_id,
                                         int client_fd) {
  (void)flags;
  const char *root_env = snapshot_root();
  if (root_env == nullptr) {
    return {CUDA_ERROR_NOT_SUPPORTED, ""};
  }

  std::string root(root_env);
  std::string objects = join_path(root, "objects");
  std::string staging_root = staging_root_for(root);
  if (!mkdir_p(objects) || !mkdir_p(staging_root)) {
    return {CUDA_ERROR_OPERATING_SYSTEM, ""};
  }

  char id_buf[LUPINE_SNAPSHOT_ID_BUFFER_BYTES] = {};
  if (requested_id != nullptr && requested_id[0] != '\0') {
    if (!valid_snapshot_id(requested_id)) {
      return {CUDA_ERROR_INVALID_VALUE, ""};
    }
    memcpy(id_buf, requested_id, LUPINE_SNAPSHOT_ID_BUFFER_BYTES);
  } else {
    random_snapshot_id(id_buf);
  }
  std::string id(id_buf);
  std::string staging = join_path(staging_root, id + ".partial");
  std::string criu_dir = join_path(staging, "criu");
  std::string logs_dir = join_path(staging, "logs");
  std::string final_dir = join_path(objects, id);

  remove_tree(staging);
  if (!mkdir_p(criu_dir) || !mkdir_p(logs_dir)) {
    remove_tree(staging);
    return {CUDA_ERROR_OPERATING_SYSTEM, ""};
  }

  std::string client_target = client_fd >= 0 ? fd_target(client_fd) : "";
  if (client_fd >= 0) {
    redirect_stdio_to_devnull();
  }
  CUresult result = cuda_checkpoint_current_process(
      criu_dir, join_path(logs_dir, "dump.log"), client_fd, client_target);
  if (result != CUDA_SUCCESS) {
    LUPINE_LOG_ERROR("Snapshot checkpoint failed for " << id
                                                       << " with CUDA result "
                                                       << result);
    if (getenv("LUPINE_SNAPSHOT_KEEP_FAILED") == nullptr) {
      remove_tree(staging);
    } else {
      LUPINE_LOG_ERROR("Keeping failed snapshot staging directory " << staging);
    }
    return {result, ""};
  }

  uint64_t bytes = tree_size(staging);
  if (!publish_artifact(staging, final_dir)) {
    remove_tree(staging);
    return {CUDA_ERROR_OPERATING_SYSTEM, ""};
  }
  if (!write_text_file(join_path(final_dir, "manifest.json"),
                       manifest_json(id, bytes, client_target))) {
    remove_tree(final_dir);
    return {CUDA_ERROR_OPERATING_SYSTEM, ""};
  }
  return {CUDA_SUCCESS, id};
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

CUresult snapshot_status_impl(const char *id, lupine_snapshot_info *info) {
  if (info == nullptr) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  *info = {};
  info->state = LUPINE_SNAPSHOT_STATE_UNKNOWN;
  const char *root_env = snapshot_root();
  if (root_env == nullptr) {
    info->state = LUPINE_SNAPSHOT_STATE_UNSUPPORTED;
    return CUDA_ERROR_NOT_SUPPORTED;
  }
  if (!valid_snapshot_id(id)) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  std::string artifact = join_path(join_path(root_env, "objects"), id);
  if (!manifest_exists(artifact)) {
    return CUDA_ERROR_NOT_FOUND;
  }
  info->state = LUPINE_SNAPSHOT_STATE_READY;
  info->bytes = tree_size(artifact);
  struct stat st = {};
  if (stat(join_path(artifact, "manifest.json").c_str(), &st) == 0) {
    info->created_unix_seconds = static_cast<int64_t>(st.st_mtime);
  }
  return CUDA_SUCCESS;
}

CUresult snapshot_delete_impl(const char *id) {
  const char *root_env = snapshot_root();
  if (root_env == nullptr) {
    return CUDA_ERROR_NOT_SUPPORTED;
  }
  if (!valid_snapshot_id(id)) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  std::string artifact = join_path(join_path(root_env, "objects"), id);
  return remove_tree(artifact) ? CUDA_SUCCESS : CUDA_ERROR_OPERATING_SYSTEM;
}

int read_snapshot_id(conn_t *conn, std::string *id) {
  uint32_t len = 0;
  if (rpc_read(conn, &len, sizeof(len)) < 0) {
    LUPINE_LOG_ERROR("Snapshot request failed while reading id length.");
    return -1;
  }
  if (len != LUPINE_SNAPSHOT_ID_HEX_BYTES) {
    LUPINE_LOG_ERROR("Snapshot request had id length " << len << ", expected "
                                                       << LUPINE_SNAPSHOT_ID_HEX_BYTES);
    return -1;
  }
  std::vector<char> buf(len + 1, '\0');
  if (rpc_read(conn, buf.data(), len) < 0) {
    LUPINE_LOG_ERROR("Snapshot request failed while reading id bytes.");
    return -1;
  }
  *id = std::string(buf.data(), len);
  if (!valid_snapshot_id(id->c_str())) {
    LUPINE_LOG_ERROR("Snapshot request had invalid id bytes: " << *id);
    return -1;
  }
  return 0;
}

} // namespace

int handle_lupine_snapshot_create(conn_t *conn) {
  unsigned int flags = 0;
  if (rpc_read(conn, &flags, sizeof(flags)) < 0) {
    return -1;
  }
  int request_id = rpc_read_end(conn);
  snapshot_result created = create_snapshot_artifact(flags, nullptr, -1);
  uint32_t id_len = static_cast<uint32_t>(created.id.size());
  if (created.cuda_result != CUDA_SUCCESS) {
    id_len = 0;
  }
  if (rpc_write_start_response(conn, request_id) < 0 ||
      rpc_write(conn, &id_len, sizeof(id_len)) < 0 ||
      (id_len != 0 && rpc_write(conn, created.id.data(), id_len) < 0) ||
      rpc_write(conn, &created.cuda_result, sizeof(created.cuda_result)) < 0 ||
      rpc_write_end(conn) < 0) {
    return -1;
  }
  return 0;
}

int handle_lupine_snapshot_status(conn_t *conn) {
  std::string id;
  if (read_snapshot_id(conn, &id) < 0) {
    return -1;
  }
  int request_id = rpc_read_end(conn);
  lupine_snapshot_info info = {};
  CUresult result = snapshot_status_impl(id.c_str(), &info);
  if (rpc_write_start_response(conn, request_id) < 0 ||
      rpc_write(conn, &info, sizeof(info)) < 0 ||
      rpc_write(conn, &result, sizeof(result)) < 0 || rpc_write_end(conn) < 0) {
    return -1;
  }
  return 0;
}

int handle_lupine_snapshot_load(conn_t *conn) {
  std::string id;
  unsigned int flags = 0;
  if (read_snapshot_id(conn, &id) < 0 ||
      rpc_read(conn, &flags, sizeof(flags)) < 0) {
    return -1;
  }
  (void)flags;
  int request_id = rpc_read_end(conn);

  lupine_snapshot_info info = {};
  CUresult result = snapshot_status_impl(id.c_str(), &info);
  if (result == CUDA_SUCCESS) {
    // Restoring a CRIU image into this existing RPC connection is unsafe
    // without the planned gateway/worker split. Keep the API explicit.
    result = CUDA_ERROR_NOT_SUPPORTED;
  }
  if (rpc_write_start_response(conn, request_id) < 0 ||
      rpc_write(conn, &result, sizeof(result)) < 0 || rpc_write_end(conn) < 0) {
    return -1;
  }
  return 0;
}

int handle_lupine_snapshot_delete(conn_t *conn) {
  std::string id;
  if (read_snapshot_id(conn, &id) < 0) {
    return -1;
  }
  int request_id = rpc_read_end(conn);
  CUresult result = snapshot_delete_impl(id.c_str());
  if (rpc_write_start_response(conn, request_id) < 0 ||
      rpc_write(conn, &result, sizeof(result)) < 0 || rpc_write_end(conn) < 0) {
    return -1;
  }
  return 0;
}

int handle_lupine_snapshot_save_and_exit(conn_t *conn) {
  std::string id;
  if (read_snapshot_id(conn, &id) < 0) {
    LUPINE_LOG_ERROR("Snapshot save request had an invalid id.");
    return -1;
  }
  int request_id = rpc_read_end(conn);

  snapshot_result saved =
      create_snapshot_artifact(0, id.c_str(), static_cast<int>(conn->connfd));
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
      recv_exact(connfd, &len, sizeof(len)) < 0 ||
      len != LUPINE_SNAPSHOT_ID_HEX_BYTES ||
      recv_exact(connfd, id, LUPINE_SNAPSHOT_ID_HEX_BYTES) < 0) {
    return -1;
  }
  id[LUPINE_SNAPSHOT_ID_HEX_BYTES] = '\0';
  return valid_snapshot_id(id) ? 1 : -1;
}

int lupine_snapshot_restore_for_connection(const char *id,
                                           lupine_socket_t connfd) {
  const char *root_env = snapshot_root();
  if (root_env == nullptr || !valid_snapshot_id(id)) {
    return -1;
  }

  std::string artifact = join_path(join_path(root_env, "objects"), id);
  std::string manifest = join_path(artifact, "manifest.json");
  if (!manifest_exists(artifact)) {
    return 1;
  }
  std::string inherited_target = read_manifest_client_fd_target(manifest);
  if (inherited_target.empty()) {
    return -1;
  }

  std::string logs_dir = join_path(artifact, "logs");
  mkdir_p(logs_dir);
  int result = run_criu_restore(join_path(artifact, "criu"),
                                join_path(logs_dir, "restore.log"), connfd,
                                inherited_target);
  return result == 0 ? 0 : -1;
}
