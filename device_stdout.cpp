#include "device_stdout.h"
#include "lupine_platform.h"
#include "rpc.h"

#include <atomic>
#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <iostream>
#if defined(__linux__)
#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>
#endif

static pthread_mutex_t lupine_stdout_capture_mutex = PTHREAD_MUTEX_INITIALIZER;
static std::atomic<bool> lupine_stdout_capture_required{false};

// The CUDA driver writes device printf output to fd 1 (process stdout) from its
// own thread once a kernel finishes (see issue #294). The lupine server writes
// all of its own diagnostics to stderr, so fd 1 is exclusively the
// device-printf channel: it is pointed at an anonymous file for the rest of the
// process, and each synchronization then reads what was appended since the
// last one through a second descriptor with its own offset.
static int lupine_stdout_capture_reader() {
  static int reader = []() -> int {
    FILE *file = nullptr;
#if defined(__linux__)
    int fd = memfd_create("lupine-stdout-capture", MFD_CLOEXEC);
    if (fd >= 0 && (file = fdopen(fd, "w+")) == nullptr) {
      close(fd);
    }
#endif
    if (file == nullptr && (file = tmpfile()) == nullptr) {
      return -1;
    }
    int writer = lupine_fd_fileno(file);
    int result = lupine_fd_reopen(writer);
    fflush(stdout);
    std::cout.flush();
    if (result < 0 || lupine_fd_dup2(writer, LUPINE_STDOUT_FD) < 0) {
      return -1;
    }
    return result;
  }();
  return reader;
}

// Redirects before any kernel of a printf-capable image can run.
void lupine_require_stdout_capture() {
  (void)lupine_stdout_capture_reader();
  lupine_stdout_capture_required.store(true, std::memory_order_release);
}

bool lupine_start_stdout_capture(lupine_captured_stdout *capture, bool force) {
  capture->output.clear();
  capture->active = (force || lupine_stdout_capture_required.load(
                                  std::memory_order_acquire)) &&
                    lupine_stdout_capture_reader() >= 0;
  return capture->active;
}

void lupine_finish_stdout_capture(lupine_captured_stdout *capture) {
  if (!capture->active) {
    return;
  }
  capture->active = false;
  fflush(stdout);
  std::cout.flush();
  int reader = lupine_stdout_capture_reader();
  pthread_mutex_lock(&lupine_stdout_capture_mutex);
  static uint64_t consumed = 0;
  char buffer[4096];
  for (;;) {
    ssize_t bytes = lupine_fd_read(reader, buffer, sizeof(buffer));
    if (bytes > 0) {
      capture->output.append(buffer, static_cast<size_t>(bytes));
      continue;
    }
    if (bytes < 0 && errno == EINTR) {
      continue;
    }
    break;
  }
  consumed += capture->output.size();
#if defined(__linux__)
  // The file only grows, so release the pages already read.
  if (!capture->output.empty()) {
    (void)fallocate(reader, FALLOC_FL_PUNCH_HOLE | FALLOC_FL_KEEP_SIZE, 0,
                    static_cast<off_t>(consumed));
  }
#endif
  pthread_mutex_unlock(&lupine_stdout_capture_mutex);
}

int lupine_write_captured_stdout(conn_t *conn,
                                 const lupine_captured_stdout &capture) {
  auto *output_size = static_cast<uint64_t *>(
      rpc_write_buffer(conn, sizeof(uint64_t), alignof(uint64_t)));
  if (output_size == nullptr) {
    return -1;
  }
  *output_size = capture.output.size();
  if (rpc_write(conn, capture.output.data(), capture.output.size()) < 0) {
    return -1;
  }
  return 0;
}

int lupine_read_captured_stdout(conn_t *conn) {
  uint64_t output_size = 0;
  if (rpc_read_buffer(conn, &output_size, sizeof(output_size)) < 0) {
    return -1;
  }
  if (output_size == 0) {
    return 0;
  }
  std::string output;
  output.resize(static_cast<size_t>(output_size));
  if (rpc_read(conn, output.data(), output.size()) < 0) {
    return -1;
  }
  fflush(stdout);
  std::cout.flush();
  return fwrite(output.data(), 1, output.size(), stdout) == output.size() ? 0
                                                                          : -1;
}
