#include "device_stdout.h"
#include "lupine_platform.h"
#include "rpc.h"

#include <atomic>
#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <iostream>
#if defined(__linux__)
#include <sys/mman.h>
#include <unistd.h>
#endif

static pthread_mutex_t lupine_stdout_capture_mutex = PTHREAD_MUTEX_INITIALIZER;
static std::atomic<bool> lupine_stdout_capture_required{false};

void lupine_require_stdout_capture() {
  lupine_stdout_capture_required.store(true, std::memory_order_release);
}

// Device printf output is drained by the CUDA driver as a write to fd 1
// (process stdout) during synchronization (see issue #294). We capture it by
// temporarily redirecting fd 1 to a backing file we can read back. The lupine
// server writes all of its own diagnostics to stderr, so fd 1 is exclusively
// the device-printf channel and nothing else can contaminate the capture.
//
// This returns a single process-global, reusable backing file: created once
// on first use and kept open for the process lifetime, so the per-synchronize
// hot path performs no filesystem open/close. On Linux it is an anonymous
// in-memory file from memfd_create() (no path, no /tmp, no inode, no page
// cache of a real file); other platforms (and old kernels without memfd)
// fall back to a single tmpfile() created once. The file is reset (truncated
// to empty) at the start of each capture.
static FILE *lupine_stdout_capture_file() {
  static FILE *file = []() -> FILE * {
#if defined(__linux__)
    int fd = memfd_create("lupine-stdout-capture", MFD_CLOEXEC);
    if (fd >= 0) {
      FILE *f = fdopen(fd, "w+");
      if (f != nullptr) {
        return f;
      }
      // fdopen failed; reclaim the fd and fall through to tmpfile().
      close(fd);
    }
#endif
    return tmpfile();
  }();
  return file;
}

bool lupine_start_stdout_capture(lupine_captured_stdout *capture, bool force) {
  if (capture == nullptr) {
    return false;
  }
  capture->saved_stdout = -1;
  capture->active = false;
  capture->output.clear();

  // Redirecting fd 1 is process-global, so only pay the serialization and
  // syscall cost after a loaded image has shown that device stdout may be used.
  if (!force &&
      !lupine_stdout_capture_required.load(std::memory_order_acquire)) {
    return false;
  }

  FILE *capture_file = lupine_stdout_capture_file();
  if (capture_file == nullptr) {
    return false;
  }
  int capture_fd = lupine_fd_fileno(capture_file);
  if (capture_fd < 0) {
    return false;
  }

  if (pthread_mutex_lock(&lupine_stdout_capture_mutex) != 0) {
    return false;
  }

  fflush(stdout);
  std::cout.flush();

  // Reset the reused backing file to empty so this capture only contains
  // output produced during the synchronization below.
  if (lupine_fd_truncate(capture_fd, 0) != 0 ||
      lupine_fd_seek(capture_fd, 0, SEEK_SET) < 0) {
    pthread_mutex_unlock(&lupine_stdout_capture_mutex);
    return false;
  }

  capture->saved_stdout = lupine_fd_dup(LUPINE_STDOUT_FD);
  if (capture->saved_stdout < 0) {
    pthread_mutex_unlock(&lupine_stdout_capture_mutex);
    return false;
  }

  if (lupine_fd_dup2(capture_fd, LUPINE_STDOUT_FD) < 0) {
    lupine_fd_close(capture->saved_stdout);
    capture->saved_stdout = -1;
    pthread_mutex_unlock(&lupine_stdout_capture_mutex);
    return false;
  }

  capture->active = true;
  return true;
}

void lupine_finish_stdout_capture(lupine_captured_stdout *capture) {
  if (capture == nullptr || !capture->active) {
    return;
  }

  fflush(stdout);
  std::cout.flush();
  lupine_fd_dup2(capture->saved_stdout, LUPINE_STDOUT_FD);
  lupine_fd_close(capture->saved_stdout);
  capture->saved_stdout = -1;

  // The backing file is process-global and reused, so read it back without
  // closing it. Its extent is exactly the bytes written during this capture
  // (it was truncated to empty on entry).
  FILE *capture_file = lupine_stdout_capture_file();
  if (capture_file != nullptr) {
    int capture_fd = lupine_fd_fileno(capture_file);
    if (capture_fd >= 0 && lupine_fd_seek(capture_fd, 0, SEEK_SET) >= 0) {
      char buffer[4096];
      for (;;) {
        ssize_t bytes = lupine_fd_read(capture_fd, buffer, sizeof(buffer));
        if (bytes > 0) {
          capture->output.append(buffer, static_cast<size_t>(bytes));
          continue;
        }
        if (bytes == 0) {
          break;
        }
        if (errno == EINTR) {
          continue;
        }
        break;
      }
    }
  }
  capture->active = false;
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
