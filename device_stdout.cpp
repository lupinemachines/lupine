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
      if (result >= 0) {
#ifdef _WIN32
        _close(result);
#else
        close(result);
#endif
      }
      fclose(file);
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
  capture->wire_size = 0;
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
#if defined(__linux__)
  static uint64_t punched = 0;
  static const uint64_t punch_page_mask = []() -> uint64_t {
    long page_size = sysconf(_SC_PAGESIZE);
    uint64_t granularity = (page_size > 0 && (page_size & (page_size - 1)) == 0)
                               ? static_cast<uint64_t>(page_size)
                               : uint64_t{4096};
    return ~(granularity - 1);
  }();
#endif
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
  capture->wire_size = capture->output.size();
  consumed += capture->wire_size;
#if defined(__linux__)
  // The file only grows; incrementally release newly completed whole pages.
  uint64_t punch_end = consumed & punch_page_mask;
  if (punch_end > punched) {
    if (fallocate(reader, FALLOC_FL_PUNCH_HOLE | FALLOC_FL_KEEP_SIZE,
                  static_cast<off_t>(punched),
                  static_cast<off_t>(punch_end - punched)) == 0 ||
        (errno != EINTR && errno != EAGAIN)) {
      punched = punch_end;
    }
  }
#endif
  pthread_mutex_unlock(&lupine_stdout_capture_mutex);
}

int lupine_write_captured_stdout(conn_t *conn,
                                 const lupine_captured_stdout &capture) {
  return rpc_write(conn, &capture.wire_size, sizeof(capture.wire_size)) < 0 ||
                 rpc_write(conn, capture.output.data(), capture.output.size()) <
                     0
             ? -1
             : 0;
}

int lupine_read_captured_stdout(conn_t *conn) {
  uint64_t output_size = 0;
  if (rpc_read_buffer(conn, &output_size, sizeof(output_size)) < 0) {
    return -1;
  }
  if (output_size == 0) {
    return 0;
  }
  fflush(stdout);
  std::cout.flush();
  char buffer[4096];
  uint64_t remaining = output_size;
  while (remaining != 0) {
    size_t chunk = remaining < sizeof(buffer) ? static_cast<size_t>(remaining)
                                              : sizeof(buffer);
    if (rpc_read(conn, buffer, chunk) < 0 ||
        fwrite(buffer, 1, chunk, stdout) != chunk) {
      return -1;
    }
    remaining -= chunk;
  }
  return 0;
}
