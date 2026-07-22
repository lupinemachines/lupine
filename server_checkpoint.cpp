#include "server_checkpoint.h"

#include "checkpoint.h"
#include "checkpoint_provider.h"
#include "lupine_log.h"

#ifndef _WIN32
#include <atomic>
#include <cerrno>
#include <cstddef>
#include <cstdlib>
#include <dlfcn.h>
#include <fcntl.h>
#include <signal.h>
#include <sys/socket.h>
#include <thread>
#include <unistd.h>
#endif

namespace {

#ifndef _WIN32

struct optional_checkpoint_provider {
  void *library = nullptr;
  const lupine_checkpoint_provider_v1 *api = nullptr;
  bool started = false;
};

struct child_checkpoint_state {
  int signal_pipe[2] = {-1, -1};
  lupine_socket_t connection = LUPINE_INVALID_SOCKET;
  std::thread signal_thread;
  struct sigaction previous_sigterm = {};
  optional_checkpoint_provider provider;
  const char *directory = nullptr;
  std::atomic<bool> checkpoint_requested{false};
  bool handler_installed = false;
  bool started = false;
};

child_checkpoint_state &checkpoint_state() {
  // The server has one connection per child. Keep this alive until process
  // exit so a late SIGTERM cannot race static destruction.
  static auto *state = new child_checkpoint_state();
  return *state;
}

volatile sig_atomic_t signal_write_fd = -1;
volatile sig_atomic_t sigterm_received = 0;

void checkpoint_sigterm_handler(int) {
  int saved_errno = errno;
  sigterm_received = 1;
  if (signal_write_fd >= 0) {
    const char event = 'T';
    ssize_t ignored =
        write(static_cast<int>(signal_write_fd), &event, sizeof(event));
    (void)ignored;
  }
  errno = saved_errno;
}

void unload_provider(optional_checkpoint_provider &provider) {
  if (provider.started && provider.api != nullptr &&
      provider.api->stop != nullptr) {
    provider.api->stop();
  }
  provider.started = false;
  provider.api = nullptr;
  if (provider.library != nullptr) {
    dlclose(provider.library);
    provider.library = nullptr;
  }
}

optional_checkpoint_provider load_provider() {
  optional_checkpoint_provider provider;
  const char *override_path = getenv("LUPINE_CHECKPOINT_LIBRARY");
  const char *candidates[] = {"liblupinecr.so.0", "liblupinecr.so"};

  if (override_path != nullptr && override_path[0] != '\0') {
    provider.library = dlopen(override_path, RTLD_NOW | RTLD_LOCAL);
  } else {
    for (const char *candidate : candidates) {
      provider.library = dlopen(candidate, RTLD_NOW | RTLD_LOCAL);
      if (provider.library != nullptr) {
        break;
      }
    }
  }
  if (provider.library == nullptr) {
    LUPINE_LOG_DEBUG("LupineCR provider is not installed; checkpointing is "
                     "disabled for this connection.");
    return provider;
  }

  auto get_provider = reinterpret_cast<lupine_checkpoint_provider_get_v1_fn>(
      dlsym(provider.library, LUPINE_CHECKPOINT_PROVIDER_SYMBOL));
  if (get_provider != nullptr) {
    provider.api = get_provider();
  }
  constexpr size_t required_size =
      offsetof(lupine_checkpoint_provider_v1, stop) +
      sizeof(lupine_checkpoint_provider_v1::stop);
  if (provider.api == nullptr || provider.api->struct_size < required_size ||
      provider.api->abi_version != LUPINE_CHECKPOINT_PROVIDER_ABI_VERSION ||
      provider.api->start == nullptr || provider.api->checkpoint == nullptr ||
      provider.api->stop == nullptr) {
    LUPINE_LOG_ERROR("Ignoring incompatible LupineCR checkpoint provider.");
    provider.api = nullptr;
    unload_provider(provider);
    return provider;
  }
  if (provider.api->start() != 0) {
    LUPINE_LOG_ERROR("LupineCR checkpoint provider failed to start.");
    provider.api = nullptr;
    unload_provider(provider);
    return provider;
  }
  provider.started = true;

  LUPINE_LOG_DEBUG("LupineCR checkpoint provider enabled.");
  return provider;
}

void wait_for_shutdown(child_checkpoint_state &state) {
  for (;;) {
    char event = 0;
    ssize_t size = read(state.signal_pipe[0], &event, sizeof(event));
    if (size < 0 && errno == EINTR) {
      continue;
    }
    if (size <= 0 || event == 'Q') {
      return;
    }
    if (event == 'T') {
      state.checkpoint_requested.store(true, std::memory_order_release);
      (void)shutdown(state.connection, SHUT_RDWR);
      return;
    }
  }
}

#endif

} // namespace

bool lupine_server_checkpoint_child_start(lupine_socket_t connection) {
#ifdef _WIN32
  (void)connection;
  return true;
#else
  child_checkpoint_state &state = checkpoint_state();
  if (state.started) {
    return false;
  }

  state.connection = connection;
  state.checkpoint_requested.store(false, std::memory_order_relaxed);
  sigterm_received = 0;
  state.directory = getenv("LUPINE_CHECKPOINT_DIR");
  if (state.directory != nullptr && state.directory[0] != '\0') {
    state.provider = load_provider();
  }

  if (pipe(state.signal_pipe) != 0) {
    unload_provider(state.provider);
    return false;
  }
  int flags = fcntl(state.signal_pipe[1], F_GETFL, 0);
  if (flags < 0 ||
      fcntl(state.signal_pipe[1], F_SETFL, flags | O_NONBLOCK) != 0) {
    close(state.signal_pipe[0]);
    close(state.signal_pipe[1]);
    state.signal_pipe[0] = state.signal_pipe[1] = -1;
    unload_provider(state.provider);
    return false;
  }

  struct sigaction action = {};
  action.sa_handler = checkpoint_sigterm_handler;
  sigemptyset(&action.sa_mask);
  if (sigaction(SIGTERM, &action, &state.previous_sigterm) != 0) {
    close(state.signal_pipe[0]);
    close(state.signal_pipe[1]);
    state.signal_pipe[0] = state.signal_pipe[1] = -1;
    unload_provider(state.provider);
    return false;
  }
  state.handler_installed = true;
  signal_write_fd = state.signal_pipe[1];

  try {
    state.signal_thread = std::thread([&state] { wait_for_shutdown(state); });
  } catch (...) {
    signal_write_fd = -1;
    (void)sigaction(SIGTERM, &state.previous_sigterm, nullptr);
    state.handler_installed = false;
    close(state.signal_pipe[0]);
    close(state.signal_pipe[1]);
    state.signal_pipe[0] = state.signal_pipe[1] = -1;
    unload_provider(state.provider);
    return false;
  }

  state.started = true;
  return true;
#endif
}

int lupine_server_checkpoint_child_finish() {
#ifdef _WIN32
  return 0;
#else
  child_checkpoint_state &state = checkpoint_state();
  if (!state.started) {
    return 0;
  }

  const char quit = 'Q';
  ssize_t ignored = write(state.signal_pipe[1], &quit, sizeof(quit));
  (void)ignored;
  if (state.signal_thread.joinable()) {
    state.signal_thread.join();
  }

  signal_write_fd = -1;
  if (state.handler_installed) {
    (void)sigaction(SIGTERM, &state.previous_sigterm, nullptr);
    state.handler_installed = false;
  }
  close(state.signal_pipe[0]);
  close(state.signal_pipe[1]);
  state.signal_pipe[0] = state.signal_pipe[1] = -1;

  int result = 0;
  bool should_checkpoint =
      sigterm_received != 0 ||
      state.checkpoint_requested.load(std::memory_order_acquire);
  if (should_checkpoint) {
    // This remains unconditional even when no provider is installed.
    lupine_checkpoint_drain_cuda_calls();
    if (state.provider.api != nullptr) {
      result = state.provider.api->checkpoint(state.directory,
                                              static_cast<uint64_t>(getpid()));
      if (result != 0) {
        LUPINE_LOG_ERROR("LupineCR failed to checkpoint connection process "
                         << getpid() << ".");
      }
    }
  }

  unload_provider(state.provider);
  state.connection = LUPINE_INVALID_SOCKET;
  state.directory = nullptr;
  state.started = false;
  return result;
#endif
}
