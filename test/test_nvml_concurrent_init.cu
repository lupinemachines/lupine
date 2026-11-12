// torch calls nvmlInit on every device query, from whichever thread asks, while
// other threads keep using handles they already hold. Real NVML is thread-safe
// and a repeated init leaves existing handles valid.

#include <nvml.h>

#include <arpa/inet.h>
#include <dlfcn.h>
#include <netdb.h>
#include <poll.h>
#include <sys/socket.h>
#include <unistd.h>

#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

namespace {

constexpr int kThreads = 8;
constexpr auto kDuration = std::chrono::seconds(5);

using init_fn = nvmlReturn_t (*)();
using shutdown_fn = nvmlReturn_t (*)();
using get_version_fn = nvmlReturn_t (*)(char *, unsigned int);
using handle_by_index_fn = nvmlReturn_t (*)(unsigned int, nvmlDevice_t *);
using get_name_fn = nvmlReturn_t (*)(nvmlDevice_t, char *, unsigned int);

template <typename T> T lookup(void *library, const char *name) {
  auto symbol = reinterpret_cast<T>(dlsym(library, name));
  if (symbol == nullptr) {
    std::fprintf(stderr, "missing NVML symbol %s\n", name);
  }
  return symbol;
}

int connect_tcp_endpoint(const std::string &endpoint) {
  auto colon = endpoint.rfind(':');
  if (colon == std::string::npos || endpoint.find(',') != std::string::npos ||
      endpoint.rfind("unix:", 0) == 0) {
    return -1;
  }
  std::string host = endpoint.substr(0, colon);
  std::string port = endpoint.substr(colon + 1);
  addrinfo hints = {};
  hints.ai_socktype = SOCK_STREAM;
  addrinfo *res = nullptr;
  if (getaddrinfo(host.c_str(), port.c_str(), &hints, &res) != 0) {
    return -1;
  }
  int fd = -1;
  for (addrinfo *ai = res; ai != nullptr; ai = ai->ai_next) {
    fd = socket(ai->ai_family, ai->ai_socktype, ai->ai_protocol);
    if (fd >= 0 && connect(fd, ai->ai_addr, ai->ai_addrlen) == 0) {
      break;
    }
    if (fd >= 0) {
      close(fd);
      fd = -1;
    }
  }
  freeaddrinfo(res);
  return fd;
}

bool verify_failed_init_rollback(init_fn nvml_init, shutdown_fn nvml_shutdown,
                                 get_version_fn get_version,
                                 handle_by_index_fn handle_by_index,
                                 get_name_fn get_name, const char *expected) {
  const char *env = std::getenv("LUPINE_SERVER");
  if (env == nullptr || env[0] == '\0') {
    return true;
  }
  std::string saved_server(env);

  // 1. Dial failure on 0 -> 1 must leave NVML uninitialized.
  setenv("LUPINE_SERVER", "127.0.0.1:1", 1);
  if (nvml_init() == NVML_SUCCESS ||
      nvml_shutdown() != NVML_ERROR_UNINITIALIZED) {
    std::fprintf(stderr, "dial failure leaked init_refcount\n");
    setenv("LUPINE_SERVER", saved_server.c_str(), 1);
    return false;
  }

  // 2. Post-open RPC failure during 0 -> 1 must roll back both init_refcount
  // and the broken transport session so a later init can reconnect.
  int upstream = connect_tcp_endpoint(saved_server);
  if (upstream >= 0) {
    int listen_fd = socket(AF_INET, SOCK_STREAM, 0);
    sockaddr_in addr = {};
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    socklen_t len = sizeof(addr);
    if (listen_fd >= 0 &&
        bind(listen_fd, reinterpret_cast<sockaddr *>(&addr), len) == 0 &&
        getsockname(listen_fd, reinterpret_cast<sockaddr *>(&addr), &len) ==
            0 &&
        listen(listen_fd, 1) == 0) {
      std::atomic<bool> drop{false};
      std::thread relay([&] {
        int client = accept(listen_fd, nullptr, nullptr);
        if (client < 0) {
          close(upstream);
          return;
        }
        pollfd fds[2] = {{client, POLLIN, 0}, {upstream, POLLIN, 0}};
        char buf[4096];
        while (!drop.load(std::memory_order_acquire) && poll(fds, 2, 20) >= 0) {
          for (int i = 0; i < 2; ++i) {
            if (fds[i].revents & (POLLIN | POLLHUP | POLLERR)) {
              ssize_t n = recv(fds[i].fd, buf, sizeof(buf), 0);
              if (n <= 0 || send(fds[1 - i].fd, buf, n, MSG_NOSIGNAL) != n) {
                drop.store(true, std::memory_order_release);
                break;
              }
            }
          }
        }
        shutdown(client, SHUT_RDWR);
        shutdown(upstream, SHUT_RDWR);
        close(client);
        close(upstream);
      });

      std::string relay_addr =
          "127.0.0.1:" + std::to_string(ntohs(addr.sin_port));
      setenv("LUPINE_SERVER", relay_addr.c_str(), 1);
      char version[NVML_SYSTEM_NVML_VERSION_BUFFER_SIZE] = {};
      bool version_ok = get_version(version, sizeof(version)) == NVML_SUCCESS;
      drop.store(true, std::memory_order_release);
      relay.join();
      close(listen_fd);

      nvmlReturn_t broken_init = nvml_init();
      nvmlReturn_t after_broken = nvml_shutdown();
      if (!version_ok || broken_init != NVML_ERROR_UNKNOWN ||
          after_broken != NVML_ERROR_UNINITIALIZED) {
        std::fprintf(stderr,
                     "RPC init failure rollback failed: ver_ok=%d init=%d "
                     "shutdown=%d\n",
                     version_ok, broken_init, after_broken);
        setenv("LUPINE_SERVER", saved_server.c_str(), 1);
        return false;
      }
    } else {
      if (listen_fd >= 0) {
        close(listen_fd);
      }
      close(upstream);
    }
  }

  setenv("LUPINE_SERVER", saved_server.c_str(), 1);
  nvmlDevice_t recovered = nullptr;
  char name[NVML_DEVICE_NAME_BUFFER_SIZE] = {};
  if (nvml_init() != NVML_SUCCESS ||
      handle_by_index(0, &recovered) != NVML_SUCCESS ||
      get_name(recovered, name, sizeof(name)) != NVML_SUCCESS ||
      std::strcmp(name, expected) != 0 || nvml_shutdown() != NVML_SUCCESS ||
      nvml_shutdown() != NVML_ERROR_UNINITIALIZED) {
    std::fprintf(stderr, "recovery after failed init failed\n");
    return false;
  }
  return true;
}

} // namespace

int main() {
  void *library = dlopen("libnvidia-ml.so.1", RTLD_NOW);
  if (library == nullptr) {
    std::fprintf(stderr, "dlopen(libnvidia-ml.so.1) failed: %s\n", dlerror());
    return 1;
  }
  auto nvml_init = lookup<init_fn>(library, "nvmlInit_v2");
  auto nvml_shutdown = lookup<shutdown_fn>(library, "nvmlShutdown");
  auto get_version =
      lookup<get_version_fn>(library, "nvmlSystemGetNVMLVersion");
  auto handle_by_index =
      lookup<handle_by_index_fn>(library, "nvmlDeviceGetHandleByIndex_v2");
  auto get_name = lookup<get_name_fn>(library, "nvmlDeviceGetName");
  if (nvml_init == nullptr || nvml_shutdown == nullptr ||
      get_version == nullptr || handle_by_index == nullptr ||
      get_name == nullptr) {
    return 1;
  }

  nvmlDevice_t device = nullptr;
  char expected[NVML_DEVICE_NAME_BUFFER_SIZE] = {};
  if (nvml_init() != NVML_SUCCESS ||
      handle_by_index(0, &device) != NVML_SUCCESS ||
      get_name(device, expected, sizeof(expected)) != NVML_SUCCESS) {
    std::fprintf(stderr, "single-threaded NVML setup failed\n");
    return 1;
  }

  std::atomic<long> iterations{0};
  std::atomic<long> failures{0};
  auto deadline = std::chrono::steady_clock::now() + kDuration;
  std::vector<std::thread> threads;
  for (int t = 0; t < kThreads; ++t) {
    threads.emplace_back([&, t] {
      while (std::chrono::steady_clock::now() < deadline) {
        nvmlReturn_t init = nvml_init();
        nvmlDevice_t handle = nullptr;
        nvmlReturn_t lookup_status = handle_by_index(0, &handle);
        char name[NVML_DEVICE_NAME_BUFFER_SIZE] = {};
        nvmlReturn_t name_status = lookup_status == NVML_SUCCESS
                                       ? get_name(handle, name, sizeof(name))
                                       : lookup_status;
        // The handle taken before the threads started must stay valid too.
        char held[NVML_DEVICE_NAME_BUFFER_SIZE] = {};
        nvmlReturn_t held_status = get_name(device, held, sizeof(held));
        if (init != NVML_SUCCESS || name_status != NVML_SUCCESS ||
            held_status != NVML_SUCCESS || std::strcmp(name, expected) != 0 ||
            std::strcmp(held, expected) != 0) {
          if (failures.fetch_add(1) < 5) {
            std::fprintf(stderr,
                         "thread %d: init %d lookup %d name %d \"%s\" held %d "
                         "\"%s\"\n",
                         t, init, lookup_status, name_status, name, held_status,
                         held);
          }
        }
        iterations.fetch_add(1);
      }
    });
  }
  for (auto &thread : threads) {
    thread.join();
  }
  for (long i = 0; i <= iterations.load(); ++i) {
    nvml_shutdown();
  }
  if (nvml_shutdown() != NVML_ERROR_UNINITIALIZED) {
    std::fprintf(stderr, "expected UNINITIALIZED after draining refcount\n");
    failures.fetch_add(1);
  }

  // Exercise concurrent 0 <-> N transitions where the last thread's
  // nvmlShutdown tears down transport while peers race into nvmlInit_v2.
  long lifecycle_cycles = 0;
  std::atomic<long> churn_iterations{0};
  auto churn_deadline =
      std::chrono::steady_clock::now() + std::chrono::seconds(2);
  threads.clear();
  for (int t = 0; t < kThreads; ++t) {
    threads.emplace_back([&, t] {
      while (std::chrono::steady_clock::now() < churn_deadline) {
        nvmlReturn_t init = nvml_init();
        if (init != NVML_SUCCESS) {
          if (failures.fetch_add(1) < 5) {
            std::fprintf(stderr, "churn thread %d: init %d\n", t, init);
          }
          continue;
        }
        nvmlDevice_t handle = nullptr;
        nvmlReturn_t lookup_status = handle_by_index(0, &handle);
        char name[NVML_DEVICE_NAME_BUFFER_SIZE] = {};
        nvmlReturn_t name_status = lookup_status == NVML_SUCCESS
                                       ? get_name(handle, name, sizeof(name))
                                       : lookup_status;
        nvmlReturn_t shut = nvml_shutdown();
        if (name_status != NVML_SUCCESS || shut != NVML_SUCCESS ||
            std::strcmp(name, expected) != 0) {
          if (failures.fetch_add(1) < 5) {
            std::fprintf(
                stderr,
                "churn thread %d: lookup %d name %d \"%s\" shutdown %d\n", t,
                lookup_status, name_status, name, shut);
          }
        }
        churn_iterations.fetch_add(1);
      }
    });
  }
  for (auto &thread : threads) {
    thread.join();
  }
  lifecycle_cycles = churn_iterations.load();
  if (nvml_shutdown() != NVML_ERROR_UNINITIALIZED) {
    std::fprintf(stderr,
                 "expected UNINITIALIZED after concurrent init/shutdown\n");
    failures.fetch_add(1);
  }
  if (!verify_failed_init_rollback(nvml_init, nvml_shutdown, get_version,
                                   handle_by_index, get_name, expected)) {
    failures.fetch_add(1);
  }
  std::printf("%ld iterations, %ld lifecycle cycles, %ld failures\n",
              iterations.load(), lifecycle_cycles, failures.load());
  return failures.load() == 0 ? 0 : 1;
}
