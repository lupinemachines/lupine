#include <cerrno>
#include <condition_variable>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <memory>
#include <mutex>
#include <stdio.h>
#include <thread>
#include <unordered_map>
#include <unordered_set>

#ifndef _WIN32
#include <algorithm>
#include <arpa/inet.h>
#include <netdb.h>
#include <netinet/tcp.h>
#include <poll.h>
#include <signal.h>
#include <sys/socket.h>
#include <sys/wait.h>
#include <unistd.h>
#include <vector>
#endif

#include "ipc.h"
#include "lupine_log.h"
#include "rpc.h"
#include "rpc_server.h"
#ifdef LUPINE_BUILD_CUDA_BACKEND
#include "checkpoint.h"
#include "copy_pipeline.h"
#include "server_checkpoint.h"
#endif

#define DEFAULT_PORT 14833
#define MAX_CLIENTS 10
#define MAX_LANES 256

#ifndef _WIN32
static volatile sig_atomic_t lupine_parent_termination_requested = 0;
static volatile sig_atomic_t lupine_parent_child_exited = 0;
static volatile sig_atomic_t lupine_child_connection_fd = -1;

static void lupine_parent_sigterm_handler(int) {
  lupine_parent_termination_requested = 1;
}

static void lupine_parent_sigchld_handler(int) {
  lupine_parent_child_exited = 1;
}

static void lupine_child_sigterm_handler(int) {
  int saved_errno = errno;
  if (lupine_child_connection_fd >= 0) {
    (void)shutdown(static_cast<int>(lupine_child_connection_fd), SHUT_RDWR);
  }
  errno = saved_errno;
}

static bool lupine_install_parent_signal_handlers() {
  struct sigaction term_action = {};
  term_action.sa_handler = lupine_parent_sigterm_handler;
  sigemptyset(&term_action.sa_mask);

  struct sigaction child_action = {};
  child_action.sa_handler = lupine_parent_sigchld_handler;
  sigemptyset(&child_action.sa_mask);

  return sigaction(SIGTERM, &term_action, nullptr) == 0 &&
         sigaction(SIGCHLD, &child_action, nullptr) == 0;
}

static bool lupine_install_child_signal_handler(lupine_socket_t connection) {
  struct sigaction action = {};
  action.sa_handler = lupine_child_sigterm_handler;
  sigemptyset(&action.sa_mask);
  lupine_child_connection_fd = connection;
  return sigaction(SIGTERM, &action, nullptr) == 0;
}

static void
lupine_reap_connection_children(std::unordered_set<pid_t> &children) {
  sigset_t child_mask;
  sigset_t previous_mask;
  sigemptyset(&child_mask);
  sigaddset(&child_mask, SIGCHLD);
  bool signal_blocked =
      sigprocmask(SIG_BLOCK, &child_mask, &previous_mask) == 0;

  for (;;) {
    int status = 0;
    pid_t child = waitpid(-1, &status, WNOHANG);
    if (child <= 0) {
      break;
    }
    if (WIFSIGNALED(status)) {
      LUPINE_LOG_ERROR("Connection child " << child << " terminated by signal "
                                           << WTERMSIG(status));
    } else if (!WIFEXITED(status) || WEXITSTATUS(status) != EXIT_SUCCESS) {
      LUPINE_LOG_ERROR("Connection child "
                       << child << " exited abnormally with status " << status);
    }
    children.erase(child);
  }
  lupine_parent_child_exited = 0;
  if (signal_blocked) {
    (void)sigprocmask(SIG_SETMASK, &previous_mask, nullptr);
  }
}
#endif

struct lupine_lane {
  uint64_t id = 0;
  std::mutex mutex;
  std::condition_variable cond;
  bool ready = false;
  int op = 0;
  std::thread worker;
};

int rpc_server_dispatch(const rpc_handler_registry &handlers, conn_t *conn,
                        int op) {
  LUPINE_TRACE_LOG("LUPINE server handling op " << op);
  auto it = handlers.find(op);
  if (it == handlers.end()) {
    LUPINE_LOG_ERROR("No RPC handler for op " << op << "; closing client.");
    return -1;
  }

  const rpc_handler &handler = it->second;
  const char *backend_name = nullptr;
  int result = -1;
  switch (handler.backend) {
  case rpc_backend::cuda:
#ifdef LUPINE_BUILD_CUDA_BACKEND
  {
    backend_name = "CUDA";
    lupine_checkpoint::cuda_call_guard guard;
    result = handler.handler(conn);
    break;
  }
#else
    break;
#endif
  case rpc_backend::nvml:
#ifdef LUPINE_BUILD_NVML_BACKEND
    backend_name = "NVML";
    result = handler.handler(conn);
#endif
    break;
  case rpc_backend::hip:
#ifdef LUPINE_BUILD_HIP_BACKEND
    backend_name = "HIP";
    result = handler.handler(conn);
#endif
    break;
  }

  if (result >= 0) {
    return 0;
  }
  if (backend_name == nullptr) {
    LUPINE_LOG_ERROR("RPC op " << op << " belongs to a disabled backend.");
  } else {
    LUPINE_LOG_ERROR("Error handling " << backend_name << " request for op "
                                       << op << ".");
  }
  return -1;
}

int client_handler(lupine_socket_t connfd) {
  const rpc_handler_registry &handlers = lupine_rpc_handlers();
  conn_t conn = {};
  if (rpc_conn_init(&conn, connfd, 1) < 0) {
    LUPINE_LOG_ERROR("Error initializing connection synchronization.");
#ifdef LUPINE_BUILD_CUDA_BACKEND
    return lupine_server_checkpoint_child_finish();
#else
    return 0;
#endif
  }

  const rpc_http2_server_metadata metadata = {
#ifdef LUPINE_BACKEND_VERSION
      LUPINE_BACKEND_VERSION,
#else
      nullptr,
#endif
  };
  int http2_init_result = rpc_http2_server_init_with_metadata(&conn, &metadata);
  if (http2_init_result < 0) {
    LUPINE_LOG_ERROR("Error initializing HTTP/2 connection.");
    rpc_conn_destroy(&conn);
#ifdef LUPINE_BUILD_CUDA_BACKEND
    return lupine_server_checkpoint_child_finish();
#else
    return 0;
#endif
  }
  if (http2_init_result != 0) {
    rpc_conn_destroy(&conn);
#ifdef LUPINE_BUILD_CUDA_BACKEND
    return lupine_server_checkpoint_child_finish();
#else
    return 0;
#endif
  }
#ifdef LUPINE_BUILD_CUDA_BACKEND
  if (!lupine_server_initialize_connection(&conn)) {
    LUPINE_LOG_ERROR("Error initializing per-connection CUDA state.");
    rpc_conn_destroy(&conn);
    return lupine_server_checkpoint_child_finish();
  }
#endif

  LUPINE_LOG_DEBUG("Client connected.");

  std::unordered_map<uint64_t, std::shared_ptr<lupine_lane>> lanes;
  bool connection_ready = false;
  while (!conn.closed) {
    if (pthread_mutex_lock(&conn.read_mutex) != 0) {
      break;
    }
    while (conn.read_id != 0 && !conn.closed) {
      pthread_cond_wait(&conn.read_cond, &conn.read_mutex);
    }
    if (conn.closed) {
      pthread_mutex_unlock(&conn.read_mutex);
      break;
    }

    int request_id = 0;
    if (rpc_read(&conn, &request_id, sizeof(request_id)) !=
            sizeof(request_id) ||
        request_id == 0) {
      pthread_mutex_unlock(&conn.read_mutex);
      LUPINE_LOG_ERROR("RPC dispatch failed; closing client.");
      break;
    }

    conn.read_id = request_id;
    if (request_id % 2 == conn.local_request_parity) {
      if (pthread_cond_broadcast(&conn.read_cond) < 0 ||
          pthread_mutex_unlock(&conn.read_mutex) < 0) {
        break;
      }
      continue;
    }

    if (rpc_read(&conn, &conn.read_lane_id, sizeof(conn.read_lane_id)) !=
            sizeof(conn.read_lane_id) ||
        rpc_read(&conn, &conn.read_op, sizeof(conn.read_op)) !=
            sizeof(conn.read_op)) {
      pthread_mutex_unlock(&conn.read_mutex);
      LUPINE_LOG_ERROR("RPC dispatch failed; closing client.");
      break;
    }
    uint64_t lane_id = conn.read_lane_id;
    int op = conn.read_op;

    if (!connection_ready) {
#ifdef LUPINE_BUILD_CUDA_BACKEND
      if (!lupine_server_checkpoint_connection_ready(
              rpc_http2_session_id(&conn))) {
        pthread_mutex_unlock(&conn.read_mutex);
        LUPINE_LOG_ERROR("Failed to restore connection checkpoint.");
        break;
      }
#endif
      connection_ready = true;
    }

    std::shared_ptr<lupine_lane> lane;
    auto it = lanes.find(lane_id);
    if (it == lanes.end()) {
      if (lanes.size() >= MAX_LANES) {
        pthread_mutex_unlock(&conn.read_mutex);
        LUPINE_LOG_ERROR("Too many active RPC lanes.");
        break;
      }
      lane = std::make_shared<lupine_lane>();
      lane->id = lane_id;
      lane->worker = std::thread([&conn, &handlers, lane]() {
        for (;;) {
          int op = 0;
          {
            std::unique_lock<std::mutex> lock(lane->mutex);
            lane->cond.wait(lock, [&lane]() { return lane->ready; });
            op = lane->op;
            lane->ready = false;
          }

          conn.read_lane_id = lane->id;
          conn.read_op = op;
          if (conn.read_id == -1 || op == LUPINE_RPC_TERMINATE_LANE) {
            rpc_read_end(&conn);
            return;
          }
          if (rpc_server_dispatch(handlers, &conn, op) >= 0) {
            continue;
          }
          rpc_read_end(&conn);
          return;
        }
      });
      lanes.emplace(lane_id, lane);
    } else {
      lane = it->second;
    }

    {
      std::lock_guard<std::mutex> lock(lane->mutex);
      lane->op = op;
      lane->ready = true;
    }
    lane->cond.notify_one();
    if (pthread_cond_broadcast(&conn.read_cond) < 0 ||
        pthread_mutex_unlock(&conn.read_mutex) < 0) {
      break;
    }

    if (op == LUPINE_RPC_TERMINATE_LANE) {
      if (lane->worker.joinable()) {
        lane->worker.join();
      }
      lanes.erase(lane_id);
    }
  }

  for (auto &entry : lanes) {
    auto &lane = entry.second;
    if (pthread_mutex_lock(&conn.read_mutex) != 0) {
      break;
    }
    while (conn.read_id != 0) {
      pthread_cond_wait(&conn.read_cond, &conn.read_mutex);
    }
    conn.read_id = -1;
    {
      std::lock_guard<std::mutex> lock(lane->mutex);
      lane->op = LUPINE_RPC_TERMINATE_LANE;
      lane->ready = true;
    }
    lane->cond.notify_one();
    if (pthread_cond_broadcast(&conn.read_cond) < 0 ||
        pthread_mutex_unlock(&conn.read_mutex) < 0) {
      break;
    }
    if (lane->worker.joinable()) {
      lane->worker.join();
    }
  }

  rpc_close_transport_socket(&conn);
  pthread_cond_broadcast(&conn.read_cond);
  if (conn.rpc_thread != 0) {
    pthread_join(conn.rpc_thread, nullptr);
    conn.rpc_thread = 0;
  }

  int checkpoint_result = 0;
#ifdef LUPINE_BUILD_CUDA_BACKEND
  // Finish checkpointing before releasing per-connection CUDA resources.
  checkpoint_result = lupine_server_checkpoint_child_finish();
  lupine_server_cleanup_connection(&conn);
#endif
  rpc_conn_destroy(&conn);
  return checkpoint_result;
}

int main() {
  if (!rpc_server_validate(lupine_rpc_handlers())) {
    LUPINE_LOG_ERROR("Invalid RPC handler registry.");
    return EXIT_FAILURE;
  }
  int port = DEFAULT_PORT;
  struct sockaddr_in servaddr, cli;
  if (lupine_socket_init() < 0) {
    LUPINE_LOG_ERROR("Socket initialization failed.");
    exit(EXIT_FAILURE);
  }

  lupine_socket_t sockfd = socket(AF_INET, SOCK_STREAM, 0);
  if (sockfd == LUPINE_INVALID_SOCKET) {
    LUPINE_LOG_ERROR("Socket creation failed.");
    exit(EXIT_FAILURE);
  }

  char *p = getenv("LUPINE_PORT");

  if (p == NULL) {
    port = DEFAULT_PORT;
  } else {
    // Validate LUPINE_PORT so a typo (e.g. "14833x" or "") can't silently
    // fall back to atoi's 0, which the kernel would reinterpret as an
    // ephemeral port.
    char *end = nullptr;
    long parsed = strtol(p, &end, 10);
    if (end == p || *end != '\0' || parsed < 1 || parsed > 65535) {
      LUPINE_LOG_ERROR("Invalid LUPINE_PORT '" << p << "'; expected 1-65535.");
      exit(EXIT_FAILURE);
    }
    port = static_cast<int>(parsed);
  }

  // Bind the socket
  memset(&servaddr, 0, sizeof(servaddr));
  servaddr.sin_family = AF_INET;
  servaddr.sin_addr.s_addr = INADDR_ANY;
  servaddr.sin_port = htons(port);

  if (lupine_socket_set_reuseaddr(sockfd) < 0) {
    LUPINE_LOG_ERROR("Socket bind failed.");
    exit(EXIT_FAILURE);
  }

  if (bind(sockfd, (struct sockaddr *)&servaddr, sizeof(servaddr)) != 0) {
    LUPINE_LOG_ERROR("Socket bind failed.");
    exit(EXIT_FAILURE);
  }

  if (listen(sockfd, MAX_CLIENTS) != 0) {
    LUPINE_LOG_ERROR("Listen failed.");
    exit(EXIT_FAILURE);
  }

  LUPINE_LOG_DEBUG("Server listening on port " << port << "...");

#ifndef _WIN32
  if (!lupine_install_parent_signal_handlers()) {
    LUPINE_LOG_ERROR("Failed to install server signal handlers.");
    lupine_socket_close(sockfd);
    exit(EXIT_FAILURE);
  }
  std::unordered_set<pid_t> connection_children;
  // One broker socket per connection child; children park and fetch backend
  // shareable fds through the runtime-neutral parent (see ipc.h).
  std::vector<int> broker_fds;
#endif

  // Server loop
  while (1) {
#ifndef _WIN32
    if (lupine_parent_child_exited != 0) {
      lupine_reap_connection_children(connection_children);
    }
    if (lupine_parent_termination_requested != 0) {
      break;
    }

    // Wait for a new connection or a broker request from a child.
    std::vector<struct pollfd> poll_fds;
    poll_fds.push_back({sockfd, POLLIN, 0});
    for (int broker_fd : broker_fds) {
      poll_fds.push_back({broker_fd, POLLIN, 0});
    }
    if (poll(poll_fds.data(), poll_fds.size(), -1) < 0) {
      if (errno != EINTR) {
        LUPINE_LOG_ERROR("Server poll failed.");
      }
      continue;
    }
    for (size_t i = 1; i < poll_fds.size(); ++i) {
      if ((poll_fds[i].revents & (POLLIN | POLLHUP | POLLERR)) == 0) {
        continue;
      }
      if (lupine_ipc_broker_parent_handle(poll_fds[i].fd) < 0) {
        close(poll_fds[i].fd);
        broker_fds.erase(
            std::remove(broker_fds.begin(), broker_fds.end(), poll_fds[i].fd),
            broker_fds.end());
      }
    }
    if ((poll_fds[0].revents & POLLIN) == 0) {
      continue;
    }
#endif
    socklen_t len = sizeof(cli);
    lupine_socket_t connfd = accept(sockfd, (struct sockaddr *)&cli, &len);

    if (connfd == LUPINE_INVALID_SOCKET) {
#ifndef _WIN32
      if (lupine_parent_child_exited != 0) {
        lupine_reap_connection_children(connection_children);
      }
      if (lupine_parent_termination_requested != 0) {
        break;
      }
      if (errno == EINTR) {
        continue;
      }
#endif
      LUPINE_LOG_ERROR("Server accept failed.");
      continue;
    }

#ifndef _WIN32
    if (lupine_parent_termination_requested != 0) {
      lupine_socket_close(connfd);
      break;
    }
#endif

#ifndef _WIN32
    // TCP_NODELAY keeps small RPC frames latency-low; keepalive (and optional
    // buffer sizing) keeps this long-lived connection from being silently
    // reaped by a NAT/load-balancer/firewall during idle gaps. See
    // lupine_socket_apply_transport_options.
    lupine_socket_apply_transport_options(connfd);
#endif

#ifndef _WIN32
    // Fork a process per connection so each client gets isolated backend
    // runtime state. A client resetting or corrupting its state cannot affect
    // other clients, and everything is released on disconnect. The parent
    // must not initialize accelerator runtimes inherited by forked children.
    fflush(stdout);
    fflush(stderr);

    sigset_t term_mask;
    sigset_t previous_mask;
    sigemptyset(&term_mask);
    sigaddset(&term_mask, SIGTERM);
    if (sigprocmask(SIG_BLOCK, &term_mask, &previous_mask) != 0) {
      LUPINE_LOG_ERROR("Failed to block SIGTERM around server fork.");
      lupine_socket_close(connfd);
      continue;
    }

    int broker_pair[2] = {-1, -1};
    if (socketpair(AF_UNIX, SOCK_SEQPACKET, 0, broker_pair) < 0) {
      (void)sigprocmask(SIG_SETMASK, &previous_mask, nullptr);
      LUPINE_LOG_ERROR("Server broker socketpair failed.");
      lupine_socket_close(connfd);
      continue;
    }

    pid_t pid = fork();
    if (pid < 0) {
      (void)sigprocmask(SIG_SETMASK, &previous_mask, nullptr);
      LUPINE_LOG_ERROR("Server fork failed.");
      close(broker_pair[0]);
      close(broker_pair[1]);
      lupine_socket_close(connfd);
      continue;
    }
    if (pid == 0) {
      struct sigaction child_action = {};
      child_action.sa_handler = SIG_DFL;
      sigemptyset(&child_action.sa_mask);
      (void)sigaction(SIGCHLD, &child_action, nullptr);

      lupine_socket_close(sockfd);
      // Drop inherited parent ends of sibling broker sockets so the parent
      // sees hangups when their owning children exit.
      for (int broker_fd : broker_fds) {
        close(broker_fd);
      }
      close(broker_pair[0]);
      lupine_ipc_set_broker_fd(broker_pair[1]);
#ifdef LUPINE_BUILD_CUDA_BACKEND
      bool child_started = lupine_server_checkpoint_child_start(connfd);
#else
      bool child_started = lupine_install_child_signal_handler(connfd);
#endif
      if (!child_started) {
        LUPINE_LOG_ERROR("Failed to initialize graceful child shutdown.");
        lupine_socket_close(connfd);
        exit(EXIT_FAILURE);
      }
      (void)sigprocmask(SIG_SETMASK, &previous_mask, nullptr);
      int checkpoint_result = client_handler(connfd);
      exit(checkpoint_result == 0 ? EXIT_SUCCESS : EXIT_FAILURE);
    }
    close(broker_pair[1]);
    broker_fds.push_back(broker_pair[0]);
    connection_children.insert(pid);
    (void)sigprocmask(SIG_SETMASK, &previous_mask, nullptr);
    lupine_socket_close(connfd);
#else
    // Windows has no fork; connections share the server process.
    std::thread client_thread(client_handler, connfd);

    // detach the thread so it runs independently
    client_thread.detach();
#endif
  }

  lupine_socket_close(sockfd);

#ifndef _WIN32
  // Every connection owns backend state in a dedicated child, so each child
  // must quiesce and optionally persist itself.
  lupine_reap_connection_children(connection_children);
  for (pid_t child : connection_children) {
    (void)kill(child, SIGTERM);
  }

  int shutdown_result = EXIT_SUCCESS;
  while (!connection_children.empty()) {
    int status = 0;
    pid_t child = waitpid(-1, &status, 0);
    if (child > 0) {
      connection_children.erase(child);
      if (!WIFEXITED(status) || WEXITSTATUS(status) != EXIT_SUCCESS) {
        shutdown_result = EXIT_FAILURE;
      }
      continue;
    }
    if (child < 0 && errno == EINTR) {
      continue;
    }
    if (child < 0 && errno == ECHILD) {
      connection_children.clear();
      break;
    }
    shutdown_result = EXIT_FAILURE;
    break;
  }
  return shutdown_result;
#else
  return 0;
#endif
}
