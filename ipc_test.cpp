#include "ipc.h"

#include <algorithm>
#include <cstring>
#include <iostream>
#include <thread>
#include <vector>

#include <poll.h>
#include <sys/mman.h>
#include <sys/socket.h>
#include <unistd.h>

namespace {

bool tokens_equal(const lupine_ipc_token &a, const lupine_ipc_token &b) {
  return memcmp(a.bytes, b.bytes, sizeof(a.bytes)) == 0;
}

int memfd_with_content(const char *content) {
  int fd = memfd_create("ipc-test-payload", 0);
  if (fd < 0) {
    return -1;
  }
  size_t len = strlen(content);
  if (pwrite(fd, content, len, 0) != static_cast<ssize_t>(len)) {
    close(fd);
    return -1;
  }
  return fd;
}

bool fd_has_content(int fd, const char *content) {
  char buffer[64] = {};
  ssize_t n = pread(fd, buffer, sizeof(buffer) - 1, 0);
  return n >= 0 && strcmp(buffer, content) == 0;
}

// Services parent ends of broker socketpairs until every one hangs up,
// mirroring the server parent's poll loop.
void serve_parent(std::vector<int> fds) {
  while (!fds.empty()) {
    std::vector<struct pollfd> poll_fds;
    for (int fd : fds) {
      poll_fds.push_back({fd, POLLIN, 0});
    }
    if (poll(poll_fds.data(), poll_fds.size(), -1) < 0) {
      if (errno == EINTR) {
        continue;
      }
      break;
    }
    for (auto &poll_fd : poll_fds) {
      if ((poll_fd.revents & (POLLIN | POLLHUP | POLLERR)) == 0) {
        continue;
      }
      if (lupine_ipc_broker_parent_handle(poll_fd.fd) < 0) {
        close(poll_fd.fd);
        fds.erase(std::remove(fds.begin(), fds.end(), poll_fd.fd), fds.end());
      }
    }
  }
}

struct broker_fixture {
  std::vector<int> child_ends;
  std::thread parent;

  explicit broker_fixture(size_t sockets) {
    std::vector<int> parent_ends;
    for (size_t i = 0; i < sockets; ++i) {
      int pair[2];
      if (socketpair(AF_UNIX, SOCK_SEQPACKET, 0, pair) < 0) {
        std::cerr << "FAIL: socketpair\n";
        abort();
      }
      parent_ends.push_back(pair[0]);
      child_ends.push_back(pair[1]);
    }
    parent = std::thread(serve_parent, parent_ends);
  }

  ~broker_fixture() {
    lupine_ipc_set_broker_fd(-1);
    for (int fd : child_ends) {
      close(fd);
    }
    parent.join();
  }
};

bool test_make_token_produces_distinct_tokens() {
  lupine_ipc_token first = {};
  lupine_ipc_token second = {};
  if (lupine_ipc_make_token(&first) != 0 ||
      lupine_ipc_make_token(&second) != 0) {
    std::cerr << "FAIL: token creation failed\n";
    return false;
  }
  lupine_ipc_token zero = {};
  if (tokens_equal(first, zero) || tokens_equal(first, second)) {
    std::cerr << "FAIL: tokens are not random\n";
    return false;
  }
  if (lupine_ipc_make_token(nullptr) == 0) {
    std::cerr << "FAIL: null token accepted\n";
    return false;
  }
  return true;
}

bool test_proxy_fd_round_trip() {
  lupine_ipc_token token;
  if (lupine_ipc_make_token(&token) != 0) {
    std::cerr << "FAIL: token creation failed\n";
    return false;
  }
  int fd = -1;
  if (lupine_ipc_create_proxy_fd(LUPINE_IPC_FD_KIND_VMM_ALLOCATION, &token,
                                 &fd) != 0 ||
      fd < 0) {
    std::cerr << "FAIL: proxy fd creation failed\n";
    return false;
  }
  // Reads must not consume the payload: an application can hand the same
  // proxy fd to several importers.
  for (int pass = 0; pass < 2; ++pass) {
    uint32_t kind = 0;
    lupine_ipc_token read_token = {};
    if (lupine_ipc_read_proxy_fd(fd, &kind, &read_token) != 0 ||
        kind != LUPINE_IPC_FD_KIND_VMM_ALLOCATION ||
        !tokens_equal(token, read_token)) {
      std::cerr << "FAIL: proxy fd round trip (pass " << pass << ")\n";
      close(fd);
      return false;
    }
  }
  close(fd);
  return true;
}

bool test_proxy_fd_rejects_garbage() {
  uint32_t kind = 0;
  lupine_ipc_token token = {};
  if (lupine_ipc_read_proxy_fd(-1, &kind, &token) == 0) {
    std::cerr << "FAIL: bad fd accepted\n";
    return false;
  }
  int empty = memfd_with_content("");
  if (lupine_ipc_read_proxy_fd(empty, &kind, &token) == 0) {
    std::cerr << "FAIL: empty payload accepted\n";
    close(empty);
    return false;
  }
  close(empty);
  int garbage = memfd_with_content("not a lupine ipc payload, definitely");
  if (lupine_ipc_read_proxy_fd(garbage, &kind, &token) == 0) {
    std::cerr << "FAIL: garbage payload accepted\n";
    close(garbage);
    return false;
  }
  close(garbage);
  return true;
}

bool test_broker_register_and_get() {
  broker_fixture broker(1);
  lupine_ipc_set_broker_fd(broker.child_ends[0]);

  lupine_ipc_token token;
  if (lupine_ipc_make_token(&token) != 0) {
    std::cerr << "FAIL: token creation failed\n";
    return false;
  }
  int payload = memfd_with_content("vmm-payload");
  if (lupine_ipc_broker_register_fd(LUPINE_IPC_FD_KIND_VMM_ALLOCATION, &token,
                                    payload) != 0) {
    std::cerr << "FAIL: register\n";
    close(payload);
    return false;
  }
  close(payload);

  int fetched =
      lupine_ipc_broker_get_fd(LUPINE_IPC_FD_KIND_VMM_ALLOCATION, &token);
  if (fetched < 0 || !fd_has_content(fetched, "vmm-payload")) {
    std::cerr << "FAIL: get did not return the registered fd\n";
    if (fetched >= 0) {
      close(fetched);
    }
    return false;
  }
  close(fetched);

  // The same fd stays parked in the parent, so it can be fetched again.
  fetched = lupine_ipc_broker_get_fd(LUPINE_IPC_FD_KIND_VMM_ALLOCATION, &token);
  if (fetched < 0 || !fd_has_content(fetched, "vmm-payload")) {
    std::cerr << "FAIL: second get failed\n";
    if (fetched >= 0) {
      close(fetched);
    }
    return false;
  }
  close(fetched);

  lupine_ipc_token unknown;
  if (lupine_ipc_make_token(&unknown) != 0 ||
      lupine_ipc_broker_get_fd(LUPINE_IPC_FD_KIND_VMM_ALLOCATION, &unknown) >=
          0) {
    std::cerr << "FAIL: unknown token was redeemed\n";
    return false;
  }

  // Kinds partition the namespace: a VMM token is not a pool token.
  if (lupine_ipc_broker_get_fd(LUPINE_IPC_FD_KIND_MEMORY_POOL, &token) >= 0) {
    std::cerr << "FAIL: kind mismatch was redeemed\n";
    return false;
  }
  return true;
}

bool test_broker_second_connection_can_fetch() {
  broker_fixture broker(2);

  lupine_ipc_token token;
  if (lupine_ipc_make_token(&token) != 0) {
    std::cerr << "FAIL: token creation failed\n";
    return false;
  }

  // Register through the first "connection child" socket.
  lupine_ipc_set_broker_fd(broker.child_ends[0]);
  int payload = memfd_with_content("cross-child");
  if (lupine_ipc_broker_register_fd(LUPINE_IPC_FD_KIND_MEMORY_POOL, &token,
                                    payload) != 0) {
    std::cerr << "FAIL: register on first socket\n";
    close(payload);
    return false;
  }
  close(payload);

  // Fetch through the second: the parent's fd table is shared across all
  // broker sockets, which is what lets one connection import another
  // connection's export.
  lupine_ipc_set_broker_fd(broker.child_ends[1]);
  int fetched =
      lupine_ipc_broker_get_fd(LUPINE_IPC_FD_KIND_MEMORY_POOL, &token);
  if (fetched < 0 || !fd_has_content(fetched, "cross-child")) {
    std::cerr << "FAIL: fetch on second socket\n";
    if (fetched >= 0) {
      close(fetched);
    }
    return false;
  }
  close(fetched);
  return true;
}

bool test_broker_reregister_overwrites() {
  broker_fixture broker(1);
  lupine_ipc_set_broker_fd(broker.child_ends[0]);

  lupine_ipc_token token;
  if (lupine_ipc_make_token(&token) != 0) {
    std::cerr << "FAIL: token creation failed\n";
    return false;
  }
  int first = memfd_with_content("stale");
  int second = memfd_with_content("fresh");
  bool registered =
      lupine_ipc_broker_register_fd(LUPINE_IPC_FD_KIND_VMM_ALLOCATION, &token,
                                    first) == 0 &&
      lupine_ipc_broker_register_fd(LUPINE_IPC_FD_KIND_VMM_ALLOCATION, &token,
                                    second) == 0;
  close(first);
  close(second);
  if (!registered) {
    std::cerr << "FAIL: re-register\n";
    return false;
  }
  int fetched =
      lupine_ipc_broker_get_fd(LUPINE_IPC_FD_KIND_VMM_ALLOCATION, &token);
  if (fetched < 0 || !fd_has_content(fetched, "fresh")) {
    std::cerr << "FAIL: re-register did not overwrite\n";
    if (fetched >= 0) {
      close(fetched);
    }
    return false;
  }
  close(fetched);
  return true;
}

bool test_broker_without_fd_fails_fast() {
  lupine_ipc_set_broker_fd(-1);
  lupine_ipc_token token;
  if (lupine_ipc_make_token(&token) != 0) {
    std::cerr << "FAIL: token creation failed\n";
    return false;
  }
  int payload = memfd_with_content("orphan");
  bool ok =
      lupine_ipc_broker_register_fd(LUPINE_IPC_FD_KIND_VMM_ALLOCATION, &token,
                                    payload) != 0 &&
      lupine_ipc_broker_get_fd(LUPINE_IPC_FD_KIND_VMM_ALLOCATION, &token) < 0;
  close(payload);
  if (!ok) {
    std::cerr << "FAIL: broker calls succeeded without a broker fd\n";
  }
  return ok;
}

bool test_parent_handle_reports_hangup() {
  int pair[2];
  if (socketpair(AF_UNIX, SOCK_SEQPACKET, 0, pair) < 0) {
    std::cerr << "FAIL: socketpair\n";
    return false;
  }
  close(pair[1]);
  bool ok = lupine_ipc_broker_parent_handle(pair[0]) < 0;
  close(pair[0]);
  if (!ok) {
    std::cerr << "FAIL: hangup not reported\n";
  }
  return ok;
}

} // namespace

int main() {
  if (!test_make_token_produces_distinct_tokens() ||
      !test_proxy_fd_round_trip() || !test_proxy_fd_rejects_garbage() ||
      !test_broker_register_and_get() ||
      !test_broker_second_connection_can_fetch() ||
      !test_broker_reregister_overwrites() ||
      !test_broker_without_fd_fails_fast() ||
      !test_parent_handle_reports_hangup()) {
    return 1;
  }
  std::cout << "ipc broker tests passed\n";
  return 0;
}
