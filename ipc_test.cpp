#include "ipc.h"

#include <algorithm>
#include <cerrno>
#include <cstring>
#include <iostream>
#include <string>
#include <thread>
#include <vector>

#include <dirent.h>
#include <fcntl.h>
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

size_t count_open_fds() {
  DIR *dir = opendir("/proc/self/fd");
  if (dir == nullptr) {
    std::cerr << "FAIL: opendir(/proc/self/fd)\n";
    abort();
  }
  size_t count = 0;
  while (struct dirent *entry = readdir(dir)) {
    if (strcmp(entry->d_name, ".") != 0 && strcmp(entry->d_name, "..") != 0) {
      ++count;
    }
  }
  closedir(dir);
  // Subtract the descriptor opendir() itself held open while scanning.
  return count > 0 ? count - 1 : 0;
}

struct raw_broker_msg {
  uint32_t cmd;
  uint32_t kind;
  int32_t status;
  uint32_t reserved;
  lupine_ipc_token token;
};

int send_raw_broker_msg(int sock, const raw_broker_msg &msg, int fd) {
  struct iovec iov = {const_cast<raw_broker_msg *>(&msg), sizeof(msg)};
  char control[CMSG_SPACE(sizeof(int))] = {};
  struct msghdr hdr = {};
  hdr.msg_iov = &iov;
  hdr.msg_iovlen = 1;
  if (fd >= 0) {
    hdr.msg_control = control;
    hdr.msg_controllen = sizeof(control);
    struct cmsghdr *cmsg = CMSG_FIRSTHDR(&hdr);
    cmsg->cmsg_level = SOL_SOCKET;
    cmsg->cmsg_type = SCM_RIGHTS;
    cmsg->cmsg_len = CMSG_LEN(sizeof(int));
    memcpy(CMSG_DATA(cmsg), &fd, sizeof(fd));
  }
  return sendmsg(sock, &hdr, 0) == static_cast<ssize_t>(sizeof(msg)) ? 0 : -1;
}

int recv_raw_broker_reply(int sock, raw_broker_msg *reply) {
  struct iovec iov = {reply, sizeof(*reply)};
  char control[CMSG_SPACE(sizeof(int))] = {};
  struct msghdr hdr = {};
  hdr.msg_iov = &iov;
  hdr.msg_iovlen = 1;
  hdr.msg_control = control;
  hdr.msg_controllen = sizeof(control);
  if (recvmsg(sock, &hdr, 0) != static_cast<ssize_t>(sizeof(*reply))) {
    return -2;
  }
  int fd = -1;
  for (struct cmsghdr *cmsg = CMSG_FIRSTHDR(&hdr); cmsg != nullptr;
       cmsg = CMSG_NXTHDR(&hdr, cmsg)) {
    if (cmsg->cmsg_level == SOL_SOCKET && cmsg->cmsg_type == SCM_RIGHTS &&
        cmsg->cmsg_len >= CMSG_LEN(sizeof(int))) {
      memcpy(&fd, CMSG_DATA(cmsg), sizeof(fd));
      break;
    }
  }
  return fd;
}

bool test_broker_get_closes_unexpected_fd() {
  broker_fixture broker(1);
  lupine_ipc_set_broker_fd(broker.child_ends[0]);

  lupine_ipc_token known = {};
  lupine_ipc_token missing = {};
  if (lupine_ipc_make_token(&known) != 0 ||
      lupine_ipc_make_token(&missing) != 0) {
    std::cerr << "FAIL: token creation failed\n";
    return false;
  }
  int payload = memfd_with_content("known-payload");
  if (lupine_ipc_broker_register_fd(LUPINE_IPC_FD_KIND_VMM_ALLOCATION, &known,
                                    payload) != 0) {
    close(payload);
    std::cerr << "FAIL: register known token\n";
    return false;
  }
  close(payload);

  const size_t baseline_fds = count_open_fds();
  constexpr uint32_t kBrokerGetCmd = 2;

  // Verify both open-file-description closure (via pipe EOF) and process fd
  // count stability across 64 hit + miss GET calls carrying unexpected fds.
  for (int i = 0; i < 64; ++i) {
    int pipe_fds[2] = {-1, -1};
    if (pipe2(pipe_fds, O_NONBLOCK | O_CLOEXEC) != 0) {
      std::cerr << "FAIL: pipe2\n";
      return false;
    }
    const bool hit = (i % 2) == 0;
    raw_broker_msg req = {kBrokerGetCmd, LUPINE_IPC_FD_KIND_VMM_ALLOCATION, 0,
                          0, hit ? known : missing};
    if (send_raw_broker_msg(broker.child_ends[0], req, pipe_fds[1]) != 0) {
      close(pipe_fds[0]);
      close(pipe_fds[1]);
      std::cerr << "FAIL: send_raw_broker_msg\n";
      return false;
    }
    close(pipe_fds[1]);

    raw_broker_msg reply = {};
    int reply_fd = recv_raw_broker_reply(broker.child_ends[0], &reply);
    if (reply_fd == -2 || (hit && (reply.status != 0 || reply_fd < 0)) ||
        (!hit && (reply.status == 0 || reply_fd >= 0))) {
      if (reply_fd >= 0) {
        close(reply_fd);
      }
      close(pipe_fds[0]);
      std::cerr << "FAIL: unexpected GET reply\n";
      return false;
    }
    if (reply_fd >= 0) {
      if (!fd_has_content(reply_fd, "known-payload")) {
        close(reply_fd);
        close(pipe_fds[0]);
        std::cerr << "FAIL: corrupted GET reply payload\n";
        return false;
      }
      close(reply_fd);
    }

    // Because the parent already processed the GET and replied, its duplicate
    // of pipe_fds[1] must already be closed: reading pipe_fds[0] must return
    // immediate EOF (0), not EAGAIN.
    char ch = 0;
    if (read(pipe_fds[0], &ch, 1) != 0) {
      close(pipe_fds[0]);
      std::cerr << "FAIL: parent leaked unexpected fd attached to GET\n";
      return false;
    }
    close(pipe_fds[0]);
  }

  if (count_open_fds() != baseline_fds) {
    std::cerr << "FAIL: open fd count grew after GET with attached fds ("
              << baseline_fds << " -> " << count_open_fds() << ")\n";
    return false;
  }
  return true;
}

bool test_broker_bounds_parked_fds_with_lru_eviction() {
  constexpr size_t kBrokerCap = 512;
  constexpr size_t kOverflow = 128;
  constexpr size_t kTotal = kBrokerCap + kOverflow;

  broker_fixture broker(1);
  lupine_ipc_set_broker_fd(broker.child_ends[0]);

  const size_t baseline_fds = count_open_fds();
  std::vector<lupine_ipc_token> tokens(kTotal);

  for (size_t i = 0; i < kBrokerCap; ++i) {
    if (lupine_ipc_make_token(&tokens[i]) != 0) {
      std::cerr << "FAIL: token creation failed\n";
      return false;
    }
    std::string label = "lru-" + std::to_string(i);
    int fd = memfd_with_content(label.c_str());
    if (fd < 0 || lupine_ipc_broker_register_fd(
                      LUPINE_IPC_FD_KIND_VMM_ALLOCATION, &tokens[i], fd) != 0) {
      if (fd >= 0) {
        close(fd);
      }
      std::cerr << "FAIL: register token " << i << "\n";
      return false;
    }
    close(fd);
  }

  // Touch token 0 via GET so it becomes MRU and tokens [1..kOverflow] become
  // the oldest entries eligible for eviction.
  int refreshed =
      lupine_ipc_broker_get_fd(LUPINE_IPC_FD_KIND_VMM_ALLOCATION, &tokens[0]);
  if (refreshed < 0 || !fd_has_content(refreshed, "lru-0")) {
    if (refreshed >= 0) {
      close(refreshed);
    }
    std::cerr << "FAIL: refresh GET on token 0\n";
    return false;
  }
  close(refreshed);

  for (size_t i = kBrokerCap; i < kTotal; ++i) {
    if (lupine_ipc_make_token(&tokens[i]) != 0) {
      std::cerr << "FAIL: token creation failed\n";
      return false;
    }
    std::string label = "lru-" + std::to_string(i);
    int fd = memfd_with_content(label.c_str());
    if (fd < 0 || lupine_ipc_broker_register_fd(
                      LUPINE_IPC_FD_KIND_VMM_ALLOCATION, &tokens[i], fd) != 0) {
      if (fd >= 0) {
        close(fd);
      }
      std::cerr << "FAIL: register overflow token " << i << "\n";
      return false;
    }
    close(fd);
  }

  const size_t full_fds = count_open_fds();
  if (full_fds > baseline_fds + kBrokerCap) {
    std::cerr << "FAIL: broker exceeded fd cap (baseline=" << baseline_fds
              << ", current=" << full_fds << ")\n";
    return false;
  }

  // Refreshed token 0 must survive; tokens [1..kOverflow] must be evicted;
  // tokens [kOverflow + 1..kTotal - 1] must all remain redeemable.
  int survivor0 =
      lupine_ipc_broker_get_fd(LUPINE_IPC_FD_KIND_VMM_ALLOCATION, &tokens[0]);
  if (survivor0 < 0 || !fd_has_content(survivor0, "lru-0")) {
    if (survivor0 >= 0) {
      close(survivor0);
    }
    std::cerr << "FAIL: refreshed token 0 was evicted\n";
    return false;
  }
  close(survivor0);

  for (size_t i = 1; i <= kOverflow; ++i) {
    int evicted =
        lupine_ipc_broker_get_fd(LUPINE_IPC_FD_KIND_VMM_ALLOCATION, &tokens[i]);
    if (evicted >= 0) {
      close(evicted);
      std::cerr << "FAIL: LRU token " << i << " was not evicted\n";
      return false;
    }
  }

  for (size_t i = kOverflow + 1; i < kTotal; ++i) {
    std::string expected = "lru-" + std::to_string(i);
    int fetched =
        lupine_ipc_broker_get_fd(LUPINE_IPC_FD_KIND_VMM_ALLOCATION, &tokens[i]);
    if (fetched < 0 || !fd_has_content(fetched, expected.c_str())) {
      if (fetched >= 0) {
        close(fetched);
      }
      std::cerr << "FAIL: recent token " << i << " was not redeemable\n";
      return false;
    }
    close(fetched);
  }

  // Register 200 additional tokens at steady-state capacity and confirm zero
  // further fd growth.
  for (size_t i = 0; i < 200; ++i) {
    lupine_ipc_token extra = {};
    if (lupine_ipc_make_token(&extra) != 0) {
      std::cerr << "FAIL: extra token creation failed\n";
      return false;
    }
    int fd = memfd_with_content("steady");
    if (fd < 0 || lupine_ipc_broker_register_fd(LUPINE_IPC_FD_KIND_MEMORY_POOL,
                                                &extra, fd) != 0) {
      if (fd >= 0) {
        close(fd);
      }
      std::cerr << "FAIL: steady-state register\n";
      return false;
    }
    close(fd);
  }

  if (count_open_fds() != full_fds) {
    std::cerr << "FAIL: steady-state registrations leaked fds (" << full_fds
              << " -> " << count_open_fds() << ")\n";
    return false;
  }
  return true;
}

} // namespace

int main() {
  if (!test_make_token_produces_distinct_tokens() ||
      !test_proxy_fd_round_trip() || !test_proxy_fd_rejects_garbage() ||
      !test_broker_register_and_get() ||
      !test_broker_second_connection_can_fetch() ||
      !test_broker_reregister_overwrites() ||
      !test_broker_without_fd_fails_fast() ||
      !test_parent_handle_reports_hangup() ||
      !test_broker_get_closes_unexpected_fd() ||
      !test_broker_bounds_parked_fds_with_lru_eviction()) {
    return 1;
  }
  std::cout << "ipc broker tests passed\n";
  return 0;
}
