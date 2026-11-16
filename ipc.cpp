#include "ipc.h"

#ifdef _WIN32

extern "C" int lupine_ipc_create_proxy_fd(uint32_t, const lupine_ipc_token *,
                                          int *) {
  return -1;
}
extern "C" int lupine_ipc_read_proxy_fd(int, uint32_t *, lupine_ipc_token *) {
  return -1;
}
extern "C" int lupine_ipc_make_token(lupine_ipc_token *) { return -1; }
extern "C" void lupine_ipc_set_broker_fd(int) {}
extern "C" int lupine_ipc_broker_register_fd(uint32_t, const lupine_ipc_token *,
                                             int) {
  return -1;
}
extern "C" int lupine_ipc_broker_get_fd(uint32_t, const lupine_ipc_token *) {
  return -1;
}
extern "C" int lupine_ipc_broker_parent_handle(int) { return -1; }
extern "C" int lupine_ipc_send_fd(int, int) { return -1; }
extern "C" int lupine_ipc_recv_fd(int) { return -1; }

#else

#include <cerrno>
#include <cstdlib>
#include <cstring>
#include <list>
#include <mutex>
#include <unordered_map>
#include <utility>

#include <sys/mman.h>
#include <sys/random.h>
#include <sys/socket.h>
#include <unistd.h>

namespace {

constexpr uint64_t LUPINE_IPC_FD_MAGIC = 0x4446454e4950554cULL; // "LUPINEFD"
constexpr uint32_t LUPINE_IPC_FD_VERSION = 1;
constexpr size_t kMaxBrokerParkedFds = 512;

struct lupine_ipc_fd_payload {
  uint64_t magic;
  uint32_t version;
  uint32_t kind;
  lupine_ipc_token token;
};

enum lupine_ipc_broker_cmd : uint32_t {
  LUPINE_IPC_BROKER_REGISTER = 1,
  LUPINE_IPC_BROKER_GET = 2,
};

struct lupine_ipc_broker_msg {
  uint32_t cmd;
  uint32_t kind;
  int32_t status;
  uint32_t reserved;
  lupine_ipc_token token;
};

// One broker socket per forked server child; RPC lanes share it, so requests
// serialize under a mutex to keep each request/reply exchange atomic.
int lupine_ipc_broker_fd = -1;
std::mutex lupine_ipc_broker_mutex;

struct lupine_ipc_key {
  uint32_t kind;
  lupine_ipc_token token;

  bool operator==(const lupine_ipc_key &other) const {
    return kind == other.kind && std::memcmp(token.bytes, other.token.bytes,
                                             sizeof(token.bytes)) == 0;
  }
};

struct lupine_ipc_key_hash {
  size_t operator()(const lupine_ipc_key &key) const noexcept {
    size_t h = 0;
    std::memcpy(&h, key.token.bytes, sizeof(h));
    return h ^ (static_cast<size_t>(key.kind) * 0x9e3779b9U);
  }
};

int lupine_temporary_proxy_fd() {
#if defined(__linux__)
  return memfd_create("lupine-ipc-proxy", 0);
#else
  char path[] = "/tmp/lupine-ipc-proxy.XXXXXX";
  int fd = mkstemp(path);
  if (fd >= 0) {
    unlink(path);
  }
  return fd;
#endif
}

int lupine_send_broker_msg(int sock, const lupine_ipc_broker_msg &msg, int fd) {
  struct iovec iov = {const_cast<lupine_ipc_broker_msg *>(&msg), sizeof(msg)};
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
#ifdef MSG_NOSIGNAL
  constexpr int kSendFlags = MSG_NOSIGNAL;
#else
  constexpr int kSendFlags = 0;
#endif
  while (sendmsg(sock, &hdr, kSendFlags) < 0) {
    if (errno != EINTR) {
      return -1;
    }
  }
  return 0;
}

// Receives one broker message; returns the attached fd or -1.
int lupine_recv_broker_msg(int sock, lupine_ipc_broker_msg *msg) {
  struct iovec iov = {msg, sizeof(*msg)};
  char control[CMSG_SPACE(sizeof(int))] = {};
  struct msghdr hdr = {};
  hdr.msg_iov = &iov;
  hdr.msg_iovlen = 1;
  hdr.msg_control = control;
  hdr.msg_controllen = sizeof(control);
  ssize_t n;
  do {
    n = recvmsg(sock, &hdr, 0);
  } while (n < 0 && errno == EINTR);
  // The broker's peers attach at most one descriptor to a message.
  int fd = -1;
  struct cmsghdr *cmsg = n > 0 ? CMSG_FIRSTHDR(&hdr) : nullptr;
  if (cmsg != nullptr && cmsg->cmsg_type == SCM_RIGHTS) {
    memcpy(&fd, CMSG_DATA(cmsg), sizeof(fd));
  }
  if (n != static_cast<ssize_t>(sizeof(*msg))) {
    memset(msg, 0, sizeof(*msg));
    return -1;
  }
  return fd;
}

// Sends one request and waits for the parent's reply; returns the reply fd
// or -1, with the reply message in *reply.
int lupine_ipc_broker_exchange(const lupine_ipc_broker_msg &msg, int fd,
                               lupine_ipc_broker_msg *reply) {
  std::lock_guard<std::mutex> lock(lupine_ipc_broker_mutex);
  if (lupine_ipc_broker_fd < 0 ||
      lupine_send_broker_msg(lupine_ipc_broker_fd, msg, fd) < 0) {
    memset(reply, 0, sizeof(*reply));
    reply->status = -1;
    return -1;
  }
  return lupine_recv_broker_msg(lupine_ipc_broker_fd, reply);
}

} // namespace

extern "C" int lupine_ipc_make_token(lupine_ipc_token *token) {
  return getentropy(token->bytes, sizeof(token->bytes));
}

extern "C" int lupine_ipc_create_proxy_fd(uint32_t kind,
                                          const lupine_ipc_token *token,
                                          int *out_fd) {
  int fd = lupine_temporary_proxy_fd();
  if (fd < 0)
    return -1;
  lupine_ipc_fd_payload payload = {LUPINE_IPC_FD_MAGIC, LUPINE_IPC_FD_VERSION,
                                   kind, *token};
  if (pwrite(fd, &payload, sizeof(payload), 0) !=
      static_cast<ssize_t>(sizeof(payload))) {
    close(fd);
    return -1;
  }
  *out_fd = fd;
  return 0;
}

extern "C" int lupine_ipc_read_proxy_fd(int fd, uint32_t *kind,
                                        lupine_ipc_token *token) {
  lupine_ipc_fd_payload payload = {};
  if (pread(fd, &payload, sizeof(payload), 0) !=
          static_cast<ssize_t>(sizeof(payload)) ||
      payload.magic != LUPINE_IPC_FD_MAGIC ||
      payload.version != LUPINE_IPC_FD_VERSION)
    return -1;
  *kind = payload.kind;
  *token = payload.token;
  return 0;
}

extern "C" void lupine_ipc_set_broker_fd(int fd) { lupine_ipc_broker_fd = fd; }

extern "C" int lupine_ipc_broker_register_fd(uint32_t kind,
                                             const lupine_ipc_token *token,
                                             int fd) {
  lupine_ipc_broker_msg msg = {LUPINE_IPC_BROKER_REGISTER, kind, 0, 0, *token};
  lupine_ipc_broker_msg reply;
  int reply_fd = lupine_ipc_broker_exchange(msg, fd, &reply);
  if (reply_fd >= 0)
    close(reply_fd);
  return reply.status == 0 ? 0 : -1;
}

extern "C" int lupine_ipc_broker_get_fd(uint32_t kind,
                                        const lupine_ipc_token *token) {
  lupine_ipc_broker_msg msg = {LUPINE_IPC_BROKER_GET, kind, 0, 0, *token};
  lupine_ipc_broker_msg reply;
  int fd = lupine_ipc_broker_exchange(msg, -1, &reply);
  if (reply.status != 0) {
    if (fd >= 0)
      close(fd);
    return -1;
  }
  return fd;
}

extern "C" int lupine_ipc_send_fd(int sock, int fd) {
  return lupine_send_broker_msg(sock, {}, fd);
}

extern "C" int lupine_ipc_recv_fd(int sock) {
  lupine_ipc_broker_msg msg;
  return lupine_recv_broker_msg(sock, &msg);
}

extern "C" int lupine_ipc_broker_parent_handle(int fd) {
  // Shared across every child's broker socket so fds registered on one
  // connection can be fetched from another, bounded with O(1) LRU eviction so
  // long-running servers do not exhaust RLIMIT_NOFILE or pin DMA-BUFs forever.
  using broker_lru_list = std::list<std::pair<lupine_ipc_key, int>>;
  static broker_lru_list lru;
  static std::unordered_map<lupine_ipc_key, broker_lru_list::iterator,
                            lupine_ipc_key_hash>
      fds;
  if (fds.empty()) {
    fds.reserve(kMaxBrokerParkedFds);
  }

  lupine_ipc_broker_msg msg = {};
  int received_fd = lupine_recv_broker_msg(fd, &msg);
  if (msg.cmd != LUPINE_IPC_BROKER_REGISTER && received_fd >= 0) {
    close(received_fd);
    received_fd = -1;
  }
  lupine_ipc_broker_msg reply = {0, msg.kind, 0, 0, msg.token};
  lupine_ipc_key key{msg.kind, msg.token};

  if (msg.cmd == LUPINE_IPC_BROKER_REGISTER) {
    if (received_fd < 0) {
      reply.status = -1;
    } else {
      auto existing = fds.find(key);
      if (existing != fds.end()) {
        if (existing->second->second != received_fd) {
          close(existing->second->second);
        }
        existing->second->second = received_fd;
        if (existing->second != lru.begin()) {
          lru.splice(lru.begin(), lru, existing->second);
        }
      } else if (fds.size() >= kMaxBrokerParkedFds) {
        auto victim = std::prev(lru.end());
        if (victim->second != received_fd) {
          close(victim->second);
        }
        fds.erase(victim->first);
        victim->first = key;
        victim->second = received_fd;
        lru.splice(lru.begin(), lru, victim);
        fds.emplace(key, victim);
      } else {
        lru.emplace_front(key, received_fd);
        fds.emplace(key, lru.begin());
      }
    }
    return lupine_send_broker_msg(fd, reply, -1);
  }
  if (msg.cmd == LUPINE_IPC_BROKER_GET) {
    auto found = fds.find(key);
    if (found == fds.end()) {
      reply.status = -1;
      return lupine_send_broker_msg(fd, reply, -1);
    }
    if (found->second != lru.begin()) {
      lru.splice(lru.begin(), lru, found->second);
    }
    return lupine_send_broker_msg(fd, reply, found->second->second);
  }
  return -1;
}

#endif
