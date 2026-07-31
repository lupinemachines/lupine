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

#else

#include <cerrno>
#include <cstring>
#include <mutex>
#include <string>
#include <unordered_map>

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/random.h>
#include <sys/socket.h>
#include <unistd.h>

namespace {

constexpr uint64_t LUPINE_IPC_FD_MAGIC = 0x4446454e4950554cULL; // "LUPINEFD"
constexpr uint32_t LUPINE_IPC_FD_VERSION = 1;

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

std::string lupine_ipc_key(uint32_t kind, const lupine_ipc_token &token) {
  std::string key(reinterpret_cast<const char *>(&kind), sizeof(kind));
  key.append(reinterpret_cast<const char *>(token.bytes), sizeof(token.bytes));
  return key;
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
  while (sendmsg(sock, &hdr, 0) < 0) {
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
  if (n != static_cast<ssize_t>(sizeof(*msg))) {
    memset(msg, 0, sizeof(*msg));
    return -1;
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
  if (token == nullptr) {
    return -1;
  }
  size_t done = 0;
  while (done < sizeof(token->bytes)) {
    ssize_t n = getrandom(token->bytes + done, sizeof(token->bytes) - done, 0);
    if (n < 0) {
      if (errno == EINTR) {
        continue;
      }
      return -1;
    }
    done += static_cast<size_t>(n);
  }
  return 0;
}

extern "C" int lupine_ipc_create_proxy_fd(uint32_t kind,
                                          const lupine_ipc_token *token,
                                          int *out_fd) {
  if (token == nullptr || out_fd == nullptr) {
    return -1;
  }
  int fd = memfd_create("lupine-ipc-proxy", 0);
  if (fd < 0) {
    return -1;
  }
  lupine_ipc_fd_payload payload = {};
  payload.magic = LUPINE_IPC_FD_MAGIC;
  payload.version = LUPINE_IPC_FD_VERSION;
  payload.kind = kind;
  payload.token = *token;
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
  if (fd < 0 || kind == nullptr || token == nullptr) {
    return -1;
  }
  lupine_ipc_fd_payload payload = {};
  if (pread(fd, &payload, sizeof(payload), 0) !=
          static_cast<ssize_t>(sizeof(payload)) ||
      payload.magic != LUPINE_IPC_FD_MAGIC ||
      payload.version != LUPINE_IPC_FD_VERSION) {
    return -1;
  }
  *kind = payload.kind;
  *token = payload.token;
  return 0;
}

extern "C" void lupine_ipc_set_broker_fd(int fd) { lupine_ipc_broker_fd = fd; }

extern "C" int lupine_ipc_broker_register_fd(uint32_t kind,
                                             const lupine_ipc_token *token,
                                             int fd) {
  if (token == nullptr || fd < 0) {
    return -1;
  }
  lupine_ipc_broker_msg msg = {};
  msg.cmd = LUPINE_IPC_BROKER_REGISTER;
  msg.kind = kind;
  msg.token = *token;
  lupine_ipc_broker_msg reply;
  int reply_fd = lupine_ipc_broker_exchange(msg, fd, &reply);
  if (reply_fd >= 0) {
    close(reply_fd);
  }
  return reply.status == 0 ? 0 : -1;
}

extern "C" int lupine_ipc_broker_get_fd(uint32_t kind,
                                        const lupine_ipc_token *token) {
  if (token == nullptr) {
    return -1;
  }
  lupine_ipc_broker_msg msg = {};
  msg.cmd = LUPINE_IPC_BROKER_GET;
  msg.kind = kind;
  msg.token = *token;
  lupine_ipc_broker_msg reply;
  int fd = lupine_ipc_broker_exchange(msg, -1, &reply);
  if (reply.status != 0) {
    if (fd >= 0) {
      close(fd);
    }
    return -1;
  }
  return fd;
}

extern "C" int lupine_ipc_broker_parent_handle(int fd) {
  // Shared across every child's broker socket so fds registered on one
  // connection can be fetched from another.
  static std::unordered_map<std::string, int> fds;

  lupine_ipc_broker_msg msg;
  int received_fd = lupine_recv_broker_msg(fd, &msg);
  lupine_ipc_broker_msg reply = {};
  reply.kind = msg.kind;
  reply.token = msg.token;
  std::string key = lupine_ipc_key(msg.kind, msg.token);

  if (msg.cmd == LUPINE_IPC_BROKER_REGISTER) {
    if (received_fd < 0) {
      reply.status = -1;
      return lupine_send_broker_msg(fd, reply, -1);
    }
    auto existing = fds.find(key);
    if (existing != fds.end()) {
      close(existing->second);
    }
    fds[key] = received_fd;
    reply.status = 0;
    return lupine_send_broker_msg(fd, reply, -1);
  }
  if (msg.cmd == LUPINE_IPC_BROKER_GET) {
    auto found = fds.find(key);
    if (found == fds.end()) {
      reply.status = -1;
      return lupine_send_broker_msg(fd, reply, -1);
    }
    reply.status = 0;
    return lupine_send_broker_msg(fd, reply, found->second);
  }
  if (received_fd >= 0) {
    close(received_fd);
  }
  return -1;
}

#endif
