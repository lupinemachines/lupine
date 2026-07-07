#include "ipc.h"

#include <cerrno>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <string>
#include <unordered_map>

#ifndef _WIN32
#include <fcntl.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/syscall.h>
#include <unistd.h>
#endif

namespace {

constexpr uint64_t LUPINE_IPC_FD_MAGIC = 0x4446454e4950554cULL;
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

static thread_local int lupine_ipc_broker_fd = -1;

std::string lupine_ipc_key(uint32_t kind, const lupine_ipc_token &token) {
  std::string key(reinterpret_cast<const char *>(&kind), sizeof(kind));
  key.append(reinterpret_cast<const char *>(token.bytes), sizeof(token.bytes));
  return key;
}

#ifndef _WIN32
ssize_t lupine_full_write(int fd, const void *data, size_t size) {
  const char *ptr = static_cast<const char *>(data);
  size_t done = 0;
  while (done < size) {
    ssize_t n = write(fd, ptr + done, size - done);
    if (n < 0 && errno == EINTR) {
      continue;
    }
    if (n <= 0) {
      return -1;
    }
    done += static_cast<size_t>(n);
  }
  return static_cast<ssize_t>(done);
}

ssize_t lupine_full_read(int fd, void *data, size_t size) {
  char *ptr = static_cast<char *>(data);
  size_t done = 0;
  while (done < size) {
    ssize_t n = read(fd, ptr + done, size - done);
    if (n < 0 && errno == EINTR) {
      continue;
    }
    if (n <= 0) {
      return -1;
    }
    done += static_cast<size_t>(n);
  }
  return static_cast<ssize_t>(done);
}

int lupine_send_broker_msg(int sock, const lupine_ipc_broker_msg &msg,
                           int fd) {
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
    if (errno == EINTR) {
      continue;
    }
    return -1;
  }
  return 0;
}

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
#endif

} // namespace

extern "C" int lupine_ipc_make_token(lupine_ipc_token *token) {
  if (token == nullptr) {
    return -1;
  }
#ifndef _WIN32
#ifdef SYS_getrandom
  size_t done = 0;
  while (done < sizeof(token->bytes)) {
    ssize_t n = syscall(SYS_getrandom, token->bytes + done,
                        sizeof(token->bytes) - done, 0);
    if (n < 0 && errno == EINTR) {
      continue;
    }
    if (n < 0) {
      break;
    }
    done += static_cast<size_t>(n);
  }
  if (done == sizeof(token->bytes)) {
    return 0;
  }
#endif
  int fd = open("/dev/urandom", O_RDONLY | O_CLOEXEC);
  if (fd < 0) {
    return -1;
  }
  int rc = lupine_full_read(fd, token->bytes, sizeof(token->bytes)) ==
                   static_cast<ssize_t>(sizeof(token->bytes))
               ? 0
               : -1;
  close(fd);
  return rc;
#else
  return -1;
#endif
}

extern "C" int lupine_ipc_create_proxy_fd(uint32_t kind,
                                          const lupine_ipc_token *token,
                                          int *out_fd) {
  if (token == nullptr || out_fd == nullptr) {
    return -1;
  }
#ifndef _WIN32
  char path[] = "/tmp/lupine-ipc-fd-XXXXXX";
  int fd = mkstemp(path);
  if (fd < 0) {
    return -1;
  }
  unlink(path);
  lupine_ipc_fd_payload payload = {};
  payload.magic = LUPINE_IPC_FD_MAGIC;
  payload.version = LUPINE_IPC_FD_VERSION;
  payload.kind = kind;
  payload.token = *token;
  if (lupine_full_write(fd, &payload, sizeof(payload)) !=
          static_cast<ssize_t>(sizeof(payload)) ||
      lseek(fd, 0, SEEK_SET) < 0) {
    close(fd);
    return -1;
  }
  *out_fd = fd;
  return 0;
#else
  return -1;
#endif
}

extern "C" int lupine_ipc_read_proxy_fd(int fd, uint32_t *kind,
                                        lupine_ipc_token *token) {
  if (fd < 0 || kind == nullptr || token == nullptr) {
    return -1;
  }
#ifndef _WIN32
  lupine_ipc_fd_payload payload = {};
  if (lseek(fd, 0, SEEK_SET) < 0 ||
      lupine_full_read(fd, &payload, sizeof(payload)) !=
          static_cast<ssize_t>(sizeof(payload)) ||
      payload.magic != LUPINE_IPC_FD_MAGIC ||
      payload.version != LUPINE_IPC_FD_VERSION) {
    return -1;
  }
  *kind = payload.kind;
  *token = payload.token;
  return 0;
#else
  return -1;
#endif
}

extern "C" void lupine_ipc_set_broker_fd(int fd) {
  lupine_ipc_broker_fd = fd;
}

extern "C" int lupine_ipc_broker_register_fd(uint32_t kind,
                                             const lupine_ipc_token *token,
                                             int fd) {
  if (token == nullptr || fd < 0 || lupine_ipc_broker_fd < 0) {
    return -1;
  }
#ifndef _WIN32
  lupine_ipc_broker_msg msg = {};
  msg.cmd = LUPINE_IPC_BROKER_REGISTER;
  msg.kind = kind;
  msg.token = *token;
  if (lupine_send_broker_msg(lupine_ipc_broker_fd, msg, fd) < 0) {
    return -1;
  }
  lupine_ipc_broker_msg reply = {};
  int reply_fd = lupine_recv_broker_msg(lupine_ipc_broker_fd, &reply);
  if (reply_fd >= 0) {
    close(reply_fd);
  }
  return reply.status == 0 ? 0 : -1;
#else
  return -1;
#endif
}

extern "C" int lupine_ipc_broker_get_fd(uint32_t kind,
                                        const lupine_ipc_token *token) {
  if (token == nullptr || lupine_ipc_broker_fd < 0) {
    return -1;
  }
#ifndef _WIN32
  lupine_ipc_broker_msg msg = {};
  msg.cmd = LUPINE_IPC_BROKER_GET;
  msg.kind = kind;
  msg.token = *token;
  if (lupine_send_broker_msg(lupine_ipc_broker_fd, msg, -1) < 0) {
    return -1;
  }
  lupine_ipc_broker_msg reply = {};
  int fd = lupine_recv_broker_msg(lupine_ipc_broker_fd, &reply);
  if (reply.status != 0) {
    if (fd >= 0) {
      close(fd);
    }
    return -1;
  }
  return fd;
#else
  return -1;
#endif
}

extern "C" int lupine_ipc_broker_parent_handle(int fd) {
#ifndef _WIN32
  static std::unordered_map<std::string, int> fds;
  lupine_ipc_broker_msg msg = {};
  int received_fd = lupine_recv_broker_msg(fd, &msg);
  if (received_fd < 0 && msg.cmd != LUPINE_IPC_BROKER_GET) {
    return -1;
  }

  lupine_ipc_broker_msg reply = {};
  reply.kind = msg.kind;
  reply.token = msg.token;
  std::string key = lupine_ipc_key(msg.kind, msg.token);
  if (msg.cmd == LUPINE_IPC_BROKER_REGISTER) {
    if (received_fd < 0) {
      reply.status = -1;
    } else {
      auto existing = fds.find(key);
      if (existing != fds.end()) {
        close(existing->second);
      }
      fds[key] = received_fd;
      received_fd = -1;
      reply.status = 0;
    }
    if (lupine_send_broker_msg(fd, reply, -1) < 0) {
      return -1;
    }
    return 0;
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
#else
  return -1;
#endif
}
