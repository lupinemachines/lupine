#ifndef LUPINE_IPC_H
#define LUPINE_IPC_H

#include <cstdint>

// CUDA exports POSIX-fd shareable handles (VMM allocations, memory pools)
// that are only meaningful on the server machine. The client instead hands
// the application a local proxy fd carrying a random token; the server parent
// process brokers the real fds between its forked connection children so an
// importing client can redeem the token from any connection.

enum lupine_ipc_fd_kind : uint32_t {
  LUPINE_IPC_FD_KIND_VMM_ALLOCATION = 1,
  LUPINE_IPC_FD_KIND_MEMORY_POOL = 2,
};

struct lupine_ipc_token {
  unsigned char bytes[16];
};

// Client side: proxy fds handed to the application.
extern "C" int lupine_ipc_create_proxy_fd(uint32_t kind,
                                          const lupine_ipc_token *token,
                                          int *out_fd);
extern "C" int lupine_ipc_read_proxy_fd(int fd, uint32_t *kind,
                                        lupine_ipc_token *token);

// Server side: children register/fetch real fds through the parent broker.
extern "C" int lupine_ipc_make_token(lupine_ipc_token *token);
extern "C" void lupine_ipc_set_broker_fd(int fd);
extern "C" int lupine_ipc_broker_register_fd(uint32_t kind,
                                             const lupine_ipc_token *token,
                                             int fd);
extern "C" int lupine_ipc_broker_get_fd(uint32_t kind,
                                        const lupine_ipc_token *token);
// Parent side: service one request on a child's broker socket; returns -1
// when the socket is dead and should be dropped.
extern "C" int lupine_ipc_broker_parent_handle(int fd);

#endif
