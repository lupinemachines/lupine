#include "address_space.h"
#include "rpc.h"

#include <cerrno>
#include <cstdlib>
#include <iostream>

#include <sys/mman.h>
#include <sys/socket.h>
#include <sys/wait.h>
#include <unistd.h>

namespace {

void require(bool condition, const char *message) {
  if (!condition) {
    std::cerr << message << std::endl;
    std::exit(1);
  }
}

void wait_success(pid_t child) {
  int status = 0;
  require(waitpid(child, &status, 0) == child, "waitpid failed");
  require(WIFEXITED(status) && WEXITSTATUS(status) == 0, "server child failed");
}

void test_identity_va_handshake() {
  int sockets[2] = {-1, -1};
  require(socketpair(AF_UNIX, SOCK_STREAM, 0, sockets) == 0,
          "socketpair failed");
  pid_t child = fork();
  require(child >= 0, "fork failed");
  if (child == 0) {
    close(sockets[0]);
    conn_t server = {};
    require(rpc_conn_init(&server, sockets[1], 1) == 0,
            "server conn init failed");
    require(rpc_http2_server_init(&server) == 0, "server handshake failed");
    require(server.va_size == LUPINE_VA_ARENA_SIZE,
            "server did not reserve negotiated arena");

#ifdef MAP_FIXED_NOREPLACE
    const size_t page_size = static_cast<size_t>(sysconf(_SC_PAGESIZE));
    errno = 0;
    void *collision =
        mmap(reinterpret_cast<void *>(server.va_base), page_size, PROT_NONE,
             MAP_PRIVATE | MAP_ANONYMOUS | MAP_FIXED_NOREPLACE, -1, 0);
    require(collision == MAP_FAILED && errno == EEXIST,
            "server arena was not retained by the kernel");

    uintptr_t replacement_address = server.va_base + page_size;
    void *replacement = mmap(reinterpret_cast<void *>(replacement_address),
                             page_size, PROT_READ | PROT_WRITE,
                             MAP_PRIVATE | MAP_ANONYMOUS | MAP_FIXED, -1, 0);
    require(replacement == reinterpret_cast<void *>(replacement_address),
            "failed to replace an owned arena page");
    *static_cast<unsigned char *>(replacement) = 42;

    errno = 0;
    collision = mmap(replacement, page_size, PROT_NONE,
                     MAP_PRIVATE | MAP_ANONYMOUS | MAP_FIXED_NOREPLACE, -1, 0);
    require(collision == MAP_FAILED && errno == EEXIST,
            "replacement page was not retained by the kernel");
    munmap(replacement, page_size);
#endif

    int32_t lane = rpc_http2_accept_stream(&server);
    char request = '\0';
    require(lane > 0 &&
                rpc_http2_read_stream(&server, lane, &request, 1) == 1 &&
                request == 'x',
            "server rejected a post-negotiation lane");
    const char response = 'y';
    std::vector<rpc_write_cursor> response_cursor = {
        rpc_write_cursor::plain(&response, sizeof(response))};
    require(rpc_http2_write_stream(&server, lane, response_cursor) == 0,
            "server lane response failed");
    rpc_conn_destroy(&server);
    std::_Exit(0);
  }

  close(sockets[1]);
  conn_t client = {};
  require(rpc_conn_init(&client, sockets[0], 0) == 0,
          "client conn init failed");
  unsigned int slot = 0;
  require(lupine_va_reserve_client(&client, 0, &slot) == 0,
          "client arena reservation failed");
  require(rpc_http2_client_init(&client) == 0, "client handshake failed");
  require(client.w_offset == LUPINE_VA_WRITE_OFFSET,
          "client did not select identity R and aliased W mappings");
  int32_t lane = rpc_http2_lane_stream(&client, 42);
  const char request = 'x';
  std::vector<rpc_write_cursor> request_cursor = {
      rpc_write_cursor::plain(&request, sizeof(request))};
  require(lane > 0 &&
              rpc_http2_write_stream(&client, lane, request_cursor) == 0,
          "client post-negotiation lane write failed");
  char response = '\0';
  require(rpc_http2_read_stream(&client, lane, &response, 1) == 1 &&
              response == 'y',
          "client post-negotiation lane response failed");
  rpc_conn_destroy(&client);
  wait_success(child);
}

void test_identity_va_conflict() {
  int sockets[2] = {-1, -1};
  int proposed[2] = {-1, -1};
  require(socketpair(AF_UNIX, SOCK_STREAM, 0, sockets) == 0 &&
              pipe(proposed) == 0,
          "conflict test setup failed");
  pid_t child = fork();
  require(child >= 0, "conflict fork failed");
  if (child == 0) {
    close(sockets[0]);
    close(proposed[1]);
    uintptr_t base = 0;
    require(read(proposed[0], &base, sizeof(base)) == sizeof(base),
            "failed to read proposed base");
    close(proposed[0]);
    const size_t page_size = static_cast<size_t>(sysconf(_SC_PAGESIZE));
    void *collision =
        mmap(reinterpret_cast<void *>(base), page_size, PROT_NONE,
             MAP_PRIVATE | MAP_ANONYMOUS | MAP_FIXED_NOREPLACE, -1, 0);
    require(collision == reinterpret_cast<void *>(base),
            "failed to install server VA collision");
    conn_t server = {};
    require(rpc_conn_init(&server, sockets[1], 1) == 0,
            "conflict server conn init failed");
    require(rpc_http2_server_init(&server) > 0,
            "server accepted a conflicting VA arena");
    rpc_conn_destroy(&server);
    munmap(collision, page_size);
    std::_Exit(0);
  }

  close(sockets[1]);
  close(proposed[0]);
  conn_t client = {};
  require(rpc_conn_init(&client, sockets[0], 0) == 0,
          "conflict client conn init failed");
  unsigned int slot = 0;
  require(lupine_va_reserve_client(&client, 0, &slot) == 0,
          "conflict client arena reservation failed");
  require(write(proposed[1], &client.va_base, sizeof(client.va_base)) ==
              sizeof(client.va_base),
          "failed to send proposed base");
  close(proposed[1]);
  require(rpc_http2_client_init(&client) == LUPINE_RPC_HTTP2_VA_CONFLICT,
          "client did not surface the VA conflict");
  rpc_conn_destroy(&client);
  wait_success(child);
}

} // namespace

void test_address_space() {
#if !defined(__SANITIZE_ADDRESS__) && !defined(__SANITIZE_THREAD__)
  test_identity_va_handshake();
  test_identity_va_conflict();
#endif
}
