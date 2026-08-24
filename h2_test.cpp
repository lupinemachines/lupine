#include "lupine_log.h"
#include "rpc.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cerrno>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <iostream>
#include <netinet/in.h>
#include <nghttp2/nghttp2.h>
#include <poll.h>
#include <string>
#include <sys/mman.h>
#include <sys/socket.h>
#include <thread>
#include <type_traits>
#include <unistd.h>
#include <utility>
#include <vector>

namespace {

template <typename T, typename = void>
struct has_owned_member : std::false_type {};
template <typename T>
struct has_owned_member<T, std::void_t<decltype(std::declval<T>().owned)>>
    : std::true_type {};

static_assert(!has_owned_member<rpc_write_cursor>::value,
              "RPC write cursors must not own their bytes");

struct h2_pair {
  conn_t client = {};
  conn_t server = {};

  h2_pair() {
    client.connfd = -1;
    server.connfd = -1;
  }

  ~h2_pair() {
    rpc_conn_destroy(&client);
    rpc_conn_destroy(&server);
    if (client.connfd >= 0) {
      close(client.connfd);
    }
    if (server.connfd >= 0) {
      close(server.connfd);
    }
  }
};

void require(bool condition, const char *message) {
  if (!condition) {
    LUPINE_LOG_ERROR(message);
    std::exit(1);
  }
}

void init_pair_sockets(h2_pair *pair);
void exchange_settings(h2_pair *pair);

h2_pair make_pair() {
  h2_pair pair;
  init_pair_sockets(&pair);
  require(rpc_http2_client_init(&pair.client) == 0, "client h2 init failed");
  rpc_http2_client_start_heartbeat(&pair.client);
  require(rpc_http2_server_init(&pair.server) == 0, "server h2 init failed");
  return pair;
}

void init_pair_sockets(h2_pair *pair) {
  int fds[2] = {-1, -1};
  require(socketpair(AF_UNIX, SOCK_STREAM, 0, fds) == 0, "socketpair failed");
  require(rpc_conn_init(&pair->client, fds[0], 0) == 0,
          "client RPC init failed");
  require(rpc_conn_init(&pair->server, fds[1], 1) == 0,
          "server RPC init failed");
}

void write_all(conn_t *conn, const std::vector<std::string> &chunks) {
  std::vector<rpc_write_cursor> cursors;
  cursors.reserve(chunks.size());
  for (const std::string &chunk : chunks) {
    cursors.push_back(rpc_write_cursor::plain(chunk.data(), chunk.size()));
  }
  require(rpc_http2_write(conn, cursors) == 0, "h2 write failed");
}

int write_bytes(conn_t *conn, const void *data, size_t size) {
  std::vector<rpc_write_cursor> cursors = {rpc_write_cursor::plain(data, size)};
  return rpc_http2_write(conn, cursors);
}

int write_stream_bytes(conn_t *conn, int32_t stream_id, const void *data,
                       size_t size) {
  std::vector<rpc_write_cursor> cursors = {rpc_write_cursor::plain(data, size)};
  return rpc_http2_write_stream(conn, stream_id, cursors);
}

std::string read_string(conn_t *conn, size_t size) {
  std::string output(size, '\0');
  require(rpc_http2_read(conn, output.data(), output.size()) ==
              static_cast<int>(output.size()),
          "h2 read failed");
  return output;
}

rpc_http2_read_stats read_stats(conn_t *conn) {
  rpc_http2_read_stats stats = {};
  require(rpc_http2_get_read_stats(conn, &stats) == 0, "read stats failed");
  return stats;
}

bool raw_write_all(lupine_socket_t socket, const unsigned char *data,
                   size_t size) {
  while (size != 0) {
    struct iovec iov = {const_cast<unsigned char *>(data), size};
    ssize_t written = lupine_socket_sendv(socket, &iov, 1);
    if (written < 0 && lupine_socket_error_is_intr()) {
      continue;
    }
    if (written <= 0) {
      return false;
    }
    data += written;
    size -= static_cast<size_t>(written);
  }
  return true;
}

bool raw_read_exact(lupine_socket_t socket, unsigned char *data, size_t size) {
  while (size != 0) {
    ssize_t received = lupine_socket_recv(socket, data, size);
    if (received < 0 && lupine_socket_error_is_intr()) {
      continue;
    }
    if (received <= 0) {
      return false;
    }
    data += received;
    size -= static_cast<size_t>(received);
  }
  return true;
}

bool raw_read_frame(lupine_socket_t socket,
                    std::array<unsigned char, 9> *header) {
  if (!raw_read_exact(socket, header->data(), header->size())) {
    return false;
  }
  size_t size = (static_cast<size_t>((*header)[0]) << 16) |
                (static_cast<size_t>((*header)[1]) << 8) | (*header)[2];
  std::vector<unsigned char> payload(size);
  return raw_read_exact(socket, payload.data(), payload.size());
}

void init_raw_server_peer(h2_pair *pair) {
  init_pair_sockets(pair);
  require(rpc_http2_client_init(&pair->client) == 0, "client h2 init failed");
  constexpr char preface[] = "PRI * HTTP/2.0\r\n\r\nSM\r\n\r\n";
  std::array<unsigned char, sizeof(preface) - 1> received = {};
  require(
      raw_read_exact(pair->server.connfd, received.data(), received.size()) &&
          memcmp(received.data(), preface, received.size()) == 0,
      "client HTTP/2 preface missing");
}

void test_response_wait_sends_transport_heartbeat() {
  h2_pair pair;
  init_raw_server_peer(&pair);
  rpc_http2_client_start_heartbeat(&pair.client);

  rpc_http2_response_wait_begin(&pair.client);
  bool received_ping = false;
  for (int frame_count = 0; frame_count < 8 && !received_ping; ++frame_count) {
    pollfd descriptor = {pair.server.connfd, POLLIN, 0};
    require(poll(&descriptor, 1, 1000) > 0,
            "response wait did not emit an HTTP/2 frame");
    std::array<unsigned char, 9> header = {};
    require(raw_read_frame(pair.server.connfd, &header),
            "heartbeat frame read failed");
    received_ping =
        header[3] == NGHTTP2_PING && (header[4] & NGHTTP2_FLAG_ACK) == 0;
  }
  rpc_http2_response_wait_end(&pair.client);
  require(received_ping, "response wait did not emit an HTTP/2 PING heartbeat");
}

void test_client_to_server() {
  h2_pair pair = make_pair();
  std::string message = "hello over h2";
  std::string received;
  std::thread reader(
      [&] { received = read_string(&pair.server, message.size()); });
  write_all(&pair.client, {message});
  reader.join();
  require(received == message, "client-to-server payload mismatch");
}

void test_server_receives_session_id() {
  const char *original = getenv("LUPINE_SESSION");
  bool had_original = original != nullptr;
  std::string saved = original == nullptr ? "" : original;
  setenv("LUPINE_SESSION", "lease-123", 1);

  {
    h2_pair pair = make_pair();
    write_all(&pair.client, {"x"});
    require(read_string(&pair.server, 1) == "x",
            "server did not receive session test payload");
    const char *session_id = rpc_http2_session_id(&pair.server);
    require(session_id != nullptr && std::string(session_id) == "lease-123",
            "server did not retain x-lupine-session");
  }

  if (had_original) {
    setenv("LUPINE_SESSION", saved.c_str(), 1);
  } else {
    unsetenv("LUPINE_SESSION");
  }
}

void test_server_to_client_after_request_headers() {
  h2_pair pair = make_pair();
  std::string request = "request";
  std::string response = "response";
  std::string received_request;
  std::thread reader(
      [&] { received_request = read_string(&pair.server, request.size()); });
  write_all(&pair.client, {request});
  reader.join();
  require(received_request == request, "server did not receive request");

  write_all(&pair.server, {response});
  require(read_string(&pair.client, response.size()) == response,
          "server-to-client payload mismatch");
}

void test_head_probe_cuda_version_metadata(const char *expected_cuda_version) {
  h2_pair pair;
  init_pair_sockets(&pair);

  const char *cuda_version = nullptr;
  std::thread probe(
      [&] { cuda_version = rpc_http2_client_probe(&pair.client); });
  int server_result = 0;
  if (expected_cuda_version != nullptr) {
    const rpc_http2_server_metadata metadata = {expected_cuda_version};
    server_result =
        rpc_http2_server_init_with_metadata(&pair.server, &metadata);
  } else {
    server_result = rpc_http2_server_init(&pair.server);
  }
  probe.join();

  require(server_result == 1, "HEAD / was not handled as a metadata request");
  if (expected_cuda_version != nullptr) {
    require(cuda_version != nullptr &&
                std::string(cuda_version) == expected_cuda_version,
            "HEAD / omitted CUDA version");
  } else {
    require(cuda_version == nullptr,
            "HEAD / advertised an unknown CUDA version");
  }
}

// The preflight refuses a peer only when both sides state an identity and they
// differ; an unstated identity must stay connectable so builds without git and
// servers predating the header are not locked out.
void test_wire_identity_compatibility_rule() {
  require(lupine_wire_identity_compatible("abc", "abc"),
          "matching identities were rejected");
  require(!lupine_wire_identity_compatible("abc", "def"),
          "mismatched identities were accepted");
  require(lupine_wire_identity_compatible("", "def"),
          "unstated local identity was treated as a mismatch");
  require(lupine_wire_identity_compatible("abc", ""),
          "unstated peer identity was treated as a mismatch");
  require(lupine_wire_identity_compatible(nullptr, nullptr),
          "null identities were treated as a mismatch");
}

// The identity rides the response on the session's own connection, so the
// build check costs a round trip rather than a second dial.
void test_client_await_ready_reports_wire_identity() {
  h2_pair pair;
  init_pair_sockets(&pair);

  std::string peer;
  int ready = -1;
  std::thread client([&] {
    require(rpc_http2_client_init(&pair.client) == 0, "client h2 init failed");
    ready = rpc_http2_client_await_ready(&pair.client);
    peer = rpc_http2_peer_wire_identity(&pair.client);
  });
  require(rpc_http2_server_init(&pair.server) == 0, "server h2 init failed");
  client.join();

  require(ready == 0, "matching builds were not accepted");
  require(peer == lupine_wire_identity(),
          "response did not carry this build's wire identity");
}

// Arena bookkeeping is pure arithmetic over the conn fields, so it is checked
// without reserving anything: the sanitizers own disjoint VA bands and neither
// can host a real reservation the other can.
void test_va_claim_bumps_within_arena() {
  conn_t conn = {};
  conn.va_base = 0x2000000000;
  conn.va_size = 0x1000;
  conn.va_next = conn.va_base;
  constexpr size_t kAlign = 0x200000;

  // The base is already aligned, so the first span fits even in a tiny arena;
  // the next one has to skip a full alignment stride and no longer does.
  uintptr_t claimed = 0;
  require(lupine_va_claim(&conn, 0x100, kAlign, &claimed) &&
              claimed == conn.va_base,
          "aligned base rejected a span that fits");
  uintptr_t beyond = 0;
  require(!lupine_va_claim(&conn, 0x100, kAlign, &beyond),
          "claim aligned past the end of the arena");
  uintptr_t cursor = conn.va_next;
  require(!lupine_va_claim(&conn, 0x100, kAlign, &beyond) &&
              conn.va_next == cursor,
          "a rejected claim advanced the arena cursor");

  conn.va_base = 0x2000000000;
  conn.va_size = 0x800000;
  conn.va_next = conn.va_base;
  require(lupine_va_claim(&conn, 0x100, kAlign, &claimed) &&
              claimed == conn.va_base,
          "first claim did not start at the arena base");
  uintptr_t second = 0;
  require(lupine_va_claim(&conn, 0x100, kAlign, &second) &&
              second == conn.va_base + kAlign,
          "second claim did not advance to the next aligned span");
  require(second - claimed >= 0x100, "claims overlapped");

  require(!lupine_va_claim(&conn, conn.va_size, kAlign, &claimed),
          "claim exceeded the arena bounds");
  require(!lupine_va_claim(&conn, 0, kAlign, &claimed),
          "zero-sized claim was accepted");
  require(!lupine_va_claim(&conn, 0x100, 0x300000, &claimed),
          "non-power-of-two alignment was accepted");
  conn.va_size = 0;
  require(!lupine_va_claim(&conn, 0x100, kAlign, &claimed),
          "claim succeeded without an arena");
}

// Concurrent claims must hand out disjoint spans; the cursor is advanced with a
// compare-exchange rather than a lock.
void test_va_claim_is_disjoint_under_contention() {
  conn_t conn = {};
  conn.va_base = 0x2000000000;
  conn.va_size = 0x4000000;
  conn.va_next = conn.va_base;
  constexpr size_t kAlign = 0x1000;
  constexpr size_t kPerThread = 512;

  // Release the workers together and give them enough iterations that a lost
  // update shows up; a handful of staggered claims never collide.
  std::array<std::vector<uintptr_t>, 8> claims;
  std::atomic<unsigned> ready{0};
  std::atomic<bool> go{false};
  std::vector<std::thread> workers;
  for (auto &bucket : claims) {
    bucket.reserve(kPerThread);
    workers.emplace_back([&bucket, &conn, &ready, &go] {
      ready.fetch_add(1, std::memory_order_release);
      while (!go.load(std::memory_order_acquire)) {
      }
      for (size_t i = 0; i < kPerThread; ++i) {
        uintptr_t claimed = 0;
        if (lupine_va_claim(&conn, kAlign, kAlign, &claimed)) {
          bucket.push_back(claimed);
        }
      }
    });
  }
  while (ready.load(std::memory_order_acquire) < claims.size()) {
  }
  go.store(true, std::memory_order_release);
  for (std::thread &worker : workers) {
    worker.join();
  }
  std::vector<uintptr_t> all;
  for (const auto &bucket : claims) {
    all.insert(all.end(), bucket.begin(), bucket.end());
  }
  require(all.size() == claims.size() * kPerThread,
          "a concurrent claim failed");
  std::sort(all.begin(), all.end());
  require(std::adjacent_find(all.begin(), all.end()) == all.end(),
          "concurrent claims returned the same span twice");
}

// A server that hosts no arena has to be distinguishable from one whose slot is
// merely taken, or the client burns every slot and then fails the connection.
void test_va_request_reports_unsupported_platform() {
  require(setenv("LUPINE_IDENTITY_VA", "0", 1) == 0, "env override failed");
  require(!lupine_va_identity_supported(),
          "identity VA stayed enabled under the override");

  h2_pair pair;
  init_pair_sockets(&pair);
  pair.client.va_base = LUPINE_VA_FIRST_BASE;
  pair.client.va_size = LUPINE_VA_ARENA_SIZE;
  pair.client.va_next = 0;
  pair.client.w_offset = static_cast<intptr_t>(LUPINE_VA_WRITE_BASE) -
                         static_cast<intptr_t>(pair.client.va_base);

  int client_result = 0;
  std::thread client(
      [&] { client_result = rpc_http2_client_init(&pair.client); });
  int server_result = rpc_http2_server_init(&pair.server);
  client.join();

  require(server_result == 1, "arena request was dispatched as an RPC stream");
  require(client_result == LUPINE_RPC_HTTP2_VA_UNSUPPORTED,
          "arena refusal did not distinguish an unsupported platform from a "
          "taken slot");
  // The arena fields are borrowed, not reserved; clear them so teardown does
  // not unmap a range this test never owned.
  pair.client.va_base = 0;
  pair.client.va_size = 0;
  pair.client.w_offset = 0;
  rpc_http2_destroy(&pair.client);
  rpc_http2_destroy(&pair.server);
  require(unsetenv("LUPINE_IDENTITY_VA") == 0, "env restore failed");
}

// The client starts from its own window and needs the peer's stated window on
// the same connection to correct itself when the two differ.
void test_client_await_ready_reports_va_window() {
  h2_pair pair;
  init_pair_sockets(&pair);

  lupine_va_window peer = {};
  bool stated = false;
  std::thread client([&] {
    require(rpc_http2_client_init(&pair.client) == 0, "client h2 init failed");
    require(rpc_http2_client_await_ready(&pair.client) == 0,
            "client was not accepted");
    stated = rpc_http2_peer_va_window(&pair.client, &peer);
  });
  require(rpc_http2_server_init(&pair.server) == 0, "server h2 init failed");
  client.join();

  const lupine_va_window local = lupine_va_local_window();
  if (local.arena_size == 0) {
    require(!stated, "a host with no arena still advertised a window");
  } else {
    require(stated, "HEAD / did not carry the arena window");
    require(peer.base == local.base && peer.arena_size == local.arena_size &&
                peer.count == local.count,
            "advertised window did not match the local one");
  }
  rpc_http2_destroy(&pair.client);
  rpc_http2_destroy(&pair.server);
}

// Every slot the server will accept must be one the client can derive from the
// advertised window, and the writable alias must stay mappable for each.
void test_va_window_slots_are_self_consistent() {
  const lupine_va_window window = lupine_va_local_window();
  if (window.arena_size == 0) {
    return;
  }
  require(window.count != 0, "a stated window offered no slots");
  require(window.arena_size <= SIZE_MAX / window.count,
          "window slots overflow the address space");
  uintptr_t span = window.arena_size * window.count;
  require(window.base <= UINTPTR_MAX - span,
          "window runs past the top of the address space");
  // Arenas and their aliases are distinct regions; an overlap would let a
  // transport write land inside the protected read view.
  uintptr_t alias_end = LUPINE_VA_WRITE_BASE + span;
  require(LUPINE_VA_WRITE_BASE >= window.base + span ||
              alias_end <= window.base,
          "writable aliases overlap the arena window");
  for (unsigned int slot = 0; slot < window.count; ++slot) {
    uintptr_t offset = static_cast<uintptr_t>(slot) * window.arena_size;
    uintptr_t base = window.base + offset;
    intptr_t w_offset = static_cast<intptr_t>(LUPINE_VA_WRITE_BASE + offset) -
                        static_cast<intptr_t>(base);
    require(static_cast<uintptr_t>(static_cast<intptr_t>(base) + w_offset) ==
                LUPINE_VA_WRITE_BASE + offset,
            "alias offset did not round-trip for this slot");
  }
}

void test_fragmented_cursors() {
  h2_pair pair = make_pair();
  std::vector<std::string> chunks = {"alpha", "", ":", "beta", ":gamma"};
  std::string received;
  std::thread reader([&] { received = read_string(&pair.server, 16); });
  write_all(&pair.client, chunks);
  reader.join();
  require(received == "alpha:beta:gamma", "fragmented payload mismatch");
}

void test_fragmented_frames_direct() {
  h2_pair pair = make_pair();
  exchange_settings(&pair);
  const rpc_http2_read_stats before = read_stats(&pair.server);
  const std::string expected = "fragmented-data";
  std::string received;
  std::thread reader(
      [&] { received = read_string(&pair.server, expected.size()); });
  usleep(20 * 1000);
  write_all(&pair.client, {"fragment"});
  write_all(&pair.client, {"ed"});
  write_all(&pair.client, {"-data"});
  reader.join();
  require(received == expected, "fragmented frame mismatch");
  const rpc_http2_read_stats after = read_stats(&pair.server);
  require(after.direct_bytes - before.direct_bytes == received.size(),
          "fragmented frames were not read directly");
  require(after.staged_bytes == before.staged_bytes,
          "fragmented frames unexpectedly staged bytes");
}

void test_partial_read_stages_only_overflow() {
  h2_pair pair = make_pair();
  exchange_settings(&pair);
  std::string payload(4096, '\0');
  for (size_t i = 0; i < payload.size(); ++i) {
    payload[i] = static_cast<char>(i & 0x7f);
  }
  std::string received(payload.size(), '\0');
  const rpc_http2_read_stats before = read_stats(&pair.server);
  std::thread reader([&] {
    require(rpc_http2_read(&pair.server, received.data(), 7) == 7,
            "partial prefix read failed");
    require(rpc_http2_read(&pair.server, received.data() + 7,
                           received.size() - 7) ==
                static_cast<int>(received.size() - 7),
            "partial suffix read failed");
  });
  usleep(20 * 1000);
  write_all(&pair.client, {payload});
  reader.join();
  require(received == payload, "partial read payload mismatch");
  const rpc_http2_read_stats after = read_stats(&pair.server);
  require(after.direct_bytes - before.direct_bytes == 7,
          "partial read direct byte count mismatch");
  require(after.staged_bytes - before.staged_bytes == payload.size() - 7,
          "partial read staged byte count mismatch");
  require(after.staged_read_bytes - before.staged_read_bytes ==
              payload.size() - 7,
          "partial read staged-copy count mismatch");
  require(after.staged_buffers - before.staged_buffers == 1,
          "partial read staging allocation count mismatch");
  require(after.peak_staged_bytes >= payload.size() - 7,
          "partial read peak staging mismatch");
}

void test_truncated_read_clears_direct_destination() {
  h2_pair pair = make_pair();
  exchange_settings(&pair);
  std::vector<unsigned char> guarded(48, 0xa5);
  const std::string prefix = "truncated";
  int read_result = 0;
  std::thread reader([&] {
    read_result = rpc_http2_read(&pair.server, guarded.data() + 8, 32);
  });
  usleep(20 * 1000);
  write_all(&pair.client, {prefix});
  require(shutdown(pair.client.connfd, SHUT_WR) == 0,
          "truncated writer shutdown failed");
  reader.join();
  require(read_result == -1, "truncated read unexpectedly succeeded");
  require(std::memcmp(guarded.data() + 8, prefix.data(), prefix.size()) == 0,
          "truncated read lost received prefix");
  for (size_t i = 0; i < guarded.size(); ++i) {
    if (i >= 8 && i < 8 + prefix.size()) {
      continue;
    }
    require(guarded[i] == 0xa5, "truncated read wrote outside prefix");
  }
}

void test_close_already_failed_transport_socket() {
  h2_pair pair = make_pair();
  exchange_settings(&pair);

  // Transport readers mark the logical connection closed before the owner
  // performs descriptor cleanup. That state must not suppress shutdown: a
  // peer can otherwise retain its per-connection resources indefinitely.
  pair.client.closed = 1;
  rpc_close_transport_socket(&pair.client);
  require(pair.client.connfd == LUPINE_INVALID_SOCKET,
          "failed transport socket was not claimed");

  char buffer[4096];
  ssize_t received = 0;
  do {
    received = recv(pair.server.connfd, buffer, sizeof(buffer), 0);
  } while (received > 0);
  require(received == 0 || (received < 0 && errno == ECONNRESET),
          "failed transport socket did not notify peer");

  // Cleanup can race the dispatch thread and the library destructor.
  rpc_close_transport_socket(&pair.client);
  require(pair.client.connfd == LUPINE_INVALID_SOCKET,
          "transport socket close was not idempotent");
}

void test_abort_failed_transport_with_queued_data() {
  int listener = socket(AF_INET, SOCK_STREAM, 0);
  require(listener >= 0, "queued close listener socket failed");

  sockaddr_in address = {};
  address.sin_family = AF_INET;
  address.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
  address.sin_port = 0;
  require(bind(listener, reinterpret_cast<sockaddr *>(&address),
               sizeof(address)) == 0,
          "queued close bind failed");
  socklen_t address_size = sizeof(address);
  require(getsockname(listener, reinterpret_cast<sockaddr *>(&address),
                      &address_size) == 0,
          "queued close getsockname failed");
  require(listen(listener, 1) == 0, "queued close listen failed");

  conn_t connection = {};
  connection.connfd = socket(AF_INET, SOCK_STREAM, 0);
  require(connection.connfd >= 0, "queued close client socket failed");
  require(lupine_socket_apply_transport_options(connection.connfd) == 0,
          "queued close transport setup failed");
#ifdef TCP_USER_TIMEOUT
  int user_timeout = 0;
  socklen_t user_timeout_size = sizeof(user_timeout);
  require(getsockopt(connection.connfd, IPPROTO_TCP, TCP_USER_TIMEOUT,
                     &user_timeout, &user_timeout_size) == 0 &&
              user_timeout == 105000,
          "unacknowledged transport data has no dead-peer timeout");
#endif
  int buffer_size = 4096;
  require(setsockopt(connection.connfd, SOL_SOCKET, SO_SNDBUF, &buffer_size,
                     sizeof(buffer_size)) == 0,
          "queued close send buffer setup failed");
  require(connect(connection.connfd, reinterpret_cast<sockaddr *>(&address),
                  sizeof(address)) == 0,
          "queued close connect failed");
  int peer = accept(listener, nullptr, nullptr);
  require(peer >= 0, "queued close accept failed");
  require(setsockopt(peer, SOL_SOCKET, SO_RCVBUF, &buffer_size,
                     sizeof(buffer_size)) == 0,
          "queued close receive buffer setup failed");

  int flags = fcntl(connection.connfd, F_GETFL, 0);
  require(flags >= 0 &&
              fcntl(connection.connfd, F_SETFL, flags | O_NONBLOCK) == 0,
          "queued close nonblocking setup failed");
  std::array<char, 64 * 1024> payload = {};
  size_t queued = 0;
  for (;;) {
    ssize_t sent =
        send(connection.connfd, payload.data(), payload.size(), MSG_NOSIGNAL);
    if (sent > 0) {
      queued += static_cast<size_t>(sent);
      continue;
    }
    require(sent < 0 && (errno == EAGAIN || errno == EWOULDBLOCK),
            "queued close fill failed");
    break;
  }
  require(queued != 0, "queued close did not queue data");

  connection.closed = 1;
  rpc_close_transport_socket(&connection);

  ssize_t received = 0;
  do {
    received = recv(peer, payload.data(), payload.size(), 0);
  } while (received > 0);
  require(received < 0 && errno == ECONNRESET,
          "queued transport close was not abortive");

  close(peer);
  close(listener);
}

void test_independent_stream_lanes() {
  h2_pair pair = make_pair();
  exchange_settings(&pair);

  int32_t client_first = rpc_http2_lane_stream(&pair.client, 101);
  int32_t client_second = rpc_http2_lane_stream(&pair.client, 202);
  require(client_first > 0 && client_second > 0 &&
              client_first != client_second,
          "client lanes did not get distinct HTTP/2 streams");
  int32_t server_first = rpc_http2_accept_stream(&pair.server);
  int32_t server_second = rpc_http2_accept_stream(&pair.server);
  require(server_first == client_first && server_second == client_second,
          "server accepted the wrong HTTP/2 lane streams");

  const std::string blocked_request = "leave this lane unread";
  const std::string independent_request = "second lane still progresses";
  require(write_stream_bytes(&pair.client, client_first, blocked_request.data(),
                             blocked_request.size()) == 0,
          "first lane request write failed");
  require(write_stream_bytes(&pair.client, client_second,
                             independent_request.data(),
                             independent_request.size()) == 0,
          "second lane request write failed");
  std::string received_request(independent_request.size(), '\0');
  require(rpc_http2_read_stream(&pair.server, server_second,
                                received_request.data(),
                                received_request.size()) ==
              static_cast<int>(received_request.size()),
          "second lane request was blocked by the first lane");
  require(received_request == independent_request,
          "second lane request payload mismatch");

  const std::string blocked_response = "leave this response unread";
  const std::string independent_response = "second response still progresses";
  require(write_stream_bytes(&pair.server, server_first,
                             blocked_response.data(),
                             blocked_response.size()) == 0,
          "first lane response write failed");
  require(write_stream_bytes(&pair.server, server_second,
                             independent_response.data(),
                             independent_response.size()) == 0,
          "second lane response write failed");
  std::string received_response(independent_response.size(), '\0');
  require(rpc_http2_read_stream(&pair.client, client_second,
                                received_response.data(),
                                received_response.size()) ==
              static_cast<int>(received_response.size()),
          "second lane response was blocked by the first lane");
  require(received_response == independent_response,
          "second lane response payload mismatch");
}

void test_socket_reader_hands_off_between_streams() {
  h2_pair pair = make_pair();
  exchange_settings(&pair);

  int32_t client_lane = rpc_http2_lane_stream(&pair.client, 404);
  int32_t server_lane = rpc_http2_accept_stream(&pair.server);
  require(client_lane > 0 && server_lane == client_lane,
          "reader handoff lane setup failed");

  char dispatch_value = '\0';
  char lane_value = '\0';
  std::atomic<bool> lane_done{false};
  std::thread dispatch_reader([&] {
    require(rpc_http2_read(&pair.client, &dispatch_value, 1) == 1,
            "dispatch stream handoff read failed");
  });
  usleep(20 * 1000);
  std::thread lane_reader([&] {
    require(rpc_http2_read_stream(&pair.client, client_lane, &lane_value, 1) ==
                1,
            "lane handoff read failed");
    lane_done.store(true, std::memory_order_release);
  });
  usleep(20 * 1000);

  require(write_stream_bytes(&pair.server, server_lane, "l", 1) == 0,
          "lane handoff write failed");
  for (int i = 0; i < 100 && !lane_done.load(std::memory_order_acquire); ++i) {
    usleep(10 * 1000);
  }
  bool handed_off = lane_done.load(std::memory_order_acquire);

  write_all(&pair.server, {"d"});
  lane_reader.join();
  dispatch_reader.join();
  require(handed_off, "socket reader reclaimed recv ahead of a waiting lane");
  require(lane_value == 'l' && dispatch_value == 'd',
          "reader handoff payload mismatch");
}

void exchange_settings(h2_pair *pair) {
  std::string request = "x";
  std::string response = "y";
  std::string received_request;
  std::thread reader(
      [&] { received_request = read_string(&pair->server, request.size()); });
  write_all(&pair->client, {request});
  reader.join();
  require(received_request == request, "settings exchange request mismatch");

  write_all(&pair->server, {response});
  require(read_string(&pair->client, response.size()) == response,
          "settings exchange response mismatch");
}

void test_large_payload() {
  h2_pair pair = make_pair();
  exchange_settings(&pair);

  std::string payload(2 * 1024 * 1024, '\0');
  for (size_t i = 0; i < payload.size(); ++i) {
    payload[i] = static_cast<char>('a' + (i % 26));
  }

  size_t midpoint = payload.size() / 2;
  std::string received;
  std::thread reader(
      [&] { received = read_string(&pair.server, payload.size()); });
  write_all(&pair.client,
            {payload.substr(0, midpoint), payload.substr(midpoint)});
  reader.join();
  require(received == payload, "large payload mismatch");
}

void test_payload_larger_than_flow_control_window() {
  h2_pair pair = make_pair();
  exchange_settings(&pair);

  constexpr size_t payload_size =
      static_cast<size_t>(INT32_MAX) + 64 * 1024 + 1;

  void *payload = mmap(nullptr, payload_size, PROT_READ,
                       MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE, -1, 0);
  require(payload != MAP_FAILED, "flow-control payload mmap failed");

  std::atomic<bool> read_failed{false};
  size_t received = 0;
  std::thread server_reader([&] {
    std::array<unsigned char, 64 * 1024> buffer = {};
    while (received < payload_size) {
      size_t chunk = std::min(buffer.size(), payload_size - received);
      if (rpc_http2_read(&pair.server, buffer.data(), chunk) !=
          static_cast<int>(chunk)) {
        read_failed = true;
        break;
      }
      if (!std::all_of(buffer.begin(), buffer.begin() + chunk,
                       [](unsigned char value) { return value == 0; })) {
        read_failed = true;
        break;
      }
      received += chunk;
    }
  });

  // Production connections always have an RPC dispatch thread reading control
  // frames. It does not receive application bytes here, but processing the
  // peer's WINDOW_UPDATE frames is what lets a large write make progress.
  std::thread client_control_reader([&] {
    unsigned char unused = 0;
    (void)rpc_http2_read(&pair.client, &unused, sizeof(unused));
  });

  int write_result = write_bytes(&pair.client, payload, payload_size);
  if (write_result != 0) {
    shutdown(pair.client.connfd, SHUT_RDWR);
    shutdown(pair.server.connfd, SHUT_RDWR);
  }
  server_reader.join();
  shutdown(pair.client.connfd, SHUT_RDWR);
  client_control_reader.join();
  munmap(payload, payload_size);

  require(write_result == 0, "flow-controlled write failed before completion");
  require(!read_failed, "flow-controlled read failed");
  require(received == payload_size, "flow-controlled payload was truncated");
}

// A server-side hold keeps received payload bytes uncredited until the staging
// they landed in retires. Held bytes saturate at a cap so the reader filling
// the hold always has credit left for the bytes it is blocked on, and the
// release hands the rest back.
void test_server_window_hold_caps_and_releases() {
  h2_pair pair = make_pair();
  exchange_settings(&pair);

  constexpr size_t kBurst = LUPINE_FF_STAGING_WINDOW_BYTES;
  std::vector<char> payload(kBurst, 'h');
  std::vector<char> received(kBurst, '\0');
  rpc_http2_window_credit held;

  // As on a production connection, the client's dispatch thread is what applies
  // the server's WINDOW_UPDATE frames; it releases when the server replies.
  std::thread client_control_reader([&] {
    unsigned char unused = 0;
    (void)rpc_http2_read(&pair.client, &unused, sizeof(unused));
  });

  std::thread reader([&] {
    rpc_http2_window_hold_begin(&pair.server);
    require(rpc_http2_read(&pair.server, received.data(), received.size()) ==
                static_cast<int>(received.size()),
            "held payload read failed");
    held = rpc_http2_window_hold_end(&pair.server);
  });
  require(write_bytes(&pair.client, payload.data(), payload.size()) == 0,
          "held write failed");
  reader.join();
  require(received == payload, "held payload mismatch");
  require(held.bytes == LUPINE_FF_STAGING_WINDOW_BYTES / 2,
          "held bytes did not saturate at the cap");

  // The cap is connection-wide, not per stream: otherwise enough busy lanes
  // could consume the connection window and starve control traffic.
  constexpr size_t kSecondLaneBurst = 1024 * 1024;
  int32_t client_lane = rpc_http2_lane_stream(&pair.client, 303);
  int32_t server_lane = rpc_http2_accept_stream(&pair.server);
  require(client_lane > 0 && server_lane == client_lane,
          "second held lane setup failed");
  std::vector<char> second_payload(kSecondLaneBurst, 'i');
  std::vector<char> second_received(kSecondLaneBurst, '\0');
  rpc_http2_window_credit second_held;
  std::thread second_reader([&] {
    require(rpc_bind_http2_stream(&pair.server, server_lane) == 0,
            "second held lane bind failed");
    rpc_http2_window_hold_begin(&pair.server);
    require(rpc_http2_read_stream(&pair.server, server_lane,
                                  second_received.data(),
                                  second_received.size()) ==
                static_cast<int>(second_received.size()),
            "second held lane read failed");
    second_held = rpc_http2_window_hold_end(&pair.server);
    rpc_unbind_http2_stream(&pair.server);
  });
  require(write_stream_bytes(&pair.client, client_lane, second_payload.data(),
                             second_payload.size()) == 0,
          "second held lane write failed");
  second_reader.join();
  require(second_received == second_payload,
          "second held lane payload mismatch");
  require(second_held.bytes == 0,
          "window hold cap was incorrectly applied per stream");

  // Still holding: the uncapped remainder must have been credited, so another
  // window's worth of payload still flows.
  std::fill(received.begin(), received.end(), '\0');
  std::thread held_reader([&] {
    require(rpc_http2_read(&pair.server, received.data(), received.size()) ==
                static_cast<int>(received.size()),
            "read under an outstanding hold failed");
  });
  require(write_bytes(&pair.client, payload.data(), payload.size()) == 0,
          "write under an outstanding hold failed");
  held_reader.join();
  require(received == payload, "payload under an outstanding hold mismatch");

  rpc_http2_window_release(&pair.server, held);
  std::string tail = "released";
  std::string received_tail;
  std::thread tail_reader(
      [&] { received_tail = read_string(&pair.server, tail.size()); });
  write_all(&pair.client, {tail});
  tail_reader.join();
  require(received_tail == tail, "payload after release mismatch");

  write_all(&pair.server, {"z"});
  client_control_reader.join();
}

void test_reset_wakes_flow_controlled_writer() {
  h2_pair pair;
  init_raw_server_peer(&pair);

  // Shrink the peer's stream window so the writer pauses after 64 KiB rather
  // than requiring another multi-gigabyte test payload.
  std::array<unsigned char, 15> settings = {};
  settings[2] = 6;
  settings[3] = NGHTTP2_SETTINGS;
  settings[10] = NGHTTP2_SETTINGS_INITIAL_WINDOW_SIZE;
  settings[13] = settings[14] = 0xff;
  require(raw_write_all(pair.server.connfd, settings.data(), settings.size()),
          "failed to send reduced-window SETTINGS");
  bool received_ack = false;
  while (!received_ack) {
    std::array<unsigned char, 9> header = {};
    require(raw_read_frame(pair.server.connfd, &header),
            "failed to read SETTINGS acknowledgement");
    received_ack =
        header[3] == NGHTTP2_SETTINGS && (header[4] & NGHTTP2_FLAG_ACK) != 0;
  }

  std::atomic<bool> reset_failed{false};
  std::thread server_reset([&] {
    std::array<unsigned char, 64 * 1024> buffer = {};
    size_t received = 0;
    bool reset_sent = false;
    for (;;) {
      ssize_t chunk =
          lupine_socket_recv(pair.server.connfd, buffer.data(), buffer.size());
      if (chunk < 0 && lupine_socket_error_is_intr()) {
        continue;
      }
      if (chunk <= 0) {
        break;
      }
      received += static_cast<size_t>(chunk);
      if (!reset_sent && received >= 32 * 1024) {
        std::array<unsigned char, 13> reset = {};
        reset[2] = 4;
        reset[3] = NGHTTP2_RST_STREAM;
        reset[8] = 1;
        reset[12] = NGHTTP2_CANCEL;
        reset_sent =
            raw_write_all(pair.server.connfd, reset.data(), reset.size());
        if (!reset_sent) {
          reset_failed = true;
          break;
        }
      }
    }
    if (!reset_sent) {
      reset_failed = true;
    }
  });

  std::string payload(128 * 1024, 'x');
  int write_result = write_bytes(&pair.client, payload.data(), payload.size());
  shutdown(pair.client.connfd, SHUT_RDWR);
  shutdown(pair.server.connfd, SHUT_RDWR);
  server_reset.join();

  require(write_result < 0,
          "reset flow-controlled write unexpectedly succeeded");
  require(!reset_failed, "failed to deliver RST_STREAM");
}

// Round-trips a multi-block LZ4-framed payload: the transport compresses it
// lazily block by block (h2.cpp) and rpc_read_payload_part decodes it with
// chunked, block-aligned reads (compress.cpp). The payload mixes
// compressible and random data so both compressed and raw block tokens are
// exercised, and plain cursors surround the framed one as in a real message.
void test_framed_payload_round_trip() {
  h2_pair pair = make_pair();
  exchange_settings(&pair);

  std::string prefix = "head";
  std::string suffix = "tail";
  std::vector<char> payload(2 * LUPINE_COMPRESS_BLOCK_BYTES + 123457);
  unsigned int seed = 42;
  for (size_t i = 0; i < payload.size() / 2; ++i) {
    payload[i] = static_cast<char>(i % 7);
  }
  for (size_t i = payload.size() / 2; i < payload.size(); ++i) {
    seed = seed * 1664525u + 1013904223u;
    payload[i] = static_cast<char>(seed >> 24);
  }

  std::string received_prefix;
  std::string received_suffix;
  std::vector<char> received(payload.size());
  std::thread reader([&] {
    received_prefix = read_string(&pair.server, prefix.size());
    size_t first = LUPINE_COMPRESS_BLOCK_BYTES;
    require(rpc_read_payload_part(&pair.server, 1, received.data(), first) ==
                static_cast<int>(first),
            "framed read part 1 failed");
    require(rpc_read_payload_part(&pair.server, 1, received.data() + first,
                                  received.size() - first) ==
                static_cast<int>(received.size() - first),
            "framed read part 2 failed");
    received_suffix = read_string(&pair.server, suffix.size());
  });

  std::vector<rpc_write_cursor> cursors = {
      rpc_write_cursor::plain(prefix.data(), prefix.size()),
      rpc_write_cursor::framed(payload.data(), payload.size()),
      rpc_write_cursor::plain(suffix.data(), suffix.size())};
  require(rpc_http2_write(&pair.client, cursors) == 0, "framed write failed");
  reader.join();
  require(received_prefix == prefix, "framed prefix mismatch");
  require(received == payload, "framed payload mismatch");
  require(received_suffix == suffix, "framed suffix mismatch");
}

void test_rpc_write_queue_grows() {
  conn_t zero_length = {};
  require(rpc_write(&zero_length, nullptr, 0) == 0,
          "zero-length rpc_write failed");
  require(rpc_write_payload(&zero_length, nullptr, 0) == 0,
          "zero-length rpc_write_payload failed");
  require(zero_length.write_queue.empty(),
          "zero-length write consumed a queue entry");

  h2_pair pair = make_pair();

  constexpr int kCount = 300;
  constexpr int kOp = 77;
  std::vector<int> values(kCount);
  for (int i = 0; i < kCount; ++i) {
    values[i] = i;
  }

  std::vector<int> received(kCount, -1);
  std::thread reader([&] {
    int32_t stream_id = rpc_http2_accept_stream(&pair.server);
    require(rpc_bind_http2_stream(&pair.server, stream_id) == 0,
            "large queue stream bind failed");
    require(rpc_dispatch(&pair.server, 0) == kOp,
            "large queue dispatch failed");
    for (int i = 0; i < kCount; ++i) {
      require(rpc_read(&pair.server, &received[i], sizeof(received[i])) ==
                  static_cast<int>(sizeof(received[i])),
              "large queue payload read failed");
    }
    require(rpc_read_end(&pair.server) > 0, "large queue read_end failed");
    rpc_unbind_http2_stream(&pair.server);
  });

  require(rpc_write_start_request(&pair.client, kOp) == 0,
          "large queue request start failed");
  for (int i = 0; i < kCount; ++i) {
    require(rpc_write(&pair.client, &values[i], sizeof(values[i])) == 0,
            "large queue rpc_write failed");
  }
  require(pair.client.write_queue.size() == kCount + 2,
          "large queue count mismatch");
  require(rpc_write_end(&pair.client) > 0, "large queue write_end failed");
  reader.join();
  require(received == values, "large queue payload mismatch");
}

void test_rpc_write_buffer_uses_fixed_allocation() {
  h2_pair pair = make_pair();
  require(rpc_write_start_response(&pair.client, 21) == 0,
          "buffered response start failed");
  uint8_t first = 17;
  uint64_t second = 19;
  uint32_t third = 23;
  require(rpc_write_buffer(&pair.client, sizeof(first), alignof(uint8_t)) ==
              nullptr,
          "rpc_write_buffer accepted data without a reservation");
  require(rpc_copy_alloc(&pair.client, 20) == 0,
          "fixed copy allocation failed");
  require(rpc_copy_alloc(&pair.client, sizeof(first)) < 0,
          "rpc_copy_alloc replaced an active reservation");
  auto *first_buffer = static_cast<uint8_t *>(
      rpc_write_buffer(&pair.client, sizeof(first), alignof(uint8_t)));
  require(first_buffer != nullptr, "fixed first buffer failed");
  *first_buffer = first;
  auto *second_buffer = static_cast<uint64_t *>(
      rpc_write_buffer(&pair.client, sizeof(second), alignof(uint64_t)));
  require(second_buffer != nullptr, "fixed second buffer failed");
  *second_buffer = second;
  auto *direct = static_cast<uint32_t *>(
      rpc_write_buffer(&pair.client, sizeof(third), alignof(uint32_t)));
  require(direct != nullptr, "direct write buffer allocation failed");
  *direct = third;
  require(rpc_write_buffer(&pair.client, 1, alignof(uint8_t)) == nullptr,
          "rpc_write_buffer exceeded its fixed allocation");
  require(pair.client.write_queue.size() == 5,
          "fixed copy queue count mismatch");
  require(pair.client.write_copy_offset == 20, "fixed copy cursor mismatch");
  require(pair.client.write_queue[2].data == pair.client.write_copy_buffer &&
              pair.client.write_queue[3].data ==
                  pair.client.write_copy_buffer + 8 &&
              pair.client.write_queue[4].data ==
                  pair.client.write_copy_buffer + 16,
          "fixed copy queued the wrong spans");
  require(*pair.client.write_queue[2].data == first &&
              *reinterpret_cast<const uint64_t *>(
                  pair.client.write_queue[3].data) == second &&
              *reinterpret_cast<const uint32_t *>(
                  pair.client.write_queue[4].data) == third,
          "fixed copy changed buffered values");
  require(rpc_write_end(&pair.client) == 21, "fixed response write end failed");
  require(pair.client.write_copy_buffer == nullptr,
          "rpc_write_end retained the copy buffer");

  require(rpc_copy_alloc(&pair.client, 32) == 0,
          "stale copy allocation failed");
  require(rpc_write_start_response(&pair.client, 22) == 0,
          "response start after abort failed");
  require(pair.client.write_copy_buffer == nullptr,
          "response reset retained a stale copy buffer");
  require(rpc_copy_alloc(&pair.client, sizeof(second)) == 0,
          "copy allocation after reset failed");
  require(rpc_write_end(&pair.client) == 22, "reset response write end failed");
}

void test_rpc_write_buffer_cleans_up_on_transport_failure_and_destroy() {
  {
    h2_pair pair = make_pair();
    require(rpc_write_start_response(&pair.client, 25) == 0,
            "failed transport response start failed");
    int value = 23;
    require(rpc_copy_alloc(&pair.client, sizeof(value)) == 0,
            "failed transport copy allocation failed");
    auto *buffer = static_cast<int *>(
        rpc_write_buffer(&pair.client, sizeof(value), alignof(int)));
    require(buffer != nullptr, "failed transport buffer allocation failed");
    *buffer = value;
    rpc_close_transport_socket(&pair.client);
    require(rpc_write_end(&pair.client) < 0,
            "failed transport unexpectedly sent copied data");
    require(pair.client.write_copy_buffer == nullptr,
            "transport failure retained the copy buffer");
  }

  conn_t conn = {};
  require(rpc_conn_init(&conn, LUPINE_INVALID_SOCKET, 0) == 0,
          "destroy connection init failed");
  require(rpc_copy_alloc(&conn, 64) == 0, "destroy copy allocation failed");
  rpc_conn_destroy(&conn);
  require(conn.write_copy_buffer == nullptr && conn.write_copy_capacity == 0 &&
              conn.write_copy_offset == 0,
          "rpc_conn_destroy retained the copy buffer");
}

void test_rpc_lz4_payload_round_trip() {
  h2_pair pair = make_pair();

  std::string prefix = "before";
  std::string suffix = "after";
  constexpr int kOp = 79;
  std::vector<char> payload(LUPINE_COMPRESS_BLOCK_BYTES + 128 * 1024);
  for (size_t i = 0; i < payload.size(); ++i) {
    payload[i] = static_cast<char>(i % 13);
  }

  std::string received_prefix(prefix.size(), '\0');
  std::string received_suffix(suffix.size(), '\0');
  std::vector<char> received(payload.size());
  const rpc_http2_read_stats before = read_stats(&pair.server);
  std::thread reader([&] {
    int32_t stream_id = rpc_http2_accept_stream(&pair.server);
    require(rpc_bind_http2_stream(&pair.server, stream_id) == 0,
            "lz4 payload stream bind failed");
    require(rpc_dispatch(&pair.server, 0) == kOp,
            "lz4 payload dispatch failed");
    require(rpc_read(&pair.server, received_prefix.data(),
                     received_prefix.size()) ==
                static_cast<int>(received_prefix.size()),
            "lz4 payload prefix read failed");
    require(rpc_read_payload(&pair.server, received.data(), received.size()) ==
                static_cast<int>(received.size()),
            "lz4 payload payload read failed");
    require(rpc_read(&pair.server, received_suffix.data(),
                     received_suffix.size()) ==
                static_cast<int>(received_suffix.size()),
            "lz4 payload suffix read failed");
    require(rpc_read_end(&pair.server) > 0, "lz4 payload read_end failed");
    rpc_unbind_http2_stream(&pair.server);
  });

  require(rpc_write_start_request(&pair.client, kOp) == 0,
          "lz4 payload request start failed");
  require(rpc_write(&pair.client, prefix.data(), prefix.size()) == 0,
          "lz4 payload prefix write failed");
  require(rpc_write_payload(&pair.client, payload.data(), payload.size()) == 0,
          "lz4 payload payload write failed");
  require(rpc_write(&pair.client, suffix.data(), suffix.size()) == 0,
          "lz4 payload suffix write failed");
  require(rpc_write_end(&pair.client) > 0, "lz4 payload write_end failed");
  reader.join();

  require(received_prefix == prefix, "lz4 payload prefix mismatch");
  require(received == payload, "lz4 payload payload mismatch");
  require(received_suffix == suffix, "lz4 payload suffix mismatch");
  const rpc_http2_read_stats after = read_stats(&pair.server);
  require(after.direct_bytes > before.direct_bytes,
          "lz4 payload did not use direct receive");
}

void test_rpc_repeated_responses_on_lane() {
  h2_pair pair = make_pair();

  constexpr int kOp = 81;
  constexpr int kChunkCount = 2;
  std::vector<char> payload(LUPINE_COMPRESS_BLOCK_BYTES, '\0');
  for (size_t i = 0; i < payload.size(); ++i) {
    payload[i] = static_cast<char>(i % 17);
  }

  std::thread server([&] {
    int32_t stream_id = rpc_http2_accept_stream(&pair.server);
    require(rpc_bind_http2_stream(&pair.server, stream_id) == 0,
            "repeated response stream bind failed");
    require(rpc_dispatch(&pair.server, 0) == kOp,
            "repeated response dispatch failed");
    int request_id = rpc_read_end(&pair.server);
    require(request_id > 0, "repeated response request end failed");

    for (int chunk = 0; chunk < kChunkCount; ++chunk) {
      require(rpc_write_start_response(&pair.server, request_id) == 0,
              "repeated response start failed");
      require(rpc_write(&pair.server, &chunk, sizeof(chunk)) == 0,
              "repeated response index write failed");
      require(rpc_write_payload(&pair.server, payload.data(), payload.size()) ==
                  0,
              "repeated response payload write failed");
      require(rpc_write_end(&pair.server) == request_id,
              "repeated response write end failed");
    }
    rpc_unbind_http2_stream(&pair.server);
  });

  require(rpc_write_start_request(&pair.client, kOp) == 0,
          "repeated response request start failed");
  int request_id = rpc_write_end(&pair.client);
  require(request_id > 0, "repeated response request write failed");
  for (int expected = 0; expected < kChunkCount; ++expected) {
    require(rpc_read_start(&pair.client, request_id) == 0,
            "repeated response read start failed");
    int chunk = -1;
    std::vector<char> received(payload.size());
    require(rpc_read(&pair.client, &chunk, sizeof(chunk)) == sizeof(chunk),
            "repeated response index read failed");
    require(rpc_read_payload(&pair.client, received.data(), received.size()) ==
                static_cast<int>(received.size()),
            "repeated response payload read failed");
    require(rpc_read_end(&pair.client) == request_id,
            "repeated response read end failed");
    require(chunk == expected, "repeated response index mismatch");
    require(received == payload, "repeated response payload mismatch");
  }
  server.join();
}

// Handlers start request chains without their own null checks; an unreachable
// server (null route conn) or a failed connection must fail the chain here
// instead of dereferencing the conn.
void test_request_start_rejects_null_and_closed_conn() {
  require(rpc_write_start_request(nullptr, 42) == -1,
          "null conn request start did not fail");
  conn_t closed = {};
  closed.closed = 1;
  require(rpc_write_start_request(&closed, 42) == -1,
          "closed conn request start did not fail");
}

// TSan supports only narrow application VA bands and owns these fixed test
// addresses itself. CUDA targets are not TSan-instrumented in this project.
#if defined(MAP_FIXED_NOREPLACE) && !defined(__SANITIZE_THREAD__)
void test_rpc_read_uses_w_offset() {
  long configured_page_size = sysconf(_SC_PAGESIZE);
  require(configured_page_size > 0, "page size lookup failed");
  size_t page_size = static_cast<size_t>(configured_page_size);

  char path[] = "/tmp/lupine-rpc-alias-XXXXXX";
  int fd = mkstemp(path);
  require(fd >= 0, "shared alias file creation failed");
  unlink(path);
  require(ftruncate(fd, static_cast<off_t>(page_size)) == 0,
          "shared alias file resize failed");

  uintptr_t server_address =
      LUPINE_HOST_ALLOCATION_SERVER_BASE + UINT64_C(0x200000);
  uintptr_t read_address = server_address + LUPINE_HOST_ALLOCATION_R_OFFSET;
  uintptr_t write_address = server_address + LUPINE_HOST_ALLOCATION_W_OFFSET;
  void *read_view =
      mmap(reinterpret_cast<void *>(read_address), page_size,
           PROT_READ | PROT_WRITE, MAP_SHARED | MAP_FIXED_NOREPLACE, fd, 0);
  void *write_view =
      mmap(reinterpret_cast<void *>(write_address), page_size,
           PROT_READ | PROT_WRITE, MAP_SHARED | MAP_FIXED_NOREPLACE, fd, 0);
  close(fd);
  require(read_view == reinterpret_cast<void *>(read_address) &&
              write_view == reinterpret_cast<void *>(write_address),
          "fixed shared alias mapping failed");
  require(mprotect(read_view, page_size, PROT_NONE) == 0,
          "read view protection failed");

  h2_pair pair = make_pair();
  pair.client.w_offset = LUPINE_HOST_ALLOCATION_W_OFFSET;
  constexpr int kOp = 83;
  const std::array<unsigned char, 8> expected = {3, 1, 4, 1, 5, 9, 2, 6};
  std::thread server([&] {
    int32_t stream_id = rpc_http2_accept_stream(&pair.server);
    require(rpc_bind_http2_stream(&pair.server, stream_id) == 0,
            "alias response stream bind failed");
    require(rpc_dispatch(&pair.server, 0) == kOp,
            "alias response dispatch failed");
    int request_id = rpc_read_end(&pair.server);
    require(request_id > 0, "alias request read end failed");
    require(rpc_write_start_response(&pair.server, request_id) == 0,
            "alias response write start failed");
    require(rpc_write(&pair.server, expected.data(), expected.size()) == 0,
            "alias response write failed");
    require(rpc_write_end(&pair.server) == request_id,
            "alias response write end failed");
    rpc_unbind_http2_stream(&pair.server);
  });

  require(rpc_write_start_request(&pair.client, kOp) == 0,
          "alias request write start failed");
  int request_id = rpc_write_end(&pair.client);
  require(request_id > 0, "alias request write end failed");
  require(rpc_read_start(&pair.client, request_id) == 0,
          "alias response read start failed");
  require(rpc_read(&pair.client, read_view, expected.size()) ==
              static_cast<int>(expected.size()),
          "alias response read failed");
  require(rpc_read_end(&pair.client) == request_id,
          "alias response read end failed");
  server.join();

  require(memcmp(read_view, expected.data(), expected.size()) == 0,
          "read view did not observe the writable alias");
  require(pair.client.host_allocation_writes.size() == 1 &&
              pair.client.host_allocation_writes[0].start == read_address &&
              pair.client.host_allocation_writes[0].size == expected.size(),
          "alias response write was not tracked");
  require(pair.client.host_allocation_writes_pending != 0,
          "alias response write was not marked pending");

  munmap(read_view, page_size);
  munmap(write_view, page_size);
}
#endif

} // namespace

int main() {
  test_request_start_rejects_null_and_closed_conn();
#if defined(MAP_FIXED_NOREPLACE) && !defined(__SANITIZE_THREAD__)
  test_rpc_read_uses_w_offset();
#endif
  test_rpc_write_queue_grows();
  test_rpc_write_buffer_uses_fixed_allocation();
  test_rpc_write_buffer_cleans_up_on_transport_failure_and_destroy();
  test_rpc_lz4_payload_round_trip();
  test_rpc_repeated_responses_on_lane();
  test_response_wait_sends_transport_heartbeat();
  test_client_to_server();
  test_server_receives_session_id();
  test_server_to_client_after_request_headers();
  test_head_probe_cuda_version_metadata(LUPINE_CUDA_VERSION);
  test_head_probe_cuda_version_metadata(nullptr);
  test_wire_identity_compatibility_rule();
  test_client_await_ready_reports_wire_identity();
  test_client_await_ready_reports_va_window();
  test_va_window_slots_are_self_consistent();
  test_va_claim_bumps_within_arena();
  test_va_claim_is_disjoint_under_contention();
  test_va_request_reports_unsupported_platform();
  test_fragmented_cursors();
  test_fragmented_frames_direct();
  test_partial_read_stages_only_overflow();
  test_truncated_read_clears_direct_destination();
  test_close_already_failed_transport_socket();
  test_abort_failed_transport_with_queued_data();
  test_independent_stream_lanes();
  test_socket_reader_hands_off_between_streams();
  test_large_payload();
  test_framed_payload_round_trip();
  test_payload_larger_than_flow_control_window();
  test_server_window_hold_caps_and_releases();
  test_reset_wakes_flow_controlled_writer();
  std::cout << "h2_test: PASS" << std::endl;
  return 0;
}
