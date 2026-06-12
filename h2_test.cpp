#include "rpc.h"

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <string>
#include <sys/socket.h>
#include <thread>
#include <unistd.h>
#include <vector>

namespace {

struct h2_pair {
  conn_t client = {};
  conn_t server = {};

  ~h2_pair() {
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
    std::cerr << message << std::endl;
    std::exit(1);
  }
}

h2_pair make_pair() {
  int fds[2] = {-1, -1};
  require(socketpair(AF_UNIX, SOCK_STREAM, 0, fds) == 0, "socketpair failed");

  h2_pair pair;
  pair.client.connfd = fds[0];
  pair.server.connfd = fds[1];
  require(rpc_http2_client_init(&pair.client) == 0, "client h2 init failed");
  require(rpc_http2_server_init(&pair.server) == 0, "server h2 init failed");
  return pair;
}

void write_all(conn_t *conn, const std::vector<std::string> &chunks) {
  std::vector<struct iovec> iov;
  iov.reserve(chunks.size());
  for (const std::string &chunk : chunks) {
    iov.push_back({const_cast<char *>(chunk.data()), chunk.size()});
  }
  require(rpc_http2_writev(conn, iov.data(), nullptr,
                           static_cast<int>(iov.size())) == 0,
          "h2 write failed");
}

std::string read_string(conn_t *conn, size_t size) {
  std::string output(size, '\0');
  require(rpc_http2_read(conn, output.data(), output.size()) ==
              static_cast<int>(output.size()),
          "h2 read failed");
  return output;
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

void test_fragmented_iovec() {
  h2_pair pair = make_pair();
  std::vector<std::string> chunks = {"alpha", "", ":", "beta", ":gamma"};
  std::string received;
  std::thread reader([&] { received = read_string(&pair.server, 16); });
  write_all(&pair.client, chunks);
  reader.join();
  require(received == "alpha:beta:gamma", "fragmented payload mismatch");
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

// Round-trips a multi-block LZ4-framed payload: the transport compresses it
// lazily block by block (h2.cpp) and rpc_read_payload_part decodes it with
// chunked, block-aligned reads (compress.cpp). The payload mixes
// compressible and random data so both compressed and raw block tokens are
// exercised, and plain iovecs surround the framed one as in a real message.
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

  struct iovec iov[3] = {
      {const_cast<char *>(prefix.data()), prefix.size()},
      {payload.data(), payload.size()},
      {const_cast<char *>(suffix.data()), suffix.size()},
  };
  unsigned char framed[3] = {0, 1, 0};
  require(rpc_http2_writev(&pair.client, iov, framed, 3) == 0,
          "framed write failed");
  reader.join();
  require(received_prefix == prefix, "framed prefix mismatch");
  require(received == payload, "framed payload mismatch");
  require(received_suffix == suffix, "framed suffix mismatch");
}

// Mirrors the CUDA fatbin layout observed from nvcc 13.1 output (see
// compress.cpp). Member field values in the fixtures below copy real dumps:
// kind 1 = PTX / 2 = ELF, version 0x0101, base flags 0x11, compression flag
// 0x2000 (lz4) or 0x8000 (zstd) plus nonzero compressed/uncompressed sizes.
struct test_fatbin_header {
  uint32_t magic;
  uint16_t version;
  uint16_t header_size;
  uint64_t files_size;
};

struct test_fatbin_member_header {
  uint16_t kind;
  uint16_t version;
  uint32_t header_size;
  uint64_t size;
  uint32_t compressed_size;
  uint32_t unknown;
  uint16_t minor;
  uint16_t major;
  uint32_t arch;
  uint32_t name_offset;
  uint32_t name_size;
  uint64_t flags;
  uint64_t reserved;
  uint64_t uncompressed_size;
};

static_assert(sizeof(test_fatbin_member_header) == 64,
              "test fixture must match the observed member header layout");

struct test_fatbin_member {
  bool compressed;
  size_t payload_size;
  uint64_t compress_flag; // 0x2000 = lz4, 0x8000 = zstd
};

std::vector<unsigned char>
make_fatbin(const std::vector<test_fatbin_member> &members) {
  std::vector<unsigned char> files;
  for (const test_fatbin_member &member : members) {
    test_fatbin_member_header header = {};
    header.kind = 2;
    header.version = 0x0101;
    header.header_size = sizeof(header);
    header.size = member.payload_size;
    header.minor = 8;
    header.major = 1;
    header.arch = 89;
    header.flags = 0x11;
    if (member.compressed) {
      header.flags |= member.compress_flag;
      header.compressed_size = static_cast<uint32_t>(member.payload_size);
      header.uncompressed_size = 4 * member.payload_size;
    }
    const auto *raw = reinterpret_cast<const unsigned char *>(&header);
    files.insert(files.end(), raw, raw + sizeof(header));
    files.insert(files.end(), member.payload_size,
                 static_cast<unsigned char>(member.compressed ? 0xc3 : 0x2e));
  }

  test_fatbin_header header = {};
  header.magic = 0xba55ed50;
  header.version = 1;
  header.header_size = sizeof(header);
  header.files_size = files.size();
  std::vector<unsigned char> fatbin;
  const auto *raw = reinterpret_cast<const unsigned char *>(&header);
  fatbin.insert(fatbin.end(), raw, raw + sizeof(header));
  fatbin.insert(fatbin.end(), files.begin(), files.end());
  return fatbin;
}

void test_precompressed_detection() {
  // All members compressed with zstd, the shape of nvcc --compress-mode=size
  // (and of default-mode fatbins whose bytes are dominated by compressed
  // members).
  std::vector<unsigned char> zstd =
      make_fatbin({{true, 520, 0x8000}, {true, 1088, 0x8000}});
  require(lupine_payload_precompressed(zstd.data(), zstd.size()) == 1,
          "zstd-compressed fatbin not detected");

  // All members compressed with lz4 (nvcc --compress-mode=speed
  // -Xfatbin --compress-all, and the CUDA 11/12 default).
  std::vector<unsigned char> lz4 =
      make_fatbin({{true, 13544, 0x2000}, {true, 26552, 0x2000}});
  require(lupine_payload_precompressed(lz4.data(), lz4.size()) == 1,
          "lz4-compressed fatbin not detected");

  // No members compressed (nvcc --compress-mode=none).
  std::vector<unsigned char> none =
      make_fatbin({{false, 1824, 0}, {false, 5224, 0}});
  require(lupine_payload_precompressed(none.data(), none.size()) == 0,
          "uncompressed fatbin misdetected");

  // Small compressed PTX next to a dominant raw SASS member (the shape nvcc
  // 13.1 produces in default mode): most bytes still benefit from LZ4, so
  // the payload must keep compressing.
  std::vector<unsigned char> mixed_raw =
      make_fatbin({{true, 7352, 0x8000}, {false, 485736, 0}});
  require(lupine_payload_precompressed(mixed_raw.data(), mixed_raw.size()) == 0,
          "raw-dominated fatbin misdetected as precompressed");

  // Dominant compressed member next to a small raw one (>75% compressed).
  std::vector<unsigned char> mixed_compressed =
      make_fatbin({{true, 485736, 0x2000}, {false, 7352, 0}});
  require(lupine_payload_precompressed(mixed_compressed.data(),
                                       mixed_compressed.size()) == 1,
          "compressed-dominated fatbin not detected");

  // Raw PTX text is not a fatbin.
  std::string ptx = ".version 8.3\n.target sm_89\n.address_size 64\n";
  require(lupine_payload_precompressed(ptx.data(), ptx.size()) == 0,
          "PTX text misdetected as precompressed");

  // Random bytes and zeros are not fatbins.
  std::vector<unsigned char> random(4096);
  unsigned int seed = 7;
  for (unsigned char &byte : random) {
    seed = seed * 1664525u + 1013904223u;
    byte = static_cast<unsigned char>(seed >> 24);
  }
  require(lupine_payload_precompressed(random.data(), random.size()) == 0,
          "random bytes misdetected as precompressed");
  std::vector<unsigned char> zeros(4096, 0);
  require(lupine_payload_precompressed(zeros.data(), zeros.size()) == 0,
          "zeros misdetected as precompressed");

  // Truncated and malformed fatbins must fall back to compressing.
  require(lupine_payload_precompressed(zstd.data(), 12) == 0,
          "truncated fatbin header misdetected");
  require(lupine_payload_precompressed(zstd.data(), zstd.size() - 1) == 0,
          "fatbin with truncated files area misdetected");
  std::vector<unsigned char> overrun = zstd;
  test_fatbin_member_header bad = {};
  memcpy(&bad, overrun.data() + sizeof(test_fatbin_header), sizeof(bad));
  bad.size = overrun.size(); // member payload overruns the files area
  memcpy(overrun.data() + sizeof(test_fatbin_header), &bad, sizeof(bad));
  require(lupine_payload_precompressed(overrun.data(), overrun.size()) == 0,
          "member overrunning the files area misdetected");
}

// A framed payload marked no-compress (write_iov_framed == 2) must reach the
// wire with every block using the raw fallback token, even when the bytes
// are highly compressible, and the unchanged receiver must decode it as any
// other framed payload.
void test_framed_no_compress_payload() {
  std::vector<char> payload(LUPINE_COMPRESS_BLOCK_BYTES + 123457, 'a');

  {
    h2_pair pair = make_pair();
    exchange_settings(&pair);
    std::vector<char> received(payload.size());
    std::thread reader([&] {
      char *dst = received.data();
      size_t remaining = received.size();
      while (remaining > 0) {
        size_t raw = std::min<size_t>(LUPINE_COMPRESS_BLOCK_BYTES, remaining);
        uint32_t token = 1;
        require(rpc_http2_read(&pair.server, &token, sizeof(token)) ==
                    static_cast<int>(sizeof(token)),
                "no-compress token read failed");
        require(token == 0, "no-compress block was LZ4 compressed");
        require(rpc_http2_read(&pair.server, dst, raw) == static_cast<int>(raw),
                "no-compress block read failed");
        dst += raw;
        remaining -= raw;
      }
    });
    struct iovec iov[1] = {{payload.data(), payload.size()}};
    unsigned char framed[1] = {2};
    require(rpc_http2_writev(&pair.client, iov, framed, 1) == 0,
            "no-compress framed write failed");
    reader.join();
    require(received == payload, "no-compress payload mismatch");
  }

  {
    h2_pair pair = make_pair();
    exchange_settings(&pair);
    std::vector<char> received(payload.size());
    std::thread reader([&] {
      require(rpc_read_payload_part(&pair.server, 1, received.data(),
                                    received.size()) ==
                  static_cast<int>(received.size()),
              "no-compress framed payload read failed");
    });
    struct iovec iov[1] = {{payload.data(), payload.size()}};
    unsigned char framed[1] = {2};
    require(rpc_http2_writev(&pair.client, iov, framed, 1) == 0,
            "no-compress framed write failed");
    reader.join();
    require(received == payload, "no-compress framed round trip mismatch");
  }
}

} // namespace

int main() {
  test_client_to_server();
  test_server_to_client_after_request_headers();
  test_fragmented_iovec();
  test_large_payload();
  test_framed_payload_round_trip();
  test_precompressed_detection();
  test_framed_no_compress_payload();
  std::cout << "h2_test: PASS" << std::endl;
  return 0;
}
