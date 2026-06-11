// Optional LZ4 compression for large host<->device memory transfer payloads.
//
// Compression is opt-in: the client sets LUPINE_COMPRESS=lz4 and advertises
// support with an "x-lupine-compress: lz4" HTTP/2 request header. The server
// mirrors the client's setting, so both directions of a connection either
// frame large payloads or leave the wire format byte-identical to before.
//
// A payload is framed only when both ends negotiated compression and the
// total (uncompressed) payload size is at least LUPINE_COMPRESS_MIN_BYTES.
// A framed payload is a sequence of blocks, each covering up to
// LUPINE_COMPRESS_BLOCK_BYTES of uncompressed data:
//
//   [uint32 token][block bytes]
//
// token == 0 means the block is stored raw (compression did not shrink it);
// otherwise token is the LZ4-compressed size of the block. The receiver knows
// each block's uncompressed size from its position in the payload, so the
// framing is self-describing. Because LUPINE_COMPRESS_BLOCK_BYTES divides the
// server's 64MB staging chunk, chunked readers stay aligned with the block
// schedule chosen by the writer.

#include "rpc.h"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <lz4.h>
#include <stdlib.h>

namespace {

constexpr size_t kLupineCompressMinBytes = 64 * 1024;
constexpr size_t kLupineCompressBlockBytes = 4 * 1024 * 1024;

static_assert(kLupineCompressBlockBytes <= LZ4_MAX_INPUT_SIZE,
              "compression block must fit a single LZ4 block");

struct lupine_payload_scratch {
  void *buffer;
  lupine_payload_scratch *next;
};

} // namespace

int lupine_payload_framed(conn_t *conn, size_t total_size) {
  return total_size >= kLupineCompressMinBytes && rpc_http2_compress_lz4(conn);
}

// rpc_write_payload writes a bulk data payload, compressing it when the
// connection negotiated compression and the payload is large enough. The
// framed bytes are kept alive on the connection until rpc_write_end() has
// flushed the request or response.
int rpc_write_payload(conn_t *conn, const void *data, size_t size) {
  if (size == 0) {
    return 0;
  }
  if (!lupine_payload_framed(conn, size)) {
    return rpc_write(conn, data, size);
  }

  size_t block_count =
      (size + kLupineCompressBlockBytes - 1) / kLupineCompressBlockBytes;
  size_t capacity = size + block_count * sizeof(uint32_t);
  int bound = LZ4_compressBound(
      static_cast<int>(std::min(size, kLupineCompressBlockBytes)));

  auto *node = static_cast<lupine_payload_scratch *>(
      malloc(sizeof(lupine_payload_scratch)));
  auto *out = static_cast<unsigned char *>(malloc(capacity));
  auto *tmp = static_cast<char *>(malloc(static_cast<size_t>(bound)));
  if (node == nullptr || out == nullptr || tmp == nullptr) {
    free(node);
    free(out);
    free(tmp);
    return -1;
  }

  const char *src = static_cast<const char *>(data);
  size_t out_len = 0;
  for (size_t offset = 0; offset < size; offset += kLupineCompressBlockBytes) {
    size_t raw = std::min(kLupineCompressBlockBytes, size - offset);
    int compressed =
        LZ4_compress_default(src + offset, tmp, static_cast<int>(raw), bound);
    uint32_t token = 0;
    if (compressed > 0 && static_cast<size_t>(compressed) < raw) {
      token = static_cast<uint32_t>(compressed);
      memcpy(out + out_len, &token, sizeof(token));
      memcpy(out + out_len + sizeof(token), tmp,
             static_cast<size_t>(compressed));
      out_len += sizeof(token) + static_cast<size_t>(compressed);
    } else {
      memcpy(out + out_len, &token, sizeof(token));
      memcpy(out + out_len + sizeof(token), src + offset, raw);
      out_len += sizeof(token) + raw;
    }
  }
  free(tmp);

  node->buffer = out;
  node->next = static_cast<lupine_payload_scratch *>(conn->payload_scratch);
  conn->payload_scratch = node;
  return rpc_write(conn, out, out_len);
}

// rpc_read_payload_part reads `size` uncompressed payload bytes. `framed`
// must be computed once per payload via lupine_payload_framed() with the
// payload's total size, so chunked readers stay consistent with the writer.
// Each part read must either be a multiple of the block size or extend to the
// end of the payload.
int rpc_read_payload_part(conn_t *conn, int framed, void *data, size_t size) {
  if (size == 0) {
    return 0;
  }
  if (!framed) {
    return rpc_read(conn, data, size);
  }

  auto *dst = static_cast<char *>(data);
  size_t remaining = size;
  char *scratch = nullptr;
  while (remaining > 0) {
    size_t raw = std::min(kLupineCompressBlockBytes, remaining);
    uint32_t token = 0;
    if (rpc_read(conn, &token, sizeof(token)) < 0) {
      free(scratch);
      return -1;
    }
    if (token == 0) {
      if (rpc_read(conn, dst, raw) < 0) {
        free(scratch);
        return -1;
      }
    } else {
      if (token >
          static_cast<uint32_t>(LZ4_compressBound(static_cast<int>(raw)))) {
        free(scratch);
        return -1;
      }
      if (scratch == nullptr) {
        scratch = static_cast<char *>(malloc(static_cast<size_t>(
            LZ4_compressBound(static_cast<int>(kLupineCompressBlockBytes)))));
        if (scratch == nullptr) {
          return -1;
        }
      }
      if (rpc_read(conn, scratch, token) < 0 ||
          LZ4_decompress_safe(scratch, dst, static_cast<int>(token),
                              static_cast<int>(raw)) != static_cast<int>(raw)) {
        free(scratch);
        return -1;
      }
    }
    dst += raw;
    remaining -= raw;
  }
  free(scratch);
  return static_cast<int>(size);
}

int rpc_read_payload(conn_t *conn, void *data, size_t size) {
  return rpc_read_payload_part(conn, lupine_payload_framed(conn, size), data,
                               size);
}

// rpc_drain_payload discards `size` uncompressed payload bytes, honoring the
// payload framing. Like rpc_read_payload_part, drains must start at a block
// boundary of the payload.
int rpc_drain_payload(conn_t *conn, int framed, size_t size) {
  if (!framed) {
    return rpc_drain(conn, size);
  }
  size_t remaining = size;
  while (remaining > 0) {
    size_t raw = std::min(kLupineCompressBlockBytes, remaining);
    uint32_t token = 0;
    if (rpc_read(conn, &token, sizeof(token)) < 0) {
      return -1;
    }
    if (token == 0) {
      if (rpc_drain(conn, raw) < 0) {
        return -1;
      }
    } else {
      if (token >
              static_cast<uint32_t>(LZ4_compressBound(static_cast<int>(raw))) ||
          rpc_drain(conn, token) < 0) {
        return -1;
      }
    }
    remaining -= raw;
  }
  return 0;
}

// rpc_release_payload_scratch frees any compressed payload buffers held for
// the in-flight request or response. Called by rpc_write_end() once the
// message has been handed to the transport.
void rpc_release_payload_scratch(conn_t *conn) {
  auto *node = static_cast<lupine_payload_scratch *>(conn->payload_scratch);
  while (node != nullptr) {
    lupine_payload_scratch *next = node->next;
    free(node->buffer);
    free(node);
    node = next;
  }
  conn->payload_scratch = nullptr;
}
