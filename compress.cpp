// LZ4 compression for host<->device memory transfer payloads.
//
// LZ4 framing is part of the wire protocol for every non-empty payload.
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
//
// Framed payloads are never materialized in full. The write side only marks
// the payload fragments (rpc_write_framed) and the HTTP/2 transport compresses
// one block at a time into a reusable scratch buffer as nghttp2 pulls data
// (see h2.cpp), so memory stays bounded by a single block and early blocks
// reach the wire while later blocks are still being compressed. The read side
// mirrors this with a single compressed-block scratch buffer.

#include "rpc.h"

#include <algorithm>
#include <cstdint>
#include <lz4.h>
#include <stdlib.h>

namespace {

constexpr size_t kLupineCompressBlockBytes = LUPINE_COMPRESS_BLOCK_BYTES;

static_assert(kLupineCompressBlockBytes <= LZ4_MAX_INPUT_SIZE,
              "compression block must fit a single LZ4 block");

} // namespace

// rpc_write_payload writes a framed data payload. The payload is compressed
// lazily by the transport as it streams to the socket; like rpc_write, the
// caller's buffer must stay valid until rpc_write_end().
int rpc_write_payload(conn_t *conn, const void *data, size_t size) {
  if (size == 0) {
    return 0;
  }
  return rpc_write_framed(conn, data, size);
}

// rpc_read_payload_part reads `size` uncompressed payload bytes. Each part
// read must either be a multiple of the block size or extend to the end of the
// payload.
int rpc_read_payload_part(conn_t *conn, void *data, size_t size) {
  if (size == 0) {
    return 0;
  }
  auto *dst = static_cast<char *>(data);
  size_t remaining = size;
  char *scratch = nullptr;
  size_t scratch_size = 0;
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
      if (scratch_size < token) {
        char *resized = static_cast<char *>(realloc(scratch, token));
        if (resized == nullptr) {
          free(scratch);
          return -1;
        }
        scratch = resized;
        scratch_size = token;
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

// rpc_drain_payload discards `size` uncompressed payload bytes. Like
// rpc_read_payload_part, drains must start at a block boundary of the payload.
int rpc_drain_payload(conn_t *conn, size_t size) {
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
