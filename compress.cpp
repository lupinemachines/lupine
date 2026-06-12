// Optional LZ4 compression for large host<->device memory transfer payloads.
//
// The client always advertises support with an "x-lupine-compress: lz4"
// HTTP/2 request header and the server mirrors it, so both directions of a
// connection frame large payloads; toward a peer that did not advertise
// (e.g. an older client) the wire format stays byte-identical to before.
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
//
// Framed payloads are never materialized in full. The write side only marks
// the payload iovec (rpc_write_framed) and the HTTP/2 transport compresses
// one block at a time into a reusable scratch buffer as nghttp2 pulls data
// (see h2.cpp), so memory stays bounded by a single block and early blocks
// reach the wire while later blocks are still being compressed. The read
// side mirrors this with a single compressed-block scratch buffer.

#include "lupine_fatbin.h"
#include "rpc.h"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <lz4.h>
#include <stdlib.h>

namespace {

constexpr size_t kLupineCompressMinBytes = 64 * 1024;
constexpr size_t kLupineCompressBlockBytes = LUPINE_COMPRESS_BLOCK_BYTES;

static_assert(kLupineCompressBlockBytes <= LZ4_MAX_INPUT_SIZE,
              "compression block must fit a single LZ4 block");

bool lupine_compress_trace_enabled() {
  const char *trace = getenv("LUPINE_TRACE");
  return trace != nullptr && trace[0] != '\0' && strcmp(trace, "0") != 0;
}

} // namespace

// lupine_payload_precompressed reports whether a payload is a CUDA fatbin
// whose bytes are predominantly (>75%) already-compressed members, in which
// case attempting LZ4 on it only wastes CPU for a ~1.05 ratio (measured on
// nvcc --compress-mode=size output and -Xfatbin --compress-all output, the
// CUDA 11/12 default shape). Fatbins dominated by members that fatbinary
// left raw still LZ4-compress 10x+, so they must keep the attempt; nvcc
// 13.1's default mode compresses only PTX members, so its SASS-bearing
// fatbins land in that bucket. The result only disables the sender's
// per-block LZ4 attempt (every block is framed with the raw fallback token);
// the wire format and the receiver are unchanged. On any parse doubt this
// returns 0, i.e. the payload is compressed exactly as before.
int lupine_payload_precompressed(const void *data, size_t size) {
  if (data == nullptr || size < sizeof(lupine_fatbin_header)) {
    return 0;
  }
  const auto *bytes = static_cast<const unsigned char *>(data);
  lupine_fatbin_header header;
  memcpy(&header, bytes, sizeof(header));
  if (header.magic != LUPINE_FATBIN_MAGIC ||
      header.header_size < sizeof(header) || header.files_size == 0 ||
      header.header_size > size ||
      header.files_size > size - header.header_size) {
    return 0;
  }
  const size_t end = static_cast<size_t>(header.header_size) +
                     static_cast<size_t>(header.files_size);
  size_t offset = header.header_size;
  size_t member_bytes = 0;
  size_t compressed_bytes = 0;
  while (offset < end) {
    if (end - offset < sizeof(lupine_fatbin_member_header)) {
      return 0;
    }
    lupine_fatbin_member_header member;
    memcpy(&member, bytes + offset, sizeof(member));
    if (member.header_size < sizeof(member) ||
        member.header_size > end - offset ||
        member.size > end - offset - member.header_size) {
      return 0;
    }
    size_t span = static_cast<size_t>(member.header_size) +
                  static_cast<size_t>(member.size);
    member_bytes += span;
    if ((member.flags &
         (LUPINE_FATBIN_MEMBER_LZ4 | LUPINE_FATBIN_MEMBER_ZSTD)) != 0 &&
        member.compressed_size != 0 && member.uncompressed_size != 0) {
      compressed_bytes += span;
    }
    offset += span;
  }
  if (member_bytes == 0) {
    return 0;
  }
  return compressed_bytes * 4 > member_bytes * 3;
}

int lupine_payload_framed(conn_t *conn, size_t total_size) {
  return total_size >= kLupineCompressMinBytes && rpc_http2_compress_lz4(conn);
}

// rpc_write_payload writes a bulk data payload, compressing it when the
// connection negotiated compression and the payload is large enough. The
// payload is compressed lazily by the transport as it streams to the socket;
// like rpc_write, the caller's buffer must stay valid until rpc_write_end().
int rpc_write_payload(conn_t *conn, const void *data, size_t size) {
  if (size == 0) {
    return 0;
  }
  if (!lupine_payload_framed(conn, size)) {
    return rpc_write(conn, data, size);
  }
  int no_compress = lupine_payload_precompressed(data, size);
  if (no_compress && lupine_compress_trace_enabled()) {
    fprintf(stderr,
            "LUPINE payload is a precompressed fatbin, skipping LZ4 attempt "
            "(size=%zu)\n",
            size);
  }
  return rpc_write_framed(conn, data, size, no_compress);
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
