#ifndef LUPINE_ADAPTIVE_COMPRESSION_H
#define LUPINE_ADAPTIVE_COMPRESSION_H

#include <array>
#include <cstddef>
#include <cstdint>
#include <lz4frame.h>
#include <vector>
#include <zstd.h>

namespace lupine_compression {
constexpr size_t block_size = 1024 * 1024;
enum class codec : uint32_t { raw = 0, lz4 = 1, zstd = 2, end = 3 };

// Per outgoing connection, shared by its lanes under the transport mutex.
// Comparisons encode the same input block; no application/GPU wait time
// enters the encoder cost. A conservative decode allowance favors LZ4 when
// the network is fast enough that extra CPU matters.
class policy {
public:
  struct choice {
    codec algorithm;
    bool sample;
    unsigned bucket;
  };
  choice choose(const unsigned char *data, size_t size);
  void observe(unsigned bucket, size_t size, size_t lz4_bytes, double lz4_ns,
               size_t zstd_bytes, double zstd_ns);
  void delivery_rate(double bytes_per_second);
  double rate() const { return bytes_per_second_; }
  uint64_t samples = 0;
  std::array<uint64_t, 3> blocks{};

private:
  struct history {
    codec selected = codec::zstd;
    unsigned observations = 0;
    bool incompressible = false;
    unsigned votes = 0;
    size_t until_sample = 0;
  };
  std::array<history, 4> histories_{};
  double bytes_per_second_ = 1250000;
};

// Eight little-endian header bytes: (encoded length << 2) | codec, decoded
// length. Decoded blocks are at most block_size, encoded blocks at most
// block_size + 64. Zstd blocks are independent frames; LZ4 blocks are linked
// frame fragments, followed by a zero-decoded-length LZ4 frame footer if used.
// codec=end with zero lengths terminates the body.
class encoder {
public:
  ~encoder();
  encoder() = default;
  encoder(const encoder &) = delete;
  encoder &operator=(const encoder &) = delete;
  bool encode(const unsigned char *data, size_t size, policy &selection,
              std::vector<unsigned char> &output);
  bool finish(std::vector<unsigned char> &output);

private:
  ZSTD_CCtx *zstd_ = nullptr;
  LZ4F_compressionContext_t lz4_ = nullptr;
  std::vector<unsigned char> lz4_output_;
  std::vector<unsigned char> zstd_output_;
};

class decoder {
public:
  using emit_fn = void (*)(void *, const unsigned char *, size_t);
  ~decoder();
  decoder() = default;
  decoder(const decoder &) = delete;
  decoder &operator=(const decoder &) = delete;
  bool consume(const unsigned char *data, size_t size, emit_fn emit,
               void *context);
  bool finished() const { return finished_; }

private:
  bool decode(emit_fn emit, void *context);
  std::array<unsigned char, 8> header_{};
  size_t header_size_ = 0;
  size_t body_size_ = 0;
  size_t encoded_size_ = 0;
  size_t decoded_size_ = 0;
  codec algorithm_ = codec::raw;
  bool finished_ = false;
  std::vector<unsigned char> body_;
  std::vector<unsigned char> decoded_;
  ZSTD_DCtx *zstd_ = nullptr;
  LZ4F_decompressionContext_t lz4_ = nullptr;
  bool lz4_finished_ = false;
};
} // namespace lupine_compression
#endif
