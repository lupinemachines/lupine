#include "adaptive_compression.h"
#include "lupine_log.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <lz4.h>

namespace lupine_compression {
namespace {
void append32(std::vector<unsigned char> &out, uint32_t n) {
  for (unsigned shift = 0; shift < 32; shift += 8)
    out.push_back(n >> shift);
}
uint32_t read32(const unsigned char *in) {
  uint32_t n = 0;
  for (unsigned i = 0; i < 4; ++i)
    n |= uint32_t(in[i]) << (i * 8);
  return n;
}
using clock = std::chrono::steady_clock;
double nanos(clock::time_point since) {
  return std::chrono::duration<double, std::nano>(clock::now() - since).count();
}
} // namespace

policy::choice policy::choose(const unsigned char *data, size_t size) {
  // Small flushes are latency-sensitive and don't amortize a Zstd context or
  // a second encoding. They never consume the bulk sampling budget.
  if (size < 4096)
    return {codec::lz4, false, 0};
  std::array<unsigned, 256> counts{};
  constexpr size_t sample_size = 1024;
  unsigned most_common = 0;
  // Contiguous mini-windows see all bytes of a float or instruction; a fixed
  // stride would repeatedly sample only its low byte and misclassify it.
  for (size_t window = 0; window < 64; ++window) {
    size_t start = window * (size - 16) / 63;
    for (size_t i = 0; i < 16; ++i)
      most_common = std::max(most_common, ++counts[data[start + i]]);
  }
  unsigned bucket = 3;
  if (most_common > sample_size / 2)
    bucket = 0;
  else if (most_common > sample_size / 8)
    bucket = 1;
  else if (most_common > sample_size / 32)
    bucket = 2;
  auto &history = histories_[bucket];
  bool sample = history.until_sample <= size;
  if (sample)
    history.until_sample =
        history.observations < 2 ? block_size : 32 * 1024 * 1024;
  else
    history.until_sample -= size;
  codec algorithm = history.selected;
  if (algorithm == codec::lz4 && history.incompressible)
    algorithm = codec::raw;
  return {algorithm, sample, bucket};
}

void policy::observe(unsigned bucket, size_t size, size_t lz4_bytes,
                     double lz4_ns, size_t zstd_bytes, double zstd_ns) {
  auto &history = histories_[bucket];
  ++history.observations;
  history.incompressible = lz4_bytes >= size * 0.99;
  ++samples;
  double wire_ns = 1e9 / bytes_per_second_;
  double lz4_cost = lz4_ns + std::min(size, lz4_bytes) * wire_ns;
  // Budget one additional nanosecond per decoded byte for Zstd. This is a
  // deliberately conservative allowance, not a claim about the peer's CPU.
  double zstd_cost = zstd_ns + size + std::min(size, zstd_bytes) * wire_ns;
  double current = history.selected == codec::zstd ? zstd_cost : lz4_cost;
  double alternative = history.selected == codec::zstd ? lz4_cost : zstd_cost;
  if (alternative < current * 0.8) {
    if (++history.votes >= 2) {
      history.selected =
          history.selected == codec::zstd ? codec::lz4 : codec::zstd;
      history.votes = 0;
    }
  } else
    history.votes = 0;
  const char *debug = std::getenv("LUPINE_DEBUG");
  if (debug && *debug && std::strcmp(debug, "0") != 0) {
    std::lock_guard<std::mutex> lock(lupine_trace_mutex());
    LUPINE_LOG_DEBUG("compression sample bucket="
                     << bucket << " bytes=" << size << " lz4=" << lz4_bytes
                     << " zstd=" << zstd_bytes << " lz4_ns=" << lz4_ns
                     << " zstd_ns=" << zstd_ns
                     << " delivery_Bps=" << bytes_per_second_ << " selected="
                     << static_cast<unsigned>(history.selected));
  }
}

void policy::delivery_rate(double rate) {
  if (!std::isfinite(rate) || rate <= 0)
    return;
  rate = std::clamp(rate, 1024.0, 10e9);
  bool changed = rate > bytes_per_second_ * 2 || rate < bytes_per_second_ / 2;
  bytes_per_second_ = rate;
  if (changed) {
    // Re-evaluate promptly, but never double-compress every bulk block.
    for (auto &history : histories_)
      history.until_sample = std::min(history.until_sample, 4 * block_size);
  }
}

encoder::~encoder() {
  ZSTD_freeCCtx(zstd_);
  LZ4F_freeCompressionContext(lz4_);
}
bool encoder::encode(const unsigned char *data, size_t size, policy &selection,
                     std::vector<unsigned char> &output) {
  if (size == 0 || size > block_size)
    return false;
  auto choice = selection.choose(data, size);
  size_t lz4_size = size, zstd_size = size;
  double lz4_ns = 0, zstd_ns = 0;
  if (choice.sample) {
    lz4_output_.resize(LZ4_compressBound(static_cast<int>(size)));
    auto start = clock::now();
    int result = LZ4_compress_default(
        reinterpret_cast<const char *>(data),
        reinterpret_cast<char *>(lz4_output_.data()), static_cast<int>(size),
        static_cast<int>(lz4_output_.size()));
    lz4_ns = nanos(start);
    if (result <= 0)
      return false;
    lz4_size = static_cast<size_t>(result);
  }
  if (choice.sample || choice.algorithm == codec::zstd) {
    if (!zstd_)
      zstd_ = ZSTD_createCCtx();
    if (!zstd_)
      return false;
    zstd_output_.resize(ZSTD_compressBound(size));
    auto start = clock::now();
    zstd_size = ZSTD_compressCCtx(zstd_, zstd_output_.data(),
                                  zstd_output_.size(), data, size, 1);
    zstd_ns = nanos(start);
    if (ZSTD_isError(zstd_size))
      return false;
  }
  if (choice.sample)
    selection.observe(choice.bucket, size, lz4_size, lz4_ns, zstd_size,
                      zstd_ns);
  const unsigned char *encoded = data;
  size_t encoded_size = size;
  codec algorithm = codec::raw;
  if (choice.algorithm == codec::zstd && zstd_size < size) {
    encoded = zstd_output_.data();
    encoded_size = zstd_size;
    algorithm = codec::zstd;
  } else if (choice.algorithm == codec::lz4) {
    // Keep the linked LZ4 history across small RPC flushes and across any
    // intervening raw/Zstd blocks. LZ4F owns its dictionary (stableSrc=false).
    LZ4F_preferences_t preferences{};
    preferences.frameInfo.blockSizeID = LZ4F_max256KB;
    preferences.autoFlush = 1;
    lz4_output_.resize(LZ4F_compressBound(size, &preferences) +
                       LZ4F_HEADER_SIZE_MAX);
    size_t header_size = 0;
    if (!lz4_) {
      if (LZ4F_isError(LZ4F_createCompressionContext(&lz4_, LZ4F_VERSION)))
        return false;
      header_size = LZ4F_compressBegin(lz4_, lz4_output_.data(),
                                       lz4_output_.size(), &preferences);
      if (LZ4F_isError(header_size))
        return false;
    }
    size_t result = LZ4F_compressUpdate(lz4_, lz4_output_.data() + header_size,
                                        lz4_output_.size() - header_size, data,
                                        size, nullptr);
    if (LZ4F_isError(result))
      return false;
    encoded = lz4_output_.data();
    encoded_size = header_size + result;
    algorithm = codec::lz4;
  }
  append32(output, (static_cast<uint32_t>(encoded_size) << 2) |
                       static_cast<uint32_t>(algorithm));
  append32(output, static_cast<uint32_t>(size));
  output.insert(output.end(), encoded, encoded + encoded_size);
  ++selection.blocks[static_cast<unsigned>(algorithm)];
  return true;
}
bool encoder::finish(std::vector<unsigned char> &output) {
  if (lz4_) {
    unsigned char end[64];
    size_t size = LZ4F_compressEnd(lz4_, end, sizeof(end), nullptr);
    if (LZ4F_isError(size))
      return false;
    append32(output, (static_cast<uint32_t>(size) << 2) |
                         static_cast<uint32_t>(codec::lz4));
    append32(output, 0);
    output.insert(output.end(), end, end + size);
  }
  append32(output, static_cast<uint32_t>(codec::end));
  append32(output, 0);
  return true;
}

decoder::~decoder() {
  ZSTD_freeDCtx(zstd_);
  LZ4F_freeDecompressionContext(lz4_);
}

bool decoder::decode(emit_fn emit, void *context) {
  const unsigned char *bytes = body_.data();
  if (algorithm_ != codec::raw) {
    decoded_.resize(decoded_size_);
    if (algorithm_ == codec::lz4) {
      if (lz4_finished_)
        return false;
      size_t header_bytes = 0;
      if (!lz4_) {
        if (LZ4F_isError(LZ4F_createDecompressionContext(&lz4_, LZ4F_VERSION)))
          return false;
        LZ4F_frameInfo_t info{};
        header_bytes = body_.size();
        size_t result =
            LZ4F_getFrameInfo(lz4_, &info, body_.data(), &header_bytes);
        if (LZ4F_isError(result) || info.blockSizeID < LZ4F_max64KB ||
            info.blockSizeID > LZ4F_max256KB || info.dictID != 0)
          return false;
      }
      size_t input = body_.size() - header_bytes;
      // One spare output byte detects a lying decoded length before emitting.
      decoded_.resize(decoded_size_ + 1);
      size_t produced = decoded_.size();
      size_t result =
          LZ4F_decompress(lz4_, decoded_.data(), &produced,
                          body_.data() + header_bytes, &input, nullptr);
      if (LZ4F_isError(result) || input != body_.size() - header_bytes ||
          produced != decoded_size_)
        return false;
      lz4_finished_ = result == 0;
      if (lz4_finished_ != (decoded_size_ == 0))
        return false;
    } else {
      if (!zstd_)
        zstd_ = ZSTD_createDCtx();
      if (!zstd_ ||
          ZSTD_getFrameContentSize(body_.data(), body_.size()) !=
              decoded_size_ ||
          ZSTD_findFrameCompressedSize(body_.data(), body_.size()) !=
              body_.size())
        return false;
      size_t result = ZSTD_decompressDCtx(
          zstd_, decoded_.data(), decoded_.size(), body_.data(), body_.size());
      if (ZSTD_isError(result) || result != decoded_size_)
        return false;
    }
    bytes = decoded_.data();
  }
  // Match the legacy transport's staging granularity.
  for (size_t offset = 0; offset < decoded_size_; offset += 64 * 1024)
    emit(context, bytes + offset,
         std::min<size_t>(64 * 1024, decoded_size_ - offset));
  return true;
}
bool decoder::consume(const unsigned char *data, size_t size, emit_fn emit,
                      void *context) {
  while (size) {
    if (finished_)
      return false;
    if (header_size_ < header_.size()) {
      size_t n = std::min(size, header_.size() - header_size_);
      std::memcpy(header_.data() + header_size_, data, n);
      header_size_ += n;
      data += n;
      size -= n;
      if (header_size_ < header_.size())
        continue;
      uint32_t packed = read32(header_.data());
      algorithm_ = static_cast<codec>(packed & 3);
      size_t encoded_size = packed >> 2;
      decoded_size_ = read32(header_.data() + 4);
      if (algorithm_ == codec::end) {
        if (encoded_size || decoded_size_ || size || (lz4_ && !lz4_finished_))
          return false;
        finished_ = true;
        return true;
      }
      if (!encoded_size || decoded_size_ > block_size ||
          encoded_size > block_size + 64 ||
          (algorithm_ != codec::lz4 &&
           (!decoded_size_ || encoded_size > decoded_size_)) ||
          (algorithm_ == codec::raw && encoded_size != decoded_size_))
        return false;
      encoded_size_ = encoded_size;
      if (algorithm_ != codec::raw)
        body_.resize(encoded_size);
      body_size_ = 0;
    }
    size_t n = std::min(size, encoded_size_ - body_size_);
    if (algorithm_ == codec::raw)
      emit(context, data, n);
    else
      std::memcpy(body_.data() + body_size_, data, n);
    body_size_ += n;
    data += n;
    size -= n;
    if (body_size_ == encoded_size_) {
      if (algorithm_ != codec::raw && !decode(emit, context))
        return false;
      header_size_ = 0;
    }
  }
  return true;
}
} // namespace lupine_compression
