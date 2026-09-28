#include "adaptive_compression.h"
#include "compression_delivery.h"
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>

using namespace lupine_compression;
static void require(bool ok) {
  if (!ok)
    std::abort();
}
static void collect(void *context, const unsigned char *data, size_t size) {
  auto &out = *static_cast<std::vector<unsigned char> *>(context);
  out.insert(out.end(), data, data + size);
}
static void put32(std::vector<unsigned char> &out, uint32_t value) {
  for (unsigned shift = 0; shift < 32; shift += 8)
    out.push_back(value >> shift);
}
static std::vector<unsigned char> header(codec type, uint32_t encoded,
                                         uint32_t decoded) {
  std::vector<unsigned char> out;
  put32(out, (encoded << 2) | static_cast<uint32_t>(type));
  put32(out, decoded);
  return out;
}
static void delivery_tests() {
  delivery_estimator meter;
  meter.begin_send(1000000000);
  require(meter.mark(1024) == 0);
  uint32_t first = meter.mark(65536);
  uint32_t second = meter.mark(131072);
  require(first && second && meter.acknowledge(99, 1100000000) == 0);
  // 150 ms RTT disappears from the estimate: 64 KiB in 50 ms between ACKs.
  require(meter.acknowledge(first, 1150000000) == 0);
  require(meter.acknowledge(second, 1200000000) == 1310720);
  require(meter.acknowledge(first, 1250000000) == 0);
  meter.end_send(1250000000);
  // A one-second application pause starts a new burst, not a slow sample.
  meter.begin_send(2250000000);
  uint32_t third = meter.mark(196608);
  require(meter.acknowledge(third, 2400000000) == 0);
  for (unsigned i = 0; i < 8; ++i)
    require(meter.mark(262144 + i * 65536) != 0);
  require(meter.mark(20 * 65536) == 0);
}

static void policy_tests() {
  std::vector<unsigned char> input(block_size, 0);
  policy p;
  require(p.choose(input.data(), 100).algorithm == codec::lz4);
  auto choice = p.choose(input.data(), input.size());
  require(choice.algorithm == codec::zstd && choice.sample);
  p.delivery_rate(1e9);
  // A modest size reduction cannot repay the extra CPU on a fast link.
  p.observe(choice.bucket, block_size, block_size, 10000, block_size * 9 / 10,
            100000);
  require(p.choose(input.data(), input.size()).algorithm == codec::zstd);
  p.observe(choice.bucket, block_size, block_size, 10000, block_size * 9 / 10,
            100000);
  require(p.choose(input.data(), input.size()).algorithm == codec::raw);
  p.delivery_rate(1250000);
  // A strong compression advantage repays it on a slow link, after two votes.
  p.observe(choice.bucket, block_size, block_size, 10000, block_size / 2,
            100000);
  require(p.choose(input.data(), input.size()).algorithm == codec::raw);
  p.observe(choice.bucket, block_size, block_size, 10000, block_size / 2,
            100000);
  require(p.choose(input.data(), input.size()).algorithm == codec::zstd);
  unsigned probes = 0;
  for (int i = 0; i < 1024; ++i) {
    auto c = p.choose(input.data(), input.size());
    if (c.sample) {
      ++probes;
      p.observe(c.bucket, block_size, block_size, 10000, block_size / 2,
                100000);
    }
  }
  require(probes <= 34);
  double old = p.rate();
  p.delivery_rate(0);
  p.delivery_rate(-1);
  require(p.rate() == old);
}
static void round_trips() {
  std::mt19937 random(42);
  policy p;
  encoder e;
  std::vector<unsigned char> wire, expected;
  for (size_t size : {size_t(1), size_t(4095), size_t(4096), block_size}) {
    for (unsigned kind = 0; kind < 3; ++kind) {
      std::vector<unsigned char> data(size);
      for (size_t i = 0; i < size; ++i) {
        if (kind == 0)
          data[i] = 0;
        else if (kind == 1)
          data[i] = random();
        else
          data[i] = i % 37;
      }
      require(e.encode(data.data(), data.size(), p, wire));
      expected.insert(expected.end(), data.begin(), data.end());
    }
  }
  // Return to a linked LZ4 fragment after other codecs have reused buffers.
  std::vector<unsigned char> tail(1000, 0);
  require(e.encode(tail.data(), tail.size(), p, wire));
  expected.insert(expected.end(), tail.begin(), tail.end());
  require(e.finish(wire));
  require(p.blocks[0] && p.blocks[1] && p.blocks[2]);
  for (size_t chunk : {size_t(1), size_t(7), size_t(1023), block_size * 2}) {
    decoder d;
    std::vector<unsigned char> actual;
    for (size_t at = 0; at < wire.size(); at += chunk)
      require(d.consume(wire.data() + at, std::min(chunk, wire.size() - at),
                        collect, &actual));
    require(d.finished() && actual == expected);
    require(!d.consume(wire.data(), 1, collect, &actual));
  }
  // Every truncation of a small body must remain visibly unfinished.
  wire.clear();
  encoder short_encoder;
  unsigned char a = 1;
  require(short_encoder.encode(&a, 1, p, wire));
  require(short_encoder.finish(wire));
  for (size_t size = 0; size < wire.size(); ++size) {
    decoder d;
    std::vector<unsigned char> actual;
    require(d.consume(wire.data(), size, collect, &actual) && !d.finished());
  }
}
static void malformed() {
  for (auto wire :
       {header(codec::raw, 0, 0), header(codec::raw, 1, block_size + 1),
        header(codec::raw, block_size + 1, block_size + 1),
        header(codec::raw, 2, 1), header(codec::raw, 1, 2),
        header(codec::end, 1, 0), header(codec::end, 0, 1),
        header(codec::zstd, 1, block_size + 1)}) {
    decoder d;
    std::vector<unsigned char> actual;
    require(!d.consume(wire.data(), wire.size(), collect, &actual));
    require(actual.empty());
  }
  for (auto type : {codec::lz4, codec::zstd}) {
    auto wire = header(type, 8, 100);
    wire.resize(16, 0xff);
    decoder d;
    std::vector<unsigned char> actual;
    require(!d.consume(wire.data(), wire.size(), collect, &actual));
    require(actual.empty());
  }
  // A small outer block must not make LZ4 allocate its 4 MiB frame window.
  LZ4F_compressionContext_t lz4 = nullptr;
  require(!LZ4F_isError(LZ4F_createCompressionContext(&lz4, LZ4F_VERSION)));
  LZ4F_preferences_t preferences{};
  preferences.frameInfo.blockSizeID = LZ4F_max4MB;
  unsigned char frame_header[LZ4F_HEADER_SIZE_MAX];
  size_t bytes =
      LZ4F_compressBegin(lz4, frame_header, sizeof(frame_header), &preferences);
  require(!LZ4F_isError(bytes));
  auto huge_window = header(codec::lz4, bytes, 1);
  huge_window.insert(huge_window.end(), frame_header, frame_header + bytes);
  decoder bounded;
  std::vector<unsigned char> unused;
  require(!bounded.consume(huge_window.data(), huge_window.size(), collect,
                           &unused));
  LZ4F_freeCompressionContext(lz4);
  // An overall terminator cannot replace the linked LZ4 frame's footer.
  encoder incomplete_encoder;
  policy selection;
  std::vector<unsigned char> unfinished;
  unsigned char one = 1;
  require(incomplete_encoder.encode(&one, 1, selection, unfinished));
  auto premature_end = header(codec::end, 0, 0);
  unfinished.insert(unfinished.end(), premature_end.begin(),
                    premature_end.end());
  decoder incomplete_decoder;
  require(!incomplete_decoder.consume(unfinished.data(), unfinished.size(),
                                      collect, &unused));
  auto wire = header(codec::end, 0, 0);
  wire.push_back(1);
  decoder d;
  std::vector<unsigned char> actual;
  require(!d.consume(wire.data(), wire.size(), collect, &actual));
}
int main() {
  delivery_tests();
  policy_tests();
  round_trips();
  malformed();
  std::puts("adaptive_compression_test: PASS");
}
