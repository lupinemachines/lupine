#ifndef LUPINE_COMPRESSION_DELIVERY_H
#define LUPINE_COMPRESSION_DELIVERY_H

#include <array>
#include <cstddef>
#include <cstdint>

namespace lupine_compression {
// PINGs are queued after DATA, so their ACKs measure progress through the
// receiver rather than copies into the sender's socket buffer. Compare ACK
// spacing within a continuous send burst; neither RTT nor application idle
// time should be mistaken for slow bandwidth. At most eight probes are live.
class delivery_estimator {
public:
  void begin_send(uint64_t now_ns) {
    if (now_ns - last_send_end_ > 20000000)
      ++epoch_;
  }
  void end_send(uint64_t now_ns) { last_send_end_ = now_ns; }
  uint32_t mark(uint64_t bytes) {
    if (count_ == markers_.size() || bytes - marked_bytes_ < 64 * 1024)
      return 0;
    if (++next_id_ == 0)
      ++next_id_;
    markers_[count_++] = {next_id_, bytes, epoch_};
    marked_bytes_ = bytes;
    return next_id_;
  }
  double acknowledge(uint32_t id, uint64_t now_ns) {
    size_t at = 0;
    while (at < count_ && markers_[at].id != id)
      ++at;
    if (at == count_)
      return 0;
    marker accepted = markers_[at];
    size_t remaining = count_ - at - 1;
    for (size_t i = 0; i < remaining; ++i)
      markers_[i] = markers_[at + 1 + i];
    count_ = remaining;
    if (ack_epoch_ != accepted.epoch) {
      ack_epoch_ = accepted.epoch;
      ack_bytes_ = accepted.bytes;
      last_ack_ = now_ns;
      return 0;
    }
    uint64_t elapsed = now_ns - last_ack_;
    if (elapsed < 10000000)
      return 0;
    double measured = (accepted.bytes - ack_bytes_) * 1e9 / elapsed;
    rate_ = rate_ == 0 ? measured : rate_ * 0.75 + measured * 0.25;
    ack_bytes_ = accepted.bytes;
    last_ack_ = now_ns;
    return rate_;
  }

private:
  struct marker {
    uint32_t id;
    uint64_t bytes;
    uint64_t epoch;
  };
  std::array<marker, 8> markers_{};
  size_t count_ = 0;
  uint32_t next_id_ = 0;
  uint64_t marked_bytes_ = 0;
  uint64_t epoch_ = 0;
  uint64_t ack_epoch_ = UINT64_MAX;
  uint64_t ack_bytes_ = 0;
  uint64_t last_send_end_ = 0;
  uint64_t last_ack_ = 0;
  double rate_ = 0;
};
} // namespace lupine_compression
#endif
