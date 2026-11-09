#pragma once

#include <algorithm>
#include <chrono>

// Per-connection adaptive polling, like KVM halt polling: grow after a short
// miss, retain successful budgets, and decay after long idle waits. The 32 us
// ceiling bounds each idle spin; it is not the target spin time.
class h2_writer_poll_policy {
  static constexpr std::chrono::nanoseconds max_poll =
      std::chrono::microseconds(32);
  static constexpr std::chrono::nanoseconds initial_poll =
      std::chrono::microseconds(1);
  std::chrono::nanoseconds budget_{0};

public:
  std::chrono::nanoseconds budget() const { return budget_; }

  // Use the first output arrival, including arrivals after sleeping, rather
  // than the time the writer is scheduled again. Ignore spurious wakeups.
  void observe(std::chrono::nanoseconds idle_time) {
    if (idle_time > max_poll) {
      budget_ /= 2;
      if (budget_ < initial_poll) {
        budget_ = std::chrono::nanoseconds{0};
      }
    } else if (idle_time > budget_) {
      budget_ = std::min(max_poll, std::max(initial_poll, budget_ * 2));
    }
  }
};
