#include "checkpoint.h"

#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <mutex>

namespace {

// Dispatches admit and release with atomics alone while no drain is pending.
// begin() publishes its count before re-reading pending_, and drain() publishes
// pending_ before reading the count (all seq_cst), so either begin() backs off
// or drain() waits for it.
class cuda_call_gate {
public:
  void begin() {
    if (!pending_.load(std::memory_order_relaxed)) {
      active_calls_.fetch_add(1, std::memory_order_seq_cst);
      if (!pending_.load(std::memory_order_seq_cst)) {
        return;
      }
      end();
    }
    std::unique_lock<std::mutex> lock(mutex_);
    condition_.wait(lock, [this] { return !draining_ && drain_waiters_ == 0; });
    active_calls_.fetch_add(1, std::memory_order_seq_cst);
  }

  void end() {
    if (active_calls_.load(std::memory_order_relaxed) == 0) {
      return;
    }
    if (active_calls_.fetch_sub(1, std::memory_order_seq_cst) == 1 &&
        pending_.load(std::memory_order_seq_cst)) {
      std::lock_guard<std::mutex> lock(mutex_);
      condition_.notify_all();
    }
  }

  void drain() {
    std::unique_lock<std::mutex> lock(mutex_);
    ++drain_waiters_;
    pending_.store(true, std::memory_order_seq_cst);
    condition_.wait(lock, [this] { return !draining_; });
    --drain_waiters_;
    draining_ = true;
    condition_.wait(lock, [this] {
      return active_calls_.load(std::memory_order_seq_cst) == 0;
    });
  }

  void resume() {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!draining_) {
      return;
    }
    draining_ = false;
    pending_.store(drain_waiters_ != 0, std::memory_order_seq_cst);
    condition_.notify_all();
  }

private:
  alignas(64) std::atomic<std::size_t> active_calls_{0};
  alignas(64) std::atomic<bool> pending_{false};
  alignas(64) std::mutex mutex_;
  std::condition_variable condition_;
  std::size_t drain_waiters_ = 0;
  bool draining_ = false;
};

// Capture lifecycle operations admit and release with atomics alone while no
// checkpoint is pending. Admitted begins (high 32 bits) and active captures
// (low 32 bits) are packed into a single 64-bit atomic state_ word so the
// begin -> active capture transition in capture_begin_complete(true) executes
// in a single atomic instruction without ever dropping to zero.
class capture_gate {
public:
  void begin() {
    if (!pending_.load(std::memory_order_relaxed)) {
      state_.fetch_add(kAdmittedBeginUnit, std::memory_order_seq_cst);
      if (!pending_.load(std::memory_order_seq_cst)) {
        return;
      }
      uint64_t prev =
          state_.fetch_sub(kAdmittedBeginUnit, std::memory_order_seq_cst);
      if (prev == kAdmittedBeginUnit &&
          pending_.load(std::memory_order_seq_cst)) {
        std::lock_guard<std::mutex> lock(mutex_);
        condition_.notify_all();
      }
    }
    std::unique_lock<std::mutex> lock(mutex_);
    condition_.wait(lock, [this] {
      return !checkpoint_active_ && checkpoint_waiters_ == 0;
    });
    state_.fetch_add(kAdmittedBeginUnit, std::memory_order_seq_cst);
  }

  void begin_complete(bool started) {
    if ((state_.load(std::memory_order_relaxed) >> 32) == 0) {
      return;
    }
    if (started) {
      state_.fetch_sub(kAdmittedBeginUnit - kActiveCaptureUnit,
                       std::memory_order_seq_cst);
      return;
    }
    uint64_t prev =
        state_.fetch_sub(kAdmittedBeginUnit, std::memory_order_seq_cst);
    if (prev == kAdmittedBeginUnit &&
        pending_.load(std::memory_order_seq_cst)) {
      std::lock_guard<std::mutex> lock(mutex_);
      condition_.notify_all();
    }
  }

  void end() {
    if ((state_.load(std::memory_order_relaxed) & 0xffffffffULL) == 0) {
      return;
    }
    uint64_t prev =
        state_.fetch_sub(kActiveCaptureUnit, std::memory_order_seq_cst);
    if (prev == kActiveCaptureUnit &&
        pending_.load(std::memory_order_seq_cst)) {
      std::lock_guard<std::mutex> lock(mutex_);
      condition_.notify_all();
    }
  }

  void wait_for_captures() {
    std::unique_lock<std::mutex> lock(mutex_);
    ++checkpoint_waiters_;
    pending_.store(true, std::memory_order_seq_cst);
    condition_.wait(lock, [this] { return !checkpoint_active_; });
    --checkpoint_waiters_;
    checkpoint_active_ = true;
    condition_.wait(
        lock, [this] { return state_.load(std::memory_order_seq_cst) == 0; });
  }

  void resume_captures() {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!checkpoint_active_) {
      return;
    }
    checkpoint_active_ = false;
    pending_.store(checkpoint_waiters_ != 0, std::memory_order_seq_cst);
    condition_.notify_all();
  }

private:
  static constexpr uint64_t kAdmittedBeginUnit = 1ULL << 32;
  static constexpr uint64_t kActiveCaptureUnit = 1ULL;

  alignas(64) std::atomic<uint64_t> state_{0};
  alignas(64) std::atomic<bool> pending_{false};
  alignas(64) std::mutex mutex_;
  std::condition_variable condition_;
  std::size_t checkpoint_waiters_ = 0;
  bool checkpoint_active_ = false;
};

capture_gate &global_capture_gate() {
  // Intentionally leak the process-wide gate so CUDA calls made during static
  // destruction cannot race its destructor.
  static capture_gate *gate = new capture_gate();
  return *gate;
}

cuda_call_gate &global_cuda_call_gate() {
  // Intentionally leak the process-wide gate so late dispatches cannot race
  // its destructor during process shutdown.
  static cuda_call_gate *gate = new cuda_call_gate();
  return *gate;
}

} // namespace

namespace lupine_checkpoint {

void cuda_call_begin() { global_cuda_call_gate().begin(); }

void cuda_call_end() { global_cuda_call_gate().end(); }

void capture_begin() { global_capture_gate().begin(); }

void capture_begin_complete(bool started) {
  global_capture_gate().begin_complete(started);
}

void capture_end() { global_capture_gate().end(); }

} // namespace lupine_checkpoint

extern "C" void lupine_checkpoint_wait_for_captures(void) {
  global_capture_gate().wait_for_captures();
}

extern "C" void lupine_checkpoint_resume_captures(void) {
  global_capture_gate().resume_captures();
}

extern "C" void lupine_checkpoint_drain_cuda_calls(void) {
  global_cuda_call_gate().drain();
}

extern "C" void lupine_checkpoint_resume_cuda_calls(void) {
  global_cuda_call_gate().resume();
}
