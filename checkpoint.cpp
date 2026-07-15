#include "checkpoint.h"

#include <condition_variable>
#include <cstddef>
#include <mutex>

namespace {

class capture_gate {
public:
  void begin() {
    std::unique_lock<std::mutex> lock(mutex_);
    condition_.wait(lock, [this] {
      return !checkpoint_active_ && checkpoint_waiters_ == 0;
    });
    ++admitted_begins_;
  }

  void begin_complete(bool started) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (admitted_begins_ == 0) {
      return;
    }
    --admitted_begins_;
    if (started) {
      ++active_captures_;
    }
    condition_.notify_all();
  }

  void end() {
    std::lock_guard<std::mutex> lock(mutex_);
    if (active_captures_ == 0) {
      return;
    }
    --active_captures_;
    condition_.notify_all();
  }

  void wait_for_captures() {
    std::unique_lock<std::mutex> lock(mutex_);
    ++checkpoint_waiters_;
    condition_.wait(lock, [this] { return !checkpoint_active_; });
    --checkpoint_waiters_;
    checkpoint_active_ = true;
    condition_.wait(lock, [this] {
      return admitted_begins_ == 0 && active_captures_ == 0;
    });
  }

  void resume_captures() {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!checkpoint_active_) {
      return;
    }
    checkpoint_active_ = false;
    condition_.notify_all();
  }

private:
  std::mutex mutex_;
  std::condition_variable condition_;
  std::size_t admitted_begins_ = 0;
  std::size_t active_captures_ = 0;
  std::size_t checkpoint_waiters_ = 0;
  bool checkpoint_active_ = false;
};

capture_gate &global_capture_gate() {
  // Intentionally leak the process-wide gate so CUDA calls made during static
  // destruction cannot race its destructor.
  static capture_gate *gate = new capture_gate();
  return *gate;
}

} // namespace

namespace lupine_checkpoint {

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
