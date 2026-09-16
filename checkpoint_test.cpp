#include "checkpoint.h"

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <functional>
#include <future>
#include <iostream>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

namespace {

using namespace std::chrono_literals;

// Each task gets its own thread: std::async may queue work on a pool (MSVC),
// so a task could start long after the test assumes it is already waiting.
class task_thread {
public:
  explicit task_thread(std::function<void()> body) {
    std::packaged_task<void()> task(std::move(body));
    future_ = task.get_future();
    thread_ = std::thread(std::move(task));
  }
  ~task_thread() { thread_.join(); }

  std::future<void> &future() { return future_; }

private:
  std::future<void> future_;
  std::thread thread_;
};

bool expect_blocked(task_thread &task, const char *message) {
  if (task.future().wait_for(100ms) == std::future_status::timeout) {
    return true;
  }
  std::cerr << "FAIL: " << message << '\n';
  return false;
}

void wait_done(task_thread &task) { task.future().get(); }

bool test_waits_for_active_capture_and_blocks_new_capture() {
  lupine_checkpoint::capture_begin();
  lupine_checkpoint::capture_begin_complete(true);
  lupine_checkpoint::capture_begin();
  lupine_checkpoint::capture_begin_complete(true);

  task_thread checkpoint([] { lupine_checkpoint_wait_for_captures(); });
  if (!expect_blocked(checkpoint,
                      "checkpoint returned while a capture was active")) {
    lupine_checkpoint::capture_end();
    return false;
  }

  lupine_checkpoint::capture_end();
  if (!expect_blocked(checkpoint,
                      "checkpoint returned before every capture ended")) {
    lupine_checkpoint::capture_end();
    return false;
  }

  lupine_checkpoint::capture_end();
  wait_done(checkpoint);

  task_thread new_capture([] {
    lupine_checkpoint::capture_begin();
    lupine_checkpoint::capture_begin_complete(true);
  });
  if (!expect_blocked(new_capture,
                      "a new capture started while checkpointing")) {
    lupine_checkpoint_resume_captures();
    return false;
  }

  lupine_checkpoint_resume_captures();
  wait_done(new_capture);
  lupine_checkpoint::capture_end();
  return true;
}

bool test_waits_for_in_flight_begin() {
  lupine_checkpoint::capture_begin();

  task_thread checkpoint([] { lupine_checkpoint_wait_for_captures(); });
  if (!expect_blocked(checkpoint,
                      "checkpoint ignored an admitted capture begin")) {
    lupine_checkpoint::capture_begin_complete(false);
    return false;
  }

  lupine_checkpoint::capture_begin_complete(false);
  wait_done(checkpoint);
  lupine_checkpoint_resume_captures();
  return true;
}

bool test_drains_all_lanes_and_blocks_new_dispatches() {
  constexpr int lane_count = 8;
  std::mutex entered_mutex;
  std::condition_variable entered_condition;
  int entered = 0;
  std::promise<void> release_promise;
  std::shared_future<void> release = release_promise.get_future().share();
  std::vector<std::unique_ptr<task_thread>> lanes;

  for (int i = 0; i < lane_count; ++i) {
    lanes.push_back(std::make_unique<task_thread>([&] {
      lupine_checkpoint::cuda_call_guard dispatch_guard;
      {
        std::lock_guard<std::mutex> lock(entered_mutex);
        ++entered;
      }
      entered_condition.notify_one();
      release.wait();
    }));
  }
  {
    std::unique_lock<std::mutex> lock(entered_mutex);
    entered_condition.wait(lock, [&] { return entered == lane_count; });
  }

  std::atomic<bool> drained{false};
  std::atomic<bool> resumed{false};
  task_thread checkpoint([&] {
    lupine_checkpoint_drain_cuda_calls();
    drained = true;
  });
  bool passed =
      expect_blocked(checkpoint, "drain returned while lanes were active");

  // This dispatch may be admitted before drain takes the gate; it must never
  // be admitted between drain returning and resume.
  std::atomic<bool> admitted_while_held{false};
  task_thread racing_dispatch([&] {
    lupine_checkpoint::cuda_call_guard dispatch_guard;
    admitted_while_held = drained && !resumed;
  });

  release_promise.set_value();
  wait_done(checkpoint);
  for (auto &lane : lanes) {
    wait_done(*lane);
  }

  task_thread new_dispatch(
      [] { lupine_checkpoint::cuda_call_guard dispatch_guard; });
  passed &= expect_blocked(
      new_dispatch, "a new lane dispatched while the drain gate was held");
  resumed = true;
  lupine_checkpoint_resume_cuda_calls();
  wait_done(new_dispatch);
  wait_done(racing_dispatch);
  if (admitted_while_held) {
    std::cerr << "FAIL: a lane was admitted while the drain gate was held\n";
    passed = false;
  }
  return passed;
}

} // namespace

int main() {
  if (!test_waits_for_active_capture_and_blocks_new_capture() ||
      !test_waits_for_in_flight_begin() ||
      !test_drains_all_lanes_and_blocks_new_dispatches()) {
    return 1;
  }
  std::cout << "checkpoint gate tests passed\n";
  return 0;
}
