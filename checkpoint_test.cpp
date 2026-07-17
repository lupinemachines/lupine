#include "checkpoint.h"

#include <chrono>
#include <condition_variable>
#include <future>
#include <iostream>
#include <mutex>
#include <vector>

namespace {

using namespace std::chrono_literals;

bool expect_blocked(std::future<void> &future, const char *message) {
  if (future.wait_for(100ms) == std::future_status::timeout) {
    return true;
  }
  std::cerr << "FAIL: " << message << '\n';
  return false;
}

bool expect_ready(std::future<void> &future, const char *message) {
  if (future.wait_for(1s) != std::future_status::ready) {
    std::cerr << "FAIL: " << message << '\n';
    return false;
  }
  future.get();
  return true;
}

bool test_waits_for_active_capture_and_blocks_new_capture() {
  lupine_checkpoint::capture_begin();
  lupine_checkpoint::capture_begin_complete(true);
  lupine_checkpoint::capture_begin();
  lupine_checkpoint::capture_begin_complete(true);

  auto checkpoint = std::async(std::launch::async,
                               [] { lupine_checkpoint_wait_for_captures(); });
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
  if (!expect_ready(checkpoint,
                    "checkpoint did not return after the capture ended")) {
    return false;
  }

  auto new_capture = std::async(std::launch::async, [] {
    lupine_checkpoint::capture_begin();
    lupine_checkpoint::capture_begin_complete(true);
  });
  if (!expect_blocked(new_capture,
                      "a new capture started while checkpointing")) {
    lupine_checkpoint_resume_captures();
    return false;
  }

  lupine_checkpoint_resume_captures();
  if (!expect_ready(new_capture,
                    "new capture did not resume after checkpointing")) {
    return false;
  }
  lupine_checkpoint::capture_end();
  return true;
}

bool test_waits_for_in_flight_begin() {
  lupine_checkpoint::capture_begin();

  auto checkpoint = std::async(std::launch::async,
                               [] { lupine_checkpoint_wait_for_captures(); });
  if (!expect_blocked(checkpoint,
                      "checkpoint ignored an admitted capture begin")) {
    lupine_checkpoint::capture_begin_complete(false);
    return false;
  }

  lupine_checkpoint::capture_begin_complete(false);
  if (!expect_ready(checkpoint,
                    "failed capture begin did not release checkpoint")) {
    return false;
  }
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
  std::vector<std::future<void>> lanes;
  lanes.reserve(lane_count);

  for (int i = 0; i < lane_count; ++i) {
    lanes.emplace_back(std::async(std::launch::async, [&] {
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
    if (!entered_condition.wait_for(lock, 1s,
                                    [&] { return entered == lane_count; })) {
      std::cerr << "FAIL: not every lane entered dispatch\n";
      release_promise.set_value();
      return false;
    }
  }

  auto checkpoint = std::async(std::launch::async,
                               [] { lupine_checkpoint_drain_cuda_calls(); });
  bool passed =
      expect_blocked(checkpoint, "drain returned while lanes were active");

  auto new_dispatch = std::async(std::launch::async, [] {
    lupine_checkpoint::cuda_call_guard dispatch_guard;
  });
  passed &= expect_blocked(new_dispatch,
                           "a new lane dispatched while drain was waiting");

  release_promise.set_value();
  passed &= expect_ready(checkpoint,
                         "drain did not return after every lane completed");
  for (auto &lane : lanes) {
    passed &= expect_ready(lane, "an existing lane did not complete");
  }

  passed &= expect_blocked(
      new_dispatch, "a new lane dispatched while the drain gate was held");
  lupine_checkpoint_resume_cuda_calls();
  passed &=
      expect_ready(new_dispatch, "new lane did not dispatch after resume");
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
