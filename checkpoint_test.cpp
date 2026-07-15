#include "checkpoint.h"

#include <chrono>
#include <future>
#include <iostream>

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

} // namespace

int main() {
  if (!test_waits_for_active_capture_and_blocks_new_capture() ||
      !test_waits_for_in_flight_begin()) {
    return 1;
  }
  std::cout << "checkpoint capture gate tests passed\n";
  return 0;
}
