#ifndef LUPINE_CUBLAS_LOG_CALLBACKS_H
#define LUPINE_CUBLAS_LOG_CALLBACKS_H

#include <cstdint>
#include <functional>
#include <mutex>
#include <unordered_map>
#include <vector>

struct conn_t;

// Log callbacks are acknowledged by the dispatch thread and run when the
// originating caller consumes its response, after releasing the RPC lane.
class pending_log_callbacks {
public:
  template <typename Callback>
  bool enqueue(conn_t *conn, int32_t stream, Callback callback,
               bool before_callbacks = false) {
    try {
      std::lock_guard<std::mutex> lock(mutex_);
      callbacks_[conn][stream].push_back(
          {std::move(callback), before_callbacks});
      return true;
    } catch (...) {
      return false;
    }
  }

  void complete(conn_t *conn, int32_t stream) {
    std::vector<pending_callback> ready;
    {
      std::lock_guard<std::mutex> lock(mutex_);
      auto connection = callbacks_.find(conn);
      if (connection == callbacks_.end()) {
        return;
      }
      auto lane = connection->second.find(stream);
      if (lane == connection->second.end()) {
        return;
      }
      ready = std::move(lane->second);
      connection->second.erase(lane);
      if (connection->second.empty()) {
        callbacks_.erase(connection);
      }
    }
    // Flush file bytes on the caller thread before user callbacks can replace
    // and close their FILE. Keeping stdio on this thread also permits callers
    // to hold flockfile() across a library call, as with the native library.
    for (auto &callback : ready) {
      if (callback.before_callbacks) {
        callback.invoke();
      }
    }
    for (auto &callback : ready) {
      if (!callback.before_callbacks) {
        callback.invoke();
      }
    }
  }

  void discard(conn_t *conn) {
    std::lock_guard<std::mutex> lock(mutex_);
    callbacks_.erase(conn);
  }

private:
  struct pending_callback {
    std::function<void()> invoke;
    bool before_callbacks;
  };
  std::mutex mutex_;
  std::unordered_map<conn_t *,
                     std::unordered_map<int32_t, std::vector<pending_callback>>>
      callbacks_;
};

#endif
