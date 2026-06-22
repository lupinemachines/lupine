#ifndef LUPINE_LOG_H
#define LUPINE_LOG_H

#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <mutex>
#include <sstream>

enum lupine_log_level {
  LUPINE_LOG_LEVEL_NONE = 0,
  LUPINE_LOG_LEVEL_ERROR = 1,
  LUPINE_LOG_LEVEL_DEBUG = 2,
};

// Verbosity is controlled by the LUPINE_LOG_LEVEL environment variable
// ("none", "error", or "debug"). The default is "debug" so existing
// diagnostics keep printing to stderr exactly as before.
inline bool lupine_log_enabled(lupine_log_level level) {
  static lupine_log_level configured = [] {
    const char *value = getenv("LUPINE_LOG_LEVEL");
    if (value == nullptr || value[0] == '\0') {
      return LUPINE_LOG_LEVEL_DEBUG;
    }
    if (strcmp(value, "none") == 0 || strcmp(value, "0") == 0) {
      return LUPINE_LOG_LEVEL_NONE;
    }
    if (strcmp(value, "error") == 0 || strcmp(value, "1") == 0) {
      return LUPINE_LOG_LEVEL_ERROR;
    }
    return LUPINE_LOG_LEVEL_DEBUG;
  }();
  return level <= configured;
}

inline std::ofstream &lupine_trace_file() {
  static std::ofstream file;
  return file;
}

inline std::ostream *lupine_trace_stream() {
  static std::ostream *stream = []() -> std::ostream * {
    const char *value = getenv("LUPINE_TRACE");
    if (value == nullptr || value[0] == '\0' || strcmp(value, "0") == 0) {
      return nullptr;
    }
    if (strcmp(value, "1") == 0) {
      return &std::cout;
    }
    if (strcmp(value, "2") == 0) {
      return &std::cerr;
    }

    std::ofstream &file = lupine_trace_file();
    file.open(value, std::ios::app);
    if (!file.is_open()) {
      std::cerr << "Failed to open LUPINE_TRACE file: " << value << std::endl;
      return nullptr;
    }
    return &file;
  }();
  return stream;
}

inline std::mutex &lupine_trace_mutex() {
  static std::mutex mutex;
  return mutex;
}

// Stream-style logging helpers, e.g.:
//   LUPINE_LOG_ERROR("failed to open " << path << " errno=" << errno);
#define LUPINE_LOG_AT(level, ...)                                              \
  do {                                                                         \
    if (lupine_log_enabled(level)) {                                           \
      std::cerr << __VA_ARGS__ << std::endl;                                   \
    }                                                                          \
  } while (0)

#define LUPINE_LOG_ERROR(...) LUPINE_LOG_AT(LUPINE_LOG_LEVEL_ERROR, __VA_ARGS__)
#define LUPINE_LOG_DEBUG(...) LUPINE_LOG_AT(LUPINE_LOG_LEVEL_DEBUG, __VA_ARGS__)

#define LUPINE_TRACE_LOG(...)                                                  \
  do {                                                                         \
    std::ostream *lupine_trace_out = lupine_trace_stream();                    \
    if (lupine_trace_out != nullptr) {                                         \
      std::ostringstream lupine_trace_message;                                 \
      lupine_trace_message << __VA_ARGS__;                                     \
      std::lock_guard<std::mutex> lupine_trace_lock(lupine_trace_mutex());     \
      (*lupine_trace_out) << lupine_trace_message.str() << std::endl;          \
    }                                                                          \
  } while (0)

#endif
