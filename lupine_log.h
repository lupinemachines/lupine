#ifndef LUPINE_LOG_H
#define LUPINE_LOG_H

#include <cstdlib>
#include <cstring>
#include <iostream>

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

#endif
