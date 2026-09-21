#pragma once

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <string>

// Where the server keeps content-addressed library images and the client its
// per-application load profiles. An empty LUPINE_LIBRARY_CACHE disables both.
inline std::filesystem::path lupine_library_cache_dir() {
  const char *cache = std::getenv("LUPINE_LIBRARY_CACHE");
  const char *xdg = std::getenv("XDG_CACHE_HOME");
  const char *home = std::getenv("HOME");
  if (cache != nullptr) {
    return cache;
  }
  if (xdg != nullptr && *xdg != '\0') {
    return std::filesystem::path(xdg) / "lupine" / "libraries";
  }
  if (home != nullptr && *home != '\0') {
    return std::filesystem::path(home) / ".cache" / "lupine" / "libraries";
  }
  return {};
}

inline std::string lupine_library_hash_name(uint64_t hash) {
  char name[17];
  std::snprintf(name, sizeof(name), "%016llx",
                static_cast<unsigned long long>(hash));
  return name;
}

// Through a temp file, so a concurrent reader never sees a partial file under
// its final name.
inline void lupine_write_file_atomically(const std::filesystem::path &path,
                                         const void *data, size_t size) {
  std::error_code ec;
  std::filesystem::create_directories(path.parent_path(), ec);
  std::filesystem::path temp = path;
  temp += "." +
          std::to_string(
              std::chrono::steady_clock::now().time_since_epoch().count()) +
          ".tmp";
  std::ofstream out(temp, std::ios::binary);
  out.write(static_cast<const char *>(data),
            static_cast<std::streamsize>(size));
  out.close();
  if (out) {
    std::filesystem::rename(temp, path, ec);
  }
  std::filesystem::remove(temp, ec);
}
