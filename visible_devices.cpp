#include "visible_devices.h"

#include <cctype>
#include <cstdlib>

namespace {

// Strips the decoration a UUID may carry so two spellings of the same device
// compare equal: surrounding space, the "GPU-" prefix, hyphens, and case.
std::string normalize_token(std::string token) {
  const size_t begin = token.find_first_not_of(" \t");
  if (begin == std::string::npos) {
    return std::string();
  }
  const size_t end = token.find_last_not_of(" \t");
  token = token.substr(begin, end - begin + 1);
  if (token.rfind("GPU-", 0) == 0) {
    token = token.substr(4);
  }
  std::string out;
  out.reserve(token.size());
  for (char c : token) {
    if (c == '-') {
      continue;
    }
    out.push_back(static_cast<char>(tolower(static_cast<unsigned char>(c))));
  }
  return out;
}

} // namespace

std::vector<int> lupine_select_visible_devices(
    const char *spec, int count,
    const std::function<std::string(int)> &uuid_hex) {
  std::vector<int> visible;
  if (count <= 0) {
    return visible;
  }
  if (spec == nullptr) {
    visible.reserve(static_cast<size_t>(count));
    for (int i = 0; i < count; ++i) {
      visible.push_back(i);
    }
    return visible;
  }
  std::string text(spec);
  if (text.empty() || text == "-1") {
    return visible;
  }

  // UUIDs are only read if one is asked for, and then only once.
  std::vector<std::string> uuids;
  auto load_uuids = [&]() {
    if (!uuids.empty()) {
      return;
    }
    uuids.assign(static_cast<size_t>(count), std::string());
    for (int i = 0; i < count; ++i) {
      uuids[static_cast<size_t>(i)] = uuid_hex ? uuid_hex(i) : std::string();
    }
  };

  size_t pos = 0;
  while (pos <= text.size()) {
    const size_t comma = text.find(',', pos);
    const std::string token = normalize_token(text.substr(
        pos, comma == std::string::npos ? std::string::npos : comma - pos));
    if (!token.empty()) {
      if (token.find_first_not_of("0123456789") == std::string::npos) {
        const long index = strtol(token.c_str(), nullptr, 10);
        if (index >= 0 && index < count) {
          visible.push_back(static_cast<int>(index));
        }
      } else if (token.size() >= 16) {
        load_uuids();
        for (int i = 0; i < count; ++i) {
          if (!uuids[static_cast<size_t>(i)].empty() &&
              uuids[static_cast<size_t>(i)] == token) {
            visible.push_back(i);
            break;
          }
        }
      }
    }
    if (comma == std::string::npos) {
      break;
    }
    pos = comma + 1;
  }
  return visible;
}
