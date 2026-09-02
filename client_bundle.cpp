#include "client_bundle.h"

#include <array>
#include <utility>

namespace {

constexpr char kClientBundlePrefix[] = "/.well-known/lupine/client/v1/";
constexpr std::array<const char *, 6> kClientPlatforms = {
    "linux/amd64", "linux/arm64",   "macos/amd64",
    "macos/arm64", "windows/amd64", "windows/arm64",
};

bool platform_supported(const std::string &platform) {
  for (const char *supported : kClientPlatforms) {
    if (platform == supported) {
      return true;
    }
  }
  return false;
}

} // namespace

const lupine_client_bundle_payload *
lupine_client_bundle_lookup(const lupine_client_bundle_registry *registry,
                            const std::string &platform) {
  if (registry == nullptr) {
    return nullptr;
  }
  for (size_t i = 0; i < registry->entry_count; ++i) {
    const lupine_client_bundle_entry &entry = registry->entries[i];
    if (entry.platform != nullptr && entry.payload != nullptr &&
        platform == entry.platform) {
      return entry.payload;
    }
  }
  return nullptr;
}

bool lupine_client_bundle_request_platform(const std::string &path,
                                           std::string *platform) {
  if (platform == nullptr || path.compare(0, sizeof(kClientBundlePrefix) - 1,
                                          kClientBundlePrefix) != 0) {
    return false;
  }
  std::string candidate = path.substr(sizeof(kClientBundlePrefix) - 1);
  if (!platform_supported(candidate)) {
    return false;
  }
  *platform = std::move(candidate);
  return true;
}
