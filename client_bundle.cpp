#include "client_bundle.h"

#include <array>
#include <fstream>
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

bool read_line(const std::string &path, std::string *value) {
  std::ifstream input(path, std::ios::binary);
  if (!input || !std::getline(input, *value)) {
    return false;
  }
  if (!value->empty() && value->back() == '\r') {
    value->pop_back();
  }
  return !value->empty() && value->find('\r') == std::string::npos &&
         value->find('\n') == std::string::npos;
}

bool valid_etag(const std::string &etag) {
  if (etag.compare(0, 8, "\"sha256:") != 0 || etag.size() != 73 ||
      etag.back() != '\"') {
    return false;
  }
  for (size_t i = 8; i + 1 < etag.size(); ++i) {
    if (!((etag[i] >= '0' && etag[i] <= '9') ||
          (etag[i] >= 'a' && etag[i] <= 'f'))) {
      return false;
    }
  }
  return true;
}

bool valid_content_digest(const std::string &digest) {
  if (digest.compare(0, 9, "sha-256=:") != 0 || digest.size() <= 10 ||
      digest.back() != ':') {
    return false;
  }
  for (size_t i = 9; i + 1 < digest.size(); ++i) {
    char c = digest[i];
    if (!((c >= '0' && c <= '9') || (c >= 'A' && c <= 'Z') ||
          (c >= 'a' && c <= 'z') || c == '+' || c == '/' || c == '=')) {
      return false;
    }
  }
  return true;
}

} // namespace

bool lupine_client_bundle_lookup(const char *root, const std::string &platform,
                                 lupine_client_bundle *bundle) {
  if (root == nullptr || root[0] == '\0' || bundle == nullptr ||
      !platform_supported(platform)) {
    return false;
  }

  std::string base(root);
  if (!base.empty() && base.back() != '/' && base.back() != '\\') {
    base += '/';
  }
  base += platform;
  base += "/client.zip";

  std::ifstream input(base, std::ios::binary | std::ios::ate);
  if (!input) {
    return false;
  }
  std::streamoff size = input.tellg();
  if (size < 0) {
    return false;
  }

  lupine_client_bundle found;
  found.path = base;
  found.size = static_cast<uint64_t>(size);
  if (!read_line(base + ".etag", &found.etag) || !valid_etag(found.etag) ||
      !read_line(base + ".digest", &found.content_digest) ||
      !valid_content_digest(found.content_digest)) {
    return false;
  }

  *bundle = std::move(found);
  return true;
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
