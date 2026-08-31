#ifndef LUPINE_CLIENT_BUNDLE_H
#define LUPINE_CLIENT_BUNDLE_H

#include <stdint.h>
#include <string>

struct lupine_client_bundle {
  std::string path;
  std::string etag;
  std::string content_digest;
  uint64_t size = 0;
};

// Looks up one immutable client bundle below root. Platform names are the
// public protocol values (for example "linux/amd64"), not build-system tags.
// Returns false for an unsupported platform or an incomplete bundle entry.
bool lupine_client_bundle_lookup(const char *root, const std::string &platform,
                                 lupine_client_bundle *bundle);

// Extracts the platform from the versioned HTTP endpoint.
bool lupine_client_bundle_request_platform(const std::string &path,
                                           std::string *platform);

#endif
