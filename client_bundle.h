#ifndef LUPINE_CLIENT_BUNDLE_H
#define LUPINE_CLIENT_BUNDLE_H

#include <stddef.h>

#include <string>

struct lupine_client_bundle_chunk {
  const char *data;
  size_t size;
};

struct lupine_client_bundle_payload {
  const char *etag;
  const char *content_digest;
  const lupine_client_bundle_chunk *chunks;
  size_t chunk_count;
  size_t size;
};

struct lupine_client_bundle_entry {
  const char *platform;
  const lupine_client_bundle_payload *payload;
};

struct lupine_client_bundle_registry {
  const lupine_client_bundle_entry *entries;
  size_t entry_count;
};

extern const lupine_client_bundle_registry lupine_embedded_client_bundles;

const lupine_client_bundle_payload *
lupine_client_bundle_lookup(const lupine_client_bundle_registry *registry,
                            const std::string &platform);

bool lupine_client_bundle_request_platform(const std::string &path,
                                           std::string *platform);

#endif
