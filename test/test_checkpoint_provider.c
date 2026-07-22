#include "checkpoint_provider.h"

#include <stdio.h>
#include <stdlib.h>

static void append_event(const char *event, const char *directory,
                         uint64_t connection_id) {
  const char *path = getenv("LUPINE_CHECKPOINT_TEST_LOG");
  if (path == NULL) {
    return;
  }
  FILE *file = fopen(path, "a");
  if (file == NULL) {
    return;
  }
  fprintf(file, "%s", event);
  if (directory != NULL) {
    fprintf(file, " %s %llu", directory, (unsigned long long)connection_id);
  }
  fputc('\n', file);
  fclose(file);
}

static int test_start(void) {
  append_event("start", NULL, 0);
  return 0;
}

static int test_checkpoint(const char *directory, uint64_t connection_id) {
  append_event("checkpoint", directory, connection_id);
  return 0;
}

static void test_stop(void) { append_event("stop", NULL, 0); }

static const lupine_checkpoint_provider_v1 provider = {
    sizeof(provider), LUPINE_CHECKPOINT_PROVIDER_ABI_VERSION, test_start,
    test_checkpoint, test_stop};

const lupine_checkpoint_provider_v1 *lupinecr_get_lupine_provider_v1(void) {
  return &provider;
}
