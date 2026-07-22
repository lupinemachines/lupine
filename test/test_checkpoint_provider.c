#include "checkpoint_provider.h"

#include <stdio.h>
#include <stdlib.h>

static void append_event(const char *event, const char *connection_id) {
  const char *path = getenv("LUPINE_CHECKPOINT_TEST_LOG");
  if (path == NULL) {
    return;
  }
  FILE *file = fopen(path, "a");
  if (file == NULL) {
    return;
  }
  fprintf(file, "%s", event);
  if (connection_id != NULL) {
    fprintf(file, " %s", connection_id);
  }
  fputc('\n', file);
  fclose(file);
}

static int test_start(void) {
  append_event("start", NULL);
  return 0;
}

static int test_restore(const char *connection_id) {
  append_event("restore", connection_id);
  return 0;
}

static int test_checkpoint(const char *connection_id) {
  append_event("checkpoint",
               connection_id == NULL ? "<unnamed>" : connection_id);
  return 0;
}

static void test_stop(void) { append_event("stop", NULL); }

static const lupine_checkpoint_provider_v2 provider = {
    sizeof(provider), LUPINE_CHECKPOINT_PROVIDER_ABI_VERSION,
    test_start,       test_restore,
    test_checkpoint,  test_stop};

const lupine_checkpoint_provider_v2 *lupinecr_get_lupine_provider_v2(void) {
  return &provider;
}
