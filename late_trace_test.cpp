#include "lupine_log.h"

#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iterator>
#include <string>

static const char *path;

static void verify() {
  std::ifstream input(path);
  std::string text((std::istreambuf_iterator<char>(input)),
                   std::istreambuf_iterator<char>());
  if (text != "early trace\nlate destructor trace\n") {
    std::fputs("late trace was lost\n", stderr);
    std::abort();
  }
}

static void late_destructor() { LUPINE_TRACE_LOG("late destructor trace"); }

int main(int argc, char **argv) {
  if (argc != 2) {
    return 1;
  }
  path = argv[1];
  std::remove(path);
#ifdef _WIN32
  if (_putenv_s("LUPINE_TRACE", path) != 0) {
#else
  if (setenv("LUPINE_TRACE", path, 1) != 0) {
#endif
    return 1;
  }
  // A dylib destructor registered before the first trace runs after the
  // logger's ordinary function-local static destructors.
  std::atexit(verify);
  std::atexit(late_destructor);
  LUPINE_TRACE_LOG("early trace");
}
