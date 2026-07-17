#include "server_checkpoint.h"

#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>

#include <signal.h>
#include <sys/socket.h>
#include <unistd.h>

namespace {

bool expect(bool condition, const char *message) {
  if (!condition) {
    std::cerr << "FAIL: " << message << '\n';
  }
  return condition;
}

} // namespace

int main(int argc, char **argv) {
  if (argc != 3) {
    std::cerr << "usage: " << argv[0] << " PROVIDER present|missing\n";
    return 2;
  }
  bool expect_provider = std::string(argv[2]) == "present";

  char directory_template[] = "/tmp/lupine-checkpoint-test.XXXXXX";
  char *directory = mkdtemp(directory_template);
  if (!expect(directory != nullptr, "failed to create test directory")) {
    return 1;
  }
  std::string log_path = std::string(directory) + "/provider.log";
  setenv("LUPINE_CHECKPOINT_DIR", directory, 1);
  setenv("LUPINE_CHECKPOINT_LIBRARY", argv[1], 1);
  setenv("LUPINE_CHECKPOINT_TEST_LOG", log_path.c_str(), 1);

  int sockets[2] = {-1, -1};
  if (!expect(socketpair(AF_UNIX, SOCK_STREAM, 0, sockets) == 0,
              "failed to create socket pair") ||
      !expect(lupine_server_checkpoint_child_start(sockets[0]),
              "failed to start checkpoint shutdown shell")) {
    return 1;
  }

  if (!expect(kill(getpid(), SIGTERM) == 0, "failed to deliver SIGTERM")) {
    return 1;
  }

  char byte = 0;
  if (!expect(read(sockets[1], &byte, sizeof(byte)) == 0,
              "SIGTERM did not shut down the connection")) {
    return 1;
  }
  if (!expect(lupine_server_checkpoint_child_finish() == 0,
              "checkpoint shutdown failed")) {
    return 1;
  }

  std::ifstream log(log_path);
  std::stringstream contents;
  contents << log.rdbuf();
  std::string expected;
  if (expect_provider) {
    expected = "start\ncheckpoint " + std::string(directory) + " " +
               std::to_string(static_cast<unsigned long long>(getpid())) +
               "\nstop\n";
  }
  bool passed = expect(contents.str() == expected,
                       "provider did not receive the expected lifecycle");
  close(sockets[0]);
  close(sockets[1]);
  return passed ? 0 : 1;
}
