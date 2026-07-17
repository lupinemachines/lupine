#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <thread>

#include <arpa/inet.h>
#include <signal.h>
#include <sys/socket.h>
#include <sys/wait.h>
#include <unistd.h>

namespace {

int reserve_port() {
  int socket_fd = socket(AF_INET, SOCK_STREAM, 0);
  if (socket_fd < 0) {
    return -1;
  }
  sockaddr_in address = {};
  address.sin_family = AF_INET;
  address.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
  address.sin_port = 0;
  if (bind(socket_fd, reinterpret_cast<sockaddr *>(&address),
           sizeof(address)) != 0) {
    close(socket_fd);
    return -1;
  }
  socklen_t size = sizeof(address);
  int port =
      getsockname(socket_fd, reinterpret_cast<sockaddr *>(&address), &size) == 0
          ? ntohs(address.sin_port)
          : -1;
  close(socket_fd);
  return port;
}

void terminate_process_group(pid_t server) {
  (void)kill(-server, SIGKILL);
  (void)waitpid(server, nullptr, 0);
}

bool connection_child_started(pid_t server) {
  std::ifstream children("/proc/" + std::to_string(server) + "/task/" +
                         std::to_string(server) + "/children");
  pid_t child = 0;
  return static_cast<bool>(children >> child);
}

} // namespace

int main(int argc, char **argv) {
  if (argc != 2 && argc != 3) {
    std::cerr << "usage: " << argv[0] << " SERVER [PROVIDER]\n";
    return 2;
  }

  char directory_template[] = "/tmp/lupine-server-sigterm-test.XXXXXX";
  char *checkpoint_directory = nullptr;
  std::string checkpoint_log;
  if (argc == 3) {
    checkpoint_directory = mkdtemp(directory_template);
    if (checkpoint_directory == nullptr) {
      std::cerr << "FAIL: could not create checkpoint test directory\n";
      return 1;
    }
    checkpoint_log = std::string(checkpoint_directory) + "/provider.log";
  }

  int port = reserve_port();
  if (port < 0) {
    std::cerr << "FAIL: could not reserve a test port\n";
    return 1;
  }

  pid_t server = fork();
  if (server < 0) {
    return 1;
  }
  if (server == 0) {
    (void)setpgid(0, 0);
    std::string port_text = std::to_string(port);
    setenv("LUPINE_PORT", port_text.c_str(), 1);
    if (checkpoint_directory != nullptr) {
      setenv("LUPINE_CHECKPOINT_DIR", checkpoint_directory, 1);
      setenv("LUPINE_CHECKPOINT_LIBRARY", argv[2], 1);
      setenv("LUPINE_CHECKPOINT_TEST_LOG", checkpoint_log.c_str(), 1);
    } else {
      unsetenv("LUPINE_CHECKPOINT_DIR");
    }
    execl(argv[1], argv[1], static_cast<char *>(nullptr));
    _exit(127);
  }
  (void)setpgid(server, server);

  int connection = -1;
  for (int attempt = 0; attempt < 100; ++attempt) {
    int early_status = 0;
    pid_t early_exit = waitpid(server, &early_status, WNOHANG);
    if (early_exit == server) {
      // CUDA stub libraries used by CPU-only builders do not necessarily
      // export every driver symbol required by the server executable.
      return WIFEXITED(early_status) && WEXITSTATUS(early_status) == 127 ? 77
                                                                         : 1;
    }
    connection = socket(AF_INET, SOCK_STREAM, 0);
    sockaddr_in address = {};
    address.sin_family = AF_INET;
    address.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    address.sin_port = htons(static_cast<uint16_t>(port));
    if (connection >= 0 &&
        connect(connection, reinterpret_cast<sockaddr *>(&address),
                sizeof(address)) == 0) {
      break;
    }
    if (connection >= 0) {
      close(connection);
      connection = -1;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(20));
  }
  bool child_started = false;
  if (connection >= 0) {
    for (int attempt = 0; attempt < 100; ++attempt) {
      if (connection_child_started(server)) {
        child_started = true;
        break;
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
  }
  if (!child_started || kill(server, SIGTERM) != 0) {
    terminate_process_group(server);
    std::cerr << "FAIL: server did not start a connection child\n";
    return 1;
  }

  int status = 0;
  for (int attempt = 0; attempt < 250; ++attempt) {
    pid_t result = waitpid(server, &status, WNOHANG);
    if (result == server) {
      close(connection);
      if (WIFEXITED(status) && WEXITSTATUS(status) == 0) {
        if (checkpoint_directory != nullptr) {
          std::ifstream log(checkpoint_log);
          std::stringstream contents;
          contents << log.rdbuf();
          std::string lifecycle = contents.str();
          std::string checkpoint_prefix =
              "start\ncheckpoint " + std::string(checkpoint_directory) + " ";
          if (lifecycle.rfind(checkpoint_prefix, 0) != 0 ||
              lifecycle.size() < checkpoint_prefix.size() + 7 ||
              lifecycle.substr(lifecycle.size() - 6) != "\nstop\n") {
            std::cerr << "FAIL: provider lifecycle was not completed\n";
            return 1;
          }
        }
        return 0;
      }
      std::cerr << "FAIL: server returned a shutdown error\n";
      return 1;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(20));
  }

  close(connection);
  terminate_process_group(server);
  std::cerr << "FAIL: server did not drain its connection child\n";
  return 1;
}
