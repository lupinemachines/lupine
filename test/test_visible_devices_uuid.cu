// CUDA_VISIBLE_DEVICES=GPU-<uuid> naming a remote device must select exactly
// that device. Resolving the UUID asks the server while the device table is
// being built, so a request the server's handler does not expect hangs cuInit
// forever; each child is therefore given a bounded wait.
#include <cuda.h>

#include <chrono>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <sys/wait.h>
#include <thread>
#include <unistd.h>
#include <vector>

namespace {

constexpr int kChildSeconds = 60;

std::string uuid_hex(const CUuuid &uuid) {
  static const char kDigits[] = "0123456789abcdef";
  std::string out;
  for (int i = 0; i < 16; ++i) {
    unsigned char b = static_cast<unsigned char>(uuid.bytes[i]);
    out.push_back(kDigits[(b >> 4) & 0xf]);
    out.push_back(kDigits[b & 0xf]);
  }
  return out;
}

std::string uuid_spec(const std::string &hex) {
  return "GPU-" + hex.substr(0, 8) + "-" + hex.substr(8, 4) + "-" +
         hex.substr(12, 4) + "-" + hex.substr(16, 4) + "-" + hex.substr(20);
}

bool enumerate_child(const char *cvd, std::vector<std::string> *uuids) {
  int pipefd[2];
  if (pipe(pipefd) != 0) {
    return false;
  }
  pid_t pid = fork();
  if (pid < 0) {
    return false;
  }
  if (pid == 0) {
    close(pipefd[0]);
    if (cvd == nullptr) {
      unsetenv("CUDA_VISIBLE_DEVICES");
    } else {
      setenv("CUDA_VISIBLE_DEVICES", cvd, 1);
    }
    int count = 0;
    if (cuInit(0) != CUDA_SUCCESS || cuDeviceGetCount(&count) != CUDA_SUCCESS) {
      _exit(2);
    }
    std::string payload;
    for (int i = 0; i < count; ++i) {
      CUdevice dev = 0;
      CUuuid uuid = {};
      if (cuDeviceGet(&dev, i) != CUDA_SUCCESS ||
          cuDeviceGetUuid_v2(&uuid, dev) != CUDA_SUCCESS) {
        _exit(3);
      }
      payload += uuid_hex(uuid) + "\n";
    }
    (void)!write(pipefd[1], payload.data(), payload.size());
    _exit(0);
  }
  close(pipefd[1]);

  int status = 0;
  pid_t done = 0;
  const auto deadline =
      std::chrono::steady_clock::now() + std::chrono::seconds(kChildSeconds);
  while ((done = waitpid(pid, &status, WNOHANG)) == 0 &&
         std::chrono::steady_clock::now() < deadline) {
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
  }
  if (done == 0) {
    kill(pid, SIGKILL);
    waitpid(pid, &status, 0);
    close(pipefd[0]);
    std::fprintf(stderr, "child with CUDA_VISIBLE_DEVICES=%s hung for %ds\n",
                 cvd ? cvd : "(unset)", kChildSeconds);
    return false;
  }
  std::string buf;
  char tmp[256];
  ssize_t n = 0;
  while ((n = read(pipefd[0], tmp, sizeof(tmp))) > 0) {
    buf.append(tmp, static_cast<size_t>(n));
  }
  close(pipefd[0]);
  if (done != pid || !WIFEXITED(status) || WEXITSTATUS(status) != 0) {
    std::fprintf(stderr, "child with CUDA_VISIBLE_DEVICES=%s failed (%d)\n",
                 cvd ? cvd : "(unset)", status);
    return false;
  }
  uuids->clear();
  for (size_t pos = 0, nl; (nl = buf.find('\n', pos)) != std::string::npos;
       pos = nl + 1) {
    uuids->push_back(buf.substr(pos, nl - pos));
  }
  return true;
}

} // namespace

int main() {
  std::vector<std::string> all;
  if (!enumerate_child(nullptr, &all)) {
    std::fprintf(stderr, "FAIL: unfiltered enumeration\n");
    return EXIT_FAILURE;
  }
  if (all.empty()) {
    std::printf("SKIP: no devices\n");
    return EXIT_SUCCESS;
  }

  int failures = 0;
  for (const std::string &uuid : all) {
    const std::string spec = uuid_spec(uuid);
    std::vector<std::string> selected;
    if (!enumerate_child(spec.c_str(), &selected) || selected.size() != 1 ||
        selected[0] != uuid) {
      std::fprintf(stderr, "FAIL: CUDA_VISIBLE_DEVICES=%s selected %zu\n",
                   spec.c_str(), selected.size());
      ++failures;
    }
  }
  if (failures != 0) {
    return EXIT_FAILURE;
  }
  std::printf("PASS: each of %zu devices selected by UUID\n", all.size());
  return EXIT_SUCCESS;
}
