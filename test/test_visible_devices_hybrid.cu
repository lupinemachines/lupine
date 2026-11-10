// LUPINE_TEST_CONNECTIONS 2
// When both local GPUs and remote Lupine servers are present (or when multiple
// local GPUs are accessed through the Lupine driver shim), CUDA_VISIBLE_DEVICES
// must filter the unified device table once rather than being applied first by
// the underlying local libcuda.so.1 and a second time by
// lupine_apply_visible_devices.
#include <cuda.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <sys/wait.h>
#include <unistd.h>
#include <vector>

namespace {

std::string uuid_hex(const CUuuid &uuid) {
  static const char kDigits[] = "0123456789abcdef";
  std::string out;
  out.reserve(32);
  for (int i = 0; i < 16; ++i) {
    unsigned char b = static_cast<unsigned char>(uuid.bytes[i]);
    out.push_back(kDigits[(b >> 4) & 0xf]);
    out.push_back(kDigits[b & 0xf]);
  }
  return out;
}

int query_visible_child(const char *cvd, std::vector<std::string> *uuids_out) {
  int pipefd[2] = {-1, -1};
  if (pipe(pipefd) != 0) {
    return -1;
  }
  pid_t pid = fork();
  if (pid < 0) {
    close(pipefd[0]);
    close(pipefd[1]);
    return -1;
  }
  if (pid == 0) {
    close(pipefd[0]);
    if (cvd == nullptr) {
      unsetenv("CUDA_VISIBLE_DEVICES");
    } else {
      setenv("CUDA_VISIBLE_DEVICES", cvd, 1);
    }
    if (cuInit(0) != CUDA_SUCCESS) {
      _exit(2);
    }
    // Verify CUDA_VISIBLE_DEVICES remains intact in the process environment.
    const char *after = getenv("CUDA_VISIBLE_DEVICES");
    if ((cvd == nullptr) != (after == nullptr) ||
        (cvd != nullptr && std::strcmp(cvd, after) != 0)) {
      _exit(3);
    }
    int count = 0;
    if (cuDeviceGetCount(&count) != CUDA_SUCCESS) {
      _exit(4);
    }
    std::string payload = std::to_string(count) + "\n";
    for (int i = 0; i < count; ++i) {
      CUdevice dev = 0;
      CUuuid uuid = {};
      int pci_bus = -1;
      int pci_dev = -1;
      int host_reg = -1;
      if (cuDeviceGet(&dev, i) != CUDA_SUCCESS ||
          cuDeviceGetUuid_v2(&uuid, dev) != CUDA_SUCCESS ||
          cuDeviceGetAttribute(&pci_bus, CU_DEVICE_ATTRIBUTE_PCI_BUS_ID, dev) !=
              CUDA_SUCCESS ||
          cuDeviceGetAttribute(&pci_dev, CU_DEVICE_ATTRIBUTE_PCI_DEVICE_ID,
                               dev) != CUDA_SUCCESS ||
          cuDeviceGetAttribute(
              &host_reg,
              CU_DEVICE_ATTRIBUTE_CAN_USE_HOST_POINTER_FOR_REGISTERED_MEM,
              dev) != CUDA_SUCCESS) {
        _exit(5);
      }
      payload += uuid_hex(uuid) + ":" + std::to_string(pci_bus) + ":" +
                 std::to_string(pci_dev) + ":" + std::to_string(host_reg) +
                 "\n";
    }
    (void)write(pipefd[1], payload.data(), payload.size());
    close(pipefd[1]);
    _exit(0);
  }
  close(pipefd[1]);
  std::string buf;
  char tmp[256];
  ssize_t n = 0;
  while ((n = read(pipefd[0], tmp, sizeof(tmp))) > 0) {
    buf.append(tmp, static_cast<size_t>(n));
  }
  close(pipefd[0]);
  int status = 0;
  if (waitpid(pid, &status, 0) < 0 || !WIFEXITED(status) ||
      WEXITSTATUS(status) != 0) {
    return -1;
  }
  uuids_out->clear();
  size_t pos = buf.find('\n');
  if (pos == std::string::npos) {
    return -1;
  }
  int count = std::atoi(buf.substr(0, pos).c_str());
  size_t cur = pos + 1;
  while (cur < buf.size()) {
    size_t nl = buf.find('\n', cur);
    if (nl == std::string::npos) {
      break;
    }
    uuids_out->push_back(buf.substr(cur, nl - cur));
    cur = nl + 1;
  }
  return count;
}

} // namespace

int main() {
  std::vector<std::string> baseline;
  int total = query_visible_child(nullptr, &baseline);
  if (total < 0) {
    std::fprintf(stderr, "FAIL: baseline device enumeration failed\n");
    return EXIT_FAILURE;
  }
  if (total < 2) {
    std::printf("SKIP: need >= 2 visible fleet devices (got %d)\n", total);
    return EXIT_SUCCESS;
  }

  int failures = 0;

  // Selecting ordinal 1 must yield 1 device matching baseline[1].
  std::vector<std::string> only_second;
  int cnt1 = query_visible_child("1", &only_second);
  if (cnt1 != 1 || only_second.size() != 1 || only_second[0] != baseline[1]) {
    std::fprintf(stderr, "FAIL: CUDA_VISIBLE_DEVICES=1 returned %d devices\n",
                 cnt1);
    ++failures;
  }

  // Reordering "1,0" must yield [baseline[1], baseline[0]].
  std::vector<std::string> swapped;
  int cnt_swap = query_visible_child("1,0", &swapped);
  if (cnt_swap != 2 || swapped.size() != 2 || swapped[0] != baseline[1] ||
      swapped[1] != baseline[0]) {
    std::fprintf(stderr,
                 "FAIL: CUDA_VISIBLE_DEVICES=1,0 returned %d devices\n",
                 cnt_swap);
    ++failures;
  }

  // Empty CUDA_VISIBLE_DEVICES must yield 0 devices.
  std::vector<std::string> none;
  int cnt_empty = query_visible_child("", &none);
  if (cnt_empty != 0 || !none.empty()) {
    std::fprintf(stderr, "FAIL: CUDA_VISIBLE_DEVICES='' returned %d devices\n",
                 cnt_empty);
    ++failures;
  }

  if (failures != 0) {
    return EXIT_FAILURE;
  }
  std::printf("PASS: CUDA_VISIBLE_DEVICES filters unified %d-device table once\n",
              total);
  return EXIT_SUCCESS;
}
