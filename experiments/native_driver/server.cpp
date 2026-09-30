#include "wire.h"
// Isolated raw-device backend for the native-driver experiment. No libcuda,
// CUDA API object table, or client pointer substitution is used here.
// Requires permission to handle userfaultfd faults caused by kernel accesses.
#include <array>
#include <atomic>
#include <cstring>
#include <fcntl.h>
#include <linux/userfaultfd.h>
#include <map>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <poll.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <sys/socket.h>
#include <sys/syscall.h>
#include <thread>
#include <vector>

static void check(bool ok, const char *what) {
  if (!ok) {
    std::perror(what);
    std::exit(2);
  }
}
struct DeviceBlock {
  Range range;
  bool mmio;
  std::vector<unsigned char> image;
};
static void publish(int peer, std::vector<DeviceBlock> &devices) {
  for (auto &block : devices) {
    size_t length = block.image.size(), offset = 0;
    if (block.mmio) {
      length = 8;
      offset = 0x80;
    }
    // Compare a single snapshot. GPU writes, including the continuously
    // advancing MMIO timer, can otherwise change between the equality
    // test and the run scan and leave the scan stuck on an empty run.
    std::vector<unsigned char> current(length);
    std::memcpy(current.data(),
                reinterpret_cast<void *>(block.range.begin + offset), length);
    changed_bytes(block.image.data() + offset, current.data(), length,
                  [&](size_t begin, size_t size) {
                    send_message(peer, {CHANGE, uint32_t(size),
                                        block.range.begin + offset + begin, 0});
                    transfer(peer, block.image.data() + offset + begin, size,
                             true);
                    transfer(peer, current.data() + begin, size, true);
                    std::memcpy(block.image.data() + offset + begin,
                                current.data() + begin, size);
                  });
  }
}
int main() {
  int listenfd = socket(AF_INET, SOCK_STREAM, 0);
  int reuse = 1;
  setsockopt(listenfd, SOL_SOCKET, SO_REUSEADDR, &reuse, sizeof(reuse));
  sockaddr_in address{};
  address.sin_family = AF_INET;
  address.sin_port = htons(16130);
  address.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
  check(bind(listenfd, reinterpret_cast<sockaddr *>(&address),
             sizeof(address)) == 0,
        "bind");
  check(listen(listenfd, 1) == 0, "listen");
  int peer = accept(listenfd, nullptr, nullptr);
  check(peer >= 0, "accept");
  close(listenfd);
  int nodelay = 1;
  setsockopt(peer, IPPROTO_TCP, TCP_NODELAY, &nodelay, sizeof(nodelay));
  int uffd = syscall(SYS_userfaultfd, O_CLOEXEC | O_NONBLOCK);
  check(uffd >= 0, "userfaultfd");
  uffdio_api api{};
  api.api = UFFD_API;
  check(ioctl(uffd, UFFDIO_API, &api) == 0, "UFFDIO_API");
  std::vector<DeviceBlock> devices;
  for (;;) {
    Message request = receive_message(peer);
    if (request.kind == OPEN) {
      char path[256]{};
      check(request.count < sizeof(path), "path size");
      transfer(peer, path, request.count, false);
      int native = open(path, request.b | O_CLOEXEC);
      int error = native < 0 ? errno : 0;
      if (native >= 0) {
        check(dup2(native, request.a) == int(request.a), "mirror fd");
        if (native != int(request.a))
          close(native);
      }
      std::fprintf(stderr, "OPEN %s fd=%llu error=%d\n", path,
                   (unsigned long long)request.a, error);
      send_message(peer,
                   {DONE, uint32_t(error), uint64_t(native >= 0 ? 0 : -1), 0});
      continue;
    }
    if (request.kind == CLOSE) {
      int result = close(request.a);
      send_message(
          peer, {DONE, uint32_t(result < 0 ? errno : 0), uint64_t(result), 0});
      continue;
    }
    if (request.kind == MAP) {
      Mapping info{};
      transfer(peer, &info, sizeof(info), false);
      void *p = mmap(
          reinterpret_cast<void *>(request.a), request.b, info.prot | PROT_READ,
          (info.flags & (MAP_SHARED | MAP_PRIVATE)) | MAP_FIXED_NOREPLACE,
          info.fd, info.offset);
      int error = p == MAP_FAILED ? errno : 0;
      std::fprintf(stderr, "MAP fd=%u addr=%llx len=%llu prot=%u error=%d\n",
                   info.fd, (unsigned long long)request.a,
                   (unsigned long long)request.b, info.prot, error);
      send_message(peer, {DONE, uint32_t(error), 0, 0});
      if (!error) {
        // The user-mode BAR contains MMIO registers. Read only its documented
        // timestamp pair, rather than touching every register in the aperture.
        uint32_t size = request.b == 65536 ? 8 : uint32_t(request.b);
        uint32_t offset = request.b == 65536 ? 0x80 : 0;
        send_message(peer, {PAGE, size, offset, 0});
        transfer(peer, static_cast<char *>(p) + offset, size, true);
        DeviceBlock block{{request.a, request.a + request.b},
                          request.b == 65536,
                          std::vector<unsigned char>(request.b)};
        std::memcpy(block.image.data() + offset,
                    static_cast<char *>(p) + offset, size);
        devices.push_back(std::move(block));
      }
      continue;
    }
    if (request.kind == UNMAP) {
      int result = munmap(reinterpret_cast<void *>(request.a), request.b);
      for (auto it = devices.begin(); it != devices.end(); ++it)
        if (it->range.begin == request.a) {
          devices.erase(it);
          break;
        }
      send_message(
          peer, {DONE, uint32_t(result < 0 ? errno : 0), uint64_t(result), 0});
      continue;
    }
    if (request.kind == SYNC) {
      std::vector<unsigned char> bytes(request.count);
      transfer(peer, bytes.data(), bytes.size(), false);
      if (request.b) {
        check(request.count == 4, "doorbell size");
        uint32_t word;
        std::memcpy(&word, bytes.data(), 4);
        std::atomic_thread_fence(std::memory_order_seq_cst);
        *reinterpret_cast<volatile uint32_t *>(request.a) = word;
      } else
        std::memcpy(reinterpret_cast<void *>(request.a), bytes.data(),
                    bytes.size());
      for (auto &block : devices)
        if (request.a >= block.range.begin &&
            request.a + request.count <= block.range.end)
          std::memcpy(block.image.data() + request.a - block.range.begin,
                      bytes.data(), bytes.size());
      continue;
    }
    if (request.kind == POLL) {
      std::vector<pollfd> fds(request.count);
      transfer(peer, fds.data(), fds.size() * sizeof(pollfd), false);
      int result = poll(fds.data(), fds.size(), 0);
      publish(peer, devices);
      send_message(
          peer, {DONE, uint32_t(result < 0 ? errno : 0), uint64_t(result), 0});
      transfer(peer, fds.data(), fds.size() * sizeof(pollfd), true);
      continue;
    }
    check(request.kind == IOCTL, "request kind");
    std::vector<Range> ranges(request.count);
    transfer(peer, ranges.data(), ranges.size() * sizeof(Range), false);
    uint64_t argument = 0;
    transfer(peer, &argument, sizeof(argument), false);
    int mapping_error = 0;
    std::vector<Range> mapped;
    for (Range range : ranges) {
      void *p = mmap(reinterpret_cast<void *>(range.begin),
                     range.end - range.begin, PROT_READ | PROT_WRITE,
                     MAP_PRIVATE | MAP_ANONYMOUS | MAP_FIXED_NOREPLACE, -1, 0);
      if (p != reinterpret_cast<void *>(range.begin)) {
        mapping_error = errno;
        break;
      }
      mapped.push_back(range);
      uffdio_register reg{};
      reg.range.start = range.begin;
      reg.range.len = range.end - range.begin;
      reg.mode = UFFDIO_REGISTER_MODE_MISSING;
      check(ioctl(uffd, UFFDIO_REGISTER, &reg) == 0, "register mirror");
    }
    std::map<uint64_t, std::array<unsigned char, 4096>> pages;
    std::atomic<bool> finished{false};
    int result = -1, error = mapping_error;
    if (!mapping_error) {
      // Long-term page pinning cannot wait for a missing-page handler under
      // every GUP calling convention. Populate the entire OS-descriptor block
      // before pinning, then retain that block for asynchronous GPU accesses.
      if (request.b == 0xc0384627) {
        // NVOS02_PARAMETERS is 56 bytes and can straddle an argument page.
        // Its OS-descriptor backing must still be populated before GUP pins it.
        uffdio_copy copy{};
        copy.len = 4096;
        for (uint64_t page = argument & ~uint64_t{4095}; page < argument + 56;
             page += 4096) {
          auto &bytes = pages[page];
          send_message(peer, {PAGE, 4096, page, 0});
          transfer(peer, bytes.data(), bytes.size(), false);
          copy.dst = page;
          copy.src = reinterpret_cast<uint64_t>(bytes.data());
          check(ioctl(uffd, UFFDIO_COPY, &copy) == 0, "prefault descriptor");
        }
        const auto *words = reinterpret_cast<const uint32_t *>(argument);
        uint64_t base = 0, limit = 0;
        std::memcpy(&base, words + 6, 8);
        std::memcpy(&limit, words + 8, 8);
        if (words[3] == 0x71 && base && limit < 64 * 1024 * 1024) {
          uint64_t begin = base & ~uint64_t{4095};
          uint64_t end = (base + limit + 4096) & ~uint64_t{4095};
          bool exists = false;
          for (auto &d : devices)
            if (begin >= d.range.begin && end <= d.range.end)
              exists = true;
          if (!exists) {
            bool mirrored = false;
            for (Range r : mapped)
              if (begin >= r.begin && end <= r.end)
                mirrored = true;
            check(mirrored, "registered CPU range is readable");
            std::vector<unsigned char> image(end - begin);
            send_message(peer, {PAGE, uint32_t(image.size()), begin, 0});
            transfer(peer, image.data(), image.size(), false);
            copy.dst = begin;
            copy.src = reinterpret_cast<uint64_t>(image.data());
            copy.len = image.size();
            check(ioctl(uffd, UFFDIO_COPY, &copy) == 0,
                  "prefault pinned block");
            uffdio_range range{begin, end - begin};
            check(ioctl(uffd, UFFDIO_UNREGISTER, &range) == 0,
                  "unregister pinned block faults");
            // This backend owns the GPU-pinned physical backing. A future
            // handoff must migrate it alongside GPU memory, at this saved VA,
            // and recreate the kernel's pin/mapping resources separately.
            devices.push_back({{begin, end}, false, std::move(image)});
            send_message(peer, {REGISTER, uint32_t(end - begin), begin, 0});
            std::fprintf(stderr, "PINNED BLOCK addr=%llx len=%llu\n",
                         (unsigned long long)begin,
                         (unsigned long long)(end - begin));
          }
        }
      }
      std::thread handler([&] {
        while (!finished) {
          pollfd polling{uffd, POLLIN, 0};
          int ready = poll(&polling, 1, 10);
          if (ready <= 0)
            continue;
          uffd_msg fault{};
          check(read(uffd, &fault, sizeof(fault)) == sizeof(fault),
                "fault event");
          check(fault.event == UFFD_EVENT_PAGEFAULT, "page fault");
          uint64_t page = fault.arg.pagefault.address & ~uint64_t{4095};
          auto &bytes = pages[page];
          send_message(peer, {PAGE, 4096, page, 0});
          transfer(peer, bytes.data(), bytes.size(), false);
          uffdio_copy copy{};
          copy.dst = page;
          copy.src = reinterpret_cast<uint64_t>(bytes.data());
          copy.len = 4096;
          check(ioctl(uffd, UFFDIO_COPY, &copy) == 0, "populate mirror");
        }
      });
      result = ioctl(request.a, request.b, reinterpret_cast<void *>(argument));
      error = result < 0 ? errno : 0;
      finished = true;
      handler.join();
      publish(peer, devices);
      for (auto &entry : pages) {
        if (std::memcmp(entry.second.data(),
                        reinterpret_cast<void *>(entry.first), 4096) == 0)
          continue;
        send_message(peer, {CHANGE, 4096, entry.first, 0});
        transfer(peer, entry.second.data(), 4096, true);
        transfer(peer, reinterpret_cast<void *>(entry.first), 4096, true);
      }
    }
    std::fprintf(
        stderr,
        "IOCTL fd=%llu cmd=%llx arg=%llx pages=%zu result=%d error=%d\n",
        (unsigned long long)request.a, (unsigned long long)request.b,
        (unsigned long long)argument, pages.size(), result, error);
    for (Range range : mapped) {
      std::vector<Range> pieces{range};
      for (const auto &block : devices)
        subtract_range(pieces, block.range);
      for (Range r : pieces)
        munmap(reinterpret_cast<void *>(r.begin), r.end - r.begin);
    }
    send_message(peer, {DONE, uint32_t(error), uint64_t(result), 0});
  }
}
