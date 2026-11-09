// Linux-only transport microbenchmark. See benchmark_h2_writer.md for build
// instructions, CSV columns, methodology, and limitations.
#include "rpc.h"
#include "test_platform.h"
#include <algorithm>
#include <array>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <sys/resource.h>
#include <thread>
#include <vector>
using Clock = std::chrono::steady_clock;
void check(bool ok) {
  if (!ok)
    std::abort();
}
double cpu(const rusage &r) {
  return r.ru_utime.tv_sec + r.ru_utime.tv_usec / 1e6 + r.ru_stime.tv_sec +
         r.ru_stime.tv_usec / 1e6;
}
int main(int argc, char **argv) {
  int gap = argc > 1 ? std::atoi(argv[1]) : 30;
  int count = argc > 2 ? std::atoi(argv[2]) : 20000;
  check(gap >= 0 && count > 0);
  conn_t client = {}, server = {};
  lupine_socket_t fds[2];
  check(lupine_test_connected_pair(fds));
  check(rpc_conn_init(&client, fds[0], 0) == 0);
  check(rpc_conn_init(&server, fds[1], 1) == 0);
  check(rpc_http2_client_init(&client) == 0);
  rpc_http2_client_start_heartbeat(&client);
  check(rpc_http2_server_init_with_metadata(&server, nullptr) == 0);
  std::vector<double> delivery;
  delivery.reserve(count);
  std::thread reader([&] {
    for (int i = 0; i < count; ++i) {
      std::array<char, 64> bytes{};
      check(rpc_http2_read_stream(&server, rpc_http2_dispatch_stream(&server),
                                  bytes.data(), bytes.size()) == 0);
      Clock::time_point sent;
      std::memcpy(&sent, bytes.data(), sizeof(sent));
      delivery.push_back(
          std::chrono::duration<double, std::micro>(Clock::now() - sent)
              .count());
    }
  });
  rusage before{}, after{}, caller_before{}, caller_after{};
  getrusage(RUSAGE_SELF, &before);
  getrusage(RUSAGE_THREAD, &caller_before);
  auto start = Clock::now();
  std::vector<double> enqueue;
  enqueue.reserve(count);
  for (int i = 0; i < count; ++i) {
    std::array<char, 64> bytes{};
    auto sent = Clock::now();
    std::memcpy(bytes.data(), &sent, sizeof(sent));
    std::vector<rpc_write_cursor> cursors{
        rpc_write_cursor(bytes.data(), bytes.size())};
    check(rpc_http2_write_stream(&client, rpc_http2_dispatch_stream(&client),
                                 cursors) == 0);
    auto done = Clock::now();
    enqueue.push_back(
        std::chrono::duration<double, std::micro>(done - sent).count());
    auto next = done + std::chrono::microseconds(gap);
    while (Clock::now() < next) {
#if defined(__aarch64__) || defined(__arm__)
      asm volatile("yield" ::: "memory");
#elif defined(__x86_64__) || defined(__i386__)
      __builtin_ia32_pause();
#endif
    }
  }
  getrusage(RUSAGE_THREAD, &caller_after);
  reader.join();
  getrusage(RUSAGE_SELF, &after);
  auto elapsed = std::chrono::duration<double>(Clock::now() - start).count();
  std::sort(enqueue.begin(), enqueue.end());
  std::sort(delivery.begin(), delivery.end());
  printf("%d,%d,%.6f,%.6f,%.6f,%ld,%ld,%.3f,%.3f,%.3f,%.3f\n", gap, count,
         elapsed, cpu(after) - cpu(before),
         cpu(caller_after) - cpu(caller_before),
         after.ru_nvcsw - before.ru_nvcsw, after.ru_nivcsw - before.ru_nivcsw,
         enqueue[count / 2], enqueue[count * 99 / 100], delivery[count / 2],
         delivery[count * 99 / 100]);
  rpc_conn_destroy(&client);
  rpc_conn_destroy(&server);
  lupine_socket_close(fds[0]);
  lupine_socket_close(fds[1]);
}
