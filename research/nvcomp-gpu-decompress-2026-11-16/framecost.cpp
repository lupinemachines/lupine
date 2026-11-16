// Wire size and single-core compress cost of the compression layouts:
// today's bulk lanes (one continuous zstd -1 stream per lane, windowLog 19,
// 4 lanes taking 4 MiB chunks round robin, flushed at each chunk) versus
// independent zstd -1 / LZ4 frames of 64 KiB, 256 KiB, 1 MiB.
#include <zstd.h>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <string>
#include <vector>

extern "C" int LZ4_compress_fast(const char *, char *, int, int, int);

static double cpu() {
  timespec t;
  clock_gettime(CLOCK_THREAD_CPUTIME_ID, &t);
  return t.tv_sec + 1e-9 * t.tv_nsec;
}

int main(int argc, char **argv) {
  FILE *f = fopen(argv[1], "rb");
  fseek(f, 0, SEEK_END);
  size_t n = ftell(f);
  fseek(f, 0, SEEK_SET);
  std::vector<char> src(n);
  if (fread(src.data(), 1, n, f) != n) return 1;
  fclose(f);
  const size_t L = 4 << 20, lanes = 4;
  std::vector<char> out(ZSTD_compressBound(16 << 20));
  const char *name = strrchr(argv[1], '/') ? strrchr(argv[1], '/') + 1 : argv[1];
  double gib = n / double(1 << 30);

  {
    std::vector<ZSTD_CCtx *> cc(lanes);
    for (auto &c : cc) {
      c = ZSTD_createCCtx();
      ZSTD_CCtx_setParameter(c, ZSTD_c_compressionLevel, -1);
      ZSTD_CCtx_setParameter(c, ZSTD_c_windowLog, 19);
    }
    size_t total = 0;
    double c0 = cpu();
    for (size_t u = 0, k = 0; u < n; u += L, k++) {
      ZSTD_CCtx *c = cc[k % lanes];
      size_t len = std::min(L, n - u);
      for (size_t o = 0; o < len; o += 256 << 10) {
        ZSTD_inBuffer in{src.data() + u + o, std::min<size_t>(256 << 10, len - o), 0};
        bool last = o + (256 << 10) >= len;
        size_t rem;
        do {
          ZSTD_outBuffer ob{out.data(), out.size(), 0};
          rem = ZSTD_compressStream2(c, &ob, &in, last ? ZSTD_e_flush : ZSTD_e_continue);
          total += ob.pos;
        } while (in.pos < in.size || (last && rem));
      }
    }
    double s = cpu() - c0;
    printf("FRAMECOST data=%s layout=stream-zstd-1 ratio=%.3f wire_mib=%.1f cpu_s_per_gib=%.3f\n", name,
           double(n) / total, total / 1048576.0, s / gib);
  }
  for (const char *codec : {"zstd-1", "lz4"}) {
    for (size_t fb : {size_t(64) << 10, size_t(256) << 10, size_t(1) << 20}) {
      ZSTD_CCtx *c = ZSTD_createCCtx();
      size_t total = 0;
      double c0 = cpu();
      for (size_t u = 0; u < n; u += L) {
        size_t len = std::min(L, n - u), frames = (len + fb - 1) / fb;
        size_t chunk = 24 + 4 * frames;
        for (size_t o = 0; o < len; o += fb) {
          size_t m = std::min(fb, len - o);
          size_t r = codec[0] == 'z'
                         ? ZSTD_compressCCtx(c, out.data(), out.size(), src.data() + u + o, m, -1)
                         : (size_t)LZ4_compress_fast(src.data() + u + o, out.data(), (int)m, (int)out.size(), 1);
          chunk += (r + 7) & ~size_t(7);
        }
        // The prototype sends a chunk raw when framing does not leave room for
        // the server's 32 B/frame of arrays.
        total += chunk + 32 * frames < len ? chunk : len;
      }
      double s = cpu() - c0;
      printf("FRAMECOST data=%s layout=frames-%s-%zuK ratio=%.3f wire_mib=%.1f cpu_s_per_gib=%.3f\n", name, codec,
             fb >> 10, double(n) / total, total / 1048576.0, s / gib);
      ZSTD_freeCCtx(c);
    }
  }
  return 0;
}
