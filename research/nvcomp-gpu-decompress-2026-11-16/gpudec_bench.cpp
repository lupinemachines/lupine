// Standalone: CPU-compressed independent chunks -> (a) CPU decompress + HtoD,
// (b) HtoD of compressed bytes + nvCOMP GPU decompress. No Lupine involved.
#include <cuda.h>
#include <nvcomp/gdeflate.h>
#include <nvcomp/lz4.h>
#include <nvcomp/zstd.h>
#include <sys/resource.h>
#include <zstd.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

extern "C" {
int LZ4_compress_fast(const char *, char *, int, int, int);
int LZ4_compressBound(int);
int LZ4_decompress_safe(const char *, char *, int, int);
struct libdeflate_gdeflate_out_page {
  void *data;
  size_t nbytes;
};
struct libdeflate_gdeflate_in_page {
  const void *data;
  size_t nbytes;
};
void *libdeflate_alloc_gdeflate_compressor(int);
size_t libdeflate_gdeflate_compress_bound(void *, size_t, size_t *);
size_t libdeflate_gdeflate_compress(void *, const void *, size_t,
                                    libdeflate_gdeflate_out_page *, size_t);
void *libdeflate_alloc_gdeflate_decompressor();
int libdeflate_gdeflate_decompress(void *, libdeflate_gdeflate_in_page *,
                                   size_t, void *, size_t, size_t *);
}

#define CK(x)                                                                  \
  do {                                                                         \
    CUresult r_ = (x);                                                         \
    if (r_ != CUDA_SUCCESS) {                                                  \
      const char *s_;                                                          \
      cuGetErrorString(r_, &s_);                                               \
      fprintf(stderr, "%s:%d %s -> %s\n", __FILE__, __LINE__, #x, s_);         \
      exit(1);                                                                 \
    }                                                                          \
  } while (0)
#define NK(x)                                                                  \
  do {                                                                         \
    nvcompStatus_t s_ = (x);                                                   \
    if (s_ != nvcompSuccess) {                                                 \
      fprintf(stderr, "%s:%d %s -> nvcomp %d\n", __FILE__, __LINE__, #x,       \
              (int)s_);                                                        \
      exit(1);                                                                 \
    }                                                                          \
  } while (0)

enum Codec { NONE, ZSTD, LZ4, GDEFLATE };
static Codec codec;
static int level = -1;
static size_t C = 65536;   // independent sub-chunk (nvCOMP chunk)
static size_t L = 4 << 20; // Lupine bulk chunk
static size_t ALIGN = 16;
static size_t N;
static std::vector<char> src;

struct Sub {
  size_t off, csize,
      usize; // offset into wire, compressed size, uncompressed size
};
struct LChunk {
  size_t uoff, ulen; // range of destination
  size_t woff, wlen; // range of wire buffer
  bool raw;
  std::vector<Sub> subs; // offsets relative to woff
};
static std::vector<LChunk> chunks;
static std::vector<char> wire;

static double now() {
  return std::chrono::duration<double>(
             std::chrono::steady_clock::now().time_since_epoch())
      .count();
}
static double cpu_s() {
  rusage ru;
  getrusage(RUSAGE_SELF, &ru);
  return ru.ru_utime.tv_sec + ru.ru_stime.tv_sec +
         1e-6 * (ru.ru_utime.tv_usec + ru.ru_stime.tv_usec);
}

static size_t compress_sub(const char *in, size_t n, char *out, size_t cap,
                           ZSTD_CCtx *zc, void *gd) {
  switch (codec) {
  case ZSTD: {
    size_t r = ZSTD_compressCCtx(zc, out, cap, in, n, level);
    if (ZSTD_isError(r))
      abort();
    return r;
  }
  case LZ4:
    return (size_t)LZ4_compress_fast(in, out, (int)n, (int)cap, 1);
  case GDEFLATE: {
    libdeflate_gdeflate_out_page pg{out, cap};
    size_t r = libdeflate_gdeflate_compress(gd, in, n, &pg, 1);
    if (!r)
      abort();
    return r;
  }
  default:
    memcpy(out, in, n);
    return n;
  }
}

static double compress_all() {
  wire.resize(N + N / 8 + 64 * 1024 * 1024);
  ZSTD_CCtx *zc = ZSTD_createCCtx();
  void *gd = codec == GDEFLATE
                 ? libdeflate_alloc_gdeflate_compressor(level < 1 ? 1 : level)
                 : nullptr;
  size_t w = 0;
  double t0 = now();
  for (size_t u = 0; u < N; u += L) {
    LChunk lc{u, std::min(L, N - u), w, 0, false, {}};
    size_t start = w;
    for (size_t o = 0; o < lc.ulen; o += C) {
      size_t n = std::min(C, lc.ulen - o);
      size_t cs =
          compress_sub(&src[u + o], n, &wire[w], wire.size() - w, zc, gd);
      lc.subs.push_back({w - start, cs, n});
      w = (w + cs + ALIGN - 1) & ~(ALIGN - 1);
    }
    lc.wlen = w - start;
    if (codec == NONE || lc.wlen * 100 >= lc.ulen * 97) {
      lc.raw = true;
      memcpy(&wire[start], &src[u], lc.ulen);
      lc.subs.clear();
      w = start + ((lc.ulen + ALIGN - 1) & ~(ALIGN - 1));
      lc.wlen = lc.ulen;
    }
    chunks.push_back(std::move(lc));
  }
  double t = now() - t0;
  wire.resize(w);
  ZSTD_freeCCtx(zc);
  return t;
}

static void *gd_dec_tls() {
  thread_local void *d = libdeflate_alloc_gdeflate_decompressor();
  return d;
}
static void cpu_decompress(const Sub &s, const char *wbase, char *out) {
  switch (codec) {
  case ZSTD: {
    thread_local ZSTD_DCtx *dc = ZSTD_createDCtx();
    size_t r = ZSTD_decompressDCtx(dc, out, s.usize, wbase + s.off, s.csize);
    if (r != s.usize)
      abort();
    break;
  }
  case LZ4:
    if ((size_t)LZ4_decompress_safe(wbase + s.off, out, (int)s.csize,
                                    (int)s.usize) != s.usize)
      abort();
    break;
  case GDEFLATE: {
    libdeflate_gdeflate_in_page pg{wbase + s.off, s.csize};
    size_t got = 0;
    if (libdeflate_gdeflate_decompress(gd_dec_tls(), &pg, 1, out, s.usize,
                                       &got) != 0 ||
        got != s.usize)
      abort();
    break;
  }
  default:
    memcpy(out, wbase + s.off, s.usize);
  }
}

static size_t gpu_temp_bytes(size_t nchunks, size_t maxu, size_t total) {
  size_t tb = 0;
  if (codec == ZSTD)
    NK(nvcompBatchedZstdDecompressGetTempSizeAsync(
        nchunks, maxu, nvcompBatchedZstdDecompressDefaultOpts, &tb, total));
  if (codec == LZ4)
    NK(nvcompBatchedLZ4DecompressGetTempSizeAsync(
        nchunks, maxu, nvcompBatchedLZ4DecompressDefaultOpts, &tb, total));
  if (codec == GDEFLATE)
    NK(nvcompBatchedGdeflateDecompressGetTempSizeAsync(
        nchunks, maxu, nvcompBatchedGdeflateDecompressDefaultOpts, &tb, total));
  return tb;
}
static void gpu_decompress(void **dcp, size_t *dcs, size_t *dus, size_t *dact,
                           size_t n, void *tmp, size_t tb, void **dout,
                           nvcompStatus_t *st, CUstream s) {
  cudaStream_t cs = (cudaStream_t)s;
  if (codec == ZSTD)
    NK(nvcompBatchedZstdDecompressAsync(dcp, dcs, dus, dact, n, tmp, tb, dout,
                                        nvcompBatchedZstdDecompressDefaultOpts,
                                        st, cs));
  if (codec == LZ4)
    NK(nvcompBatchedLZ4DecompressAsync(dcp, dcs, dus, dact, n, tmp, tb, dout,
                                       nvcompBatchedLZ4DecompressDefaultOpts,
                                       st, cs));
  if (codec == GDEFLATE)
    NK(nvcompBatchedGdeflateDecompressAsync(
        dcp, dcs, dus, dact, n, tmp, tb, dout,
        nvcompBatchedGdeflateDecompressDefaultOpts, st, cs));
}

static CUcontext ctx;
static CUdeviceptr dst;
static size_t slot_bytes, nmax, temp_per_slot;
static std::atomic<double> gpu_busy_ms{0};

// Per-slot pinned/device layout: [ptrs nmax][csizes nmax][usizes nmax][outptrs
// nmax][data L+pad]
static void worker(int t, int T, bool gpu) {
  CK(cuCtxSetCurrent(ctx));
  CUstream sts[2];
  CK(cuStreamCreate(&sts[0], CU_STREAM_NON_BLOCKING));
  CK(cuStreamCreate(&sts[1], CU_STREAM_NON_BLOCKING));
  char *h[2];
  CUdeviceptr d[2] = {0, 0}, tmp[2] = {0, 0}, act[2] = {0, 0}, stat[2] = {0, 0};
  CUevent done[2], a[2], b[2];
  bool pend[2] = {false, false}, timed[2] = {false, false};
  for (int s = 0; s < 2; s++) {
    CK(cuMemHostAlloc((void **)&h[s], slot_bytes, 0));
    CK(cuEventCreate(&done[s],
                     CU_EVENT_BLOCKING_SYNC | CU_EVENT_DISABLE_TIMING));
    CK(cuEventCreate(&a[s], CU_EVENT_BLOCKING_SYNC));
    CK(cuEventCreate(&b[s], CU_EVENT_BLOCKING_SYNC));
    if (gpu) {
      CK(cuMemAlloc(&d[s], slot_bytes));
      if (temp_per_slot)
        CK(cuMemAlloc(&tmp[s], temp_per_slot));
      CK(cuMemAlloc(&act[s], nmax * 8));
      CK(cuMemAlloc(&stat[s], nmax * 4));
    }
  }
  double busy = 0;
  size_t k = 0;
  const size_t meta = 4 * nmax * 8;
  for (size_t i = t; i < chunks.size(); i += T, k++) {
    int s = k & 1;
    CUstream st = sts[s];
    if (pend[s]) {
      CK(cuEventSynchronize(done[s]));
      if (timed[s]) {
        float ms;
        CK(cuEventElapsedTime(&ms, a[s], b[s]));
        busy += ms;
      }
    }
    const LChunk &lc = chunks[i];
    const char *wb = &wire[lc.woff];
    timed[s] = false;
    if (lc.raw) {
      memcpy(h[s], wb, lc.ulen);
      CK(cuMemcpyHtoDAsync(dst + lc.uoff, h[s], lc.ulen, st));
    } else if (!gpu) {
      for (const Sub &sb : lc.subs)
        cpu_decompress(sb, wb, h[s] + (&sb - &lc.subs[0]) * C);
      CK(cuMemcpyHtoDAsync(dst + lc.uoff, h[s], lc.ulen, st));
    } else {
      size_t n = lc.subs.size();
      uint64_t *hp = (uint64_t *)h[s], *hc = hp + nmax, *hu = hc + nmax,
               *ho = hu + nmax;
      for (size_t j = 0; j < n; j++) {
        hp[j] = d[s] + meta + lc.subs[j].off;
        hc[j] = lc.subs[j].csize;
        hu[j] = lc.subs[j].usize;
        ho[j] = dst + lc.uoff + j * C;
      }
      memcpy(h[s] + meta, wb, lc.wlen);
      CK(cuMemcpyHtoDAsync(d[s], h[s], meta + lc.wlen, st));
      CK(cuEventRecord(a[s], st));
      CUdeviceptr base = d[s];
      gpu_decompress((void **)base, (size_t *)(base + nmax * 8),
                     (size_t *)(base + 2 * nmax * 8), (size_t *)act[s], n,
                     (void *)tmp[s], temp_per_slot,
                     (void **)(base + 3 * nmax * 8), (nvcompStatus_t *)stat[s],
                     st);
      CK(cuEventRecord(b[s], st));
      timed[s] = true;
    }
    CK(cuEventRecord(done[s], st));
    pend[s] = true;
  }
  for (int s = 0; s < 2; s++)
    if (pend[s]) {
      CK(cuEventSynchronize(done[s]));
      if (timed[s]) {
        float ms;
        CK(cuEventElapsedTime(&ms, a[s], b[s]));
        busy += ms;
      }
    }
  double cur = gpu_busy_ms.load();
  while (!gpu_busy_ms.compare_exchange_weak(cur, cur + busy)) {
  }
  for (int s = 0; s < 2; s++) {
    cuMemFreeHost(h[s]);
    if (d[s])
      cuMemFree(d[s]);
    if (tmp[s])
      cuMemFree(tmp[s]);
    if (act[s])
      cuMemFree(act[s]);
    if (stat[s])
      cuMemFree(stat[s]);
  }
  cuStreamDestroy(sts[0]);
  cuStreamDestroy(sts[1]);
}

static bool verify() {
  std::vector<char> back(N);
  CK(cuMemcpyDtoH(back.data(), dst, N));
  return memcmp(back.data(), src.data(), N) == 0;
}

// Whole buffer resident on device, one batched decompress: pure decoder
// throughput.
static void kernel_only(double *gibs, size_t *tb_out) {
  std::vector<uint64_t> hp, hc, hu, ho;
  for (const LChunk &lc : chunks)
    for (size_t j = 0; j < lc.subs.size(); j++) {
      hp.push_back(lc.woff + lc.subs[j].off);
      hc.push_back(lc.subs[j].csize);
      hu.push_back(lc.subs[j].usize);
      ho.push_back(dst + lc.uoff + j * C);
    }
  *gibs = 0;
  *tb_out = 0;
  if (hp.empty())
    return;
  size_t n = hp.size(), ubytes = 0;
  for (auto u : hu)
    ubytes += u;
  CUdeviceptr dw, arr, act, stat, tmp = 0;
  CK(cuMemAlloc(&dw, wire.size()));
  CK(cuMemcpyHtoD(dw, wire.data(), wire.size()));
  for (auto &p : hp)
    p += dw;
  CK(cuMemAlloc(&arr, 4 * n * 8));
  CK(cuMemcpyHtoD(arr, hp.data(), n * 8));
  CK(cuMemcpyHtoD(arr + n * 8, hc.data(), n * 8));
  CK(cuMemcpyHtoD(arr + 2 * n * 8, hu.data(), n * 8));
  CK(cuMemcpyHtoD(arr + 3 * n * 8, ho.data(), n * 8));
  CK(cuMemAlloc(&act, n * 8));
  CK(cuMemAlloc(&stat, n * 4));
  size_t tb = gpu_temp_bytes(n, C, ubytes);
  if (codec == ZSTD) {
    size_t sync_tb = 0;
    NK(nvcompBatchedZstdDecompressGetTempSizeSync(
        (void **)arr, (size_t *)(arr + n * 8), n, C, &sync_tb, ubytes,
        nvcompBatchedZstdDecompressDefaultOpts, (nvcompStatus_t *)stat, 0));
    fprintf(stderr, "TEMP n=%zu chunk=%zu async=%zu sync=%zu\n", n, C, tb,
            sync_tb);
    if (getenv("USE_SYNC_TEMP"))
      tb = sync_tb;
  }
  if (tb)
    CK(cuMemAlloc(&tmp, tb));
  *tb_out = tb;
  CUevent e0, e1;
  CK(cuEventCreate(&e0, 0));
  CK(cuEventCreate(&e1, 0));
  float best = 1e30f;
  for (int r = 0; r < 4; r++) {
    CK(cuEventRecord(e0, 0));
    gpu_decompress((void **)arr, (size_t *)(arr + n * 8),
                   (size_t *)(arr + 2 * n * 8), (size_t *)act, n, (void *)tmp,
                   tb, (void **)(arr + 3 * n * 8), (nvcompStatus_t *)stat, 0);
    CK(cuEventRecord(e1, 0));
    CK(cuEventSynchronize(e1));
    float ms;
    CK(cuEventElapsedTime(&ms, e0, e1));
    if (r)
      best = std::min(best, ms);
  }
  std::vector<int> sts(n);
  CK(cuMemcpyDtoH(sts.data(), stat, n * 4));
  for (size_t i = 0; i < n; i++)
    if (sts[i] != nvcompSuccess) {
      fprintf(stderr, "kernel_only chunk %zu status %d\n", i, sts[i]);
      exit(2);
    }
  *gibs = ubytes / (best / 1e3) / (1 << 30);
  cuMemFree(dw);
  cuMemFree(arr);
  cuMemFree(act);
  cuMemFree(stat);
  if (tmp)
    cuMemFree(tmp);
}

int main(int argc, char **argv) {
  // gpudec_bench <file> <dev> <codec> <level> <chunkKiB> <threads> <reps>
  if (argc < 8)
    return 1;
  std::string cn = argv[3];
  codec = cn == "zstd"       ? ZSTD
          : cn == "lz4"      ? LZ4
          : cn == "gdeflate" ? GDEFLATE
                             : NONE;
  level = atoi(argv[4]);
  C = (size_t)atoi(argv[5]) << 10;
  int T = atoi(argv[6]), reps = atoi(argv[7]);
  if (codec == GDEFLATE)
    C = 65536;
  FILE *f = fopen(argv[1], "rb");
  fseek(f, 0, SEEK_END);
  N = ftell(f);
  fseek(f, 0, SEEK_SET);
  src.resize(N);
  if (fread(src.data(), 1, N, f) != N)
    return 1;
  fclose(f);

  double ct = compress_all();
  size_t raw_chunks = 0;
  for (auto &c : chunks)
    raw_chunks += c.raw;

  CK(cuInit(0));
  CUdevice dev;
  CK(cuDeviceGet(&dev, atoi(argv[2])));
  CK(cuDevicePrimaryCtxSetFlags(dev, CU_CTX_SCHED_BLOCKING_SYNC));
  CK(cuDevicePrimaryCtxRetain(&ctx, dev));
  CK(cuCtxSetCurrent(ctx));
  char name[128];
  cuDeviceGetName(name, sizeof name, dev);
  CK(cuMemAlloc(&dst, N));
  nmax = L / C;
  slot_bytes = 4 * nmax * 8 + L + nmax * ALIGN + 4096;
  temp_per_slot = codec == NONE ? 0 : gpu_temp_bytes(nmax, C, L);

  double kgibs = 0;
  size_t ktb = 0;
  if (codec != NONE && raw_chunks < chunks.size()) {
    kernel_only(&kgibs, &ktb);
    CK(cuMemsetD8(dst, 0xab, N));
  }

  for (int gpu = 0; gpu < 2; gpu++) {
    if (gpu && codec == NONE)
      break;
    std::vector<double> rates, cpus;
    double busy_total = 0;
    bool ok = true;
    for (int r = 0; r < reps; r++) {
      CK(cuMemsetD8(dst, 0xab, N));
      CK(cuCtxSynchronize());
      gpu_busy_ms = 0;
      double c0 = cpu_s(), t0 = now();
      std::vector<std::thread> th;
      for (int t = 0; t < T; t++)
        th.emplace_back(worker, t, T, gpu != 0);
      for (auto &x : th)
        x.join();
      double el = now() - t0, c1 = cpu_s();
      rates.push_back(N / el / (1 << 30));
      cpus.push_back((c1 - c0) / (N / double(1 << 30)));
      busy_total += gpu_busy_ms.load();
      if (r == reps - 1)
        ok = verify();
    }
    std::vector<double> rs = rates, cs = cpus;
    std::sort(rs.begin(), rs.end());
    std::sort(cs.begin(), cs.end());
    double busy_frac =
        busy_total / reps / 1e3 * (rs[rs.size() / 2] * (1 << 30) / N);
    printf("RESULT gpu=\"%s\" file=%s codec=%s level=%d chunk_kib=%zu "
           "threads=%d path=%s ratio=%.3f raw_lchunks=%zu/%zu "
           "comp_mibps_1t=%.0f gibs_med=%.3f gibs_min=%.3f gibs_max=%.3f "
           "cpu_s_per_gib=%.3f gpu_dec_ms_per_gib=%.1f "
           "gpu_busy_frac=%.2f temp_per_4mib_batch=%zu kernel_only_gibs=%.2f "
           "kernel_only_temp=%zu verify=%s\n",
           name, argv[1], cn.c_str(), level, C >> 10, T, gpu ? "gpu" : "cpu",
           double(N) / wire.size(), raw_chunks, chunks.size(),
           N / ct / (1 << 20), rs[rs.size() / 2], rs.front(), rs.back(),
           cs[cs.size() / 2],
           gpu ? busy_total / reps / (N / double(1 << 30)) : 0.0,
           gpu ? busy_frac : 0.0, gpu ? temp_per_slot : 0, kgibs, ktb,
           ok ? "ok" : "FAIL");
    fflush(stdout);
  }
  return 0;
}
