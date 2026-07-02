#include "gpu_snapshot.h"

#include <algorithm>
#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <mutex>
#include <string>
#include <vector>

#ifndef _WIN32
#include <fcntl.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#endif

#include "lupine_fatbin.h"
#include "lupine_log.h"

namespace {

constexpr char kPayloadMagic[8] = {'L', 'U', 'P', 'C', 'K', 'P', 'T', '1'};
constexpr size_t kMaxMapPath = 256;

std::mutex g_mutex;
bool g_restored = false;

struct ModuleRec {
  unsigned int kind;
  std::vector<unsigned char> image;
};

std::map<CUmodule, ModuleRec> g_modules;
std::map<CUmodule, CUlibrary> g_library_modules;
std::map<CUfunction, std::pair<CUmodule, std::string>> g_functions;
std::map<CUlibrary, ModuleRec> g_libraries;
std::map<CUkernel, std::pair<CUlibrary, std::string>> g_kernels;
std::map<CUfunction, CUkernel> g_kernel_functions;

std::map<CUmodule, CUmodule> g_module_remap;
std::map<CUlibrary, CUlibrary> g_library_remap;
std::map<CUkernel, CUkernel> g_kernel_remap;
std::map<CUfunction, CUfunction> g_function_remap;
std::map<CUstream, CUstream> g_stream_remap;

constexpr unsigned int K_FATBINC_V1 = 1;
constexpr unsigned int K_FATBIN_RAW = 2;
constexpr unsigned int K_FATBINC_V2 = 3;

struct PayloadHeader {
  char magic[8];
  uint64_t size;
};

struct Map {
  uintptr_t start = 0;
  uintptr_t end = 0;
  char perms[5] = {};
  char path[kMaxMapPath] = {};
};

using MapList = std::vector<Map>;

ModuleRec make_module_rec(unsigned int kind, const void *image, size_t size) {
  ModuleRec rec;
  rec.kind = kind;
  if (image != nullptr && size != 0) {
    const auto *bytes = static_cast<const unsigned char *>(image);
    rec.image.assign(bytes, bytes + size);
  }
  return rec;
}

template <typename Handle>
Handle translate_handle(const std::map<Handle, Handle> &remap, Handle handle) {
  auto it = remap.find(handle);
  return it == remap.end() ? handle : it->second;
}

uint64_t fnv1a(const char *s) {
  uint64_t h = 1469598103934665603ULL;
  for (; *s; ++s) {
    h ^= static_cast<unsigned char>(*s);
    h *= 1099511628211ULL;
  }
  return h;
}

bool mkdir_if_needed(const std::string &path) {
#ifdef _WIN32
  (void)path;
  return false;
#else
  return mkdir(path.c_str(), 0777) == 0 || errno == EEXIST;
#endif
}

std::string snapshot_dir(const char *id) {
  const char *base = getenv("LUPINE_SNAPSHOT_DIR");
  std::string dir = base && base[0] ? base : "/tmp/lupine-snapshots";
  if (!mkdir_if_needed(dir)) {
    LUPINE_LOG_ERROR("gpu_snapshot: cannot create " << dir);
    return "";
  }

  char hex[17];
  snprintf(hex, sizeof(hex), "%016llx", (unsigned long long)fnv1a(id));
  dir += "/";
  dir += hex;
  if (!mkdir_if_needed(dir)) {
    LUPINE_LOG_ERROR("gpu_snapshot: cannot create " << dir);
    return "";
  }
  return dir;
}

std::string join(const std::string &dir, const char *name) {
  std::string out = dir;
  if (!out.empty() && out.back() != '/') out.push_back('/');
  return out + name;
}

int read_id(conn_t *conn, char *id, size_t cap) {
  uint32_t len = 0;
  if (rpc_read(conn, &len, sizeof(len)) < 0) return -1;
  if (len == 0 || len >= cap) return -1;
  if (rpc_read(conn, id, len) < 0) return -1;
  id[len] = '\0';
  return 0;
}

void write_u64(FILE *f, uint64_t v) { fwrite(&v, sizeof(v), 1, f); }

bool read_u64(FILE *f, uint64_t *v) {
  return fread(v, sizeof(*v), 1, f) == 1;
}

void write_blob(FILE *f, const std::vector<unsigned char> &blob) {
  write_u64(f, blob.size());
  if (!blob.empty()) fwrite(blob.data(), 1, blob.size(), f);
}

bool read_blob(FILE *f, std::vector<unsigned char> &blob) {
  uint64_t size = 0;
  if (!read_u64(f, &size)) return false;
  blob.resize(size);
  return size == 0 || fread(blob.data(), 1, size, f) == size;
}

void write_str(FILE *f, const std::string &s) {
  write_u64(f, s.size());
  if (!s.empty()) fwrite(s.data(), 1, s.size(), f);
}

bool read_str(FILE *f, std::string &s) {
  uint64_t size = 0;
  if (!read_u64(f, &size)) return false;
  s.assign(size, '\0');
  return size == 0 || fread(&s[0], 1, size, f) == size;
}

bool save_objects(const std::string &dir) {
  FILE *f = fopen(join(dir, "gpu.objects").c_str(), "wb");
  if (!f) return false;

  write_u64(f, g_modules.size());
  for (auto &kv : g_modules) {
    write_u64(f, reinterpret_cast<uint64_t>(kv.first));
    write_u64(f, kv.second.kind);
    write_blob(f, kv.second.image);
  }

  write_u64(f, g_libraries.size());
  for (auto &kv : g_libraries) {
    write_u64(f, reinterpret_cast<uint64_t>(kv.first));
    write_u64(f, kv.second.kind);
    write_blob(f, kv.second.image);
  }

  write_u64(f, g_library_modules.size());
  for (auto &kv : g_library_modules) {
    write_u64(f, reinterpret_cast<uint64_t>(kv.first));
    write_u64(f, reinterpret_cast<uint64_t>(kv.second));
  }

  write_u64(f, g_functions.size());
  for (auto &kv : g_functions) {
    write_u64(f, reinterpret_cast<uint64_t>(kv.first));
    write_u64(f, reinterpret_cast<uint64_t>(kv.second.first));
    write_str(f, kv.second.second);
  }

  write_u64(f, g_kernels.size());
  for (auto &kv : g_kernels) {
    write_u64(f, reinterpret_cast<uint64_t>(kv.first));
    write_u64(f, reinterpret_cast<uint64_t>(kv.second.first));
    write_str(f, kv.second.second);
  }

  write_u64(f, g_kernel_functions.size());
  for (auto &kv : g_kernel_functions) {
    write_u64(f, reinterpret_cast<uint64_t>(kv.first));
    write_u64(f, reinterpret_cast<uint64_t>(kv.second));
  }

  fclose(f);
  return true;
}

CUresult load_module_image(unsigned int kind, const unsigned char *image,
                           CUmodule *out) {
  if (kind == K_FATBINC_V1 || kind == K_FATBINC_V2)
    return cuModuleLoadFatBinary(out, image);
  if (kind == K_FATBIN_RAW) return cuModuleLoadData(out, image);
  return CUDA_ERROR_NOT_SUPPORTED;
}

CUresult load_library_image(unsigned int kind, const unsigned char *image,
                            CUlibrary *out) {
  if (kind == K_FATBINC_V1 || kind == K_FATBINC_V2) {
    lupine_fatbin_wrapper wrapper = {
        LUPINE_FATBINC_MAGIC, kind == K_FATBINC_V2 ? 2U : 1U, image, nullptr};
    return cuLibraryLoadData(out, &wrapper, nullptr, nullptr, 0, nullptr,
                             nullptr, 0);
  }
  if (kind == K_FATBIN_RAW)
    return cuLibraryLoadData(out, image, nullptr, nullptr, 0, nullptr, nullptr,
                             0);
  return CUDA_ERROR_NOT_SUPPORTED;
}

CUresult replay_objects(const std::string &dir) {
  FILE *f = fopen(join(dir, "gpu.objects").c_str(), "rb");
  if (!f) return CUDA_SUCCESS;

  uint64_t n = 0;
  CUresult rc = CUDA_ERROR_UNKNOWN;
  if (!read_u64(f, &n)) goto done;
  for (uint64_t i = 0; i < n; i++) {
    uint64_t old_h = 0, kind = 0;
    std::vector<unsigned char> image;
    if (!read_u64(f, &old_h) || !read_u64(f, &kind) || !read_blob(f, image))
      goto done;
    CUmodule module = nullptr;
    rc = load_module_image(static_cast<unsigned int>(kind), image.data(),
                           &module);
    if (rc != CUDA_SUCCESS) goto done;
    g_modules[module] = ModuleRec{static_cast<unsigned int>(kind),
                                  std::move(image)};
    g_module_remap[reinterpret_cast<CUmodule>(old_h)] = module;
  }

  if (!read_u64(f, &n)) goto done;
  for (uint64_t i = 0; i < n; i++) {
    uint64_t old_h = 0, kind = 0;
    std::vector<unsigned char> image;
    if (!read_u64(f, &old_h) || !read_u64(f, &kind) || !read_blob(f, image))
      goto done;
    CUlibrary library = nullptr;
    rc = load_library_image(static_cast<unsigned int>(kind), image.data(),
                            &library);
    if (rc != CUDA_SUCCESS) goto done;
    g_libraries[library] = ModuleRec{static_cast<unsigned int>(kind),
                                     std::move(image)};
    g_library_remap[reinterpret_cast<CUlibrary>(old_h)] = library;
  }

  if (!read_u64(f, &n)) goto done;
  for (uint64_t i = 0; i < n; i++) {
    uint64_t old_mod = 0, old_lib = 0;
    if (!read_u64(f, &old_mod) || !read_u64(f, &old_lib)) goto done;
    auto it = g_library_remap.find(reinterpret_cast<CUlibrary>(old_lib));
    if (it == g_library_remap.end()) goto done;
    CUmodule module = nullptr;
    rc = cuLibraryGetModule(&module, it->second);
    if (rc != CUDA_SUCCESS) goto done;
    g_library_modules[module] = it->second;
    g_module_remap[reinterpret_cast<CUmodule>(old_mod)] = module;
  }

  if (!read_u64(f, &n)) goto done;
  for (uint64_t i = 0; i < n; i++) {
    uint64_t old_fn = 0, old_mod = 0;
    std::string name;
    if (!read_u64(f, &old_fn) || !read_u64(f, &old_mod) || !read_str(f, name))
      goto done;
    auto it = g_module_remap.find(reinterpret_cast<CUmodule>(old_mod));
    if (it == g_module_remap.end()) goto done;
    CUfunction fn = nullptr;
    rc = cuModuleGetFunction(&fn, it->second, name.c_str());
    if (rc != CUDA_SUCCESS) goto done;
    g_functions[fn] = {it->second, name};
    g_function_remap[reinterpret_cast<CUfunction>(old_fn)] = fn;
  }

  if (!read_u64(f, &n)) goto done;
  for (uint64_t i = 0; i < n; i++) {
    uint64_t old_k = 0, old_lib = 0;
    std::string name;
    if (!read_u64(f, &old_k) || !read_u64(f, &old_lib) || !read_str(f, name))
      goto done;
    auto it = g_library_remap.find(reinterpret_cast<CUlibrary>(old_lib));
    if (it == g_library_remap.end()) goto done;
    CUkernel kernel = nullptr;
    rc = cuLibraryGetKernel(&kernel, it->second, name.c_str());
    if (rc != CUDA_SUCCESS) goto done;
    g_kernels[kernel] = {it->second, name};
    g_kernel_remap[reinterpret_cast<CUkernel>(old_k)] = kernel;
  }

  if (!read_u64(f, &n)) goto done;
  for (uint64_t i = 0; i < n; i++) {
    uint64_t old_fn = 0, old_k = 0;
    if (!read_u64(f, &old_fn) || !read_u64(f, &old_k)) goto done;
    auto it = g_kernel_remap.find(reinterpret_cast<CUkernel>(old_k));
    if (it == g_kernel_remap.end()) goto done;
    CUfunction fn = nullptr;
    rc = cuKernelGetFunction(&fn, it->second);
    if (rc != CUDA_SUCCESS) goto done;
    g_kernel_functions[fn] = it->second;
    g_function_remap[reinterpret_cast<CUfunction>(old_fn)] = fn;
  }

  rc = CUDA_SUCCESS;
done:
  fclose(f);
  return rc;
}

#ifdef _WIN32
MapList read_private_anon_maps() { return {}; }
int open_self_mem(int) { return -1; }
#else
bool is_private_anon_rw(const Map &map) {
  return map.perms[0] == 'r' && map.perms[1] == 'w' &&
         map.perms[2] == '-' && map.perms[3] == 'p' && map.path[0] == '\0';
}

MapList read_private_anon_maps() {
  FILE *f = fopen("/proc/self/maps", "r");
  if (!f) return {};

  MapList maps;
  char line[1024];
  while (fgets(line, sizeof(line), f)) {
    Map map;
    unsigned long start = 0, end = 0, offset = 0, inode = 0;
    char dev[64] = {};
    char path[kMaxMapPath] = {};
    int fields = sscanf(line, "%lx-%lx %4s %lx %63s %lu %255[^\n]", &start,
                        &end, map.perms, &offset, dev, &inode, path);
    if (fields < 6) continue;
    map.start = static_cast<uintptr_t>(start);
    map.end = static_cast<uintptr_t>(end);
    if (fields == 7) {
      char *p = path;
      while (*p == ' ') p++;
      snprintf(map.path, sizeof(map.path), "%s", p);
    }
    if (is_private_anon_rw(map)) maps.push_back(map);
  }
  fclose(f);
  return maps;
}

int open_self_mem(int flags) { return open("/proc/self/mem", flags); }
#endif

bool same_map(const Map &a, const Map &b) {
  return a.start == b.start && a.end == b.end &&
         strcmp(a.perms, b.perms) == 0 && strcmp(a.path, b.path) == 0;
}

bool contains_map(const MapList &maps, const Map &needle) {
  for (const Map &map : maps) {
    if (same_map(map, needle)) return true;
  }
  return false;
}

bool find_payload_map(const MapList &before, const MapList &after,
                      size_t expected_size, Map *out) {
  Map best;
  for (const Map &map : after) {
    size_t size = map.end - map.start;
    if (contains_map(before, map)) continue;
    if (expected_size != 0 && size != expected_size) continue;
    if (best.start == 0 || size > best.end - best.start) best = map;
  }
  if (best.start == 0) return false;
  *out = best;
  return true;
}

void log_new_payload_candidates(const MapList &before, const MapList &after) {
  for (const Map &map : after) {
    if (contains_map(before, map)) continue;
    LUPINE_LOG_ERROR("gpu_snapshot: new anon map candidate "
                     << (void *)map.start << "-" << (void *)map.end
                     << " size=" << (map.end - map.start));
  }
}

bool read_exact(int fd, void *buf, size_t size, off_t offset) {
  char *p = static_cast<char *>(buf);
  while (size != 0) {
    ssize_t n = pread(fd, p, size, offset);
    if (n < 0 && errno == EINTR) continue;
    if (n <= 0) return false;
    p += n;
    offset += n;
    size -= static_cast<size_t>(n);
  }
  return true;
}

bool write_exact(int fd, const void *buf, size_t size, off_t offset) {
  const char *p = static_cast<const char *>(buf);
  while (size != 0) {
    ssize_t n = pwrite(fd, p, size, offset);
    if (n < 0 && errno == EINTR) continue;
    if (n <= 0) return false;
    p += n;
    offset += n;
    size -= static_cast<size_t>(n);
  }
  return true;
}

CUresult unlock_self() {
  CUcheckpointUnlockArgs unlock_args = {};
  return cuCheckpointProcessUnlock(getpid(), &unlock_args);
}

CUresult checkpoint_self() {
  CUcheckpointLockArgs lock_args = {};
  CUcheckpointCheckpointArgs checkpoint_args = {};
  CUresult result = cuCheckpointProcessLock(getpid(), &lock_args);
  if (result != CUDA_SUCCESS) return result;
  result = cuCheckpointProcessCheckpoint(getpid(), &checkpoint_args);
  if (result != CUDA_SUCCESS) unlock_self();
  return result;
}

CUresult restore_self() {
  CUcheckpointRestoreArgs restore_args = {};
  CUresult result = cuCheckpointProcessRestore(getpid(), &restore_args);
  if (result != CUDA_SUCCESS) return result;
  return unlock_self();
}

CUresult ensure_context() {
  CUcontext cur = nullptr;
  CUresult result = cuCtxGetCurrent(&cur);
  if (result == CUDA_SUCCESS && cur != nullptr) return CUDA_SUCCESS;

  result = cuInit(0);
  if (result != CUDA_SUCCESS) return result;
  CUdevice dev = 0;
  result = cuDeviceGet(&dev, 0);
  if (result != CUDA_SUCCESS) return result;
  CUcontext ctx = nullptr;
  result = cuDevicePrimaryCtxRetain(&ctx, dev);
  if (result != CUDA_SUCCESS) return result;
  return cuCtxSetCurrent(ctx);
}

// The in-kernel malloc heap (CU_LIMIT_MALLOC_HEAP_SIZE) is only materialized
// when a kernel actually calls malloc(); setting the limit alone reserves
// nothing. A workload that used device-side malloc therefore has a heap
// reservation in its checkpoint payload that a bare fresh worker lacks, so the
// worker's payload comes out smaller and the shapes don't line up. To match, we
// re-apply the saved limit and launch a tiny malloc so the driver reserves the
// same heap before the worker checkpoints itself.
const char kWarmupPtx[] = R"ptx(
.version 7.0
.target sm_70
.address_size 64
.extern .func (.param .b64 func_retval0) malloc (.param .b64 malloc_param_0);
.extern .func free (.param .b64 free_param_0);
.visible .entry lupine_heap_warmup(.param .u64 lupine_heap_warmup_param_0)
{
  .reg .pred %p<2>;
  .reg .b64 %rd<3>;
  ld.param.u64 %rd2, [lupine_heap_warmup_param_0];
  { .param .b64 param0;
    st.param.b64 [param0+0], %rd2;
    .param .b64 retval0;
    call.uni (retval0), malloc, (param0);
    ld.param.b64 %rd1, [retval0+0]; }
  setp.eq.s64 %p1, %rd1, 0;
  @%p1 bra $L__BB0_2;
  { .param .b64 param0;
    st.param.b64 [param0+0], %rd1;
    call.uni free, (param0); }
$L__BB0_2:
  ret;
}
)ptx";

CUresult reproduce_malloc_heap(uint64_t limit) {
  CUresult result = cuCtxSetLimit(CU_LIMIT_MALLOC_HEAP_SIZE,
                                  static_cast<size_t>(limit));
  if (result != CUDA_SUCCESS) return result;
  CUmodule mod = nullptr;
  result = cuModuleLoadData(&mod, kWarmupPtx);
  if (result != CUDA_SUCCESS) return result;
  CUfunction fn = nullptr;
  result = cuModuleGetFunction(&fn, mod, "lupine_heap_warmup");
  if (result == CUDA_SUCCESS) {
    unsigned long long bytes = 16;
    void *args[] = {&bytes};
    result = cuLaunchKernel(fn, 1, 1, 1, 1, 1, 1, 0, 0, args, nullptr);
    if (result == CUDA_SUCCESS) result = cuCtxSynchronize();
  }
  // The heap reservation is a context resource; it outlives the module that
  // triggered it, so unloading here keeps the worker's payload clean.
  cuModuleUnload(mod);
  return result;
}

bool save_heap_limit(const std::string &dir) {
  size_t limit = 0;
  if (cuCtxGetLimit(&limit, CU_LIMIT_MALLOC_HEAP_SIZE) != CUDA_SUCCESS)
    limit = 0;
  FILE *f = fopen(join(dir, "gpu.meta").c_str(), "wb");
  if (!f) return false;
  uint64_t v = limit;
  bool ok = fwrite(&v, sizeof(v), 1, f) == 1;
  fclose(f);
  return ok;
}

uint64_t read_saved_heap_limit(const std::string &dir) {
  FILE *f = fopen(join(dir, "gpu.meta").c_str(), "rb");
  if (!f) return 0;
  uint64_t v = 0;
  if (fread(&v, sizeof(v), 1, f) != 1) v = 0;
  fclose(f);
  return v;
}

// Checkpoint the worker and locate the newly-created payload mapping of the
// expected size. On success the worker is left checkpointed (the caller must
// restore it); on a shape mismatch the worker is restored so the caller can
// retry from a running state or bail cleanly.
bool checkpoint_and_locate(size_t expected_size, Map *payload) {
  MapList before = read_private_anon_maps();
  if (checkpoint_self() != CUDA_SUCCESS) return false;
  MapList after = read_private_anon_maps();
  if (find_payload_map(before, after, expected_size, payload)) return true;
  log_new_payload_candidates(before, after);
  restore_self();
  return false;
}

bool write_payload_header(FILE *f, uint64_t size) {
  PayloadHeader header = {};
  memcpy(header.magic, kPayloadMagic, sizeof(header.magic));
  header.size = size;
  return fwrite(&header, sizeof(header), 1, f) == 1;
}

bool read_payload_header(FILE *f, uint64_t *size) {
  PayloadHeader header = {};
  bool ok = fread(&header, sizeof(header), 1, f) == 1 &&
            memcmp(header.magic, kPayloadMagic, sizeof(header.magic)) == 0;
  if (ok) *size = header.size;
  return ok;
}

CUresult read_payload_size(const std::string &path, uint64_t *size) {
  FILE *f = fopen(path.c_str(), "rb");
  if (!f) return CUDA_ERROR_UNKNOWN;
  bool ok = read_payload_header(f, size);
  fclose(f);
  return ok ? CUDA_SUCCESS : CUDA_ERROR_UNKNOWN;
}

bool copy_mem_to_payload_file(const std::string &path, int mem, uintptr_t addr,
                              size_t size) {
  FILE *f = fopen(path.c_str(), "wb");
  if (!f) return false;

  bool ok = write_payload_header(f, size);
  std::vector<unsigned char> chunk(1 << 20);
  size_t copied = 0;
  while (ok && copied < size) {
    size_t want = std::min(chunk.size(), size - copied);
    ok = read_exact(mem, chunk.data(), want, static_cast<off_t>(addr + copied)) &&
         fwrite(chunk.data(), 1, want, f) == want;
    if (ok) copied += want;
  }
  fclose(f);
  return ok;
}

bool copy_payload_file_to_mem(const std::string &path, int mem, uintptr_t addr,
                              size_t size) {
  FILE *f = fopen(path.c_str(), "rb");
  if (!f) return false;

  uint64_t payload_size = 0;
  bool ok = read_payload_header(f, &payload_size) && payload_size == size;
  std::vector<unsigned char> chunk(1 << 20);
  size_t copied = 0;
  while (ok && copied < size) {
    size_t want = std::min(chunk.size(), size - copied);
    ok = fread(chunk.data(), 1, want, f) == want &&
         write_exact(mem, chunk.data(), want, static_cast<off_t>(addr + copied));
    if (ok) copied += want;
  }
  fclose(f);
  return ok;
}

} // namespace

CUresult lupine_gpu_alloc(CUdeviceptr *dptr, size_t bytesize) {
  return cuMemAlloc_v2(dptr, bytesize);
}

CUresult lupine_gpu_free(CUdeviceptr dptr) { return cuMemFree_v2(dptr); }

int lupine_gpu_owns(CUdeviceptr) { return 0; }

CUresult lupine_gpu_snapshot_save(const char *id) {
  std::lock_guard<std::mutex> lock(g_mutex);
  if (id == nullptr || id[0] == '\0') return CUDA_ERROR_INVALID_VALUE;

  CUresult result = ensure_context();
  if (result != CUDA_SUCCESS) return result;

  std::string dir = snapshot_dir(id);
  if (dir.empty()) return CUDA_ERROR_UNKNOWN;
  if (!save_objects(dir)) return CUDA_ERROR_UNKNOWN;
  // Best-effort: lets a fresh worker reproduce the in-kernel malloc heap
  // reservation before it checkpoints. A miss only affects malloc workloads.
  save_heap_limit(dir);

  int mem = open_self_mem(O_RDONLY);
  if (mem < 0) return CUDA_ERROR_UNKNOWN;

  MapList before = read_private_anon_maps();
  result = checkpoint_self();
  if (result != CUDA_SUCCESS) {
    close(mem);
    return result;
  }
  MapList after = read_private_anon_maps();

  Map payload;
  if (!find_payload_map(before, after, 0, &payload)) {
    close(mem);
    restore_self();
    return CUDA_ERROR_UNKNOWN;
  }

  size_t payload_size = payload.end - payload.start;
  result = copy_mem_to_payload_file(join(dir, "gpu.payload"), mem,
                                    payload.start, payload_size)
               ? CUDA_SUCCESS
               : CUDA_ERROR_UNKNOWN;
  close(mem);
  CUresult resume = restore_self();
  return result != CUDA_SUCCESS ? result : resume;
}

CUresult lupine_gpu_snapshot_restore(const char *id) {
  std::lock_guard<std::mutex> lock(g_mutex);
  if (id == nullptr || id[0] == '\0') return CUDA_ERROR_INVALID_VALUE;

  CUresult result = ensure_context();
  if (result != CUDA_SUCCESS) return result;

  std::string dir = snapshot_dir(id);
  if (dir.empty()) return CUDA_ERROR_UNKNOWN;

  std::string payload_path = join(dir, "gpu.payload");
  uint64_t payload_size = 0;
  result = read_payload_size(payload_path, &payload_size);
  if (result != CUDA_SUCCESS) return result;

  int mem = open_self_mem(O_RDWR);
  if (mem < 0) return CUDA_ERROR_UNKNOWN;

  size_t want = static_cast<size_t>(payload_size);
  Map payload;
  bool located = checkpoint_and_locate(want, &payload);
  if (!located) {
    // A bare worker's payload is smaller than a kernel-malloc workload's,
    // which carries a device malloc heap reservation. Re-apply the saved heap
    // limit, launch a malloc to materialize it, and checkpoint again so the
    // worker's payload shape matches.
    uint64_t heap_limit = read_saved_heap_limit(dir);
    if (heap_limit != 0 && reproduce_malloc_heap(heap_limit) == CUDA_SUCCESS)
      located = checkpoint_and_locate(want, &payload);
  }
  if (!located) {
    close(mem);
    LUPINE_LOG_ERROR("gpu_snapshot: restored worker produced incompatible "
                     "CUDA checkpoint payload size, expected " << payload_size);
    return CUDA_ERROR_UNKNOWN;
  }

  if (!copy_payload_file_to_mem(payload_path, mem, payload.start,
                                static_cast<size_t>(payload_size))) {
    close(mem);
    restore_self();
    return CUDA_ERROR_UNKNOWN;
  }
  close(mem);

  result = restore_self();
  if (result == CUDA_SUCCESS) result = replay_objects(dir);
  if (result == CUDA_SUCCESS) g_restored = true;
  return result;
}

int handle_manual_cuMemAlloc_v2(conn_t *conn) {
  CUdeviceptr dptr = 0;
  size_t bytesize = 0;
  if (rpc_read(conn, &dptr, sizeof(dptr)) < 0 ||
      rpc_read(conn, &bytesize, sizeof(bytesize)) < 0)
    return -1;
  int request_id = rpc_read_end(conn);
  if (request_id < 0) return -1;

  CUresult result = cuMemAlloc_v2(&dptr, bytesize);
  if (rpc_write_start_response(conn, request_id) < 0 ||
      rpc_write(conn, &dptr, sizeof(dptr)) < 0 ||
      rpc_write(conn, &result, sizeof(result)) < 0 || rpc_write_end(conn) < 0)
    return -1;
  return 0;
}

int handle_manual_cuMemFree_v2(conn_t *conn) {
  CUdeviceptr dptr = 0;
  if (rpc_read(conn, &dptr, sizeof(dptr)) < 0) return -1;
  int request_id = rpc_read_end(conn);
  if (request_id < 0) return -1;

  CUresult result = cuMemFree_v2(dptr);
  if (rpc_write_start_response(conn, request_id) < 0 ||
      rpc_write(conn, &result, sizeof(result)) < 0 || rpc_write_end(conn) < 0)
    return -1;
  return 0;
}

int handle_gpu_snapshot_save(conn_t *conn) {
  char id[256];
  if (read_id(conn, id, sizeof(id)) < 0) return -1;
  int request_id = rpc_read_end(conn);
  if (request_id < 0) return -1;
  CUresult result = lupine_gpu_snapshot_save(id);
  if (rpc_write_start_response(conn, request_id) < 0 ||
      rpc_write(conn, &result, sizeof(result)) < 0 || rpc_write_end(conn) < 0)
    return -1;
  return 0;
}

int handle_gpu_snapshot_restore(conn_t *conn) {
  char id[256];
  if (read_id(conn, id, sizeof(id)) < 0) return -1;
  int request_id = rpc_read_end(conn);
  if (request_id < 0) return -1;
  CUresult result = lupine_gpu_snapshot_restore(id);
  if (rpc_write_start_response(conn, request_id) < 0 ||
      rpc_write(conn, &result, sizeof(result)) < 0 || rpc_write_end(conn) < 0)
    return -1;
  return 0;
}

void lupine_gpu_track_module(CUmodule m, unsigned int kind, const void *image,
                             size_t size) {
  if (!m) return;
  std::lock_guard<std::mutex> lock(g_mutex);
  g_modules[m] = make_module_rec(kind, image, size);
}

void lupine_gpu_track_library_module(CUmodule m, CUlibrary lib) {
  if (!m || !lib) return;
  std::lock_guard<std::mutex> lock(g_mutex);
  g_library_modules[m] = lib;
}

void lupine_gpu_track_function(CUfunction fn, CUmodule m, const char *name) {
  if (!fn) return;
  std::lock_guard<std::mutex> lock(g_mutex);
  g_functions[fn] = {m, std::string(name ? name : "")};
}

void lupine_gpu_track_library(CUlibrary lib, unsigned int kind,
                              const void *image, size_t size) {
  if (!lib) return;
  std::lock_guard<std::mutex> lock(g_mutex);
  g_libraries[lib] = make_module_rec(kind, image, size);
}

void lupine_gpu_track_kernel(CUkernel k, CUlibrary lib, const char *name) {
  if (!k) return;
  std::lock_guard<std::mutex> lock(g_mutex);
  g_kernels[k] = {lib, std::string(name ? name : "")};
}

void lupine_gpu_track_kernel_function(CUfunction fn, CUkernel k) {
  if (!fn) return;
  std::lock_guard<std::mutex> lock(g_mutex);
  g_kernel_functions[fn] = k;
}

CUfunction lupine_gpu_translate_function(CUfunction fn) {
  std::lock_guard<std::mutex> lock(g_mutex);
  auto it = g_function_remap.find(fn);
  if (it != g_function_remap.end()) return it->second;
  auto k = g_kernel_remap.find(reinterpret_cast<CUkernel>(fn));
  if (k != g_kernel_remap.end()) return reinterpret_cast<CUfunction>(k->second);
  return fn;
}

CUmodule lupine_gpu_translate_module(CUmodule m) {
  std::lock_guard<std::mutex> lock(g_mutex);
  return translate_handle(g_module_remap, m);
}

CUlibrary lupine_gpu_translate_library(CUlibrary lib) {
  std::lock_guard<std::mutex> lock(g_mutex);
  return translate_handle(g_library_remap, lib);
}

CUkernel lupine_gpu_translate_kernel(CUkernel k) {
  std::lock_guard<std::mutex> lock(g_mutex);
  return translate_handle(g_kernel_remap, k);
}

CUcontext lupine_gpu_translate_context(CUcontext ctx) {
  if (ctx == nullptr) return nullptr;
  std::lock_guard<std::mutex> lock(g_mutex);
  if (!g_restored) return ctx;
  CUcontext current = nullptr;
  return cuCtxGetCurrent(&current) == CUDA_SUCCESS ? current : nullptr;
}

CUstream lupine_gpu_translate_stream(CUstream s) {
  std::lock_guard<std::mutex> lock(g_mutex);
  if (!g_restored || s == nullptr) return s;
  auto it = g_stream_remap.find(s);
  if (it != g_stream_remap.end()) return it->second;

  CUstream replacement = nullptr;
  CUresult result = cuStreamCreate(&replacement, CU_STREAM_DEFAULT);
  if (result != CUDA_SUCCESS) return nullptr;
  g_stream_remap[s] = replacement;
  return replacement;
}

int lupine_gpu_restored(void) {
  std::lock_guard<std::mutex> lock(g_mutex);
  return g_restored ? 1 : 0;
}

int handle_manual_cuModuleGetFunction_tracked(conn_t *conn) {
  CUmodule hmod = nullptr;
  size_t name_len = 0;
  if (rpc_read(conn, &hmod, sizeof(hmod)) < 0 ||
      rpc_read(conn, &name_len, sizeof(name_len)) < 0)
    return -1;
  std::vector<char> name(name_len + 1, '\0');
  if (name_len && rpc_read(conn, name.data(), name_len) < 0) return -1;
  int request_id = rpc_read_end(conn);
  if (request_id < 0) return -1;

  hmod = lupine_gpu_translate_module(hmod);
  CUfunction hfunc = nullptr;
  CUresult result = cuModuleGetFunction(&hfunc, hmod, name.data());
  if (result == CUDA_SUCCESS) lupine_gpu_track_function(hfunc, hmod, name.data());

  if (rpc_write_start_response(conn, request_id) < 0 ||
      rpc_write(conn, &hfunc, sizeof(hfunc)) < 0 ||
      rpc_write(conn, &result, sizeof(result)) < 0 || rpc_write_end(conn) < 0)
    return -1;
  return 0;
}

int handle_manual_cuLibraryGetKernel_tracked(conn_t *conn) {
  CUlibrary library = nullptr;
  size_t name_len = 0;
  if (rpc_read(conn, &library, sizeof(library)) < 0 ||
      rpc_read(conn, &name_len, sizeof(name_len)) < 0)
    return -1;
  std::vector<char> name(name_len + 1, '\0');
  if (name_len && rpc_read(conn, name.data(), name_len) < 0) return -1;
  int request_id = rpc_read_end(conn);
  if (request_id < 0) return -1;

  library = lupine_gpu_translate_library(library);
  CUkernel kernel = nullptr;
  CUresult result = cuLibraryGetKernel(&kernel, library, name.data());
  if (result == CUDA_SUCCESS) lupine_gpu_track_kernel(kernel, library, name.data());

  if (rpc_write_start_response(conn, request_id) < 0 ||
      rpc_write(conn, &kernel, sizeof(kernel)) < 0 ||
      rpc_write(conn, &result, sizeof(result)) < 0 || rpc_write_end(conn) < 0)
    return -1;
  return 0;
}

int handle_manual_cuKernelGetFunction_tracked(conn_t *conn) {
  CUkernel kernel = nullptr;
  if (rpc_read(conn, &kernel, sizeof(kernel)) < 0) return -1;
  int request_id = rpc_read_end(conn);
  if (request_id < 0) return -1;

  kernel = lupine_gpu_translate_kernel(kernel);
  CUfunction fn = nullptr;
  CUresult result = cuKernelGetFunction(&fn, kernel);
  if (result == CUDA_SUCCESS) lupine_gpu_track_kernel_function(fn, kernel);

  if (rpc_write_start_response(conn, request_id) < 0 ||
      rpc_write(conn, &fn, sizeof(fn)) < 0 ||
      rpc_write(conn, &result, sizeof(result)) < 0 || rpc_write_end(conn) < 0)
    return -1;
  return 0;
}
