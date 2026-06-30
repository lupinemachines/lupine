#include "gpu_snapshot.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <mutex>
#include <string>
#include <vector>

#ifdef _WIN32
#include <direct.h>
#define LUPINE_MKDIR(p) _mkdir(p)
#else
#include <sys/stat.h>
#define LUPINE_MKDIR(p) mkdir((p), 0777)
#endif

#include "lupine_fatbin.h"
#include "lupine_log.h"

namespace {

struct Allocation {
  size_t size;    // bytes the client asked for
  size_t rounded; // bytes actually mapped (granularity-aligned)
  CUmemGenericAllocationHandle handle;
};

// One VMM virtual-address arena per worker. Device pointers are offsets into
// g_arena_base; with ASLR disabled the driver picks the same base in every
// worker, so the layout is reproducible by re-reserving that base on restore.
std::mutex g_mutex;
bool g_inited = false;
CUdeviceptr g_arena_base = 0;
size_t g_arena_size = 0;
size_t g_granularity = 0;
int g_device = 0;
size_t g_bump = 0; // next never-used offset
std::map<CUdeviceptr, Allocation> g_allocs;
std::map<size_t, size_t> g_free; // freed (offset -> rounded size) for reuse

// --- opaque-object tracking + remap ---------------------------------------
// A fresh worker created on restore has new CUmodule/CUfunction/CUstream/
// CUcontext values, but the client still holds the originals (it never
// dereferences them). We record how each was created, replay them on restore,
// and translate the client's originals to the live handles.
struct ModuleRec {
  unsigned int kind;
  std::vector<unsigned char> image;
};
std::map<CUmodule, ModuleRec> g_modules;
std::map<CUfunction, std::pair<CUmodule, std::string>> g_functions;
std::map<CUlibrary, ModuleRec> g_libraries;
std::map<CUkernel, std::pair<CUlibrary, std::string>> g_kernels;
std::map<CUfunction, CUkernel> g_kernel_functions;
std::map<CUstream, unsigned int> g_streams;
std::map<CUevent, unsigned int> g_events;
CUcontext g_primary_ctx = nullptr;
std::map<CUmodule, CUmodule> g_module_remap;
std::map<CUlibrary, CUlibrary> g_library_remap;
std::map<CUkernel, CUkernel> g_kernel_remap;
std::map<CUfunction, CUfunction> g_function_remap;
std::map<CUstream, CUstream> g_stream_remap;
std::map<CUevent, CUevent> g_event_remap;
std::map<CUcontext, CUcontext> g_context_remap;

// Mirror of manual_server.cpp's module-image kinds.
constexpr unsigned int K_FATBINC_V1 = 1, K_FATBIN_RAW = 2, K_FATBINC_V2 = 3;

size_t round_up(size_t x, size_t g) {
  if (g == 0) return x;
  return ((x + g - 1) / g) * g;
}

size_t default_arena_bytes() {
  if (const char *env = getenv("LUPINE_GPU_ARENA_BYTES")) {
    unsigned long long v = strtoull(env, nullptr, 0);
    if (v > 0) return (size_t)v;
  }
  return (size_t)64 << 30; // 64 GiB of VA (cheap; backed physically on demand)
}

void fill_prop(CUmemAllocationProp *prop) {
  memset(prop, 0, sizeof(*prop));
  prop->type = CU_MEM_ALLOCATION_TYPE_PINNED;
  prop->location.type = CU_MEM_LOCATION_TYPE_DEVICE;
  prop->location.id = g_device;
}

// Reserve the arena. fixed_base != 0 requests that exact base (restore).
CUresult reserve_arena(CUdeviceptr fixed_base) {
  if (cuCtxGetDevice(&g_device) != CUDA_SUCCESS) g_device = 0;
  CUmemAllocationProp prop;
  fill_prop(&prop);
  size_t gran = 0;
  CUresult r = cuMemGetAllocationGranularity(
      &gran, &prop, CU_MEM_ALLOC_GRANULARITY_RECOMMENDED);
  if (r != CUDA_SUCCESS) return r;
  g_granularity = gran;
  g_arena_size = round_up(default_arena_bytes(), gran);

  CUdeviceptr base = 0;
  r = cuMemAddressReserve(&base, g_arena_size, gran, fixed_base, 0);
  if (r != CUDA_SUCCESS) return r;
  if (fixed_base != 0 && base != fixed_base) {
    LUPINE_LOG_ERROR("gpu_snapshot: arena base moved on restore (wanted "
                     << (void *)fixed_base << " got " << (void *)base
                     << "); is ASLR disabled?");
    cuMemAddressFree(base, g_arena_size);
    return CUDA_ERROR_OUT_OF_MEMORY;
  }
  g_arena_base = base;
  g_bump = 0;
  g_inited = true;
  return CUDA_SUCCESS;
}

// Map physical memory at [base+offset, +rounded) with device RW access.
CUresult map_block(size_t offset, size_t rounded,
                   CUmemGenericAllocationHandle *out) {
  CUmemAllocationProp prop;
  fill_prop(&prop);
  CUmemGenericAllocationHandle handle = 0;
  CUresult r = cuMemCreate(&handle, rounded, &prop, 0);
  if (r != CUDA_SUCCESS) return r;
  CUdeviceptr addr = g_arena_base + offset;
  r = cuMemMap(addr, rounded, 0, handle, 0);
  if (r != CUDA_SUCCESS) {
    cuMemRelease(handle);
    return r;
  }
  CUmemAccessDesc access;
  memset(&access, 0, sizeof(access));
  access.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
  access.location.id = g_device;
  access.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
  r = cuMemSetAccess(addr, rounded, &access, 1);
  if (r != CUDA_SUCCESS) {
    cuMemUnmap(addr, rounded);
    cuMemRelease(handle);
    return r;
  }
  *out = handle;
  return CUDA_SUCCESS;
}

// Pick an offset for a rounded-byte block: reuse a freed block or bump.
bool take_offset(size_t rounded, size_t *offset) {
  for (auto it = g_free.begin(); it != g_free.end(); ++it) {
    if (it->second >= rounded) {
      *offset = it->first;
      if (it->second > rounded)
        g_free.emplace(it->first + rounded, it->second - rounded);
      g_free.erase(it);
      return true;
    }
  }
  if (g_bump + rounded > g_arena_size) return false;
  *offset = g_bump;
  g_bump += rounded;
  return true;
}

uint64_t fnv1a(const char *s) {
  uint64_t h = 1469598103934665603ULL;
  for (; *s; ++s) {
    h ^= (unsigned char)*s;
    h *= 1099511628211ULL;
  }
  return h;
}

// base dir + hashed id, created if needed. Returns "" on failure.
std::string snapshot_dir(const char *id) {
  const char *base = getenv("LUPINE_SNAPSHOT_DIR");
  std::string dir = base && base[0] ? base : "/tmp/lupine-snapshots";
  LUPINE_MKDIR(dir.c_str());
  char hex[17];
  snprintf(hex, sizeof(hex), "%016llx", (unsigned long long)fnv1a(id));
  dir += "/";
  dir += hex;
  if (LUPINE_MKDIR(dir.c_str()) != 0 && errno != EEXIST) {
    LUPINE_LOG_ERROR("gpu_snapshot: cannot create " << dir);
    return "";
  }
  return dir;
}

std::string join(const std::string &dir, const char *name) {
  std::string s = dir;
  if (!s.empty() && s.back() != '/') s.push_back('/');
  return s + name;
}

struct ManifestHeader {
  char magic[8]; // "LUPGPU01"
  uint64_t arena_base;
  uint64_t granularity;
  uint64_t count;
};

struct ManifestEntry {
  uint64_t offset; // addr - arena_base
  uint64_t size;   // requested bytes (also bytes stored in gpu.mem)
  uint64_t rounded;
};

// Read a length-prefixed snapshot id off the wire into id[<=cap].
int read_id(conn_t *conn, char *id, size_t cap) {
  uint32_t len = 0;
  if (rpc_read(conn, &len, sizeof(len)) < 0) return -1;
  if (len == 0 || len >= cap) return -1;
  if (rpc_read(conn, id, len) < 0) return -1;
  id[len] = '\0';
  return 0;
}

void write_u64(FILE *f, uint64_t v) { fwrite(&v, sizeof(v), 1, f); }
bool read_u64(FILE *f, uint64_t *v) { return fread(v, sizeof(*v), 1, f) == 1; }

void write_blob(FILE *f, const std::vector<unsigned char> &b) {
  write_u64(f, b.size());
  if (!b.empty()) fwrite(b.data(), 1, b.size(), f);
}
void write_str(FILE *f, const std::string &s) {
  write_u64(f, s.size());
  if (!s.empty()) fwrite(s.data(), 1, s.size(), f);
}
bool read_blob(FILE *f, std::vector<unsigned char> &b) {
  uint64_t n;
  if (!read_u64(f, &n)) return false;
  b.resize(n);
  return n == 0 || fread(b.data(), 1, n, f) == n;
}
bool read_str(FILE *f, std::string &s) {
  uint64_t n;
  if (!read_u64(f, &n)) return false;
  s.assign(n, '\0');
  return n == 0 || fread(&s[0], 1, n, f) == n;
}

// Serialize the tracked objects. Called while holding g_mutex. Sections are
// ordered so restore can replay dependencies first (module before its
// functions; library before kernels before kernel-functions).
bool save_objects(const std::string &dir) {
  fprintf(stderr,
          "gpu_snapshot save: modules=%zu funcs=%zu libs=%zu kernels=%zu "
          "kfuncs=%zu streams=%zu events=%zu ctx=%p\n",
          g_modules.size(), g_functions.size(), g_libraries.size(),
          g_kernels.size(), g_kernel_functions.size(), g_streams.size(),
          g_events.size(), (void *)g_primary_ctx);
  FILE *f = fopen(join(dir, "gpu.objects").c_str(), "wb");
  if (!f) return false;
  write_u64(f, g_modules.size());
  for (auto &kv : g_modules) {
    write_u64(f, (uint64_t)kv.first);
    write_u64(f, kv.second.kind);
    write_blob(f, kv.second.image);
  }
  write_u64(f, g_functions.size());
  for (auto &kv : g_functions) {
    write_u64(f, (uint64_t)kv.first);
    write_u64(f, (uint64_t)kv.second.first);
    write_str(f, kv.second.second);
  }
  write_u64(f, g_libraries.size());
  for (auto &kv : g_libraries) {
    write_u64(f, (uint64_t)kv.first);
    write_u64(f, kv.second.kind);
    write_blob(f, kv.second.image);
  }
  write_u64(f, g_kernels.size());
  for (auto &kv : g_kernels) {
    write_u64(f, (uint64_t)kv.first);
    write_u64(f, (uint64_t)kv.second.first);
    write_str(f, kv.second.second);
  }
  write_u64(f, g_kernel_functions.size());
  for (auto &kv : g_kernel_functions) {
    write_u64(f, (uint64_t)kv.first);
    write_u64(f, (uint64_t)kv.second);
  }
  write_u64(f, g_streams.size());
  for (auto &kv : g_streams) {
    write_u64(f, (uint64_t)kv.first);
    write_u64(f, kv.second);
  }
  write_u64(f, g_events.size());
  for (auto &kv : g_events) {
    write_u64(f, (uint64_t)kv.first);
    write_u64(f, kv.second);
  }
  write_u64(f, (uint64_t)g_primary_ctx);
  fclose(f);
  return true;
}

CUresult load_module_image(unsigned int kind, const unsigned char *image,
                           CUmodule *out) {
  if (kind == K_FATBINC_V1 || kind == K_FATBINC_V2)
    return cuModuleLoadFatBinary(out, image);
  if (kind == K_FATBIN_RAW)
    return cuModuleLoadData(out, image);
  return CUDA_ERROR_NOT_SUPPORTED;
}

CUresult load_library_image(unsigned int kind, const unsigned char *image,
                            CUlibrary *out) {
  if (kind == K_FATBINC_V1 || kind == K_FATBINC_V2) {
    lupine_fatbin_wrapper wrapper = {LUPINE_FATBINC_MAGIC,
                                     kind == K_FATBINC_V2 ? 2U : 1U, image,
                                     nullptr};
    return cuLibraryLoadData(out, &wrapper, nullptr, nullptr, 0, nullptr,
                             nullptr, 0);
  }
  if (kind == K_FATBIN_RAW)
    return cuLibraryLoadData(out, image, nullptr, nullptr, 0, nullptr, nullptr,
                             0);
  return CUDA_ERROR_NOT_SUPPORTED;
}

// Replay tracked objects and build the old->new remaps. Called while holding
// g_mutex, after device memory + a current context are restored.
CUresult restore_objects(const std::string &dir) {
  FILE *f = fopen(join(dir, "gpu.objects").c_str(), "rb");
  if (!f) return CUDA_SUCCESS; // pure-driver snapshot has no objects
  uint64_t n = 0;
  CUresult rc = CUDA_ERROR_UNKNOWN;
  if (!read_u64(f, &n)) goto done;
  for (uint64_t i = 0; i < n; i++) { // modules
    uint64_t old_h, kind;
    std::vector<unsigned char> image;
    if (!read_u64(f, &old_h) || !read_u64(f, &kind) || !read_blob(f, image))
      goto done;
    CUmodule m = nullptr;
    rc = load_module_image((unsigned)kind, image.data(), &m);
    if (rc != CUDA_SUCCESS) goto done;
    g_modules[m] = ModuleRec{(unsigned)kind, std::move(image)};
    g_module_remap[(CUmodule)old_h] = m;
  }
  if (!read_u64(f, &n)) { rc = CUDA_ERROR_UNKNOWN; goto done; }
  for (uint64_t i = 0; i < n; i++) { // module functions
    uint64_t old_fn, old_mod;
    std::string name;
    if (!read_u64(f, &old_fn) || !read_u64(f, &old_mod) || !read_str(f, name)) {
      rc = CUDA_ERROR_UNKNOWN;
      goto done;
    }
    auto it = g_module_remap.find((CUmodule)old_mod);
    if (it == g_module_remap.end()) { rc = CUDA_ERROR_UNKNOWN; goto done; }
    CUfunction fn = nullptr;
    rc = cuModuleGetFunction(&fn, it->second, name.c_str());
    if (rc != CUDA_SUCCESS) goto done;
    g_functions[fn] = {it->second, name};
    g_function_remap[(CUfunction)old_fn] = fn;
  }
  if (!read_u64(f, &n)) { rc = CUDA_ERROR_UNKNOWN; goto done; }
  for (uint64_t i = 0; i < n; i++) { // libraries
    uint64_t old_h, kind;
    std::vector<unsigned char> image;
    if (!read_u64(f, &old_h) || !read_u64(f, &kind) || !read_blob(f, image)) {
      rc = CUDA_ERROR_UNKNOWN;
      goto done;
    }
    CUlibrary lib = nullptr;
    rc = load_library_image((unsigned)kind, image.data(), &lib);
    if (rc != CUDA_SUCCESS) goto done;
    g_libraries[lib] = ModuleRec{(unsigned)kind, std::move(image)};
    g_library_remap[(CUlibrary)old_h] = lib;
  }
  if (!read_u64(f, &n)) { rc = CUDA_ERROR_UNKNOWN; goto done; }
  for (uint64_t i = 0; i < n; i++) { // kernels
    uint64_t old_k, old_lib;
    std::string name;
    if (!read_u64(f, &old_k) || !read_u64(f, &old_lib) || !read_str(f, name)) {
      rc = CUDA_ERROR_UNKNOWN;
      goto done;
    }
    auto it = g_library_remap.find((CUlibrary)old_lib);
    if (it == g_library_remap.end()) { rc = CUDA_ERROR_UNKNOWN; goto done; }
    CUkernel k = nullptr;
    rc = cuLibraryGetKernel(&k, it->second, name.c_str());
    if (rc != CUDA_SUCCESS) goto done;
    g_kernels[k] = {it->second, name};
    g_kernel_remap[(CUkernel)old_k] = k;
  }
  if (!read_u64(f, &n)) { rc = CUDA_ERROR_UNKNOWN; goto done; }
  for (uint64_t i = 0; i < n; i++) { // kernel functions
    uint64_t old_fn, old_k;
    if (!read_u64(f, &old_fn) || !read_u64(f, &old_k)) {
      rc = CUDA_ERROR_UNKNOWN;
      goto done;
    }
    auto it = g_kernel_remap.find((CUkernel)old_k);
    if (it == g_kernel_remap.end()) { rc = CUDA_ERROR_UNKNOWN; goto done; }
    CUfunction fn = nullptr;
    rc = cuKernelGetFunction(&fn, it->second);
    if (rc != CUDA_SUCCESS) goto done;
    g_kernel_functions[fn] = it->second;
    g_function_remap[(CUfunction)old_fn] = fn;
  }
  if (!read_u64(f, &n)) { rc = CUDA_ERROR_UNKNOWN; goto done; }
  for (uint64_t i = 0; i < n; i++) { // streams
    uint64_t old_s, flags;
    if (!read_u64(f, &old_s) || !read_u64(f, &flags)) {
      rc = CUDA_ERROR_UNKNOWN;
      goto done;
    }
    CUstream s = nullptr;
    rc = cuStreamCreate(&s, (unsigned)flags);
    if (rc != CUDA_SUCCESS) goto done;
    g_streams[s] = (unsigned)flags;
    g_stream_remap[(CUstream)old_s] = s;
  }
  if (!read_u64(f, &n)) { rc = CUDA_ERROR_UNKNOWN; goto done; }
  for (uint64_t i = 0; i < n; i++) { // events
    uint64_t old_e, flags;
    if (!read_u64(f, &old_e) || !read_u64(f, &flags)) {
      rc = CUDA_ERROR_UNKNOWN;
      goto done;
    }
    CUevent e = nullptr;
    rc = cuEventCreate(&e, (unsigned)flags);
    if (rc != CUDA_SUCCESS) goto done;
    g_events[e] = (unsigned)flags;
    g_event_remap[(CUevent)old_e] = e;
  }
  {
    uint64_t old_ctx = 0;
    read_u64(f, &old_ctx);
    if (old_ctx) {
      CUcontext cur = nullptr;
      cuCtxGetCurrent(&cur);
      if (cur) {
        g_context_remap[(CUcontext)old_ctx] = cur;
        g_primary_ctx = cur;
      }
    }
  }
  rc = CUDA_SUCCESS;
done:
  fclose(f);
  return rc;
}

} // namespace

CUresult lupine_gpu_alloc(CUdeviceptr *dptr, size_t bytesize) {
  std::lock_guard<std::mutex> lock(g_mutex);
  if (!g_inited) {
    CUresult r = reserve_arena(0);
    if (r != CUDA_SUCCESS) return r;
  }
  size_t rounded = round_up(bytesize == 0 ? 1 : bytesize, g_granularity);
  size_t offset = 0;
  if (!take_offset(rounded, &offset)) return CUDA_ERROR_OUT_OF_MEMORY;
  CUmemGenericAllocationHandle handle = 0;
  CUresult r = map_block(offset, rounded, &handle);
  if (r != CUDA_SUCCESS) {
    g_free.emplace(offset, rounded);
    return r;
  }
  CUdeviceptr addr = g_arena_base + offset;
  g_allocs[addr] = Allocation{bytesize, rounded, handle};
  *dptr = addr;
  return CUDA_SUCCESS;
}

CUresult lupine_gpu_free(CUdeviceptr dptr) {
  std::lock_guard<std::mutex> lock(g_mutex);
  auto it = g_allocs.find(dptr);
  if (it == g_allocs.end()) return CUDA_ERROR_INVALID_VALUE;
  size_t rounded = it->second.rounded;
  cuMemUnmap(dptr, rounded);
  cuMemRelease(it->second.handle);
  g_free.emplace((size_t)(dptr - g_arena_base), rounded);
  g_allocs.erase(it);
  return CUDA_SUCCESS;
}

int lupine_gpu_owns(CUdeviceptr dptr) {
  std::lock_guard<std::mutex> lock(g_mutex);
  if (!g_inited || dptr < g_arena_base || dptr >= g_arena_base + g_arena_size)
    return 0;
  return g_allocs.find(dptr) != g_allocs.end() ? 1 : 0;
}

CUresult lupine_gpu_snapshot_save(const char *id) {
  std::lock_guard<std::mutex> lock(g_mutex);
  std::string dir = snapshot_dir(id);
  if (dir.empty()) return CUDA_ERROR_UNKNOWN;
  FILE *mf = fopen(join(dir, "gpu.manifest").c_str(), "wb");
  FILE *df = fopen(join(dir, "gpu.mem").c_str(), "wb");
  if (!mf || !df) {
    if (mf) fclose(mf);
    if (df) fclose(df);
    LUPINE_LOG_ERROR("gpu_snapshot: cannot open snapshot files under " << dir);
    return CUDA_ERROR_UNKNOWN;
  }

  ManifestHeader hdr;
  memset(&hdr, 0, sizeof(hdr));
  memcpy(hdr.magic, "LUPGPU01", 8);
  hdr.arena_base = g_arena_base;
  hdr.granularity = g_granularity;
  hdr.count = g_allocs.size();
  CUresult result = CUDA_SUCCESS;
  if (fwrite(&hdr, sizeof(hdr), 1, mf) != 1) result = CUDA_ERROR_UNKNOWN;

  std::vector<unsigned char> staging;
  for (const auto &kv : g_allocs) {
    if (result != CUDA_SUCCESS) break;
    ManifestEntry e;
    e.offset = (uint64_t)(kv.first - g_arena_base);
    e.size = kv.second.size;
    e.rounded = kv.second.rounded;
    if (fwrite(&e, sizeof(e), 1, mf) != 1) {
      result = CUDA_ERROR_UNKNOWN;
      break;
    }
    if (e.size == 0) continue;
    staging.resize(e.size);
    CUresult r = cuMemcpyDtoH(staging.data(), kv.first, e.size);
    if (r != CUDA_SUCCESS) {
      result = r;
      break;
    }
    if (fwrite(staging.data(), 1, e.size, df) != e.size)
      result = CUDA_ERROR_UNKNOWN;
  }

  fclose(mf);
  fclose(df);
  if (result == CUDA_SUCCESS && !save_objects(dir)) result = CUDA_ERROR_UNKNOWN;
  return result;
}

CUresult lupine_gpu_snapshot_restore(const char *id) {
  std::lock_guard<std::mutex> lock(g_mutex);
  if (g_inited) {
    LUPINE_LOG_ERROR("gpu_snapshot: restore into an already-initialized arena");
    return CUDA_ERROR_UNKNOWN;
  }
  // A reconnecting worker receives restore as its first op, before the client
  // re-establishes a context, so set one up ourselves.
  CUcontext cur = nullptr;
  if (cuCtxGetCurrent(&cur) != CUDA_SUCCESS || cur == nullptr) {
    cuInit(0);
    CUdevice dev;
    CUcontext ctx;
    if (cuDeviceGet(&dev, 0) != CUDA_SUCCESS ||
        cuDevicePrimaryCtxRetain(&ctx, dev) != CUDA_SUCCESS ||
        cuCtxSetCurrent(ctx) != CUDA_SUCCESS) {
      LUPINE_LOG_ERROR("gpu_snapshot: restore could not establish a context");
      return CUDA_ERROR_UNKNOWN;
    }
  }
  std::string dir = snapshot_dir(id);
  if (dir.empty()) return CUDA_ERROR_UNKNOWN;
  FILE *mf = fopen(join(dir, "gpu.manifest").c_str(), "rb");
  FILE *df = fopen(join(dir, "gpu.mem").c_str(), "rb");
  if (!mf || !df) {
    if (mf) fclose(mf);
    if (df) fclose(df);
    LUPINE_LOG_ERROR("gpu_snapshot: cannot open snapshot files under " << dir);
    return CUDA_ERROR_UNKNOWN;
  }

  ManifestHeader hdr;
  if (fread(&hdr, sizeof(hdr), 1, mf) != 1 ||
      memcmp(hdr.magic, "LUPGPU01", 8) != 0) {
    fclose(mf);
    fclose(df);
    LUPINE_LOG_ERROR("gpu_snapshot: bad manifest in " << dir);
    return CUDA_ERROR_UNKNOWN;
  }

  CUresult r = reserve_arena((CUdeviceptr)hdr.arena_base);
  if (r != CUDA_SUCCESS) {
    fclose(mf);
    fclose(df);
    return r;
  }

  CUresult result = CUDA_SUCCESS;
  std::vector<unsigned char> staging;
  size_t max_end = 0;
  for (uint64_t i = 0; i < hdr.count; ++i) {
    ManifestEntry e;
    if (fread(&e, sizeof(e), 1, mf) != 1) {
      result = CUDA_ERROR_UNKNOWN;
      break;
    }
    CUmemGenericAllocationHandle handle = 0;
    r = map_block(e.offset, e.rounded, &handle);
    if (r != CUDA_SUCCESS) {
      result = r;
      break;
    }
    CUdeviceptr addr = g_arena_base + e.offset;
    if (e.size > 0) {
      staging.resize(e.size);
      if (fread(staging.data(), 1, e.size, df) != e.size) {
        result = CUDA_ERROR_UNKNOWN;
        break;
      }
      r = cuMemcpyHtoD(addr, staging.data(), e.size);
      if (r != CUDA_SUCCESS) {
        result = r;
        break;
      }
    }
    g_allocs[addr] = Allocation{e.size, e.rounded, handle};
    if (e.offset + e.rounded > max_end) max_end = e.offset + e.rounded;
  }
  g_bump = max_end;

  fclose(mf);
  fclose(df);
  if (result == CUDA_SUCCESS) result = restore_objects(dir);
  return result;
}

int handle_manual_cuMemAlloc_v2(conn_t *conn) {
  CUdeviceptr dptr;
  size_t bytesize;
  if (rpc_read(conn, &dptr, sizeof(CUdeviceptr)) < 0 ||
      rpc_read(conn, &bytesize, sizeof(size_t)) < 0)
    return -1;
  int request_id = rpc_read_end(conn);
  if (request_id < 0) return -1;
  CUresult result = lupine_gpu_alloc(&dptr, bytesize);
  if (rpc_write_start_response(conn, request_id) < 0 ||
      rpc_write(conn, &dptr, sizeof(CUdeviceptr)) < 0 ||
      rpc_write(conn, &result, sizeof(CUresult)) < 0 ||
      rpc_write_end(conn) < 0)
    return -1;
  return 0;
}

int handle_manual_cuMemFree_v2(conn_t *conn) {
  CUdeviceptr dptr;
  if (rpc_read(conn, &dptr, sizeof(CUdeviceptr)) < 0) return -1;
  int request_id = rpc_read_end(conn);
  if (request_id < 0) return -1;
  CUresult result =
      lupine_gpu_owns(dptr) ? lupine_gpu_free(dptr) : cuMemFree_v2(dptr);
  if (rpc_write_start_response(conn, request_id) < 0 ||
      rpc_write(conn, &result, sizeof(CUresult)) < 0 ||
      rpc_write_end(conn) < 0)
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
      rpc_write(conn, &result, sizeof(CUresult)) < 0 ||
      rpc_write_end(conn) < 0)
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
      rpc_write(conn, &result, sizeof(CUresult)) < 0 ||
      rpc_write_end(conn) < 0)
    return -1;
  return 0;
}

// --- object tracking + handle translation ---------------------------------

void lupine_gpu_track_module(CUmodule m, unsigned int kind, const void *image,
                             size_t size) {
  if (!m) return;
  std::lock_guard<std::mutex> lock(g_mutex);
  ModuleRec r;
  r.kind = kind;
  r.image.assign((const unsigned char *)image,
                 (const unsigned char *)image + size);
  g_modules[m] = std::move(r);
}

void lupine_gpu_track_function(CUfunction fn, CUmodule m, const char *name) {
  if (!fn) return;
  std::lock_guard<std::mutex> lock(g_mutex);
  g_functions[fn] = {m, std::string(name ? name : "")};
}

void lupine_gpu_track_stream(CUstream s, unsigned int flags) {
  if (!s) return;
  std::lock_guard<std::mutex> lock(g_mutex);
  g_streams[s] = flags;
}

void lupine_gpu_track_primary_ctx(CUcontext ctx) {
  std::lock_guard<std::mutex> lock(g_mutex);
  g_primary_ctx = ctx;
}

CUfunction lupine_gpu_xlate_function(CUfunction fn) {
  std::lock_guard<std::mutex> lock(g_mutex);
  auto it = g_function_remap.find(fn);
  if (it != g_function_remap.end()) return it->second;
  // torch launches a CUkernel directly as a CUfunction (no cuKernelGetFunction),
  // so fall back to the kernel remap.
  auto k = g_kernel_remap.find((CUkernel)fn);
  if (k != g_kernel_remap.end()) return (CUfunction)k->second;
  return fn;
}

CUmodule lupine_gpu_xlate_module(CUmodule m) {
  std::lock_guard<std::mutex> lock(g_mutex);
  auto it = g_module_remap.find(m);
  return it == g_module_remap.end() ? m : it->second;
}

CUstream lupine_gpu_xlate_stream(CUstream s) {
  std::lock_guard<std::mutex> lock(g_mutex);
  auto it = g_stream_remap.find(s);
  return it == g_stream_remap.end() ? s : it->second;
}

CUcontext lupine_gpu_xlate_context(CUcontext c) {
  std::lock_guard<std::mutex> lock(g_mutex);
  auto it = g_context_remap.find(c);
  return it == g_context_remap.end() ? c : it->second;
}

int handle_manual_cuModuleGetFunction_tracked(conn_t *conn) {
  CUmodule hmod;
  size_t name_len;
  if (rpc_read(conn, &hmod, sizeof(CUmodule)) < 0 ||
      rpc_read(conn, &name_len, sizeof(size_t)) < 0)
    return -1;
  std::vector<char> name(name_len + 1, '\0');
  if (name_len && rpc_read(conn, name.data(), name_len) < 0) return -1;
  int request_id = rpc_read_end(conn);
  if (request_id < 0) return -1;
  CUfunction hfunc = nullptr;
  CUresult result = cuModuleGetFunction(&hfunc, hmod, name.data());
  if (result == CUDA_SUCCESS) lupine_gpu_track_function(hfunc, hmod, name.data());
  if (rpc_write_start_response(conn, request_id) < 0 ||
      rpc_write(conn, &hfunc, sizeof(CUfunction)) < 0 ||
      rpc_write(conn, &result, sizeof(CUresult)) < 0 || rpc_write_end(conn) < 0)
    return -1;
  return 0;
}

int handle_manual_cuStreamCreate_tracked(conn_t *conn) {
  CUstream phStream;
  unsigned int Flags;
  if (rpc_read(conn, &phStream, sizeof(CUstream)) < 0 ||
      rpc_read(conn, &Flags, sizeof(unsigned int)) < 0)
    return -1;
  int request_id = rpc_read_end(conn);
  if (request_id < 0) return -1;
  CUresult result = cuStreamCreate(&phStream, Flags);
  if (result == CUDA_SUCCESS) lupine_gpu_track_stream(phStream, Flags);
  if (rpc_write_start_response(conn, request_id) < 0 ||
      rpc_write(conn, &phStream, sizeof(CUstream)) < 0 ||
      rpc_write(conn, &result, sizeof(CUresult)) < 0 || rpc_write_end(conn) < 0)
    return -1;
  return 0;
}

int handle_manual_cuDevicePrimaryCtxRetain_tracked(conn_t *conn) {
  CUcontext pctx;
  CUdevice dev;
  if (rpc_read(conn, &dev, sizeof(CUdevice)) < 0) return -1;
  int request_id = rpc_read_end(conn);
  if (request_id < 0) return -1;
  CUresult result = cuDevicePrimaryCtxRetain(&pctx, dev);
  if (result == CUDA_SUCCESS) lupine_gpu_track_primary_ctx(pctx);
  if (rpc_write_start_response(conn, request_id) < 0 ||
      rpc_write(conn, &pctx, sizeof(CUcontext)) < 0 ||
      rpc_write(conn, &result, sizeof(CUresult)) < 0 || rpc_write_end(conn) < 0)
    return -1;
  return 0;
}

void lupine_gpu_track_library(CUlibrary lib, unsigned int kind,
                              const void *image, size_t size) {
  if (!lib) return;
  std::lock_guard<std::mutex> lock(g_mutex);
  ModuleRec r;
  r.kind = kind;
  r.image.assign((const unsigned char *)image,
                 (const unsigned char *)image + size);
  g_libraries[lib] = std::move(r);
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

int handle_manual_cuLibraryGetKernel_tracked(conn_t *conn) {
  CUlibrary library;
  size_t name_len;
  if (rpc_read(conn, &library, sizeof(CUlibrary)) < 0 ||
      rpc_read(conn, &name_len, sizeof(size_t)) < 0)
    return -1;
  std::vector<char> name(name_len + 1, '\0');
  if (name_len && rpc_read(conn, name.data(), name_len) < 0) return -1;
  int request_id = rpc_read_end(conn);
  if (request_id < 0) return -1;
  CUkernel pKernel = nullptr;
  CUresult result = cuLibraryGetKernel(&pKernel, library, name.data());
  if (result == CUDA_SUCCESS)
    lupine_gpu_track_kernel(pKernel, library, name.data());
  if (rpc_write_start_response(conn, request_id) < 0 ||
      rpc_write(conn, &pKernel, sizeof(CUkernel)) < 0 ||
      rpc_write(conn, &result, sizeof(CUresult)) < 0 || rpc_write_end(conn) < 0)
    return -1;
  return 0;
}

int handle_manual_cuKernelGetFunction_tracked(conn_t *conn) {
  CUkernel kernel;
  if (rpc_read(conn, &kernel, sizeof(CUkernel)) < 0) return -1;
  int request_id = rpc_read_end(conn);
  if (request_id < 0) return -1;
  CUfunction pFunc = nullptr;
  CUresult result = cuKernelGetFunction(&pFunc, kernel);
  if (result == CUDA_SUCCESS) lupine_gpu_track_kernel_function(pFunc, kernel);
  if (rpc_write_start_response(conn, request_id) < 0 ||
      rpc_write(conn, &pFunc, sizeof(CUfunction)) < 0 ||
      rpc_write(conn, &result, sizeof(CUresult)) < 0 || rpc_write_end(conn) < 0)
    return -1;
  return 0;
}

void lupine_gpu_track_event(CUevent e, unsigned int flags) {
  if (!e) return;
  std::lock_guard<std::mutex> lock(g_mutex);
  g_events[e] = flags;
}

CUevent lupine_gpu_xlate_event(CUevent e) {
  std::lock_guard<std::mutex> lock(g_mutex);
  auto it = g_event_remap.find(e);
  return it == g_event_remap.end() ? e : it->second;
}

int handle_manual_cuEventCreate_tracked(conn_t *conn) {
  CUevent phEvent;
  unsigned int Flags;
  if (rpc_read(conn, &phEvent, sizeof(CUevent)) < 0 ||
      rpc_read(conn, &Flags, sizeof(unsigned int)) < 0)
    return -1;
  int request_id = rpc_read_end(conn);
  if (request_id < 0) return -1;
  CUresult result = cuEventCreate(&phEvent, Flags);
  if (result == CUDA_SUCCESS) lupine_gpu_track_event(phEvent, Flags);
  if (rpc_write_start_response(conn, request_id) < 0 ||
      rpc_write(conn, &phEvent, sizeof(CUevent)) < 0 ||
      rpc_write(conn, &result, sizeof(CUresult)) < 0 || rpc_write_end(conn) < 0)
    return -1;
  return 0;
}
