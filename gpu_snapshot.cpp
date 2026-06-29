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
  return result;
}

CUresult lupine_gpu_snapshot_restore(const char *id) {
  std::lock_guard<std::mutex> lock(g_mutex);
  if (g_inited) {
    LUPINE_LOG_ERROR("gpu_snapshot: restore into an already-initialized arena");
    return CUDA_ERROR_UNKNOWN;
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
