#include "checkpoint_objects.h"
#include "codegen/gen_rpc_ids.h"
#include "lupine_log.h"

#include <algorithm>
#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string_view>
#ifndef _WIN32
#include <unistd.h>
#endif
#include <map>
#include <mutex>

extern "C" CUresult CUDAAPI cuCtxCreate_v2(CUcontext *, unsigned, CUdevice);

namespace lupine_objects {
namespace {
struct object {
  kind type;
  uintptr_t id, parent;
  uint32_t flags;
  int32_t value;
  std::vector<unsigned char> data;
  uintptr_t handle;
};
std::mutex mutex;
std::map<uintptr_t, object> objects;
uintptr_t next_id = 0x10000;
std::string unsupported_reason;
thread_local std::vector<std::vector<uintptr_t>> wire_buffers;

uintptr_t find(kind type, uintptr_t handle) {
  for (const auto &entry : objects)
    if (entry.second.type == type && entry.second.handle == handle)
      return entry.first;
  return 0;
}

void add(kind type, uintptr_t handle, uintptr_t parent, unsigned flags,
         int value, const void *bytes = nullptr, size_t size = 0) {
  if (!enabled() || !handle)
    return;
  std::lock_guard<std::mutex> lock(mutex);
  if (find(type, handle))
    return;
  object entry{type, next_id++, parent, flags, value, {}, handle};
  if (size)
    entry.data.assign(static_cast<const unsigned char *>(bytes),
                      static_cast<const unsigned char *>(bytes) + size);
  objects.emplace(entry.id, std::move(entry));
}

uintptr_t current_context() {
  CUcontext context = nullptr;
  if (cuCtxGetCurrent(&context) != CUDA_SUCCESS || !context) {
    unsupported("object without a current context");
    return 0;
  }
  return wire(kind::context, reinterpret_cast<uintptr_t>(context));
}
} // namespace

bool enabled() {
  static const bool value = [] {
    const char *setting = std::getenv("LUPINE_CHECKPOINT_OBJECTS");
    return setting && std::strcmp(setting, "1") == 0;
  }();
  return value;
}

void unsupported(const char *reason) {
  if (!enabled())
    return;
  std::lock_guard<std::mutex> lock(mutex);
  if (unsupported_reason.empty())
    unsupported_reason = reason;
}

uintptr_t native(kind type, uintptr_t id) {
  if (!enabled() || id <= 2)
    return id; // null, legacy and per-thread streams
  std::lock_guard<std::mutex> lock(mutex);
  auto entry = objects.find(id);
  return entry != objects.end() && entry->second.type == type
             ? entry->second.handle
             : id;
}

uintptr_t wire(kind type, uintptr_t handle, bool required) {
  if (!enabled() || handle <= 2)
    return handle;
  std::lock_guard<std::mutex> lock(mutex);
  uintptr_t id = find(type, handle);
  if (!id && required && unsupported_reason.empty())
    unsupported_reason = "untracked CUDA handle";
  return id ? id : handle;
}

const void *wire_values(kind type, const void *handles, size_t size) {
  wire_buffers.emplace_back(size / sizeof(uintptr_t));
  auto &buffer = wire_buffers.back();
  std::memcpy(buffer.data(), handles, size);
  for (auto &value : buffer)
    value = wire(type, value);
  return buffer.data();
}
void finish_call() { wire_buffers.clear(); }

void context(CUcontext handle, CUdevice device, unsigned flags, bool primary) {
  add(kind::context, reinterpret_cast<uintptr_t>(handle), primary, flags,
      device);
}
// Recreating a module image does not recreate mutable module globals. Accept
// plain PTX/cubin only when the image proves there are none; opaque fatbins
// need separate coverage before they can participate in a coordinated
// checkpoint.
static bool module_without_mutable_globals(const void *image, size_t size,
                                           bool fatbinary) {
  if (!image || !size || fatbinary)
    return false;
  const auto *bytes = static_cast<const unsigned char *>(image);
  if (size >= 4 && std::memcmp(bytes,
                               "\x7f"
                               "ELF",
                               4) == 0) {
    Elf64_Ehdr header;
    if (size < sizeof(header))
      return false;
    std::memcpy(&header, bytes, sizeof(header));
    if (header.e_ident[EI_CLASS] != ELFCLASS64 ||
        header.e_ident[5] != 1 /* little-endian */ ||
        header.e_shentsize != sizeof(Elf64_Shdr) || !header.e_shnum ||
        header.e_shoff > size ||
        header.e_shnum > (size - header.e_shoff) / sizeof(Elf64_Shdr))
      return false;
    for (size_t i = 0; i < header.e_shnum; ++i) {
      Elf64_Shdr section;
      std::memcpy(&section, bytes + header.e_shoff + i * sizeof(section),
                  sizeof(section));
      if (section.sh_size && (section.sh_flags & 3) == 3)
        return false; // SHF_WRITE | SHF_ALLOC
    }
    return true;
  }
  std::string_view ptx(static_cast<const char *>(image), size);
  if (ptx.find(".version") == ptx.npos)
    return false;
  for (size_t at = 0; (at = ptx.find(".global", at)) != ptx.npos; at += 7)
    if (at + 7 < size && ptx[at + 7] != '.')
      return false;
  return true;
}

void module(CUmodule handle, const void *image, size_t size, bool fatbinary) {
  if (enabled()) {
    if (!module_without_mutable_globals(image, size, fatbinary))
      unsupported("module globals or uninspected module image");
    add(kind::module, reinterpret_cast<uintptr_t>(handle), current_context(),
        fatbinary, 0, image, size);
  }
}
void function(CUfunction handle, CUmodule parent, const char *name) {
  if (enabled())
    add(kind::function, reinterpret_cast<uintptr_t>(handle),
        wire(kind::module, reinterpret_cast<uintptr_t>(parent)), 0, 0, name,
        std::strlen(name) + 1);
}
void stream(CUstream handle, unsigned flags, int priority) {
  if (enabled())
    add(kind::stream, reinterpret_cast<uintptr_t>(handle), current_context(),
        flags, priority);
}
void event(CUevent handle, unsigned flags) {
  if (enabled())
    add(kind::event, reinterpret_cast<uintptr_t>(handle), current_context(),
        flags, 0);
}
void erase(kind type, uintptr_t handle) {
  if (!enabled())
    return;
  std::lock_guard<std::mutex> lock(mutex);
  uintptr_t id = find(type, handle);
  if (!id)
    return;
  // Context and module destruction also invalidates their child objects.
  for (auto it = objects.begin(); it != objects.end();) {
    if (it->first == id || (type == kind::module && it->second.parent == id &&
                            it->second.type == kind::function))
      it = objects.erase(it);
    else
      ++it;
  }
  if (type == kind::context)
    unsupported_reason = "context destroyed before checkpoint";
}

std::vector<CUcontext> contexts() {
  std::lock_guard<std::mutex> lock(mutex);
  std::vector<CUcontext> result;
  for (const auto &entry : objects)
    if (entry.second.type == kind::context)
      result.push_back(reinterpret_cast<CUcontext>(entry.second.handle));
  return result;
}

bool supported() {
  std::lock_guard<std::mutex> lock(mutex);
  for (const auto &item : objects)
    if (item.second.type == kind::event &&
        !(item.second.flags & CU_EVENT_DISABLE_TIMING))
      unsupported_reason = "timed events require timestamp preservation";
  if (!unsupported_reason.empty())
    LUPINE_LOG_ERROR("Cannot checkpoint CUDA objects: " << unsupported_reason);
  return unsupported_reason.empty();
}

int save(const char *path) {
  if (!enabled())
    return 0;
  if (!supported())
    return -1;
  std::lock_guard<std::mutex> lock(mutex);
  FILE *file = std::fopen(path, "wb");
  if (!file)
    return -1;
  uint64_t header[] = {0x314a424f4e50554cULL, next_id, objects.size()};
  bool ok = std::fwrite(header, sizeof(header), 1, file) == 1;
  for (const auto &item : objects) {
    const auto &o = item.second;
    uint64_t fields[] = {
        static_cast<uint64_t>(o.type),  o.id,         o.parent, o.flags,
        static_cast<uint32_t>(o.value), o.data.size()};
    ok = ok && std::fwrite(fields, sizeof(fields), 1, file) == 1 &&
         (o.data.empty() ||
          std::fwrite(o.data.data(), o.data.size(), 1, file) == 1);
  }
  ok = ok && std::fflush(file) == 0;
#ifndef _WIN32
  ok = ok && fsync(fileno(file)) == 0;
#endif
  return std::fclose(file) == 0 && ok ? 0 : -1;
}

int restore(const char *path) {
  if (!enabled())
    return 0;
  std::lock_guard<std::mutex> lock(mutex);
  if (!objects.empty())
    return -1;
  FILE *file = std::fopen(path, "rb");
  if (!file)
    return -1;
  uint64_t header[3];
  bool ok = std::fread(header, sizeof(header), 1, file) == 1 &&
            header[0] == 0x314a424f4e50554cULL && header[2] <= 1000000;
  CUcontext previous = nullptr;
  ok = ok && cuCtxGetCurrent(&previous) == CUDA_SUCCESS;
  for (uint64_t i = 0; ok && i < header[2]; ++i) {
    uint64_t fields[6];
    ok = std::fread(fields, sizeof(fields), 1, file) == 1 &&
         fields[0] <= static_cast<uint64_t>(kind::event) &&
         fields[1] >= 0x10000 && fields[1] < header[1] &&
         !objects.count(fields[1]) && fields[5] <= (1ULL << 30);
    if (!ok)
      break;
    object o{static_cast<kind>(fields[0]),
             fields[1],
             fields[2],
             static_cast<uint32_t>(fields[3]),
             static_cast<int32_t>(fields[4]),
             {},
             0};
    o.data.resize(fields[5]);
    ok = o.data.empty() ||
         std::fread(o.data.data(), o.data.size(), 1, file) == 1;
    if (!ok)
      break;
    auto parent = objects.find(o.parent);
    if (o.type != kind::context) {
      kind expected = o.type == kind::function ? kind::module : kind::context;
      ok = parent != objects.end() && parent->second.type == expected;
      if (!ok)
        break;
      uintptr_t ctx = expected == kind::context
                          ? parent->second.handle
                          : objects.at(parent->second.parent).handle;
      ok = cuCtxSetCurrent(reinterpret_cast<CUcontext>(ctx)) == CUDA_SUCCESS;
      if (!ok)
        break;
    }
    CUresult status = CUDA_ERROR_INVALID_VALUE;
    switch (o.type) {
    case kind::context: {
      CUcontext ctx = nullptr;
      if (o.parent == 1) {
        unsigned flags = 0;
        int active = 0;
        status = cuDevicePrimaryCtxGetState(o.value, &flags, &active);
        if (status == CUDA_SUCCESS && active && flags != o.flags)
          status = CUDA_ERROR_INVALID_VALUE;
        if (status == CUDA_SUCCESS && !active)
          status = cuDevicePrimaryCtxSetFlags(o.value, o.flags);
        if (status == CUDA_SUCCESS)
          status = cuDevicePrimaryCtxRetain(&ctx, o.value);
      } else if (o.parent == 0)
        status = cuCtxCreate_v2(&ctx, o.flags, o.value);
      o.handle = reinterpret_cast<uintptr_t>(ctx);
      break;
    }
    case kind::module: {
      CUmodule module = nullptr;
      if (!o.data.empty())
        status = o.flags ? cuModuleLoadFatBinary(&module, o.data.data())
                         : cuModuleLoadData(&module, o.data.data());
      o.handle = reinterpret_cast<uintptr_t>(module);
      break;
    }
    case kind::function: {
      CUfunction function = nullptr;
      if (!o.data.empty() && o.data.back() == 0)
        status = cuModuleGetFunction(
            &function, reinterpret_cast<CUmodule>(parent->second.handle),
            reinterpret_cast<const char *>(o.data.data()));
      o.handle = reinterpret_cast<uintptr_t>(function);
      break;
    }
    case kind::stream: {
      CUstream stream = nullptr;
      status = cuStreamCreateWithPriority(&stream, o.flags, o.value);
      o.handle = reinterpret_cast<uintptr_t>(stream);
      break;
    }
    case kind::event: {
      CUevent event = nullptr;
      if (o.flags & CU_EVENT_DISABLE_TIMING)
        status = cuEventCreate(&event, o.flags);
      // The coordinated checkpoint drains all work. Re-record once so queries
      // and waits on an event recorded before eviction see completed work.
      if (status == CUDA_SUCCESS)
        status = cuEventRecord(event, nullptr);
      o.handle = reinterpret_cast<uintptr_t>(event);
      break;
    }
    }
    ok = status == CUDA_SUCCESS;
    if (ok)
      objects.emplace(o.id, std::move(o));
  }
  ok = ok && std::fgetc(file) == EOF && !std::ferror(file);
  std::fclose(file);
  if (cuCtxSetCurrent(previous) != CUDA_SUCCESS)
    ok = false;
  if (ok)
    next_id = header[1];
  return ok ? 0 : -1;
}
void observe_call(int operation) {
  if (!enabled())
    return;
  switch (operation) {
  case LUPINE_RPC_lupineDeviceSnapshot:
  case LUPINE_RPC_lupineEventCreateBatch:
  case LUPINE_RPC_lupineStreamPoolInit:
  case LUPINE_RPC_lupineFunctionAttributeSnapshot:
  case LUPINE_RPC_lupineFunctionParamLayoutSnapshot:
  case RPC_cuGetErrorString:
  case RPC_cuGetErrorName:
  case RPC_cuInit:
  case RPC_cuDriverGetVersion:
  case RPC_cuDeviceGet:
  case RPC_cuDeviceGetCount:
  case RPC_cuDeviceGetName:
  case RPC_cuDeviceGetUuid_v2:
  case RPC_cuDeviceGetLuid:
  case RPC_cuDeviceTotalMem_v2:
  case RPC_cuDeviceGetTexture1DLinearMaxWidth:
  case RPC_cuDeviceGetAttribute:
  case RPC_cuDeviceGetExecAffinitySupport:
  case RPC_cuDeviceGetProperties:
  case RPC_cuDevicePrimaryCtxRetain:
  case RPC_cuDevicePrimaryCtxGetState:
  case RPC_cuCtxPushCurrent_v2:
  case RPC_cuCtxPopCurrent_v2:
  case RPC_cuCtxSetCurrent:
  case RPC_cuCtxGetCurrent:
  case RPC_cuCtxGetDevice:
  case RPC_cuCtxGetDevice_v2:
  case RPC_cuCtxGetFlags:
  case RPC_cuCtxGetId:
  case RPC_cuCtxSynchronize:
  case RPC_cuCtxGetLimit:
  case RPC_cuCtxGetCacheConfig:
  case RPC_cuCtxGetApiVersion:
  case RPC_cuCtxGetStreamPriorityRange:
  case RPC_cuCtxGetExecAffinity:
  case RPC_cuCtxGetSharedMemConfig:
  case RPC_cuModuleLoad:
  case RPC_cuModuleLoadDataEx:
  case RPC_cuModuleLoadData:
  case RPC_cuModuleUnload:
  case RPC_cuModuleGetFunction:
  case RPC_cuMemGetInfo_v2:
  case RPC_cuMemAlloc_v2:
  case RPC_cuMemFree_v2:
  case RPC_cuMemGetAddressRange_v2:
  case RPC_cuDeviceGetByPCIBusId:
  case RPC_cuDeviceGetPCIBusId:
  case RPC_cuMemcpy:
  case RPC_cuMemcpyHtoD_v2:
  case RPC_cuMemcpyDtoH_v2:
  case RPC_cuMemcpyDtoD_v2:
  case RPC_cuMemcpyHtoDAsync_v2:
  case RPC_cuMemcpyDtoHAsync_v2:
  case RPC_cuMemcpyDtoDAsync_v2:
  case RPC_cuMemcpy2D_v2:
  case RPC_cuMemcpy2DUnaligned_v2:
  case RPC_cuMemcpy3D_v2:
  case RPC_cuMemcpy2DAsync_v2:
  case RPC_cuMemcpy3DAsync_v2:
  case RPC_cuMemsetD8_v2:
  case RPC_cuMemsetD16_v2:
  case RPC_cuMemsetD32_v2:
  case RPC_cuMemsetD8Async:
  case RPC_cuMemsetD16Async:
  case RPC_cuMemsetD32Async:
  case RPC_cuMemsetD2D8_v2:
  case RPC_cuMemsetD2D16_v2:
  case RPC_cuMemsetD2D32_v2:
  case RPC_cuMemsetD2D8Async:
  case RPC_cuMemsetD2D16Async:
  case RPC_cuMemsetD2D32Async:
  case RPC_cuMemGetAccess:
  case RPC_cuMemGetAllocationGranularity:
  case RPC_cuMemGetAllocationPropertiesFromHandle:
  case RPC_cuPointerGetAttributes:
  case RPC_cuStreamCreate:
  case RPC_cuStreamCreateWithPriority:
  case RPC_cuStreamGetPriority:
  case RPC_cuStreamGetFlags:
  case RPC_cuStreamGetId:
  case RPC_cuStreamGetCtx:
  case RPC_cuStreamWaitEvent:
  case RPC_cuStreamQuery:
  case RPC_cuStreamSynchronize:
  case RPC_cuStreamDestroy_v2:
  case RPC_cuStreamGetAttribute:
  case RPC_cuEventCreate:
  case RPC_cuEventRecord:
  case RPC_cuEventRecordWithFlags:
  case RPC_cuEventQuery:
  case RPC_cuEventSynchronize:
  case RPC_cuEventDestroy_v2:
  case RPC_cuFuncGetAttribute:
  case RPC_cuFuncGetModule:
  case RPC_cuFuncGetName:
  case RPC_cuFuncGetParamInfo:
  case RPC_cuLaunchKernel:
  case RPC_cuLaunchCooperativeKernel:
  case RPC_cuOccupancyMaxActiveBlocksPerMultiprocessor:
  case RPC_cuOccupancyMaxActiveBlocksPerMultiprocessorWithFlags:
  case RPC_cuOccupancyMaxPotentialBlockSize:
  case RPC_cuOccupancyMaxPotentialBlockSizeWithFlags:
  case RPC_cuOccupancyAvailableDynamicSMemPerBlock:
  case RPC_cuOccupancyMaxPotentialClusterSize:
  case RPC_cuOccupancyMaxActiveClusters:
  case RPC_cuDeviceGetP2PAttribute:
  case RPC_cuCtxCreate_v2:
  case RPC_cuDeviceGetGraphMemAttribute:
  case RPC_cuMemGetHandleForAddressRange:
  case RPC_cuPointerGetAttribute:
    return;
  default:
    unsupported("RPC outside the supported object recovery surface");
  }
}

} // namespace lupine_objects
