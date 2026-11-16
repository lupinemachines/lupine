#include "nvcomp_htod.h"

#include <nvcomp/lz4.h>
#include <nvcomp/zstd.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>

#include "rpc.h"

namespace {

struct nvcomp_api {
  decltype(&nvcompBatchedZstdDecompressGetTempSizeAsync) zstd_temp = nullptr;
  decltype(&nvcompBatchedZstdDecompressAsync) zstd = nullptr;
  decltype(&nvcompBatchedLZ4DecompressGetTempSizeAsync) lz4_temp = nullptr;
  decltype(&nvcompBatchedLZ4DecompressAsync) lz4 = nullptr;
};

// Loaded on first use so a server without libnvcomp behaves as before.
const nvcomp_api *nvcomp() {
  static const nvcomp_api api = [] {
    nvcomp_api loaded;
    if (getenv("LUPINE_NVCOMP_HTOD") == nullptr) {
      return loaded;
    }
    void *library = dlopen("libnvcomp.so.5", RTLD_NOW | RTLD_LOCAL);
    if (library == nullptr) {
      return loaded;
    }
    loaded.zstd_temp = reinterpret_cast<decltype(loaded.zstd_temp)>(
        dlsym(library, "nvcompBatchedZstdDecompressGetTempSizeAsync"));
    loaded.zstd = reinterpret_cast<decltype(loaded.zstd)>(
        dlsym(library, "nvcompBatchedZstdDecompressAsync"));
    loaded.lz4_temp = reinterpret_cast<decltype(loaded.lz4_temp)>(
        dlsym(library, "nvcompBatchedLZ4DecompressGetTempSizeAsync"));
    loaded.lz4 = reinterpret_cast<decltype(loaded.lz4)>(
        dlsym(library, "nvcompBatchedLZ4DecompressAsync"));
    if (!loaded.zstd_temp || !loaded.zstd || !loaded.lz4_temp || !loaded.lz4) {
      loaded = nvcomp_api();
    }
    return loaded;
  }();
  return api.zstd != nullptr ? &api : nullptr;
}

// Chunks decoding at once; each holds one device slot and one host buffer.
constexpr size_t kSlots = 4;
constexpr size_t kSlotBytes = LUPINE_RPC_TRANSFER_CHUNK_BYTES + 64 * 1024;

} // namespace

struct lupine_nvcomp_htod {
  CUdeviceptr destination = 0;
  CUdeviceptr slots = 0;
  CUdeviceptr temp = 0;
  size_t temp_bytes = 0;
  CUevent done[kSlots] = {};
  void *tags[kSlots] = {};
  std::deque<size_t> queued;
  size_t next = 0;
  size_t peak_device_bytes = 0;
  size_t chunks = 0;
  size_t frames = 0;
};

bool lupine_nvcomp_htod_enabled() { return nvcomp() != nullptr; }

lupine_nvcomp_htod *lupine_nvcomp_htod_begin(CUdeviceptr destination) {
  CUmemorytype type = CU_MEMORYTYPE_HOST;
  // nvCOMP's zstd output alignment.
  if (nvcomp() == nullptr || destination % 8 != 0 ||
      cuPointerGetAttribute(&type, CU_POINTER_ATTRIBUTE_MEMORY_TYPE,
                            destination) != CUDA_SUCCESS ||
      type != CU_MEMORYTYPE_DEVICE) {
    return nullptr;
  }
  auto *state = new lupine_nvcomp_htod();
  state->destination = destination;
  bool ok = cuMemAllocAsync(&state->slots, kSlots * kSlotBytes,
                            CU_STREAM_LEGACY) == CUDA_SUCCESS;
  for (size_t i = 0; ok && i < kSlots; ++i) {
    // Blocking, so a copy waiting on its decodes yields the core instead of
    // spinning on it.
    ok = cuEventCreate(&state->done[i], CU_EVENT_DISABLE_TIMING |
                                            CU_EVENT_BLOCKING_SYNC) ==
         CUDA_SUCCESS;
  }
  if (!ok) {
    std::vector<void *> none;
    lupine_nvcomp_htod_end(state, none);
    return nullptr;
  }
  state->peak_device_bytes = kSlots * kSlotBytes;
  return state;
}

CUresult lupine_nvcomp_htod_chunk(lupine_nvcomp_htod *state,
                                  unsigned char *host, size_t offset,
                                  void *tag) {
  const nvcomp_api &api = *nvcomp();
  lupine_framed_chunk_header header;
  memcpy(&header, host, sizeof(header));
  size_t count = header.frames;
  size_t temp_bytes = 0;
  nvcompStatus_t sized =
      header.codec == LUPINE_FRAME_LZ4
          ? api.lz4_temp(count, header.frame_bytes,
                         nvcompBatchedLZ4DecompressDefaultOpts, &temp_bytes,
                         header.decoded)
          : api.zstd_temp(count, header.frame_bytes,
                          nvcompBatchedZstdDecompressDefaultOpts, &temp_bytes,
                          header.decoded);
  if (sized != nvcompSuccess) {
    return CUDA_ERROR_UNKNOWN;
  }
  // Decodes share the legacy stream, so they take turns with one scratch.
  if (temp_bytes > state->temp_bytes) {
    if (state->temp != 0) {
      cuMemFreeAsync(state->temp, CU_STREAM_LEGACY);
    }
    state->temp_bytes = 0;
    CUresult result =
        cuMemAllocAsync(&state->temp, temp_bytes, CU_STREAM_LEGACY);
    if (result != CUDA_SUCCESS) {
      state->temp = 0;
      return result;
    }
    state->temp_bytes = temp_bytes;
    state->peak_device_bytes =
        std::max(state->peak_device_bytes, kSlots * kSlotBytes + temp_bytes);
  }
  size_t slot = state->next++ % kSlots;
  CUdeviceptr device = state->slots + slot * kSlotBytes;
  size_t source = lupine_framed_payload_offset(header.frames);
  size_t body = source;
  for (size_t i = 0; i < count; ++i) {
    uint32_t compressed;
    memcpy(&compressed, host + sizeof(header) + 4 * i, sizeof(compressed));
    body += lupine_framed_align(compressed);
  }
  auto *pointers = reinterpret_cast<uint64_t *>(host + body);
  uint64_t *compressed_sizes = pointers + count;
  uint64_t *decoded_sizes = compressed_sizes + count;
  uint64_t *outputs = decoded_sizes + count;
  for (size_t i = 0; i < count; ++i) {
    uint32_t compressed;
    memcpy(&compressed, host + sizeof(header) + 4 * i, sizeof(compressed));
    pointers[i] = device + source;
    compressed_sizes[i] = compressed;
    decoded_sizes[i] = std::min<size_t>(
        header.frame_bytes, header.decoded - i * header.frame_bytes);
    outputs[i] = state->destination + offset + i * header.frame_bytes;
    source += lupine_framed_align(compressed);
  }
  size_t uploaded = body + count * LUPINE_FRAMED_ARRAY_BYTES_PER_FRAME;
  CUdeviceptr arrays = device + body;
  CUdeviceptr actual = device + uploaded;
  CUdeviceptr statuses = actual + count * sizeof(uint64_t);
  CUresult result = cuMemcpyHtoDAsync(device, host, uploaded, CU_STREAM_LEGACY);
  if (result != CUDA_SUCCESS) {
    return result;
  }
  auto *stream = reinterpret_cast<cudaStream_t>(CU_STREAM_LEGACY);
  auto *in = reinterpret_cast<const void *const *>(arrays);
  auto *in_bytes = reinterpret_cast<const size_t *>(arrays + 8 * count);
  auto *out_bytes = reinterpret_cast<const size_t *>(arrays + 16 * count);
  auto *out = reinterpret_cast<void *const *>(arrays + 24 * count);
  nvcompStatus_t launched =
      header.codec == LUPINE_FRAME_LZ4
          ? api.lz4(in, in_bytes, out_bytes, reinterpret_cast<size_t *>(actual),
                    count, reinterpret_cast<void *>(state->temp),
                    state->temp_bytes, out,
                    nvcompBatchedLZ4DecompressDefaultOpts,
                    reinterpret_cast<nvcompStatus_t *>(statuses), stream)
          : api.zstd(in, in_bytes, out_bytes,
                     reinterpret_cast<size_t *>(actual), count,
                     reinterpret_cast<void *>(state->temp), state->temp_bytes,
                     out, nvcompBatchedZstdDecompressDefaultOpts,
                     reinterpret_cast<nvcompStatus_t *>(statuses), stream);
  if (launched != nvcompSuccess) {
    return CUDA_ERROR_UNKNOWN;
  }
  result = cuEventRecord(state->done[slot], CU_STREAM_LEGACY);
  state->tags[slot] = tag;
  state->queued.push_back(slot);
  ++state->chunks;
  state->frames += count;
  return result;
}

bool lupine_nvcomp_htod_full(const lupine_nvcomp_htod *state) {
  return state->queued.size() == kSlots;
}

bool lupine_nvcomp_htod_idle(const lupine_nvcomp_htod *state) {
  return state->queued.empty();
}

CUresult lupine_nvcomp_htod_retire(lupine_nvcomp_htod *state, bool all,
                                   std::vector<void *> &released) {
  CUresult result = CUDA_SUCCESS;
  while (!state->queued.empty()) {
    size_t slot = state->queued.front();
    CUresult synced = cuEventSynchronize(state->done[slot]);
    if (result == CUDA_SUCCESS) {
      result = synced;
    }
    state->queued.pop_front();
    released.push_back(state->tags[slot]);
    if (!all) {
      break;
    }
  }
  return result;
}

CUresult lupine_nvcomp_htod_end(lupine_nvcomp_htod *state,
                                std::vector<void *> &released) {
  CUresult result = lupine_nvcomp_htod_retire(state, true, released);
  if (state->temp != 0) {
    cuMemFreeAsync(state->temp, CU_STREAM_LEGACY);
  }
  if (state->slots != 0) {
    cuMemFreeAsync(state->slots, CU_STREAM_LEGACY);
  }
  for (CUevent event : state->done) {
    if (event != nullptr) {
      cuEventDestroy(event);
    }
  }
  if (getenv("LUPINE_NVCOMP_HTOD_STATS") != nullptr) {
    fprintf(stderr,
            "nvcomp_htod chunks=%zu frames=%zu peak_device_bytes=%zu "
            "temp_bytes=%zu\n",
            state->chunks, state->frames, state->peak_device_bytes,
            state->temp_bytes);
  }
  delete state;
  return result;
}
