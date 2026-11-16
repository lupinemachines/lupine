#pragma once

// Exploratory: striped host-to-device chunks decompressed on the GPU.
// LUPINE_NVCOMP_HTOD on the client frames each bulk chunk; a server built with
// LUPINE_NVCOMP and run with LUPINE_NVCOMP_HTOD decodes the frames with nvCOMP
// on the device, any other server decodes them on its CPU.

#include <cuda.h>

#include <dlfcn.h>

#include <cstddef>
#include <cstdint>
#include <vector>

// A lupineBulkChunk whose byte count has this bit set carries its chunk as
// independent frames: this header, a uint32 compressed size per frame, then
// the frames, each starting 8-byte aligned (nvCOMP's zstd input alignment).
// Every frame but the last decodes to frame_bytes. The client only frames a
// chunk when the body plus 32 bytes per frame still fits in the chunk, so the
// server can build nvCOMP's argument arrays behind the body in place.
constexpr uint64_t LUPINE_BULK_CHUNK_FRAMED = uint64_t{1} << 63;
enum lupine_frame_codec : uint32_t {
  LUPINE_FRAME_ZSTD = 0,
  LUPINE_FRAME_LZ4 = 1
};
struct lupine_framed_chunk_header {
  uint64_t decoded;
  uint32_t frames;
  uint32_t frame_bytes;
  uint32_t codec;
  uint32_t reserved;
};
constexpr size_t LUPINE_FRAMED_ARRAY_BYTES_PER_FRAME = 4 * sizeof(uint64_t);

inline size_t lupine_framed_align(size_t bytes) {
  return (bytes + 7) & ~size_t{7};
}

inline size_t lupine_framed_payload_offset(uint32_t frames) {
  return lupine_framed_align(sizeof(lupine_framed_chunk_header) +
                             sizeof(uint32_t) * frames);
}

// liblz4 is not a Lupine dependency; the exploratory LZ4 framing loads the
// system copy.
struct lupine_lz4_api {
  int (*compress)(const char *, char *, int, int, int) = nullptr;
  int (*decompress)(const char *, char *, int, int) = nullptr;
};
inline const lupine_lz4_api &lupine_lz4() {
  static const lupine_lz4_api api = [] {
    lupine_lz4_api loaded;
    if (void *library = dlopen("liblz4.so.1", RTLD_NOW | RTLD_LOCAL)) {
      loaded.compress = reinterpret_cast<decltype(loaded.compress)>(
          dlsym(library, "LZ4_compress_fast"));
      loaded.decompress = reinterpret_cast<decltype(loaded.decompress)>(
          dlsym(library, "LZ4_decompress_safe"));
    }
    return loaded;
  }();
  return api;
}

struct lupine_nvcomp_htod;

#ifdef LUPINE_NVCOMP
bool lupine_nvcomp_htod_enabled();
// Null when the destination cannot take a device-side decode; the caller then
// decodes on the CPU.
lupine_nvcomp_htod *lupine_nvcomp_htod_begin(CUdeviceptr destination);
// Queues the upload and decode of a framed chunk staged in page-locked `host`
// on the legacy stream. `host` stays in use until a retire returns `tag`.
CUresult lupine_nvcomp_htod_chunk(lupine_nvcomp_htod *state,
                                  unsigned char *host, size_t offset,
                                  void *tag);
// Waits for the oldest queued chunk (every one with `all`) and returns the
// tags whose host buffers are free again.
CUresult lupine_nvcomp_htod_retire(lupine_nvcomp_htod *state, bool all,
                                   std::vector<void *> &released);
bool lupine_nvcomp_htod_full(const lupine_nvcomp_htod *state);
bool lupine_nvcomp_htod_idle(const lupine_nvcomp_htod *state);
CUresult lupine_nvcomp_htod_end(lupine_nvcomp_htod *state,
                                std::vector<void *> &released);
#else
inline bool lupine_nvcomp_htod_enabled() { return false; }
inline lupine_nvcomp_htod *lupine_nvcomp_htod_begin(CUdeviceptr) {
  return nullptr;
}
inline CUresult lupine_nvcomp_htod_chunk(lupine_nvcomp_htod *, unsigned char *,
                                         size_t, void *) {
  return CUDA_ERROR_NOT_SUPPORTED;
}
inline CUresult lupine_nvcomp_htod_retire(lupine_nvcomp_htod *, bool,
                                          std::vector<void *> &) {
  return CUDA_SUCCESS;
}
inline bool lupine_nvcomp_htod_full(const lupine_nvcomp_htod *) {
  return false;
}
inline bool lupine_nvcomp_htod_idle(const lupine_nvcomp_htod *) { return true; }
inline CUresult lupine_nvcomp_htod_end(lupine_nvcomp_htod *,
                                       std::vector<void *> &) {
  return CUDA_SUCCESS;
}
#endif
