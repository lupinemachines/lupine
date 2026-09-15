// cuFile (libcufile) on the LUPINE client.
//
// GPUDirect Storage exists to keep the host out of the transfer: the storage
// device DMAs straight into GPU memory. Under LUPINE the file is on the client
// and the memory is on a server, so nothing can DMA across that gap and every
// byte has to cross the wire. The shim runs the compatibility path cuFile
// itself falls back to when there is no nvidia-fs driver, with the two halves
// on different machines: the client reads the file with POSIX calls and the
// driver shim moves the staging buffer. The API's contract holds - the bytes
// land where the caller asked, in the order the caller asked - while the
// direct part of GPUDirect is gone, and the driver properties report no
// GPUDirect capability rather than advertise one the wire cannot provide.
//
// Entry points that only a real GPUDirect stack can answer, the nvidia-fs
// tunables and the P2P flags and the driver's statistics, return
// CU_FILE_PLATFORM_NOT_SUPPORTED instead of a value they cannot mean.

#include <cuda.h>
#include <cufile.h>

#include <fcntl.h>
#include <sys/types.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <cstdio>
#include <cstring>
#include <map>
#include <mutex>
#include <string>
#include <vector>

namespace {

// One wire transfer per chunk, so a large request needs no large allocation.
constexpr size_t staging_bytes = 16u << 20;

CUfileError_t status(CUfileOpError error) {
  CUfileError_t result;
  result.err = error;
  result.cu_err = CUDA_SUCCESS;
  return result;
}

CUfileError_t success() { return status(CU_FILE_SUCCESS); }

CUfileError_t not_supported() { return status(CU_FILE_PLATFORM_NOT_SUPPORTED); }

struct registered_file {
  int fd;
  bool owns_fd;
};

// The caller opens its file O_DIRECT because GPUDirect needs it, and O_DIRECT
// then demands block-aligned offsets, lengths and buffers of the reads this
// shim does instead. O_DIRECT belongs to the open file description, so a dup
// carries it; reopening through /proc gives a second description of the same
// file without it and leaves the caller's own descriptor untouched.
int reopen_buffered(int fd) {
  const int flags = fcntl(fd, F_GETFL);
  if (flags < 0) {
    return -1;
  }
  char path[32];
  snprintf(path, sizeof(path), "/proc/self/fd/%d", fd);
  return open(path, (flags & O_ACCMODE) | O_CLOEXEC);
}

// A short count is the answer to a short read, -1 leaves errno as the
// filesystem set it, and any other failure is the negated CUfileOpError the
// API asks for.
ssize_t read_into_device(int fd, CUdeviceptr buffer, size_t size,
                         off_t file_offset) {
  std::vector<char> staging(std::min(size, staging_bytes));
  size_t done = 0;
  while (done < size) {
    const size_t want = std::min(staging.size(), size - done);
    const ssize_t got =
        pread(fd, staging.data(), want, file_offset + static_cast<off_t>(done));
    if (got < 0) {
      return -1;
    }
    if (got == 0) {
      break;
    }
    if (cuMemcpyHtoD(buffer + done, staging.data(), static_cast<size_t>(got)) !=
        CUDA_SUCCESS) {
      return -CU_FILE_CUDA_DRIVER_ERROR;
    }
    done += static_cast<size_t>(got);
    if (static_cast<size_t>(got) < want) {
      break;
    }
  }
  return static_cast<ssize_t>(done);
}

ssize_t write_from_device(int fd, CUdeviceptr buffer, size_t size,
                          off_t file_offset) {
  std::vector<char> staging(std::min(size, staging_bytes));
  size_t done = 0;
  while (done < size) {
    const size_t want = std::min(staging.size(), size - done);
    if (cuMemcpyDtoH(staging.data(), buffer + done, want) != CUDA_SUCCESS) {
      return -CU_FILE_CUDA_DRIVER_ERROR;
    }
    const ssize_t put = pwrite(fd, staging.data(), want,
                               file_offset + static_cast<off_t>(done));
    if (put < 0) {
      return -1;
    }
    done += static_cast<size_t>(put);
    if (static_cast<size_t>(put) < want) {
      break;
    }
  }
  return static_cast<ssize_t>(done);
}

CUdeviceptr device_address(const void *base, off_t offset) {
  return reinterpret_cast<CUdeviceptr>(base) + static_cast<CUdeviceptr>(offset);
}

int descriptor_of(CUfileHandle_t fh) {
  return static_cast<const registered_file *>(fh)->fd;
}

// A batch runs inside its submit, so every request it was given is already an
// event by the time the call returns.
struct batch {
  std::vector<CUfileIOEvents_t> completed;
};

std::atomic<long> driver_uses{0};

#if LUPINE_CUFILE_HAS_PARAMETERS
std::mutex parameters_mutex;
// cuFile keeps a parameter staged until the driver opens and this driver never
// opens one, so what a program sets is what it reads back. Compatibility mode
// is the only mode here, and reads true from the start.
std::map<int, size_t> sizet_parameters;
std::map<int, bool> bool_parameters{
    {CUFILE_PARAM_PROPERTIES_ALLOW_COMPAT_MODE, true},
    {CUFILE_PARAM_FORCE_COMPAT_MODE, true}};
std::map<int, std::string> string_parameters;
#endif

} // namespace

extern "C" CUfileError_t cuFileHandleRegister(CUfileHandle_t *fh,
                                              CUfileDescr_t *descr) {
  if (descr->type != CU_FILE_HANDLE_TYPE_OPAQUE_FD) {
    // A userspace filesystem's callbacks run inside the library that reads the
    // file, which here is the client's own process, but the descriptor names
    // no file the shim can read itself.
    return status(CU_FILE_IO_NOT_SUPPORTED);
  }
  const int buffered = reopen_buffered(descr->handle.fd);
  registered_file *file = new registered_file;
  file->fd = buffered < 0 ? descr->handle.fd : buffered;
  file->owns_fd = buffered >= 0;
  *fh = file;
  return success();
}

extern "C" void cuFileHandleDeregister(CUfileHandle_t fh) {
  registered_file *file = static_cast<registered_file *>(fh);
  if (file->owns_fd) {
    close(file->fd);
  }
  delete file;
}

// Nothing is pinned for a DMA that never happens: the bytes travel through the
// staging buffer whether or not the caller registers its memory.
extern "C" CUfileError_t cuFileBufRegister(const void *bufPtr_base,
                                           size_t length, int flags) {
  return success();
}

extern "C" CUfileError_t cuFileBufDeregister(const void *bufPtr_base) {
  return success();
}

extern "C" ssize_t cuFileRead(CUfileHandle_t fh, void *bufPtr_base, size_t size,
                              off_t file_offset, off_t bufPtr_offset) {
  return read_into_device(descriptor_of(fh),
                          device_address(bufPtr_base, bufPtr_offset), size,
                          file_offset);
}

extern "C" ssize_t cuFileWrite(CUfileHandle_t fh, const void *bufPtr_base,
                               size_t size, off_t file_offset,
                               off_t bufPtr_offset) {
  return write_from_device(descriptor_of(fh),
                           device_address(bufPtr_base, bufPtr_offset), size,
                           file_offset);
}

extern "C" CUfileError_t cuFileDriverOpen(void) {
  driver_uses.fetch_add(1);
  return success();
}

extern "C" CUfileError_t cuFileDriverClose(void) {
  driver_uses.fetch_sub(1);
  return success();
}

// The header maps the name to cuFileDriverClose_v2, which the definition above
// took; NVIDIA's library exports both, so a program built before the rename
// resolves too.
#undef cuFileDriverClose
extern "C" CUfileError_t cuFileDriverClose(void) {
  return cuFileDriverClose_v2();
}

extern "C" long cuFileUseCount(void) { return driver_uses.load(); }

extern "C" CUfileError_t cuFileDriverGetProperties(CUfileDrvProps_t *props) {
  *props = CUfileDrvProps_t{};
  // No nvidia-fs and no GPUDirect-capable filesystem: a program that reads
  // these flags to choose between GDS and its own POSIX path is told there is
  // nothing to choose, and compatibility mode is what it gets either way.
  props->nvfs.dcontrolflags = 1u << CU_FILE_ALLOW_COMPAT_MODE;
  props->nvfs.max_direct_io_size = staging_bytes / 1024;
  // The shim caps no batch; this is the library's own default, so a caller
  // sizing its batches from this number gets a usable one.
  props->max_batch_io_size = 128;
  return success();
}

// The nvidia-fs data path these tune is the one the wire replaces.
extern "C" CUfileError_t cuFileDriverSetPollMode(bool poll,
                                                 size_t poll_threshold_size) {
  return not_supported();
}

extern "C" CUfileError_t
cuFileDriverSetMaxDirectIOSize(size_t max_direct_io_size) {
  return not_supported();
}

extern "C" CUfileError_t cuFileDriverSetMaxCacheSize(size_t max_cache_size) {
  return not_supported();
}

extern "C" CUfileError_t
cuFileDriverSetMaxPinnedMemSize(size_t max_pinned_size) {
  return not_supported();
}

extern "C" CUfileError_t cuFileBatchIOSetUp(CUfileBatchHandle_t *batch_idp,
                                            unsigned nr) {
  *batch_idp = new batch;
  return success();
}

extern "C" CUfileError_t cuFileBatchIOSubmit(CUfileBatchHandle_t batch_idp,
                                             unsigned nr,
                                             CUfileIOParams_t *iocbp,
                                             unsigned int flags) {
  batch *pending = static_cast<batch *>(batch_idp);
  for (unsigned i = 0; i < nr; ++i) {
    const CUfileIOParams_t &params = iocbp[i];
    const CUdeviceptr buffer = device_address(params.u.batch.devPtr_base,
                                              params.u.batch.devPtr_offset);
    const ssize_t moved =
        params.opcode == CUFILE_READ
            ? read_into_device(descriptor_of(params.fh), buffer,
                               params.u.batch.size, params.u.batch.file_offset)
            : write_from_device(descriptor_of(params.fh), buffer,
                                params.u.batch.size,
                                params.u.batch.file_offset);
    CUfileIOEvents_t event;
    event.cookie = params.cookie;
    event.status = moved < 0 ? CUFILE_FAILED : CUFILE_COMPLETE;
    event.ret = static_cast<size_t>(moved);
    pending->completed.push_back(event);
  }
  return success();
}

// Submission ran the whole batch, so there is nothing left for min_nr or the
// timeout to wait for.
extern "C" CUfileError_t cuFileBatchIOGetStatus(CUfileBatchHandle_t batch_idp,
                                                unsigned min_nr, unsigned *nr,
                                                CUfileIOEvents_t *iocbp,
                                                struct timespec *timeout) {
  batch *pending = static_cast<batch *>(batch_idp);
  const size_t count = std::min<size_t>(*nr, pending->completed.size());
  const auto first = pending->completed.begin();
  std::copy(first, first + count, iocbp);
  pending->completed.erase(first, first + count);
  *nr = static_cast<unsigned>(count);
  return success();
}

extern "C" CUfileError_t cuFileBatchIOCancel(CUfileBatchHandle_t batch_idp) {
  return success();
}

extern "C" void cuFileBatchIODestroy(CUfileBatchHandle_t batch_idp) {
  delete static_cast<batch *>(batch_idp);
}

#if LUPINE_CUFILE_HAS_ASYNC
// The library reads the sizes and offsets when the stream reaches the
// operation; the shim waits for the stream and then reads them, which puts the
// transfer in the same place in the stream's order. It blocks the caller to
// get there, because the transfer is host work on this side of the wire and a
// stream callback that made RPCs would deadlock against the connection it
// waits on.
extern "C" CUfileError_t cuFileReadAsync(CUfileHandle_t fh, void *bufPtr_base,
                                         size_t *size_p, off_t *file_offset_p,
                                         off_t *bufPtr_offset_p,
                                         ssize_t *bytes_read_p,
                                         CUstream stream) {
  if (cuStreamSynchronize(stream) != CUDA_SUCCESS) {
    return status(CU_FILE_CUDA_DRIVER_ERROR);
  }
  *bytes_read_p = read_into_device(
      descriptor_of(fh), device_address(bufPtr_base, *bufPtr_offset_p), *size_p,
      *file_offset_p);
  return success();
}

extern "C" CUfileError_t cuFileWriteAsync(CUfileHandle_t fh, void *bufPtr_base,
                                          size_t *size_p, off_t *file_offset_p,
                                          off_t *bufPtr_offset_p,
                                          ssize_t *bytes_written_p,
                                          CUstream stream) {
  if (cuStreamSynchronize(stream) != CUDA_SUCCESS) {
    return status(CU_FILE_CUDA_DRIVER_ERROR);
  }
  *bytes_written_p = write_from_device(
      descriptor_of(fh), device_address(bufPtr_base, *bufPtr_offset_p), *size_p,
      *file_offset_p);
  return success();
}

// A registration only buys the library somewhere to keep the resources an
// asynchronous submission needs, and this one submits nothing.
extern "C" CUfileError_t cuFileStreamRegister(CUstream stream, unsigned flags) {
  return success();
}

extern "C" CUfileError_t cuFileStreamDeregister(CUstream stream) {
  return success();
}
#endif

#if LUPINE_CUFILE_HAS_VERSION
extern "C" CUfileError_t cuFileGetVersion(int *version) {
  *version = LUPINE_CUFILE_VERSION;
  return success();
}
#endif

#if LUPINE_CUFILE_HAS_PARAMETERS
extern "C" CUfileError_t
cuFileGetParameterSizeT(CUFileSizeTConfigParameter_t param, size_t *value) {
  const std::lock_guard<std::mutex> lock(parameters_mutex);
  const auto staged = sizet_parameters.find(param);
  *value = staged == sizet_parameters.end() ? 0 : staged->second;
  return success();
}

extern "C" CUfileError_t
cuFileGetParameterBool(CUFileBoolConfigParameter_t param, bool *value) {
  const std::lock_guard<std::mutex> lock(parameters_mutex);
  const auto staged = bool_parameters.find(param);
  *value = staged == bool_parameters.end() ? false : staged->second;
  return success();
}

extern "C" CUfileError_t
cuFileGetParameterString(CUFileStringConfigParameter_t param, char *desc_str,
                         int len) {
  const std::lock_guard<std::mutex> lock(parameters_mutex);
  const auto staged = string_parameters.find(param);
  snprintf(desc_str, static_cast<size_t>(std::max(len, 0)), "%s",
           staged == string_parameters.end() ? "" : staged->second.c_str());
  return success();
}

extern "C" CUfileError_t
cuFileSetParameterSizeT(CUFileSizeTConfigParameter_t param, size_t value) {
  const std::lock_guard<std::mutex> lock(parameters_mutex);
  sizet_parameters[param] = value;
  return success();
}

extern "C" CUfileError_t
cuFileSetParameterBool(CUFileBoolConfigParameter_t param, bool value) {
  const std::lock_guard<std::mutex> lock(parameters_mutex);
  bool_parameters[param] = value;
  return success();
}

extern "C" CUfileError_t
cuFileSetParameterString(CUFileStringConfigParameter_t param,
                         const char *desc_str) {
  const std::lock_guard<std::mutex> lock(parameters_mutex);
  string_parameters[param] = desc_str;
  return success();
}
#endif

#if LUPINE_CUFILE_HAS_PARAMETER_RANGE
extern "C" CUfileError_t
cuFileGetParameterMinMaxValue(CUFileSizeTConfigParameter_t param,
                              size_t *min_value, size_t *max_value) {
  return not_supported();
}
#endif

#if LUPINE_CUFILE_HAS_STATS
extern "C" CUfileError_t cuFileSetStatsLevel(int level) {
  return not_supported();
}

extern "C" CUfileError_t cuFileGetStatsLevel(int *level) {
  return not_supported();
}

extern "C" CUfileError_t cuFileStatsStart(void) { return not_supported(); }

extern "C" CUfileError_t cuFileStatsStop(void) { return not_supported(); }

extern "C" CUfileError_t cuFileStatsReset(void) { return not_supported(); }

extern "C" CUfileError_t cuFileGetStatsL1(CUfileStatsLevel1_t *stats) {
  return not_supported();
}

extern "C" CUfileError_t cuFileGetStatsL2(CUfileStatsLevel2_t *stats) {
  return not_supported();
}

extern "C" CUfileError_t cuFileGetStatsL3(CUfileStatsLevel3_t *stats) {
  return not_supported();
}
#endif

#if LUPINE_CUFILE_HAS_SLAB_ARRAYS
extern "C" CUfileError_t
cuFileSetParameterPosixPoolSlabArray(const size_t *size_values,
                                     const size_t *count_values, int len) {
  return not_supported();
}

extern "C" CUfileError_t
cuFileGetParameterPosixPoolSlabArray(size_t *size_values, size_t *count_values,
                                     int len) {
  return not_supported();
}

extern "C" CUfileError_t cuFileSetParameterGpuBounceBufferSlabArray(
    const size_t *size_values, const size_t *count_values, int len) {
  return not_supported();
}

extern "C" CUfileError_t
cuFileGetParameterGpuBounceBufferSlabArray(size_t *size_values,
                                           size_t *count_values, int len) {
  return not_supported();
}
#endif

#if LUPINE_CUFILE_HAS_P2P_FLAGS
extern "C" CUfileError_t
cuFileDriverGetP2PFlags(CUfileDriverStatusFlags_t status_flag,
                        CUfileP2PFlags_t *p2p_flags) {
  return not_supported();
}

extern "C" CUfileError_t
cuFileDriverSetP2PFlags(CUfileDriverStatusFlags_t status_flag,
                        CUfileP2PFlags_t p2p_flags) {
  return not_supported();
}
#endif

#if LUPINE_CUFILE_HAS_BAR_SIZE
extern "C" CUfileError_t cuFileGetBARSizeInKB(int gpuIndex, size_t *barSize) {
  return not_supported();
}
#endif

#if LUPINE_CUFILE_HAS_TOPOLOGY
extern "C" CUfileError_t cuFileExportPCIeTopology(const char *filename) {
  return not_supported();
}
#endif
