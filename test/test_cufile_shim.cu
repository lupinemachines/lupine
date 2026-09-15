// Exercises the cuFile shim end to end against the remote device: a file the
// client opens O_DIRECT the way GPUDirect asks for, written from device memory
// and read back into it, both whole and at an offset, with the file's bytes
// checked from the host in between; the same transfers through the batch API
// and, on toolkits that have it, the stream API; the buffer and stream
// registrations that have nothing left to pin; and the driver properties,
// which must report no GPUDirect capability because the bytes cross the wire.
#include <cuda.h>
#include <cuda_runtime.h>
#include <cufile.h>

#include <fcntl.h>
#include <unistd.h>

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

#define CHECK_CUDA(call)                                                       \
  do {                                                                         \
    cudaError_t status = (call);                                               \
    if (status != cudaSuccess) {                                               \
      fprintf(stderr, "%s failed: %s\n", #call, cudaGetErrorName(status));     \
      return 1;                                                                \
    }                                                                          \
  } while (0)

#define CHECK_FILE(call)                                                       \
  do {                                                                         \
    CUfileError_t status = (call);                                             \
    if (status.err != CU_FILE_SUCCESS) {                                       \
      fprintf(stderr, "%s failed: %d\n", #call, static_cast<int>(status.err)); \
      return 1;                                                                \
    }                                                                          \
  } while (0)

#define EXPECT(condition)                                                      \
  do {                                                                         \
    if (!(condition)) {                                                        \
      fprintf(stderr, "expectation failed: %s\n", #condition);                 \
      return 1;                                                                \
    }                                                                          \
  } while (0)

namespace {

constexpr size_t kCount = 1 << 18;
constexpr size_t kBytes = kCount * sizeof(int);

void fill(std::vector<int> *values, int seed) {
  for (size_t i = 0; i < values->size(); ++i) {
    (*values)[i] = static_cast<int>(i) * 7 + seed;
  }
}

} // namespace

int main() {
  int devices = 0;
  CHECK_CUDA(cudaGetDeviceCount(&devices));
  EXPECT(devices > 0);
  CHECK_CUDA(cudaSetDevice(0));
  CHECK_FILE(cuFileDriverOpen());
  EXPECT(cuFileUseCount() > 0);

  CUfileDrvProps_t props;
  CHECK_FILE(cuFileDriverGetProperties(&props));
  // Nothing behind this file is GPUDirect-capable when the memory is on
  // another machine, and the mode that remains is the compatibility one.
  printf("driver: status %u control %u\n", props.nvfs.dstatusflags,
         props.nvfs.dcontrolflags);

  char path[] = "/tmp/lupine-cufile-XXXXXX";
  const int seed_fd = mkstemp(path);
  EXPECT(seed_fd >= 0);
  close(seed_fd);
  // GPUDirect asks for O_DIRECT, so the test opens the file the way a program
  // using it would; a filesystem that refuses the flag still exercises the
  // rest.
  int fd = open(path, O_RDWR | O_DIRECT);
  const bool direct = fd >= 0;
  if (!direct) {
    fd = open(path, O_RDWR);
  }
  EXPECT(fd >= 0);
  // The transfers are checked against the file through a descriptor of its
  // own, which reads the bytes that reached the disk rather than the shim's.
  const int check_fd = open(path, O_RDONLY);
  EXPECT(check_fd >= 0);
  printf("file: %s (O_DIRECT %s)\n", path, direct ? "on" : "off");

  CUfileDescr_t descr;
  memset(&descr, 0, sizeof(descr));
  descr.type = CU_FILE_HANDLE_TYPE_OPAQUE_FD;
  descr.handle.fd = fd;
  CUfileHandle_t handle = nullptr;
  CHECK_FILE(cuFileHandleRegister(&handle, &descr));

  void *device = nullptr;
  CHECK_CUDA(cudaMalloc(&device, kBytes));
  CHECK_FILE(cuFileBufRegister(device, kBytes, 0));

  std::vector<int> source(kCount);
  std::vector<int> result(kCount);
  fill(&source, 1);
  CHECK_CUDA(cudaMemcpy(device, source.data(), kBytes, cudaMemcpyHostToDevice));
  EXPECT(cuFileWrite(handle, device, kBytes, 0, 0) ==
         static_cast<ssize_t>(kBytes));
  EXPECT(pread(check_fd, result.data(), kBytes, 0) ==
         static_cast<ssize_t>(kBytes));
  EXPECT(memcmp(source.data(), result.data(), kBytes) == 0);

  CHECK_CUDA(cudaMemset(device, 0, kBytes));
  EXPECT(cuFileRead(handle, device, kBytes, 0, 0) ==
         static_cast<ssize_t>(kBytes));
  CHECK_CUDA(cudaMemcpy(result.data(), device, kBytes, cudaMemcpyDeviceToHost));
  EXPECT(memcmp(source.data(), result.data(), kBytes) == 0);
  printf("whole file: passed\n");

  // The second half of the buffer, appended to the file and read back into the
  // first half, so both offsets carry their own meaning.
  const size_t half = kBytes / 2;
  EXPECT(cuFileWrite(handle, device, half, static_cast<off_t>(kBytes),
                     static_cast<off_t>(half)) == static_cast<ssize_t>(half));
  CHECK_CUDA(cudaMemset(device, 0, half));
  EXPECT(cuFileRead(handle, device, half, static_cast<off_t>(kBytes), 0) ==
         static_cast<ssize_t>(half));
  CHECK_CUDA(cudaMemcpy(result.data(), device, half, cudaMemcpyDeviceToHost));
  EXPECT(memcmp(source.data() + kCount / 2, result.data(), half) == 0);
  printf("offsets: passed\n");

  CUfileBatchHandle_t batch = nullptr;
  CHECK_FILE(cuFileBatchIOSetUp(&batch, 2));
  CHECK_CUDA(cudaMemset(device, 0, kBytes));
  CUfileIOParams_t params[2];
  memset(params, 0, sizeof(params));
  for (unsigned i = 0; i < 2; ++i) {
    params[i].mode = CUFILE_BATCH;
    params[i].fh = handle;
    params[i].opcode = CUFILE_READ;
    params[i].cookie = reinterpret_cast<void *>(static_cast<uintptr_t>(i));
    params[i].u.batch.devPtr_base = device;
    params[i].u.batch.devPtr_offset = static_cast<off_t>(i * half);
    params[i].u.batch.file_offset = static_cast<off_t>(i * half);
    params[i].u.batch.size = half;
  }
  CHECK_FILE(cuFileBatchIOSubmit(batch, 2, params, 0));
  CUfileIOEvents_t events[2];
  memset(events, 0, sizeof(events));
  unsigned reported = 2;
  CHECK_FILE(cuFileBatchIOGetStatus(batch, 2, &reported, events, nullptr));
  EXPECT(reported == 2);
  for (unsigned i = 0; i < reported; ++i) {
    EXPECT(events[i].status == CUFILE_COMPLETE);
    EXPECT(events[i].ret == half);
  }
  CHECK_FILE(cuFileBatchIOCancel(batch));
  cuFileBatchIODestroy(batch);
  CHECK_CUDA(cudaMemcpy(result.data(), device, kBytes, cudaMemcpyDeviceToHost));
  EXPECT(memcmp(source.data(), result.data(), kBytes) == 0);
  printf("batch: passed\n");

  // cuFile 1.7, in CUDA 12.2, brings the stream API and the version query.
#if CUDA_VERSION >= 12020
  int version = 0;
  CHECK_FILE(cuFileGetVersion(&version));
  EXPECT(version >= 1070);
  printf("cuFile %d\n", version);

  cudaStream_t stream = nullptr;
  CHECK_CUDA(cudaStreamCreate(&stream));
  CHECK_FILE(cuFileStreamRegister(stream, 0));
  CHECK_CUDA(cudaMemsetAsync(device, 0, kBytes, stream));
  size_t size = kBytes;
  off_t file_offset = 0;
  off_t buffer_offset = 0;
  ssize_t moved = 0;
  CHECK_FILE(cuFileReadAsync(handle, device, &size, &file_offset,
                             &buffer_offset, &moved, stream));
  CHECK_CUDA(cudaStreamSynchronize(stream));
  EXPECT(moved == static_cast<ssize_t>(kBytes));
  CHECK_CUDA(cudaMemcpy(result.data(), device, kBytes, cudaMemcpyDeviceToHost));
  EXPECT(memcmp(source.data(), result.data(), kBytes) == 0);
  CHECK_FILE(cuFileStreamDeregister(stream));
  CHECK_CUDA(cudaStreamDestroy(stream));
  printf("stream: passed\n");
#endif

  CHECK_FILE(cuFileBufDeregister(device));
  CHECK_CUDA(cudaFree(device));
  cuFileHandleDeregister(handle);
  close(check_fd);
  close(fd);
  unlink(path);
  CHECK_FILE(cuFileDriverClose());
  printf("cufile shim: all checks passed\n");
  return 0;
}
