// Integration test for the cuLaunchHostFunc device->host callback path.
//
// The only way device->host bytes reach the client through the op=1 host-func
// callback (lupine_graph_host_callback -> rpc_client_dispatch_thread op==1) is
// a cuMemcpyDtoHAsync issued *during stream capture* followed by a
// cuLaunchHostFunc on the same stream: the server records the copy into the
// stream's capture resources and replays it when the host func fires.
//
// This test issues such a copy large enough (>= LUPINE_COMPRESS_MIN_BYTES,
// 64 KiB) that the server frames it with rpc_write_payload. If the client
// reads it with plain rpc_read (the bug), it ingests the LZ4 framing bytes as
// raw data and the host buffer ends up garbage -> the host func sees a
// mismatch. With the matching rpc_read_payload on the client, the buffer is
// correct.
//
// Built/run like the other local.sh samples (nvcc -cudart=shared, LD_PRELOAD
// the lupine client shim against a running server).
#include <atomic>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vector>

// 1 MiB of floats -> comfortably above the 64 KiB compression threshold and
// within a single 4 MiB LZ4 block.
static const size_t kFloats = (1u << 20) / sizeof(float);

static float *g_host = nullptr;
static std::atomic<int> g_result{0}; // 0 = not run, 1 = pass, -1 = fail

static void CUDART_CB host_fn(void * /*userdata*/) {
  // g_host was filled by the op=1 callback just before invoking us.
  int ok = 1;
  for (size_t i = 0; i < kFloats; ++i) {
    if (g_host[i] != (float)i) {
      ok = -1;
      break;
    }
  }
  g_result.store(ok, std::memory_order_release);
}

int main() {
  std::vector<float> src(kFloats);
  for (size_t i = 0; i < kFloats; ++i) {
    src[i] = (float)i;
  }
  g_host = (float *)calloc(kFloats, sizeof(float));
  if (g_host == nullptr) {
    printf("RESULT: ERROR calloc\n");
    return 2;
  }

  float *d_src = nullptr;
  cudaError_t e;
  e = cudaMalloc((void **)&d_src, kFloats * sizeof(float));
  if (e != cudaSuccess) {
    printf("RESULT: ERROR cudaMalloc %s\n", cudaGetErrorString(e));
    return 2;
  }
  // Seed device memory with the pattern the host func will check.
  e = cudaMemcpy(d_src, src.data(), kFloats * sizeof(float),
                 cudaMemcpyHostToDevice);
  if (e != cudaSuccess) {
    printf("RESULT: ERROR HtoD %s\n", cudaGetErrorString(e));
    return 2;
  }

  cudaStream_t stream;
  cudaGraph_t graph;
  cudaGraphExec_t exec;
  e = cudaStreamCreate(&stream);
  if (e != cudaSuccess) {
    printf("RESULT: ERROR stream %s\n", cudaGetErrorString(e));
    return 2;
  }

  // Capture: a large DtoH copy followed by a host func. The server records the
  // copy into the capture resources and ships it back via op=1 when the host
  // func fires.
  e = cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);
  if (e != cudaSuccess) {
    printf("RESULT: ERROR beginCapture %s\n", cudaGetErrorString(e));
    return 2;
  }
  e = cudaMemcpyAsync(g_host, d_src, kFloats * sizeof(float),
                      cudaMemcpyDeviceToHost, stream);
  if (e != cudaSuccess) {
    printf("RESULT: ERROR DtoH async %s\n", cudaGetErrorString(e));
    return 2;
  }
  e = cudaLaunchHostFunc(stream, host_fn, nullptr);
  if (e != cudaSuccess) {
    printf("RESULT: ERROR launchHostFunc %s\n", cudaGetErrorString(e));
    return 2;
  }
  e = cudaStreamEndCapture(stream, &graph);
  if (e != cudaSuccess) {
    printf("RESULT: ERROR endCapture %s\n", cudaGetErrorString(e));
    return 2;
  }

  e = cudaGraphInstantiate(&exec, graph, nullptr, nullptr, 0);
  if (e != cudaSuccess) {
    printf("RESULT: ERROR instantiate %s\n", cudaGetErrorString(e));
    return 2;
  }
  e = cudaGraphLaunch(exec, stream);
  if (e != cudaSuccess) {
    printf("RESULT: ERROR launch %s\n", cudaGetErrorString(e));
    return 2;
  }
  // Sync waits for the graph, which includes the host node; the host node only
  // completes after the client processes the op=1 callback (and runs host_fn).
  e = cudaStreamSynchronize(stream);
  if (e != cudaSuccess) {
    printf("RESULT: ERROR sync %s\n", cudaGetErrorString(e));
    return 2;
  }

  int r = g_result.load(std::memory_order_acquire);
  cudaGraphExecDestroy(exec);
  cudaGraphDestroy(graph);
  cudaStreamDestroy(stream);
  cudaFree(d_src);
  free(g_host);

  if (r == 1) {
    printf("RESULT: PASS (host func received correct DtoH data)\n");
    return 0;
  }
  if (r == -1) {
    printf("RESULT: FAIL (host func received mismatched DtoH data)\n");
    return 1;
  }
  printf("RESULT: FAIL (host func never ran)\n");
  return 1;
}
