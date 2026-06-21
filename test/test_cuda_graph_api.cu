// Driver-API coverage for the CUDA graph query and node-params APIs that lupine
// marshals specially: the optional out-array queries (cuGraphGetNodes /
// GetRootNodes / GetEdges / NodeGetDependencies / NodeGetDependentNodes), the
// host-node params (callback trampoline), the batch-mem-op node params
// (embedded array, exercised end-to-end), and the array/texture-object
// creators. Runs through the lupine client shim against a remote server, so it
// validates the RPC wire format, not just local CUDA.
//
// Pure driver API, no kernels (so it is arch-independent). Exits non-zero on
// the first failed assertion.
#include <cstdio>
#include <cstdlib>
#include <cuda.h>
#include <set>
#include <vector>

static int g_failures = 0;
#define CHECK(cond, msg)                                                       \
  do {                                                                         \
    if (!(cond)) {                                                             \
      fprintf(stderr, "FAIL: %s (line %d)\n", msg, __LINE__);                  \
      g_failures++;                                                            \
    } else {                                                                   \
      fprintf(stderr, "ok:   %s\n", msg);                                      \
    }                                                                          \
  } while (0)

static const char *errstr(CUresult r) {
  const char *s = nullptr;
  cuGetErrorName(r, &s);
  return s ? s : "?";
}
#define DRV(call)                                                              \
  do {                                                                         \
    CUresult _r = (call);                                                      \
    if (_r != CUDA_SUCCESS) {                                                  \
      fprintf(stderr, "FATAL: %s -> %s (line %d)\n", #call, errstr(_r),        \
              __LINE__);                                                       \
      return 1;                                                                \
    }                                                                          \
  } while (0)

static int g_host_calls = 0;
static void *g_host_user = nullptr;
static void CUDA_CB host_cb(void *user) {
  g_host_calls++;
  g_host_user = user;
}

int main() {
  DRV(cuInit(0));
  CUdevice dev;
  DRV(cuDeviceGet(&dev, 0));
  CUcontext ctx; // primary context: signature is stable across CUDA versions.
  DRV(cuDevicePrimaryCtxRetain(&ctx, dev));
  DRV(cuCtxSetCurrent(ctx));

  CUgraph graph;
  DRV(cuGraphCreate(&graph, 0));

  // Topology:  A -> B -> C ;  B -> D ;  C -> D
  CUgraphNode A, B, C, D;
  DRV(cuGraphAddEmptyNode(&A, graph, nullptr, 0));
  CUgraphNode depA[] = {A};
  DRV(cuGraphAddEmptyNode(&B, graph, depA, 1));

  CUDA_HOST_NODE_PARAMS hp = {};
  hp.fn = host_cb;
  hp.userData = (void *)0xCAFEF00D;
  CUgraphNode depB[] = {B};
  DRV(cuGraphAddHostNode(&C, graph, depB, 1, &hp));

  CUgraphNode depBC[] = {B, C};
  DRV(cuGraphAddEmptyNode(&D, graph, depBC, 2));

  // ---- cuGraphGetNodes (OPTIONAL out-array: count query then fill) ----
  size_t n = 0;
  DRV(cuGraphGetNodes(graph, nullptr, &n));
  CHECK(n == 4, "cuGraphGetNodes count == 4");
  std::vector<CUgraphNode> nodes(n);
  DRV(cuGraphGetNodes(graph, nodes.data(), &n));
  CHECK(n == 4, "cuGraphGetNodes fill returned 4");

  // ---- cuGraphGetRootNodes ----
  size_t nroots = 0;
  DRV(cuGraphGetRootNodes(graph, nullptr, &nroots));
  CHECK(nroots == 1, "cuGraphGetRootNodes count == 1");
  std::vector<CUgraphNode> roots(nroots);
  DRV(cuGraphGetRootNodes(graph, roots.data(), &nroots));
  CHECK(nroots == 1 && roots[0] == A, "root node is A");

  // ---- cuGraphGetEdges (cuda.h adds an edgeData out-array in the _v2 form;
  // pick the signature the active header exposes) ----
  size_t nedges = 0;
#ifdef cuGraphGetEdges
  DRV(cuGraphGetEdges(graph, nullptr, nullptr, nullptr, &nedges));
#else
  DRV(cuGraphGetEdges(graph, nullptr, nullptr, &nedges));
#endif
  CHECK(nedges == 4, "cuGraphGetEdges count == 4 (A->B,B->C,B->D,C->D)");
  std::vector<CUgraphNode> from(nedges), to(nedges);
#ifdef cuGraphGetEdges
  DRV(cuGraphGetEdges(graph, from.data(), to.data(), nullptr, &nedges));
#else
  DRV(cuGraphGetEdges(graph, from.data(), to.data(), &nedges));
#endif
  std::set<std::pair<CUgraphNode, CUgraphNode>> edges;
  for (size_t i = 0; i < nedges; i++)
    edges.insert({from[i], to[i]});
  CHECK(edges.count({A, B}) && edges.count({B, C}) && edges.count({B, D}) &&
            edges.count({C, D}),
        "cuGraphGetEdges returned the expected edge set");

  // ---- cuGraphNodeGetDependencies (D depends on B and C) ----
  size_t ndeps = 0;
#ifdef cuGraphNodeGetDependencies
  DRV(cuGraphNodeGetDependencies(D, nullptr, nullptr, &ndeps));
#else
  DRV(cuGraphNodeGetDependencies(D, nullptr, &ndeps));
#endif
  CHECK(ndeps == 2, "cuGraphNodeGetDependencies(D) count == 2");
  std::vector<CUgraphNode> deps(ndeps);
#ifdef cuGraphNodeGetDependencies
  DRV(cuGraphNodeGetDependencies(D, deps.data(), nullptr, &ndeps));
#else
  DRV(cuGraphNodeGetDependencies(D, deps.data(), &ndeps));
#endif
  {
    std::set<CUgraphNode> s(deps.begin(), deps.end());
    CHECK(s.count(B) && s.count(C), "cuGraphNodeGetDependencies(D) == {B,C}");
  }

  // ---- cuGraphNodeGetDependentNodes (A's dependents == {B}) ----
  size_t ndependents = 0;
#ifdef cuGraphNodeGetDependentNodes
  DRV(cuGraphNodeGetDependentNodes(A, nullptr, nullptr, &ndependents));
#else
  DRV(cuGraphNodeGetDependentNodes(A, nullptr, &ndependents));
#endif
  CHECK(ndependents == 1, "cuGraphNodeGetDependentNodes(A) count == 1");
  std::vector<CUgraphNode> dependents(ndependents);
#ifdef cuGraphNodeGetDependentNodes
  DRV(cuGraphNodeGetDependentNodes(A, dependents.data(), nullptr,
                                   &ndependents));
#else
  DRV(cuGraphNodeGetDependentNodes(A, dependents.data(), &ndependents));
#endif
  CHECK(ndependents == 1 && dependents[0] == B,
        "cuGraphNodeGetDependentNodes(A) == {B}");

  // ---- host-node params: GetParams unwraps the trampoline to the original
  // fn/userData; SetParams updates them; the callback fires on launch ----
  CUDA_HOST_NODE_PARAMS got = {};
  DRV(cuGraphHostNodeGetParams(C, &got));
  CHECK(got.fn == host_cb && got.userData == (void *)0xCAFEF00D,
        "cuGraphHostNodeGetParams returns original fn/userData");
  CUDA_HOST_NODE_PARAMS np = {};
  np.fn = host_cb;
  np.userData = (void *)0xBEEF;
  DRV(cuGraphHostNodeSetParams(C, &np));
  CUDA_HOST_NODE_PARAMS got2 = {};
  DRV(cuGraphHostNodeGetParams(C, &got2));
  CHECK(got2.userData == (void *)0xBEEF,
        "cuGraphHostNodeSetParams updated userData");

  CUgraphExec exec;
  DRV(cuGraphInstantiateWithFlags(&exec, graph, 0));
  CUstream stream;
  DRV(cuStreamCreate(&stream, 0));
  DRV(cuGraphLaunch(exec, stream));
  DRV(cuStreamSynchronize(stream));
  CHECK(g_host_calls >= 1 && g_host_user == (void *)0xBEEF,
        "host callback fired with updated userData during launch");

  // ---- batch-mem-op node (embedded paramArray), exercised end-to-end ----
  CUdeviceptr dptr = 0;
  DRV(cuMemAlloc(&dptr, sizeof(unsigned int)));
  unsigned int zero = 0;
  DRV(cuMemcpyHtoD(dptr, &zero, sizeof(zero)));
  CUstreamBatchMemOpParams op = {};
  op.operation = CU_STREAM_MEM_OP_WRITE_VALUE_32;
  op.writeValue.address = dptr;
  op.writeValue.value = 0xABCD1234u;
  CUDA_BATCH_MEM_OP_NODE_PARAMS bp = {};
  bp.ctx = ctx;
  bp.count = 1;
  bp.paramArray = &op;
  CUgraph bgraph;
  DRV(cuGraphCreate(&bgraph, 0));
  CUgraphNode bnode;
  CUresult br = cuGraphAddBatchMemOpNode(&bnode, bgraph, nullptr, 0, &bp);
  if (br == CUDA_ERROR_NOT_SUPPORTED) {
    fprintf(stderr, "skip: batch mem ops not supported on this device\n");
  } else {
    CHECK(br == CUDA_SUCCESS, "cuGraphAddBatchMemOpNode succeeded");
    CUDA_BATCH_MEM_OP_NODE_PARAMS bgot = {};
    DRV(cuGraphBatchMemOpNodeGetParams(bnode, &bgot));
    CHECK(bgot.count == 1 && bgot.paramArray != nullptr &&
              bgot.paramArray[0].writeValue.address == dptr &&
              bgot.paramArray[0].writeValue.value == 0xABCD1234u,
          "cuGraphBatchMemOpNodeGetParams round-trips paramArray");
    CUgraphExec bexec;
    DRV(cuGraphInstantiateWithFlags(&bexec, bgraph, 0));
    DRV(cuGraphLaunch(bexec, stream));
    DRV(cuStreamSynchronize(stream));
    unsigned int readback = 0;
    DRV(cuMemcpyDtoH(&readback, dptr, sizeof(readback)));
    CHECK(readback == 0xABCD1234u, "batch mem op wrote value to device memory");
    DRV(cuGraphExecDestroy(bexec));
  }
  DRV(cuGraphDestroy(bgraph));
  DRV(cuMemFree(dptr));

  // ---- array / texture-object creators (DEREF / NULLABLE codegen) ----
  CUDA_ARRAY_DESCRIPTOR ad = {};
  ad.Width = 16;
  ad.Height = 16;
  ad.Format = CU_AD_FORMAT_FLOAT;
  ad.NumChannels = 1;
  CUarray arr = nullptr;
  CHECK(cuArrayCreate(&arr, &ad) == CUDA_SUCCESS && arr != nullptr,
        "cuArrayCreate created a 2D array");
  CUDA_ARRAY3D_DESCRIPTOR ad3 = {};
  ad3.Width = 8;
  ad3.Height = 8;
  ad3.Depth = 2;
  ad3.Format = CU_AD_FORMAT_FLOAT;
  ad3.NumChannels = 1;
  CUarray arr3 = nullptr;
  CHECK(cuArray3DCreate(&arr3, &ad3) == CUDA_SUCCESS && arr3 != nullptr,
        "cuArray3DCreate created a 3D array");
  CUDA_RESOURCE_DESC rd = {};
  rd.resType = CU_RESOURCE_TYPE_ARRAY;
  rd.res.array.hArray = arr;
  CUDA_TEXTURE_DESC td = {};
  td.addressMode[0] = CU_TR_ADDRESS_MODE_CLAMP;
  td.addressMode[1] = CU_TR_ADDRESS_MODE_CLAMP;
  td.filterMode = CU_TR_FILTER_MODE_POINT;
  CUtexObject tex = 0;
  CHECK(cuTexObjectCreate(&tex, &rd, &td, nullptr) == CUDA_SUCCESS && tex != 0,
        "cuTexObjectCreate created a tex object from the array");
  if (tex)
    DRV(cuTexObjectDestroy(tex));
  if (arr)
    DRV(cuArrayDestroy(arr));
  if (arr3)
    DRV(cuArrayDestroy(arr3));

  DRV(cuStreamDestroy(stream));
  DRV(cuGraphExecDestroy(exec));
  DRV(cuGraphDestroy(graph));
  cuDevicePrimaryCtxRelease(dev);

  fprintf(stderr, "\n%s (%d failure(s))\n",
          g_failures == 0 ? "ALL PASSED" : "FAILURES", g_failures);
  return g_failures == 0 ? 0 : 1;
}
