// nvSHMEM's host library (libnvshmem_host.so.3) on the LUPINE client.
//
// nvSHMEM is a partitioned global address space over the GPUs of a job. Its
// PEs read and write each other's symmetric heap directly, mostly from inside
// kernels, over NVLink or a network fabric, and the host library exists to
// bootstrap that and to drive it from the host. None of it survives the split
// between a LUPINE client and a server:
//
//   - A PE is a process that owns a GPU. Under LUPINE the process holding the
//     GPU is the server, which serves many clients at once, and nvSHMEM keeps
//     its PE identity, its symmetric heap and its teams in process-global
//     state with no handle to separate one job from another. A server cannot
//     be two PEs, so forwarding the host API would make two clients of one
//     server silently share one job's state.
//   - The symmetric heap is mapped by nvSHMEM at addresses of its own
//     choosing, outside the identity VA arena a client reserves, so a pointer
//     from nvshmem_malloc is not one the client can hold.
//   - The device-side half runs inside kernels, from the device library linked
//     into the caller's own binary, and never reaches a host shim at all.
//
// So this shim forwards nothing and has no server side. Every entry point
// reports once and refuses: initialization returns NVSHMEMX_ERROR_NOT_SUPPORTED
// and the rest, which a correct program only reaches after initialization
// succeeded, return an error, a null pointer or a zero. Answering a collective
// or a peer pointer with a plausible value instead would be a wrong answer
// inside a communication library, which is worse than no answer.
//
// What it does do is load. PyTorch's libtorch_nvshmem.so, pulled in by
// libtorch_cuda.so's NEEDED, resolves twenty nvSHMEM symbols at load time
// without calling any of them unless a program asks for symmetric memory, and
// that is the whole of the immediate motivation.

#include <nvshmem_host.h>

#include <atomic>
#include <cstdlib>
#include <cstring>

#include "lupine_log.h"

namespace {

// The first entry point a caller reaches names itself; the rest stay quiet so
// a loop cannot flood the log.
void report(const char *entry_point) {
  static std::atomic<bool> reported{false};
  if (reported.exchange(true)) {
    return;
  }
  LUPINE_LOG_ERROR(
      "lupine: nvSHMEM is not available over a forwarded connection ("
      << entry_point
      << "). A PE owns its GPU and its symmetric heap in process-global "
         "state, which a server shared by many clients cannot hold.");
}

int refuse_status(const char *entry_point) {
  report(entry_point);
  return NVSHMEMX_ERROR_NOT_SUPPORTED;
}

template <typename T> T refuse_value(const char *entry_point) {
  report(entry_point);
  return T();
}

} // namespace

#define LUPINE_NVSHMEM_STRINGIFY(name) #name
#define LUPINE_NVSHMEM_NAME(name) LUPINE_NVSHMEM_STRINGIFY(name)

// NVIDIA's header undefines its own declaration macros but leaves the
// NVSHMEMI_REPT_* drivers standing, so the same repertoire that declared these
// entry points defines them here and the shim's surface follows the header it
// was built against.
#define LUPINE_NVSHMEM_VOID(signature, name)                                   \
  signature { report(LUPINE_NVSHMEM_NAME(name)); }
#define LUPINE_NVSHMEM_VALUE(type, signature, name)                            \
  signature { return refuse_value<type>(LUPINE_NVSHMEM_NAME(name)); }
#define LUPINE_NVSHMEM_STATUS(signature, name)                                 \
  signature { return refuse_status(LUPINE_NVSHMEM_NAME(name)); }

// The library declares half-precision fetch, set and swap atomics but defines
// only the float and double ones, so the shim's extended-AMO repertoire has to
// drop half for exactly those three families to export what NVIDIA exports.
#define LUPINE_REPT_EXTENDED_AMO_NO_HALF(OPGRPNAME, opname)                    \
  NVSHMEMI_DECL_TYPE_##OPGRPNAME(float, float, opname)                         \
      NVSHMEMI_DECL_TYPE_##OPGRPNAME(double, double, opname)
#define LUPINE_REPT_EXTENDED_AMO(OPGRPNAME, opname)                            \
  NVSHMEMI_DECL_TYPE_##OPGRPNAME(half, __half, opname)                         \
      LUPINE_REPT_EXTENDED_AMO_NO_HALF(OPGRPNAME, opname)

//////////////////// Initialization and status ////////////////////

// The header's nvshmem_init is a static inline that reaches the library
// through nvshmemi_init_thread in the device archive, so NVIDIA's host library
// does not define it and the reference device linking leaves behind is weak
// and unversioned. The shim defines it anyway so nothing can call through a
// null pointer, and exits the way the header's inline does when initialization
// fails, a void return having no other way to say so.
extern "C" void lupine_nvshmem_init() asm("nvshmem_init");
void lupine_nvshmem_init() {
  report("nvshmem_init");
  exit(-1);
}

int nvshmemid_hostlib_init_attr(int, int *, unsigned int,
                                nvshmemx_init_attr_t *, nvshmemi_version_t,
                                nvshmemx_device_lib_init_cb) {
  return refuse_status("nvshmemid_hostlib_init_attr");
}

int nvshmemx_hostlib_init_attr(unsigned int, nvshmemx_init_attr_t *) {
  return refuse_status("nvshmemx_hostlib_init_attr");
}

void nvshmemid_hostlib_finalize(void *, void *) {
  report("nvshmemid_hostlib_finalize");
}

void nvshmemx_hostlib_finalize() { report("nvshmemx_hostlib_finalize"); }

// Nothing was ever initialized, which is the honest answer and the one that
// lets a caller that probes first take its own path.
int nvshmemid_init_status() { return NVSHMEM_STATUS_NOT_INITIALIZED; }

int nvshmemx_init_status() { return NVSHMEM_STATUS_NOT_INITIALIZED; }

void nvshmem_query_thread(int *provided) {
  if (provided != nullptr) {
    *provided = NVSHMEM_THREAD_SINGLE;
  }
}

void nvshmem_global_exit(int status) { exit(status); }

// The version a caller asks about is the API this library speaks, which the
// header it was built against fixes, so these three answer from the header
// rather than refuse.
void nvshmem_info_get_version(int *major, int *minor) {
  if (major != nullptr) {
    *major = NVSHMEM_MAJOR_VERSION;
  }
  if (minor != nullptr) {
    *minor = NVSHMEM_MINOR_VERSION;
  }
}

void nvshmem_info_get_name(char *name) {
  if (name != nullptr) {
    strncpy(name, NVSHMEM_VENDOR_STRING, NVSHMEM_MAX_NAME_LEN - 1);
    name[NVSHMEM_MAX_NAME_LEN - 1] = '\0';
  }
}

void nvshmemx_vendor_get_version_info(int *major, int *minor, int *patch) {
  if (major != nullptr) {
    *major = NVSHMEM_VENDOR_MAJOR_VERSION;
  }
  if (minor != nullptr) {
    *minor = NVSHMEM_VENDOR_MINOR_VERSION;
  }
  if (patch != nullptr) {
    *patch = NVSHMEM_VENDOR_PATCH_VERSION;
  }
}

int nvshmem_my_pe() { return refuse_status("nvshmem_my_pe"); }
int nvshmem_n_pes() { return refuse_status("nvshmem_n_pes"); }

// NVIDIA's host library exports this as data from nvSHMEM 3.5 on, where its
// header starts declaring it, so the shim carries it over the same range.
#if NVSHMEM_VENDOR_VERSION >= 30500
long nvshmem_error = 0;
#endif

//////////////////// Heap management ////////////////////

void *nvshmem_malloc(size_t) { return refuse_value<void *>("nvshmem_malloc"); }
void *nvshmem_calloc(size_t, size_t) {
  return refuse_value<void *>("nvshmem_calloc");
}
void *nvshmem_align(size_t, size_t) {
  return refuse_value<void *>("nvshmem_align");
}
void nvshmem_free(void *) { report("nvshmem_free"); }
void *nvshmem_ptr(const void *, int) {
  return refuse_value<void *>("nvshmem_ptr");
}
void *nvshmemx_mc_ptr(nvshmem_team_t, const void *) {
  return refuse_value<void *>("nvshmemx_mc_ptr");
}

int nvshmemx_buffer_register(void *, size_t) {
  return refuse_status("nvshmemx_buffer_register");
}
int nvshmemx_buffer_unregister(void *) {
  return refuse_status("nvshmemx_buffer_unregister");
}
void nvshmemx_buffer_unregister_all() {
  report("nvshmemx_buffer_unregister_all");
}
void *nvshmemx_buffer_register_symmetric(void *, size_t, int) {
  return refuse_value<void *>("nvshmemx_buffer_register_symmetric");
}
void *nvshmemx_buffer_register_symmetric_at_preferred_address(void *, size_t,
                                                              void *, int) {
  return refuse_value<void *>(
      "nvshmemx_buffer_register_symmetric_at_preferred_address");
}
int nvshmemx_buffer_unregister_symmetric(void *, size_t) {
  return refuse_status("nvshmemx_buffer_unregister_symmetric");
}

//////////////////// Bootstrap and module registration ////////////////////

int nvshmemx_set_attr_uniqueid_args(const int, const int,
                                    const nvshmemx_uniqueid_t *,
                                    nvshmemx_init_attr_t *) {
  return refuse_status("nvshmemx_set_attr_uniqueid_args");
}
int nvshmemx_set_attr_mpi_comm_args(void *, nvshmemx_init_attr_t *) {
  return refuse_status("nvshmemx_set_attr_mpi_comm_args");
}
int nvshmemx_get_uniqueid(nvshmemx_uniqueid_t *) {
  return refuse_status("nvshmemx_get_uniqueid");
}
int nvshmemx_cumodule_init(CUmodule) {
  return refuse_status("nvshmemx_cumodule_init");
}
int nvshmemx_cumodule_finalize(CUmodule) {
  return refuse_status("nvshmemx_cumodule_finalize");
}
int nvshmemx_culibrary_init(CUlibrary) {
  return refuse_status("nvshmemx_culibrary_init");
}
int nvshmemx_culibrary_finalize(CUlibrary) {
  return refuse_status("nvshmemx_culibrary_finalize");
}
// Queue pairs arrive with nvSHMEM 3.5; older headers declare no handle type for
// this to hand back.
#if NVSHMEM_VENDOR_VERSION >= 30500
int nvshmemx_qp_create(int, nvshmemx_qp_handle_t **) {
  return refuse_status("nvshmemx_qp_create");
}
#endif

//////////////////// Teams ////////////////////

int nvshmem_team_my_pe(nvshmem_team_t) {
  return refuse_status("nvshmem_team_my_pe");
}
int nvshmem_team_n_pes(nvshmem_team_t) {
  return refuse_status("nvshmem_team_n_pes");
}
void nvshmem_team_get_config(nvshmem_team_t, nvshmem_team_config_t *) {
  report("nvshmem_team_get_config");
}
int nvshmem_team_translate_pe(nvshmem_team_t, int, nvshmem_team_t) {
  return refuse_status("nvshmem_team_translate_pe");
}
int nvshmem_team_split_strided(nvshmem_team_t, int, int, int,
                               const nvshmem_team_config_t *, long,
                               nvshmem_team_t *) {
  return refuse_status("nvshmem_team_split_strided");
}
int nvshmem_team_split_2d(nvshmem_team_t, int, const nvshmem_team_config_t *,
                          long, nvshmem_team_t *, const nvshmem_team_config_t *,
                          long, nvshmem_team_t *) {
  return refuse_status("nvshmem_team_split_2d");
}
void nvshmem_team_destroy(nvshmem_team_t) { report("nvshmem_team_destroy"); }
int nvshmemx_team_get_uniqueid(nvshmemx_team_uniqueid_t *) {
  return refuse_status("nvshmemx_team_get_uniqueid");
}
int nvshmemx_team_init(nvshmem_team_t *, nvshmem_team_config_t *, long, int,
                       int) {
  return refuse_status("nvshmemx_team_init");
}

//////////////////// Atomics ////////////////////

#define NVSHMEMI_DECL_TYPE_INC(type, TYPE, opname)                             \
  LUPINE_NVSHMEM_VOID(void nvshmem_##type##_atomic_##opname(TYPE *, int),      \
                      nvshmem_##type##_atomic_##opname)

#define NVSHMEMI_DECL_TYPE_FINC(type, TYPE, opname)                            \
  LUPINE_NVSHMEM_VALUE(TYPE,                                                   \
                       TYPE nvshmem_##type##_atomic_##opname(TYPE *, int),     \
                       nvshmem_##type##_atomic_##opname)

#define NVSHMEMI_DECL_TYPE_FETCH(type, TYPE, opname)                           \
  LUPINE_NVSHMEM_VALUE(                                                        \
      TYPE, TYPE nvshmem_##type##_atomic_##opname(const TYPE *, int),          \
      nvshmem_##type##_atomic_##opname)

#define NVSHMEMI_DECL_TYPE_ADD_SET(type, TYPE, opname)                         \
  LUPINE_NVSHMEM_VOID(                                                         \
      void nvshmem_##type##_atomic_##opname(TYPE *, TYPE, int),                \
      nvshmem_##type##_atomic_##opname)

#define NVSHMEMI_DECL_TYPE_FADD_SWAP(type, TYPE, opname)                       \
  LUPINE_NVSHMEM_VALUE(                                                        \
      TYPE, TYPE nvshmem_##type##_atomic_##opname(TYPE *, TYPE, int),          \
      nvshmem_##type##_atomic_##opname)

#define NVSHMEMI_DECL_TYPE_CSWAP(type, TYPE, opname)                           \
  LUPINE_NVSHMEM_VALUE(                                                        \
      TYPE, TYPE nvshmem_##type##_atomic_##opname(TYPE *, TYPE, TYPE, int),    \
      nvshmem_##type##_atomic_##opname)

#define NVSHMEMI_DECL_TYPE_XADD(type, TYPE, opname)                            \
  LUPINE_NVSHMEM_VOID(                                                         \
      void nvshmemx_##type##_atomic_##opname(TYPE *, TYPE, int),               \
      nvshmemx_##type##_atomic_##opname)

#define NVSHMEMI_DECL_TYPE_XFADD(type, TYPE, opname)                           \
  LUPINE_NVSHMEM_VALUE(                                                        \
      TYPE, TYPE nvshmemx_##type##_atomic_##opname(TYPE *, TYPE, int),         \
      nvshmemx_##type##_atomic_##opname)

#define NVSHMEMI_DECL_TYPE_AND_OR_XOR(type, TYPE, opname)                      \
  LUPINE_NVSHMEM_VOID(                                                         \
      void nvshmem_##type##_atomic_##opname(TYPE *, TYPE, int),                \
      nvshmem_##type##_atomic_##opname)

#define NVSHMEMI_DECL_TYPE_FAND_FOR_FXOR(type, TYPE, opname)                   \
  LUPINE_NVSHMEM_VALUE(                                                        \
      TYPE, TYPE nvshmem_##type##_atomic_fetch_##opname(TYPE *, TYPE, int),    \
      nvshmem_##type##_atomic_fetch_##opname)

NVSHMEMI_REPT_OPGROUP_FOR_BITWISE_AMO(INC, inc)
NVSHMEMI_REPT_OPGROUP_FOR_STANDARD_AMO(INC, inc)

NVSHMEMI_REPT_OPGROUP_FOR_BITWISE_AMO(FINC, fetch_inc)
NVSHMEMI_REPT_OPGROUP_FOR_STANDARD_AMO(FINC, fetch_inc)

NVSHMEMI_REPT_OPGROUP_FOR_BITWISE_AMO(FETCH, fetch)
NVSHMEMI_REPT_OPGROUP_FOR_STANDARD_AMO(FETCH, fetch)
LUPINE_REPT_EXTENDED_AMO_NO_HALF(FETCH, fetch)

NVSHMEMI_REPT_OPGROUP_FOR_BITWISE_AMO(ADD_SET, add)
NVSHMEMI_REPT_OPGROUP_FOR_STANDARD_AMO(ADD_SET, add)

LUPINE_REPT_EXTENDED_AMO(XADD, add)
LUPINE_REPT_EXTENDED_AMO(XFADD, fetch_add)

NVSHMEMI_REPT_OPGROUP_FOR_BITWISE_AMO(ADD_SET, set)
NVSHMEMI_REPT_OPGROUP_FOR_STANDARD_AMO(ADD_SET, set)
LUPINE_REPT_EXTENDED_AMO_NO_HALF(ADD_SET, set)

NVSHMEMI_REPT_OPGROUP_FOR_BITWISE_AMO(FADD_SWAP, fetch_add)
NVSHMEMI_REPT_OPGROUP_FOR_STANDARD_AMO(FADD_SWAP, fetch_add)

NVSHMEMI_REPT_OPGROUP_FOR_BITWISE_AMO(FADD_SWAP, swap)
NVSHMEMI_REPT_OPGROUP_FOR_STANDARD_AMO(FADD_SWAP, swap)
LUPINE_REPT_EXTENDED_AMO_NO_HALF(FADD_SWAP, swap)

NVSHMEMI_REPT_OPGROUP_FOR_BITWISE_AMO(CSWAP, compare_swap)
NVSHMEMI_REPT_OPGROUP_FOR_STANDARD_AMO(CSWAP, compare_swap)

NVSHMEMI_REPT_OPGROUP_FOR_BITWISE_AMO(AND_OR_XOR, and)
NVSHMEMI_REPT_OPGROUP_FOR_BITWISE_AMO(AND_OR_XOR, or)
NVSHMEMI_REPT_OPGROUP_FOR_BITWISE_AMO(AND_OR_XOR, xor)

NVSHMEMI_REPT_OPGROUP_FOR_BITWISE_AMO(FAND_FOR_FXOR, and)
NVSHMEMI_REPT_OPGROUP_FOR_BITWISE_AMO(FAND_FOR_FXOR, or)
NVSHMEMI_REPT_OPGROUP_FOR_BITWISE_AMO(FAND_FOR_FXOR, xor)

//////////////////// Remote memory access ////////////////////

#define LUPINE_DECL_TYPE_P(NAME, TYPE)                                         \
  LUPINE_NVSHMEM_VOID(void nvshmem_##NAME##_p(TYPE *, const TYPE, int),        \
                      nvshmem_##NAME##_p)

#define LUPINE_DECL_TYPE_G(NAME, TYPE)                                         \
  LUPINE_NVSHMEM_VALUE(TYPE, TYPE nvshmem_##NAME##_g(const TYPE *, int),       \
                       nvshmem_##NAME##_g)

#define LUPINE_DECL_TYPE_RMA(NAME, TYPE)                                       \
  LUPINE_NVSHMEM_VOID(                                                         \
      void nvshmem_##NAME##_put(TYPE *, const TYPE *, size_t, int),            \
      nvshmem_##NAME##_put)                                                    \
  LUPINE_NVSHMEM_VOID(                                                         \
      void nvshmem_##NAME##_put_nbi(TYPE *, const TYPE *, size_t, int),        \
      nvshmem_##NAME##_put_nbi)                                                \
  LUPINE_NVSHMEM_VOID(                                                         \
      void nvshmem_##NAME##_get(TYPE *, const TYPE *, size_t, int),            \
      nvshmem_##NAME##_get)                                                    \
  LUPINE_NVSHMEM_VOID(                                                         \
      void nvshmem_##NAME##_get_nbi(TYPE *, const TYPE *, size_t, int),        \
      nvshmem_##NAME##_get_nbi)                                                \
  LUPINE_NVSHMEM_VOID(void nvshmem_##NAME##_iput(TYPE *, const TYPE *,         \
                                                 ptrdiff_t, ptrdiff_t, size_t, \
                                                 int),                         \
                      nvshmem_##NAME##_iput)                                   \
  LUPINE_NVSHMEM_VOID(void nvshmem_##NAME##_iget(TYPE *, const TYPE *,         \
                                                 ptrdiff_t, ptrdiff_t, size_t, \
                                                 int),                         \
                      nvshmem_##NAME##_iget)

#define LUPINE_DECL_SIZE_RMA(NAME)                                             \
  LUPINE_NVSHMEM_VOID(                                                         \
      void nvshmem_put##NAME(void *, const void *, size_t, int),               \
      nvshmem_put##NAME)                                                       \
  LUPINE_NVSHMEM_VOID(                                                         \
      void nvshmem_put##NAME##_nbi(void *, const void *, size_t, int),         \
      nvshmem_put##NAME##_nbi)                                                 \
  LUPINE_NVSHMEM_VOID(                                                         \
      void nvshmem_get##NAME(void *, const void *, size_t, int),               \
      nvshmem_get##NAME)                                                       \
  LUPINE_NVSHMEM_VOID(                                                         \
      void nvshmem_get##NAME##_nbi(void *, const void *, size_t, int),         \
      nvshmem_get##NAME##_nbi)                                                 \
  LUPINE_NVSHMEM_VOID(void nvshmem_iput##NAME(void *, const void *, ptrdiff_t, \
                                              ptrdiff_t, size_t, int),         \
                      nvshmem_iput##NAME)                                      \
  LUPINE_NVSHMEM_VOID(void nvshmem_iget##NAME(void *, const void *, ptrdiff_t, \
                                              ptrdiff_t, size_t, int),         \
                      nvshmem_iget##NAME)

NVSHMEMI_REPT_FOR_STANDARD_RMA_TYPES(LUPINE_DECL_TYPE_P)
NVSHMEMI_REPT_FOR_STANDARD_RMA_TYPES(LUPINE_DECL_TYPE_G)
NVSHMEMI_REPT_FOR_STANDARD_RMA_TYPES(LUPINE_DECL_TYPE_RMA)
NVSHMEMI_REPT_FOR_SIZES(LUPINE_DECL_SIZE_RMA)

void nvshmem_putmem(void *, const void *, size_t, int) {
  report("nvshmem_putmem");
}
void nvshmem_putmem_nbi(void *, const void *, size_t, int) {
  report("nvshmem_putmem_nbi");
}
void nvshmem_getmem(void *, const void *, size_t, int) {
  report("nvshmem_getmem");
}
void nvshmem_getmem_nbi(void *, const void *, size_t, int) {
  report("nvshmem_getmem_nbi");
}

uint64_t nvshmem_signal_fetch(uint64_t *) {
  return refuse_value<uint64_t>("nvshmem_signal_fetch");
}
void nvshmem_quiet() { report("nvshmem_quiet"); }
void nvshmem_fence() { report("nvshmem_fence"); }
void nvshmemx_flush() { report("nvshmemx_flush"); }

//////////////////// Collectives ////////////////////

#define LUPINE_DECL_TYPE_COLL(NAME, TYPE)                                      \
  LUPINE_NVSHMEM_STATUS(int nvshmem_##NAME##_alltoall(nvshmem_team_t, TYPE *,  \
                                                      const TYPE *, size_t),   \
                        nvshmem_##NAME##_alltoall)                             \
  LUPINE_NVSHMEM_STATUS(int nvshmem_##NAME##_broadcast(nvshmem_team_t, TYPE *, \
                                                       const TYPE *, size_t,   \
                                                       int),                   \
                        nvshmem_##NAME##_broadcast)                            \
  LUPINE_NVSHMEM_STATUS(int nvshmem_##NAME##_fcollect(nvshmem_team_t, TYPE *,  \
                                                      const TYPE *, size_t),   \
                        nvshmem_##NAME##_fcollect)

#define NVSHMEMI_DECL_TEAM_REDUCE(NAME, TYPE, OP)                              \
  LUPINE_NVSHMEM_STATUS(int nvshmem_##NAME##_##OP##_reduce(                    \
                            nvshmem_team_t, TYPE *, const TYPE *, size_t),     \
                        nvshmem_##NAME##_##OP##_reduce)                        \
  LUPINE_NVSHMEM_STATUS(int nvshmem_##NAME##_##OP##_reducescatter(             \
                            nvshmem_team_t, TYPE *, const TYPE *, size_t),     \
                        nvshmem_##NAME##_##OP##_reducescatter)

NVSHMEMI_REPT_FOR_STANDARD_RMA_TYPES(LUPINE_DECL_TYPE_COLL)

NVSHMEMI_REPT_FOR_BITWISE_REDUCE_TYPES(NVSHMEMI_DECL_TEAM_REDUCE, and)
NVSHMEMI_REPT_FOR_BITWISE_REDUCE_TYPES(NVSHMEMI_DECL_TEAM_REDUCE, or)
NVSHMEMI_REPT_FOR_BITWISE_REDUCE_TYPES(NVSHMEMI_DECL_TEAM_REDUCE, xor)
NVSHMEMI_REPT_FOR_STANDARD_REDUCE_TYPES(NVSHMEMI_DECL_TEAM_REDUCE, max)
NVSHMEMI_REPT_FOR_STANDARD_REDUCE_TYPES(NVSHMEMI_DECL_TEAM_REDUCE, min)
NVSHMEMI_REPT_FOR_ARITH_REDUCE_TYPES(NVSHMEMI_DECL_TEAM_REDUCE, sum)
NVSHMEMI_REPT_FOR_ARITH_REDUCE_TYPES(NVSHMEMI_DECL_TEAM_REDUCE, prod)

int nvshmem_alltoallmem(nvshmem_team_t, void *, const void *, size_t) {
  return refuse_status("nvshmem_alltoallmem");
}
int nvshmem_broadcastmem(nvshmem_team_t, void *, const void *, size_t, int) {
  return refuse_status("nvshmem_broadcastmem");
}
int nvshmem_fcollectmem(nvshmem_team_t, void *, const void *, size_t) {
  return refuse_status("nvshmem_fcollectmem");
}
int nvshmem_barrier(nvshmem_team_t) { return refuse_status("nvshmem_barrier"); }
void nvshmem_barrier_all() { report("nvshmem_barrier_all"); }
int nvshmem_team_sync(nvshmem_team_t) {
  return refuse_status("nvshmem_team_sync");
}
void nvshmem_sync_all() { report("nvshmem_sync_all"); }

//////////////////// Stream-ordered operations ////////////////////

#define LUPINE_DECL_TYPE_P_ON_STREAM(NAME, TYPE)                               \
  LUPINE_NVSHMEM_VOID(void nvshmemx_##NAME##_p_on_stream(TYPE *, const TYPE,   \
                                                         int, cudaStream_t),   \
                      nvshmemx_##NAME##_p_on_stream)

#define LUPINE_DECL_TYPE_G_ON_STREAM(NAME, TYPE)                               \
  LUPINE_NVSHMEM_VALUE(                                                        \
      TYPE,                                                                    \
      TYPE nvshmemx_##NAME##_g_on_stream(const TYPE *, int, cudaStream_t),     \
      nvshmemx_##NAME##_g_on_stream)

#define LUPINE_DECL_TYPE_RMA_ON_STREAM(NAME, TYPE)                             \
  LUPINE_NVSHMEM_VOID(void nvshmemx_##NAME##_put_on_stream(                    \
                          TYPE *, const TYPE *, size_t, int, cudaStream_t),    \
                      nvshmemx_##NAME##_put_on_stream)                         \
  LUPINE_NVSHMEM_VOID(void nvshmemx_##NAME##_put_nbi_on_stream(                \
                          TYPE *, const TYPE *, size_t, int, cudaStream_t),    \
                      nvshmemx_##NAME##_put_nbi_on_stream)                     \
  LUPINE_NVSHMEM_VOID(void nvshmemx_##NAME##_get_on_stream(                    \
                          TYPE *, const TYPE *, size_t, int, cudaStream_t),    \
                      nvshmemx_##NAME##_get_on_stream)                         \
  LUPINE_NVSHMEM_VOID(void nvshmemx_##NAME##_get_nbi_on_stream(                \
                          TYPE *, const TYPE *, size_t, int, cudaStream_t),    \
                      nvshmemx_##NAME##_get_nbi_on_stream)                     \
  LUPINE_NVSHMEM_VOID(void nvshmemx_##NAME##_iput_on_stream(                   \
                          TYPE *, const TYPE *, ptrdiff_t, ptrdiff_t, size_t,  \
                          int, cudaStream_t),                                  \
                      nvshmemx_##NAME##_iput_on_stream)                        \
  LUPINE_NVSHMEM_VOID(void nvshmemx_##NAME##_iget_on_stream(                   \
                          TYPE *, const TYPE *, ptrdiff_t, ptrdiff_t, size_t,  \
                          int, cudaStream_t),                                  \
                      nvshmemx_##NAME##_iget_on_stream)                        \
  LUPINE_NVSHMEM_VOID(void nvshmemx_##NAME##_put_signal_on_stream(             \
                          TYPE *, const TYPE *, size_t, uint64_t *, uint64_t,  \
                          int, int, cudaStream_t),                             \
                      nvshmemx_##NAME##_put_signal_on_stream)                  \
  LUPINE_NVSHMEM_VOID(void nvshmemx_##NAME##_put_signal_nbi_on_stream(         \
                          TYPE *, const TYPE *, size_t, uint64_t *, uint64_t,  \
                          int, int, cudaStream_t),                             \
                      nvshmemx_##NAME##_put_signal_nbi_on_stream)

#define LUPINE_DECL_SIZE_RMA_ON_STREAM(NAME)                                   \
  LUPINE_NVSHMEM_VOID(void nvshmemx_put##NAME##_on_stream(                     \
                          void *, const void *, size_t, int, cudaStream_t),    \
                      nvshmemx_put##NAME##_on_stream)                          \
  LUPINE_NVSHMEM_VOID(void nvshmemx_put##NAME##_nbi_on_stream(                 \
                          void *, const void *, size_t, int, cudaStream_t),    \
                      nvshmemx_put##NAME##_nbi_on_stream)                      \
  LUPINE_NVSHMEM_VOID(void nvshmemx_get##NAME##_on_stream(                     \
                          void *, const void *, size_t, int, cudaStream_t),    \
                      nvshmemx_get##NAME##_on_stream)                          \
  LUPINE_NVSHMEM_VOID(void nvshmemx_get##NAME##_nbi_on_stream(                 \
                          void *, const void *, size_t, int, cudaStream_t),    \
                      nvshmemx_get##NAME##_nbi_on_stream)                      \
  LUPINE_NVSHMEM_VOID(void nvshmemx_iput##NAME##_on_stream(                    \
                          void *, const void *, ptrdiff_t, ptrdiff_t, size_t,  \
                          int, cudaStream_t),                                  \
                      nvshmemx_iput##NAME##_on_stream)                         \
  LUPINE_NVSHMEM_VOID(void nvshmemx_iget##NAME##_on_stream(                    \
                          void *, const void *, ptrdiff_t, ptrdiff_t, size_t,  \
                          int, cudaStream_t),                                  \
                      nvshmemx_iget##NAME##_on_stream)                         \
  LUPINE_NVSHMEM_VOID(void nvshmemx_put##NAME##_signal_on_stream(              \
                          void *, const void *, size_t, uint64_t *, uint64_t,  \
                          int, int, cudaStream_t),                             \
                      nvshmemx_put##NAME##_signal_on_stream)                   \
  LUPINE_NVSHMEM_VOID(void nvshmemx_put##NAME##_signal_nbi_on_stream(          \
                          void *, const void *, size_t, uint64_t *, uint64_t,  \
                          int, int, cudaStream_t),                             \
                      nvshmemx_put##NAME##_signal_nbi_on_stream)

#define LUPINE_DECL_WAIT_ON_STREAM(NAME, TYPE)                                 \
  LUPINE_NVSHMEM_VOID(void nvshmemx_##NAME##_wait_until_on_stream(             \
                          TYPE *, int, TYPE, cudaStream_t),                    \
                      nvshmemx_##NAME##_wait_until_on_stream)                  \
  LUPINE_NVSHMEM_VOID(                                                         \
      void nvshmemx_##NAME##_wait_until_all_on_stream(                         \
          TYPE *, size_t, const int *, int, TYPE, cudaStream_t),               \
      nvshmemx_##NAME##_wait_until_all_on_stream)                              \
  LUPINE_NVSHMEM_VOID(                                                         \
      void nvshmemx_##NAME##_wait_until_all_vector_on_stream(                  \
          TYPE *, size_t, const int *, int, TYPE *, cudaStream_t),             \
      nvshmemx_##NAME##_wait_until_all_vector_on_stream)

#define LUPINE_DECL_TYPE_COLL_ON_STREAM(NAME, TYPE)                            \
  LUPINE_NVSHMEM_STATUS(                                                       \
      int nvshmemx_##NAME##_alltoall_on_stream(                                \
          nvshmem_team_t, TYPE *, const TYPE *, size_t, cudaStream_t),         \
      nvshmemx_##NAME##_alltoall_on_stream)                                    \
  LUPINE_NVSHMEM_STATUS(                                                       \
      int nvshmemx_##NAME##_broadcast_on_stream(                               \
          nvshmem_team_t, TYPE *, const TYPE *, size_t, int, cudaStream_t),    \
      nvshmemx_##NAME##_broadcast_on_stream)                                   \
  LUPINE_NVSHMEM_STATUS(                                                       \
      int nvshmemx_##NAME##_fcollect_on_stream(                                \
          nvshmem_team_t, TYPE *, const TYPE *, size_t, cudaStream_t),         \
      nvshmemx_##NAME##_fcollect_on_stream)

#define NVSHMEMI_DECL_REDUCE_ONSTREAM(NAME, TYPE, OP)                          \
  LUPINE_NVSHMEM_STATUS(                                                       \
      int nvshmemx_##NAME##_##OP##_reduce_on_stream(                           \
          nvshmem_team_t, TYPE *, const TYPE *, size_t, cudaStream_t),         \
      nvshmemx_##NAME##_##OP##_reduce_on_stream)                               \
  LUPINE_NVSHMEM_STATUS(                                                       \
      int nvshmemx_##NAME##_##OP##_reducescatter_on_stream(                    \
          nvshmem_team_t, TYPE *, const TYPE *, size_t, cudaStream_t),         \
      nvshmemx_##NAME##_##OP##_reducescatter_on_stream)

NVSHMEMI_REPT_FOR_STANDARD_RMA_TYPES(LUPINE_DECL_TYPE_P_ON_STREAM)
NVSHMEMI_REPT_FOR_STANDARD_RMA_TYPES(LUPINE_DECL_TYPE_G_ON_STREAM)
NVSHMEMI_REPT_FOR_STANDARD_RMA_TYPES(LUPINE_DECL_TYPE_RMA_ON_STREAM)
NVSHMEMI_REPT_FOR_SIZES(LUPINE_DECL_SIZE_RMA_ON_STREAM)
NVSHMEMI_REPT_FOR_WAIT_TYPES(LUPINE_DECL_WAIT_ON_STREAM)
NVSHMEMI_REPT_FOR_STANDARD_RMA_TYPES(LUPINE_DECL_TYPE_COLL_ON_STREAM)

NVSHMEMI_REPT_FOR_BITWISE_REDUCE_TYPES(NVSHMEMI_DECL_REDUCE_ONSTREAM, and)
NVSHMEMI_REPT_FOR_BITWISE_REDUCE_TYPES(NVSHMEMI_DECL_REDUCE_ONSTREAM, or)
NVSHMEMI_REPT_FOR_BITWISE_REDUCE_TYPES(NVSHMEMI_DECL_REDUCE_ONSTREAM, xor)
NVSHMEMI_REPT_FOR_STANDARD_REDUCE_TYPES(NVSHMEMI_DECL_REDUCE_ONSTREAM, max)
NVSHMEMI_REPT_FOR_STANDARD_REDUCE_TYPES(NVSHMEMI_DECL_REDUCE_ONSTREAM, min)
NVSHMEMI_REPT_FOR_ARITH_REDUCE_TYPES(NVSHMEMI_DECL_REDUCE_ONSTREAM, sum)
NVSHMEMI_REPT_FOR_ARITH_REDUCE_TYPES(NVSHMEMI_DECL_REDUCE_ONSTREAM, prod)

void nvshmemx_putmem_on_stream(void *, const void *, size_t, int,
                               cudaStream_t) {
  report("nvshmemx_putmem_on_stream");
}
void nvshmemx_putmem_nbi_on_stream(void *, const void *, size_t, int,
                                   cudaStream_t) {
  report("nvshmemx_putmem_nbi_on_stream");
}
void nvshmemx_getmem_on_stream(void *, const void *, size_t, int,
                               cudaStream_t) {
  report("nvshmemx_getmem_on_stream");
}
void nvshmemx_getmem_nbi_on_stream(void *, const void *, size_t, int,
                                   cudaStream_t) {
  report("nvshmemx_getmem_nbi_on_stream");
}
void nvshmemx_putmem_signal_on_stream(void *, const void *, size_t, uint64_t *,
                                      uint64_t, int, int, cudaStream_t) {
  report("nvshmemx_putmem_signal_on_stream");
}
void nvshmemx_putmem_signal_nbi_on_stream(void *, const void *, size_t,
                                          uint64_t *, uint64_t, int, int,
                                          cudaStream_t) {
  report("nvshmemx_putmem_signal_nbi_on_stream");
}
void nvshmemx_quiet_on_stream(cudaStream_t) {
  report("nvshmemx_quiet_on_stream");
}
void nvshmemx_flush_on_stream(cudaStream_t) {
  report("nvshmemx_flush_on_stream");
}
void nvshmemx_signal_op_on_stream(uint64_t *, uint64_t, int, int,
                                  cudaStream_t) {
  report("nvshmemx_signal_op_on_stream");
}
void nvshmemx_signal_wait_until_on_stream(uint64_t *, int, uint64_t,
                                          cudaStream_t) {
  report("nvshmemx_signal_wait_until_on_stream");
}

int nvshmemx_alltoallmem_on_stream(nvshmem_team_t, void *, const void *, size_t,
                                   cudaStream_t) {
  return refuse_status("nvshmemx_alltoallmem_on_stream");
}
int nvshmemx_broadcastmem_on_stream(nvshmem_team_t, void *, const void *,
                                    size_t, int, cudaStream_t) {
  return refuse_status("nvshmemx_broadcastmem_on_stream");
}
int nvshmemx_fcollectmem_on_stream(nvshmem_team_t, void *, const void *, size_t,
                                   cudaStream_t) {
  return refuse_status("nvshmemx_fcollectmem_on_stream");
}
int nvshmemx_barrier_on_stream(nvshmem_team_t, cudaStream_t) {
  return refuse_status("nvshmemx_barrier_on_stream");
}
void nvshmemx_barrier_all_on_stream(cudaStream_t) {
  report("nvshmemx_barrier_all_on_stream");
}
int nvshmemx_team_sync_on_stream(nvshmem_team_t, cudaStream_t) {
  return refuse_status("nvshmemx_team_sync_on_stream");
}
void nvshmemx_sync_all_on_stream(cudaStream_t) {
  report("nvshmemx_sync_all_on_stream");
}
