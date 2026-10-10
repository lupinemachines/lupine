/* Native code linked into the AOT product. No ELF loader or separate runtime. */
#include <stdint.h>
#include <stddef.h>
#include <unistd.h>
#include <errno.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/resource.h>
#include <sys/uio.h>
#include <sys/time.h>
#include <time.h>
#include <pthread.h>
#include <stdlib.h>
#include <crt_externs.h>

struct tlv { void *(*address)(struct tlv *); uintptr_t key, offset; };
extern struct tlv aot_tls_descriptor;
extern void g___ctype_init(void);
extern unsigned char g___libc_single_threaded;
extern unsigned char g___environ[];
extern void aot_original_init_0(int, char **, char **);
extern void aot_original_init_1(int, char **, char **);
extern void aot_original_init_2(int, char **, char **);
static _Thread_local int aot_thread_initialized;

uintptr_t aot_import___stack_chk_guard = 0x51a7603952b34800;
uintptr_t aot_import___pointer_chk_guard = 0x91c57380ea2764bc;
unsigned char aot_import__rtld_global_ro[0x10000];
unsigned char aot_import__rtld_global[0x10000];
char **aot_import__dl_argv;
int aot_import___libc_enable_secure;
void *aot_import___libc_stack_end;
static uintptr_t auxiliary[] = {6, 16384, 16, 0, 26, 0, 0, 0};

void *aot_thread_pointer(void) {
    unsigned char *tp = (unsigned char *)aot_tls_descriptor.address(&aot_tls_descriptor)+0x1000;
    if (!aot_thread_initialized) {
        aot_thread_initialized = 1;
        uint64_t tid;
        if (pthread_main_np()) tid=(uint64_t)getpid();
        else {
            pthread_threadid_np(NULL,&tid);
            __atomic_store_n(&g___libc_single_threaded,0,__ATOMIC_RELAXED);
        }
        /* This profile's pthread descriptor is 0x740 bytes before TP; tid is
         * its 32-bit member at offset 0xd0. Recursive mutexes need a nonzero ID. */
        *(uint32_t *)(tp-0x670)=(uint32_t)(tid & 0x3fffffff);
        /* The original glibc initializer fills its own ctype TLS slots. */
        g___ctype_init();
    }
    return tp;
}
void *aot_import___tls_get_addr(const uintptr_t *index) {
    if (index[0] != 1) __builtin_trap();
    return (unsigned char *)aot_thread_pointer()+16+index[1];
}
/* AOT_TUNABLE_CODE */
int aot_import___tunable_is_initialized(void) { return 0; }
void aot_import__dl_audit_preinit(void) {}
void *aot_import__dl_find_dso_for_object(void *address) { (void)address; return NULL; }
int aot_import__dl_catch_exception(void *exception, void (*operation)(void *), void *argument) {
    (void)exception; operation(argument); return 0;
}
void aot_import__dl_signal_error(void) { __builtin_trap(); }
void aot_import__dl_signal_exception(void) { __builtin_trap(); }
void aot_import__dl_audit_symbind_alt(void) { __builtin_trap(); }
void aot_import__dl_deallocate_tls(void) { __builtin_trap(); }
void aot_import__dl_allocate_tls(void) { __builtin_trap(); }
void aot_import__dl_allocate_tls_init(void) { __builtin_trap(); }
void aot_import__dl_rtld_di_serinfo(void) { __builtin_trap(); }
void aot_import___nptl_change_stack_perm(void) { __builtin_trap(); }

__attribute__((constructor)) static void aot_initialize(void) {
    *(uintptr_t *)(aot_import__rtld_global_ro+24) = 16384;
    *(uintptr_t *)(aot_import__rtld_global_ro+104) = (uintptr_t)auxiliary;
    *(uint32_t *)(aot_import__rtld_global_ro+120) = 64;
    aot_import__dl_argv = *_NSGetArgv();
    *(char ***)g___environ = *_NSGetEnviron();
    g___libc_single_threaded = 1;
    aot_thread_pointer();
    int argc = *_NSGetArgc();
    char **argv = *_NSGetArgv(), **env = *_NSGetEnviron();
    aot_original_init_0(argc, argv, env);
    aot_original_init_1(argc, argv, env);
    aot_original_init_2(argc, argv, env);
}

static int aot_linux_errno(int error) {
    switch (error) {
#define E(host, linux) case host: return linux;
        E(EPERM,1) E(ENOENT,2) E(ESRCH,3) E(EINTR,4) E(EIO,5) E(ENXIO,6)
        E(E2BIG,7) E(ENOEXEC,8) E(EBADF,9) E(ECHILD,10) E(EAGAIN,11)
        E(ENOMEM,12) E(EACCES,13) E(EFAULT,14) E(EBUSY,16) E(EEXIST,17)
        E(EXDEV,18) E(ENODEV,19) E(ENOTDIR,20) E(EISDIR,21) E(EINVAL,22)
        E(ENFILE,23) E(EMFILE,24) E(ENOTTY,25) E(EFBIG,27) E(ENOSPC,28)
        E(ESPIPE,29) E(EROFS,30) E(EMLINK,31) E(EPIPE,32) E(EDOM,33)
        E(ERANGE,34) E(EDEADLK,35) E(ENAMETOOLONG,36) E(ENOLCK,37)
        E(ENOSYS,38) E(ENOTEMPTY,39) E(ELOOP,40) E(EOVERFLOW,75)
        E(ENOTSOCK,88) E(EDESTADDRREQ,89) E(EMSGSIZE,90) E(EPROTOTYPE,91)
        E(ENOPROTOOPT,92) E(EPROTONOSUPPORT,93) E(EOPNOTSUPP,95)
        E(EAFNOSUPPORT,97) E(EADDRINUSE,98) E(EADDRNOTAVAIL,99)
        E(ENETDOWN,100) E(ENETUNREACH,101) E(ECONNABORTED,103)
        E(ECONNRESET,104) E(ENOBUFS,105) E(EISCONN,106) E(ENOTCONN,107)
        E(ETIMEDOUT,110) E(ECONNREFUSED,111) E(EHOSTUNREACH,113)
        E(EALREADY,114) E(EINPROGRESS,115) E(ECANCELED,125)
#undef E
        default: return 38;
    }
}
static long aot_result(long result) { return result == -1 ? -aot_linux_errno(errno) : result; }
static int aot_open_flags(int flags) {
    const int supported = 3|0100|0200|01000|02000|04000|0100000|0200000|0400000|02000000;
    if ((flags & ~supported) || (flags & 3) == 3) return -1;
    int native = flags & 3;
    if (flags & 0100) native |= O_CREAT;
    if (flags & 0200) native |= O_EXCL;
    if (flags & 01000) native |= O_TRUNC;
    if (flags & 02000) native |= O_APPEND;
    if (flags & 04000) native |= O_NONBLOCK;
    if (flags & 0200000) native |= O_DIRECTORY;
    if (flags & 0400000) native |= O_NOFOLLOW;
    if (flags & 02000000) native |= O_CLOEXEC;
    return native;
}
static long aot_openat(long dir, long path, long flags, long mode) {
    int native = aot_open_flags((int)flags);
    if (native < 0) return -22;
    return aot_result(openat(dir == -100 ? AT_FDCWD : (int)dir, (const char *)path, native, (unsigned)mode));
}
static long aot_mmap(long address, long size, long protection, long flags, long fd, long offset) {
    const long supported = 3|0x10|0x20|0x1000|0x4000|0x8000|0x10000|0x20000;
    if ((flags & ~supported) || (protection & ~7L) || ((flags & 3) != 1 && (flags & 3) != 2)) return -22;
    int native = (flags & 1) ? MAP_SHARED : MAP_PRIVATE;
    if (flags & 0x10) native |= MAP_FIXED;
    if (flags & 0x20) native |= MAP_ANON;
    return aot_result((long)mmap((void *)address, (size_t)size, (int)protection, native, (int)fd, offset));
}
static pthread_mutex_t brk_lock = PTHREAD_MUTEX_INITIALIZER;
static unsigned char *brk_base, *brk_top;
static long aot_brk(long requested) {
    pthread_mutex_lock(&brk_lock);
    if (!brk_base) {
        void *region = mmap(NULL, 128*1024*1024, PROT_NONE, MAP_PRIVATE|MAP_ANON, -1, 0);
        if (region == MAP_FAILED) { pthread_mutex_unlock(&brk_lock); return 0; }
        brk_base = brk_top = region;
    }
    if (requested && (uintptr_t)requested >= (uintptr_t)brk_base && (uintptr_t)requested <= (uintptr_t)brk_base+128*1024*1024) {
        size_t length = ((uintptr_t)requested-(uintptr_t)brk_base+16383)&~(size_t)16383;
        if (!length || !mprotect(brk_base, length, PROT_READ|PROT_WRITE)) brk_top = (void *)requested;
    }
    long result = (long)brk_top;
    pthread_mutex_unlock(&brk_lock);
    return result;
}
struct linux_timespec { int64_t seconds, nanoseconds; };
static long aot_clock_gettime(long id, long out) {
    clockid_t native;
    switch (id) { case 0: case 5: native=CLOCK_REALTIME; break; case 1: case 6: native=CLOCK_MONOTONIC; break;
        case 2: native=CLOCK_PROCESS_CPUTIME_ID; break; case 3: native=CLOCK_THREAD_CPUTIME_ID; break;
        case 4: native=CLOCK_MONOTONIC_RAW; break; default: return -22; }
    struct timespec ts;
    if (clock_gettime(native,&ts)) return -aot_linux_errno(errno);
    struct linux_timespec *result = (void *)out;
    result->seconds=ts.tv_sec; result->nanoseconds=ts.tv_nsec;
    return 0;
}
static long aot_prlimit(long pid, long resource, long newer, long older) {
    if (pid && pid != getpid()) return -38;
    int native;
    switch (resource) {
        case 0: native=RLIMIT_CPU; break; case 1: native=RLIMIT_FSIZE; break;
        case 2: native=RLIMIT_DATA; break; case 3: native=RLIMIT_STACK; break;
        case 4: native=RLIMIT_CORE; break; case 5: native=RLIMIT_RSS; break;
        case 6: native=RLIMIT_NPROC; break; case 7: native=RLIMIT_NOFILE; break;
        case 8: native=RLIMIT_MEMLOCK; break; case 9: native=RLIMIT_AS; break;
        default: return -22;
    }
    struct rlimit limits;
    if (older) {
        if (getrlimit(native,&limits)) return -aot_linux_errno(errno);
        uint64_t *values=(void *)older;
        values[0]=limits.rlim_cur==RLIM_INFINITY ? UINT64_MAX : limits.rlim_cur;
        values[1]=limits.rlim_max==RLIM_INFINITY ? UINT64_MAX : limits.rlim_max;
    }
    if (newer) {
        const uint64_t *values=(void *)newer;
        limits.rlim_cur=values[0]==UINT64_MAX ? RLIM_INFINITY : values[0];
        limits.rlim_max=values[1]==UINT64_MAX ? RLIM_INFINITY : values[1];
        if (setrlimit(native,&limits)) return -aot_linux_errno(errno);
    }
    return 0;
}
long aot_unresolved_syscall(void) { return -38; }

/* AOT_FUTEX_CODE */

/* Native callers put arm64 varargs on the stack. The original glibc va_arg
 * implementation can consume those scalar slots through its overflow area. */
#include <stdarg.h>
struct linux_va_list { void *stack, *gr_top, *vr_top; int gr_offs, vr_offs; };
extern int aot_linux_vsnprintf(char *,size_t,const char *,struct linux_va_list *);
extern int aot_linux_vfprintf(void *,const char *,struct linux_va_list *);
extern void *g_stdout;
#define PUBLIC __attribute__((visibility("default")))
PUBLIC int g_vsnprintf(char *buffer,size_t size,const char *format,va_list args) {
    struct linux_va_list ap = {(void *)args,NULL,NULL,0,0};
    return aot_linux_vsnprintf(buffer,size,format,&ap);
}
PUBLIC int g_snprintf(char *buffer,size_t size,const char *format,...) {
    va_list args; va_start(args,format);
    int result=g_vsnprintf(buffer,size,format,args); va_end(args); return result;
}
PUBLIC int g_vfprintf(void *file,const char *format,va_list args) {
    struct linux_va_list ap = {(void *)args,NULL,NULL,0,0};
    return aot_linux_vfprintf(file,format,&ap);
}
PUBLIC int g_fprintf(void *file,const char *format,...) {
    va_list args; va_start(args,format);
    int result=g_vfprintf(file,format,args); va_end(args); return result;
}
PUBLIC int g_vprintf(const char *format,va_list args) { return g_vfprintf(g_stdout,format,args); }
PUBLIC int g_printf(const char *format,...) {
    va_list args; va_start(args,format);
    int result=g_vprintf(format,args); va_end(args); return result;
}
PUBLIC int g___snprintf_chk(char *buffer,size_t size,int flag,size_t maximum,const char *format,...) {
    (void)flag; if (size>maximum) __builtin_trap();
    va_list args; va_start(args,format);
    int result=g_vsnprintf(buffer,size,format,args); va_end(args); return result;
}
PUBLIC int g___printf_chk(int flag,const char *format,...) {
    (void)flag; va_list args; va_start(args,format);
    int result=g_vprintf(format,args); va_end(args); return result;
}
PUBLIC int g___fprintf_chk(void *file,int flag,const char *format,...) {
    (void)flag; va_list args; va_start(args,format);
    int result=g_vfprintf(file,format,args); va_end(args); return result;
}
