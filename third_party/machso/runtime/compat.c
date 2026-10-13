/* Selected Linux libc semantics. Compiled for Darwin, never from input source. */
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <errno.h>
#include <fcntl.h>
#include <unistd.h>
#include <stdarg.h>
#include <sys/mman.h>

FILE *elf_stdout, *elf_stderr;
static uint16_t ctype_table[384];
static int32_t upper_table[384];
static const uint16_t *ctype_ptr = ctype_table+128;
static const int32_t *upper_ptr = upper_table+128;
__attribute__((constructor)) static void setup(void) {
    elf_stdout = stdout;
    elf_stderr = stderr;
    for (int c = -128; c < 256; ++c) {
        unsigned char u = (unsigned char)c;
        /* glibc ctype flags are stored in network byte order. ASCII only. */
        uint16_t b = 0;
        if (u < 128) {
            if (isupper(u)) b |= 0x100;
            if (islower(u)) b |= 0x200;
            if (isalpha(u)) b |= 0x400;
            if (isdigit(u)) b |= 0x800;
            if (isxdigit(u)) b |= 0x1000;
            if (isspace(u)) b |= 0x2000;
            if (isprint(u)) b |= 0x4000;
            if (isgraph(u)) b |= 0x8000;
            if (u == ' ' || u == '\t') b |= 1;
            if (iscntrl(u)) b |= 2;
            if (ispunct(u)) b |= 4;
            if (isalnum(u)) b |= 8;
        }
        ctype_table[c+128] = c == -1 ? 0 : b;
        upper_table[c+128] = c == -1 ? -1 : (u < 128 ? toupper(u) : u);
    }
}
const uint16_t **elf___ctype_b_loc(void) { return &ctype_ptr; }
const int32_t **elf___ctype_toupper_loc(void) { return &upper_ptr; }
static _Thread_local int linux_errno;
/* Explicit subset; fail instead of returning a wrong Linux errno number. */
static int linux_error(int e) {
    switch (e) {
        case 0: return 0;
#define E(host, guest) case host: return guest;
        E(EPERM,1) E(ENOENT,2) E(ESRCH,3) E(EINTR,4) E(EIO,5) E(ENXIO,6)
        E(E2BIG,7) E(ENOEXEC,8) E(EBADF,9) E(ECHILD,10) E(EAGAIN,11)
        E(ENOMEM,12) E(EACCES,13) E(EFAULT,14) E(EBUSY,16) E(EEXIST,17)
        E(EXDEV,18) E(ENODEV,19) E(ENOTDIR,20) E(EISDIR,21) E(EINVAL,22)
        E(ENFILE,23) E(EMFILE,24) E(ENOTTY,25) E(EFBIG,27) E(ENOSPC,28)
        E(ESPIPE,29) E(EROFS,30) E(EMLINK,31) E(EPIPE,32) E(EDOM,33)
        E(ERANGE,34) E(ENAMETOOLONG,36) E(ENOSYS,38) E(ENOTEMPTY,39)
        E(ELOOP,40) E(EOVERFLOW,75)
#undef E
        default: fprintf(stderr,"elfcompat: unsupported Darwin errno %d\n",e); abort();
    }
}
int *elf___errno_location(void) { return &linux_errno; }
static int translate_open_flags(int flags) {
    int supported = 3 | 0100 | 0200 | 01000 | 02000 | 04000 | 0200000 | 0400000 | 02000000;
    if ((flags & ~supported) || (flags & 3) == 3) { errno = EINVAL; return -1; }
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
static long libc_result(long result) {
    if (result == -1) linux_errno = linux_error(errno);
    return result;
}
int elf_open(const char *path, int flags, unsigned mode) {
    int native = translate_open_flags(flags);
    if (native == -1) return (int)libc_result(-1);
    return (int)libc_result(open(path, native, mode));
}
int elf_close(int fd) { return (int)libc_result(close(fd)); }
ssize_t elf_read(int fd, void *buf, size_t count) { return libc_result(read(fd, buf, count)); }
ssize_t elf_write(int fd, const void *buf, size_t count) { return libc_result(write(fd, buf, count)); }

/* Linux AArch64 syscall numbers, never Darwin syscall numbers. Fixed arguments
 * deliberately match the register ABI of Linux's variadic syscall() entry. */
long elf_syscall(long number, long a, long b, long c, long d, long e, long f) {
    long result;
    switch (number) {
        case 56: { /* openat */
            int flags = translate_open_flags((int)c);
            if (flags == -1) return libc_result(-1);
            int dirfd = a == -100 ? AT_FDCWD : (int)a;
            result = openat(dirfd, (const char *)b, flags, (unsigned)d);
            break;
        }
        case 57: result = close((int)a); break;
        case 62: result = lseek((int)a, b, (int)c); break;
        case 63: result = read((int)a, (void *)b, (size_t)c); break;
        case 64: result = write((int)a, (const void *)b, (size_t)c); break;
        case 82: result = fsync((int)a); break;
        case 172: return getpid();
        case 173: return getppid();
        case 174: return getuid();
        case 175: return geteuid();
        case 176: return getgid();
        case 177: return getegid();
        case 215: result = munmap((void *)a, (size_t)b); break;
        case 222: { /* mmap: shared/private and optional anonymous only */
            if ((c & ~7L) || (d & ~0x23L) || ((d & 3) != 1 && (d & 3) != 2)) {
                linux_errno = 22; return -1;
            }
            int flags = (d & 1) ? MAP_SHARED : MAP_PRIVATE;
            if (d & 0x20) flags |= MAP_ANON;
            int prot = ((c & 1) ? PROT_READ : 0) | ((c & 2) ? PROT_WRITE : 0) | ((c & 4) ? PROT_EXEC : 0);
            result = (long)mmap((void *)a, (size_t)b, prot, flags, (int)e, f);
            break;
        }
        case 226: { /* mprotect */
            if (c & ~7L) { linux_errno = 22; return -1; }
            int prot = ((c & 1) ? PROT_READ : 0) | ((c & 2) ? PROT_WRITE : 0) | ((c & 4) ? PROT_EXEC : 0);
            result = mprotect((void *)a, (size_t)b, prot);
            break;
        }
        default: linux_errno = 38; return -1; /* ENOSYS */
    }
    return libc_result(result);
}

/* Entry assembly captures the Linux AAPCS64 integer and FP argument banks. */
struct frame { uint64_t x[8]; uint64_t d[8]; uint64_t *stack; };
static void unsupported_format(void) {
    fputs("elfcompat: unsupported printf format\n", stderr); abort();
}
static uint64_t next_arg(struct frame *f, int *gp, int *fp, int floating) {
    if (floating && *fp < 8) return f->d[(*fp)++];
    if (!floating && *gp < 8) return f->x[(*gp)++];
    return *f->stack++;
}
/* Darwin arm64 va_list is a pointer to consecutive 8-byte vararg slots. */
static void marshal(const char *fmt, struct frame *f, int gp, uint64_t slots[256]) {
    unsigned n = 0;
    int fp = 0;
#define ARG(floating) do { if (n == 256) unsupported_format(); slots[n++] = next_arg(f, &gp, &fp, floating); } while (0)
    for (const char *p = fmt; *p; ++p) {
        if (*p != '%') continue;
        ++p;
        if (*p == '%') continue;
        while (*p && strchr("-+ #0'", *p)) ++p;
        if (*p == '*') { ARG(0); ++p; }
        else while (isdigit((unsigned char)*p)) ++p;
        if (*p == '.') {
            ++p;
            if (*p == '*') { ARG(0); ++p; }
            else while (isdigit((unsigned char)*p)) ++p;
        }
        char length = 0;
        if (*p && strchr("hljztL", *p)) {
            length = *p++;
            if (*p == length && (length == 'h' || length == 'l')) ++p;
        }
        if (!*p || *p == '$' || length == 'L') unsupported_format();
        if (strchr("aAeEfFgG", *p)) ARG(1);
        else if (strchr("diouxXcsp", *p) && !(length == 'l' && (*p == 's' || *p == 'c'))) ARG(0);
        else unsupported_format();
    }
#undef ARG
}
/* kind: snprintf, printf, fprintf, and their glibc fortify variants. */
int elf_format(struct frame *f, int kind) {
    int gp, fi;
    switch (kind) {
        case 0: gp = 3; fi = 2; break;
        case 1: gp = 1; fi = 0; break;
        case 2: gp = 2; fi = 1; break;
        case 3: gp = 5; fi = 4; break;
        case 4: gp = 2; fi = 1; break;
        case 5: gp = 3; fi = 2; break;
        default: abort();
    }
    const char *fmt = (const char *)f->x[fi];
    uint64_t slots[256] = {0};
    marshal(fmt, f, gp, slots);
    va_list ap = (char *)slots;
    if (kind == 0 || kind == 3) {
        if (kind == 3 && f->x[1] > f->x[3]) abort();
        return vsnprintf((char *)f->x[0], f->x[1], fmt, ap);
    }
    return vfprintf(kind == 2 || kind == 5 ? (FILE *)f->x[0] : stdout, fmt, ap);
}
