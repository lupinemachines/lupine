#include <unistd.h>
#include <errno.h>
#include <sys/syscall.h>
#include <sys/mman.h>
#include <fcntl.h>
extern int middle_add(int);
extern int leaf_value;
int graph_result(void) { return middle_add(0) + leaf_value; }
int kernel_checks(const char *path) {
    if (syscall(SYS_getpid) != getpid()) return 1;
    long fd = syscall(SYS_openat, AT_FDCWD, path, O_CREAT | O_WRONLY | O_TRUNC, 0600);
    if (fd < 0) return 2;
    if (syscall(SYS_write, fd, "mapped", 6) != 6) return 3;
    if (syscall(SYS_close, fd)) return 4;
    fd = syscall(SYS_openat, AT_FDCWD, path, O_RDONLY, 0);
    if (fd < 0) return 5;
    char buf[6];
    if (syscall(SYS_read, fd, buf, 6) != 6 || buf[0] != 'm' || buf[5] != 'd') return 6;
    if (syscall(SYS_lseek, fd, 0, SEEK_SET) != 0) return 7;
    if (syscall(SYS_close, fd)) return 8;
    char *p = (char *)syscall(SYS_mmap, 0, 16384, PROT_READ | PROT_WRITE,
                            MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (p == MAP_FAILED) return 9;
    p[0] = 123;
    if (syscall(SYS_mprotect, p, 16384, PROT_READ) || p[0] != 123) return 10;
    if (syscall(SYS_munmap, p, 16384)) return 11;
    if (syscall(SYS_openat, AT_FDCWD, "/nonexistent/machso", O_RDONLY, 0) != -1 || errno != 2) return 12;
    if (syscall(99999) != -1 || errno != 38) return 13;
    /* EBADF differs numerically from Darwin ENOSYS; test Linux error contract. */
    if (syscall(SYS_close, -1) != -1 || errno != 9) return 14;
    return 0;
}
