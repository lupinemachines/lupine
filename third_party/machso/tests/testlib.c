#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include <ctype.h>
#include <fcntl.h>
#include <unistd.h>

/* exported data, including pointers that need rebasing */
int counter = 42;
const char *greeting = "hello from linux";
static int table[] = {1, 2, 3, 4, 5};
int *table_ptr = &table[2];

static int ctor_ran = 0;
__attribute__((constructor)) static void init_me(void) { ctor_ran = 1234; }

int get_ctor_ran(void) { return ctor_ran; }

int add(int a, int b) { return a + b; }

static int sq(int x) { return x * x; }
typedef int (*fn_t)(int);
static fn_t fns[] = {sq, abs};          /* local + imported fn pointers in data */
int apply(int which, int x) { return fns[which](x); }

int bump(void) { return ++counter; }
int deref_table(void) { return *table_ptr; }

char *dup_upper(const char *s) {
    char *r = strdup(s);
    for (char *p = r; *p; p++) *p = toupper((unsigned char)*p);
    return r;
}

int count_alpha(const char *s) {
    int n = 0;
    for (; *s; s++) if (isalpha((unsigned char)*s)) n++;
    return n;
}

static int cmp(const void *a, const void *b) { return *(const int *)a - *(const int *)b; }
void sort_ints(int *v, int n) { qsort(v, n, sizeof(int), cmp); }

/* variadic call into libc: the arm64 ABI hazard */
int fmt(char *buf, size_t n, int i, double d, const char *s) {
    return snprintf(buf, n, "i=%d d=%.3f s=%s l=%ld", i, d, s, (long)i * 1000000000L);
}

void say(const char *s) { printf("say: %s\n", s); fprintf(stderr, "err: %s\n", s); fflush(stdout); }

int bad_open(void) {
    int fd = open("/nonexistent/xyz", O_RDONLY);
    return fd < 0 ? errno : 0;
}

int make_file(const char *path) {
    int fd = open(path, O_CREAT | O_WRONLY | O_TRUNC, 0644);
    if (fd < 0) return -1;
    write(fd, "abc", 3);
    close(fd);
    return 0;
}

/* buffer to exercise stack protector */
int stack_user(const char *s) {
    char buf[64];
    strncpy(buf, s, sizeof buf - 1);
    buf[63] = 0;
    return (int)strlen(buf);
}
