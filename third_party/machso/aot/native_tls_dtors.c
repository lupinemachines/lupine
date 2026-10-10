/* Lower the C++ TLS lifetime ABI to Darwin pthread keys. The ELF closure is
 * fixed for this dylib's lifetime, so no Linux link_map refcounts are needed.
 * Match glibc's reverse registration order, including destructor registrations
 * made by another destructor. */
#include <pthread.h>
#include <stdlib.h>
#ifdef __APPLE__
#include <dlfcn.h>
#endif
struct aot_tls_dtor {void (*function)(void *); void *object; struct aot_tls_dtor *next;};
static pthread_key_t aot_tls_dtor_key;
static pthread_once_t aot_tls_dtor_once = PTHREAD_ONCE_INIT;
static int aot_tls_dtor_error;
static void aot_tls_dtors_run(void *head) {
    struct aot_tls_dtor *item = head;
    while (item) {
        pthread_setspecific(aot_tls_dtor_key, item->next);
        item->function(item->object);
        free(item);
        item = pthread_getspecific(aot_tls_dtor_key);
    }
}
static void aot_tls_dtors_initialize(void) {
#ifdef __APPLE__
    // A pending thread destructor must keep its ELF function alive even if a
    // caller closes the original dylib handle on another thread.
    Dl_info self;
    if (!dladdr((void *)&aot_tls_dtors_initialize, &self) ||
        !dlopen(self.dli_fname, RTLD_LAZY | RTLD_LOCAL | RTLD_NODELETE)) {
        aot_tls_dtor_error = 1;
        return;
    }
#endif
    aot_tls_dtor_error = pthread_key_create(&aot_tls_dtor_key, aot_tls_dtors_run);
}
int aot_linux___cxa_thread_atexit_impl(void (*function)(void *), void *object, void *dso) {
    (void)dso;
    pthread_once(&aot_tls_dtor_once, aot_tls_dtors_initialize);
    if (aot_tls_dtor_error) return -1;
    struct aot_tls_dtor *item = malloc(sizeof(*item));
    if (!item) return -1;
    *item = (struct aot_tls_dtor){function, object, pthread_getspecific(aot_tls_dtor_key)};
    if (pthread_setspecific(aot_tls_dtor_key, item)) {free(item); return -1;}
    return 0;
}
void aot_linux___call_tls_dtors(void) {
    pthread_once(&aot_tls_dtor_once, aot_tls_dtors_initialize);
    if (!aot_tls_dtor_error) aot_tls_dtors_run(pthread_getspecific(aot_tls_dtor_key));
}
