/* Static lowering of Linux loader entry points to native Darwin entry points. */
#include <dlfcn.h>
#include <mach-o/dyld.h>
#include <stdint.h>
#include <string.h>
extern int *g___errno_location(void);
static int aot_dlopen_flags(int flags) {
    const int supported=1|2|4|8|0x100|0x1000;
    if ((flags & ~supported) || ((flags & 3)!=1 && (flags & 3)!=2)) return -1;
    int native=(flags & 2) ? RTLD_NOW : RTLD_LAZY;
    native |= (flags & 0x100) ? RTLD_GLOBAL : RTLD_LOCAL;
    if (flags & 4) native |= RTLD_NOLOAD;
    if (flags & 0x1000) native |= RTLD_NODELETE;
    return native;
}
void *aot_linux_dlopen(const char *path,int flags) {
    int native=aot_dlopen_flags(flags);
    if (native<0) {*g___errno_location()=22;return NULL;}
    void *handle=dlopen(path,native);
    if (handle || !path || strchr(path,'/')) return handle;
    /* An explicitly preloaded native provider may have a Darwin filename
     * while its Linux caller requests a SONAME such as libfoo.so.1. */
    const char *suffix=strstr(path,".so");
    if (!suffix || (suffix[3] && suffix[3]!='.')) return NULL;
    for (const char *version=suffix+3; *version; version++)
        if (*version!='.' && (*version<'0' || *version>'9')) return NULL;
    size_t stem=(size_t)(suffix-path);
    for (uint32_t i=0;i<_dyld_image_count();i++) {
        const char *image=_dyld_get_image_name(i);
        if (!image) continue;
        const char *name=strrchr(image,'/');
        name=name ? name+1 : image;
        if (strlen(name)==stem+6 && !strncmp(name,path,stem) && !strcmp(name+stem,".dylib"))
            return dlopen(image,native);
    }
    return NULL;
}
void *aot_linux_dlmopen(long namespace,const char *path,int flags) {
    if (namespace) {*g___errno_location()=38;return NULL;}
    return aot_linux_dlopen(path,flags);
}
void *aot_linux_dlsym(void *handle,const char *name) {
    if (!handle) handle=RTLD_DEFAULT;
    else if (handle==(void *)-1) handle=RTLD_NEXT;
    return dlsym(handle,name);
}
void *aot_linux_dlvsym(void *handle,const char *name,const char *version) {
    (void)version;
    return aot_linux_dlsym(handle,name);
}
int aot_linux_dlclose(void *handle) {return dlclose(handle);}
char *aot_linux_dlerror(void) {return dlerror();}
