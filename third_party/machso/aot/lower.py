"""Generate a separate native function for each statically identified syscall."""
from pathlib import Path

# These expressions are compiled into each call site target, not a runtime dispatcher.
OPERATIONS = {
    23: 'aot_result(dup((int)a))',
    49: 'aot_result(chdir((const char *)a))',
    50: 'aot_result(fchdir((int)a))',
    56: 'aot_openat(a,b,c,d)',
    57: 'aot_result(close((int)a))',
    62: 'aot_result(lseek((int)a,b,(int)c))',
    63: 'aot_result(read((int)a,(void *)b,(size_t)c))',
    64: 'aot_result(write((int)a,(const void *)b,(size_t)c))',
    65: 'aot_result(readv((int)a,(const struct iovec *)b,(int)c))',
    66: 'aot_result(writev((int)a,(const struct iovec *)b,(int)c))',
    67: 'aot_result(pread((int)a,(void *)b,(size_t)c,d))',
    68: 'aot_result(pwrite((int)a,(const void *)b,(size_t)c,d))',
    81: '(sync(),0)',
    82: 'aot_result(fsync((int)a))',
    83: 'aot_result(fsync((int)a))',
    93: '(_exit((int)a),0)',
    94: '(_exit((int)a),0)',
    98: 'aot_futex(a,b,c,d,e,f)',
    172: 'getpid()', 173: 'getppid()', 174: 'getuid()', 175: 'geteuid()',
    176: 'getgid()', 177: 'getegid()',
    178: '*(uint32_t *)((unsigned char *)aot_thread_pointer()-0x670)',
    214: 'aot_brk(a)',
    215: 'aot_result(munmap((void *)a,(size_t)b))',
    222: 'aot_mmap(a,b,c,d,e,f)',
    226: '(c & ~7L) ? -22 : aot_result(mprotect((void *)a,(size_t)b,(int)c))',
    113: 'aot_clock_gettime(a,b)',
    169: 'aot_result(gettimeofday((struct timeval *)a,(void *)b))',
    261: 'aot_prlimit(a,b,c,d)',
    278: '(c & ~3L) ? -22 : (arc4random_buf((void *)a,(size_t)b),b)',
}

ARITIES = {23:1,49:1,50:1,56:4,57:1,62:3,63:3,64:3,65:3,66:3,67:4,68:4,
           81:0,82:1,83:1,93:1,94:1,98:6,113:2,169:2,172:0,173:0,174:0,175:0,
           176:0,177:0,178:0,214:1,215:2,222:6,226:3,261:4,278:3}


def emit_native(directory, manifest, tunables=None):
    template = Path(__file__).with_name('native.c').read_text()
    template = template.replace('/* AOT_FUTEX_CODE */', Path(__file__).with_name('native_futex.c').read_text())
    if tunables is None:
        tunable_code = 'void aot_import___tunable_get_val(int id,void *value,void *callback) {(void)id;(void)value;(void)callback;__builtin_trap();}'
    else:
        from aot.tunables import native_code
        tunable_code = native_code(tunables)
        manifest['loader_tunables'] = tunables
    template = template.replace('/* AOT_TUNABLE_CODE */', tunable_code)
    pieces = [template]
    for number in sorted(set(manifest['numbers']) | OPERATIONS.keys()):
        expression = OPERATIONS.get(number, '-38')
        pieces.append(f'long aot_nr_{number}(long a,long b,long c,long d,long e,long f) {{\n'
                      '    (void)a;(void)b;(void)c;(void)d;(void)e;(void)f;\n'
                      f'    return {expression};\n}}\n')
    pieces.append('long aot_dynamic_syscall(long a,long b,long c,long d,long e,long f,long number) {\n'
                  '    switch (number) {\n'+''.join(
                      f'        case {number}: return aot_nr_{number}(a,b,c,d,e,f);\n'
                      for number in sorted(OPERATIONS))+'        default: return -38;\n    }\n}\n')
    pieces.append('extern int *g___errno_location(void);\n'
                  'PUBLIC long g_syscall(long number,...) {\n    int count;\n    switch (number) {\n'+''.join(
                      f'        case {number}: count={ARITIES[number]}; break;\n'
                      for number in sorted(OPERATIONS))+
                  '        default: *g___errno_location()=38; return -1;\n    }\n'
                  '    long values[6]={0}; va_list args; va_start(args,number);\n'
                  '    for (int i=0;i<count;i++) values[i]=va_arg(args,long);\n    va_end(args);\n'
                  '    long result=aot_dynamic_syscall(values[0],values[1],values[2],values[3],values[4],values[5],number);\n'
                  '    if (result<0 && result>=-4095) {*g___errno_location()=(int)-result;return -1;}\n'
                  '    return result;\n}\n')
    Path(directory, 'native.c').write_text('\n'.join(pieces))
    manifest['lowered_syscalls'] = sorted(set(manifest['numbers']) & OPERATIONS.keys())
    manifest['unsupported_syscalls'] = sorted(set(manifest['numbers']) - OPERATIONS.keys())
    manifest['dynamic_syscall_lowering'] = 'native compiled dispatch; unsupported numbers return ENOSYS'
    manifest['unsupported_loader_imports'] = ['_dl_signal_error','_dl_signal_exception','_dl_audit_symbind_alt',
                                             '_dl_deallocate_tls','_dl_allocate_tls','_dl_allocate_tls_init',
                                             '_dl_rtld_di_serinfo','__nptl_change_stack_perm']
    return manifest
