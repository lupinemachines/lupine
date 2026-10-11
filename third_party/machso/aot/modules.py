"""Generate module-scoped symbol lookup for late-loaded converted libraries."""
from collections import deque
import json
from aot.graph import exported


def emit_modules(nodes, plans, native_providers=()):
    nodes = [*nodes, *native_providers]
    indexes = {node.index: i for i, node in enumerate(nodes)}
    lines = ['#include <dlfcn.h>', '#include <string.h>',
             'struct aot_symbol {const char *name,*version; const void *address; uintptr_t tls; int hidden,is_tls,unsupported;};',
             'struct aot_module {const char *name,*path; const struct aot_symbol *symbols; size_t count; '
             'const unsigned *scope; size_t scope_count; void *native;};']
    rows = []
    for node in nodes:
        definitions = [(i, s) for i, s in enumerate(node.elf.symbols) if exported(s)]
        entries = []
        for i, symbol in sorted(definitions, key=lambda item: (item[1]['name'], item[1]['version'] or '')):
            tls = (symbol['info'] & 15 == 6 or
                   bool(node.native and symbol['name'] in node.native.get('tls_exports', [])))
            unsupported = bool(node.native and symbol['name'] in
                               node.native.get('signature_report', {}).get('unsupported', {}))
            signature = node.native.get('signatures', {}).get(symbol['name']) if node.native else None
            if unsupported or node.native and not signature:
                address, offset = 'NULL', 0
            elif tls:
                address = 'NULL'
                offset = plans[node.index].tls_offset+symbol['value']
            else:
                alias = f'aot_symbol_{node.identifier}_{i}'
                target = node.symbol_name(i)
                if node.native:
                    target = f'_aot_native_{node.identifier}_s{i}'
                lines.append(f'extern const unsigned char {alias}[] __asm__({json.dumps(target)});')
                address, offset = alias, 0
            entries.append('{'+','.join((json.dumps(symbol['name']),
                                          json.dumps(symbol['version']) if symbol['version'] else 'NULL',
                                          address, str(offset), str(int(symbol['version_hidden'])), str(int(tls)),
                                          str(int(unsupported))))+'}')
        table = 'aot_symbols_'+node.identifier
        # A sentinel keeps an empty module valid in strict C; count excludes it.
        lines.append(f'static const struct aot_symbol {table}[]={{'+','.join(entries or ['{NULL,NULL,NULL,0,0,0,0}'])+'};')
        queue, seen, scope = deque([node]), set(), []
        while queue:
            member = queue.popleft()
            if member.index in seen:
                continue
            seen.add(member.index)
            if member.index in indexes:
                scope.append(indexes[member.index])
            queue.extend(member.dependencies)
        # Retain the converted C++ ABI domain before native implementations.
        scope = [i for i in scope if not nodes[i].native]+[i for i in scope if nodes[i].native]
        scope_name = 'aot_scope_'+node.identifier
        lines.append(f'static const unsigned {scope_name}[]={{'+','.join(map(str, scope))+'};')
        path = json.dumps(node.native['native_install_name']) if node.native else 'NULL'
        rows.append('{'+','.join((json.dumps(node.name), path, table, str(len(entries)),
                                 scope_name, str(len(scope)), 'NULL'))+'}')
    lines.append('static struct aot_module aot_modules[]={'+','.join(rows)+'};')
    lines.append('''
#define AOT_MODULE_LOOKUP 1
static _Thread_local const char *aot_module_error;
static struct aot_module *aot_module_handle(void *handle) {
    for (size_t i=0;i<sizeof(aot_modules)/sizeof(aot_modules[0]);i++)
        if (handle==&aot_modules[i]) return &aot_modules[i];
    return NULL;
}
static void *aot_module_open(const char *name,int flags) {
    for (size_t i=0;i<sizeof(aot_modules)/sizeof(aot_modules[0]);i++) {
        struct aot_module *module=&aot_modules[i];
        if (strcmp(name,module->name)) continue;
        const char *path=module->path;
        Dl_info self;
        if (!path) {
            if (!dladdr((void *)&aot_module_open,&self)) return NULL;
            path=self.dli_fname;
        }
        void *handle=dlopen(path,flags);
        if (!handle) return NULL;
        __atomic_store_n(&module->native,handle,__ATOMIC_RELEASE);
        return module;
    }
    return NULL;
}
static void *aot_module_symbol(struct aot_module *module,const char *name,const char *version) {
    aot_module_error=NULL;
    (void)dlerror();
    for (size_t i=0;i<module->scope_count;i++) {
        const struct aot_module *owner=&aot_modules[module->scope[i]];
        size_t low=0,high=owner->count;
        while (low<high) {
            size_t middle=low+(high-low)/2;
            if (strcmp(owner->symbols[middle].name,name)<0) low=middle+1;
            else high=middle;
        }
        for (;low<owner->count && !strcmp(owner->symbols[low].name,name);low++) {
            const struct aot_symbol *symbol=&owner->symbols[low];
            if (version ? (!symbol->version || strcmp(version,symbol->version)) : symbol->hidden) continue;
            if (symbol->unsupported) {
                aot_module_error="native symbol has an unsupported ABI signature";
                return NULL;
            }
            if (owner->path && !symbol->address) {
                void *handle=dlopen(owner->path,RTLD_LAZY|RTLD_LOCAL);
                if (!handle) return NULL;
                void *address=dlsym(handle,name);
                const char *error=dlerror();
                dlclose(handle);
                if (error) aot_module_error="native symbol not found";
                if (symbol->is_tls && address && !error) {
                    // Darwin's dlsym returns the exported TLV descriptor.
                    // Invoke its bootstrap to obtain this thread's value.
                    struct aot_native_tlv {
                        void *(*address)(struct aot_native_tlv *);
                        uintptr_t key,offset;
                    };
                    struct aot_native_tlv *descriptor=address;
                    return descriptor->address(descriptor);
                }
                return address;
            }
            if (symbol->is_tls) return (unsigned char *)aot_thread_pointer()+symbol->tls;
            return (void *)symbol->address;
        }
    }
    aot_module_error="symbol or version not found in converted module scope";
    return NULL;
}
''')
    return '\n'.join(lines)+'\n'
