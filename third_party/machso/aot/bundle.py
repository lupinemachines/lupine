"""Compile a dependency closure as one native dylib using ELF-derived bindings."""
from dataclasses import dataclass, field
import json
from pathlib import Path
from machso import align
from aot.graph import Graph, exported
from aot.emit import emit, ELF_BASE
from aot.lower import emit_native


@dataclass
class Plan:
    identifier: str
    base: int
    segments: list
    core: bool = False
    definitions: dict = field(default_factory=dict)
    public: dict = field(default_factory=dict)
    imports: dict = field(default_factory=dict)
    module_id: int = 0
    tls_offset: int = 16
    tls_padding: int = 0
    first_tls: bool = False
    tls_alignment: int = 16
    tls_symbols: dict = field(default_factory=dict)
    tls_modules: dict = field(default_factory=dict)
    overrides: dict = field(default_factory=dict)


DYNAMIC_LINKER_APIS = {'dlopen', 'dlmopen', 'dlsym', 'dlvsym', 'dlclose', 'dlerror'}


class Bundle:
    def __init__(self, source, library_paths=()):
        self.graph = Graph(source, library_paths)
        if self.graph.root.elf.machine != 183:
            raise ValueError('native bundle backend requires ARM64 ELF; x86-64 needs a different code-generation backend')
        if self.graph.missing:
            names = [f'{n.name}:{s["name"]}@{s["version"]}' for n, s in self.graph.missing]
            raise ValueError('unresolved required symbols: '+', '.join(names))
        self.nodes = [n for n in self.graph.nodes if not n.intrinsic]
        cores = [n for n in self.nodes if any(s['name'] == 'gnu_get_libc_version' and s['shndx'] for s in n.elf.symbols)]
        if len(cores) > 1:
            raise ValueError('multiple libc bootstrap providers in the same bundle')
        self.core = cores[0] if cores else None
        self.link_order = ([self.core] if self.core else [])+[n for n in self.nodes if n is not self.core]
        self.plans = {}
        native_bytes = 0x200000+2*sum(p[5] for n in self.nodes for p in n.elf.loads if p[1] & 1)
        base = max(ELF_BASE, align(native_bytes, 0x400000))
        for node in self.link_order:
            names = [f'__{node.identifier.upper()}L{i}' for i in range(len(node.elf.loads))]
            plan = Plan(node.identifier, base, names, core=node is self.core)
            for index, symbol in enumerate(node.elf.symbols):
                if exported(symbol):
                    plan.definitions[index] = node.symbol_name(index)
                    if plan.core and symbol['name'] in DYNAMIC_LINKER_APIS:
                        plan.overrides[symbol['value']] = '_aot_linux_'+symbol['name']
            self.plans[node.index] = plan
            base += align(max(p[3]+p[6] for p in node.elf.loads), 0x10000)+0x10000
        # Public native names use the group's ELF search order. Versioned imports
        # retain separate internal symbols and bind to their exact definitions.
        self.public = {}
        for node in self.graph.scope(self.graph.root):
            if node.intrinsic:
                continue
            for index, symbol in enumerate(node.elf.symbols):
                if exported(symbol) and not symbol['version_hidden'] and symbol['name'] not in self.public:
                    self.public[symbol['name']] = (node, index)
                    self.plans[node.index].public[index] = symbol['name']
        # A single static TLS image replaces loader-managed dynamic modules.
        self.link_order = ([self.core] if self.core else [])+[n for n in self.nodes if n is not self.core]
        offset = 16
        alignment = max([16]+[n.elf.tls[7] for n in self.nodes if n.elf.tls])
        for i, node in enumerate(self.link_order):
            plan = self.plans[node.index]
            plan.first_tls = i == 0
            plan.tls_alignment = alignment
            if node.elf.tls:
                aligned = align(offset, node.elf.tls[7])
                plan.tls_offset = aligned
                plan.tls_padding = aligned-offset
                plan.module_id = i+1
                offset = aligned+node.elf.tls[6]
        self.tls_size = offset
        for node in self.nodes:
            plan = self.plans[node.index]
            for index, symbol in enumerate(node.elf.symbols):
                if symbol['shndx']:
                    plan.tls_symbols[index] = plan.tls_offset+symbol['value']
                    plan.tls_modules[index] = plan.module_id
                else:
                    provider = self.graph.bindings.get((node.index, index))
                    if provider:
                        owner, definition_index = provider
                        if owner.intrinsic:
                            plan.imports[index] = '_aot_import_'+symbol['name']
                        else:
                            plan.imports[index] = owner.symbol_name(definition_index)
                            definition = owner.elf.symbols[definition_index]
                            provider_plan = self.plans[owner.index]
                            plan.tls_symbols[index] = provider_plan.tls_offset+definition['value']
                            plan.tls_modules[index] = provider_plan.module_id
                    elif symbol['info'] >> 4 == 2:
                        plan.imports[index] = None

    def emit(self, directory):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        components = {}
        for node in self.nodes:
            components[node.index] = emit(node.path, directory/node.identifier, self.plans[node.index])
        numbers = sorted({number for component in components.values() for number in component['numbers']})
        report = self.graph.report()
        report.update({'components': {self.graph.nodes[i].name: component for i, component in components.items()},
                       'numbers': numbers, 'public_names': sorted(self.public), 'tls_size': self.tls_size,
                       'tls_layout': [{'name': n.name, 'module': self.plans[n.index].module_id,
                                       'offset': self.plans[n.index].tls_offset, 'size': n.elf.tls[6]}
                                      for n in self.link_order if n.elf.tls],
                       'segments': {name: address for component in components.values() for name, address in component['segments'].items()},
                       'complete': False})
        if self.core:
            from aot.tunables import extract
            loader = next((node for node in self.graph.nodes if node.intrinsic and
                           any(s['name']=='__tunable_get_val' and s['shndx'] for s in node.elf.symbols)), None)
            if loader is None:
                raise ValueError('glibc bootstrap requires its matching loader binary for tunable defaults')
            report = emit_native(directory, report, extract(loader.elf))
            native = (directory/'native.c').read_text()
            # Generate static module access from the TLS plan, without a loader.
            native = native.replace('if (index[0] != 1) __builtin_trap();\n    return (unsigned char *)aot_thread_pointer()+16+index[1];',
                                    'switch (index[0]) {\n'+''.join(
                                        f'    case {self.plans[n.index].module_id}: return (unsigned char *)aot_thread_pointer()+{self.plans[n.index].tls_offset}+index[1];\n'
                                        for n in self.link_order if n.elf.tls)+'    default: __builtin_trap();\n    }')
            native += '\n'+Path(__file__).with_name('native_dl.c').read_text()
        else:
            if numbers:
                raise ValueError('syscall lowering currently needs the libc platform profile')
            native = '#include <stdint.h>\n#include <crt_externs.h>\n'
            native += 'struct tlv {void *(*address)(struct tlv *); uintptr_t key,offset;};\nextern struct tlv aot_tls_descriptor;\n'
            native += 'void *aot_thread_pointer(void) {return (unsigned char *)aot_tls_descriptor.address(&aot_tls_descriptor)+0x1000;}\n'
            native += '__attribute__((constructor)) static void aot_initialize(void) {int argc=*_NSGetArgc();char **argv=*_NSGetArgv(), **env=*_NSGetEnviron();\n/* BUNDLE_INIT */\n}\n'
            native = native.replace('/* BUNDLE_INIT */', '(void)argc;(void)argv;(void)env;\n/* BUNDLE_INIT */')
            report['unsupported_syscalls'] = []
        declarations, calls, finalizers = [], [], []
        for node in self.graph.initialization_order():
            component = components[node.index]
            if node is not self.core:
                for i in range(len(component['initializers'])):
                    name = f'aot_{node.identifier}_init_{i}'
                    declarations.append(f'extern void {name}(int,char **,char **);')
                    calls.append(f'    {name}(argc,argv,env);')
            for i in range(len(component['finalizers'])):
                name = f'aot_{node.identifier}_fini_{i}'
                declarations.append(f'extern void {name}(void);')
                finalizers.append(name+'();')
        if self.core:
            native = native.replace('aot_original_init_2(argc, argv, env);', 'aot_original_init_2(argc, argv, env);\n'+'\n'.join(calls))
        else:
            native = native.replace('/* BUNDLE_INIT */', '\n'.join(calls))
        # Declarations must precede the constructor calls.
        native = '\n'.join(declarations)+'\n'+native
        if finalizers:
            native += '\n__attribute__((destructor)) static void aot_finalize(void) {'+' '.join(reversed(finalizers))+'}\n'
        (directory/'native.c').write_text(native)
        (directory/'exports.txt').write_text(''.join('_g_'+name+'\n' for name in sorted(self.public)))
        (directory/'manifest.json').write_text(json.dumps(report, indent=2)+'\n')
        return report
