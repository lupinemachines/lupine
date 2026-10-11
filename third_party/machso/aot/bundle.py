"""Compile a dependency closure as one native dylib using ELF-derived bindings."""
from dataclasses import dataclass, field
import json
from pathlib import Path
from machso import align
from aot.graph import Graph, exported
from aot.emit import emit, scan, save_frame, restore_frame, lower_x18, ELF_BASE
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
    sites: list = field(default_factory=list)
    x18_sites: list = field(default_factory=list)
    thunk_segment: str = ''
    thunk_address: int = 0
    thunks: list = field(default_factory=list)
    site_thunks: dict = field(default_factory=dict)


def trampoline_layout(start, extent, sites, identifier):
    """Reserve nearby prefix/suffix islands without moving original ELF LOADs."""
    sites = sorted(sites, key=lambda site: site['address'])
    sizes = [0]
    for site in sites:
        words = len(save_frame())+(
            len(lower_x18(site)[0])+4 if site['kind']=='x18' else len(restore_frame())+8)
        sizes.append(sizes[-1]+words*4)

    def allocation(begin, end):
        return align(64+sizes[end]-sizes[begin], 0x4000)

    extent = align(extent, 0x4000)
    prefix = allocation(0, len(sites))
    groups = [sites]
    if sites and max(s['address'] for s in sites)+prefix >= 1 << 27:
        # Choose the largest reachable prefix, accounting for the island's
        # own instructions. Splitting at the image midpoint fails when a
        # dense cluster makes either island much larger than the other.
        low, high = 0, len(sites)
        while low < high:
            middle = (low+high+1)//2
            if allocation(0, middle)+sites[middle-1]['address'] <= (1 << 27)-4:
                low = middle
            else:
                high = middle-1
        groups = [sites[:low], sites[low:]]
        prefix = allocation(0, low)
    image_base = start+prefix
    thunks = []
    for index, group in enumerate(groups):
        address = start if index == 0 else image_base+extent
        size = prefix if index == 0 else allocation(len(groups[0]), len(sites))
        for site in group:
            origin = image_base+site['address']
            if address-origin < -(1 << 27)+4 or address+size-origin >= 1 << 27:
                raise ValueError('patch sites exceed the ARM64 trampoline branch range')
        thunks.append({'segment': f'__{identifier.upper()}T'+(str(index) if index else ''),
                       'address': address, 'size': size, 'sites': group})
    return image_base, thunks, max(image_base+extent, thunks[-1]['address']+thunks[-1]['size'])


DYNAMIC_LINKER_APIS = {'dlopen', 'dlmopen', 'dlsym', 'dlvsym', 'dlclose', 'dlerror',
                       '__cxa_thread_atexit_impl', '__call_tls_dtors'}


class Bundle:
    def __init__(self, source, library_paths=(), additional_roots=(), native_providers=(), dependency_provider=None,
                 export_interface=None):
        self.graph = Graph(source, library_paths, additional_roots, native_providers, dependency_provider)
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
        self.native_bridges = {}
        base = ELF_BASE
        for node in self.link_order:
            names = [f'__{node.identifier.upper()}L{i}' for i in range(len(node.elf.loads))]
            sites, x18 = scan(node.elf)
            # Reserve a nearby executable region before each ELF image. Large
            # read-only GPU blobs can make a LOAD hundreds of MB wide even
            # though CPU code and its branch sites fit within ARM64 B26 reach.
            extent = max(p[3]+p[6] for p in node.elf.loads)
            try:
                image_base, thunks, end = trampoline_layout(base, extent, sites, node.identifier)
            except ValueError as error:
                raise ValueError(f'{node.name}: {error}') from error
            plan = Plan(node.identifier, image_base, names, core=node is self.core,
                        sites=sites, x18_sites=x18, thunk_segment=f'__{node.identifier.upper()}T',
                        thunk_address=base, thunks=thunks,
                        site_thunks={site['address']: thunk['segment'] for thunk in thunks for site in thunk['sites']})
            for index, symbol in enumerate(node.elf.symbols):
                if exported(symbol):
                    plan.definitions[index] = node.symbol_name(index)
                    if plan.core and symbol['name'] in DYNAMIC_LINKER_APIS:
                        plan.overrides[symbol['value']] = '_aot_linux_'+symbol['name']
            self.plans[node.index] = plan
            base = align(end, 0x10000)+0x10000
        # Public native names use the group's ELF search order. Versioned imports
        # retain separate internal symbols and bind to their exact definitions.
        self.public = self.graph.public_definitions()
        for name, (node, index) in self.public.items():
            self.plans[node.index].public[index] = name
        self.export_bridges = {}
        self.export_interface = None
        if export_interface is not None:
            from aot.native_abi import validate_signatures
            self.export_interface = json.loads(Path(export_interface).read_text())
            if (not isinstance(self.export_interface, dict) or self.export_interface.get('schema') != 1 or
                    self.export_interface.get('architecture') not in ('arm64', 'aarch64')):
                raise ValueError('export interface requires schema 1 and ARM64 architecture')
            names = self.export_interface.get('exports')
            if not isinstance(names, list) or any(not isinstance(name, str) or not name for name in names):
                raise ValueError('export interface requires exported names')
            signatures = validate_signatures(self.export_interface)
            for name, signature in signatures.items():
                if name not in self.public:
                    raise ValueError(f'{name}: export signature does not name a public ELF definition')
                node, index = self.public[name]
                if node.elf.symbols[index]['info'] & 15 != 2:
                    raise ValueError(f'{name}: export ABI bridges require regular ELF functions')
                self.export_bridges[name] = (node, index, signature)
                del self.plans[node.index].public[index]
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
                        if owner.native:
                            plan.imports[index] = '_'+symbol['name']
                            signature = owner.native.get('signatures', {}).get(symbol['name'])
                            if signature:
                                label = f'_aot_native_{owner.identifier}_s{definition_index}'
                                plan.imports[index] = label
                                self.native_bridges[label] = (symbol['name'], signature)
                        elif owner.intrinsic:
                            plan.imports[index] = '_aot_import_'+symbol['name']
                        else:
                            plan.imports[index] = owner.symbol_name(definition_index)
                            definition = owner.elf.symbols[definition_index]
                            provider_plan = self.plans[owner.index]
                            plan.tls_symbols[index] = provider_plan.tls_offset+definition['value']
                            plan.tls_modules[index] = provider_plan.module_id
                    elif symbol['info'] >> 4 == 2:
                        plan.imports[index] = None
        # Loader lookups can request functions absent from the ELF import table.
        # Prepare the declared ABI bridges before those pointers escape to Linux.
        self.native_pointer_providers = []
        self.native_pointer_resolvers = {}
        for owner in self.graph.nodes:
            if not owner.native:
                continue
            signatures = owner.native.get('signatures', {})
            pointer_outputs = any(s.get('function_pointer_output') for s in signatures.values())
            if pointer_outputs:
                self.native_pointer_providers.append(owner)
            if self.core or pointer_outputs:
                for index, symbol in enumerate(owner.elf.symbols):
                    if symbol['name'] in signatures:
                        label = f'_aot_native_{owner.identifier}_s{index}'
                        self.native_bridges[label] = (symbol['name'], signatures[symbol['name']])
                        if signatures[symbol['name']].get('function_pointer_output'):
                            self.native_pointer_resolvers[label] = f'_aot_native_{owner.identifier}_function_pointer'

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
            loader = next((node for node in self.graph.nodes if node.intrinsic and not node.native and
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
            # Late-bound modules can be included as roots even when they have
            # no DT_NEEDED edge. Their SONAMEs resolve to this same native image
            # and share one libc/TLS/dependency closure, with no hand-made map.
            from aot.modules import emit_modules
            providers = [node for node in self.graph.nodes if node.native]
            native += '\n'+emit_modules(self.nodes, self.plans, providers)
            report['native_dynamic_exports'] = {
                node.name: [symbol['name'] for symbol in node.elf.symbols
                            if symbol['name'] not in node.native.get('signature_report', {}).get('unsupported', {})]
                for node in providers}
            native += '\n'+Path(__file__).with_name('native_dl.c').read_text()
            native += '\n'+Path(__file__).with_name('native_tls_dtors.c').read_text()
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
                    declarations.append(f'extern void {name}(int,char **,char **);\n'
                                        f'static void (*volatile {name}_pointer)(int,char **,char **)={name};')
                    calls.append(f'    {name}_pointer(argc,argv,env);')
            for i in range(len(component['finalizers'])):
                name = f'aot_{node.identifier}_fini_{i}'
                declarations.append(f'extern void {name}(void);\n'
                                    f'static void (*volatile {name}_pointer)(void)={name};')
                finalizers.append(name+'_pointer();')
        if self.core:
            native = native.replace('aot_original_init_2(argc, argv, env);', 'aot_original_init_2(argc, argv, env);\n'+'\n'.join(calls))
        else:
            native = native.replace('/* BUNDLE_INIT */', '\n'.join(calls))
        # Declarations must precede the constructor calls.
        native = '\n'.join(declarations)+'\n'+native
        if finalizers:
            native += '\n__attribute__((destructor)) static void aot_finalize(void) {'+('aot_linux___call_tls_dtors(); ' if self.core else '')+' '.join(reversed(finalizers))+'}\n'
        from aot.native_abi import emit_pointer_resolver
        native += ''.join(emit_pointer_resolver(owner) for owner in self.native_pointer_providers)
        for owner in self.native_pointer_providers:
            required = set(owner.native.get('signatures', {}))
            required.update(owner.native.get('signature_report', {}).get('unsupported', {}))
            existing = report.setdefault('native_dynamic_exports', {}).get(owner.name, [])
            report['native_dynamic_exports'][owner.name] = sorted(set(existing) | required)
        if self.export_bridges:
            from aot.native_abi import emit_export_pointer_resolver
            unsupported = [self.public[name] for name in
                           self.export_interface.get('signature_report', {}).get('unsupported', {})
                           if name in self.public and self.public[name][0].elf.symbols[self.public[name][1]]['info'] & 15 == 2]
            native = '#define AOT_EXPORT_ABI 1\nvoid *aot_import_function_pointer(void *);\n'+native
            native += emit_export_pointer_resolver(self.export_bridges, self.native_bridges, unsupported)
        (directory/'native.c').write_text(native)
        if self.native_bridges or self.export_bridges:
            from aot.native_abi import emit_bridge
            assembly = ''.join(
                emit_bridge(label, '_'+name, signature, self.native_pointer_resolvers.get(label))
                for label, (name, signature) in self.native_bridges.items())
            assembly += ''.join(
                emit_bridge('_g_'+name, node.symbol_name(index), signature,
                            '_aot_export_function_pointer' if signature.get('function_pointer_output') else None,
                            caller_abi='darwin', indirect=True, private=False)
                for name, (node, index, signature) in self.export_bridges.items())
            (directory/'native_abi.S').write_text(assembly)
        report['native_abi_bridges'] = sorted(name for name, _ in self.native_bridges.values())
        report['export_abi_bridges'] = sorted(self.export_bridges)
        if self.export_interface is not None:
            report['export_interface'] = self.export_interface
        (directory/'exports.txt').write_text(''.join('_g_'+name+'\n' for name in sorted(self.public)))
        (directory/'manifest.json').write_text(json.dumps(report, indent=2)+'\n')
        return report
