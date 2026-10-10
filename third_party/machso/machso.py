#!/usr/bin/env python3
"""Repackage a restricted AArch64 ELF DSO as a native Mach-O dylib."""
import argparse
import json
import struct
from pathlib import Path


def pack(fmt, *args):
    return struct.pack('<' + fmt, *args)


def align(n, a=0x4000):
    return (n + a - 1) & -a


def uleb(n):
    if n < 0:
        raise ValueError('negative ULEB128')
    out = bytearray()
    while n >= 128:
        out.append((n & 127) | 128)
        n >>= 7
    return bytes(out + bytes([n]))


class ELF:
    def __init__(self, data, allow_tls=False, machines=(183,)):
        self.data = data
        if data[:7] != b'\x7fELF\x02\x01\x01':
            raise ValueError('requires little-endian ELF64')
        h = struct.unpack_from('<HHIQQQIHHHHHH', data, 16)
        if h[0] != 3 or h[1] not in machines:
            raise ValueError('requires a supported ET_DYN ISA (AArch64 for native compilation)')
        self.machine = h[1]
        self.architecture = {183: 'arm64', 62: 'x86_64'}.get(h[1], str(h[1]))
        self.ph = [struct.unpack_from('<IIQQQQQQ', data, h[4] + i*h[8]) for i in range(h[9])]
        self.tls = next((p for p in self.ph if p[0] == 7), None)
        if self.tls and not allow_tls:
            raise ValueError('ELF TLS is unsupported')
        self.loads = [p for p in self.ph if p[0] == 1]
        self.tags = {}
        dyn = next(p for p in self.ph if p[0] == 2)
        for off in range(dyn[2], dyn[2] + dyn[5], 16):
            tag, val = struct.unpack_from('<qQ', data, off)
            if not tag:
                break
            self.tags.setdefault(tag, []).append(val)
        if 22 in self.tags or self.tag(30) & 4:
            raise ValueError('text relocations are unsupported')
        strings = self.read(self.tag(5), self.tag(10))
        self.string = lambda off: strings[off:strings.index(0, off)].decode()
        self.needed = [self.string(v) for v in self.tags.get(1, [])]
        if self.tag(4):
            count = struct.unpack('<II', self.read(self.tag(4), 8))[1]
        else:
            gh = self.tag(0x6ffffef5)
            nb, start, bloom, _ = struct.unpack('<IIII', self.read(gh, 16))
            buckets = struct.unpack('<' + 'I'*nb, self.read(gh+16+bloom*8, nb*4))
            count = max(buckets)
            if count >= start:
                chain = gh+16+bloom*8+nb*4
                while not struct.unpack('<I', self.read(chain+(count-start)*4, 4))[0] & 1:
                    count += 1
                count += 1
            else:
                count = start
        self.symbols = []
        for i in range(count):
            name, info, other, shndx, value, size = struct.unpack('<IBBHQQ', self.read(self.tag(6)+24*i, 24))
            self.symbols.append(dict(name=self.string(name), info=info, other=other, shndx=shndx, value=value, size=size))
        self.relocs = []
        if self.tag(17) or self.tag(36):
            raise ValueError('REL/RELR relocations are unsupported')
        for addr, size in [(self.tag(7), self.tag(8)), (self.tag(23), self.tag(2))]:
            if size and ((addr == self.tag(23) and self.tag(20) != 7) or size % 24):
                raise ValueError('requires RELA relocations')
            for off in range(0, size, 24):
                target, info, addend = struct.unpack('<QQq', self.read(addr+off, 24))
                self.relocs.append((target, info & 0xffffffff, info >> 32, addend))
        self.soname = self.string(self.tag(14)) if self.tag(14) else None
        self.runpath = self.string(self.tag(29)) if self.tag(29) else None
        self.rpath = self.string(self.tag(15)) if self.tag(15) else None
        self.parse_versions()

    def parse_versions(self):
        version_names = {}
        version_files = {}
        address = self.tag(0x6ffffffc)
        for _ in range(self.tag(0x6ffffffd)):
            _, _, index, _, _, auxiliary, following = struct.unpack('<HHHHIII', self.read(address, 20))
            name, _ = struct.unpack('<II', self.read(address+auxiliary, 8))
            version_names[index] = self.string(name)
            if not following:
                break
            address += following
        address = self.tag(0x6ffffffe)
        for _ in range(self.tag(0x6fffffff)):
            _, count, filename, auxiliary, following = struct.unpack('<HHIII', self.read(address, 16))
            item = address+auxiliary
            for _ in range(count):
                _, _, index, name, next_aux = struct.unpack('<IHHII', self.read(item, 16))
                version_names[index & 0x7fff] = self.string(name)
                version_files[index & 0x7fff] = self.string(filename)
                if not next_aux:
                    break
                item += next_aux
            if not following:
                break
            address += following
        versions = struct.unpack('<'+'H'*len(self.symbols), self.read(self.tag(0x6ffffff0), len(self.symbols)*2)) if self.tag(0x6ffffff0) else [0]*len(self.symbols)
        for symbol, index in zip(self.symbols, versions):
            symbol['version'] = version_names.get(index & 0x7fff)
            symbol['version_file'] = version_files.get(index & 0x7fff)
            symbol['version_hidden'] = bool(index & 0x8000)

    def tag(self, key):
        return self.tags.get(key, [0])[0]

    def read(self, addr, size):
        for p in self.loads:
            if p[3] <= addr and addr+size <= p[3]+p[5]:
                return self.data[p[2]+addr-p[3]:p[2]+addr-p[3]+size]
        raise ValueError(f'ELF address not file-backed: {addr:#x}')


SYSTEM_LIBRARIES = {'libc.so.6', 'ld-linux-aarch64.so.1'}
SHIMS = {'stdout', 'stderr', '__errno_location', '__ctype_b_loc', '__ctype_toupper_loc', 'syscall', 'read', 'write', 'close',
         'open', '__snprintf_chk', '__printf_chk', '__fprintf_chk', 'snprintf', 'printf', 'fprintf'}
DIRECT = {'strlen', 'strncpy', 'strdup', 'qsort', 'abs', 'getpid', 'fflush',
          '__stack_chk_guard', '__stack_chk_fail', '__cxa_finalize', 'malloc', 'calloc',
          'realloc', 'free', 'memcpy', 'memmove', 'memset', 'memcmp', 'strcmp', 'puts'}


def export_trie(exports):
    # A byte trie avoids the 255-child limit of a flat root for large DSOs.
    nodes = [{'children': {}, 'terminal': b''}]
    for symbol in exports:
        index = 0
        for character in ('_'+symbol['name']).encode():
            children = nodes[index]['children']
            if character not in children:
                children[character] = len(nodes)
                nodes.append({'children': {}, 'terminal': b''})
            index = children[character]
        flags = 4 if symbol['info'] >> 4 == 2 else 0
        nodes[index]['terminal'] = uleb(flags)+uleb(symbol['value']+0x4000)
    offsets = [0]*len(nodes)
    while True:
        blobs = []
        for node in nodes:
            terminal = node['terminal']
            children = node['children']
            blobs.append(uleb(len(terminal))+terminal+bytes([len(children)])+
                         b''.join(bytes([character, 0])+uleb(offsets[child])
                                  for character, child in children.items()))
        new, pos = [], 0
        for blob in blobs:
            new.append(pos)
            pos += len(blob)
        if new == offsets:
            return b''.join(blobs)
        offsets = new


def section(name, seg, addr, size, offset, flags=0):
    return pack('16s16sQQIIIIIIII', name.encode(), seg.encode(), addr, size, offset, 3, 0, 0, flags, 0, 0, 0)


def segment(name, vm, size, offset, prot, sections=()):
    return pack('II16sQQQQiiII', 0x19, 72+80*len(sections), name.encode(), vm, size, offset, size, prot, prot, len(sections), 0)+b''.join(sections)


def dylib_command(cmd, name):
    raw = name.encode()+b'\0'
    size = align(24+len(raw), 8)
    return pack('IIIIII', cmd, size, 24, 0, 0x10000, 0x10000)+raw+b'\0'*(size-24-len(raw))


def convert(src, dst, dependency_names=(), providers=None):
    elf = ELF(src.read_bytes())
    providers = providers or {}
    if not dependency_names and set(elf.needed) - SYSTEM_LIBRARIES:
        raise ValueError('ELF dependencies require --recursive and a --library-path')
    if len(elf.loads) != 2 or elf.loads[0][1] != 5 or elf.loads[1][1] != 6 or elf.loads[0][3] != 0:
        raise ValueError('requires two LOAD segments: RX at zero, then RW')
    rx, rw = elf.loads
    executable = elf.data[rx[2]:rx[2]+rx[5]]
    for offset in range(0, len(executable)-3, 4):
        if struct.unpack_from('<I', executable, offset)[0] & 0xffe0001f == 0xd4000001:
            raise ValueError(f'possible direct SVC at ELF address {rx[3]+offset:#x}; syscall rewriting is not implemented')
    bias = 0x4000
    data_vm = (rw[3] & -0x4000)+bias
    text_size = align(rx[6]+bias)
    if data_vm < text_size:
        raise ValueError('LOAD segments overlap at the Mach-O page size')
    init_values = ([elf.tag(12)] if elf.tag(12) else [])
    if elf.tag(27):
        init_values += list(struct.unpack('<'+'Q'*(elf.tag(27)//8), elf.read(elf.tag(25), elf.tag(27))))
        # RELATIVE addends are authoritative; the stored pointers can be zero.
        for i in range(elf.tag(27)//8):
            for target, kind, _, addend in elf.relocs:
                if target == elf.tag(25)+i*8:
                    if kind != 1027:
                        raise ValueError('imported initializers unsupported')
                    init_values[i + bool(elf.tag(12))] = addend
    init_addr = align(rw[3]+rw[6]+bias, 8)
    data_size = align(init_addr+len(init_values)*8-data_vm)
    link_vm = data_vm+data_size
    image = bytearray(link_vm)
    image[bias:bias+rx[5]] = elf.data[rx[2]:rx[2]+rx[5]]
    image[rw[3]+bias:rw[3]+bias+rw[5]] = elf.data[rw[2]:rw[2]+rw[5]]
    rebases, binds = [], []
    def location(addr):
        if not rw[3]+bias <= addr <= rw[3]+bias+rw[6]-8:
            raise ValueError(f'relocation outside writable ELF segment: {addr:#x}')
        return addr-data_vm
    for target, kind, symidx, addend in elf.relocs:
        addr = target+bias
        loc = location(addr)
        if kind == 1027:
            value = addend+bias
        elif kind in (257, 1025, 1026):
            sym = elf.symbols[symidx]
            if sym['shndx']:
                if sym['info'] & 15 not in (0, 1, 2) or sym['shndx'] >= 0xff00:
                    raise ValueError('unsupported defined relocation symbol')
                value = sym['value']+bias+addend
            else:
                name = sym['name']
                if name in providers:
                    binds.append((loc, providers[name], '_'+name))
                elif name in SHIMS:
                    binds.append((loc, 1, '_elf_'+name))
                elif name in DIRECT:
                    binds.append((loc, 2, '_'+name))
                elif sym['info'] >> 4 != 2:
                    raise ValueError(f'unsupported import: {name}')
                if addend:
                    raise ValueError('import addends unsupported')
                image[addr:addr+8] = pack('Q', 0)
                continue
        else:
            raise ValueError(f'unsupported AArch64 relocation {kind}')
        image[addr:addr+8] = pack('Q', value)
        rebases.append(loc)
    for i, value in enumerate(init_values):
        addr = init_addr+i*8
        image[addr:addr+8] = pack('Q', value+bias)
        rebases.append(addr-data_vm)
    rebase = b'\x11'+b''.join(b'\x21'+uleb(o)+b'\x51' for o in rebases)+b'\0'
    bind = b''.join(b'\x20'+uleb(ordinal)+b'\x40'+name.encode()+b'\0\x51\x71'+uleb(o)+b'\x90' for o, ordinal, name in binds)+b'\0'
    exports = [s for s in elf.symbols if s['shndx'] and s['shndx'] < 0xff00 and s['info'] >> 4 in (1, 2) and s['other'] & 3 in (0, 3)]
    if any(s['info'] & 15 not in (1, 2) for s in exports):
        raise ValueError('unsupported exports (TLS/IFUNC)')
    trie = export_trie(exports)
    strings = bytearray(b'\0')
    symtab = bytearray()
    for s in exports:
        index = len(strings)
        strings += ('_'+s['name']).encode()+b'\0'
        weak = 0x80 if s['info'] >> 4 == 2 else 0
        symtab += pack('IBBHQ', index, 0xf, 1 if s['info'] & 15 == 2 else 2, weak, s['value']+bias)
    blobs = [rebase, bind, trie, bytes(symtab), bytes(strings)]
    offsets = []
    for blob in blobs:
        offsets.append(len(image))
        image += blob
    link_size = align(len(image)-link_vm)
    image += b'\0'*(link_vm+link_size-len(image))
    commands = [segment('__TEXT', 0, text_size, 0, 5, [section('__elf', '__TEXT', bias, rx[6], bias, 0x80000400)]),
                segment('__DATA', data_vm, data_size, data_vm, 3, [section('__elf', '__DATA', rw[3]+bias, rw[6], rw[3]+bias), section('__mod_init_func', '__DATA', init_addr, len(init_values)*8, init_addr, 9)]),
                segment('__LINKEDIT', link_vm, link_size, link_vm, 1),
                dylib_command(0xd, '@rpath/'+dst.name),
                dylib_command(0xc, '@loader_path/libelfcompat.dylib'),
                dylib_command(0xc, '/usr/lib/libSystem.B.dylib'),
                pack('IIIIII', 0x32, 24, 1, 0x000b0000, 0x000b0000, 0),
                pack('IIIIIIIIIIII', 0x80000022, 48, offsets[0], len(rebase), offsets[1], len(bind), 0, 0, 0, 0, offsets[2], len(trie)),
                pack('IIIIII', 2, 24, offsets[3], len(exports), offsets[4], len(strings)),
                pack('I'*20, 0xb, 80, 0, 0, 0, len(exports), len(exports), 0, *([0]*12))]
    commands += [dylib_command(0xc, '@loader_path/'+name) for name in dependency_names]
    header = pack('IIIIIIII', 0xfeedfacf, 0x100000c, 0, 6, len(commands), sum(map(len, commands)), 0x85, 0)+b''.join(commands)
    if len(header) > bias:
        raise ValueError('load commands overflow header page')
    image[:len(header)] = header
    dst.write_bytes(image)
    return dict(architecture='arm64', exports=[s['name'] for s in exports], dependencies=elf.needed, rebases=len(rebases), binds=len(binds), initializers=len(init_values), signing_required=True)


class DependencyGraph:
    """Discover the complete ELF closure before selecting Mach-O binding ordinals."""
    def __init__(self, source, output, library_paths=()):
        self.library_paths = list(library_paths)
        self.nodes = {}
        self.outputs = {source.resolve(): output}
        self.root = source.resolve()
        self.directory = output.parent
        self.visit(self.root)

    def visit(self, path):
        if path in self.nodes:
            return
        elf = ELF(path.read_bytes())
        edges = []
        self.nodes[path] = (elf, edges)
        for name in elf.needed:
            if name in SYSTEM_LIBRARIES:
                continue
            if Path(name).name != name:
                raise ValueError(f'dependency needs a simple library name: {name}')
            candidates = [directory/name for directory in [path.parent, *self.library_paths]]
            dependency = next((p.resolve() for p in candidates if p.is_file()), None)
            if dependency is None:
                raise ValueError(f'{path.name}: missing {name}; add --library-path')
            if dependency not in self.outputs:
                destination = self.directory/(name+'.dylib')
                if destination in self.outputs.values():
                    raise ValueError(f'dependency output name collision: {destination.name}')
                self.outputs[dependency] = destination
            edges.append(dependency)
            self.visit(dependency)

    def closure(self, path):
        # Breadth-first dependency order approximates ELF lookup in an isolated group.
        seen = {path}
        queue = list(self.nodes[path][1])
        result = []
        for dependency in queue:
            if dependency in seen:
                continue
            seen.add(dependency)
            result.append(dependency)
            queue.extend(self.nodes[dependency][1])
        return result

    def convert(self):
        # Stage every result first: an unsupported leaf must not leave a partial bundle.
        import tempfile
        reports = {}
        with tempfile.TemporaryDirectory() as temporary:
            staged = []
            for path in self.nodes:
                dependencies = self.closure(path)
                providers = {}
                for ordinal, dependency in enumerate(dependencies, 3):
                    for symbol in self.nodes[dependency][0].symbols:
                        if (symbol['shndx'] and symbol['shndx'] < 0xff00
                                and symbol['info'] >> 4 in (1, 2)
                                and symbol['other'] & 3 in (0, 3)):
                            providers.setdefault(symbol['name'], ordinal)
                destination = self.outputs[path]
                stage = Path(temporary)/destination.name
                report = convert(path, stage, [self.outputs[p].name for p in dependencies], providers)
                reports[destination.name] = report
                staged.append((stage, destination))
            self.directory.mkdir(parents=True, exist_ok=True)
            for stage, destination in staged:
                destination.write_bytes(stage.read_bytes())
        return reports


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('input', type=Path)
    ap.add_argument('output', type=Path)
    ap.add_argument('--recursive', action='store_true', help='convert the DT_NEEDED dependency closure')
    ap.add_argument('-L', '--library-path', action='append', default=[], type=Path)
    args = ap.parse_args()
    try:
        if args.recursive:
            report = DependencyGraph(args.input, args.output, args.library_path).convert()
        else:
            report = convert(args.input, args.output)
        print(json.dumps(report, indent=2))
    except (ValueError, OSError, struct.error, IndexError, StopIteration) as exc:
        ap.exit(1, f'machso: {exc}\n')


if __name__ == '__main__':
    main()
