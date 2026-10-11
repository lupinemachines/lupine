"""Discover ELF dependencies and resolve symbols from binary metadata."""
from collections import deque
from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from machso import ELF
from aot.native_abi import validate_signatures


@dataclass
class Node:
    path: Path
    elf: ELF
    index: int
    dependencies: list = field(default_factory=list)
    intrinsic: bool = False
    native: dict | None = None

    @property
    def name(self):
        return self.elf.soname or self.path.name

    @property
    def identifier(self):
        return f'n{self.index}'

    def symbol_name(self, index):
        return f'_aot_{self.identifier}_s{index}'


def exported(symbol):
    return (symbol['shndx'] and symbol['shndx'] < 0xff00 and
            symbol['info'] >> 4 in (1, 2) and symbol['other'] & 3 in (0, 3))


class Graph:
    def __init__(self, source, library_paths=(), additional_roots=(), native_providers=(), dependency_provider=None):
        self.paths = [Path(p).resolve() for p in library_paths]
        self.dependency_provider = dependency_provider
        self.nodes = []
        self.by_path = {}
        self.edges = []
        self.native_paths = {}
        self.native_declarations = {}
        for path in native_providers:
            path = Path(path).resolve()
            declaration = json.loads(path.read_text())
            if declaration.get('schema') != 1 or declaration.get('architecture') != 'arm64':
                raise ValueError('native provider requires schema 1 and ARM64 architecture')
            name = declaration.get('soname')
            if not isinstance(name, str) or not name or '/' in name or name in self.native_paths:
                raise ValueError(f'invalid or duplicate native provider SONAME: {name}')
            install_name = declaration.get('native_install_name')
            names = declaration.get('exports')
            if (not isinstance(install_name, str) or not install_name or
                    not isinstance(names, list) or not names or
                    any(not isinstance(symbol, str) or not symbol for symbol in names)):
                raise ValueError(f'{name}: native install name and exports are required')
            self.native_paths[name] = path
            tls_exports = declaration.get('tls_exports', [])
            if (not isinstance(tls_exports, list) or
                    any(not isinstance(symbol, str) or symbol not in names for symbol in tls_exports)):
                raise ValueError(f'{name}: native TLS exports must be declared exported names')
            validate_signatures(declaration)
            if set(tls_exports) & declaration.get('signatures', {}).keys():
                raise ValueError(f'{name}: native TLS exports cannot have function signatures')
            self.native_declarations[path] = declaration
        self.root = self.load(Path(source).resolve())
        self.roots = [self.root]
        for source in additional_roots:
            node = self.load(Path(source).resolve())
            if node.elf.machine != self.root.elf.machine:
                raise ValueError(f'ISA mismatch in additional root {node.name}')
            if node not in self.roots:
                self.roots.append(node)
        queue = deque(self.roots)
        while queue:
            node = queue.popleft()
            for name in node.elf.needed:
                path = self.find(node, name)
                was_known = path in self.by_path
                dependency = self.load(path)
                if dependency.elf.machine != self.root.elf.machine:
                    raise ValueError(f'ISA mismatch: {node.name} needs {name}, got {dependency.elf.architecture}')
                node.dependencies.append(dependency)
                self.edges.append((node, name, dependency))
                if not was_known:
                    queue.append(dependency)
        # Linux loader functionality is compiled into platform bootstrap code.
        # Its entire ELF image must not be initialized as another process loader.
        for node in self.nodes:
            node.intrinsic = bool(node.native) or node.name.startswith('ld-linux-')
        self.bindings = {}
        self.missing = []
        for node in self.nodes:
            if node.intrinsic:
                continue
            for index, symbol in enumerate(node.elf.symbols):
                if symbol['shndx'] or not symbol['name'] or symbol['info'] >> 4 == 0:
                    continue
                provider = self.resolve(node, symbol)
                self.bindings[(node.index, index)] = provider
                if provider is None and symbol['info'] >> 4 != 2:
                    self.missing.append((node, symbol))

    def load(self, path):
        path = path.resolve()
        if path not in self.by_path:
            if path in self.native_declarations:
                declaration = self.native_declarations[path]
                symbols = [{'name': name, 'info': 0x12, 'shndx': 1, 'other': 0,
                            'version': None, 'version_hidden': False, 'value': 0}
                           for name in declaration['exports']]
                elf = SimpleNamespace(soname=declaration['soname'], symbols=symbols,
                                      needed=[], tls=None, machine=183, architecture='aarch64',
                                      data=path.read_bytes())
                node = Node(path, elf, len(self.nodes), intrinsic=True, native=declaration)
                self.nodes.append(node)
                self.by_path[path] = node
                return node
            elf = ELF.from_file(path, allow_tls=True, machines=(183, 62))
            node = Node(path, elf, len(self.nodes))
            self.nodes.append(node)
            self.by_path[path] = node
        return self.by_path[path]

    def find(self, node, name):
        if name in self.native_paths:
            return self.native_paths[name]
        if '/' in name:
            candidate = Path(name)
            if not candidate.is_absolute():
                candidate = node.path.parent/candidate
            if candidate.is_file():
                return candidate.resolve()
            raise ValueError(f'{node.name}: missing dependency {name}')
        directories = []
        for entry in (node.elf.runpath or node.elf.rpath or '').split(':'):
            if entry:
                entry = entry.replace('${ORIGIN}', str(node.path.parent)).replace('$ORIGIN', str(node.path.parent))
                if '$' in entry:
                    raise ValueError(f'{node.name}: unsupported dynamic path token: {entry}')
                directories.append(Path(entry))
        directories += [node.path.parent, *self.paths]
        mismatches = []
        for directory in directories:
            candidate = directory/name
            if not candidate.is_file():
                continue
            elf = ELF.from_file(candidate, allow_tls=True, machines=(183, 62))
            if elf.machine == node.elf.machine:
                return candidate.resolve()
            mismatches.append(str(candidate))
        suffix = f'; wrong ISA candidates: {mismatches}' if mismatches else ''
        if self.dependency_provider:
            candidate = self.dependency_provider.find(name, node.elf.machine)
            elf = ELF.from_file(candidate, allow_tls=True, machines=(183, 62))
            if elf.machine != node.elf.machine:
                raise ValueError(f'{name}: dependency provider returned the wrong ISA')
            return candidate.resolve()
        raise ValueError(f'{node.name}: missing {name}; supply --library-path{suffix}')

    def scope(self, node):
        seen, result = set(), []
        queue = deque([node, *self.nodes])
        while queue:
            candidate = queue.popleft()
            if candidate.index in seen:
                continue
            seen.add(candidate.index)
            result.append(candidate)
            queue.extend(candidate.dependencies)
        # Converted C++/libc definitions must retain their own ABI domain.
        # Native providers satisfy imports after regular ELF definitions.
        return [n for n in result if not n.native]+[n for n in result if n.native]

    def resolve(self, node, symbol):
        for candidate in self.scope(node):
            for index, definition in enumerate(candidate.elf.symbols):
                if not exported(definition) or definition['name'] != symbol['name']:
                    continue
                if symbol['version']:
                    if definition['version'] != symbol['version']:
                        continue
                elif definition['version_hidden']:
                    continue
                if candidate.native:
                    if symbol['name'] in candidate.native.get('tls_exports', []):
                        raise ValueError(f"{candidate.name}:{symbol['name']}: static native TLS imports are unsupported; use dlsym")
                    reason = candidate.native.get('signature_report', {}).get('unsupported', {}).get(symbol['name'])
                    if reason:
                        raise ValueError(f"{candidate.name}:{symbol['name']}: {reason}")
                return candidate, index
        return None

    def initialization_order(self):
        result, seen = [], set()
        def visit(node):
            if node.index in seen:
                return
            seen.add(node.index)
            for dependency in node.dependencies:
                visit(dependency)
            if not node.intrinsic:
                result.append(node)
        for root in self.roots:
            visit(root)
        return result

    def report(self):
        bindings = []
        for (owner, index), provider in self.bindings.items():
            symbol = self.nodes[owner].elf.symbols[index]
            bindings.append({'owner': self.nodes[owner].name, 'name': symbol['name'],
                             'version': symbol['version'],
                             'provider': provider[0].name if provider else None,
                             'weak': symbol['info'] >> 4 == 2,
                             'intrinsic': provider[0].intrinsic if provider else False})
        return {'root': self.root.name, 'roots': [n.name for n in self.roots],
                'architecture': self.root.elf.architecture,
                'nodes': [{'id': n.identifier, 'name': n.name, 'input': str(n.path),
                           'sha256': hashlib.sha256(n.elf.data).hexdigest(),
                           'needed': n.elf.needed, 'tls_bytes': n.elf.tls[6] if n.elf.tls else 0,
                           'intrinsic': n.intrinsic} for n in self.nodes],
                'native_providers': [n.native for n in self.nodes if n.native],
                'system_packages': self.dependency_provider.report() if self.dependency_provider else [],
                'edges': [{'owner': n.name, 'needed': name, 'provider': dep.name} for n, name, dep in self.edges],
                'bindings': bindings,
                'missing_required': [{'owner': n.name, 'name': s['name'], 'version': s['version']} for n, s in self.missing],
                'initialization_order': [n.name for n in self.initialization_order()]}
