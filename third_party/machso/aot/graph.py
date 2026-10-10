"""Discover ELF dependencies and resolve symbols from binary metadata."""
from collections import deque
from dataclasses import dataclass, field
import hashlib
from pathlib import Path
from machso import ELF


@dataclass
class Node:
    path: Path
    elf: ELF
    index: int
    dependencies: list = field(default_factory=list)
    intrinsic: bool = False

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
    def __init__(self, source, library_paths=()):
        self.paths = [Path(p).resolve() for p in library_paths]
        self.nodes = []
        self.by_path = {}
        self.edges = []
        self.root = self.load(Path(source).resolve())
        queue = deque([self.root])
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
            node.intrinsic = node.name.startswith('ld-linux-')
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
            elf = ELF(path.read_bytes(), allow_tls=True, machines=(183, 62))
            node = Node(path, elf, len(self.nodes))
            self.nodes.append(node)
            self.by_path[path] = node
        return self.by_path[path]

    def find(self, node, name):
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
            elf = ELF(candidate.read_bytes(), allow_tls=True, machines=(183, 62))
            if elf.machine == node.elf.machine:
                return candidate.resolve()
            mismatches.append(str(candidate))
        suffix = f'; wrong ISA candidates: {mismatches}' if mismatches else ''
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
        return result

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
        visit(self.root)
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
        return {'root': self.root.name, 'architecture': self.root.elf.architecture,
                'nodes': [{'id': n.identifier, 'name': n.name, 'input': str(n.path),
                           'sha256': hashlib.sha256(n.elf.data).hexdigest(),
                           'needed': n.elf.needed, 'tls_bytes': n.elf.tls[6] if n.elf.tls else 0,
                           'intrinsic': n.intrinsic} for n in self.nodes],
                'edges': [{'owner': n.name, 'needed': name, 'provider': dep.name} for n, name, dep in self.edges],
                'bindings': bindings,
                'missing_required': [{'owner': n.name, 'name': s['name'], 'version': s['version']} for n, s in self.missing],
                'initialization_order': [n.name for n in self.initialization_order()]}
