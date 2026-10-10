"""Inspect native layout and restore public names after linking private ELF names."""
import struct
from pathlib import Path
from machso import uleb, align


def load_commands(data):
    offset = 32
    for _ in range(struct.unpack_from('<I', data, 16)[0]):
        cmd, size = struct.unpack_from('<II', data, offset)
        yield cmd, offset, size
        offset += size


def read_uleb(data, offset):
    value, shift = 0, 0
    while True:
        byte = data[offset]
        offset += 1
        value |= (byte & 127) << shift
        if not byte & 128:
            return value, offset
        shift += 7


def exports(data):
    command = next(o for cmd, o, _ in load_commands(data) if cmd == 0x80000033)
    start, size = struct.unpack_from('<II', data, command+8)
    trie = data[start:start+size]
    result = {}

    def visit(offset, prefix):
        length, pos = read_uleb(trie, offset)
        terminal = trie[pos:pos+length]
        if terminal:
            result[prefix] = terminal
        pos += length
        count = trie[pos]
        pos += 1
        for _ in range(count):
            end = trie.index(0, pos)
            edge = trie[pos:end].decode()
            child, pos = read_uleb(trie, end+1)
            visit(child, prefix+edge)

    visit(0, '')
    return command, start, size, result


def make_trie(entries):
    nodes = [{'children': {}, 'terminal': b''}]
    for name, terminal in entries.items():
        index = 0
        for character in name.encode():
            children = nodes[index]['children']
            if character not in children:
                children[character] = len(nodes)
                nodes.append({'children': {}, 'terminal': b''})
            index = children[character]
        nodes[index]['terminal'] = terminal
    # Compress paths to fit into the linker's existing export allocation.
    def compressed(index):
        node = nodes[index]
        children = []
        for character, child in node['children'].items():
            edge = bytes([character])
            while not nodes[child]['terminal'] and len(nodes[child]['children']) == 1:
                character, child = next(iter(nodes[child]['children'].items()))
                edge += bytes([character])
            children.append((edge, compressed(child)))
        return {'terminal': node['terminal'], 'children': children}

    root = compressed(0)
    flat = []
    def flatten(node):
        node['index'] = len(flat)
        flat.append(node)
        for _, child in node['children']:
            flatten(child)
    flatten(root)
    offsets = [0]*len(flat)
    while True:
        blobs, new, position = [], [], 0
        for node in flat:
            terminal = node['terminal']
            blob = uleb(len(terminal))+terminal+bytes([len(node['children'])])
            blob += b''.join(edge+b'\0'+uleb(offsets[child['index']]) for edge, child in node['children'])
            blobs.append(blob)
            new.append(position)
            position += len(blob)
        if new == offsets:
            return b''.join(blobs)
        offsets = new


def finish(path, expected_layout):
    data = bytearray(Path(path).read_bytes())
    for name, expected in expected_layout.items():
        cmd = next(o for cmd, o, _ in load_commands(data)
                   if cmd == 0x19 and data[o+8:o+24].rstrip(b'\0').decode() == name)
        actual = struct.unpack_from('<Q', data, cmd+24)[0]
        if actual != expected:
            raise ValueError(f'{name}: linker changed address {actual:#x}, expected {expected:#x}')
    command, start, size, entries = exports(data)
    renamed = {(name[2:] if name.startswith('_g_') else name): terminal for name, terminal in entries.items()}
    if len(renamed) != len(entries):
        raise ValueError('public export name collision')
    trie = make_trie(renamed)
    signature = next((o for cmd, o, _ in load_commands(data) if cmd == 0x1d), None)
    if signature is not None:
        # Produce an unsigned final binary; signing is performed after all edits.
        signature_start = struct.unpack_from('<I', data, signature+8)[0]
        del data[signature_start:]
    start = align(len(data), 8)
    data.extend(b'\0'*(start-len(data)))
    data.extend(trie)
    struct.pack_into('<II', data, command+8, start, len(trie))
    # Leave string bytes intact: imports may share suffixes with exported names.
    symcmd = next(o for cmd, o, _ in load_commands(data) if cmd == 2)
    symoff, count, stroff, _ = struct.unpack_from('<IIII', data, symcmd+8)
    for i in range(count):
        offset = symoff+i*16
        string_index, kind = struct.unpack_from('<IB', data, offset)
        if kind & 0x0e and data[stroff+string_index:stroff+string_index+3] == b'_g_':
            struct.pack_into('<I', data, offset, string_index+2)
    for cmd, offset, _ in load_commands(data):
        if cmd == 0x19 and data[offset+8:offset+24].rstrip(b'\0') == b'__LINKEDIT':
            fileoff = struct.unpack_from('<Q', data, offset+40)[0]
            struct.pack_into('<Q', data, offset+32, align(len(data)-fileoff))
            struct.pack_into('<Q', data, offset+48, len(data)-fileoff)
    commands = [bytes(data[offset:offset+size]) for cmd, offset, size in load_commands(data) if cmd != 0x1d]
    old_size = struct.unpack_from('<I', data, 20)[0]
    command_bytes = b''.join(commands)
    data[32:32+old_size] = command_bytes+b'\0'*(old_size-len(command_bytes))
    struct.pack_into('<II', data, 16, len(commands), len(command_bytes))
    Path(path).write_bytes(data)
    return len(renamed)
