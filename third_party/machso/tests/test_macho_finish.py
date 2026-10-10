"""Final export edits preserve large payloads and shared symbol strings."""
import hashlib
import mmap
from pathlib import Path
import struct
import tempfile
import unittest
from unittest.mock import patch

from aot.macho import exports, file_hash, finish, load_commands, make_trie


class MachOFinishTests(unittest.TestCase):
    def fixture(self, path):
        end = 64*1024*1024+3
        symoff, stroff, trieoff = 4096, 4128, 4160
        strings = b'\0_g_answer\0'
        trie = make_trie({'_g_answer': b'\0\x10'})
        segment = struct.pack('<II16sQQQQIIII', 0x19, 72, b'__N0L0',
                              0x400000, 4096, 0, 4096, 5, 5, 0, 0)
        linkedit = struct.pack('<II16sQQQQIIII', 0x19, 72, b'__LINKEDIT',
                               0x500000, end+16, 4096, end-4096+16, 1, 1, 0, 0)
        symtab = struct.pack('<IIIIII', 2, 24, symoff, 2, stroff, len(strings))
        export_command = struct.pack('<IIII', 0x80000033, 16, trieoff, len(trie))
        signature = struct.pack('<IIII', 0x1d, 16, end, 16)
        commands = segment+linkedit+symtab+export_command+signature
        header = struct.pack('<IIIIIIII', 0xfeedfacf, 0x100000c, 0, 6, 5, len(commands), 0, 0)
        with path.open('wb') as output:
            output.write(header+commands)
            output.seek(symoff)
            # A defined symbol and an import share the original string bytes.
            output.write(struct.pack('<IBBHQ', 1, 0xf, 1, 0, 0x400010))
            output.write(struct.pack('<IBBHQ', 1, 1, 0, 0, 0))
            output.seek(stroff); output.write(strings)
            output.seek(trieoff); output.write(trie)
            output.seek(16*1024*1024); output.write(b'unchanged GPU payload')
            output.seek(end); output.write(b'signature bytes!')
        return symoff, stroff, end

    def test_sparse_payload_and_import_strings_are_untouched(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)/'image.dylib'
            symoff, stroff, end = self.fixture(path)
            with patch.object(Path, 'read_bytes', side_effect=AssertionError('whole-file read')):
                self.assertEqual(finish(path, {'__N0L0': 0x400000}), 1)
                self.assertEqual(len(file_hash(path)), 64)
            with path.open('rb') as source, mmap.mmap(source.fileno(), 0, access=mmap.ACCESS_READ) as data:
                self.assertEqual(set(exports(data)[3]), {'_answer'})
                self.assertFalse(any(kind==0x1d for kind, _, _ in load_commands(data)))
                self.assertEqual(struct.unpack_from('<I', data, symoff)[0], 3)
                self.assertEqual(struct.unpack_from('<I', data, symoff+16)[0], 1)
                self.assertEqual(data[stroff:stroff+11], b'\0_g_answer\0')
                self.assertEqual(data[16*1024*1024:16*1024*1024+21], b'unchanged GPU payload')
                self.assertEqual(exports(data)[1], (end+7)&-8)

    def test_layout_rejection_does_not_modify_file(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)/'image.dylib'
            self.fixture(path)
            before = file_hash(path)
            with self.assertRaisesRegex(ValueError, 'linker changed address'):
                finish(path, {'__N0L0': 0x800000})
            self.assertEqual(file_hash(path), before)

    def test_streaming_hash_matches_standard_sha256(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)/'bytes'
            data = bytes(range(256))*9000
            path.write_bytes(data)
            self.assertEqual(file_hash(path), hashlib.sha256(data).hexdigest())
