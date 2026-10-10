import struct
import tempfile
import unittest
from pathlib import Path
import sys
import shutil

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from machso import ELF, DependencyGraph, convert
from build_dependencies import build_dependencies

ROOT = Path(__file__).resolve().parents[1]


class ConverterTests(unittest.TestCase):
    def convert_bytes(self, data):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory)/'input.so'
            output = Path(directory)/'output.dylib'
            source.write_bytes(data)
            report = convert(source, output)
            return output.read_bytes(), report

    def test_no_section_headers_or_static_symbols_required(self):
        data = bytearray((ROOT/'tests/libtest.aarch64.so').read_bytes())
        original, report = self.convert_bytes(data)
        struct.pack_into('<Q', data, 40, 0)  # e_shoff
        struct.pack_into('<HHH', data, 58, 0, 0, 0)
        stripped, _ = self.convert_bytes(data)
        # The copied ELF header changes too; no conversion metadata depends on it.
        self.assertEqual(original[:0x4000], stripped[:0x4000])
        self.assertEqual(original[0x4040:], stripped[0x4040:])
        self.assertEqual(len(report['exports']), 16)
        elf = ELF(data)
        rx = elf.loads[0]
        # All bytes in the original executable segment survive unchanged.
        self.assertEqual(stripped[0x4000:0x4000+rx[5]], data[rx[2]:rx[2]+rx[5]])
        self.assertEqual(struct.unpack_from('<I', stripped)[0], 0xfeedfacf)

    def test_wrong_architecture_rejected(self):
        with self.assertRaisesRegex(ValueError, 'AArch64'):
            self.convert_bytes((ROOT/'tests/libtest.x86_64.so').read_bytes())

    def test_unknown_relocation_rejected(self):
        data = bytearray((ROOT/'tests/libtest.aarch64.so').read_bytes())
        elf = ELF(data)
        relocation = elf.tag(7)
        struct.pack_into('<Q', data, relocation+3*24+8, 1031)  # TLS DTPREL
        with self.assertRaisesRegex(ValueError, 'relocation 1031'):
            self.convert_bytes(data)

    def test_unknown_import_rejected(self):
        data = (ROOT/'tests/libtest.aarch64.so').read_bytes().replace(b'strdup\0', b'zzzzzz\0')
        with self.assertRaisesRegex(ValueError, 'unsupported import: zzzzzz'):
            self.convert_bytes(data)

    def test_direct_syscall_rejected(self):
        data = bytearray((ROOT/'tests/libtest.aarch64.so').read_bytes())
        elf = ELF(data)
        add = next(s for s in elf.symbols if s['name'] == 'add')
        struct.pack_into('<I', data, add['value'], 0xd4000001)
        with self.assertRaisesRegex(ValueError, 'direct SVC'):
            self.convert_bytes(data)


@unittest.skipUnless(shutil.which('aarch64-linux-gnu-gcc'), 'needs cross compiler')
class DependencyTests(unittest.TestCase):
    def test_transitive_conversion_and_missing_dependency(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            fixtures = directory/'elf'
            build_dependencies(fixtures)
            output = directory/'converted/root.dylib'
            graph = DependencyGraph(fixtures/'libroot.so', output, [fixtures])
            self.assertEqual([p.name for p in graph.closure(graph.root)],
                             ['libmiddle.so', 'libleaf.so'])
            reports = graph.convert()
            self.assertEqual(set(reports), {'root.dylib', 'libmiddle.so.dylib', 'libleaf.so.dylib'})
            self.assertIn(b'@loader_path/libleaf.so.dylib\0', output.read_bytes())
            (fixtures/'libleaf.so').unlink()
            with self.assertRaisesRegex(ValueError, 'missing libleaf.so'):
                DependencyGraph(fixtures/'libroot.so', directory/'missing/root.dylib', [fixtures])
            self.assertFalse((directory/'missing').exists())

    def test_cycles_during_discovery(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            fixtures = directory/'elf'
            build_dependencies(fixtures)
            # Add a reverse DT_NEEDED edge to create a middle -> leaf -> middle cycle.
            from unittest.mock import patch
            real_elf = ELF

            def cyclic_elf(data):
                elf = real_elf(data)
                if any(s['name'] == 'leaf_value' and s['shndx'] for s in elf.symbols):
                    elf.needed.append('libmiddle.so')
                return elf

            with patch('machso.ELF', side_effect=cyclic_elf):
                graph = DependencyGraph(fixtures/'libroot.so', directory/'root.dylib', [fixtures])
                self.assertEqual(len(graph.nodes), 3)
                self.assertEqual(len(graph.closure((fixtures/'libmiddle.so').resolve())), 1)


if __name__ == '__main__':
    unittest.main()
