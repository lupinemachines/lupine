import ctypes
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from machso import ELF
from aot.tunables import extract, native_code


class TunableTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        loader = Path('/usr/aarch64-linux-gnu/lib/ld-linux-aarch64.so.1')
        if not loader.exists():
            raise unittest.SkipTest('requires an ARM64 glibc loader binary')
        cls.entries = extract(ELF(loader.read_bytes(), allow_tls=True))

    def test_defaults_come_from_loader_binary(self):
        defaults = {entry['name']: entry for entry in self.entries}
        self.assertEqual(defaults['glibc.mem.tagging']['value'], 0)
        self.assertEqual(defaults['glibc.mem.tagging']['type'], 0)
        self.assertEqual(defaults['glibc.malloc.top_pad']['value'], 131072)
        self.assertEqual(defaults['glibc.malloc.top_pad']['type'], 2)

    @unittest.skipUnless(shutil.which('cc'), 'requires a C compiler')
    def test_lowering_writes_exact_type_width_and_preserves_neighbors(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            source = directory/'tunables.c'
            source.write_text('#include <stdint.h>\n#include <stddef.h>\n'+native_code(self.entries))
            library = directory/'tunables.so'
            subprocess.run(['cc', '-shared', '-fPIC', str(source), '-o', str(library)], check=True)
            function = ctypes.CDLL(str(library)).aot_import___tunable_get_val
            function.argtypes = [ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p]
            function.restype = None
            for index, entry in enumerate(self.entries):
                buffer = (ctypes.c_ubyte * 32)(*([0xa5] * 32))
                function(index, ctypes.byref(buffer, 8), None)
                width = 4 if entry['type'] == 0 else 16 if entry['type'] == 3 else 8
                expected = entry['value'].to_bytes(width, 'little')
                self.assertEqual(bytes(buffer)[8:8+width], expected)
                self.assertEqual(bytes(buffer)[:8], b'\xa5' * 8)
                self.assertEqual(bytes(buffer)[8+width:], b'\xa5' * (24-width))


if __name__ == '__main__':
    unittest.main()
