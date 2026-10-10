import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
HAS_CAPSTONE=importlib.util.find_spec('capstone') is not None


@unittest.skipUnless(HAS_CAPSTONE,'run uv sync')
class AOTTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from machso import ELF
        cls.input=Path('/usr/aarch64-linux-gnu/lib/libc.so.6')
        if not cls.input.exists(): raise unittest.SkipTest('requires glibc AArch64 binary')
        cls.elf=ELF(cls.input.read_bytes(),allow_tls=True)

    def test_ifuncs_resolved_to_original_executable_code(self):
        from aot.ifunc import Resolver
        resolver=Resolver(self.elf)
        addresses={s['value'] for s in self.elf.symbols if s['info'] & 15==10}
        self.assertGreater(len(addresses),0)
        for address in addresses:
            result=resolver.resolve(address)
            self.assertNotEqual(result,address)
            self.assertTrue(any(p[1]&1 and p[3]<=result<p[3]+p[5] for p in self.elf.loads))

    def test_platform_instructions_have_static_replacements(self):
        from aot.emit import emit
        from aot.lower import emit_native
        with tempfile.TemporaryDirectory() as directory:
            manifest=emit_native(directory,emit(self.input,directory))
            image=Path(directory,'image.S').read_text()
            self.assertEqual(manifest['exports'],2852)
            self.assertEqual(manifest['tls_sites'],1530)
            self.assertEqual(len(manifest['syscall_sites']),517)
            self.assertEqual(len(manifest['x18_sites']),318)
            for site in manifest['syscall_sites']+manifest['x18_sites']:
                self.assertIn(f'Laot_{site["address"]:x}:',image)
            native=Path(directory,'native.c').read_text()
            self.assertNotIn('elfcompat',native)
            self.assertFalse(manifest['complete'])
            self.assertTrue(manifest['unsupported_syscalls'])

    def test_native_product_has_public_symbols_and_only_libsystem_dependency(self):
        from aot.macho import exports,load_commands
        import struct
        product=ROOT/'build/glibc.dylib'
        if not product.exists(): self.skipTest('compile build/glibc.dylib with link_bundle.py first')
        data=product.read_bytes()
        _,_,_,names=exports(data)
        self.assertEqual(len(names),2852)
        self.assertIn('_malloc',names)
        self.assertIn('_gnu_get_libc_version',names)
        self.assertFalse(any(name.startswith('_g_') for name in names))
        dependencies=[]
        for cmd,offset,_ in load_commands(data):
            if cmd==12:
                start=offset+struct.unpack_from('<I',data,offset+8)[0]
                dependencies.append(data[start:data.index(0,start)].decode())
        self.assertEqual(dependencies,['/usr/lib/libSystem.B.dylib'])


if __name__=='__main__': unittest.main()
