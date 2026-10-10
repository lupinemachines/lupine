"""Real cross-compiled callers for native scalar argument bridges."""
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

from compile_bundle import prepare_bundle
from aot.graph import Graph

TYPES = {'i8': 'signed char', 'u8': 'unsigned char', 'i16': 'short',
         'u16': 'unsigned short', 'i32': 'int', 'u32': 'unsigned int',
         'i64': 'long long', 'u64': 'unsigned long long', 'ptr': 'void *',
         'f32': 'float', 'f64': 'double'}
CASES = {'integers': ['u64']*8+['i32', 'u32', 'i16', 'u8', 'i32'],
         'mixed': ['i32', 'f32']*11+['ptr', 'u8', 'f64'],
         'narrow': ['i8', 'u8', 'i16', 'u16']*3}


def prepare_fixture(directory):
    provider, declarations, calls, expected = ['#include <stdint.h>'], [], [], {}
    for name, arguments in CASES.items():
        parameters = ', '.join(f'{TYPES[kind]} a{i}' for i, kind in enumerate(arguments))
        declaration = f'double native_{name}({parameters})'
        values = [-(i+1) if kind.startswith('i') else i+1 for i, kind in enumerate(arguments)]
        expected[name] = sum((i+1)*value for i, value in enumerate(values))
        expression = '+'.join(f'{i+1}*(double)'+(f'(uintptr_t)a{i}' if kind == 'ptr' else f'a{i}')
                              for i, kind in enumerate(arguments))
        provider.append(declaration+' {return '+expression+';}')
        declarations.append(declaration+';')
        constants = ', '.join(f'(void *)(uintptr_t){value}' if kind == 'ptr' else str(value)
                              for kind, value in zip(arguments, values))
        calls.append(f'double check_{name}(void) {{return native_{name}({constants});}}')
    (directory/'provider.c').write_text('\n'.join(provider)+'\n')
    (directory/'root.c').write_text('#include <stdint.h>\n'+'\n'.join(declarations+calls)+'\n')
    compiler = ['aarch64-linux-gnu-gcc', '-shared', '-fPIC', '-nostdlib', '-O2']
    subprocess.run([*compiler, str(directory/'provider.c'), '-Wl,-soname,provider.so',
                    '-o', str(directory/'provider.so')], check=True)
    subprocess.run([*compiler, str(directory/'root.c'), '-L'+str(directory), '-l:provider.so',
                    '-o', str(directory/'root.so')], check=True)
    interface = {'schema': 1, 'architecture': 'arm64', 'soname': 'provider.so',
                 'native_install_name': '@rpath/libscalar-provider.dylib',
                 'exports': ['native_'+name for name in CASES],
                 'signatures': {'native_'+name: {'arguments': args, 'result': 'f64'}
                                for name, args in CASES.items()}}
    (directory/'provider.json').write_text(json.dumps(interface))
    (directory/'expected.json').write_text(json.dumps(expected))
    return prepare_bundle(directory/'root.so', directory/'prepared',
                          native_providers=[directory/'provider.json'])


@unittest.skipUnless(shutil.which('aarch64-linux-gnu-gcc'), 'needs ARM64 cross compiler')
class NativeProviderABITests(unittest.TestCase):
    def test_fixed_scalar_callers_use_bridges(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            report = prepare_fixture(directory)
            self.assertEqual(report['native_abi_bridges'], sorted('native_'+name for name in CASES))
            self.assertEqual(report['native_build']['modules'], ['n0'])
            self.assertEqual(report['native_build']['abi_bridge'], 'native_abi.S')
            self.assertFalse(report['missing_required'])

    def test_unsupported_signature_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            prepare_fixture(directory)
            interface = directory/'provider.json'
            declaration = json.loads(interface.read_text())
            declaration['signatures']['native_integers']['arguments'] = ['struct']
            interface.write_text(json.dumps(declaration))
            with self.assertRaisesRegex(ValueError, 'fixed scalar'):
                Graph(directory/'root.so', native_providers=[interface])
