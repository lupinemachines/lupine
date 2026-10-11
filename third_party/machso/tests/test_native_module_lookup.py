"""Late-bound native dependencies preserve module scopes and scalar ABIs."""
import json
from pathlib import Path
import shutil
import struct
import subprocess
import tempfile
import unittest

from compile_bundle import prepare_bundle
from aot.graph import Graph
from aot.macho import make_trie, native_interface
from machso import align, uleb
from test_native_provider_abi import CASES, TYPES, prepare_fixture as prepare_scalar_fixture


def prepare_fixture(directory):
    prepare_scalar_fixture(directory)
    provider = directory/'provider.c'
    provider.write_text(provider.read_text()+'\nint native_shadow(void){return 99;}\nint not_supported(void){return 123;}\n'
                        '_Thread_local int native_local=17; int native_data=55;\n'
                        'int read_native_local(void){return native_local;}\n')
    interface_path = directory/'provider.json'
    interface = json.loads(interface_path.read_text())
    interface['exports'] += ['native_shadow', 'not_supported', 'native_local', 'native_data', 'read_native_local']
    interface['tls_exports'] = ['native_local']
    interface['signatures']['native_shadow'] = {'arguments': [], 'result': 'i32'}
    interface['signatures']['read_native_local'] = {'arguments': [], 'result': 'i32'}
    interface['signature_report'] = {'unsupported': {'not_supported': 'aggregate arguments unsupported'}}
    interface_path.write_text(json.dumps(interface))
    source = ['#define _GNU_SOURCE', '#include <stdint.h>', '#include <dlfcn.h>',
              'int native_shadow(void){return 42;}']
    for case, arguments in CASES.items():
        parameters = ','.join(TYPES[kind] for kind in arguments)
        values = [-(i+1) if kind.startswith('i') else i+1 for i, kind in enumerate(arguments)]
        constants = ','.join(f'(void *)(uintptr_t){value}' if kind=='ptr' else str(value)
                             for kind, value in zip(arguments, values))
        for owner in ('caller', 'provider'):
            source.append(f'double check_{owner}_{case}(void){{'
                          f'void*h=dlopen("{owner}.so",2);if(!h)return -1;'
                          f'double(*f)({parameters})=dlsym(h,"native_{case}");if(!f)return -2;'
                          f'double result=f({constants});dlclose(h);return result;}}')
    source.append('int check_scope(void){void*h=dlopen("caller.so",2);if(!h)return -1;'
                  'int(*f)(void)=dlsym(h,"native_shadow");if(!f)return -2;'
                  'int result=f();dlclose(h);h=dlopen("provider.so",2);if(!h)return -3;'
                  'f=dlsym(h,"native_shadow");if(!f)return -4;result=result*100+f();'
                  'if(dlsym(h,"not_supported")||!dlerror()||dlerror())return -5;'
                  'if(dlvsym(h,"native_shadow","MISSING")||!dlerror())return -6;'
                  'dlclose(h);return result;}')
    source.append('int check_native_data(void){void*h=dlopen("caller.so",2);if(!h)return -1;'
                  'int*local=dlsym(h,"native_local");int*data=dlsym(h,"native_data");'
                  'int(*read)(void)=dlsym(h,"read_native_local");if(!local||!data||!read)return -2;'
                  'int initial=*local;*local=18;int result=(initial==17&&read()==18&&*data==55);'
                  '*local=17;dlclose(h);return result;}')
    (directory/'root.c').write_text('\n'.join(source)+'\n')
    compiler = ['aarch64-linux-gnu-gcc', '-shared', '-fPIC', '-nostdlib', '-O2']
    subprocess.run([*compiler, str(provider), '-Wl,-soname,provider.so',
                    '-o', str(directory/'provider.so')], check=True)
    subprocess.run([*compiler, str(directory/'root.c'), '-Wl,-soname,caller.so',
                    '-Wl,--no-as-needed', '-L'+str(directory), '-l:provider.so', '-lc',
                    '-o', str(directory/'root.so')], check=True)
    return prepare_bundle(directory/'root.so', directory/'prepared',
                          ['/usr/aarch64-linux-gnu/lib'], experimental=True,
                          native_providers=[interface_path])


@unittest.skipUnless(shutil.which('aarch64-linux-gnu-gcc') and Path('/usr/aarch64-linux-gnu/lib/libc.so.6').is_file(),
                     'needs ARM64 cross compiler and glibc')
class NativeModuleLookupTests(unittest.TestCase):
    def test_late_bound_native_exports_get_bridges_and_module_scope(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            report = prepare_fixture(directory)
            self.assertFalse(report['missing_required'])
            self.assertFalse(any(b['provider']=='provider.so' for b in report['bindings']))
            self.assertEqual(report['native_abi_bridges'],
                             sorted([*('native_'+case for case in CASES), 'native_shadow', 'read_native_local']))
            self.assertNotIn('not_supported', report['native_dynamic_exports']['provider.so'])
            native = (directory/'prepared/native.c').read_text()
            self.assertIn('"@rpath/libscalar-provider.dylib"', native)
            self.assertIn('native symbol has an unsupported ABI signature', native)
            self.assertRegex(native, r'__asm__\("_aot_native_n\d+_s\d+"\)')

    def test_static_native_tls_import_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            prepare_fixture(directory)
            (directory/'static.c').write_text(
                'extern _Thread_local int native_local; int read(void){return native_local;}\n')
            subprocess.run(['aarch64-linux-gnu-gcc', '-shared', '-fPIC', '-nostdlib',
                            str(directory/'static.c'), str(directory/'provider.so'),
                            '-o', str(directory/'static.so')], check=True)
            with self.assertRaisesRegex(ValueError, 'static native TLS imports are unsupported'):
                Graph(directory/'static.so', native_providers=[directory/'provider.json'])


class NativeTLSInspectionTests(unittest.TestCase):
    def test_native_interface_distinguishes_thread_local_export_kind(self):
        install_name = b'@rpath/provider.dylib\0'
        size = align(24+len(install_name), 8)
        identity = struct.pack('<6I', 13, size, 24, 0, 0, 0)+install_name
        identity += bytes(size-len(identity))
        trie = make_trie({'_native_local': uleb(1)+uleb(0x2000),
                          '_native_data': uleb(0)+uleb(0x3000),
                          '_native_fn': uleb(0)+uleb(0x1000),
                          '_constant': uleb(2)+uleb(123)})
        header = struct.pack('<8I', 0xfeedfacf, 0x100000c, 0, 6, 2, size+16, 0, 0)
        command = struct.pack('<4I', 0x80000033, 16, 32+size+16, len(trie))
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)/'provider.dylib'
            path.write_bytes(header+identity+command+trie)
            interface = native_interface(path, 'provider.so')
            self.assertEqual(interface['tls_exports'], ['native_local'])
            self.assertEqual(interface['exports'], ['constant', 'native_data', 'native_fn', 'native_local'])
