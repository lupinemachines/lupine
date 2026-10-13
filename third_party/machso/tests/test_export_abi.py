"""Native entry points and escaped pointers retain the caller's ABI domain."""
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

from compile_bundle import prepare_bundle
from test_native_function_pointers import prepare_fixture as prepare_queries
from test_native_provider_abi import CASES, TYPES


def prepare_fixture(directory):
    prepare_queries(directory)
    subprocess.run(['aarch64-linux-gnu-gcc', '-shared', '-fPIC', '-nostdlib', '-O2',
                    str(directory/'provider.c'), str(directory/'root.c'),
                    '-Wl,-soname,provider.so', '-o', str(directory/'provider.so')], check=True)
    return prepare_bundle(directory/'provider.so', directory/'export-prepared',
                          export_interface=directory/'provider.json')


def prepare_forward_fixture(directory):
    prepare_queries(directory)
    (directory/'forward.c').write_text(
        'int native_query(int,void**);\n'
        'int forward_query(int choice,void **output){return native_query(choice,output);}\n')
    subprocess.run(['aarch64-linux-gnu-gcc', '-shared', '-fPIC', '-nostdlib', '-O2',
                    str(directory/'forward.c'), '-L'+str(directory), '-l:provider.so',
                    '-o', str(directory/'forward.so')], check=True)
    interface = {'schema': 1, 'architecture': 'arm64', 'exports': ['forward_query'],
                 'signatures': {'forward_query': {'arguments': ['i32', 'ptr'], 'result': 'i32',
                     'function_pointer_output': {'argument': 1, 'unavailable_result': 500}}}}
    (directory/'export-interface.json').write_text(json.dumps(interface))
    return prepare_bundle(directory/'forward.so', directory/'forward-prepared',
                          native_providers=[directory/'provider.json'],
                          export_interface=directory/'export-interface.json')


def prepare_scope_fixture(directory):
    prepare_fixture(directory)
    arguments = CASES['integers']
    parameters = ','.join(TYPES[kind] for kind in arguments)
    values = [-(i+1) if kind.startswith('i') else i+1 for i, kind in enumerate(arguments)]
    constants = ','.join(str(value) for value in values)
    (directory/'scope.c').write_text(
        '#include <dlfcn.h>\n'
        f'double check_default_scope(void){{double(*f)({parameters})=dlsym(0,"native_integers");'
        f'if(!f)return -1;return f({constants});}}\n')
    subprocess.run(['aarch64-linux-gnu-gcc', '-shared', '-fPIC', '-nostdlib', '-O2',
                    str(directory/'provider.c'), str(directory/'root.c'), str(directory/'scope.c'),
                    '-Wl,-soname,provider.so', '-Wl,--no-as-needed', '-lc',
                    '-o', str(directory/'scope.so')], check=True)
    return prepare_bundle(directory/'scope.so', directory/'scope-prepared',
                          ['/usr/aarch64-linux-gnu/lib'], experimental=True,
                          export_interface=directory/'provider.json')


@unittest.skipUnless(shutil.which('aarch64-linux-gnu-gcc'), 'needs ARM64 cross compiler')
class ExportABITests(unittest.TestCase):
    def test_export_only_bundle_records_bridge_input(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            report = prepare_fixture(directory)
            self.assertFalse(report['missing_required'])
            self.assertFalse(report['native_abi_bridges'])
            self.assertEqual(len(report['export_abi_bridges']), 6)
            self.assertEqual(report['native_build']['abi_bridge'], 'native_abi.S')

    def test_native_dependency_and_exports_share_function_pointer_metadata(self):
        with tempfile.TemporaryDirectory() as temporary:
            report = prepare_forward_fixture(Path(temporary))
            self.assertEqual(report['export_abi_bridges'], ['forward_query'])
            self.assertEqual(len(report['native_abi_bridges']), 6)
            self.assertFalse(report['missing_required'])

    def test_nonexistent_export_signature_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            prepare_queries(directory)
            interface = {'schema': 1, 'architecture': 'arm64', 'exports': ['missing'],
                         'signatures': {'missing': {'arguments': [], 'result': 'void'}}}
            path = directory/'export-interface.json'
            path.write_text(json.dumps(interface))
            with self.assertRaisesRegex(ValueError, 'public ELF definition'):
                prepare_bundle(directory/'provider.so', directory/'bad', export_interface=path)

    def test_data_export_cannot_be_misdeclared_as_a_function(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            (directory/'data.c').write_text('int exported_data=3;\n')
            subprocess.run(['aarch64-linux-gnu-gcc', '-shared', '-fPIC', '-nostdlib',
                            str(directory/'data.c'), '-o', str(directory/'data.so')], check=True)
            interface = {'schema': 1, 'architecture': 'arm64', 'exports': ['exported_data'],
                         'signatures': {'exported_data': {'arguments': [], 'result': 'i32'}}}
            path = directory/'export-interface.json'
            path.write_text(json.dumps(interface))
            with self.assertRaisesRegex(ValueError, 'regular ELF functions'):
                prepare_bundle(directory/'data.so', directory/'bad', export_interface=path)

    @unittest.skipUnless(Path('/usr/aarch64-linux-gnu/lib/libc.so.6').is_file(), 'needs ARM64 glibc')
    def test_default_lookup_has_a_converted_abi_domain(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            report = prepare_scope_fixture(directory)
            self.assertFalse(report['missing_required'])
            self.assertEqual(len(report['export_abi_bridges']), 6)
