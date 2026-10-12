"""Native API queries must return callable Linux ABI function pointers."""
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

from aot.native_abi import validate_signatures
from compile_bundle import prepare_bundle
from test_native_provider_abi import CASES, TYPES, prepare_fixture as prepare_scalar_fixture


def prepare_fixture(directory):
    prepare_scalar_fixture(directory)
    provider = directory/'provider.c'
    provider.write_text(provider.read_text()+'''
int native_unsupported(void){return 123;}
int native_unknown(void){return 321;}
int native_query(int choice,void **output){
    if(choice==5)return 17;
    void *functions[]={native_integers,native_mixed,native_narrow,native_unsupported,native_unknown,0};
    *output=functions[choice==6?5:choice];return 0;
}
int native_query_stack(unsigned long long a,unsigned long long b,unsigned long long c,
                       unsigned long long d,unsigned long long e,unsigned long long f,
                       unsigned long long g,unsigned long long h,int choice,void **output){
    if(a+b+c+d+e+f+g+h!=36)return 19;
    return native_query(choice,output);
}
int native_query_wide(int choice,void **output){return native_query(choice,output);}
''')
    path = directory/'provider.json'
    interface = json.loads(path.read_text())
    interface['exports'] += ['native_unsupported', 'native_unknown', 'native_query',
                             'native_query_stack', 'native_query_wide']
    for name, arguments, index, unavailable in (
            ('native_query', ['i32', 'ptr'], 1, 500),
            ('native_query_stack', ['u64']*8+['i32', 'ptr'], 9, 500),
            ('native_query_wide', ['i32', 'ptr'], 1, 65537)):
        interface['signatures'][name] = {
            'arguments': arguments, 'result': 'i32',
            'function_pointer_output': {'argument': index, 'unavailable_result': unavailable}}
    interface['signature_report'] = {'unsupported': {'native_unsupported': 'aggregate arguments unsupported'}}
    path.write_text(json.dumps(interface))
    source = ['#include <stdint.h>', 'int native_query(int,void **);',
              'int native_query_wide(int,void **);', 'int native_unknown(void);',
              'int native_query_stack(unsigned long long,unsigned long long,unsigned long long,'
              + 'unsigned long long,unsigned long long,unsigned long long,unsigned long long,'
              + 'unsigned long long,int,void **);']
    for choice, (case, arguments) in enumerate(CASES.items()):
        parameters = ','.join(TYPES[kind] for kind in arguments)
        values = [-(i+1) if kind.startswith('i') else i+1 for i, kind in enumerate(arguments)]
        constants = ','.join(f'(void *)(uintptr_t){value}' if kind=='ptr' else str(value)
                             for kind, value in zip(arguments, values))
        source.append(f'double check_query_{case}(void){{void *p=0;'
                      f'if(native_query({choice},&p))return -1;'
                      f'return ((double(*)({parameters}))p)({constants});}}')
        if choice == 0:
            source.append(f'double check_query_stack(void){{void *p=0;'
                          f'if(native_query_stack(1,2,3,4,5,6,7,8,0,&p))return -1;'
                          f'return ((double(*)({parameters}))p)({constants});}}')
    source += [
        'int check_query_unsupported(void){void*p=(void*)1;return native_query(3,&p)==500&&!p;}',
        'int check_query_error(void){void*p=(void*)0x12345678;return native_query(5,&p)==17&&p==(void*)0x12345678;}',
        'int check_query_null(void){void*p=(void*)1;return native_query(6,&p)==500&&!p;}',
        'int check_query_unknown(void){void*p=0;return !native_query(4,&p)&&p==(void*)native_unknown&&((int(*)(void))p)()==321;}',
        'int check_query_wide_error(void){void*p=0;return native_query_wide(3,&p)==65537&&!p;}']
    (directory/'root.c').write_text('\n'.join(source)+'\n')
    compiler = ['aarch64-linux-gnu-gcc', '-shared', '-fPIC', '-nostdlib', '-O2']
    subprocess.run([*compiler, str(provider), '-Wl,-soname,provider.so',
                    '-o', str(directory/'provider.so')], check=True)
    subprocess.run([*compiler, str(directory/'root.c'), '-L'+str(directory), '-l:provider.so',
                    '-o', str(directory/'root.so')], check=True)
    return prepare_bundle(directory/'root.so', directory/'prepared', native_providers=[path])


@unittest.skipUnless(shutil.which('aarch64-linux-gnu-gcc'), 'needs ARM64 cross compiler')
class NativeFunctionPointerTests(unittest.TestCase):
    def test_query_prepares_all_callable_targets_without_libc(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            report = prepare_fixture(directory)
            self.assertFalse(report['missing_required'])
            self.assertEqual(len(report['native_abi_bridges']), 6)
            required = report['native_dynamic_exports']['provider.so']
            self.assertIn('native_unsupported', required)
            self.assertIn('native_mixed', required)
            assembly = (directory/'prepared/native_abi.S').read_text()
            self.assertIn('function_pointer', assembly)
            self.assertIn('movk w20, #1, lsl #16', assembly)


class FunctionPointerContractTests(unittest.TestCase):
    def test_invalid_outputs_are_rejected(self):
        for output in ([], {}, {'argument': True, 'unavailable_result': 500},
                       {'argument': -1, 'unavailable_result': 500},
                       {'argument': 1, 'unavailable_result': 500},
                       {'argument': 0, 'unavailable_result': 0}):
            with self.subTest(output=output), self.assertRaisesRegex(ValueError, 'function pointer output'):
                validate_signatures({'exports': ['query'], 'signatures': {
                    'query': {'arguments': ['ptr'], 'result': 'i32', 'function_pointer_output': output}}})
