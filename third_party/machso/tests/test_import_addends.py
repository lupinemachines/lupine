"""External data relocations retain their ELF RELA addends."""
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

from compile_bundle import prepare_bundle
from machso import ELF


def prepare_fixture(directory):
    (directory/'values.c').write_text('int values[]={3,5,7};\n')
    (directory/'root.c').write_text(
        'extern int values[]; static int *after=values+2, *before=values-1;\n'
        'int answer(void){return 10*(*after)+before[1];}\n')
    compiler=['aarch64-linux-gnu-gcc','-shared','-fPIC','-nostdlib']
    subprocess.run([*compiler,str(directory/'values.c'),'-Wl,-soname,values.so',
                    '-o',str(directory/'values.so')],check=True)
    subprocess.run([*compiler,str(directory/'root.c'),str(directory/'values.so'),
                    '-o',str(directory/'root.so')],check=True)
    return prepare_bundle(directory/'root.so',directory/'prepared',[directory],experimental=True)


@unittest.skipUnless(shutil.which('aarch64-linux-gnu-gcc'),'needs ARM64 cross compiler')
class ImportAddendTests(unittest.TestCase):
    def test_positive_and_negative_external_data_offsets(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory=Path(temporary)
            report=prepare_fixture(directory)
            self.assertFalse(report['missing_required'])
            root=ELF.from_file(directory/'root.so')
            addends={a for _,kind,i,a in root.relocs if kind==257 and root.symbols[i]['name']=='values'}
            self.assertEqual(addends,{-4,8})
            assembly=(directory/'prepared/n0/image.S').read_text()
            self.assertRegex(assembly,r'\.quad _aot_n1_s\d+\+8\b')
            self.assertRegex(assembly,r'\.quad _aot_n1_s\d+-4\b')
