"""Relocation destinations do not need exported labels unless referenced."""
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

from aot.bundle import Bundle


@unittest.skipUnless(shutil.which('aarch64-linux-gnu-gcc'), 'requires ARM64 cross compiler')
class SparseSymbolTests(unittest.TestCase):
    def test_pointer_table_slots_preserve_relocations_without_unused_labels(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root/'fixture.c').write_text(
                'static int first(void){return 3;} static int second(void){return 7;}\n'
                'int (*table[2])(void)={first,second}; int call_second(void){return table[1]();}')
            subprocess.run(['aarch64-linux-gnu-gcc', '-shared', '-fPIC', '-nostdlib',
                            str(root/'fixture.c'), '-o', str(root/'fixture.so')], check=True)
            bundle = Bundle(root/'fixture.so')
            table = next(s for s in bundle.graph.root.elf.symbols if s['name']=='table')
            slot = table['value']+8
            relocation = next(r for r in bundle.graph.root.elf.relocs if r[0]==slot)
            bundle.emit(root/'prepared')
            image = (root/'prepared/n0/image.S').read_text()
            # The table's second slot needs a relocation, but no other ELF
            # address refers to the slot itself. Its function target is labeled.
            self.assertNotIn(f'_aot_n0_elf_{slot:x}:', image)
            self.assertIn(f'.quad _aot_n0_elf_{relocation[3]:x}', image)
            self.assertIn(f'_aot_n0_elf_{relocation[3]:x}:', image)
