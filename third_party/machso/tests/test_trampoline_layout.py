import unittest
from pathlib import Path
import shutil
import subprocess
import tempfile

from aot.bundle import trampoline_layout
from compile_bundle import prepare_bundle


class TrampolineLayoutTests(unittest.TestCase):
    def test_large_image_uses_two_reachable_islands(self):
        sites = [{'address': 0x1000, 'kind': 'tls', 'register': 0},
                 {'address': 0x8100000, 'kind': 'tls', 'register': 0}]
        base, thunks, end = trampoline_layout(0x400000, 0x8200000, sites, 'n0')
        self.assertEqual(len(thunks), 2)
        self.assertEqual(thunks[0]['sites'], sites[:1])
        self.assertEqual(thunks[1]['sites'], sites[1:])
        self.assertGreaterEqual(thunks[1]['address'], base+0x8200000)
        self.assertGreater(end, thunks[1]['address'])
        for thunk in thunks:
            for site in thunk['sites']:
                delta = thunk['address']-(base+site['address'])
                self.assertTrue(-(1<<27) < delta < 1<<27)

    def test_single_far_code_cluster_uses_suffix(self):
        sites = [{'address': 0x8100000, 'kind': 'tls', 'register': 0}]
        _, thunks, _ = trampoline_layout(0x400000, 0x8200000, sites, 'n0')
        self.assertFalse(thunks[0]['sites'])
        self.assertEqual(thunks[1]['sites'], sites)

    def test_unreachable_middle_cluster_is_rejected(self):
        sites = [{'address': 0x10000000, 'kind': 'tls', 'register': 0}]
        with self.assertRaisesRegex(ValueError, 'branch range'):
            trampoline_layout(0x400000, 0x21000000, sites, 'n0')

    @unittest.skipUnless(shutil.which('aarch64-linux-gnu-gcc'), 'needs ARM64 cross compiler')
    def test_distant_code_sections_emit_both_trampoline_segments(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            source = directory/'root.c'
            source.write_text('int early(void) { __asm__ volatile("mrs x0,tpidr_el0" ::: "x0"); return 10; }\n'
                              '__attribute__((section(".far"))) int late(void) { '
                              '__asm__ volatile("mrs x0,tpidr_el0" ::: "x0"); return 32; }\n'
                              'int answer(void) { return early()+late(); }\n')
            script = directory/'layout.ld'
            script.write_text('PHDRS { early PT_LOAD FILEHDR PHDRS FLAGS(5); late PT_LOAD FLAGS(5); '
                              'data PT_LOAD FLAGS(6); dynamic PT_DYNAMIC FLAGS(6); }\n'
                              'SECTIONS { . = 0x1000; .text : { *(.text) *(.plt*) } :early '
                              '. = 0x8100000; .far : { *(.far) } :late '
                              '. = ALIGN(0x4000); .dynamic : { *(.dynamic) } :data :dynamic '
                              '.got : { *(.got*) } :data .data : { *(.data*) } :data '
                              '.bss : { *(.bss*) } :data }\n')
            subprocess.run(['aarch64-linux-gnu-gcc', '-shared', '-fPIC', '-nostdlib', str(source),
                            '-Wl,--build-id=none', '-Wl,-T,'+str(script), '-o', str(directory/'root.so')], check=True)
            report = prepare_bundle(directory/'root.so', directory/'prepared')
            self.assertIn('__N0T', report['segments'])
            self.assertIn('__N0T1', report['segments'])
            assembly = (directory/'prepared/n0/image.S').read_text()
            self.assertIn('.section __N0T1,__text', assembly)
            self.assertEqual(sum(c['tls_sites'] for c in report['components'].values()), 2)
