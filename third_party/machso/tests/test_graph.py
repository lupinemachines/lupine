import importlib.util
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from machso import ELF
from aot.graph import Graph


@unittest.skipUnless(shutil.which('aarch64-linux-gnu-gcc'),'needs ARM64 cross compiler')
class GraphTests(unittest.TestCase):
    def compile(self, directory, name, source, options=()):
        path=directory/(name+'.c')
        path.write_text(source)
        output=directory/(name+'.so')
        subprocess.run(['aarch64-linux-gnu-gcc','-shared','-fPIC','-nostdlib',str(path),'-o',str(output),*options],check=True)
        return output

    def test_exact_symbol_versions(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory=Path(temporary)
            script=directory/'versions.map'
            script.write_text('V1 {}; V2 {} V1;')
            provider=self.compile(directory,'provider',
                'int first(void){return 1;} int second(void){return 2;}\n'
                '__asm__(".symver first,answer@V1"); __asm__(".symver second,answer@@V2");',
                ['-Wl,-soname,provider.so','-Wl,--version-script='+str(script)])
            consumer=self.compile(directory,'consumer',
                'extern int old(void); __asm__(".symver old,answer@V1"); int call(void){return old();}',
                ['-L'+str(directory),'-l:provider.so'])
            graph=Graph(consumer,[directory])
            imported=next(i for i,s in enumerate(graph.root.elf.symbols) if s['name']=='answer')
            owner,index=graph.bindings[(graph.root.index,imported)]
            self.assertEqual(owner.path,provider.resolve())
            self.assertEqual(owner.elf.symbols[index]['version'],'V1')
            self.assertTrue(owner.elf.symbols[index]['version_hidden'])

    def test_runpath_origin_and_no_filename_mapping(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory=Path(temporary)
            libraries=directory/'unusual'
            libraries.mkdir()
            self.compile(libraries,'unexpected_name','int some_value(void){return 17;}',
                         ['-Wl,-soname,unexpected_name.so'])
            root=self.compile(directory,'root','extern int some_value(void); int f(void){return some_value();}',
                              ['-L'+str(libraries),'-l:unexpected_name.so','-Wl,-rpath,$ORIGIN/unusual'])
            report=Graph(root).report()
            self.assertFalse(report['missing_required'])
            self.assertEqual(report['nodes'][1]['name'],'unexpected_name.so')

    def test_required_unresolved_symbol_is_reported(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory=Path(temporary)
            source=self.compile(directory,'missing','extern int unavailable(void); int f(void){return unavailable();}')
            report=Graph(source).report()
            self.assertEqual(report['missing_required'][0]['name'],'unavailable')

    def test_segment_layout_and_tls_modules_are_planned(self):
        if not importlib.util.find_spec('capstone'): self.skipTest('needs capstone')
        from aot.bundle import Bundle
        with tempfile.TemporaryDirectory() as temporary:
            directory=Path(temporary)
            self.compile(directory,'dependency','__thread int local=11; int f(void){return local;}',
                         ['-Wl,-soname,dependency.so','-mtls-dialect=desc'])
            root=self.compile(directory,'root','extern int f(void); __thread int other; int g(void){return f()+other;}',
                              ['-L'+str(directory),'-l:dependency.so','-mtls-dialect=desc'])
            bundle=Bundle(root,[directory])
            offsets=[p.tls_offset for p in bundle.plans.values()]
            self.assertEqual(len(set(offsets)),2)
            self.assertTrue(all(p.module_id for p in bundle.plans.values()))
            starts=[bundle.plans[n.index].base for n in bundle.link_order]
            self.assertEqual(starts,sorted(starts))


if __name__=='__main__':unittest.main()
