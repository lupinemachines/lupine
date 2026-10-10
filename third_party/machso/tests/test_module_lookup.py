"""A forwarding library must resolve its implementation inside another module."""
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

from compile_bundle import prepare_bundle


def prepare_fixture(directory):
    (directory/'implementation.c').write_text(
        '__thread int local=17; int answer(void){return 42;} int read_local(void){return local;}\n'
        'int first(void){return 1;} int second(void){return 2;}\n'
        '__asm__(".symver first,value@V1"); __asm__(".symver second,value@@V2");\n')
    (directory/'versions.map').write_text('V1 {}; V2 {} V1;')
    (directory/'root.c').write_text(
        '#define _GNU_SOURCE\n#include <dlfcn.h>\n'
        'int answer(void){void*h=dlopen("implementation.so",2); if(!h)return -1; '
        'int(*f)(void)=dlsym(h,"answer"); if(!f)return -2; int result=f(); dlclose(h); return result;}\n'
        'int check_versions(void){void*h=dlopen("implementation.so",2); if(!h)return -1; '
        'int(*a)(void)=dlsym(h,"value"); int(*b)(void)=dlvsym(h,"value","V1"); '
        'if(!a||!b)return -2; int result=10*a()+b(); '
        'if(dlvsym(h,"value","MISSING")||!dlerror()||dlerror())return -3; dlclose(h); return result;}\n'
        'int check_tls(void){void*h=dlopen("implementation.so",2); if(!h)return -1; '
        'int*p=dlsym(h,"local"); int(*f)(void)=dlsym(h,"read_local"); if(!p||!f)return -2; '
        'int initial=*p; *p=initial+1; int result=(initial==17 && f()==18); *p=17; dlclose(h); return result;}\n')
    compiler = ['aarch64-linux-gnu-gcc','-shared','-fPIC','-nostdlib']
    subprocess.run([*compiler,str(directory/'implementation.c'),'-Wl,-soname,implementation.so',
                    '-Wl,--version-script='+str(directory/'versions.map'),
                    '-o',str(directory/'implementation.so')],check=True)
    subprocess.run([*compiler,str(directory/'root.c'),'-Wl,--no-as-needed','-lc',
                    '-o',str(directory/'root.so')],check=True)
    return prepare_bundle(directory/'root.so',directory/'prepared',['/usr/aarch64-linux-gnu/lib'],
                          experimental=True,additional_roots=[directory/'implementation.so'])


@unittest.skipUnless(shutil.which('aarch64-linux-gnu-gcc') and Path('/usr/aarch64-linux-gnu/lib/libc.so.6').is_file(),
                     'needs ARM64 cross compiler and glibc')
class ModuleLookupTests(unittest.TestCase):
    def test_forwarder_versions_and_tls_have_module_scoped_definitions(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory=Path(temporary)
            report=prepare_fixture(directory)
            self.assertFalse(report['missing_required'])
            self.assertIn('answer',report['public_names'])
            native=(directory/'prepared/native.c').read_text()
            self.assertIn('aot_module_symbol',native)
            self.assertIn('"implementation.so"',native)
            self.assertIn('"V1"',native)
            self.assertIn('"V2"',native)
            implementation=next(n for n in report['nodes'] if n['name']=='implementation.so')
            assembly=(directory/'prepared'/implementation['id']/'image.S').read_text()
            section=None
            for line in assembly.splitlines():
                if line.startswith('.section '):
                    section=line
                if line=='_g_local:':
                    self.assertIn('__thread_vars',section)
