"""C++ TLS teardown follows thread lifetime and reverse registration order."""
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest


@unittest.skipUnless(shutil.which('cc'), 'requires a C compiler')
class TLSDestructorTests(unittest.TestCase):
    def test_thread_exit_and_destructors_registering_destructors(self):
        support = Path(__file__).resolve().parents[1]/'aot/native_tls_dtors.c'
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root/'test.c'
            source.write_text(r'''
#include <assert.h>
#include <pthread.h>
#include <stdint.h>
int aot_linux___cxa_thread_atexit_impl(void (*)(void *), void *, void *);
void aot_linux___call_tls_dtors(void);
static _Thread_local int sequence;
static int worker_result;
static void last(void *value) {sequence = sequence*10 + (int)(intptr_t)value;}
static void nested(void *value) {
    last(value);
    assert(aot_linux___cxa_thread_atexit_impl(last, (void *)3, 0) == 0);
}
static void first(void *value) {
    last(value);
    worker_result = sequence;
}
static void *worker(void *unused) {
    assert(aot_linux___cxa_thread_atexit_impl(first, (void *)1, 0) == 0);
    assert(aot_linux___cxa_thread_atexit_impl(nested, (void *)2, 0) == 0);
    return 0;
}
int main(void) {
    assert(aot_linux___cxa_thread_atexit_impl(last, (void *)4, 0) == 0);
    pthread_t thread;
    assert(pthread_create(&thread, 0, worker, 0) == 0);
    assert(pthread_join(thread, 0) == 0);
    assert(worker_result == 231);
    assert(sequence == 0);
    aot_linux___call_tls_dtors();
    assert(sequence == 4);
    aot_linux___call_tls_dtors();
    assert(sequence == 4);
}
''')
            subprocess.run(['cc', '-pthread', str(source), str(support), '-o', str(root/'test')], check=True)
            subprocess.run([str(root/'test')], check=True)
