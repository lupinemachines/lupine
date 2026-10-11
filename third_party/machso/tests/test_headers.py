import importlib.util
from pathlib import Path
import shutil
import tempfile
import unittest

from aot.headers import read_signatures


@unittest.skipUnless(importlib.util.find_spec('clang') and shutil.which('clang'),
                     'needs optional headers extra and clang')
class HeaderTests(unittest.TestCase):
    def parse(self, source, exports, typedefs=None):
        with tempfile.TemporaryDirectory() as temporary:
            header = Path(temporary)/'native.h'
            header.write_text(source)
            return read_signatures([header], exports, signature_typedefs=typedefs)

    def test_explicit_historical_function_pointer_typedef(self):
        report = self.parse('typedef int (*old_query)(void *, unsigned long);',
                            ['query'], {'query': 'old_query'})
        self.assertEqual(report['signatures']['query'], {'arguments': ['ptr', 'u64'], 'result': 'i32'})
        self.assertEqual(report['signature_typedefs'], {'query': 'old_query'})

    def test_missing_signature_typedef_fails(self):
        with self.assertRaisesRegex(ValueError, 'typedefs not found'):
            self.parse('int query(void *);', ['query'], {'query': 'missing'})

    def test_conflicting_or_non_function_typedef_is_unsupported(self):
        for source in ('int query(int); typedef int (*old_query)(void *);',
                       'typedef int old_query;'):
            with self.subTest(source=source):
                report = self.parse(source, ['query'], {'query': 'old_query'})
                self.assertNotIn('query', report['signatures'])
                self.assertIn('query', report['unsupported'])

    def test_typedefs_enums_and_scalar_widths(self):
        report = self.parse('typedef unsigned long handle; typedef enum { YES=1 } choice;\n'
                            'double native(handle, choice, signed char, _Bool, float, void *, int [4]);\n',
                            ['native', 'undeclared'])
        self.assertEqual(report['signatures']['native'],
                         {'arguments': ['u64', 'u32', 'i8', 'u8', 'f32', 'ptr', 'ptr'], 'result': 'f64'})
        self.assertEqual(report['missing_declarations'], ['undeclared'])

    def test_aggregates_and_variadic_functions_are_reported(self):
        report = self.parse('struct pair { int a,b; };\n'
                            'int aggregate(struct pair); int variadic(int, ...);\n',
                            ['aggregate', 'variadic'])
        self.assertFalse(report['signatures'])
        self.assertEqual(set(report['unsupported']), {'aggregate', 'variadic'})

    def test_unprototyped_declaration_is_not_treated_as_zero_arguments(self):
        report = self.parse('int unknown();', ['unknown'])
        self.assertNotIn('unknown', report['signatures'])
        self.assertIn('unknown', report['unsupported'])

    def test_header_errors_fail_instead_of_emitting_partial_signatures(self):
        with self.assertRaisesRegex(ValueError, 'header parsing failed'):
            self.parse('int good(int); int broken(unknown_type);', ['good', 'broken'])

    def test_only_exported_functions_are_selected(self):
        report = self.parse('int exported(int); int unused(float);', ['exported'])
        self.assertEqual(set(report['signatures']), {'exported'})

    @unittest.skipUnless(shutil.which('aarch64-linux-gnu-gcc'), 'needs ARM64 cross compiler')
    def test_headers_match_mac_verified_scalar_fixture_signatures(self):
        from test_native_provider_abi import prepare_fixture
        import json
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            prepare_fixture(directory)
            interface = json.loads((directory/'provider.json').read_text())
            report = read_signatures([directory/'provider.h'], interface['exports'])
            self.assertEqual(report['signatures'], interface['signatures'])
            self.assertFalse(report['unsupported'])
            self.assertFalse(report['missing_declarations'])
