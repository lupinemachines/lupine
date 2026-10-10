import gzip
import hashlib
import lzma
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

from aot.graph import Graph
from aot.ubuntu import UbuntuDependencies


class UbuntuMetadataTests(unittest.TestCase):
    def test_corrupt_cached_download_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            provider = UbuntuDependencies(temporary)
            (Path(temporary)/'payload').write_bytes(b'corrupted')
            digest = hashlib.sha256(b'expected').hexdigest()
            with self.assertRaisesRegex(ValueError, 'cached archive SHA256 mismatch'):
                provider.download('payload', digest)

    def test_archive_paths_cannot_escape_cache(self):
        with tempfile.TemporaryDirectory() as temporary:
            provider = UbuntuDependencies(temporary)
            for path in ('../payload', '/payload'):
                with self.assertRaisesRegex(ValueError, 'invalid archive path'):
                    provider.download(path)

    def test_ambiguous_library_providers_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            index = root/'Contents-arm64.gz'
            index.write_bytes(gzip.compress(b'usr/lib/aarch64-linux-gnu/libodd.so libs/one,libs/two\n'))
            provider = UbuntuDependencies(root)
            provider.index = lambda relative: index
            with self.assertRaisesRegex(ValueError, 'ambiguous Ubuntu package providers'):
                provider.package_for('libodd.so')

    @unittest.skipUnless(shutil.which('gpgv'), 'needs gpgv')
    def test_invalid_release_signature_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root/'keyring').write_bytes(b'invalid keyring')
            (root/'dists/noble').mkdir(parents=True)
            (root/'dists/noble/Release').write_text('SHA256:\n abc 3 Contents-arm64.gz\n')
            (root/'dists/noble/Release.gpg').write_bytes(b'invalid signature')
            provider = UbuntuDependencies(root, root/'keyring')
            with self.assertRaisesRegex(ValueError, 'signature verification failed'):
                provider.release()
            self.assertIsNone(provider.hashes)


@unittest.skipUnless(shutil.which('dpkg-deb') and shutil.which('aarch64-linux-gnu-gcc'),
                     'needs dpkg-deb and ARM64 cross compiler')
class UbuntuClosureTests(unittest.TestCase):
    def test_missing_dependencies_are_discovered_and_acquired_recursively(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            mirror = root/'mirror'
            mirror.mkdir()
            libraries = root/'libraries'
            libraries.mkdir()
            def compile(name, source, extra=()):
                path = libraries/(name+'.c')
                path.write_text(source)
                output = libraries/(name+'.so')
                subprocess.run(['aarch64-linux-gnu-gcc', '-shared', '-fPIC', '-nostdlib', str(path),
                                '-Wl,-soname,'+name+'.so', '-o', str(output), *extra], check=True)
                return output
            secondary = compile('unusual_secondary', 'int secondary(void){return 41;}')
            primary = compile('unusual_primary', 'extern int secondary(void); int primary(void){return secondary()+1;}',
                              ['-L'+str(libraries), '-l:unusual_secondary.so'])
            source = compile('consumer', 'extern int primary(void); int answer(void){return primary();}',
                             ['-L'+str(libraries), '-l:unusual_primary.so'])
            consumer = root/'consumer.so'
            shutil.copyfile(source, consumer)
            contents, stanzas = [], []
            for number, library in enumerate((primary, secondary)):
                package = f'fixture-package-{number}'
                tree = root/package
                (tree/'DEBIAN').mkdir(parents=True)
                (tree/'DEBIAN/control').write_text(f'Package: {package}\nVersion: 1.0\nArchitecture: arm64\n'
                                                 'Maintainer: Test <test@example.com>\nDescription: Fixture\n')
                payload = tree/'usr/lib/aarch64-linux-gnu'
                payload.mkdir(parents=True)
                shutil.copyfile(library, payload/library.name)
                archive = mirror/(package+'.deb')
                subprocess.run(['dpkg-deb', '--build', str(tree), str(archive)], check=True, capture_output=True)
                digest = hashlib.sha256(archive.read_bytes()).hexdigest()
                contents.append(f'usr/lib/aarch64-linux-gnu/{library.name} libs/{package}\n')
                stanzas.append(f'Package: {package}\nVersion: 1.0\nArchitecture: arm64\nFilename: {archive.name}\nSHA256: {digest}\n\n')
            indexes = {'Contents-arm64.gz': gzip.compress(''.join(contents).encode()),
                       'main/binary-arm64/Packages.xz': lzma.compress(''.join(stanzas).encode())}
            for name, data in indexes.items():
                path = mirror/'dists/noble'/name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(data)
            provider = UbuntuDependencies(root/'cache', mirror=mirror.as_uri())
            # This test supplies a fixture index; real signature verification
            # is exercised independently and by the live Ubuntu smoke test.
            provider.hashes = {name: hashlib.sha256(data).hexdigest() for name, data in indexes.items()}
            release = provider.cache/'dists/noble/Release'
            release.parent.mkdir(parents=True)
            release.write_text('fixture release')
            graph = Graph(consumer, dependency_provider=provider)
            self.assertFalse(graph.missing)
            self.assertEqual(len(graph.nodes), 3)
            self.assertEqual({r['package'] for r in graph.report()['system_packages']},
                             {'fixture-package-0', 'fixture-package-1'})
