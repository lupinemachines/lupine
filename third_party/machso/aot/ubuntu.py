"""Optional, metadata-driven ARM64 Ubuntu dependency acquisition."""
import gzip
import lzma
from pathlib import Path, PurePosixPath
import subprocess
import tarfile
import tempfile
import urllib.request

from aot.macho import file_hash


class UbuntuDependencies:
    def __init__(self, cache, keyring='/usr/share/keyrings/ubuntu-archive-keyring.gpg',
                 mirror='https://ports.ubuntu.com/ubuntu-ports/'):
        # The current libc lowering profile targets Noble's glibc 2.39.
        self.cache = Path(cache).resolve()
        self.keyring = Path(keyring)
        self.mirror = mirror.rstrip('/')+'/'
        self.hashes = None
        self.records = {}

    def download(self, relative, digest=None):
        path = PurePosixPath(relative)
        if path.is_absolute() or '..' in path.parts:
            raise ValueError(f'invalid archive path: {relative}')
        output = self.cache/path
        if not output.is_file():
            output.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(dir=output.parent, delete=False) as target:
                temporary = Path(target.name)
                try:
                    with urllib.request.urlopen(self.mirror+relative, timeout=60) as response:
                        for chunk in iter(lambda: response.read(1024*1024), b''):
                            target.write(chunk)
                    target.flush()
                    if digest and file_hash(temporary) != digest:
                        raise ValueError(f'archive SHA256 mismatch: {relative}')
                    temporary.replace(output)
                finally:
                    temporary.unlink(missing_ok=True)
        if digest and file_hash(output) != digest:
            raise ValueError(f'cached archive SHA256 mismatch: {relative}')
        return output

    def release(self):
        if self.hashes is not None:
            return
        if not self.keyring.is_file():
            raise ValueError('Ubuntu acquisition requires the Ubuntu archive keyring; supply --ubuntu-keyring')
        release = self.download('dists/noble/Release')
        signature = self.download('dists/noble/Release.gpg')
        result = subprocess.run(['gpgv', '--keyring', str(self.keyring), str(signature), str(release)],
                                text=True, capture_output=True)
        if result.returncode:
            raise ValueError('Ubuntu Release signature verification failed: '+result.stderr)
        hashes, in_sha256 = {}, False
        for line in release.read_text().splitlines():
            if not line.startswith(' '):
                in_sha256 = line == 'SHA256:'
            elif in_sha256:
                digest, _, relative = line.split()
                hashes[relative] = digest
        if not hashes:
            raise ValueError('Ubuntu Release has no SHA256 index')
        self.hashes = hashes

    def index(self, relative):
        self.release()
        if relative not in self.hashes:
            raise ValueError(f'index absent from signed Ubuntu Release: {relative}')
        return self.download('dists/noble/'+relative, self.hashes[relative])

    def package_for(self, name):
        candidates = set()
        paths = {'usr/lib/aarch64-linux-gnu/'+name, 'lib/aarch64-linux-gnu/'+name}
        with gzip.open(self.index('Contents-arm64.gz'), 'rt') as contents:
            for line in contents:
                fields = line.rsplit(None, 1)
                if len(fields) == 2 and fields[0] in paths:
                    candidates.update(p.rsplit('/', 1)[-1] for p in fields[1].split(','))
        if len(candidates) != 1:
            raise ValueError(f'{name}: missing or ambiguous Ubuntu package providers: {sorted(candidates)}')
        return candidates.pop()

    def package_record(self, package):
        for component in ('main', 'universe', 'restricted', 'multiverse'):
            with lzma.open(self.index(component+'/binary-arm64/Packages.xz'), 'rt') as index:
                record = {}
                # Also accept a final stanza without a trailing blank line.
                for line in index:
                    if not line.strip():
                        if record.get('Package') == package:
                            return record
                        record = {}
                    elif not line[0].isspace() and ':' in line:
                        key, value = line.split(':', 1)
                        record[key] = value.strip()
                if record.get('Package') == package:
                    return record
        raise ValueError(f'{package}: absent from ARM64 Ubuntu package indexes')

    def find(self, name, machine):
        if machine != 183 or '/' in name:
            raise ValueError('Ubuntu acquisition supports ARM64 SONAME dependencies')
        package = self.package_for(name)
        record = self.package_record(package)
        if record.get('Architecture') not in ('arm64', 'all'):
            raise ValueError(f'{package}: wrong package architecture')
        archive = self.download(record['Filename'], record['SHA256'])
        destination = self.cache/'extracted'/record['SHA256']
        if not destination.is_dir():
            if not hasattr(tarfile, 'data_filter'):
                raise ValueError('Ubuntu acquisition requires Python with tarfile data filtering (3.10.13 or newer)')
            destination.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.TemporaryDirectory(dir=destination.parent) as temporary:
                with subprocess.Popen(['dpkg-deb', '--fsys-tarfile', str(archive)], stdout=subprocess.PIPE) as process:
                    try:
                        with tarfile.open(fileobj=process.stdout, mode='r|') as contents:
                            contents.extractall(temporary, filter='data')
                        # Drain trailing tar padding before waiting on the
                        # producer, which may otherwise block on a full pipe.
                        for chunk in iter(lambda: process.stdout.read(1024*1024), b''):
                            pass
                    except BaseException:
                        process.kill()
                        raise
                    if process.wait():
                        raise ValueError(f'{package}: could not extract package payload')
                Path(temporary).rename(destination)
        library = next((destination/path/name for path in ('usr/lib/aarch64-linux-gnu', 'lib/aarch64-linux-gnu')
                        if (destination/path/name).is_file()), None)
        if library is None:
            raise ValueError(f'{package}: package does not provide {name}')
        self.records[name] = {'soname': name, 'package': package, 'version': record['Version'],
                              'sha256': record['SHA256'], 'url': self.mirror+record['Filename'],
                              'release_sha256': file_hash(self.cache/'dists/noble/Release')}
        return library.resolve()

    def report(self):
        return list(self.records.values())
