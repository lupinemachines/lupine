#!/usr/bin/env python3
"""Prepare NVIDIA's ARM64 runtime for the macOS client using vendored machso."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'third_party/machso'))
from compile_bundle import prepare_bundle

CUDA_RUNTIME_VERSION = '13.3.29'


def download_runtime(directory, version):
    subprocess.run([
        sys.executable, '-m', 'pip', 'download', '--dest', str(directory),
        '--only-binary=:all:', '--no-deps', '--implementation', 'py',
        '--python-version', '3', '--abi', 'none',
        '--platform', 'manylinux2014_aarch64', '--platform', 'manylinux_2_17_aarch64',
        'nvidia-cuda-runtime=='+version,
    ], check=True)
    wheels = list(directory.glob('*.whl'))
    if len(wheels) != 1:
        raise ValueError('expected exactly one CUDA runtime wheel')
    with zipfile.ZipFile(wheels[0]) as archive:
        candidates = [name for name in archive.namelist()
                      if '/lib/' in name and Path(name).name == 'libcudart.so.13']
        if len(candidates) != 1:
            raise ValueError('wheel must contain one libcudart.so.13')
        output = directory/'libcudart.so.13'
        output.write_bytes(archive.read(candidates[0]))
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('-L', '--library-path', type=Path, action='append', required=True)
    parser.add_argument('--runtime', type=Path, help='use an existing ARM64 Linux runtime binary')
    parser.add_argument('--version', default=CUDA_RUNTIME_VERSION)
    args = parser.parse_args()
    with tempfile.TemporaryDirectory() as temporary:
        source = args.runtime or download_runtime(Path(temporary), args.version)
        report = prepare_bundle(source, args.output, args.library_path, experimental=True)
        report['runtime_source'] = ({'path': str(args.runtime)} if args.runtime else
                                    {'package': 'nvidia-cuda-runtime', 'version': args.version})
        (args.output/'manifest.json').write_text(json.dumps(report, indent=2)+'\n')


if __name__ == '__main__':
    main()
