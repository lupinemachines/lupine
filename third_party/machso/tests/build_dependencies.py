"""Build independent Linux test fixtures; the converter only consumes their binaries."""
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def build_dependencies(directory):
    directory.mkdir(parents=True, exist_ok=True)
    for name in ['leaf', 'middle', 'root']:
        command = ['aarch64-linux-gnu-gcc', '-shared', '-fPIC', '-Wall', '-Wextra', '-Werror',
                   f'-Wl,-soname,lib{name}.so', str(ROOT/f'tests/dependencies/{name}.c'),
                   '-o', str(directory/f'lib{name}.so')]
        if name == 'middle':
            command += ['-L'+str(directory), '-lleaf']
        if name == 'root':
            command += ['-L'+str(directory), '-lmiddle']
        subprocess.run(command, check=True)
