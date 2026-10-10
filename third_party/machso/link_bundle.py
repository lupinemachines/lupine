#!/usr/bin/env python3
"""Assemble and link a prepared machso directory using the local macOS toolchain."""
import argparse
import json
from pathlib import Path
import subprocess

from aot.macho import finish, file_hash


def run(command, directory):
    result = subprocess.run(command, cwd=directory, text=True, capture_output=True)
    if result.stderr:
        print(result.stderr, end='')
    result.check_returncode()
    return result.stdout


def link_bundle(directory, output):
    directory = Path(directory).resolve()
    output = Path(output).resolve()
    report = json.loads((directory/'manifest.json').read_text())
    build = report['native_build']
    if build['schema'] != 1:
        raise ValueError('unsupported native build schema')
    output.parent.mkdir(parents=True, exist_ok=True)
    objects = ['native.o']
    try:
        run(['clang', '-arch', 'arm64', '-fvisibility=hidden', '-O2', '-Wall', '-Wextra', '-Werror',
             '-c', 'native.c', '-o', 'native.o'], directory)
        for module in build['modules']:
            # Retain optional Linux CPU dispatch variants. The platform profile
            # advertises baseline ARM64 capabilities; it must not select SVE/SME
            # code on Apple CPUs. These flags only assemble original snippets.
            run(['clang', '-arch', 'arm64', '-march=armv9.4-a+sve+sve2+sme+sme2',
                 '-c', 'image.S', '-o', 'image.o'], directory/module)
            objects.append(module+'/image.o')
        run(['clang', '-dynamiclib', *objects, '-Wl,'+','.join(build['linker_flags']),
             '-Wl,-exported_symbols_list,exports.txt', '-install_name', '@rpath/'+output.name,
             '-o', str(output)], directory)
        report['native_exports'] = finish(output, report['segments'])
        report['native_dependencies'] = run(['otool', '-L', str(output)], directory)
        report['output_sha256'] = file_hash(output)
        report['build_status'] = 'linked'
        report['signed'] = False
    except subprocess.CalledProcessError as error:
        report['build_status'] = 'failed'
        report['build_error'] = {'command': error.cmd, 'stderr': error.stderr, 'stdout': error.stdout}
        raise
    finally:
        Path(str(output)+'.json').write_text(json.dumps(report, indent=2)+'\n')
    print(f'Linked {output}: {report["native_exports"]} public symbols; signing is a separate step.')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    try:
        link_bundle(args.directory, args.output)
    except (ValueError, OSError, KeyError, subprocess.CalledProcessError) as error:
        parser.exit(1, f'link_bundle: {error}\n')


if __name__ == '__main__':
    main()
