#!/usr/bin/env python3
"""Assemble and link a prepared machso directory using the local macOS toolchain."""
import argparse
import json
from pathlib import Path
import subprocess

from aot.macho import finish, file_hash, native_interface


def run(command, directory):
    result = subprocess.run(command, cwd=directory, text=True, capture_output=True)
    if result.stderr:
        print(result.stderr, end='')
    result.check_returncode()
    return result.stdout


def link_bundle(directory, output, native_libraries=()):
    directory = Path(directory).resolve()
    output = Path(output).resolve()
    report = json.loads((directory/'manifest.json').read_text())
    build = report['native_build']
    if build['schema'] != 1:
        raise ValueError('unsupported native build schema')
    available = {}
    for library in native_libraries:
        library = Path(library).resolve()
        interface = native_interface(library, '')
        available[interface['native_install_name']] = (library, interface)
    dependencies = []
    for provider in report.get('native_providers', []):
        item = available.get(provider['native_install_name'])
        if item is None:
            raise ValueError(f"{provider['soname']}: supply native library {provider['native_install_name']}")
        library, interface = item
        required = {binding['name'] for binding in report['bindings']
                    if binding['provider'] == provider['soname']}
        missing = required-set(interface['exports'])
        if missing:
            raise ValueError(f'{library}: missing native exports: {sorted(missing)}')
        dependencies.append(str(library))
    output.parent.mkdir(parents=True, exist_ok=True)
    objects = ['native.o']
    try:
        run(['clang', '-arch', 'arm64', '-fvisibility=hidden', '-O2', '-Wall', '-Wextra', '-Werror',
             '-c', 'native.c', '-o', 'native.o'], directory)
        if build.get('abi_bridge'):
            run(['clang', '-arch', 'arm64', '-c', build['abi_bridge'], '-o', 'native_abi.o'], directory)
            objects.append('native_abi.o')
        for module in build['modules']:
            # Retain optional Linux CPU dispatch variants. The platform profile
            # advertises baseline ARM64 capabilities; it must not select SVE/SME
            # code on Apple CPUs. These flags only assemble original snippets.
            run(['clang', '-arch', 'arm64', '-march=armv9.4-a+sve+sve2+sme+sme2',
                 '-c', 'image.S', '-o', 'image.o'], directory/module)
            objects.append(module+'/image.o')
        run(['clang', '-dynamiclib', *objects, *dependencies, '-Wl,'+','.join(build['linker_flags']),
             *(['-Wl,-rpath,@loader_path'] if dependencies else []),
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
    parser.add_argument('--native-library', type=Path, action='append', default=[])
    args = parser.parse_args()
    try:
        link_bundle(args.directory, args.output, args.native_library)
    except (ValueError, OSError, KeyError, subprocess.CalledProcessError) as error:
        parser.exit(1, f'link_bundle: {error}\n')


if __name__ == '__main__':
    main()
