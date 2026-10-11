#!/usr/bin/env python3
"""Assemble and link a prepared machso directory using the local macOS toolchain."""
import argparse
import json
from pathlib import Path
import subprocess
import struct

from aot.macho import finish, file_hash, native_interface, load_commands
from machso import align


def reserve_native_space(report, directory, objects):
    """Keep fixed ELF segments beyond the linker-generated native sections."""
    size = 0x100000  # Linker headers, stubs, and alignment padding.
    custom = set(report['segments'])
    for name in objects:
        with (directory/name).open('rb') as source:
            header = source.read(32)
            if header[:4] != b'\xcf\xfa\xed\xfe':
                raise ValueError(f'{name}: expected a Mach-O object')
            data = header+source.read(struct.unpack_from('<I', header, 20)[0])
        for command, offset, _ in load_commands(data):
            if command != 0x19:
                continue
            for index in range(struct.unpack_from('<I', data, offset+64)[0]):
                section = offset+72+index*80
                segment = data[section+16:section+32].rstrip(b'\0').decode()
                if segment not in custom:
                    length = struct.unpack_from('<Q', data, section+40)[0]
                    alignment = 1 << struct.unpack_from('<I', data, section+52)[0]
                    size += align(length, alignment)
    delta = align(max(0, size-min(report['segments'].values())), 0x10000)
    if not delta:
        return
    flags = report['native_build']['linker_flags']
    for index, flag in enumerate(flags):
        if flag == '-segaddr':
            flags[index+2] = hex(int(flags[index+2], 16)+delta)
    for addresses in [report['segments'], *(c['segments'] for c in report['components'].values())]:
        for name in addresses:
            addresses[name] += delta
    report['native_build']['native_space_shift'] = delta


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
        required.update(report.get('native_dynamic_exports', {}).get(provider['soname'], []))
        missing = required-set(interface['exports'])
        if missing:
            raise ValueError(f'{library}: missing native exports: {sorted(missing)}')
        dependencies.append(str(library))
    output.parent.mkdir(parents=True, exist_ok=True)
    objects = ['native.o']
    exports = 'exports.txt'
    export_bridges = report.get('export_abi_bridges', [])
    if export_bridges:
        from aot.native_abi import emit_export_metadata
        if 'machso_native_export_abi' in report['public_names']:
            raise ValueError('ELF exports collide with the compiler ABI metadata entry point')
        (directory/'export_abi.c').write_text(emit_export_metadata(export_bridges))
        (directory/'native_exports.txt').write_text((directory/'exports.txt').read_text()+'_g_machso_native_export_abi\n')
        exports = 'native_exports.txt'
    try:
        run(['clang', '-arch', 'arm64', '-fvisibility=hidden', '-O2', '-Wall', '-Wextra', '-Werror',
             '-c', 'native.c', '-o', 'native.o'], directory)
        if build.get('abi_bridge'):
            run(['clang', '-arch', 'arm64', '-c', build['abi_bridge'], '-o', 'native_abi.o'], directory)
            objects.append('native_abi.o')
        if export_bridges:
            run(['clang', '-arch', 'arm64', '-O2', '-c', 'export_abi.c', '-o', 'export_abi.o'], directory)
            objects.append('export_abi.o')
        for module in build['modules']:
            # Retain optional Linux CPU dispatch variants. The platform profile
            # advertises baseline ARM64 capabilities; it must not select SVE/SME
            # code on Apple CPUs. These flags only assemble original snippets.
            run(['clang', '-arch', 'arm64', '-march=armv9.4-a+sve+sve2+sme+sme2',
                 '-c', 'image.S', '-o', 'image.o'], directory/module)
            objects.append(module+'/image.o')
        reserve_native_space(report, directory, objects)
        # A converted SDK also exports Linux libc names. Native bootstrap
        # calls must bind to Darwin, including pthread's incompatible structs.
        run(['clang', '-dynamiclib', *objects, '-lSystem', *dependencies, '-Wl,'+','.join(build['linker_flags']),
             # Widely separated functions can exceed compact unwind's 24-bit
             # compressed offsets. Retain DWARF unwind information instead.
             *(['-Wl,-no_compact_unwind'] if build.get('native_space_shift') else []),
             *(['-Wl,-rpath,@loader_path'] if dependencies else []),
             '-Wl,-exported_symbols_list,'+exports, '-install_name', '@rpath/'+output.name,
             '-o', str(output)], directory)
        report['native_exports'] = finish(output, report['segments'])
        report['native_dependencies'] = run(['otool', '-L', str(output)], directory)
        report['output_sha256'] = file_hash(output)
        report['build_status'] = 'linked'
        report['signed'] = False
        if export_bridges:
            report['export_abi_metadata'] = 'machso_native_export_abi'
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
