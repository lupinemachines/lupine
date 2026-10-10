#!/usr/bin/env python3
"""Compile ARM64 ELF dependencies into a portable native build directory."""
import argparse
import json
from pathlib import Path

from aot.bundle import Bundle
from aot.graph import Graph


def prepare_bundle(source, directory, library_paths=(), experimental=False):
    bundle = Bundle(source, library_paths)
    directory = Path(directory)
    report = bundle.emit(directory)
    report['build_status'] = 'prepared'
    if report['unsupported_syscalls'] and not experimental:
        raise ValueError('the libc platform profile has unsupported syscall lowerings; use --experimental to prepare an incomplete artifact')
    flags = []
    for node in bundle.nodes:
        plan = bundle.plans[node.index]
        for name, load in zip(plan.segments, node.elf.loads):
            address = plan.base+(load[3] & -0x4000)
            protection = ''.join(letter for flag, letter in [(4,'r'),(2,'w'),(1,'x')] if load[1] & flag)
            flags += ['-segaddr', name, hex(address), '-segprot', name, protection, protection]
    report['native_build'] = {
        'schema': 1,
        'modules': [node.identifier for node in bundle.link_order],
        'linker_flags': flags,
    }
    (directory/'manifest.json').write_text(json.dumps(report, indent=2)+'\n')
    print(f'Prepared {len(bundle.nodes)} ELF libraries in {directory}; required unresolved imports: 0.')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input', type=Path)
    parser.add_argument('output', type=Path, help='build directory, or JSON file with --plan-only')
    parser.add_argument('-L', '--library-path', type=Path, action='append', default=[])
    parser.add_argument('--plan-only', action='store_true')
    parser.add_argument('--experimental', action='store_true')
    args = parser.parse_args()
    try:
        if args.plan_only:
            report = Graph(args.input, args.library_path).report()
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(report, indent=2)+'\n')
            print(f'Planned {len(report["nodes"])} dependencies; {len(report["missing_required"])} unresolved required imports.')
        else:
            prepare_bundle(args.input, args.output, args.library_path, args.experimental)
    except (ValueError, OSError) as error:
        parser.exit(1, f'compile_bundle: {error}\n')


if __name__ == '__main__':
    main()
